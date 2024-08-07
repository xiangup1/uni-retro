# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, Optional

import torch
from torch import Tensor, nn
from unicore.modules.softmax_dropout import softmax_dropout


class SelfMultiheadAttentionAug(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.1,
        bias=True,
        scaling_factor=1,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        self.num_heads = num_heads
        self.dropout = dropout

        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"
        self.scaling = (self.head_dim * scaling_factor) ** -0.5

        self.in_proj = nn.Linear(embed_dim, embed_dim * 3, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def forward(
        self,
        query,
        key_padding_mask: Optional[Tensor] = None,
        attn_bias: Optional[Tensor] = None,
        return_attn: bool = False,
    ) -> Tensor:

        bsz, tgt_len, embed_dim = query.size()
        assert embed_dim == self.embed_dim

        q, k, v = self.in_proj(query).chunk(3, dim=-1)

        q = (
            q.view(bsz, tgt_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
            .view(bsz * self.num_heads, -1, self.head_dim)
            * self.scaling
        )
        if k is not None:
            k = (
                k.view(bsz, -1, self.num_heads, self.head_dim)
                .transpose(1, 2)
                .contiguous()
                .view(bsz * self.num_heads, -1, self.head_dim)
            )
        if v is not None:
            v = (
                v.view(bsz, -1, self.num_heads, self.head_dim)
                .transpose(1, 2)
                .contiguous()
                .view(bsz * self.num_heads, -1, self.head_dim)
            )

        assert k is not None
        src_len = k.size(1)

        # This is part of a workaround to get around fork/join parallelism
        # not supporting Optional types.
        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        attn_weights = torch.bmm(q, k.transpose(1, 2))

        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        if key_padding_mask is not None:
            # don't attend to padding symbols
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights.masked_fill_(
                key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool), float("-inf")
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if not return_attn:
            attn = softmax_dropout(
                attn_weights, self.dropout, self.training, bias=attn_bias,
            )
        else:
            attn_weights += attn_bias
            attn = softmax_dropout(
                attn_weights, self.dropout, self.training, inplace=False,
            )

        o = torch.bmm(attn, v)
        assert list(o.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]

        o = (
            o.view(bsz, self.num_heads, tgt_len, self.head_dim)
            .transpose(1, 2)
            .contiguous()
            .view(bsz, tgt_len, embed_dim)
        )
        o = self.out_proj(o)
        if not return_attn:
            return o
        else:
            return o, attn_weights, attn


class CrossMultiheadAttentionAug(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.1,
        bias=True,
        scaling_factor=1,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        self.num_heads = num_heads
        self.dropout = dropout

        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"
        self.scaling = (self.head_dim * scaling_factor) ** -0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def forward(
        self,
        query,
        key,
        value,
        key_padding_mask: Optional[Tensor] = None,
        local_mask: Optional[Tensor] = None,
        attn_bias: Optional[Tensor] = None,
    ) -> Tensor:

        bsz, tgt_len, embed_dim = query.size()
        assert embed_dim == self.embed_dim

        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        q = (
            q.view(bsz, tgt_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
            * self.scaling
        )
        q_global = q[:,:self.num_heads//2].contiguous().view(bsz * self.num_heads//2, -1, self.head_dim)
        q_local =  q[:,self.num_heads//2:].contiguous().view(bsz * self.num_heads//2, -1, self.head_dim)
        if k is not None:
            k0 = (
                k.view(bsz, -1, self.num_heads, self.head_dim)
                .transpose(1, 2)
                .contiguous()
                .view(bsz * self.num_heads, -1, self.head_dim)
            )
            k = (
                k.view(bsz, -1, self.num_heads, self.head_dim)
                .transpose(1, 2)
                .contiguous()
            )
            k_global = k[:,:self.num_heads//2].contiguous().view(bsz * self.num_heads//2, -1, self.head_dim)
            k_local =  k[:,self.num_heads//2:].contiguous().view(bsz * self.num_heads//2, -1, self.head_dim)
        if v is not None:
            v = (
                v.view(bsz, -1, self.num_heads, self.head_dim)
                .transpose(1, 2)
                .contiguous()
            )
            v_global = v[:,:self.num_heads//2].contiguous().view(bsz * self.num_heads//2, -1, self.head_dim)
            v_local =  v[:,self.num_heads//2:].contiguous().view(bsz * self.num_heads//2, -1, self.head_dim)

        assert k is not None
        src_len = k0.size(1)

        # This is part of a workaround to get around fork/join parallelism
        # not supporting Optional types.
        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        attn_weights_global = torch.bmm(q_global, k_global.transpose(1, 2))
        assert list(attn_weights_global.size()) == [bsz * self.num_heads//2, tgt_len, src_len]

        if key_padding_mask is not None:
            # don't attend to padding symbols
            attn_weights_global = attn_weights_global.view(bsz, self.num_heads//2, tgt_len, src_len)
            attn_weights_global.masked_fill_(
                key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool), float("-inf")
            )
            attn_weights_global = attn_weights_global.view(bsz * self.num_heads//2, tgt_len, src_len)

        attn_global = softmax_dropout(attn_weights_global, self.dropout, self.training, bias=attn_bias)

        o_global = torch.bmm(attn_global, v_global)
        assert list(o_global.size()) == [bsz * self.num_heads//2, tgt_len, self.head_dim]

        o_global = (
            o_global.view(bsz, self.num_heads//2, tgt_len, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )

        if local_mask is not None and local_mask.dim() == 0:
            local_mask = None

        if local_mask is not None:
            assert local_mask.size(0) == bsz
            assert local_mask.size(1) == src_len

        attn_weights_local = torch.bmm(q_local, k_local.transpose(1, 2))

        assert list(attn_weights_local.size()) == [bsz * self.num_heads//2, tgt_len, src_len]

        if local_mask is not None:
            # don't attend to padding symbols
            attn_weights_local = attn_weights_local.view(bsz, self.num_heads//2, tgt_len, src_len)
            attn_weights_local.masked_fill_(
                local_mask.unsqueeze(1).unsqueeze(2).to(torch.bool), float("-inf")
            )
            attn_weights_local = attn_weights_local.view(bsz * self.num_heads//2, tgt_len, src_len)

        attn_local = softmax_dropout(attn_weights_local, self.dropout, self.training, bias=attn_bias)

        o_local = torch.bmm(attn_local, v_local)
        assert list(o_local.size()) == [bsz * self.num_heads//2, tgt_len, self.head_dim]

        o_local = (
            o_local.view(bsz, self.num_heads//2, tgt_len, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )
        o = torch.cat([o_global, o_local], dim = 2).contiguous().view(bsz, tgt_len, embed_dim)
        o = self.out_proj(o)
        return o
