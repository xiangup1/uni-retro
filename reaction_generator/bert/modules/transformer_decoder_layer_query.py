import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from unicore import utils
from torch import nn
from unicore.modules import  LayerNorm 
from .multihead_attention import SelfMultiheadAttention, CrossMultiheadAttention
from .multihead_attention_aug import CrossMultiheadAttentionAug

class TransformerDecoderLayerQuery(nn.Module):
    """
    Implements a Transformer Encoder Layer used in BERT/XLM style pre-trained
    models.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        ffn_embed_dim: int = 3072,
        attention_heads: int = 8,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.0,
        activation_fn: str = "gelu",
        post_ln = False,
        talking_heads = False,
    ) -> None:
        super().__init__()

        # Initialize parameters
        self.embed_dim = embed_dim
        self.attention_heads = attention_heads
        self.attention_dropout = attention_dropout

        self.dropout = dropout
        self.activation_dropout = activation_dropout
        self.activation_fn = utils.get_activation_fn(activation_fn)

        self.self_attn = CrossMultiheadAttention(
            self.embed_dim,
            attention_heads,
            dropout=attention_dropout,
        )

        # layer norm associated with the self attention layer
        self.self_attn_layer_norm = LayerNorm(self.embed_dim)

        self.encoder_attn = CrossMultiheadAttention(
            self.embed_dim,
            attention_heads,
            dropout=attention_dropout,
            talking_heads = talking_heads,
        )

        # layer norm associated with the self attention layer
        self.encoder_attn_layer_norm = LayerNorm(self.embed_dim)

        self.fc1 = nn.Linear(self.embed_dim, ffn_embed_dim)
        self.fc2 = nn.Linear(ffn_embed_dim, self.embed_dim)
        self.final_layer_norm = LayerNorm(self.embed_dim)
        self.post_ln = post_ln


    def forward(
        self,
        x: torch.Tensor,
        encoder_out:torch.Tensor=None,
        attn_bias: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        encoder_attn_bias: Optional[torch.Tensor] = None,
        encoder_padding_mask: Optional[torch.Tensor] = None,
        atten_score_flag = False, 
    ) -> torch.Tensor:
        """
        LayerNorm is applied either before or after the self-attention/ffn
        modules similar to the original Transformer implementation.
        """
        residual = x
        if not self.post_ln:
            x = self.self_attn_layer_norm(x)  # ???
        x = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=padding_mask,
            attn_bias=attn_bias,
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if self.post_ln:
            x = self.self_attn_layer_norm(x)
        if encoder_out is not None:
            residual = x
            if not self.post_ln:
                x = self.encoder_attn_layer_norm(x)
            if atten_score_flag:
                x, atten_weight = self.encoder_attn(
                    query=x,
                    key=encoder_out,
                    value=encoder_out,
                    key_padding_mask=encoder_padding_mask,
                    attn_bias=encoder_attn_bias,
                    atten_score_flag = atten_score_flag,
                )
            else:
                x = self.encoder_attn(
                    query=x,
                    key=encoder_out,
                    value=encoder_out,
                    key_padding_mask=encoder_padding_mask,
                    attn_bias=encoder_attn_bias,
                )
            #x = self.dropout_module(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = residual + x
            if self.post_ln:
                x = self.encoder_attn_layer_norm(x)

        residual = x
        if not self.post_ln:
            x = self.final_layer_norm(x)
        x = self.fc1(x)
        x = self.activation_fn(x)
        x = F.dropout(x, p=self.activation_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if self.post_ln:
            x = self.final_layer_norm(x)

        if atten_score_flag:
            return x, atten_weight
        else:
            return x


class TransformerEncoderLayerMerge(TransformerDecoderLayerQuery):

    def __init__(
        self,
        embed_dim: int = 768,
        ffn_embed_dim: int = 3072,
        attention_heads: int = 8,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.0,
        activation_fn: str = "gelu",
        post_ln = False,
    ) -> None:
        super().__init__()

        # Initialize parameters
        self.embed_dim = embed_dim
        self.attention_heads = attention_heads
        self.attention_dropout = attention_dropout

        self.dropout = dropout
        self.activation_dropout = activation_dropout
        self.activation_fn = utils.get_activation_fn(activation_fn)

        self.self_attn = CrossMultiheadAttention(
            self.embed_dim,
            attention_heads,
            dropout=attention_dropout,
        )

        # layer norm associated with the self attention layer
        self.self_attn_layer_norm = LayerNorm(self.embed_dim)

        self.encoder_attn = CrossMultiheadAttention(
            self.embed_dim,
            attention_heads,
            dropout=attention_dropout,
        )

        # layer norm associated with the self attention layer
        self.encoder_attn_layer_norm = LayerNorm(self.embed_dim)

        self.fc1 = nn.Linear(self.embed_dim, ffn_embed_dim)
        self.fc2 = nn.Linear(ffn_embed_dim, self.embed_dim)
        self.final_layer_norm = LayerNorm(self.embed_dim)
        self.post_ln = post_ln
        
    def forward(
        self,
        x: torch.Tensor,
        encoder_out:torch.Tensor=None,
        padding_mask: Optional[torch.Tensor] = None,
        encoder_attn_bias: Optional[torch.Tensor] = None,
        encoder_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        LayerNorm is applied either before or after the self-attention/ffn
        modules similar to the original Transformer implementation.
        """
        residual = x
        if self.post_ln:
            x = self.self_attn_layer_norm(x)
        if encoder_out is not None:
            residual = x
            if not self.post_ln:
                x = self.encoder_attn_layer_norm(x)
            x = self.encoder_attn(
                query=x,
                key=encoder_out,
                value=encoder_out,
                key_padding_mask=encoder_padding_mask,
                attn_bias=encoder_attn_bias,
            )
            #x = self.dropout_module(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = residual + x
            if self.post_ln:
                x = self.encoder_attn_layer_norm(x)

        residual = x
        if not self.post_ln:
            x = self.final_layer_norm(x)
        x = self.fc1(x)
        x = self.activation_fn(x)
        x = F.dropout(x, p=self.activation_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if self.post_ln:
            x = self.final_layer_norm(x)
        return x  


class TransformerDecoderLayerQueryAug(nn.Module):
    """
    Implements a Transformer Encoder Layer used in BERT/XLM style pre-trained
    models.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        ffn_embed_dim: int = 3072,
        attention_heads: int = 8,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.0,
        activation_fn: str = "gelu",
        post_ln = False,
        talking_heads = False,
    ) -> None:
        super().__init__()

        # Initialize parameters
        self.embed_dim = embed_dim
        self.attention_heads = attention_heads
        self.attention_dropout = attention_dropout

        self.dropout = dropout
        self.activation_dropout = activation_dropout
        self.activation_fn = utils.get_activation_fn(activation_fn)

        self.self_attn = CrossMultiheadAttention(
            self.embed_dim,
            attention_heads,
            dropout=attention_dropout,
        )

        # layer norm associated with the self attention layer
        self.self_attn_layer_norm = LayerNorm(self.embed_dim)

        self.encoder_attn = CrossMultiheadAttentionAug(
            self.embed_dim,
            attention_heads,
            dropout=attention_dropout,
            talking_heads = talking_heads,
        )

        # layer norm associated with the self attention layer
        self.encoder_attn_layer_norm = LayerNorm(self.embed_dim)

        self.fc1 = nn.Linear(self.embed_dim, ffn_embed_dim)
        self.fc2 = nn.Linear(ffn_embed_dim, self.embed_dim)
        self.final_layer_norm = LayerNorm(self.embed_dim)
        self.post_ln = post_ln


    def forward(
        self,
        x: torch.Tensor,
        encoder_out:torch.Tensor=None,
        attn_bias: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        encoder_attn_bias: Optional[torch.Tensor] = None,
        encoder_padding_mask: Optional[torch.Tensor] = None,
        reaction_center_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        LayerNorm is applied either before or after the self-attention/ffn
        modules similar to the original Transformer implementation.
        """
        residual = x
        if not self.post_ln:
            x = self.self_attn_layer_norm(x)  # ???
        x = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=padding_mask,
            attn_bias=attn_bias,
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if self.post_ln:
            x = self.self_attn_layer_norm(x)
        if encoder_out is not None:
            residual = x
            if not self.post_ln:
                x = self.encoder_attn_layer_norm(x)
            x = self.encoder_attn(
                query=x,
                key=encoder_out,
                value=encoder_out,
                key_padding_mask=encoder_padding_mask,
                local_mask = reaction_center_mask,
                attn_bias=encoder_attn_bias,
            )
            #x = self.dropout_module(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = residual + x
            if self.post_ln:
                x = self.encoder_attn_layer_norm(x)

        residual = x
        if not self.post_ln:
            x = self.final_layer_norm(x)
        x = self.fc1(x)
        x = self.activation_fn(x)
        x = F.dropout(x, p=self.activation_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if self.post_ln:
            x = self.final_layer_norm(x)
        return x

