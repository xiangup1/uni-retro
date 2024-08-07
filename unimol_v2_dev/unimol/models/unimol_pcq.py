# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from unicore import utils
from unicore.models import BaseUnicoreModel, register_model, register_model_architecture
from unicore.modules import LayerNorm, init_bert_params
from .transformer_encoder_with_pair import TransformerEncoderWithPair
from .transformer_encoder_with_tri_pair import TransformerEncoderWithTriPair
from typing import Dict, Any, List

BACKBONE = {
    "transformer": TransformerEncoderWithPair,
    "transformer_tri": TransformerEncoderWithTriPair,
}

logger = logging.getLogger(__name__)


@register_model("unimol_pcq")
class UniMolPCQModel(BaseUnicoreModel):
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument(
            "--encoder-layers", type=int, metavar="L", help="num encoder layers"
        )
        parser.add_argument(
            "--encoder-embed-dim",
            type=int,
            metavar="H",
            help="encoder embedding dimension",
        )
        parser.add_argument(
            "--encoder-ffn-embed-dim",
            type=int,
            metavar="F",
            help="encoder embedding dimension for FFN",
        )
        parser.add_argument(
            "--encoder-attention-heads",
            type=int,
            metavar="A",
            help="num encoder attention heads",
        )
        parser.add_argument(
            "--activation-fn",
            choices=utils.get_available_activation_fns(),
            help="activation function to use",
        )
        parser.add_argument(
            "--pooler-activation-fn",
            choices=utils.get_available_activation_fns(),
            help="activation function to use for pooler layer",
        )
        parser.add_argument(
            "--emb-dropout",
            type=float,
            metavar="D",
            help="dropout probability for embeddings",
        )
        parser.add_argument(
            "--dropout", type=float, metavar="D", help="dropout probability"
        )
        parser.add_argument(
            "--attention-dropout",
            type=float,
            metavar="D",
            help="dropout probability for attention weights",
        )
        parser.add_argument(
            "--activation-dropout",
            type=float,
            metavar="D",
            help="dropout probability after activation in FFN",
        )
        parser.add_argument(
            "--pooler-dropout",
            type=float,
            metavar="D",
            help="dropout probability in the masked_lm pooler layers",
        )
        parser.add_argument(
            "--max-seq-len", type=int, help="number of positional embeddings to learn"
        )
        parser.add_argument(
            "--post-ln", type=bool, help="use post layernorm or pre layernorm"
        )
        parser.add_argument(
            "--backbone",
            type=str,
            default="transformer",
            choices=BACKBONE.keys(),
            help="backbone of unimol model",
        )
        parser.add_argument(
            "--masked-token-loss",
            type=float,
            metavar="D",
            help="mask loss ratio",
        )
        parser.add_argument(
            "--masked-dist-loss",
            type=float,
            metavar="D",
            help="masked distance loss ratio",
        )
        parser.add_argument(
            "--masked-coord-loss",
            type=float,
            metavar="D",
            help="masked coord loss ratio",
        )
        parser.add_argument(
            "--masked-coord-dist-loss",
            type=float,
            metavar="D",
            help="masked coord dist loss ratio",
        )
        parser.add_argument(
            "--x-norm-loss",
            type=float,
            metavar="D",
            help="x norm loss ratio",
        )
        parser.add_argument(
            "--delta-pair-repr-norm-loss",
            type=float,
            metavar="D",
            help="delta encoder pair repr norm loss ratio",
        )
        parser.add_argument(
            "--lddt-loss",
            type=float,
            metavar="D",
            help="lddt loss ratio",
        )
        parser.add_argument(
            "--homo-lumo-loss",
            type=float,
            metavar="D",
            help="pred homo lumo loss ratio",
        )
        parser.add_argument(
            "--homo-lumo-bin-loss",
            type=float,
            metavar="D",
            help="pred homo lumo bin loss ratio",
        )
        parser.add_argument(
            "--p-homo-lumo-loss",
            type=float,
            metavar="D",
            help="pred of pred homo lumo loss ratio",
        )
        parser.add_argument(
            "--gaussian-layer-eps",
            type=float,
            metavar="D",
            help="guassian layer eps",
        )
        parser.add_argument(
            "--pre-rmsd-loss",
            type=float,
            metavar="D",
            help="pre rmsd loss ratio",
        )
        parser.add_argument(
            "--pre-dist-loss",
            type=float,
            metavar="D",
            help="pre dist loss ratio",
        )
        parser.add_argument(
            "--num-recycle",
            type=int,
            help="number of cycles to use for coordinate prediction",
        )
        parser.add_argument(
            "--num-block",
            type=int,
            help="number of cycles to use for coordinate prediction",
        )
        parser.add_argument(
            "--num-gbf",
            type=int,
            help="number of cycles to use for coordinate prediction",
        )
        parser.add_argument(
            "--max-dist",
            type=float,
            metavar="D",
            help="pred homo lumo loss ratio",
        )
        parser.add_argument(
            "--dist-regular",
            type=float,
            metavar="D",
            help="pred homo lumo loss ratio",
        )

    def __init__(self, args, dictionary):
        super().__init__()
        base_architecture(args)
        self.args = args
        self.padding_idx = dictionary.pad()
        self.embed_tokens = nn.Embedding(
            len(dictionary), args.encoder_embed_dim, self.padding_idx
        )
        # self.shortest_path_embedder = nn.Embedding(
        #     512, args.encoder_attention_heads, 511
        # )
        # self.degree_embedder = nn.Embedding(512, args.encoder_embed_dim, 0)
        self.atom_feat_embedder = nn.Embedding(8 * 16, args.encoder_embed_dim, 0)
        self.bond_embedder = nn.Embedding(4 * 8, self.args.encoder_attention_heads, 0)
        self._num_updates = None
        self.encoder = BACKBONE[args.backbone](
            encoder_layers=args.encoder_layers,
            embed_dim=args.encoder_embed_dim,
            ffn_embed_dim=args.encoder_ffn_embed_dim,
            attention_heads=args.encoder_attention_heads,
            emb_dropout=args.emb_dropout,
            dropout=args.dropout,
            attention_dropout=args.attention_dropout,
            activation_dropout=args.activation_dropout,
            max_seq_len=args.max_seq_len,
            activation_fn=args.activation_fn,
            no_final_head_layer_norm=args.delta_pair_repr_norm_loss < 0,
            droppath_prob=0.1,
        )
        if args.masked_token_loss > 0:
            self.lm_head = MaskLMHead(
                embed_dim=args.encoder_embed_dim,
                output_dim=len(dictionary),
                activation_fn=args.activation_fn,
                weight=None,
            )

        K = args.num_gbf
        n_edge_type = len(dictionary) * len(dictionary)
        # self.edge_to_node_proj = nn.Linear(K, args.encoder_embed_dim)
        self.gbf_proj = NonLinearHead(
            K, args.encoder_attention_heads, args.activation_fn
        )
        self.gbf = GaussianLayer(K, n_edge_type, eps=self.args.gaussian_layer_eps)

        if args.masked_coord_loss > 0:
            self.pair2coord_proj = NonLinearHead(
                args.encoder_attention_heads, 1, args.activation_fn
            )
        if args.masked_dist_loss > 0:
            self.dist_head = DistanceHead(
                args.encoder_attention_heads, args.activation_fn
            )
        if args.lddt_loss > 0:
            self.lddt_head = NonLinearHead(
                args.encoder_embed_dim, 50, args.activation_fn
            )
        if args.p_homo_lumo_loss > 0:
            self.p_homo_lumo_head = NonLinearHead(
                args.encoder_embed_dim, 50, args.activation_fn
            )
        if self.args.homo_lumo_loss > 0:
            self.homo_lumo_head = NonLinearHead(
                args.encoder_embed_dim, 1, args.activation_fn
            )

        self.classification_heads = nn.ModuleDict()
        self.apply(init_bert_params)
        # zero moving at the beginning
        self.pair2coord_proj.zero_init()

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        return cls(args, task.dictionary)

    def forward(
        self,
        src_tokens,
        src_coord,
        # src_shortest_path,
        # src_degree,
        src_atom_feat,
        src_bond,
        src_edge_type,
        encoder_masked_tokens=None,
        features_only=False,
        classification_head_name=None,
        **kwargs
    ):

        if classification_head_name is not None:
            features_only = True

        assert src_coord.dtype == torch.float

        padding_mask = src_tokens.eq(self.padding_idx)
        if not padding_mask.any():
            padding_mask = None
        if padding_mask is not None:
            atom_num = (torch.sum(1 - padding_mask.type_as(src_coord), dim=1) - 1).view(
                -1, 1, 1, 1
            ) + 1e-5
        else:
            atom_num = float(src_coord.shape[1] - 1) + 1e-5

        src_x = (
            self.embed_tokens(src_tokens)
            # + self.degree_embedder(src_degree)
            + self.atom_feat_embedder(src_atom_feat).sum(dim=-2)
        )
        # attn_bias_2d = self.shortest_path_embedder(src_shortest_path)
        # sumup all bond features
        attn_bias_2d = self.bond_embedder(src_bond).sum(dim=-2)
        # TODO: 2d to node feature
        def get_dist_features(dist, et):
            n_node = dist.size(-1)
            gbf_feature = self.gbf(dist, et)
            gbf_result = self.gbf_proj(gbf_feature)
            graph_attn_bias = gbf_result
            graph_attn_bias = graph_attn_bias + attn_bias_2d
            graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
            graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
            return graph_attn_bias

        with utils.torch_seed(self.args.seed, self._num_updates, 2):
            if self.training:
                num_recycle = int(torch.randint(0, self.args.num_recycle + 1, (1,))[0])
            else:
                num_recycle = self.args.num_recycle

        def one_block(src_coord):
            src_distance = (src_coord.unsqueeze(1) - src_coord.unsqueeze(2)).norm(
                dim=-1
            )
            graph_attn_bias = get_dist_features(src_distance, src_edge_type)
            (
                encoder_rep,
                encoder_pair_rep,
                delta_encoder_pair_rep,
                x_norm,
                delta_encoder_pair_rep_norm,
            ) = self.encoder(
                src_x, padding_mask=padding_mask, attn_mask=graph_attn_bias
            )
            coords_emb = src_coord
            delta_pos = coords_emb.unsqueeze(1) - coords_emb.unsqueeze(2)
            attn_probs = self.pair2coord_proj(delta_encoder_pair_rep)
            coord_update = delta_pos / atom_num * attn_probs
            coord_update = torch.sum(coord_update, dim=2)
            encoder_coord = coords_emb + coord_update
            return (
                encoder_coord,
                encoder_rep,
                encoder_pair_rep,
                delta_encoder_pair_rep,
                x_norm,
                delta_encoder_pair_rep_norm,
            )

        def one_iteration(src_coord):
            for _ in range(self.args.num_block):
                (
                    src_coord,
                    encoder_rep,
                    encoder_pair_rep,
                    delta_encoder_pair_rep,
                    x_norm,
                    delta_encoder_pair_rep_norm,
                ) = one_block(src_coord)
            return (
                src_coord,
                encoder_rep,
                encoder_pair_rep,
                delta_encoder_pair_rep,
                x_norm,
                delta_encoder_pair_rep_norm,
            )

        for _ in range(num_recycle):
            with torch.no_grad():
                (
                    src_coord,
                    _,
                    _,
                    _,
                    _,
                    _,
                ) = one_iteration(src_coord)

        pre_coord = src_coord.clone()
        (
            encoder_coord,
            encoder_rep,
            encoder_pair_rep,
            delta_encoder_pair_rep,
            x_norm,
            delta_encoder_pair_rep_norm,
        ) = one_iteration(src_coord)
        encoder_pair_rep[encoder_pair_rep == float("-inf")] = 0

        logits = None
        encoder_distance = None

        if not features_only:
            if self.args.masked_token_loss > 0:
                logits = self.lm_head(encoder_rep, encoder_masked_tokens)
            if self.args.masked_dist_loss > 0:
                encoder_distance = self.dist_head(encoder_pair_rep)

        if classification_head_name is not None:
            logits = self.classification_heads[classification_head_name](encoder_rep)

        plddt = None
        if self.args.lddt_loss > 0:
            plddt = self.lddt_head(encoder_rep)

        p_homo_lumo = None
        if self.args.p_homo_lumo_loss > 0:
            p_homo_lumo = self.p_homo_lumo_head(encoder_rep[:, 0, :])

        pred_homo_lumo = None
        if self.args.homo_lumo_loss > 0:
            pred_homo_lumo = self.homo_lumo_head(encoder_rep[:, 0, :])

        return (
            logits,
            encoder_distance,
            encoder_coord,
            x_norm,
            delta_encoder_pair_rep_norm,
            pre_coord,
            plddt,
            p_homo_lumo,
            pred_homo_lumo,
        )

    def register_classification_head(
        self, name, num_classes=None, inner_dim=None, **kwargs
    ):
        """Register a classification head."""
        if name in self.classification_heads:
            prev_num_classes = self.classification_heads[name].out_proj.out_features
            prev_inner_dim = self.classification_heads[name].dense.out_features
            if num_classes != prev_num_classes or inner_dim != prev_inner_dim:
                logger.warning(
                    're-registering head "{}" with num_classes {} (prev: {}) '
                    "and inner_dim {} (prev: {})".format(
                        name, num_classes, prev_num_classes, inner_dim, prev_inner_dim
                    )
                )
        self.classification_heads[name] = ClassificationHead(
            input_dim=self.args.encoder_embed_dim,
            inner_dim=inner_dim or self.args.encoder_embed_dim,
            num_classes=num_classes,
            activation_fn=self.args.pooler_activation_fn,
            pooler_dropout=self.args.pooler_dropout,
        )

    def set_num_updates(self, num_updates):
        """State from trainer to pass along to model at every update."""
        self._num_updates = num_updates

    def get_num_updates(self):
        return self._num_updates

    def __make_input_float__(self):
        self.gbf = self.gbf.float()
        self.pair2coord_proj = self.pair2coord_proj.float()
        self.homo_lumo_head = self.homo_lumo_head.float()

    def half(self):
        super().half()
        self.dtype = torch.half
        self.__make_input_float__()
        return self

    def bfloat16(self):
        super().bfloat16()
        self.dtype = torch.bfloat16
        self.__make_input_float__()
        return self


class MaskLMHead(nn.Module):
    """Head for masked language modeling."""

    def __init__(self, embed_dim, output_dim, activation_fn, weight=None):
        super().__init__()
        self.dense = nn.Linear(embed_dim, embed_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.layer_norm = LayerNorm(embed_dim)

        if weight is None:
            weight = nn.Linear(embed_dim, output_dim, bias=False).weight
        self.weight = weight
        self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, features, masked_tokens=None, **kwargs):
        # Only project the masked tokens while training,
        # saves both memory and computation
        if masked_tokens is not None:
            features = features[masked_tokens, :]

        x = self.dense(features)
        x = self.activation_fn(x)
        x = self.layer_norm(x)
        # project back to size of vocabulary with bias
        x = F.linear(x, self.weight) + self.bias
        return x


class ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(
        self,
        input_dim,
        inner_dim,
        num_classes,
        activation_fn,
        pooler_dropout,
    ):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class NonLinearHead(nn.Module):
    """Head for simple classification tasks."""

    def __init__(
        self,
        input_dim,
        out_dim,
        activation_fn,
        hidden=None,
    ):
        super().__init__()
        hidden = input_dim if not hidden else hidden
        self.linear1 = nn.Linear(input_dim, hidden)
        self.linear2 = nn.Linear(hidden, out_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)

    def forward(self, x):
        x = x.type_as(self.linear1.weight)
        x = self.linear1(x)
        x = self.activation_fn(x)
        x = self.linear2(x)
        return x

    def zero_init(self):
        nn.init.zeros_(self.linear2.weight)
        nn.init.zeros_(self.linear2.bias)


class DistanceHead(nn.Module):
    def __init__(
        self,
        heads,
        activation_fn,
    ):
        super().__init__()
        self.dense = nn.Linear(heads, heads)
        self.layer_norm = nn.LayerNorm(heads)
        self.out_proj = nn.Linear(heads, 1)
        self.activation_fn = utils.get_activation_fn(activation_fn)

    def forward(self, x):
        bsz, seq_len, seq_len, _ = x.size()
        # x[x == float('-inf')] = 0
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.layer_norm(x)
        x = self.out_proj(x).view(bsz, seq_len, seq_len)
        x = (x + x.transpose(-1, -2)) * 0.5
        return x


@torch.jit.script
def gaussian(x, mean, std):
    pi = 3.14159
    a = (2 * pi) ** 0.5
    return torch.exp(-0.5 * (((x - mean) / std) ** 2)) / (a * std)


class GaussianLayer(nn.Module):
    def __init__(self, K=128, edge_types=1024, eps=1e-5):
        super().__init__()
        self.K = K
        self.means = nn.Embedding(1, K)
        self.stds = nn.Embedding(1, K)
        self.mul = nn.Embedding(edge_types, 1)
        self.bias = nn.Embedding(edge_types, 1)
        self.eps = eps
        nn.init.uniform_(self.means.weight, 0, 3)
        nn.init.uniform_(self.stds.weight, 0, 3)
        nn.init.constant_(self.bias.weight, 0)
        nn.init.constant_(self.mul.weight, 1)

    def forward(self, x, edge_type):
        mul = self.mul(edge_type)
        bias = self.bias(edge_type)
        x = mul * x.unsqueeze(-1) + bias
        x = x.expand(-1, -1, -1, self.K)
        mean = self.means.weight.float().view(-1)
        std = self.stds.weight.float().view(-1).abs() + self.eps
        return gaussian(x.float(), mean, std)


@register_model_architecture("unimol_pcq", "unimol_pcq")
def base_architecture(args):
    args.encoder_layers = getattr(args, "encoder_layers", 12)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4096)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 64)
    args.dropout = getattr(args, "dropout", 0.0)
    args.emb_dropout = getattr(args, "emb_dropout", 0.0)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.pooler_dropout = getattr(args, "pooler_dropout", 0.0)
    args.max_seq_len = getattr(args, "max_seq_len", 512)
    args.activation_fn = getattr(args, "activation_fn", "gelu")
    args.pooler_activation_fn = getattr(args, "pooler_activation_fn", "tanh")
    args.post_ln = getattr(args, "post_ln", False)
    args.backbone = getattr(args, "backbone", "transformer")
    args.masked_token_loss = getattr(args, "masked_token_loss", -1.0)
    args.masked_coord_loss = getattr(args, "masked_coord_loss", 2.0)
    args.masked_dist_loss = getattr(args, "masked_dist_loss", 0.5)
    args.x_norm_loss = getattr(args, "x_norm_loss", 0.01)
    args.delta_pair_repr_norm_loss = getattr(args, "delta_pair_repr_norm_loss", 0.01)
    args.lddt_loss = getattr(args, "lddt_loss", 0.01)
    args.p_homo_lumo_loss = getattr(args, "p_homo_lumo", 0.01)
    args.gaussian_layer_eps = getattr(args, "gaussian_layer_eps", 1e-2)
    args.homo_lumo_loss = getattr(args, "homo_lumo_loss", 1.0)
    args.num_recycle = getattr(args, "num_recycle", 3)
    args.num_block = getattr(args, "num_block", 2)
    args.num_gbf = getattr(args, "num_gbf", 128)
    args.max_dist = getattr(args, "max_dist", 1.0)
    args.dist_regular_loss = getattr(args, "dist_regular_loss", 0.01)


@register_model_architecture("unimol_pcq", "unimol_pcq_base")
def unimol_base_architecture(args):
    base_architecture(args)
