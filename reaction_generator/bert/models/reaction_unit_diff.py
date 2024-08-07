import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from unicore import utils
from unicore.models import BaseUnicoreModel, register_model, register_model_architecture
from unicore.data import Dictionary
from unicore.modules import LayerNorm, TransformerEncoder,TransformerDecoder, init_bert_params
from ..modules import TransformerEncoderQuery, TransformerDecoderQueryDiff
from typing import Callable, Optional, Dict, Tuple, Any, NamedTuple, List
import numpy as np
import math
from torch import Tensor


logger = logging.getLogger(__name__)

torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_override_can_fuse_on_cpu(True)
torch._C._jit_override_can_fuse_on_gpu(True)

@register_model("reaction_unit_diff")
class ReactionUnitDiffModel(BaseUnicoreModel):
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
            "--decoder-layers", type=int, metavar="L", help="num decoder layers"
        )
        parser.add_argument(
            "--decoder-embed-dim",
            type=int,
            metavar="H",
            help="decoder embedding dimension",
        )
        parser.add_argument(
            "--decoder-attention-heads",
            type=int,
            metavar="A",
            help="num decoder attention heads",
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
            "--emb-dropout", type=float, metavar="D", help="dropout probability for embeddings"
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
            "--position-type",
            default='normal',
            choices=['sinusoidal', 'relative', 'normal'],
            help="noise type in coordinate noise",
        )
        parser.add_argument(
            "--vq-embedding-nums",
            type=int,
            help="num vq embedding",
        )
        parser.add_argument(
            "--vq-sampling-tau",
            type=float,
            help="tau of vq gumbel softmax",
        )
        parser.add_argument(
            "--transformer-type",
            default='normal',
            choices=['simple', 'normal'],
            help="noise type in coordinate noise",
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
            "--contrastive-global-negative", action='store_true', help="use contrastive learning or not"
        )
        parser.add_argument(
            "--rel-pos", action='store_true', help="relative pos in transformer or not"
        )
        parser.add_argument(
            "--auto-regressive", action='store_true', help="use auto regressive generative or not"
        )
        parser.add_argument(
            "--class-embedding", action='store_true', help="use class embedding or not"
        )
        parser.add_argument(
            "--vq-embedding", action='store_true', help="use vq embedding or not"
        )
        parser.add_argument(
            "--recycle-num", type=int, metavar="L", help="num time recycle"
        )
        parser.add_argument(
            "--local-attn-size", type=int, help="attention window for decoder to reduce teacher-forcing denpendency"
        )
        parser.add_argument(
            "--use-decoder", action='store_true', help="use decoder or not"
        )
        parser.add_argument(
            "--smoothl1-beta",
            default=1.0,
            type=float,
            help="beta in pair distance smoothl1 loss"
        )
        parser.add_argument(
            "--mse-loss-weight",
            default = -1.0,
            type = float,
            help = "mse loss weight"
        )
        parser.add_argument(
            "--ce-loss-weight",
            default = -1.0,
            type = float,
            help = "cross entropy loss weight"
        )
        parser.add_argument(
            "--length-loss-weight",
            default = -1.0,
            type = float,
            help = "length loss weight"
        )


    def __init__(self, args, dictionary):
        super().__init__()
        base_architecture(args)
        self.args = args
        self.padding_idx = dictionary.pad()
        self.embed_tokens = nn.Embedding(len(dictionary), args.encoder_embed_dim, self.padding_idx)
        self.encoder = TransformerEncoderQuery(
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
            rel_pos = args.rel_pos,
        )
        self.encoder_rec = TransformerEncoderQuery(
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
            rel_pos = args.rel_pos,
        )
        # self.lm_head = MaskLMHead(embed_dim=args.encoder_embed_dim,
        #     output_dim=len(dictionary),
        #     activation_fn=args.activation_fn,
        #     weight=None,
        # )
        self.auto_regressive = args.auto_regressive
        self.use_class_embedding = args.class_embedding
        self.use_vq_embedding = args.vq_embedding
        self.use_decoder = args.use_decoder
        # self.embed_positions = self.get_position_embedding('test', args.max_seq_len, args.encoder_embed_dim)
        if self.use_vq_embedding:
            self.vq_sampling_tau = args.vq_sampling_tau
            self.vq_embeddings = nn.Embedding(args.vq_embedding_nums, args.encoder_embed_dim)
            self.attention_head_tar = AttentionVQ(args.dropout)
            self.attention_head_rec = AttentionVQ(args.dropout)
            self.classification_head = ClassificationHead(args.encoder_embed_dim, args.encoder_embed_dim, 10, "gelu", args.pooler_dropout)
            self.length_classification_head = ClassificationHead(args.encoder_embed_dim, args.encoder_embed_dim, 80, "gelu", args.pooler_dropout)
            # self.gumbel_softmax = GumbelSoftmax()
            self.recycle_num = args.recycle_num

        self.embed_positions = self.get_position_embedding(args.position_type, args.max_seq_len, args.encoder_embed_dim)
        # self.embed_positions = nn.Embedding(args.max_seq_len, args.encoder_embed_dim)
        if self.use_class_embedding:
            self.class_embedding = nn.Embedding(100, args.encoder_embed_dim)           
        if args.auto_regressive:
            self.use_decoder = True
        if self.use_decoder:
            # self.decoder_embed_positions = self.get_position_embedding('test', args.max_seq_len, args.decoder_embed_dim)
            # self.decoder_embed_positions = nn.Embedding(args.max_seq_len, args.decoder_embed_dim)
            self.decoder_embed_positions = self.get_position_embedding(args.position_type, args.max_seq_len, args.decoder_embed_dim)
            self.decoder_embed_tokens = self.embed_tokens  
            # self.decoder_embed_tokens = nn.Embedding(len(dictionary), args.decoder_embed_dim, self.padding_idx)
            self.decoder = TransformerDecoderQueryDiff(
                decoder_layers=args.decoder_layers,
                embed_dim=args.decoder_embed_dim,
                ffn_embed_dim=args.decoder_ffn_embed_dim,
                attention_heads=args.decoder_attention_heads,
                emb_dropout=args.emb_dropout,
                dropout=args.dropout,
                attention_dropout=args.attention_dropout,
                activation_dropout=args.activation_dropout,
                # emb_dropout=0,
                # dropout=0,
                # attention_dropout=0,
                # activation_dropout=0,
                max_seq_len=args.max_seq_len,
                activation_fn=args.activation_fn,
                rel_pos = args.rel_pos,
                auto_regressive=args.auto_regressive,
                local_attn_size=args.local_attn_size,
            )            
            self.decoder_lm_head = MaskLMHead(embed_dim=args.decoder_embed_dim,
                output_dim=len(dictionary),
                activation_fn=args.activation_fn,
                weight=None,
            )
            self.mid_emb_class = ClassMidEmbHead(embed_dim=args.decoder_embed_dim,
                output_dim=len(dictionary),
                activation_fn=args.activation_fn,
                weight=None,)

            # self.decoder_lm_head = nn.Linear(args.decoder_embed_dim, len(dictionary))
        self.classification_heads = nn.ModuleDict()

        self.apply(init_bert_params)

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        return cls(args, task.dictionary)


    def get_position_embedding(self, position_type, max_seq_len, embed_dim):

        if position_type == "sinusoidal":
            pe = torch.zeros(max_seq_len, embed_dim)
            position = torch.arange(0, max_seq_len).unsqueeze(1)
            div_term = torch.exp((torch.arange(0, embed_dim, 2, dtype=torch.float) *
                                -(math.log(10000.0) / embed_dim)))
            pe[:, 0::2] = torch.sin(position.float() * div_term)
            pe[:, 1::2] = torch.cos(position.float() * div_term)
            pe1 = nn.Embedding(max_seq_len, embed_dim)
            pe1.weight = nn.Parameter(pe, requires_grad=False) 
            return pe1

        elif position_type == "relative":
            # relative_pe = nn.Embedding(max_seq_len * 2 + 2, embed_dim)
            pe = torch.zeros(max_seq_len, embed_dim//2)
            position = torch.arange(0, max_seq_len).unsqueeze(1)
            div_term = torch.exp((torch.arange(0, (embed_dim//2), 2, dtype=torch.float) *
                                -(math.log(10000.0) / (embed_dim//2))))
            pe[:, 0::2] = torch.sin(position.float() * div_term)
            pe[:, 1::2] = torch.cos(position.float() * div_term) 
            pe1 = nn.Embedding(max_seq_len, embed_dim//2)
            pe1.weight = nn.Parameter(pe, requires_grad=False) 
            relative = nn.Embedding(max_seq_len, embed_dim//2)  
            relative_pe = torch.cat((relative,pe1), -1)   
            return relative_pe
        else:
            return nn.Embedding(max_seq_len, embed_dim)

    def forward(
        self,
        src_tokens,
        reverse_src_tokens,
        reverse_tgt_tokens,
        masked_tokens,
        features_only=False,
        classification_head_name=None,
        **kwargs
    ):
        if classification_head_name is not None:
            features_only = True

        # target
        padding_mask = reverse_src_tokens.eq(self.padding_idx)
        if not padding_mask.any():
            padding_mask = None
        x0 = self.embed_tokens(reverse_src_tokens)
        seq_len = reverse_src_tokens.size(1)
        x = x0 + self.embed_positions.weight[:seq_len, :]
        # if self.use_class_embedding:
        #     x[:,0,:] += self.class_embedding(relation_type)
        if padding_mask is not None:
            x1 = x * (1 - padding_mask.unsqueeze(-1).type_as(x))
        encoder_rep = self.encoder(x1, padding_mask=padding_mask, attn_mask=None)

        # rec
        padding_mask_rec = src_tokens.eq(self.padding_idx)
        if not padding_mask_rec.any():
            padding_mask_rec = None
        x_rec = self.decoder_embed_tokens(src_tokens)
        seq_len2 = src_tokens.size(1)
        x_rec += self.decoder_embed_positions.weight[:seq_len2, :]
        # if self.use_class_embedding:
        #     x[:,0,:] += self.class_embedding(relation_type)

        if padding_mask_rec is not None:
            x_rec = x_rec * (1 - padding_mask_rec.unsqueeze(-1).type_as(x_rec))
        encoder_rep_rec = self.encoder_rec(x_rec, padding_mask=padding_mask_rec, attn_mask=None)

        classifier_logits = None
        vq1 = None
        vq2 = None
        if self.use_vq_embedding:
            vq1 = self.attention_head_tar(encoder_rep, self.vq_embeddings)
            vq2 = self.attention_head_rec(encoder_rep_rec, self.vq_embeddings)
            # encoder_vq = self.gumbel_softmax(encoder_rep, self.vq_embeddings)
            encoder_vq = vq1
            classifier_logits = self.classification_head(encoder_vq)
            length_class_logits = self.length_classification_head(encoder_rep[:,0,:].clone())
            encoder_rep[:,0,:] = encoder_vq

        decoder_outprob = None
        # vae_kl_loss = None
        if self.use_decoder:
            ### length 训练时 可以 teacher forcing，推理时需要 采样一个值
            decoder_padding_mask = reverse_tgt_tokens.eq(self.padding_idx)
            if not decoder_padding_mask.any():
                decoder_padding_mask = None
            x_decoder2 = self.decoder_embed_tokens(reverse_tgt_tokens)
            seq_len = reverse_tgt_tokens.size(1)
            x_decoder = torch.zeros_like(x_decoder2)
            x_decoder += self.decoder_embed_positions.weight[:seq_len, :]
            if decoder_padding_mask is not None:
                x_decoder = x_decoder * (1 - decoder_padding_mask.unsqueeze(-1).type_as(x_decoder))
            
            encoder_cls = encoder_rep

            mid_x_decoder_list = [] 

            x_decoder = self.decoder(x_decoder, padding_mask = decoder_padding_mask, encoder_padding_mask = padding_mask, encoder_out=encoder_cls, attn_mask=None)
            for i in range(self.recycle_num):

                if decoder_padding_mask is not None:
                    x_decoder = x_decoder * (1 - decoder_padding_mask.unsqueeze(-1).type_as(x_decoder))
                
                x_decoder2 = x_decoder.detach()
                x_decoder_class = self.mid_emb_class(x_decoder)
                batch_size, seq_len, num_clas = x_decoder_class.shape
                x_decoder_class2 = x_decoder_class.view(-1, num_clas)
                x_decoder_class3 = x_decoder_class2.clone()
                x_decoder_class3[:,0] = 0
                x_decoder_class_ind = torch.multinomial(x_decoder_class3, 1)
                x_decoder_class_ind = x_decoder_class_ind.view(batch_size, seq_len)
                x_decoder = self.decoder_embed_tokens(x_decoder_class_ind) 

                mid_x_decoder_list.append(x_decoder_class)

                x_decoder = self.decoder(x_decoder, padding_mask = decoder_padding_mask, encoder_padding_mask = padding_mask, encoder_out=encoder_cls, attn_mask=None)

            decoder_outprob = self.decoder_lm_head(x_decoder)
            
        # contrast_out = None
        # if not features_only:
        #     logits = self.lm_head(encoder_rep, masked_tokens)
        return length_class_logits, classifier_logits, decoder_outprob, vq1, vq2, mid_x_decoder_list

    def forward_encoder(
        self,
        reverse_src_tokens,
        relation_type,
        features_only=False,
        classification_head_name=None,
        **kwargs
    ):
        if classification_head_name is not None:
            features_only = True

        # target
        padding_mask = reverse_src_tokens.eq(self.padding_idx)
        if not padding_mask.any():
            padding_mask = None
        x0 = self.embed_tokens(reverse_src_tokens)
        seq_len = reverse_src_tokens.size(1)
        x = x0 + self.embed_positions.weight[:seq_len, :]
        # if self.use_class_embedding:
        #     x[:,0,:] += self.class_embedding(relation_type)
        if padding_mask is not None:
            x1 = x * (1 - padding_mask.unsqueeze(-1).type_as(x))
        encoder_rep = self.encoder(x1, padding_mask=padding_mask, attn_mask=None)

        classifier_logits = None
        vq1 = None
        if self.use_vq_embedding:
            vq1 = self.attention_head_tar(encoder_rep, self.vq_embeddings)
            # encoder_vq = self.gumbel_softmax(encoder_rep, self.vq_embeddings)
            encoder_vq = vq1
            classifier_logits = self.classification_head(encoder_vq)
            length_class_logits = self.length_classification_head(encoder_vq)
            encoder_rep[:,0,:] = encoder_vq
        
        encoder_cls = encoder_rep
        
        return encoder_cls, padding_mask
    
    def forward_decoder(
        self,
        reverse_tgt_tokens,
        encoder_cls,
        temperature,
        padding_mask,
        **kwargs
    ):

        decoder_outprob = None
        # vae_kl_loss = None
        # if self.use_decoder:
        ### 训练时 可以 teacher forcing，推理时需要 采样一个值
        decoder_padding_mask = reverse_tgt_tokens.eq(self.padding_idx)
        if not decoder_padding_mask.any():
            decoder_padding_mask = None
        x_decoder2 = self.decoder_embed_tokens(reverse_tgt_tokens)
        seq_len = reverse_tgt_tokens.size(1)
        x_decoder = torch.zeros_like(x_decoder2)
        x_decoder += self.decoder_embed_positions.weight[:seq_len, :]
        if decoder_padding_mask is not None:
            x_decoder = x_decoder * (1 - decoder_padding_mask.unsqueeze(-1).type_as(x_decoder))

        for i in range(self.recycle_num):
            x_decoder_class = self.mid_emb_class(x_decoder)
            batch_size, seq_len, num_clas = x_decoder_class.shape
            x_decoder_class = x_decoder_class.view(-1, num_clas)
            x_decoder_class_ind = torch.multinomial(x_decoder_class, 1)
            x_decoder_class_ind = x_decoder_class_ind.view(batch_size, seq_len)
            x_decoder = self.decoder_embed_tokens(x_decoder_class_ind)

        x_decoder = self.decoder(x_decoder, padding_mask = decoder_padding_mask, encoder_padding_mask = padding_mask, encoder_out=encoder_cls, attn_mask=None)
        decoder_outprob = self.decoder_lm_head(x_decoder)
        probs = self.get_normalized_probs(
            decoder_outprob, temperature, log_probs=True, sample=None
        )     

        return probs, None



    def get_normalized_probs(self, net_output, temperature, log_probs, sample=None):
        """Get normalized probabilities (or log probs) from a net's output."""
        logits = net_output#[0].float()
        if log_probs:
            return F.log_softmax(logits, dim=-1)
        else:
            return torch.log(F.softmax(logits/temperature, dim=-1))

    # def register_classification_head(
    #     self, name, num_classes=None, inner_dim=None, **kwargs
    # ):
    #     """Register a classification head."""
    #     if name in self.classification_heads:
    #         prev_num_classes = self.classification_heads[name].out_proj.out_features
    #         prev_inner_dim = self.classification_heads[name].dense.out_features
    #         if num_classes != prev_num_classes or inner_dim != prev_inner_dim:
    #             logger.warning(
    #                 're-registering head "{}" with num_classes {} (prev: {}) '
    #                 "and inner_dim {} (prev: {})".format(
    #                     name, num_classes, prev_num_classes, inner_dim, prev_inner_dim
    #                 )
    #             )
    #     self.classification_heads[name] = ClassificationHead(
    #         input_dim=self.args.encoder_embed_dim,
    #         inner_dim=inner_dim or self.args.encoder_embed_dim,
    #         num_classes=num_classes,
    #         activation_fn=self.args.pooler_activation_fn,
    #         pooler_dropout=self.args.pooler_dropout,
    #     )


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


class ClassMidEmbHead(nn.Module):
    """Head for classifying embedding of middle process."""

    def __init__(self, embed_dim, output_dim, activation_fn, weight=None):
        super().__init__()
        self.dense = nn.Linear(embed_dim, embed_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.layer_norm = LayerNorm(embed_dim)
        if weight is None:
            weight = nn.Linear(embed_dim, output_dim, bias=False).weight
        self.weight = weight
        self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = self.activation_fn(x)
        x = self.layer_norm(x)
        # project back to size of vocabulary with bias
        x = F.linear(x, self.weight) + self.bias
        x = F.softmax(x, dim=-1)
        return x


class AttentionVQ(nn.Module):
    def __init__(self, dropout, tau = 1.0):
        super().__init__()
        self.dropout = dropout
        self.tau = tau
    def forward(self, inputs, vq_embeddings):
        
        inputs_mean = torch.mean(inputs, 1)
        attn_weights = torch.mm(inputs_mean, vq_embeddings.weight.transpose(0, 1))
        attn_probs = F.softmax(attn_weights/self.tau, dim=-1)
        attn_vq = torch.mm(attn_probs, vq_embeddings.weight)
        return attn_vq


# class GumbelSoftmax(nn.Module):
#     # index = one_hot(argmax(logits)) continuous version
#     def __init__(self, soft_sw, dropout, tau, eps=1e-20):
#         super().__init__()
#         self.soft_sw = soft_sw
#         self.dropout = dropout
#         self.tau = tau
#         self.eps = eps

#     def forward(self, inputs, vq_embeddings, **kwargs):
#         inputs_mean = torch.mean(inputs, 1)
#         attn_weights = torch.mm(inputs_mean, vq_embeddings.transpose(0, 1))
#         logits = F.softmax(attn_weights, dim = -1)
#         y_index = self.gumbel_softmax_sample(logits, self.tau)
        
#         shape = attn_weights.size()
#         if self.soft_sw:
#             y_index = y_index.view(-1, shape[-1])
#             return torch.mm(y_index, vq_embeddings)
#         else:
#             _, ind = y_index.max(dim=-1)
#             y_hard = torch.zeros_like(y_index).view(-1, shape[-1])
#             y_hard.scatter_(1, ind.view(-1, 1), 1)
#             y_hard = y_hard.view(*shape)
#             # Set gradients w.r.t. y_hard gradients w.r.t. y
#             y_hard = (y_hard - y_index).detach() + y_index
#             y_hard.view(-1, shape[-1])
#             y_index = y_hard
#             return vq_embeddings[y_index]

#     def sample_gumbel(self, shape):
#         u = torch.rand(shape)
#         return -torch.log(-torch.log(u + self.eps) + self.eps)

#     def gumbel_softmax_sample(self, logits):
#         y = logits + self.sample_gumbel(logits.size())
#         return F.softmax(y/self.tau, dim=-1)


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
        # x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = features    
        x = self.dropout(x)
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x



@register_model_architecture("reaction_unit_diff", "reaction_unit_diff")
def base_architecture(args):
    args.encoder_layers = getattr(args, "encoder_layers", 15)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 64)
    args.dropout = getattr(args, "dropout", 0.1)
    args.emb_dropout = getattr(args, "emb_dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.pooler_dropout = getattr(args, "pooler_dropout", 0.0)
    args.max_seq_len = getattr(args, "max_seq_len", 512)
    args.activation_fn = getattr(args, "activation_fn", "gelu")
    args.rel_pos = getattr(args, "rel_pos", False)
    args.pooler_activation_fn = getattr(args, "pooler_activation_fn", "tanh")
    args.post_ln = getattr(args, "post_ln", False)
    args.contrastive_global_negative = getattr(args, "contrastive_global_negative", False)
    args.auto_regressive = getattr(args, "auto_regressive", False)
    args.use_decoder = getattr(args, "use_decoder", False)
    args.class_embedding = getattr(args, "class_embedding", False)
    args.vq_embedding = getattr(args, "vq_embedding", False)
    args.vq_embedding_nums = getattr(args, "vq_embedding_nums", 2048)
    args.vq_sampling_tau = getattr(args, "vq_sampling_tau", 0.2)
    args.recycle_num = getattr(args, "recycle_num", 3)

    args.decoder_layers = getattr(args, "decoder_layers", 15)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 512)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 2048)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 64)
    args.decoder_loss = getattr(args, "decoder_loss", 1)
    args.decoder_loss_weight = getattr(args, "mse_loss_weight", -1) 
    args.ce_loss_weight = getattr(args, "ce_loss_weight", -1)    
    args.length_loss_weight = getattr(args, "length_loss_weight", -1)   
    args.local_attn_size = getattr(args, "local_attn_size", -1)
    

@register_model_architecture("reaction_unit_diff", "reaction_unit_diff_base")
def reaction_unit_base_architecture(args):
    base_architecture(args)
