import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from unicore import utils
from unicore.models import BaseUnicoreModel, register_model, register_model_architecture
from unicore.data import Dictionary
from unicore.modules import LayerNorm, TransformerEncoder,TransformerDecoder, init_bert_params
from ..modules import TransformerEncoderQuery, TransformerDecoderQuery
from typing import Callable, Optional, Dict, Tuple, Any, NamedTuple, List
import numpy as np
import math
from torch import Tensor

logger = logging.getLogger(__name__)

torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_override_can_fuse_on_cpu(True)
torch._C._jit_override_can_fuse_on_gpu(True)

@register_model("unimol_reaction_forward")
class UniMolReactionForwardModel(BaseUnicoreModel):
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
            "--auto-regressive", action='store_true', help="use auto regressive generative or not"
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
        )
        self.lm_head = MaskLMHead(embed_dim=args.encoder_embed_dim,
            output_dim=len(dictionary),
            activation_fn=args.activation_fn,
            weight=None,
        )
        self.auto_regressive = args.auto_regressive
        self.use_decoder = args.use_decoder
        # self.embed_positions = self.get_position_embedding('test', args.max_seq_len, args.encoder_embed_dim)

        self.embed_positions = self.get_position_embedding(args.position_type, args.max_seq_len, args.encoder_embed_dim)
        # self.embed_positions = nn.Embedding(args.max_seq_len, args.encoder_embed_dim)
        if args.auto_regressive:
            self.use_decoder = True
        if self.use_decoder:
            # self.decoder_embed_positions = self.get_position_embedding('test', args.max_seq_len, args.decoder_embed_dim)
            # self.decoder_embed_positions = nn.Embedding(args.max_seq_len, args.decoder_embed_dim)
            self.decoder_embed_positions = self.get_position_embedding(args.position_type, args.max_seq_len, args.decoder_embed_dim)
            self.decoder_embed_tokens = self.embed_tokens  #FFFFFF
            # self.decoder_embed_tokens = nn.Embedding(len(dictionary), args.decoder_embed_dim, self.padding_idx)
            self.decoder = TransformerDecoderQuery(
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
                auto_regressive=args.auto_regressive,
                local_attn_size=args.local_attn_size,
            )              
            self.decoder_lm_head = MaskLMHead(embed_dim=args.decoder_embed_dim,
                output_dim=len(dictionary),
                activation_fn=args.activation_fn,
                weight=None,
            )
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
        decoder_src_tokens,
        masked_tokens,
        features_only=False,
        classification_head_name=None,
        **kwargs
    ):
        if classification_head_name is not None:
            features_only = True
        padding_mask = src_tokens.eq(self.padding_idx)
        if not padding_mask.any():
            padding_mask = None
        x = self.embed_tokens(src_tokens)
        seq_len = src_tokens.size(1)
        x = x * math.sqrt(x.shape[-1])  # FFFFFF
        x += self.embed_positions.weight[:seq_len, :]
        if padding_mask is not None:
            x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))
        encoder_rep = self.encoder(x, padding_mask=padding_mask, attn_mask=None)
        # print('test encoder_rep: ', encoder_rep.shape, encoder_rep)
        decoder_outprob = None
        vae_kl_loss = None
        if self.use_decoder:
            decoder_padding_mask = decoder_src_tokens.eq(self.padding_idx)
            if not decoder_padding_mask.any():
                decoder_padding_mask = None
            x_decoder = self.decoder_embed_tokens(decoder_src_tokens)
            x_decoder = x_decoder * math.sqrt(x_decoder.shape[-1])  # FFFFFF
            seq_len = decoder_src_tokens.size(1)
            x_decoder += self.decoder_embed_positions.weight[:seq_len, :]
            if decoder_padding_mask is not None:
                x_decoder = x_decoder * (1 - decoder_padding_mask.unsqueeze(-1).type_as(x_decoder))
            
            encoder_cls = encoder_rep
            decoder_rep = self.decoder(x_decoder, padding_mask = decoder_padding_mask, encoder_padding_mask = padding_mask, encoder_out=encoder_cls, attn_mask=None)
            decoder_outprob = self.decoder_lm_head(decoder_rep)
        contrast_out = None
        
        if not features_only:
            logits = self.lm_head(encoder_rep, masked_tokens)

        return logits, decoder_outprob, contrast_out, vae_kl_loss

    def forward_encoder(
        self,
        src_tokens,
        features_only=False,
        classification_head_name=None,
        **kwargs
    ):
        # print('test type0: ', self.args.transformer_type, self.args.position_type)
        # print('wz check src_tokens0',src_tokens.shape)
        if classification_head_name is not None:
            features_only = True

        padding_mask = src_tokens.eq(self.padding_idx)
        if not padding_mask.any():
            padding_mask = None
        x = self.embed_tokens(src_tokens)
        seq_len = src_tokens.size(1)
        x = x * math.sqrt(x.shape[-1])  # FFFFFF
        x += self.embed_positions.weight[:seq_len, :]
        if padding_mask is not None:
            x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))
        encoder_rep = self.encoder(x, padding_mask=padding_mask, attn_mask=None)
        vae_kl_loss = None
        if self.use_decoder:
            # encoder_cls = encoder_rep[:,0,:]
            encoder_cls = encoder_rep
        # encoder_cls = latent_z.transpose(0, 1).unsqueeze(1)

        return encoder_cls, padding_mask
    
    def forward_decoder(
        self,
        decoder_src_tokens,
        encoder_cls,
        temperature,
        encoder_padding_mask,
        **kwargs
    ):
        # print('test type1: ', self.args.transformer_type, self.args.position_type)

        decoder_outprob = None
        vae_kl_loss = None
        if self.use_decoder:          
            decoder_padding_mask = decoder_src_tokens.eq(self.padding_idx)
            if not decoder_padding_mask.any():
                decoder_padding_mask = None
            x_decoder = self.decoder_embed_tokens(decoder_src_tokens)
            x_decoder = x_decoder * math.sqrt(x_decoder.shape[-1])  # FFFFFF
            seq_len = decoder_src_tokens.size(1)
            # print('test seq_len: ', seq_len, x_decoder.shape, decoder_src_tokens.shape, x_decoder, decoder_src_tokens)
            x_decoder += self.decoder_embed_positions.weight[:seq_len, :]
            if decoder_padding_mask is not None:
                x_decoder = x_decoder * (1 - decoder_padding_mask.unsqueeze(-1).type_as(x_decoder))
            # x_decoder[:,-1,:].unsqueeze(1), 
            # decoder_rep = self.decoder(x_decoder[:,-1,:].unsqueeze(1), padding_mask=decoder_padding_mask, encoder_out=encoder_cls, attn_mask=None)
            # import time
            # torch.cuda.synchronize()
            # time_0 = time.time() 
            
            decoder_rep = self.decoder(x_decoder[:,-1,:].unsqueeze(1), padding_mask=decoder_padding_mask, encoder_padding_mask=encoder_padding_mask, encoder_out=encoder_cls, attn_mask=None)
            # torch.cuda.synchronize()
            # time_1 = time.time()
            # print('test decoder 1: ', time_1 - time_0)
            decoder_outprob = self.decoder_lm_head(decoder_rep)
            # torch.cuda.synchronize()
            # time_2 = time.time()   
            # print('test decoder 2: ', time_2 - time_1)
            probs = self.get_normalized_probs(
                decoder_outprob, temperature, log_probs=True, sample=None
            )   
            # torch.cuda.synchronize()
            # time_3 = time.time() 
            # print('test decoder 3: ', time_3 - time_2)        
            probs = probs[:, -1, :]
        return probs, None

    def get_normalized_probs(self, net_output, temperature, log_probs, sample=None):
        """Get normalized probabilities (or log probs) from a net's output."""
        logits = net_output#[0].float()
        if log_probs:
            return F.log_softmax(logits, dim=-1)
        else:
            return torch.log(F.softmax(logits/temperature, dim=-1))

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



@register_model_architecture("unimol_reaction_forward", "unimol_reaction_forward")
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
    args.pooler_activation_fn = getattr(args, "pooler_activation_fn", "tanh")
    args.post_ln = getattr(args, "post_ln", False)
    args.contrastive_global_negative = getattr(args, "contrastive_global_negative", False)
    args.auto_regressive = getattr(args, "auto_regressive", False)
    args.use_decoder = getattr(args, "use_decoder", False)

    args.decoder_layers = getattr(args, "decoder_layers", 15)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 512)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 2048)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 64)
    args.decoder_loss = getattr(args, "decoder_loss", 1)
    args.local_attn_size = getattr(args, "local_attn_size", -1)
    

@register_model_architecture("unimol_reaction_forward", "unimol_reaction_forward_base")
def unimol_reaction_forward_base_architecture(args):
    base_architecture(args)
