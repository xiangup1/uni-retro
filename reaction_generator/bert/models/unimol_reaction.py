import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from unicore import utils
from unicore.models import BaseUnicoreModel, register_model, register_model_architecture
from unicore.data import Dictionary
from unicore.modules import LayerNorm, TransformerEncoder,TransformerDecoder, init_bert_params
from unicore.modules.softmax_dropout import softmax_dropout
from ..modules import TransformerEncoderQuery, TransformerDecoderQuery
from ..modules.classfier import ClassificationHead, MaskLMHead, FPClassfierHead
from typing import Callable, Optional, Dict, Tuple, Any, NamedTuple, List
import numpy as np
import math
from torch import Tensor

logger = logging.getLogger(__name__)

torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_override_can_fuse_on_cpu(True)
torch._C._jit_override_can_fuse_on_gpu(True)

@register_model("unimol_reaction")
class UniMolReactionModel(BaseUnicoreModel):
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
            "--class-embedding", action='store_true', help="use class embedding or not"
        )
        parser.add_argument(
            "--regdot-embedding", action='store_true', help="use regdot embedding or not"
        )
        parser.add_argument(
            "--special-token-embedding", action='store_true', help="use special token embedding or not"
        )
        parser.add_argument(
            "--digtal-token-embedding", action='store_true', help="use digtal token embedding or not"
        )
        parser.add_argument(
            "--local-attn-size", type=int, help="attention window for decoder to reduce teacher-forcing denpendency"
        )
        parser.add_argument(
            "--use-decoder", action='store_true', help="use decoder or not"
        )
        parser.add_argument(
            "--rel-pos", action='store_true', help="relative pos in transformer or not"
        )
        parser.add_argument(
            "--smoothl1-beta",
            default=1.0,
            type=float,
            help="beta in pair distance smoothl1 loss"
        )
        parser.add_argument(
            "--ce-loss-weight",
            default = -1.0,
            type = float,
            help = "cross entropy loss weight"
        )
        parser.add_argument(
            "--rc-pred-loss-weight",
            default = -1.0,
            type = float,
            help = "reaction center predict loss weight"
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
        self.class_encoder = TransformerEncoderQuery(
            encoder_layers=1,
            embed_dim=args.encoder_embed_dim,
            ffn_embed_dim=128,
            attention_heads=1,
            emb_dropout=args.emb_dropout,
            dropout=0.5,
            attention_dropout=args.attention_dropout,
            activation_dropout=args.activation_dropout,
            max_seq_len=args.max_seq_len,
            activation_fn=args.activation_fn,
            rel_pos = args.rel_pos,
        )
        self.classfier_macc_resnet = FPClassfierHead(
            input_dim = 167,
            inner_dim = args.encoder_embed_dim,
            num_layer = 3,
            num_classes = 10,
            activation_fn = "gelu",
            pooler_dropout = args.pooler_dropout,
        )
        self.classfier_ecfp_resnet = FPClassfierHead(
            input_dim = 256,
            inner_dim = args.encoder_embed_dim,
            num_layer = 3,
            num_classes = 10,
            activation_fn = "gelu",
            pooler_dropout = args.pooler_dropout,
        )
        self.lm_head = MaskLMHead(embed_dim=args.encoder_embed_dim,
            output_dim=len(dictionary),
            activation_fn=args.activation_fn,
            weight=None,
        )
        self.auto_regressive = args.auto_regressive
        self.use_class_embedding = args.class_embedding
        self.use_regdot_embedding = args.regdot_embedding
        self.use_special_token_embedding = args.special_token_embedding
        self.use_digtal_token_embedding = args.digtal_token_embedding
        self.use_decoder = args.use_decoder
        self.classification_head = ClassificationHead(args.encoder_embed_dim, args.encoder_embed_dim, 10, "gelu", args.pooler_dropout)
        self.embed_positions = self.get_position_embedding(args.position_type, args.max_seq_len, args.encoder_embed_dim)
        # self.embed_positions = nn.Embedding(args.max_seq_len, args.encoder_embed_dim)
        if self.use_class_embedding:    
            self.class_embedding = nn.Embedding(100, args.encoder_embed_dim) 

        self.dot_idx = 17

        self.atom_identifier = nn.Sequential(nn.Linear(args.encoder_embed_dim, 1),
                                                nn.Sigmoid())

        if self.use_regdot_embedding: 
        # {18: .}   
            self.regdot_embedding = nn.Embedding(4, args.encoder_embed_dim)  
        if self.use_special_token_embedding: 
        # {7: (, 8: ), 12: =, 19: -, 25: # ,32: / ,39: \, others}  
            self.special_embedding = nn.Embedding(40, args.encoder_embed_dim) 
        if self.use_digtal_token_embedding: 
        # {9:1, 10:0, 11:2, 15:3, 20:4, 31:5, 40:6, 49:7, 61:8, others}  
            self.digtal_embedding = nn.Embedding(40, args.encoder_embed_dim)       
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
                rel_pos = args.rel_pos,
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
        relation_type,
        masked_tokens,
        features_only=False,
        classification_head_name=None,
        pro_macc_fp = None,
        reg_label = None, 
        reg_spe_token = None, 
        reg_num_token = None,
        tar_spe_token = None, 
        tar_num_token = None,
        rc_product = None,
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

        if self.use_digtal_token_embedding:
            x += self.digtal_embedding(tar_num_token.long())
        if self.use_special_token_embedding:
            x += self.special_embedding(tar_spe_token.long())

        if padding_mask is not None:
            x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))
    
        # y = x.clone()
        # class_embedding_rep = self.class_encoder(y, padding_mask=padding_mask, attn_mask=None)
        # classifier_logits = self.classification_head(class_embedding_rep, padding_mask) 

        pro_macc_fp = pro_macc_fp.type_as(x)
        classifier_logits = self.classfier_macc_resnet(pro_macc_fp)

        if self.use_class_embedding:
            x[:,0,:] += self.class_embedding(relation_type)

        encoder_rep = self.encoder(x, padding_mask=padding_mask, attn_mask=None)
        atom_score = self.atom_identifier(encoder_rep)

        # if self.use_class_embedding:
        #     encoder_rep[:,0,:] += self.class_embedding(relation_type) 
        # 很神奇的是, 这是不行的, 很奇怪, 我不能解释
        # 与之前的比较可以得出的一个结论还真是encoder/alighment的行为决定了一个比较大的效果 gap，那就从头重新考虑下怎么加上这个效应；
        # 或者在 encoder 重新加一下 alignment guided infomation

       
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

            if self.use_regdot_embedding:
                x_decoder += self.regdot_embedding(reg_label.long())
            if self.use_digtal_token_embedding:
                x_decoder += self.digtal_embedding(reg_num_token.long())
            if self.use_special_token_embedding:
                x_decoder += self.special_embedding(reg_spe_token.long())

            if decoder_padding_mask is not None:
                x_decoder = x_decoder * (1 - decoder_padding_mask.unsqueeze(-1).type_as(x_decoder))           
            encoder_cls = encoder_rep
            rc_product = ~rc_product
            decoder_rep = self.decoder(x_decoder, padding_mask = decoder_padding_mask, encoder_padding_mask = padding_mask, encoder_out=encoder_cls, attn_mask=None, reaction_center_mask= rc_product, )
            decoder_outprob = self.decoder_lm_head(decoder_rep)
        contrast_out = None
        
        if not features_only:
            logits = self.lm_head(encoder_rep, masked_tokens)

        return atom_score, classifier_logits, logits, decoder_outprob, contrast_out, vae_kl_loss

    def forward_encoder(
        self,
        src_tokens,
        relation_type,
        features_only=False,
        classification_head_name=None,
        pro_macc_fp = None,
        tar_spe_token = None, 
        tar_num_token = None,
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
        # classifier_logits = self.classification_head(x, padding_mask, training_flag = True) 

        # y = x.clone()
        # class_embedding_rep = self.class_encoder(y, padding_mask=padding_mask, attn_mask=None)
        # classifier_logits = self.classification_head(class_embedding_rep, padding_mask) 

        pro_macc_fp = pro_macc_fp.type_as(x)
        classifier_logits = self.classfier_macc_resnet(pro_macc_fp)
        class_indices = torch.multinomial(classifier_logits, 1)
        # logits, class_indices = classifier_logits.topk(2, dim = -1, largest=True, sorted=True)
        # print('test logits: ', logits.shape, class_indices.shape, class_indices)
        # set_id = torch.randint(2, (1, 1))[0][0] 
        # print('test set_id: ', set_id)
        if self.use_class_embedding:
            x[:,0,:] += self.class_embedding(class_indices[:,0] + 1)

        if self.use_digtal_token_embedding:
            x += self.digtal_embedding(tar_num_token.long())
        if self.use_special_token_embedding:
            x += self.special_embedding(tar_spe_token.long())

        encoder_rep = self.encoder(x, padding_mask=padding_mask, attn_mask=None)
        atom_score = self.atom_identifier(encoder_rep)
        vae_kl_loss = None
        # if self.use_class_embedding:
        #     encoder_rep[:,0,:] += self.class_embedding(relation_type)
            # classifier_logits = self.classification_head(encoder_rep)   
            # classifier_logits_pred = torch.argmax(classifier_logits, dim=-1) 
            # encoder_rep[:,0,:] += self.class_embedding(classifier_logits_pred)
        if self.use_decoder:
            # encoder_cls = encoder_rep[:,0,:]
            encoder_cls = encoder_rep
        # encoder_cls = latent_z.transpose(0, 1).unsqueeze(1)
        return atom_score, encoder_cls, padding_mask

    
    def forward_decoder(
        self,
        decoder_src_tokens,
        encoder_cls,
        temperature,
        encoder_padding_mask,
        rc_product = None,
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

            if self.use_regdot_embedding:
                # split reg
                reg_dot_idx = self.get_special_split_mask(decoder_src_tokens).to(x_decoder.device)
                regdot_embedding = self.regdot_embedding(reg_dot_idx.long())
                x_decoder += regdot_embedding
                # x_decoder += self.regdot_embedding(reg_label.long())
            if self.use_digtal_token_embedding:
                reg_num_idx = self.get_number_mask(decoder_src_tokens).to(x_decoder.device)
                x_decoder += self.digtal_embedding(reg_num_idx.long())
            if self.use_special_token_embedding:
                reg_spe_idx = self.get_special_mask(decoder_src_tokens).to(x_decoder.device)
                x_decoder += self.special_embedding(reg_spe_idx.long())

            if decoder_padding_mask is not None:
                x_decoder = x_decoder * (1 - decoder_padding_mask.unsqueeze(-1).type_as(x_decoder))
            # x_decoder[:,-1,:].unsqueeze(1), 
            # decoder_rep = self.decoder(x_decoder, padding_mask=decoder_padding_mask, encoder_out=encoder_cls, attn_mask=None)
            rc_product = ~rc_product            
            decoder_rep = self.decoder(x_decoder, padding_mask=decoder_padding_mask, encoder_padding_mask=encoder_padding_mask, encoder_out=encoder_cls, attn_mask=None, reaction_center_mask = rc_product, )
            decoder_outprob = self.decoder_lm_head(decoder_rep)           
            probs = self.get_normalized_probs(
                decoder_outprob, temperature, log_probs=True, sample=None
            )          
            probs = probs[:, -1, :]
        return probs, None

    def get_number_mask(self, tokens):
        numtokens = torch.zeros_like(tokens)
        for idx, item in enumerate(tokens):
            numtokens[idx] = self.get_number_array(item)
        return numtokens

    def get_number_array(self, arraydata):
        token_dict = {"0":10, "1":1, "2":2, "3":3,"4":4,"5":5,"6":6,"7":7,"8":8}
        array_data_num_set = []
        second_pro = False
        for j in range(len(arraydata)):
            if arraydata[j] not in set(token_dict.keys()):
                array_data_num_set.append(20)
            else:
                array_data_num_set.append(token_dict[arraydata[j]])
        return torch.from_numpy(np.array(array_data_num_set))


    def get_special_mask(self, tokens):
        special_tokens = torch.zeros_like(tokens)
        for idx, item in enumerate(tokens):
            special_tokens[idx] = self.get_sepcial_array(item)
        return special_tokens

    def get_sepcial_array(self, arraydata):
        token_dict = {"(":1, ")":2, "=":3, ".":4,"-":5,"#":6,"/":7,"\\":8}
        array_data_spe_set = []
        for j in range(len(arraydata)):

            if arraydata[j] in set(token_dict.keys()):
                array_data_spe_set.append(token_dict[arraydata[j]])
            else:
                array_data_spe_set.append(20)

        return torch.from_numpy(np.array(array_data_spe_set))

    def get_special_split_mask(self, tokens):
        dottokens = torch.zeros_like(tokens)
        for idx, item in enumerate(tokens):
            dottokens[idx] = self.get_dot_array(item)
        return dottokens

    def get_dot_array(self, arraydata):
        array_data_set = torch.zeros_like(arraydata)
        second_pro = False
        for j in range(len(arraydata)):
            array_data_set[j] = 1
            if arraydata[j].eq(self.dot_idx):
                array_data_set[j] = 2
                second_pro = True
                continue
            if second_pro: 
                array_data_set[j] = 3  
            if arraydata[j].eq(self.padding_idx):
                array_data_set[j] = 0
        return array_data_set


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



@register_model_architecture("unimol_reaction", "unimol_reaction")
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
    args.rel_pos = getattr(args, "rel_pos", True)
    args.activation_fn = getattr(args, "activation_fn", "gelu")
    args.pooler_activation_fn = getattr(args, "pooler_activation_fn", "tanh")
    args.post_ln = getattr(args, "post_ln", False)
    args.contrastive_global_negative = getattr(args, "contrastive_global_negative", False)
    args.auto_regressive = getattr(args, "auto_regressive", False)
    args.use_decoder = getattr(args, "use_decoder", False)
    args.class_embedding = getattr(args, "class_embedding", False)
    args.regdot_embedding = getattr(args, "regdot_embedding", False)
    args.special_token_embedding = getattr(args, "special_token_embedding", False)
    args.digtal_token_embedding = getattr(args, "digtal_token_embedding", False)

    args.decoder_layers = getattr(args, "decoder_layers", 15)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 512)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 2048)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 64)
    args.decoder_loss = getattr(args, "decoder_loss", 1)
    args.local_attn_size = getattr(args, "local_attn_size", -1)
    args.ce_loss_weight = getattr(args, "ce_loss_weight", -1) 
    args.rc_pred_loss_weight = getattr(args, "rc_pred_loss_weight", -1)      

@register_model_architecture("unimol_reaction", "unimol_reaction_base")
def unimol_reaction_base_architecture(args):
    base_architecture(args)
