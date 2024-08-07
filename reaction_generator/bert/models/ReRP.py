import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from unicore import utils
from unicore.models import BaseUnicoreModel, register_model, register_model_architecture
from typing import Callable, Optional, Dict, Tuple, Any, NamedTuple, List
import math
from ..modules import (
    init_xavier_params,
    init_bert_params,
    TransformerEncoderQuery,
    TransformerDecoderQuery,
    TransformerDecoderQueryAug,
    TransformerEncoderMerge, 
    MaskLMHead,
    ClassificationHead,
)


logger = logging.getLogger(__name__)

torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_override_can_fuse_on_cpu(True)
torch._C._jit_override_can_fuse_on_gpu(True)


class ReRPBaseModel(BaseUnicoreModel):

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
            "--position-type",
            default="normal",
            choices=["sinusoidal", "relative", "normal"],
            help="noise type in coordinate noise",
        )
        parser.add_argument(
            "--transformer-type",
            default="normal",
            choices=["simple", "normal"],
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
            "--contrastive-global-negative",
            action="store_true",
            help="use contrastive learning or not",
        )
        parser.add_argument(
            "--auto-regressive",
            action="store_true",
            help="use auto regressive generative or not",
        )
        parser.add_argument(
            "--train-force-decoding",
            action="store_true",
            help="use train force decoding or not",
        )
        parser.add_argument(
            "--atten-score",
            action="store_true",
            help="get attention score or not",
        )
        parser.add_argument(
            "--class-embedding", action="store_true", help="use class embedding or not"
        )
        parser.add_argument(
            "--local-attn-size",
            type=int,
            help="attention window for decoder to reduce teacher-forcing denpendency",
        )
        parser.add_argument(
            "--use-decoder", action="store_true", help="use decoder or not"
        )
        parser.add_argument(
            "--concat-encoder-emb", action="store_true", help="concat encoder or not"
        )
        parser.add_argument(
            "--align-encoder-emb", action="store_true", help="alignment encoder or not"
        )
        parser.add_argument(
            "--talking-heads", action="store_true", help="talking heads or not"
        )
        parser.add_argument(
            "--train-force-rate",
            default = 0.9,
            type = float,
            help = "train forcing rate"
        )
        parser.add_argument(
            "--mix-force-step",
            default = 10000,
            type=int,
            help="mix teacher forcing step",
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
        parser.add_argument(
            "--attention-align-loss-weight",
            default = -1.0,
            type = float,
            help = "attention alignment loss weight"
        )
        
        parser.add_argument(
            "--smoothl1-beta",
            default=1.0,
            type=float,
            help="beta in pair distance smoothl1 loss",
        )

    def __init__(self, args, dictionary, **kwargs):
        super().__init__()
        base_architecture(args)
        self.args = args
        self.padding_idx = dictionary.pad()
        self.mask_idx = -1
        self._num_updates = None
        self.embed_tokens = nn.Embedding(
            len(dictionary), args.encoder_embed_dim, self.padding_idx
        )
        self.encoder = self.get_encoder(kwargs)

        self.lm_head = MaskLMHead(
            embed_dim=args.encoder_embed_dim,
            output_dim=len(dictionary),
            activation_fn=args.activation_fn,
            weight=None,
        )
        self.auto_regressive = args.auto_regressive
        self.train_force_decoding = args.train_force_decoding
        self.atten_score = args.atten_score
        self.train_force_rate = args.train_force_rate
        self.mix_force_step = args.mix_force_step
        self.talking_heads = args.talking_heads

        self.use_decoder = args.use_decoder
        # self.embed_positions = self.get_position_embedding('test', args.max_seq_len, args.encoder_embed_dim)
        
        self.classification_head = ClassificationHead(args.encoder_embed_dim, args.encoder_embed_dim, 10, "gelu", args.pooler_dropout)
        
        self.atom_identifier = nn.Sequential(nn.Linear(args.encoder_embed_dim, 1),
                                                nn.Sigmoid())

        self.embed_positions = self.get_position_embedding(
            args.position_type, args.max_seq_len, args.encoder_embed_dim
        )
        # self.embed_positions = nn.Embedding(args.max_seq_len, args.encoder_embed_dim)
        self.rc_pred_loss_weight = args.rc_pred_loss_weight
        self.use_class_embedding = args.class_embedding
        self.concat_encoder_emb = args.concat_encoder_emb
        self.align_encoder_emb = args.align_encoder_emb

        if self.use_class_embedding:
            self.class_embedding = nn.Embedding(100, args.encoder_embed_dim)    
        if self.concat_encoder_emb:
            self.concat_encoder_position_embedding = nn.Embedding(100, args.encoder_embed_dim)  

        if args.auto_regressive:
            self.use_decoder = True
        if self.use_decoder:
            # self.decoder_embed_positions = self.get_position_embedding('test', args.max_seq_len, args.decoder_embed_dim)
            # self.decoder_embed_positions = nn.Embedding(args.max_seq_len, args.decoder_embed_dim)
            self.decoder_embed_positions = self.get_position_embedding(
                args.position_type, args.max_seq_len, args.decoder_embed_dim
            )
            self.decoder_embed_tokens = self.embed_tokens  # FFFFFF
            # self.decoder_embed_tokens = nn.Embedding(len(dictionary), args.decoder_embed_dim, self.padding_idx)
            self.decoder = self.get_decoder()
            self.encoder_merge = self.get_encoder_merge()
            self.decoder_lm_head = MaskLMHead(
                embed_dim=args.decoder_embed_dim,
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
        if hasattr(task, "encoder_dictionary"):
            return cls(
                args, task.dictionary, encoder_dictionary=task.encoder_dictionary
            )
        return cls(args, task.dictionary)

    def get_default_encoder(self):
        encoder = TransformerEncoderQuery(
            encoder_layers=self.args.encoder_layers,
            embed_dim=self.args.encoder_embed_dim,
            ffn_embed_dim=self.args.encoder_ffn_embed_dim,
            attention_heads=self.args.encoder_attention_heads,
            emb_dropout=self.args.emb_dropout,
            dropout=self.args.dropout,
            attention_dropout=self.args.attention_dropout,
            activation_dropout=self.args.activation_dropout,
            max_seq_len=self.args.max_seq_len,
            activation_fn=self.args.activation_fn,
        )
        return encoder

    def get_encoder(self, kwargs):
        encoder = self.get_default_encoder()
        return encoder

    def get_encoder_merge(self):
        encoder_merge = TransformerEncoderMerge(
            decoder_layers=1,
            embed_dim=self.args.decoder_embed_dim,
            ffn_embed_dim=self.args.decoder_ffn_embed_dim,
            attention_heads=self.args.decoder_attention_heads,
            emb_dropout=self.args.emb_dropout,
            dropout=self.args.dropout,
            attention_dropout=self.args.attention_dropout,
            activation_dropout=self.args.activation_dropout,
            # emb_dropout=0,
            # dropout=0,
            # attention_dropout=0,
            # activation_dropout=0,
            max_seq_len=self.args.max_seq_len,
            activation_fn=self.args.activation_fn,
            # auto_regressive=self.args.auto_regressive,
            local_attn_size=self.args.local_attn_size,
        )
        return encoder_merge

    def get_decoder(self):
        if self.rc_pred_loss_weight > 0:
            decoder = TransformerDecoderQueryAug(
                decoder_layers=self.args.decoder_layers,
                embed_dim=self.args.decoder_embed_dim,
                ffn_embed_dim=self.args.decoder_ffn_embed_dim,
                attention_heads=self.args.decoder_attention_heads,
                emb_dropout=self.args.emb_dropout,
                dropout=self.args.dropout,
                attention_dropout=self.args.attention_dropout,
                activation_dropout=self.args.activation_dropout,
                # emb_dropout=0,
                # dropout=0,
                # attention_dropout=0,
                # activation_dropout=0,
                max_seq_len=self.args.max_seq_len,
                activation_fn=self.args.activation_fn,
                auto_regressive=self.args.auto_regressive,
                local_attn_size=self.args.local_attn_size,
            )
        else:
            decoder = TransformerDecoderQuery(
                decoder_layers=self.args.decoder_layers,
                embed_dim=self.args.decoder_embed_dim,
                ffn_embed_dim=self.args.decoder_ffn_embed_dim,
                attention_heads=self.args.decoder_attention_heads,
                emb_dropout=self.args.emb_dropout,
                dropout=self.args.dropout,
                attention_dropout=self.args.attention_dropout,
                activation_dropout=self.args.activation_dropout,
                # emb_dropout=0,
                # dropout=0,
                # attention_dropout=0,
                # activation_dropout=0,
                max_seq_len=self.args.max_seq_len,
                activation_fn=self.args.activation_fn,
                auto_regressive=self.args.auto_regressive,
                local_attn_size=self.args.local_attn_size,
                talking_heads =self.talking_heads, 
            )
        return decoder

    def get_position_embedding(self, position_type, max_seq_len, embed_dim):

        if position_type == "sinusoidal":
            pe = torch.zeros(max_seq_len, embed_dim)
            position = torch.arange(0, max_seq_len).unsqueeze(1)
            div_term = torch.exp(
                (
                    torch.arange(0, embed_dim, 2, dtype=torch.float)
                    * -(math.log(10000.0) / embed_dim)
                )
            )
            pe[:, 0::2] = torch.sin(position.float() * div_term)
            pe[:, 1::2] = torch.cos(position.float() * div_term)
            pe1 = nn.Embedding(max_seq_len, embed_dim)
            pe1.weight = nn.Parameter(pe, requires_grad=False)
            return pe1

        elif position_type == "relative":
            # relative_pe = nn.Embedding(max_seq_len * 2 + 2, embed_dim)
            pe = torch.zeros(max_seq_len, embed_dim // 2)
            position = torch.arange(0, max_seq_len).unsqueeze(1)
            div_term = torch.exp(
                (
                    torch.arange(0, (embed_dim // 2), 2, dtype=torch.float)
                    * -(math.log(10000.0) / (embed_dim // 2))
                )
            )
            pe[:, 0::2] = torch.sin(position.float() * div_term)
            pe[:, 1::2] = torch.cos(position.float() * div_term)
            pe1 = nn.Embedding(max_seq_len, embed_dim // 2)
            pe1.weight = nn.Parameter(pe, requires_grad=False)
            relative = nn.Embedding(max_seq_len, embed_dim // 2)
            relative_pe = torch.cat((relative, pe1), -1)
            return relative_pe
        else:
            return nn.Embedding(max_seq_len, embed_dim)

    def forward(
        self,
        src_tokens,
        decoder_src_tokens,
        # relation_type,
        # masked_tokens=None,
        features_only=False,
        classification_head_name=None,
        **kwargs
    ):

        if classification_head_name is not None:
            features_only = True
        encoder_rep, padding_mask, masked_tokens, unimol_classifier_logits, atom_score = self.forward_encoder(
            src_tokens=src_tokens, **kwargs
        )
        atom_score = atom_score.squeeze(-1)

        if self.rc_pred_loss_weight > 0:
            # rc_mask_padding = kwargs['smiles_src_tokens'].eq(self.padding_idx)
            mask_smiles_token_padding = torch.ones_like(kwargs['smiles_src_tokens']).bool()
            mask_atom_token_padding = ~kwargs['can_rc_pro_dataset']
            rc_mask_padding = torch.cat((mask_smiles_token_padding, mask_atom_token_padding),-1)

        # self.mix_force_step
        # if self._num_updates < self.mix_force_step: 
        #     mask_smiles_token_padding = torch.ones_like(kwargs['smiles_src_tokens']).bool()
        #     mask_atom_token_padding = ~kwargs['can_rc_pro_dataset']
        #     rc_mask_padding = torch.cat((mask_smiles_token_padding, mask_atom_token_padding),-1)
        # else:
        #     mask_smiles_token_padding = torch.ones_like(kwargs['smiles_src_tokens']).bool()
        #     atom_score2 = atom_score.detach()
        #     mask_atom_token_padding = atom_score2 > 0.9
        #     # print('test atom_score2: ', atom_score.shape, atom_score, mask_atom_token_padding)
        #     rc_mask_padding = torch.cat((mask_smiles_token_padding, mask_atom_token_padding),-1)  

        # print('test kwargs: ', mask_atom_token_padding.shape, atom_score.shape)
        if self.rc_pred_loss_weight > 0:
            decoder_outprob, vae_kl_loss = self.forward_decoder_aug(
                decoder_src_tokens=decoder_src_tokens,
                encoder_cls=encoder_rep,
                temperature=None,
                encoder_padding_mask=padding_mask,
                train_force_decoding = self.train_force_decoding,
                want_probs=False,
                rc_center_mask = rc_mask_padding,
            )
        else:
            decoder_outprob, vae_kl_loss, atten_score_list = self.forward_decoder(
                decoder_src_tokens=decoder_src_tokens,
                encoder_cls=encoder_rep,
                temperature=None,
                encoder_padding_mask=padding_mask,
                train_force_decoding = self.train_force_decoding,
                want_probs=False,
            )

        contrast_out = None
        if not features_only:
            logits = self.lm_head(encoder_rep, masked_tokens)
        else:
            logits = encoder_rep

        return logits, decoder_outprob, contrast_out, vae_kl_loss, unimol_classifier_logits, atom_score, atten_score_list

    def forward_default_encoder(self, encoder, src_tokens, **kwargs):
        padding_mask = src_tokens.eq(self.padding_idx)
        masked_tokens = ~padding_mask
        tmp_padding_mask = padding_mask
        if not padding_mask.any():
            padding_mask = None
        x = self.embed_tokens(src_tokens)
        seq_len = src_tokens.size(1)
        x = x * math.sqrt(x.shape[-1])  # FFFFFF
        x += self.embed_positions.weight[:seq_len, :]
        # if self.use_class_embedding:
        #     x[:,0,:] += self.class_embedding(reaction_class)
        if padding_mask is not None:
            x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))
        encoder_rep = encoder(x, padding_mask=padding_mask, attn_mask=None)
        return encoder_rep, padding_mask, tmp_padding_mask, masked_tokens

    def forward_encoder(
        self,
        src_tokens,
        # reaction_class,
        # masked_tokens=None,
        **kwargs
    ):
        encoder_rep, padding_mask, _, masked_tokens = self.forward_default_encoder(
            self.encoder, src_tokens
        )
        return encoder_rep, padding_mask, masked_tokens

    def forward_decoder_aug(
        self,
        decoder_src_tokens,
        encoder_cls,
        temperature,
        encoder_padding_mask,
        train_force_decoding = False,
        want_probs = True,
        rc_center_mask = None,
        **kwargs
    ):
        decoder_outprob = None
        vae_kl_loss = None


        if self.use_decoder:
            decoder_padding_mask = decoder_src_tokens.eq(self.padding_idx)
            if not decoder_padding_mask.any():
                decoder_padding_mask = None
            x_decoder = self.decoder_embed_tokens(decoder_src_tokens)
            x_decoder = x_decoder * math.sqrt(x_decoder.shape[-1])  
            seq_len = decoder_src_tokens.size(1)
            x_decoder += self.decoder_embed_positions.weight[:seq_len, :]
            if decoder_padding_mask is not None:
                x_decoder = x_decoder * (
                    1 - decoder_padding_mask.unsqueeze(-1).type_as(x_decoder)
                )
            # print('test4: ', x_decoder.shape, x_decoder.device)
#  and self._num_updates > self.mix_force_step
            if train_force_decoding and self._num_updates > self.mix_force_step:         
                with torch.no_grad():    
                    decoder_rep = self.decoder(
                        x_decoder,
                        padding_mask=decoder_padding_mask,
                        encoder_padding_mask=encoder_padding_mask,
                        encoder_out=encoder_cls,
                        attn_mask=None,
                        reaction_center_mask= rc_center_mask, 
                    )
                    mid_decoder_outprob = self.decoder_lm_head(decoder_rep)
                    # mid_probs = self.get_normalized_probs(
                    #     mid_decoder_outprob, temperature, log_probs=True, sample=None
                    # )
                    mid_probs = F.softmax(mid_decoder_outprob, dim=-1)
                    batch_size, seq_len, num_clas = mid_probs.shape
                    mid_probs2 = mid_probs.view(-1, num_clas).clone()
                    mid_probs2[:,0] = 0
                    x_decoder_class_ind = torch.multinomial(mid_probs2, 1)
                    x_decoder_class_ind = x_decoder_class_ind.view(batch_size, seq_len)
                    # x_decoder_pred = self.decoder_embed_tokens(x_decoder_class_ind) 

                mask_probs = torch.rand(decoder_rep.shape[:-1]).to(decoder_src_tokens)
                x_decoder_idx_mix = torch.where(mask_probs > self.train_force_rate, x_decoder_class_ind, decoder_src_tokens)
                x_decoder_mix = self.decoder_embed_tokens(x_decoder_idx_mix) 
                decoder_rep = self.decoder(
                        x_decoder_mix,
                        padding_mask=decoder_padding_mask,
                        encoder_padding_mask=encoder_padding_mask,
                        encoder_out=encoder_cls,
                        attn_mask=None,
                        reaction_center_mask= rc_center_mask, 
                    )

            else: 
                decoder_rep = self.decoder(
                    x_decoder,
                    padding_mask=decoder_padding_mask,
                    encoder_padding_mask=encoder_padding_mask,
                    encoder_out=encoder_cls,
                    attn_mask=None,
                    reaction_center_mask= rc_center_mask, 
                )
            decoder_outprob = self.decoder_lm_head(decoder_rep)
            if want_probs:
                probs = self.get_normalized_probs(
                    decoder_outprob, temperature, log_probs=True, sample=None
                )
                probs = probs[:, -1, :]
                return probs, None

        return decoder_outprob, vae_kl_loss

    def forward_decoder(
        self,
        decoder_src_tokens,
        encoder_cls,
        temperature,
        encoder_padding_mask,
        train_force_decoding = False,
        want_probs = True,
        **kwargs
    ):
        decoder_outprob = None
        vae_kl_loss = None


        if self.use_decoder:
            decoder_padding_mask = decoder_src_tokens.eq(self.padding_idx)
            if not decoder_padding_mask.any():
                decoder_padding_mask = None
            x_decoder = self.decoder_embed_tokens(decoder_src_tokens)
            x_decoder = x_decoder * math.sqrt(x_decoder.shape[-1])  
            seq_len = decoder_src_tokens.size(1)
            x_decoder += self.decoder_embed_positions.weight[:seq_len, :]
            if decoder_padding_mask is not None:
                x_decoder = x_decoder * (
                    1 - decoder_padding_mask.unsqueeze(-1).type_as(x_decoder)
                )

            if train_force_decoding and self._num_updates > self.mix_force_step:         
                with torch.no_grad():    
                    decoder_rep = self.decoder(
                        x_decoder,
                        padding_mask=decoder_padding_mask,
                        encoder_padding_mask=encoder_padding_mask,
                        encoder_out=encoder_cls,
                        attn_mask=None,
                    )
                    mid_decoder_outprob = self.decoder_lm_head(decoder_rep)
                    # mid_probs = self.get_normalized_probs(
                    #     mid_decoder_outprob, temperature, log_probs=True, sample=None
                    # )
                    mid_probs = F.softmax(mid_decoder_outprob, dim=-1)
                    batch_size, seq_len, num_clas = mid_probs.shape
                    mid_probs2 = mid_probs.view(-1, num_clas).clone()
                    mid_probs2[:,0] = 0
                    x_decoder_class_ind = torch.multinomial(mid_probs2, 1)
                    x_decoder_class_ind = x_decoder_class_ind.view(batch_size, seq_len)
                    # x_decoder_pred = self.decoder_embed_tokens(x_decoder_class_ind) 

                mask_probs = torch.rand(decoder_rep.shape[:-1]).to(decoder_src_tokens)
                x_decoder_idx_mix = torch.where(mask_probs > self.train_force_rate, x_decoder_class_ind, decoder_src_tokens)
                x_decoder_mix = self.decoder_embed_tokens(x_decoder_idx_mix) 
                decoder_rep = self.decoder(
                        x_decoder_mix,
                        padding_mask=decoder_padding_mask,
                        encoder_padding_mask=encoder_padding_mask,
                        encoder_out=encoder_cls,
                        attn_mask=None,
                    )

            else: 
                decoder_rep, atten_score_list = self.decoder(
                    x_decoder,
                    padding_mask=decoder_padding_mask,
                    encoder_padding_mask=encoder_padding_mask,
                    encoder_out=encoder_cls,
                    attn_mask=None,
                    atten_score_flag = self.atten_score, 
                )
            decoder_outprob = self.decoder_lm_head(decoder_rep)
            if want_probs:
                probs = self.get_normalized_probs(
                    decoder_outprob, temperature, log_probs=True, sample=None
                )
                probs = probs[:, -1, :]
                return probs, None

        return decoder_outprob, vae_kl_loss, atten_score_list

    def get_normalized_probs(self, net_output, temperature, log_probs, sample=None):
        """Get normalized probabilities (or log probs) from a net's output."""
        logits = net_output  # [0].float()
        if log_probs:
            return F.log_softmax(logits, dim=-1)
        else:
            return torch.log(F.softmax(logits / temperature, dim=-1))

    def set_num_updates(self, num_updates):
        """State from trainer to pass along to model at every update."""

        self._num_updates = num_updates

    def get_num_updates(self):

        return self._num_updates

    def concat_encoder_rep_attention(self, smile_encoder_rep, unimol_encoder_rep, smile_encoder_rep_padding, unimol_encoder_rep_padding, **kwargs):

        # per_index = kwargs['aug_index_dataset'] + 1
        # position_index_embedding = self.concat_encoder_position_embedding(per_index)
        if smile_encoder_rep_padding is not None:
            smile_encoder_rep = smile_encoder_rep * (
                1 - smile_encoder_rep_padding.unsqueeze(-1).type_as(smile_encoder_rep)
            )

        # unimol_encoder_rep = unimol_encoder_rep + position_index_embedding       
        if unimol_encoder_rep_padding is not None:
            unimol_encoder_rep = unimol_encoder_rep * (
                1 - unimol_encoder_rep_padding.unsqueeze(-1).type_as(unimol_encoder_rep)
            )

        concat_encoder_rep = self.encoder_merge(
                    emb = smile_encoder_rep, 
                    encoder_out = unimol_encoder_rep,
                    padding_mask = smile_encoder_rep_padding, 
                    encoder_padding_mask = unimol_encoder_rep_padding, 
                    attn_mask=None,)
        concat_encoder_rep = concat_encoder_rep + smile_encoder_rep
        return concat_encoder_rep, smile_encoder_rep_padding

    def align_reaction_position_smile(self, unimol_encoder_rep, reaction_padding, **kwargs):
        # 将反应位点对齐到 smile
        # align 3d atom rep
        per_index = kwargs['aug_index_dataset'] + 1
        per_unimol_encoder_reaction_padding = torch.zeros_like(kwargs['smiles_src_tokens']).bool()
        for i in range(unimol_encoder_rep.shape[0]):
            for j in range(reaction_padding.shape[1]):
                per_unimol_encoder_reaction_padding[i, per_index[i,j]] = reaction_padding[i,j]

        # align 3d atom rep padding
        smiles_padding_rep_padding = torch.zeros_like(kwargs['smiles_src_tokens']).bool()
        atom_index_dataset = kwargs['atom_index_dataset']
        atom_index_dataset_padding = kwargs['atom_index_dataset'].ne(self.mask_idx)
        for i in range(default_encoder_rep.shape[0]):
            for j in range(atom_index_dataset.shape[1]):
                atom_num = atom_index_dataset[i,j]
                if atom_num.ne(self.mask_idx):
                    atom_num = atom_num + 1
                    smiles_padding_rep_padding[i, atom_num] = per_unimol_encoder_reaction_padding[i, j]

        return smiles_padding_rep

    def concat_encoder_rep_index(self, unimol_encoder_rep, default_encoder_rep, **kwargs):
        # 根据 atom map num 手动 match 上去

        # align 3d atom rep
        per_index = kwargs['aug_index_dataset'] + 1
        per_unimol_encoder_rep = torch.zeros_like(unimol_encoder_rep)
        for i in range(unimol_encoder_rep.shape[0]):
            for j in range(unimol_encoder_rep.shape[1]):
                per_unimol_encoder_rep[i, per_index[i,j]] = unimol_encoder_rep[i,j]
        # expand as 1d atom rep
        smile_rep = torch.zeros_like(default_encoder_rep)
        atom_index_dataset = kwargs['atom_index_dataset']

        for i in range(default_encoder_rep.shape[0]):
            for j in range(atom_index_dataset.shape[1]):
                atom_num = atom_index_dataset[i,j]
                if atom_num.ne(self.mask_idx):
                    try:
                        atom_num = atom_num + 1
                        smile_rep[i, atom_num] = per_unimol_encoder_rep[i, j]
                    except:
                        smile_rep[i, atom_num] = per_unimol_encoder_rep[i, j]
                        # print('test dim: ', per_index[i], atom_index_dataset[i], atom_index_dataset[i].shape, smile_rep.shape, per_unimol_encoder_rep.shape)
        return smile_rep

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


@register_model("ReRP")
class ReRPModel(ReRPBaseModel):
    @staticmethod
    def add_args(parser):
        ReRPBaseModel.add_args(parser)
        parser.add_argument(
            "--encoder-type",
            default="default",
            choices=["default", "unimol", "default_and_unimol"],
            help="model chosen as encoder",
        )

    def get_unimol_encoder(self, kwargs):
        encoder_dictionary = (
            kwargs["encoder_dictionary"]
            if "encoder_dictionary" in kwargs.keys()
            else None
        )
        assert encoder_dictionary is not None
        from .unimol_encoder import CustomizedUniMolModel

        encoder = CustomizedUniMolModel(self.args, encoder_dictionary)
        return encoder

    def get_encoder(self, kwargs):
        if self.args.encoder_type == "default":
            encoder = self.get_default_encoder()
        elif self.args.encoder_type == "unimol":
            encoder = self.get_unimol_encoder(kwargs)
        elif self.args.encoder_type == "default_and_unimol":
            encoder = nn.ModuleList(
                [self.get_default_encoder(), self.get_unimol_encoder(kwargs)]
            )
        return encoder

    def forward_unimol_encoder(self, encoder, src_tokens, **kwargs):
        padding_mask = src_tokens.eq(self.padding_idx)
        masked_tokens = ~padding_mask
        if not padding_mask.any():
            padding_mask = None
        tmp_padding_mask = padding_mask
        encoder_input_dict = {
            "src_tokens": src_tokens,
            "encoder_masked_tokens": masked_tokens,
            "src_distance": kwargs["src_distance"],
            "src_coord": kwargs["src_coord"],
            "src_edge_type": kwargs["src_edge_type"],
            "features_only": True,
        }

        encoder_rep, _, _, _, _, = encoder(
            **encoder_input_dict,
        )
        return encoder_rep, padding_mask, tmp_padding_mask, masked_tokens

    def forward_encoder(
        self,
        src_tokens,
        # reaction_class,
        # masked_tokens=None,
        **kwargs
    ):
        if self.args.encoder_type == "default":
            encoder_rep, padding_mask, _, masked_tokens = self.forward_default_encoder(
                self.encoder, src_tokens
            )
            unimol_classifier_logits = self.classification_head(encoder_rep) 
            atom_score = self.atom_identifier(encoder_rep)

        elif self.args.encoder_type == "unimol":
            encoder_rep, padding_mask, _, masked_tokens = self.forward_unimol_encoder(
                self.encoder, src_tokens, **kwargs
            )

            unimol_classifier_logits = self.classification_head(encoder_rep) 
            atom_score = self.atom_identifier(unimol_encoder_rep)

        elif self.args.encoder_type == "default_and_unimol":

            (
                unimol_encoder_rep,
                unimol_padding_mask,
                unimol_padding_mask_tmp,
                unimol_masked_tokens,
            ) = self.forward_unimol_encoder(
                self.encoder[1], src_tokens, **kwargs
            )

            # aug_index_dataset = kwargs['aug_index_dataset']
            # aug_index_padding = aug_index_dataset.ne(self.mask_idx)
            # # aug_index_dataset[aug_index_padding]
            # print('test unimol_encoder_rep: ', aug_index_dataset[aug_index_padding].shape, aug_index_dataset[aug_index_padding])
            unimol_classifier_logits = self.classification_head(unimol_encoder_rep) 
            atom_score = self.atom_identifier(unimol_encoder_rep)
            # if kwargs["split"] in ['valid', 'test']:
            # reaction_class_t = torch.argmax(unimol_classifier_logits, dim=-1) 
            # else: 
            reaction_class_t = kwargs['reaction_type']
            # reaction_class = reaction_class_t

            (
                default_encoder_rep,
                default_padding_mask,
                default_padding_mask_tmp,
                default_masked_tokens,
            ) = self.forward_default_encoder( 
                self.encoder[0], src_tokens=kwargs["smiles_src_tokens"],  **kwargs
            )

            encoder_rep = torch.cat([default_encoder_rep, unimol_encoder_rep], 1)
            masked_tokens = torch.cat([default_masked_tokens, unimol_masked_tokens], 1)
            if default_padding_mask is None and unimol_padding_mask is None:
                padding_mask = None
            elif default_padding_mask is None:
                padding_mask = unimol_padding_mask
            elif unimol_padding_mask is None:
                padding_mask = default_padding_mask
            else:
                padding_mask = torch.cat(
                    [default_padding_mask_tmp, unimol_padding_mask_tmp], 1
                )

            if self.concat_encoder_emb:
                # print('test shape: ', default_encoder_rep.shape, unimol_encoder_rep.shape)
                encoder_rep, padding_mask = self.concat_encoder_rep_attention(default_encoder_rep, unimol_encoder_rep, default_padding_mask, unimol_padding_mask, **kwargs)
            elif self.align_encoder_emb:
                padding_mask = default_padding_mask_tmp
                encoder_rep = self.concat_encoder_rep_index(unimol_encoder_rep, default_encoder_rep, **kwargs)
        return encoder_rep, padding_mask, masked_tokens, unimol_classifier_logits, atom_score


@register_model_architecture("ReRP", "ReRP")
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
    args.contrastive_global_negative = getattr(
        args, "contrastive_global_negative", False
    )
    args.auto_regressive = getattr(args, "auto_regressive", False)
    args.train_force_decoding = getattr(args, "train_force_decoding", False)
    args.atten_score = getattr(args, "atten_score", False)    
    args.train_force_rate = getattr(args, "train_force_rate", 0.9)
    args.mix_force_step = getattr(args, "mix_force_step", 10000)
    args.use_decoder = getattr(args, "use_decoder", False)
    args.class_embedding = getattr(args, "class_embedding", False)
    args.ce_loss_weight = getattr(args, "ce_loss_weight", -1) 
    args.talking_heads = getattr(args, "talking_heads", False)
    

    args.decoder_layers = getattr(args, "decoder_layers", 15)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 512)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 2048)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 64)
    args.decoder_loss = getattr(args, "decoder_loss", 1)
    args.local_attn_size = getattr(args, "local_attn_size", -1)
    args.encoder_type = getattr(args, "encoder_type", "default")
    args.concat_encoder_emb = getattr(args, "concat_encoder_emb", False)
    args.align_encoder_emb = getattr(args, "align_encoder_emb", False)
    args.rc_pred_loss_weight = getattr(args, "rc_pred_loss_weight", -1)  


@register_model_architecture("ReRP", "ReRP_base")
def ReRP_base_architecture(args):
    base_architecture(args)


@register_model_architecture("ReRP", "ReRP_unimol")
def ReRP_unimol_architecture(args):
    args.encoder_type = "unimol"
    base_architecture(args)


@register_model_architecture("ReRP", "ReRP_DnU")
def ReRP_default_and_unimol_architecture(args):
    args.encoder_type = "default_and_unimol"
    base_architecture(args)
