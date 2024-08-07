import logging
import os

import contextlib
from typing import Optional

import numpy as np
from unicore.data import (
    MaskTokensDataset,
    NestedDictionaryDataset,
    NumelDataset,
    NumSamplesDataset,
    LMDBDataset,
    AppendTokenDataset,
    PrependTokenDataset,
    RightPadDataset,
    EpochShuffleDataset,
    RawLabelDataset,
    TokenizeDataset,
    data_utils,
    SortDataset,
    FromNumpyDataset,
)
from ..data import Dictionary, KeyDataset, SplitDotDataset, TensorDataset

from .customized_unicore_task import CustomizedUnicoreTask as UnicoreTask
from unicore.tasks import register_task
# import selfies as sf
import rdkit.Chem as Chem

logger = logging.getLogger(__name__)


@register_task("unimol_reaction")
class UniMolReactionTask(UnicoreTask):
    """Task for training transformer auto-encoder models."""

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument(
            "data",
            help="colon separated path to data directories list, \
                            will be iterated upon during epochs in round-robin manner",
        )
        parser.add_argument(
            "--mask-prob",
            default=0.25,
            type=float,
            help="probability of replacing a token with mask",
        )
        parser.add_argument(
            "--leave-unmasked-prob",
            default=0.05,
            type=float,
            help="probability that a masked token is unmasked",
        )
        parser.add_argument(
            "--random-token-prob",
            default=0.05,
            type=float,
            help="probability of replacing a token with a random token",
        )
        parser.add_argument(
            "--noise-type",
            default='normal',
            choices=['trunc_normal', 'uniform', 'normal', 'none'],
            help="noise type in coordinate noise",
        )
        parser.add_argument(
            "--noise",
            default=1,
            type=float,
            help="coordinate noise for masked atoms",
        )
        parser.add_argument(
            "--use-selfies",
            default=0,
            type=int,
            help="coordinate noise for masked atoms",
        )
        parser.add_argument(
            "--use-class-embedding",
            default=0,
            type=int,
            help="class embedding or not",
        )
        parser.add_argument(
            '--results-smi-path', 
            metavar='RESDIR', 
            type=str, 
            default=None,
            help='path to save eval smile results (optional)"'
        )
        parser.add_argument(
            '--results-smi-file', 
            metavar='RESDIR', 
            type=str, 
            default=None,
            help='file to save eval smile results (optional)"'
        )
        parser.add_argument(
            '--training-shuffle',
            action='store_true',
            help='disable progress bar'
        )


    def __init__(self, args, dictionary):
        super().__init__(args)
        self.dictionary = dictionary
        self.seed = args.seed
        # add mask token
        self.use_selfies = args.use_selfies
        self.use_class_embedding = args.use_class_embedding
        self.mask_idx = dictionary.add_symbol("[MASK]", is_special=True)

        self.src_sizes = np.array([])
        self.tgt_sizes = np.array([])
        self.indices = np.array([])
        self.dot_idx = 17

    @classmethod
    def setup_task(cls, args, **kwargs):
        dictionary = Dictionary.load(os.path.join(args.data, "dict.txt"))
        logger.info("dictionary: {} types".format(len(dictionary)))
        return cls(args, dictionary)

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        return max(
            self.src_sizes[index],
            self.tgt_sizes[index] if self.tgt_sizes is not None else 0,
        )

    def num_tokens_vec(self, indices):
        """Return the number of tokens for a set of positions defined by indices.
        This value is used to enforce ``--max-tokens`` during batching."""
        sizes = self.src_sizes[indices]
        if self.tgt_sizes is not None:
            sizes = np.maximum(sizes, self.tgt_sizes[indices])
        return sizes

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return (
            self.src_sizes[index],
            self.tgt_sizes[index] if self.tgt_sizes is not None else 0,
        )
    def get_size(self, sizes):
        tmp_size_data = np.array([0] * len(sizes))
        for i in range(len(sizes)):
            tmp_size_data[i] = sizes[i]
        return tmp_size_data

    def ordered_indices(self, dataset_length):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        indices = np.arange(dataset_length, dtype=np.int64)
        if self.tgt_sizes is not None: 
            indices = indices[np.argsort(self.src_sizes[indices], kind="mergesort")]                       
            indices[np.argsort(self.tgt_sizes[indices], kind="mergesort")]
        else:
            indices = np.arange(dataset_length, dtype=np.int64)
        return indices


    def load_dataset(self, split, combine=False, **kwargs):
        """Load a given dataset split.
        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        split_path = os.path.join(self.args.data, split + '.lmdb')
        # todo: load data from example data
        raw_dataset = LMDBDataset(split_path)
        def one_dataset(raw_dataset, coord_seed, mask_seed):
            src_sizes = KeyDataset(raw_dataset, 'tar_size') 
            tgt_sizes = KeyDataset(raw_dataset, 'rec_size')
            
            self.src_sizes  = self.get_size(src_sizes)
            self.tgt_sizes  = self.get_size(tgt_sizes)
            self.indices = self.ordered_indices(len(raw_dataset))
            if self.use_selfies == 0:
                # dataset = ReorderDataset(raw_dataset, coord_seed, 'smi', 'atoms','coordinates')
                token_dataset = KeyDataset(raw_dataset, 'smiles_target')
                token_target_dataset = KeyDataset(raw_dataset, 'smiles_train')
            else:
                # dataset = ReorderDataset(raw_dataset, coord_seed, 'selfies', 'atoms','coordinates')
                token_dataset = KeyDataset(raw_dataset, 'selfies_target')
                token_target_dataset = KeyDataset(raw_dataset, 'selfies_train')

            pro_macc_fp = KeyDataset(raw_dataset, 'MACC_fp_pro')
            pro_ecfp_fp = KeyDataset(raw_dataset, 'ECFP_fp_pro')

            if self.use_class_embedding:
                class_dataset = KeyDataset(raw_dataset, 'class')
            else:
                class_dataset = KeyDataset(raw_dataset, 'class')

            token_dataset = TokenizeDataset(token_dataset, self.dictionary, max_seq_len=self.args.max_seq_len)
            token_target_dataset = TokenizeDataset(token_target_dataset, self.dictionary, max_seq_len=self.args.max_seq_len)

            reaction_point = KeyDataset(raw_dataset, 'smiles_target_label')  
            rc_product = KeyDataset(raw_dataset, 'rc_product')  
            reg_label = KeyDataset(raw_dataset, 'reg_label')            
            reg_spe_token = KeyDataset(raw_dataset, 'reg_spe_token')  
            reg_num_token = KeyDataset(raw_dataset, 'reg_num_token')  
            tar_spe_token = KeyDataset(raw_dataset, 'tar_spe_token')  
            tar_num_token = KeyDataset(raw_dataset, 'tar_num_token')  
            # split_dot_data = SplitDotDataset(token_target_dataset, self.dictionary.pad(), self.dot_idx)
            # dotdataset = split_dot_data.get_dot_dataset()
            # src_dataset, tgt_dataset = MaskTokensDataset.apply_mask(
            #     token_dataset,
            #     self.dictionary,
            #     pad_idx=self.dictionary.pad(),
            #     mask_idx=self.mask_idx,
            #     seed=mask_seed,
            #     mask_prob=self.args.mask_prob,
            #     leave_unmasked_prob=self.args.leave_unmasked_prob,
            #     random_token_prob=self.args.random_token_prob,
            # )
            
            def PrependAndAppend(dataset, pre_token, app_token):
                dataset = PrependTokenDataset(dataset, pre_token)
                return AppendTokenDataset(dataset, app_token)

            src_dataset = token_dataset
            tgt_dataset = token_dataset

            src_dataset = PrependAndAppend(src_dataset, self.dictionary.bos(), self.dictionary.eos())
            tgt_dataset = PrependAndAppend(tgt_dataset, self.dictionary.pad(), self.dictionary.pad())

            decoder_src_dataset = PrependAndAppend(token_target_dataset, self.dictionary.bos(), self.dictionary.eos())
            decoder_target = PrependAndAppend(token_target_dataset, self.dictionary.bos(), self.dictionary.pad())       

            reaction_point = TensorDataset(reaction_point) 
            rc_product = TensorDataset(rc_product)       
            reg_label = TensorDataset(reg_label)
            reg_spe_token = TensorDataset(reg_spe_token)
            reg_num_token = TensorDataset(reg_num_token)
            tar_spe_token = TensorDataset(tar_spe_token)
            tar_num_token = TensorDataset(tar_num_token)

            rc_product = PrependAndAppend(rc_product, self.dictionary.pad(), self.dictionary.pad()) 

            reaction_point = PrependAndAppend(reaction_point, self.dictionary.pad(), self.dictionary.pad()) 
            reg_label = PrependAndAppend(reg_label, self.dictionary.bos(), self.dictionary.eos())            
            reg_spe_token = PrependAndAppend(reg_spe_token, self.dictionary.bos(), self.dictionary.eos())
            reg_num_token = PrependAndAppend(reg_num_token, self.dictionary.bos(), self.dictionary.eos()) 
            tar_spe_token = PrependAndAppend(tar_spe_token, self.dictionary.bos(), self.dictionary.eos())
            tar_num_token = PrependAndAppend(tar_num_token, self.dictionary.bos(), self.dictionary.eos())

            return {
                "src_tokens": RightPadDataset(
                        src_dataset,
                        pad_idx=self.dictionary.pad(),
                    ),
                "decoder_src_tokens":RightPadDataset(
                        decoder_src_dataset,
                        pad_idx=self.dictionary.pad(),
                    ),
                "reaction_type": RawLabelDataset(class_dataset),

                "rc_product": RightPadDataset(rc_product, pad_idx=self.dictionary.pad(),),
                "reaction_point": RightPadDataset(reaction_point, pad_idx=self.dictionary.pad(),),
                "reg_label": RightPadDataset(reg_label, pad_idx=self.dictionary.pad(),),
                "reg_spe_token": RightPadDataset(reg_spe_token, pad_idx=self.dictionary.pad(),),
                "reg_num_token": RightPadDataset(reg_num_token, pad_idx=self.dictionary.pad(),),
                "tar_spe_token": RightPadDataset(tar_spe_token, pad_idx=self.dictionary.pad(),),
                "tar_num_token": RightPadDataset(tar_num_token, pad_idx=self.dictionary.pad(),),
                "pro_macc_fp": RawLabelDataset(pro_macc_fp),
                "pro_ecfp_fp": RawLabelDataset(pro_ecfp_fp),
            }, {
                # 'encoder': RightPadDataset(
                #     tgt_dataset, 
                #     pad_idx=self.dictionary.pad()
                #     ),
                # 'decoder':RightPadDataset(
                #     decoder_target, 
                #     pad_idx=self.dictionary.pad()
                #     )
                }
        net_input, target = one_dataset(raw_dataset, self.args.seed, self.args.seed)
        dataset = {'net_input': net_input, 'target': target}

        dataset =  NestedDictionaryDataset(
            dataset
        )
        if split in ['train', 'train.small']:
            # self.args.training_shuffle and 
            dataset = EpochShuffleDataset(dataset, len(dataset), self.args.seed)
        
        self.datasets[split] = dataset

    def build_model(self, args):
        from unicore import models
        model = models.build_model(args, self)
        return model

    def test_step(self, args, sample, model, loss, step, seed):
        
        # tgt_tokens = sample['target']['decoder']
        src_tokens = sample['net_input']['src_tokens']
        tgt_tokens = sample['net_input']['decoder_src_tokens']
        pred, vae_kl_loss, likelyhood = model._generate(sample)
        tgt_tokens = tgt_tokens.cpu().numpy()
        beam_size = model.beam_size
        # vae_kl_loss = vae_kl_loss.mean()
        for i in range(len(tgt_tokens)):  
            a = tgt_tokens[i]
            # print('test tgt_tokens a: ', a)
            mol_str = []
            for t in a[1:]:
                tt =  self.dictionary.decode(t)
                if tt == '[SEP]':
                    break
                mol_str.append(tt)
            mol_str = ''.join(mol_str)

            # b = src_tokens[i]
            # mol_str2 = []
            # for t in b[1:]:
            #     tt =  self.dictionary.decode(t.item())
            #     if tt == '[SEP]':
            #         break
            #     mol_str2.append(tt)
            # mol_str2 = ''.join(mol_str2)

            # if self.use_selfies ==1:
            #     mol_str = sf.decoder(mol_str)

            file_path = args.results_smi_path
            if not os.path.exists(file_path):
                os.makedirs(file_path)
            file_check = os.path.join(file_path, args.results_smi_file)
            # file_check2 = os.path.join(file_path, 'test_sample_total.smi')

            # with open(file_check2, 'a') as w1: 
            #     w1.write(mol_str2+'\t'+mol_str+'\n')
                
            with open(file_check, 'a') as w: 
                w.write(mol_str+'\t'+'target'+'\n')
                for j in range(beam_size):
                    # print('test tgt_tokens beam_size: ', pred[j + i * 10])
                    mol_smi = []
                    mol_str2 = []
                    # a = pred[i][j]["tokens"].cpu().numpy()
                    # for t in a:
                    #     tt =  self.dictionary.decode(t)
                    #     if tt == '[SEP]':
                    #         break
                    #     mol_str2.append(tt)
                    # mol_str2 = ''.join(mol_str2)
                    a = pred[j + i * beam_size].cpu().numpy()
                    score = likelyhood[j + i * beam_size]
                    for t in a[1:]:
                        tt =  self.dictionary.decode(t)
                        if tt == '[PAD]':
                            break
                        mol_str2.append(tt)
                    mol_str2 = ''.join(mol_str2)

                    # if self.use_selfies ==1:
                    #     smiles = sf.decoder(mol_str2)
                    #     #加一个rdkit转分子对象能否成功的判断
                    #     # if Chem.MolFromSmiles(smiles) is not None:
                    #     w.write(smiles+'\t'+'predicted'+str(j)+'\n')
                    #     mol_smi.append(smiles)
                    # else:

                    #加一个rdkit转分子对象能否成功的判断
                    smiles = mol_str2
                    # if Chem.MolFromSmiles(smiles) is not None:
                    w.write(smiles+'\t'+ str(score) +'\t'+'predicted'+'\t'+str(j)+'\n')
                    mol_smi.append(smiles)                                      
                        
        return pred, {"vae_kl_loss": vae_kl_loss, "sample_size":1, "bsz": sample['net_input']['decoder_src_tokens'].size(0),
                "seq_len": sample['net_input']['decoder_src_tokens'].size(1) * sample['net_input']['decoder_src_tokens'].size(0),}
