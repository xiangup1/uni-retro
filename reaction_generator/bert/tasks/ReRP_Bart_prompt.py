import logging
import os
import time

import contextlib
from typing import Optional
import sentencepiece as spm
import torch

import numpy as np
import pandas as pd
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
    # RightPadDatasetCross2D,
    RightPadDataset2D,
    SortDataset,
    FromNumpyDataset,
)
from ..data import (
    CustomizedUnicoreDataset,
    KeyDataset,
    SizeDataset,
    ReorderDataset,
    ListShuffleDataset,
    RandomSmilesDataset,
    BartTokenDataset,
    BpeTokenDataset,
    CatDataset,
    CleanMapNumberDataset,
    AugPerMapDataset, 
    SmilesTokenizerDataset,
    AtomIdxSmilePosDataset,
    AtomReactionMaskDataset,
    TensorDataset,
    CannoicalSmilerDataset, 
    ConcatDataset,
    CutDataset,
    RandomReactionSmilesDataset,
    RandomReactionSmilesAugDataset,
    ReRandomReactionSmilesAugDataset,
    RRandomReactionSmilesAugDataset,
    ReactionSmilesNumberAugDataset,
    TensorDimDataset,
    AlignmentMatrixDataset,
    PrependAndAppend2DDataset,
    FromListDataset,
    RollDataset,
    InferSmilesDataset,
    InferMultiReactionSmilesDataset,
    InferMultiReactionSmilesProProcessDataset,
    PadListDataset,
    ReverseAugDataset,
)

from unicore.tasks import register_task
from unicore import checkpoint_utils
from .customized_unicore_task import CustomizedUnicoreTask as UnicoreTask
from .ReRP import ReRPTask
from ..data import Dictionary
# import selfies as sf
import rdkit.Chem as Chem

logger = logging.getLogger(__name__)


@register_task("ReRP_Bart_P")
class ReRPPrBartTask(ReRPTask):
    """Task for training transformer auto-encoder models."""

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        ReRPPrBartTask.add_other_args(parser)
        ReRPPrBartTask.add_default_encoder_args(parser)

    @staticmethod
    def add_other_args(parser):
        parser.add_argument(
            "data",
            help="colon separated path to data directories list, \
                            will be iterated upon during epochs in round-robin manner",
        )

        parser.add_argument(
            "--dict-name",
            default='tokenizer-smiles-bart/dict.txt',
            help="dict name",
        )

        parser.add_argument(
            "--bart-dict-path",
            default='tokenizer-smiles-bart',
            help="bart dict path",
        )

        parser.add_argument(
            '--use-old-dataset',
            action='store_true',
            help='use-old-dataset'
        )

        parser.add_argument(
            "--task-type",
            default='retrosynthetic',
            choices=['retrosynthetic', 'synthetic'],
            help="task type for training",
        )

        parser.add_argument(
            "--infer-task-type",
            default='retrosynthetic',
            choices=['retrosynthetic', 'synthetic'],
            help="task type for inference",
        )

        parser.add_argument(
            "--random-aug-strategy-type",
            default='number',
            choices=['number', 'root_aug', 'fragment_index_aug'],
            help="task type for num augment of smile",
        )

        parser.add_argument(
            "--use-selfies",
            default=0,
            type=int,
            help="coordinate noise for masked atoms",
        )
        parser.add_argument(
            "--aug-infer-smile",
            default=0,
            type=int,
            help="augment augument smiles",
        )
        parser.add_argument(
            "--use-map-data",
            default=0,
            type=int,
            help="use map dataset for reaction",
        )

        parser.add_argument(
            "--aug-strategy-epoch",
            default=0,
            type=int,
            help="augument strategy epoch number for reaction",
        )

        parser.add_argument(
            "--use-class-embedding",
            default=0,
            type=int,
            help="class embedding or not",
        )
        parser.add_argument(
            "--use-align-augment",
            default=0,
            type=int,
            help="reaction alignment augment or not",
        )

        parser.add_argument(
            "--use-num-align-augment",
            default=0,
            type=int,
            help="number reaction alignment augment or not",
        )

        parser.add_argument(
            "--use-multi-smile-infer-augment",
            default=0,
            type=int,
            help="multi smile augment or not",
        )
        parser.add_argument(
            "--use_smile_times",
            default=0,
            type=int,
            help="smile times",
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
            '--bpe-token-file',
            metavar='RESDIR',
            type=str,
            default=None,
            help='file to save bpe token model(optional)"'
        )
        parser.add_argument(
            '--training-shuffle',
            action='store_true',
            help='disable progress bar'
        )
        parser.add_argument(
            "--pro-aug-prob",
            default=0.90,
            type=float,
            help="probability of product augument probability",
        )
        parser.add_argument(
            "--rea-aug-prob",
            default=0.90,
            type=float,
            help="probability of reactant augument probability",
        )
        parser.add_argument(
            "--use-inchi-data",
            default=0,
            type=int,
            help="use inchi dataset for reaction",
        )
        parser.add_argument(
            "--use-precursor-data",
            default=0,
            type=int,
            help="use precursor dataset for reaction",
        )
        parser.add_argument(
            "--syn_aug_prob",
            default=0.50,
            type=float,
            help="probability of reactant and product augument probability",
        )
        parser.add_argument(
            "--use_syn_aug",
            default=0,
            type=int,
            help="reactant and product augument",
        )

    @staticmethod
    def add_default_encoder_args(parser):
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
            "--bart-path",
            default="None",
            help="bart-path",
        )

        parser.add_argument(
            "--freeze-bart",
            action="store_true",
            help="freeze-bart",
        )


    def __init__(self, args, dictionary):
        super().__init__(args, dictionary)
        self.bpe_token_file = args.bpe_token_file
        self.bpe = spm.SentencePieceProcessor()
        if self.bpe_token_file:
            self.bpe.Load(self.bpe_token_file)
        self.source_dictionary = self.dictionary
        self.target_dictionary = self.dictionary
        self.pro_aug_prob = args.pro_aug_prob
        self.rea_aug_prob = args.rea_aug_prob
        self.use_num_align_augment = args.use_num_align_augment
        self.use_multi_smile_infer_augment = args.use_multi_smile_infer_augment
        self.use_smile_times = args.use_smile_times
        self.infer_task_type = args.infer_task_type
        self.random_aug_strategy_type = args.random_aug_strategy_type
        self.syn_aug_prob = args.syn_aug_prob

    def one_dataset(self, raw_dataset, coord_seed, mask_seed, **kwargs):
        
        if self.use_map_data > 0 or self.use_align_augment > 0 or self.use_num_align_augment > 0: 
            if self.task_type == 'retrosynthetic':
                input_name, output_name = 'smiles_mapnumber_target_list', 'smiles_mapnumber_reactant_list'
            elif self.task_type == 'synthetic':
                input_name, output_name = 'smiles_mapnumber_reactant_list', 'smiles_mapnumber_target_list' 
        else:
            if self.task_type == 'retrosynthetic':
                input_name, output_name = 'selfies_target_list', 'selfies_reactant_list'
            elif self.task_type == 'synthetic':
                input_name, output_name = 'selfies_reactant_list', 'selfies_target_list'

        token_dataset = KeyDataset(raw_dataset, input_name)
        token_target_dataset = KeyDataset(raw_dataset, output_name)

        class_dataset = KeyDataset(raw_dataset, 'class')
        
        def PrependAndAppend(dataset, pre_token, app_token):
            dataset = PrependTokenDataset(dataset, pre_token)
            return AppendTokenDataset(dataset, app_token)

        def data_aug_dataset(dataset):
            bart_dict_path = os.path.join(
                self.args.data, self.args.bart_dict_path)
            if kwargs["split"] in ['valid', 'test']:
                prob = 0
            else:
                prob = 0.9
            dataset = ListShuffleDataset(dataset, self.args.seed, prob=prob, epoch_t = self.aug_strategy_epoch)
            dataset = RandomSmilesDataset(dataset, self.args.seed, prob=prob, epoch_t = self.aug_strategy_epoch)

            dataset = SmilesTokenizerDataset(dataset)
            dataset = TokenizeDataset(
                dataset, self.dictionary, max_seq_len=self.args.max_seq_len)
            # dataset = BartTokenDataset(
            #     dataset, bart_dict_path, max_seq_len=self.args.max_seq_len
            # )
            return dataset

        def map_data_aug_dataset(dataset):
            bart_dict_path = os.path.join(
                self.args.data, self.args.bart_dict_path)
            if kwargs["split"] in ['valid', 'test']:
                prob = 0
            else:
                prob = self.rea_aug_prob
            dataset = ListShuffleDataset(dataset, self.args.seed, prob=prob)
            dataset = RandomSmilesDataset(dataset, self.args.seed, prob=prob, epoch_t = self.aug_strategy_epoch)
            dataset = CleanMapNumberDataset(dataset)

            dataset = BpeTokenDataset(dataset, self.bpe, self.args.max_seq_len)
            dataset = CutDataset(dataset, self.args.max_seq_len)
            dataset = TokenizeDataset(
                dataset, self.dictionary, max_seq_len=self.args.max_seq_len)

            return dataset

        def reverse_data_aug_dataset(token_dataset, token_target_dataset):

            if kwargs["split"] in ['valid', 'test']:
                prob = self.syn_aug_prob
            else:
                prob = self.syn_aug_prob
            # print('wz test syn_prob: ', prob)
            process_dataset = ReverseAugDataset(token_dataset, token_target_dataset, self.args.seed, prob)
            flag_dataset = TensorDimDataset(process_dataset, 0)
            token_dataset = TensorDimDataset(process_dataset, 1)
            token_target_dataset = TensorDimDataset(process_dataset, 2)
            return flag_dataset, token_dataset, token_target_dataset

        def map_data_aug_product_dataset(dataset):
            bart_dict_path = os.path.join(
                self.args.data, self.args.bart_dict_path)
            if kwargs["split"] in ['valid', 'test']:
                prob = 0.0
            else:
                prob = self.pro_aug_prob
            dataset = ListShuffleDataset(dataset, self.args.seed, prob=prob, epoch_t = self.aug_strategy_epoch)
            dataset = RandomSmilesDataset(dataset, self.args.seed, prob=prob, epoch_t = self.aug_strategy_epoch)

            # aug_index_dataset = AugPerMapDataset(dataset)
            dataset = CleanMapNumberDataset(dataset)

            dataset = BpeTokenDataset(dataset, self.bpe, self.args.max_seq_len)
            dataset = CutDataset(dataset, self.args.max_seq_len)
            
            dataset = TokenizeDataset(
                dataset, self.dictionary, max_seq_len=self.args.max_seq_len)

            return dataset

        def map_data_align_check_aug_dataset(pro_dataset, rea_dataset):
            if kwargs["split"] in ['valid', 'test']: 
                prob = 0.0
                process_dataset = RRandomReactionSmilesAugDataset(pro_dataset, rea_dataset, self.args.seed, prob=prob, epoch_t = self.aug_strategy_epoch)
                pro_dataset = TensorDimDataset(process_dataset, 0)
                rea_dataset = TensorDimDataset(process_dataset, 1)
                pro_dataset1 = TensorDimDataset(process_dataset, 2)
                rea_dataset1 = TensorDimDataset(process_dataset, 3)
            else:         
                prob = self.pro_aug_prob
                process_dataset = RRandomReactionSmilesAugDataset(pro_dataset, rea_dataset, self.args.seed, prob=prob, epoch_t = self.aug_strategy_epoch)
                pro_dataset = TensorDimDataset(process_dataset, 0)
                rea_dataset = TensorDimDataset(process_dataset, 1)
                pro_dataset1 = TensorDimDataset(process_dataset, 2)
                rea_dataset1 = TensorDimDataset(process_dataset, 3)
           
            pro_dataset = BpeTokenDataset(pro_dataset, self.bpe, self.args.max_seq_len)
            pro_dataset = CutDataset(pro_dataset, self.args.max_seq_len)
            pro_dataset = TokenizeDataset(
                pro_dataset, self.dictionary, max_seq_len=self.args.max_seq_len)

            rea_dataset = BpeTokenDataset(rea_dataset, self.bpe, self.args.max_seq_len)
            rea_dataset = CutDataset(rea_dataset, self.args.max_seq_len)    
            rea_dataset = TokenizeDataset(
                rea_dataset, self.dictionary, max_seq_len=self.args.max_seq_len)
     
            return pro_dataset, rea_dataset


        def map_data_multi_infer_dataset(pro_dataset, rea_dataset):
            process_dataset = InferMultiReactionSmilesDataset(pro_dataset, rea_dataset, self.args.seed, times = self.use_smile_times)
            pros_dataset = TensorDimDataset(process_dataset, 0)
            rea_dataset = TensorDimDataset(process_dataset, 1)
            pros_dataset = InferMultiReactionSmilesProProcessDataset(pros_dataset, self.bpe, self.dictionary, self.args.max_seq_len, times = self.use_smile_times)           
          
            rea_dataset = BpeTokenDataset(rea_dataset, self.bpe, self.args.max_seq_len)
            rea_dataset = CutDataset(rea_dataset, self.args.max_seq_len)    
            rea_dataset = TokenizeDataset(
                rea_dataset, self.dictionary, max_seq_len=self.args.max_seq_len)
     
            return pros_dataset, rea_dataset

        def map_data_align_number_aug_dataset(pro_dataset, rea_dataset):
            if kwargs["split"] in ['valid', 'test']: 
                prob = 0.0
                process_dataset = ReactionSmilesNumberAugDataset(pro_dataset, rea_dataset, self.args.seed, prob=prob, epoch_t = self.aug_strategy_epoch, aug_strategy = self.random_aug_strategy_type)
                pro_dataset = TensorDimDataset(process_dataset, 0)
                rea_dataset = TensorDimDataset(process_dataset, 1)
                pro_dataset1 = TensorDimDataset(process_dataset, 2)
                rea_dataset1 = TensorDimDataset(process_dataset, 3)
            else:         
                prob = self.pro_aug_prob
                process_dataset = ReactionSmilesNumberAugDataset(pro_dataset, rea_dataset, self.args.seed, prob=prob, epoch_t = self.aug_strategy_epoch, aug_strategy = self.random_aug_strategy_type)
                pro_dataset = TensorDimDataset(process_dataset, 0)
                rea_dataset = TensorDimDataset(process_dataset, 1)
                pro_dataset1 = TensorDimDataset(process_dataset, 2)
                rea_dataset1 = TensorDimDataset(process_dataset, 3)
           
            pro_dataset = BpeTokenDataset(pro_dataset, self.bpe, self.args.max_seq_len)
            pro_dataset = CutDataset(pro_dataset, self.args.max_seq_len)
            pro_dataset = TokenizeDataset(
                pro_dataset, self.dictionary, max_seq_len=self.args.max_seq_len)


            rea_dataset = BpeTokenDataset(rea_dataset, self.bpe, self.args.max_seq_len)  
            rea_dataset = CutDataset(rea_dataset, self.args.max_seq_len)  
            rea_dataset = TokenizeDataset(
                rea_dataset, self.dictionary, max_seq_len=self.args.max_seq_len)

            return pro_dataset, rea_dataset

        def map_data_align_aug_dataset(pro_dataset, rea_dataset):

            if kwargs["split"] in ['valid', 'test']:
                prob = 0.5
                pro_dataset = RandomSmilesDataset(pro_dataset, self.args.seed, prob=prob)
                rea_dataset = RandomSmilesDataset(rea_dataset, self.args.seed, prob=prob)
                # pro_dataset = CannoicalSmilerDataset(pro_dataset)   
                # rea_dataset = CannoicalSmilerDataset(rea_dataset) 
                pro_dataset1 = pro_dataset
                aug_index_dataset = AugPerMapDataset(pro_dataset1)
                align_matrix_dataset = AlignmentMatrixDataset(pro_dataset, rea_dataset, 1)
                # atom_index_dataset = AtomIdxSmilePosDataset(pro_dataset1)

            else:
                prob = self.pro_aug_prob
                process_dataset = RandomReactionSmilesDataset(pro_dataset, rea_dataset, self.args.seed, prob=prob, epoch_t = self.aug_strategy_epoch)
                pro_dataset = TensorDimDataset(process_dataset, 0)
                rea_dataset = TensorDimDataset(process_dataset, 1)
                pro_dataset1 = TensorDimDataset(process_dataset, 2)
                rea_dataset1 = TensorDimDataset(process_dataset, 3)
                aug_index_dataset = AugPerMapDataset(pro_dataset1)
                align_matrix_dataset = AlignmentMatrixDataset(pro_dataset1, rea_dataset1)
                # atom_index_dataset = AtomIdxSmilePosDataset(pro_dataset1)

            pro_dataset = CleanMapNumberDataset(pro_dataset)
            rea_dataset = CleanMapNumberDataset(rea_dataset)

            pro_dataset = SmilesTokenizerDataset(pro_dataset)
            rea_dataset = SmilesTokenizerDataset(rea_dataset)
            pro_dataset = TokenizeDataset(
                pro_dataset, self.dictionary, max_seq_len=self.args.max_seq_len)
            rea_dataset = TokenizeDataset(
                rea_dataset, self.dictionary, max_seq_len=self.args.max_seq_len)          
            return pro_dataset, rea_dataset, aug_index_dataset, align_matrix_dataset 
        
        rea_flag_dataset = None
        if kwargs["split"] in ['valid', 'test'] and self.use_multi_smile_infer_augment > 0:
            token_dataset, token_target_dataset = map_data_multi_infer_dataset(token_dataset, token_target_dataset)
            token_target_dataset = PrependAndAppend(token_target_dataset, self.dictionary.bos(), self.dictionary.eos())

        elif self.use_align_augment > 0:
            token_dataset, token_target_dataset = map_data_align_check_aug_dataset(token_dataset, token_target_dataset)

            token_dataset = PrependAndAppend(token_dataset, self.dictionary.bos(), self.dictionary.eos()) 
            token_target_dataset = PrependAndAppend(token_target_dataset, self.dictionary.bos(), self.dictionary.eos())

        elif self.use_num_align_augment > 0:
            token_dataset, token_target_dataset = map_data_align_number_aug_dataset(token_dataset, token_target_dataset)

            token_dataset = PrependAndAppend(token_dataset, self.dictionary.bos(), self.dictionary.eos()) 
            token_target_dataset = PrependAndAppend(token_target_dataset, self.dictionary.bos(), self.dictionary.eos())

            # token_dataset = PrependAndAppend(token_dataset, self.dictionary.bos(), self.dictionary.eos()) 
            # token_target_dataset = PrependAndAppend(token_target_dataset, self.dictionary.bos(), self.dictionary.eos()) 

        elif self.use_map_data > 0:

            if self.args.use_syn_aug > 0:
                rea_flag_dataset, token_dataset, token_target_dataset = reverse_data_aug_dataset(token_dataset, token_target_dataset)
            
            token_dataset = map_data_aug_product_dataset(token_dataset)
            token_target_dataset = map_data_aug_dataset(token_target_dataset)

            token_dataset = PrependAndAppend(token_dataset, self.dictionary.bos(), self.dictionary.eos()) 
            token_target_dataset = PrependAndAppend(token_target_dataset, self.dictionary.bos(), self.dictionary.eos()) 
        else:
            token_dataset = data_aug_dataset(token_dataset)
            token_target_dataset = data_aug_dataset(token_target_dataset)

        if self.use_multi_smile_infer_augment > 0:
            return {
                "src_tokens": PadListDataset(
                    token_dataset,
                    pad_idx=self.dictionary.pad(),
                ),
                "decoder_src_tokens": RightPadDataset(
                    token_target_dataset,
                    pad_idx=self.dictionary.pad(),
                ),
                # "reaction_type": RawLabelDataset(class_dataset),
            }, {

            }              
        elif self.args.use_syn_aug > 0:
            return {
            "src_tokens": RightPadDataset(
                token_dataset,
                pad_idx=self.dictionary.pad(),
            ),
            "decoder_src_tokens": RightPadDataset(
                token_target_dataset,
                pad_idx=self.dictionary.pad(),
            ),
            "flag_reaction":RawLabelDataset(rea_flag_dataset),
            # "reaction_type": RawLabelDataset(class_dataset),
        }, {

        }
        return {
            "src_tokens": RightPadDataset(
                token_dataset,
                pad_idx=self.dictionary.pad(),
            ),
            "decoder_src_tokens": RightPadDataset(
                token_target_dataset,
                pad_idx=self.dictionary.pad(),
            ),
            # "reaction_type": RawLabelDataset(class_dataset),
        }, {

        }

    def one_dataset_inference(self, raw_dataset, **kwargs):

        if self.task_type == 'retrosynthetic':
            input_name, output_name = 'smiles_mapnumber_target_list', 'smiles_mapnumber_reactant_list'
        elif self.task_type == 'synthetic':
            input_name, output_name = 'smiles_mapnumber_reactant_list', 'smiles_mapnumber_target_list' 

        token_dataset = KeyDataset(raw_dataset, input_name)
        token_target_dataset = KeyDataset(raw_dataset, output_name)

        # token_dataset1 = KeyDataset(raw_dataset, input_name)
        # token_target_dataset1 = KeyDataset(raw_dataset, output_name)

        class_dataset = KeyDataset(raw_dataset, 'class')

        def PrependAndAppend(dataset, pre_token, app_token):
            dataset = PrependTokenDataset(dataset, pre_token)
            return AppendTokenDataset(dataset, app_token)

        def map_data_aug_dataset(dataset):
            prob = 0.0
            dataset = RandomSmilesDataset(dataset, self.args.seed, prob=prob)
            dataset = BpeTokenDataset(dataset, self.bpe, self.args.max_seq_len)
            dataset = CutDataset(dataset, self.args.max_seq_len)
            dataset = TokenizeDataset(
                dataset, self.dictionary, max_seq_len=self.args.max_seq_len)
            return dataset

        token_dataset = map_data_aug_dataset(token_dataset)
        token_target_dataset = map_data_aug_dataset(token_target_dataset)
        token_dataset = PrependAndAppend(token_dataset, self.dictionary.bos(), self.dictionary.eos()) 
        token_target_dataset = PrependAndAppend(token_target_dataset, self.dictionary.bos(), self.dictionary.eos())

        return {
            "src_tokens": RightPadDataset(
                token_dataset,
                pad_idx=self.dictionary.pad(),
            ),
            "decoder_src_tokens": RightPadDataset(
                token_target_dataset,
                pad_idx=self.dictionary.pad(),
            ),
            # "reaction_type": RawLabelDataset(class_dataset),
        }, {

        }

    def load_infer_dataset(self, smiles, **kwargs):
        
        #先规定少一点做测试
        assert len(smiles) < 1000
        raw_dataset = InferSmilesDataset(smiles, self.infer_task_type, **kwargs)
        net_input, target = self.one_dataset_inference(
            raw_dataset)
        dataset = {'net_input': net_input, 'target': target}
        dataset = NestedDictionaryDataset(
            dataset
        )
        return dataset, raw_dataset

    def test_list_step(self, args, sample, generator, loss, step, seed):

        total_src_tokens = sample['net_input']['src_tokens'] 
        for i in range(total_src_tokens.shape[1]):
            tgt_tokens = sample['net_input']['decoder_src_tokens']
            src_tokens = total_src_tokens[:, i, :] 
            sample['net_input']['src_tokens'] = src_tokens

            if args.search_strategies == "SimpleGenerator":
                pred, vae_kl_loss,likelyhood = generator._generate(sample)
            else:
                pred = generator(sample)

            tgt_tokens = tgt_tokens.cpu().numpy()
            beam_size = generator.beam_size
            self.translate_tokens(args, pred, src_tokens, tgt_tokens, beam_size)

        return pred, { "sample_size": 1, }


    def translate_tokens(self, args, pred, src_tokens, tgt_tokens, beam_size):

        for i in range(len(tgt_tokens)):

            a = tgt_tokens[i]
            mol_str = []
            for t in a[1:]:
                tt = self.dictionary.decode(t)
                if tt == '[SEP]':
                    break
                mol_str.append(tt)
            mol_str = self.bpe_decoder(mol_str)

            b = src_tokens[i]
            mol_str2 = []
            for t in b[:]:
                tt =  self.dictionary.decode(t.item())
                if tt == '[SEP]':
                    break
                mol_str2.append(tt)
            mol_str2 = self.bpe_decoder(mol_str2)

            file_path = args.results_smi_path
            if not os.path.exists(file_path):
                os.makedirs(file_path)

            file_check2 = os.path.join(file_path, 'raw_data_'+args.results_smi_file)
            with open(file_check2, 'a') as w1:
                w1.write(mol_str2+'\t'+mol_str+'\n')

            file_check = os.path.join(file_path, args.results_smi_file)
            with open(file_check, 'a') as w:
                w.write(mol_str+'\t'+'target'+'\n')

                for j in range(beam_size):
                    mol_smi = []
                    mol_str2 = []
                    if args.search_strategies == "SimpleGenerator":
                        a = pred[j + i * beam_size].cpu().numpy()
                        score = likelyhood[j + i * beam_size]
                        for t in a[1:]:
                            tt = self.dictionary.decode(t)
                            if tt == "[PAD]":
                                break
                            mol_str2.append(tt)
                        mol_str2 = self.bpe_decoder(mol_str2)
                    else:
                        c = pred[i][j]['tokens'].cpu().numpy()
                        score = pred[i][j]['score'].cpu().detach().numpy()
                        for t in c[:]:
                            tt = self.dictionary.decode(t)
                            if tt == '[SEP]':
                                break
                            mol_str2.append(tt)
                        mol_str2 = self.bpe_decoder(mol_str2)

                    # 加一个rdkit转分子对象能否成功的判断
                    smiles = mol_str2
                    # if Chem.MolFromSmiles(smiles) is not None:
                    w.write(smiles+'\t'+ str(score) +'\t'+'predicted'+'\t'+str(j)+'\n')
                    mol_smi.append(smiles)

    def translate_tokens(self, args, pred, src_tokens, tgt_tokens, beam_size):

        for i in range(len(tgt_tokens)):

            a = tgt_tokens[i]
            mol_str = []
            for t in a[1:]:
                tt = self.dictionary.decode(t)
                if tt == '[SEP]':
                    break
                mol_str.append(tt)
            mol_str = self.bpe_decoder(mol_str)

            b = src_tokens[i]
            mol_str2 = []
            for t in b[:]:
                tt =  self.dictionary.decode(t.item())
                if tt == '[SEP]':
                    break
                mol_str2.append(tt)
            mol_str2 = self.bpe_decoder(mol_str2)

            file_path = args.results_smi_path
            if not os.path.exists(file_path):
                os.makedirs(file_path)

            file_check2 = os.path.join(file_path, 'raw_data_'+args.results_smi_file)
            with open(file_check2, 'a') as w1:
                w1.write(mol_str2+'\t'+mol_str+'\n')

            file_check = os.path.join(file_path, args.results_smi_file)
            with open(file_check, 'a') as w:
                w.write(mol_str+'\t'+'target'+'\n')

                for j in range(beam_size):
                    mol_smi = []
                    mol_str2 = []

                    # a = pred[j + i * beam_size].cpu().numpy()
                    # score = likelyhood[j + i * beam_size]
                    # for t in a[1:]:
                    #     tt = self.dictionary.decode(t)
                    #     if tt == "[PAD]":
                    #         break
                    #     mol_str2.append(tt)
                    # mol_str2 = self.bpe_decoder(mol_str2)

                    c = pred[i][j]['tokens'].cpu().numpy()
                    score = pred[i][j]['score'].cpu().detach().numpy()
                    for t in c[:]:
                        tt = self.dictionary.decode(t)
                        if tt == '[SEP]':
                            break
                        mol_str2.append(tt)
                    mol_str2 = self.bpe_decoder(mol_str2)

                    # 加一个rdkit转分子对象能否成功的判断
                    smiles = mol_str2
                    # if Chem.MolFromSmiles(smiles) is not None:
                    w.write(smiles+'\t'+ str(score) +'\t'+'predicted'+'\t'+str(j)+'\n')
                    mol_smi.append(smiles)


    def test_step(self, args, sample, generator, loss, step, seed):

        tgt_tokens = sample['net_input']['decoder_src_tokens']
        src_tokens = sample['net_input']['src_tokens'] 
        
        if args.search_strategies == "SimpleGenerator":
            pred, tgt_lengths, likelyhood = generator._generate(sample)
        else:
            pred = generator(sample)

        tgt_tokens = tgt_tokens.cpu().numpy()
        beam_size = generator.beam_size

        for i in range(len(tgt_tokens)):

            a = tgt_tokens[i]
            mol_str = []
            for t in a[1:]:
                tt = self.dictionary.decode(t)
                if tt == '[SEP]':
                    break
                mol_str.append(tt)
            mol_str = self.bpe_decoder(mol_str)

            b = src_tokens[i]
            mol_str2 = []
            for t in b[:]:
                tt =  self.dictionary.decode(t.item())
                if tt == '[SEP]':
                    break
                mol_str2.append(tt)
            mol_str2 = self.bpe_decoder(mol_str2)

            file_path = args.results_smi_path
            if not os.path.exists(file_path):
                os.makedirs(file_path)
            file_check = os.path.join(file_path, args.results_smi_file)
            file_check2 = os.path.join(file_path, 'raw_data_'+args.results_smi_file)

            with open(file_check2, 'a') as w1:
                w1.write(mol_str2+'\t'+mol_str+'\n')

            with open(file_check, 'a') as w:
                w.write(mol_str+'\t'+'target'+'\n')

                for j in range(beam_size):
                    mol_smi = []
                    mol_str2 = []

                    if args.search_strategies == "SimpleGenerator":
                        a = pred[j + i * beam_size].cpu().numpy()
                        score = likelyhood[j + i * beam_size]
                        for t in a[1:]:
                            tt = self.dictionary.decode(t)
                            if tt == "[PAD]":
                                break
                            mol_str2.append(tt)
                        mol_str2 = self.bpe_decoder(mol_str2)
                        
                    else:
                        a = pred[i][j]['tokens'].cpu().numpy()
                        score = pred[i][j]['score'].cpu().detach().numpy()
                        for t in a[:]:
                            tt = self.dictionary.decode(t)
                            if tt == '[SEP]':
                                break
                            mol_str2.append(tt)
                        mol_str2 = self.bpe_decoder(mol_str2)

                    # 加一个rdkit转分子对象能否成功的判断
                    smiles = mol_str2
                    # if Chem.MolFromSmiles(smiles) is not None:
                    w.write(smiles+'\t'+ str(score) +'\t'+'predicted'+'\t'+str(j)+'\n')
                    mol_smi.append(smiles)

        return pred, { "sample_size": 1, "bsz": sample['net_input']['decoder_src_tokens'].size(0),
                      "seq_len": sample['net_input']['decoder_src_tokens'].size(1) * sample['net_input']['decoder_src_tokens'].size(0), }


    def infer_step(self, args, sample, generator, **kwargs):

        tgt_tokens = sample['net_input']['decoder_src_tokens']
        src_tokens = sample['net_input']['src_tokens']      
        pred = generator(sample)
        # pred, vae_kl_loss,likelyhood = generator._generate(sample)
        beam_size = generator.beam_size

        search_res_list = []
        for i in range(len(src_tokens)):

            a = src_tokens[i].cpu().numpy()
            mol_str = []
            for t in a[1:]:
                tt = self.dictionary.decode(t)
                if tt == '[SEP]':
                    break
                mol_str.append(tt)
            mol_str = self.bpe_decoder(mol_str)

            for j in range(beam_size):    
                mol_str2 = []
                b = pred[i][j]['tokens'].cpu().numpy()
                score = pred[i][j]['score'].cpu().detach().numpy()
                for t in b[:]:
                    tt = self.dictionary.decode(t)
                    if tt == '[SEP]':
                        break
                    mol_str2.append(tt)
                mol_str2 = self.bpe_decoder(mol_str2)

                search_res_dict = {}
                search_res_dict['target'] = mol_str
                pre_col = 'pred'
                pre_col_score = 'pre_score' 
                search_res_dict[pre_col] = mol_str2
                search_res_dict[pre_col_score] = score 
                search_res_list.append(search_res_dict)

        mol_search_res = pd.DataFrame.from_dict(search_res_list)

        return mol_search_res

    def bpe_decoder(self, mol_str_list):
        mol_str = self.bpe.decode(mol_str_list)
        return mol_str

    def build_model(self, args):
        from unicore import models
        model = models.build_model(args, self)
        model_dict = model.state_dict()

        if args.bart_path != "None":
            pretrained_state = checkpoint_utils.load_checkpoint_to_cpu(args.bart_path)
            pretrained_state = pretrained_state["model"]
            load_pretrained_state = {}
            for k, v in pretrained_state.items():
                if (k in model_dict and v.shape == model_dict[k].shape):
                    load_pretrained_state.update({k:v})
                elif (k in model_dict and v.shape != model_dict[k].shape):
                    load_pretrained_state.update(copy_model_parameters(k, v, model_dict[k]))

            # pretrained_state = {k: v for k, v in pretrained_state.items() if (k in model_dict and v.shape == model_dict[k].shape)}
            model_dict.update(load_pretrained_state)
            model.load_state_dict(model_dict, strict=False)

            logger.info("load weight bart smile. ")

        if args.freeze_bart:   
            freeze_layer(model)
            logger.info("freeze part bart smile. ")
        return model

def freeze_embedding(model):

    for name, param in model.named_parameters():
        # param.requires_grad and 
        if param.requires_grad and 'embed_tokens' in name:
            param.requires_grad=False

def freeze_layer(model):
    for name, param in model.named_parameters():
        if param.requires_grad and ('encoder.layers.0.' in name or 'encoder.layers.1.' in name or 'decoder.layers.1.' in name or 'decoder.layers.1.' in name):
            param.requires_grad=False

def copy_model_parameters(key, pre_train_v, new_model_v):
    model_shape = new_model_v.shape
    pretrain_shape = pre_train_v.shape
    assert len(model_shape) == len(pretrain_shape)
    assert len(pretrain_shape) <= 2
    if len(model_shape) == 1:
        new_model_v = copy_one_dim_parameters(pretrain_shape, model_shape, pre_train_v, new_model_v)
    elif len(model_shape) == 2:
        new_model_v = copy_two_dim_parameters(pretrain_shape, model_shape, pre_train_v, new_model_v)  
    return {key:new_model_v}
    
def copy_one_dim_parameters(pretrain_shape, model_shape, pre_train_v, new_model_v):
    copy_shape = min(model_shape, pretrain_shape)
    if model_shape < pretrain_shape:
        new_model_v = pre_train_v[:copy_shape]
        v_new = new_model_v
    elif model_shape > pretrain_shape:
        new_model_v[:copy_shape] = pre_train_v
        v_new = new_model_v
    else:
        v_new = pretrain_shape
    return v_new

def copy_two_dim_parameters(pretrain_shape, model_shape, pre_train_v, new_model_v):
    copy_shape_x = min(model_shape[0], pretrain_shape[0])  
    copy_shape_y = min(model_shape[1], pretrain_shape[1])   
    new_model_v[:copy_shape_x,:copy_shape_y] = pre_train_v
    v_new = new_model_v
    return v_new

@register_task("ReRP_Bart_RS_Pr")
class ReRPPrBartTask_RS(ReRPPrBartTask):
    def __init__(self, args, dictionary):
        super().__init__(args, dictionary)
        self.task_type = "retrosynthetic"


@register_task("ReRP_Bart_S_Pr")
class ReRPPrBartTask_S(ReRPPrBartTask):
    def __init__(self, args, dictionary):
        super().__init__(args, dictionary)
        self.task_type = "synthetic"
