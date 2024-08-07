# Copyright (c) DP Techonology, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from unittest.mock import NonCallableMagicMock
from functools import lru_cache
from unicore.data import BaseWrapperDataset
from unicore.data import data_utils
import torch
import numpy as np
from rdkit import Chem
import re
import random
import logging
logger = logging.getLogger(__name__)
from .data_utils import smi_tokenizer


class InferMultiReactionSmilesProProcessDataset(BaseWrapperDataset):
    def __init__(self, pros_dataset, bpe, dictionary, max_seq_len, seed = 42, times = 10):
        self.dataset = pros_dataset
        self.bpe = bpe
        self.dictionary = dictionary
        self.bos = self.dictionary.bos()
        self.eos = self.dictionary.eos()
        self.max_seq_len = max_seq_len
        self.seed = seed
        self.set_epoch(None)
        self.epoch = 0
        self.times = times

    def __len__(self):
        return len(self.dataset)

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch

    def get_epoch(self):
        return self.epoch

    def bpe_encode(self, line):
        ids = self.bpe.EncodeAsPieces(line)
        output = list(map(str, ids))
        output = np.array(output)
        return output

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        pro_smis = self.get_multi_infer_pro_smiles(self.dataset[idx])
        return pro_smis

    def get_multi_infer_pro_smiles(self, products):
        pro_token_list = []
        for i, pro in enumerate(products):
            pro = self.bpe_encode(pro)
            pro = pro if len(pro) < self.max_seq_len-2 else pro[:self.max_seq_len-2]
            pro = torch.from_numpy(self.dictionary.vec_index(pro)).long()
            pro = torch.cat([torch.full_like(pro[0], self.bos).unsqueeze(0), pro], dim=0)
            pro = torch.cat([pro, torch.full_like(pro[0], self.eos).unsqueeze(0)], dim=0)
            pro_token_list.append(pro)   
        return pro_token_list

