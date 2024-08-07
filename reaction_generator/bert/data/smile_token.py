# Copyright (c) DP Techonology, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from unittest.mock import NonCallableMagicMock
from functools import lru_cache
from unicore.data import BaseWrapperDataset
from rdkit import Chem
import random
import re
import logging
logger = logging.getLogger(__name__)


class SmilesTokenizerDataset(BaseWrapperDataset):
    def __init__(self, dataset, prob=1.0):
        self.dataset = dataset
        self.prob = prob

    def __len__(self):
        return len(self.dataset)

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch

    def smi_tokenizer(self, smi):
        """Tokenize a SMILES sequence or reaction"""
        pattern = "(\[[^\]]+]|Bi|Br?|Ge|Te|Mo|K|Ti|Zr|Y|Na|125I|Al|Ce|Cr|Cl?|Ni?|O|S|Pd?|Fe?|I|b|c|Mn|n|o|s|<unk>|>>|Li|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
        regex = re.compile(pattern)
        tokens = [token for token in regex.findall(smi)]
        if smi != ''.join(tokens):
            print('test illegal smile: ', smi, ''.join(tokens))
        assert smi == ''.join(tokens)
        return tokens

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        return self.smi_tokenizer(self.dataset[idx])
