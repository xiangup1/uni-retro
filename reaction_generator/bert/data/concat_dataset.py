# Copyright (c) DP Techonology, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from unittest.mock import NonCallableMagicMock
from functools import lru_cache
from unicore.data import BaseWrapperDataset
from unicore.data import data_utils
from rdkit import Chem
import random
import logging
logger = logging.getLogger(__name__)


class ConcatDataset(BaseWrapperDataset):
    def __init__(self, dataset, sep_token = '.'):
        self.dataset = dataset
        self.sep_token = sep_token

    def __len__(self):
        return len(self.dataset)

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch

    def get_epoch(self):
        return self.epoch

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        if len(self.dataset[idx]) > 1:
            return self.sep_token.join([i for i in self.dataset[idx]])
        else:
            return self.dataset[idx][0]
