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


class RandomSmilesDataset(BaseWrapperDataset):
    def __init__(self, dataset, seed, prob=1.0, epoch_t = 0):
        self.dataset = dataset
        self.prob = prob
        self.seed = seed
        self.set_epoch(None)
        self.epoch = 0
        self.epoch_t = epoch_t

    def __len__(self):
        return len(self.dataset)

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch

    def get_epoch(self):
        return self.epoch

    def get_random_smiles(self, smi):
        with data_utils.numpy_seed(self.seed, self.epoch):
            sample_p = random.random()

        if self.prob == 0 or sample_p >= self.prob:
            return smi
        if self.epoch > self.epoch_t and self.epoch_t > 0:
            return smi
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                return smi
            return Chem.MolToSmiles(mol, doRandom=True)
        except:
            return smi

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        smi = ".".join([self.get_random_smiles(i) for i in self.dataset[idx]])
        return smi
