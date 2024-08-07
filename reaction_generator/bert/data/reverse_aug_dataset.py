# Copyright (c) DP Techonology, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import numpy as np
from functools import lru_cache
from unicore.data import BaseWrapperDataset
import logging
logger = logging.getLogger(__name__)
from unicore.data import data_utils
import random

class ReverseAugDataset(BaseWrapperDataset):
    def __init__(self, dataset, dataset2, seed, prob):
        self.dataset = dataset
        self.dataset2 = dataset2
        self.size1 = len(self.dataset)
        self.size2 = len(self.dataset2)
        self.prob = prob
        self.seed = seed

    def __len__(self):
        return self.size1

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        with data_utils.numpy_seed(self.seed, self.epoch):
            sample_p = random.random()   
        if self.prob != 0 and sample_p < self.prob:
            return 2, self.dataset2[idx], self.dataset[idx]
        else:
            return 1, self.dataset[idx], self.dataset2[idx]



