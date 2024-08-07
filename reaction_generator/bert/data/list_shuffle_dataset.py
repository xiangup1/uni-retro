# Copyright (c) DP Techonology, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from functools import lru_cache
from unicore.data import BaseWrapperDataset
from unicore.data import data_utils
import random

import logging
logger = logging.getLogger(__name__)


class ListShuffleDataset(BaseWrapperDataset):
    def __init__(self, dataset, seed, prob=1.0, epoch_t = 0):
        self.dataset = dataset
        self.prob = prob
        self.set_epoch(None)
        self.seed = seed
        self.epoch_t = epoch_t

    def __len__(self):
        return len(self.dataset)

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        tmp_list = self.dataset[idx]
        with data_utils.numpy_seed(self.seed, self.epoch):
            sample_p = random.random()
            
        if self.prob != 0 and sample_p < self.prob and self.epoch < self.epoch_t and self.epoch_t > 0:
            random.shuffle(tmp_list)
        return tmp_list
