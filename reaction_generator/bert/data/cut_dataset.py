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

class CutDataset(BaseWrapperDataset):
    def __init__(self, dataset, max_seq_len):
        self.max_seq_len = max_seq_len
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
            if len(self.dataset[idx]) < self.max_seq_len-2:
                return self.dataset[idx]
            else:
                print('test aug large data length: ', len(self.dataset[idx]), len(self.dataset[idx][:self.max_seq_len-2]))
                return self.dataset[idx][:self.max_seq_len-2]
    


