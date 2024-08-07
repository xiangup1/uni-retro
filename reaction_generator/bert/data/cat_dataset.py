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

class CatDataset(BaseWrapperDataset):
    def __init__(self, dataset1, dataset2):
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.size1 = len(self.dataset1)
        self.size2 = len(self.dataset2)

    def __len__(self):
        return self.size1 + self.size2

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
            if idx < len(self.dataset1):
                return self.dataset1[idx]
            else:
                return self.dataset2[idx - self.size1]




