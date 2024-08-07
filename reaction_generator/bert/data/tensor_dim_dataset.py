# Copyright (c) DP Techonology, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from functools import lru_cache
from unicore.data import BaseWrapperDataset
import logging
import torch
import numpy as np
logger = logging.getLogger(__name__)

class TensorDimDataset(BaseWrapperDataset):
    def __init__(self, dataset, dim):
        self.dataset = dataset
        self.dim = dim

    def __len__(self):
        return len(self.dataset)

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
            return self.dataset[idx][self.dim]