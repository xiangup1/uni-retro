# Copyright (c) DP Techonology, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from functools import lru_cache
from unicore.data import BaseWrapperDataset
import numpy as np
import logging
logger = logging.getLogger(__name__)


class BpeTokenDataset(BaseWrapperDataset):
    def __init__(self, dataset, bpe, max_seq_len: int = 512):
        self.dataset = dataset
        self.bpe = bpe
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.dataset)

    def encode(self, line):
        ids = self.bpe.EncodeAsPieces(line)
        return list(map(str, ids))
    
    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        tmp = self.dataset[idx]
        output = self.encode(tmp)
        output = np.array(output)
        # assert len(output) < self.max_seq_len and len(output) > 2
        return output
