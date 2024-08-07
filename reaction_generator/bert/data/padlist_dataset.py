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

from ..data import data_utils



class PadListDataset(BaseWrapperDataset):
    def __init__(self, dataset, pad_idx, left_pad = False):
        super().__init__(dataset)
        self.pad_idx = pad_idx
        self.left_pad = left_pad

    def collater(self, samples):
        return data_utils.collate_list_tokens(samples, self.pad_idx, left_pad=self.left_pad, pad_to_multiple=8)
