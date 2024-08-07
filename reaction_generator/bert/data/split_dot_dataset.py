# Copyright (c) DP Techonology, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from functools import lru_cache
from unicore.data import BaseWrapperDataset
import numpy as np
import torch
import logging
logger = logging.getLogger(__name__)

class SplitDotDataset(BaseWrapperDataset):
    def __init__(self, dataset, padding_id, dot_id):
        self.dataset = dataset
        self.padding_id = padding_id
        self.dot_id = dot_id
        self.dotdataset = []

    def __len__(self):
        return len(self.dataset)

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
            return self.dotdataset[idx]

    def get_dot_dataset(self):
        for idx, item in enumerate(self.dataset):
            self.dotdataset.append(self.get_dot_array(item))
        return self.dotdataset

    def get_dot_array(self, arraydata):
        array_data_set = np.array([0]*len(arraydata))
        second_pro = False
        for j in range(len(arraydata)):
            array_data_set[j] = 1
            if arraydata[j].eq(self.dot_id):
                array_data_set[j] = 2
                second_pro = True
                continue
            if second_pro: 
                array_data_set[j] = 3  
            if arraydata[j].eq(self.padding_id):
                array_data_set[j] = 0
        return torch.from_numpy(array_data_set.astype(np.int32))