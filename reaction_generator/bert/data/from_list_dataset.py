# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from functools import lru_cache
import numpy as np
from unicore.data import BaseWrapperDataset


class FromListDataset(BaseWrapperDataset):
    def __init__(self, dataset):
        super().__init__(dataset)

    def __len__(self):
        return len(dataset)

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        array_data = np.array(self.dataset[idx])
        return torch.from_numpy(array_data)