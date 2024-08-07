# Copyright (c) DP Techonology, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from functools import lru_cache
from unicore.data import BaseWrapperDataset
import logging
from rdkit import Chem
import torch
import numpy as np
import re
from .data_utils import smi_tokenizer

logger = logging.getLogger(__name__)

class AlignmentMatrixDataset(BaseWrapperDataset):
    # mainly from https://github.com/yuewan2/Retroformer/tree/4e94540e06f5a0ed52ac7b7c4acdd98e92aeda56/retroformer
    def __init__(self, dataset, rea_dataset, training_flag = 0):
        self.dataset = dataset
        self.rea_dataset = rea_dataset
        self.training_flag = training_flag

    def __len__(self):
        return len(self.dataset)

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        if self.training_flag == 0:
            return self.get_alignment_matrix(self.dataset[idx], self.rea_dataset[idx])
        else:
            return torch.zeros((3,3))


    def get_alignment_matrix(self, pro_smi, rea_smi):

        pro_smi_l =  smi_tokenizer(pro_smi)
        rea_smi_l =  smi_tokenizer(rea_smi)

        # prodToken2posIdx = {}
        # position_mapping_list = []
        # mp = 0
        # for i, token in enumerate(pro_smi_l):
        #     if token[0] == '[' and token[-1] == ']' and re.match('.*:([0-9]+)]', token):
        #         am = int(re.match('.*:([0-9]+)]', token).group(1))
        #         prodToken2posIdx[am] = i
        # for j, token in enumerate(rea_smi_l):
        #     if token[0] == '[' and token[-1] == ']' and re.match('.*:([0-9]+)]', token):
        #         am2 = int(re.match('.*:([0-9]+)]', token).group(1)) 
        #         posidx = prodToken2posIdx.get(am2, -1) 
        #         mp = max(mp, posidx)
        #         if  posidx != -1 and (j, posidx) not in position_mapping_list:
        #             position_mapping_list.append((j, posidx))

        # assert mp < len(pro_smi_l) 

        prodToken2posIdx = {}
        for i, token in enumerate(pro_smi_l):
            if token[0] == '[' and token[-1] == ']' and re.match('.*:([0-9]+)]', token):
                am = int(re.match('.*:([0-9]+)]', token).group(1))
                prodToken2posIdx[am] = i
            else:
                prodToken2posIdx[token] = prodToken2posIdx.get(token, []) + [i]
        
        position_mapping_list = []
        for i, token in enumerate(rea_smi_l):
            if token[0] == '[' and token[-1] == ']' and re.match('.*:([0-9]+)]', token):
                am = int(re.match('.*:([0-9]+)]', token).group(1))
                prod_posIdx = prodToken2posIdx.get(am, -1)

                if prod_posIdx != -1:
                    if (i, prod_posIdx) not in position_mapping_list:
                        position_mapping_list.append((i, prod_posIdx))

        alignment_matrix = torch.zeros(
            (len(rea_smi_l), len(pro_smi_l)))
        for (i, j) in position_mapping_list:
            alignment_matrix[i][j] = 1
        return alignment_matrix


