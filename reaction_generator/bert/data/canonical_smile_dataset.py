# Copyright (c) DP Techonology, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from functools import lru_cache
from unicore.data import BaseWrapperDataset
import logging
from rdkit import Chem
logger = logging.getLogger(__name__)

class CannoicalSmilerDataset(BaseWrapperDataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
            return ".".join([self.canoical_smile_number(i) for i in self.dataset[idx].split('.')])

    def canoical_smile_number(self, smi):
        """Clear the atom mapping number of a SMILES sequence"""
        try: 
            mol = Chem.MolFromSmiles(smi)
            return Chem.MolToSmiles(mol)
        except:
            return smi