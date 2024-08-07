# Copyright (c) DP Techonology, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from functools import lru_cache
from unicore.data import BaseWrapperDataset
import logging
from rdkit import Chem
import re
logger = logging.getLogger(__name__)

class CleanMapNumberDataset(BaseWrapperDataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
            return ".".join([self.clear_map_number(i) for i in self.dataset[idx].split('.')])

    def clear_map_number(self, smi):
        """Clear the atom mapping number of a SMILES sequence"""
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            for atom in mol.GetAtoms():
                if atom.HasProp('molAtomMapNumber'):
                    atom.ClearProp('molAtomMapNumber')
            return Chem.MolToSmiles(mol, canonical = False)
        else:
            return smi
    
    # def re_clean_smile_num(self, smiles):
    #     # 去掉非@后面的H以及数字
    #     regx1 = re.compile(r'([^@])(H[0-9]*[:]?[0-9]*)')
    #     # 去掉所有的:和数字
    #     regex11 = re.compile(r'[:][0-9]*')
    #     smiles = regx1.sub(r'\g<1>', smiles) 
    #     smiles = regex11.sub(r'', smiles) 
    #     # 方案1：[]中没有@符号的，一律去掉两边的[]
    #     regx2 = re.compile(r'(\[)([a-zA-Z0-9:=]+)(\])')
    #     smiles = regx2.sub("\g<2>", smiles)
    #     return smiles
    # def check_re_smile(self, smiles):
    #     mol1 = Chem.MolFromSmiles(smiles)
    #     smiles2 = Chem.MolToSmiles(mol1)
    #     return smiles