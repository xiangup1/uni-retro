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
logger = logging.getLogger(__name__)

class AugPerMapDataset(BaseWrapperDataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
            return self.get_3d_atom_permuation_list(self.dataset[idx])

    def get_3d_atom_permuation_list(self, aug_smi2):

        try: 
            aug_smi = self.reset_atom_map_num(aug_smi2)
            mol2 = Chem.MolFromSmiles(aug_smi) 
            can_smi = Chem.MolToSmiles(mol2) 
            mol = Chem.MolFromSmiles(can_smi)
            canSmiIdx_to_permuteIdx_list = [atom.GetIdx() for atom in mol.GetAtoms()]

            can_atom_idx_to_atomNum = {}
            for atom in mol.GetAtoms():
                can_atom_idx_to_atomNum[atom.GetIdx()] = (atom.GetSymbol(), atom.GetAtomMapNum(), atom.GetIdx())

            per_atom_atomNum_to_idx = {}
            for atom in mol2.GetAtoms():
                per_atom_atomNum_to_idx[atom.GetAtomMapNum()] = (atom.GetSymbol(), atom.GetAtomMapNum(), atom.GetIdx())

            for atom in mol.GetAtoms():
                atomMapNum = can_atom_idx_to_atomNum[atom.GetIdx()][1]
                permuteAtomIdx = per_atom_atomNum_to_idx[atomMapNum][2]
                canSmiIdx_to_permuteIdx_list[atom.GetIdx()] = permuteAtomIdx
            assert len(list(mol2.GetAtoms())) == len(canSmiIdx_to_permuteIdx_list)
            return torch.from_numpy(np.array(canSmiIdx_to_permuteIdx_list))
        except:
            mol = Chem.MolFromSmiles(aug_smi2) 
            if mol is not None:
                canSmiIdx_to_permuteIdx_list = [atom.GetIdx() for atom in mol.GetAtoms()]
            else:
                canSmiIdx_to_permuteIdx_list = []
                print('ilgeral ami: ', aug_smi2)
            return torch.from_numpy(np.array(canSmiIdx_to_permuteIdx_list))            
    def reset_atom_map_num(self, aug_smi):
        mol2 = Chem.MolFromSmiles(aug_smi)
        try:
            for i, atom in enumerate(mol2.GetAtoms()):
                atom.SetAtomMapNum(i+1)
            return Chem.MolToSmiles(mol2, canonical = False)
        except:
            return aug_smi
