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
logger = logging.getLogger(__name__)

class AtomIdxSmilePosDataset(BaseWrapperDataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
            return self.get_atom_idx_smile_pos_list(self.dataset[idx])

    def get_atom_idx_smile_pos_list(self, aug_smi):

        smi_list = self.smi_tokenizer(aug_smi)
        mol = Chem.MolFromSmiles(aug_smi)
        idx_transfer_list = [i for i in range(len(mol.GetAtoms()))]
        atom_info = {}
        atom_info_list = [(atom.GetSymbol(), atom.GetAtomMapNum(), atom.GetIdx()) for atom in mol.GetAtoms()]
        for atom in mol.GetAtoms():
            atom_info[atom.GetAtomMapNum()] = (atom.GetSymbol(), atom.GetIdx())

        for i, token in enumerate(smi_list):
            if re.match('.*:([0-9]+)]', token):
                token_mol = Chem.MolFromSmarts(token)
                l = [(atom.GetSymbol(), atom.GetAtomMapNum(), atom.GetIdx()) for atom in token_mol.GetAtoms()]
                smile_position = i
                atom_map_num = l[0][1]
                atom_idx = atom_info[atom_map_num][1]
                idx_transfer_list[atom_idx] = smile_position
        # assert len(idx_transfer_list) == len(atom_info_list), 'atom number should be same.'
        # print('test get_atom_idx_smile_pos_list: ', atom_info_list)
        # print('test                    smi_list: ', smi_list, len(smi_list))
        # print('test           idx_transfer_list: ', idx_transfer_list)
        return torch.from_numpy(np.array(idx_transfer_list))

    def smi_tokenizer(self, smi):
        """Tokenize a SMILES sequence or reaction"""
        pattern = "(\[[^\]]+]|Bi|Br?|Ge|Te|Mo|K|Ti|Zr|Y|Na|125I|Al|Ce|Cr|Cl?|Ni?|O|S|Pd?|Fe?|I|b|c|Mn|n|o|s|<unk>|>>|Li|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
        regex = re.compile(pattern)
        tokens = [token for token in regex.findall(smi)]
        assert smi == ''.join(tokens)
        return tokens
