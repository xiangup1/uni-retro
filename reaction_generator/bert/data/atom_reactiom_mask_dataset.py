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

class AtomReactionMaskDataset(BaseWrapperDataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        return self.get_atom_reaction_mask_list(self.dataset[idx])

    def get_atom_reaction_mask_list(self, rc_product, smi):
        ### 暂时不这样用
        smi_list = self.smi_tokenizer(smi)
        mol = Chem.MolFromSmiles(smi)
        reaction_center_list = [False for atoms in mol.GetAtoms()]
        atom_info = {}
        for atom in mol.GetAtoms():
            atom_info[atom.GetAtomMapNum()] = (atom.GetSymbol(), atom.GetIdx())

        for i,reaction_flag in enumerate(rc_product):
            if reaction_flag:
                token_reaction = smi_list[i]
                if re.match('.*:([0-9]+)]', token_reaction):
                    token_mol = Chem.MolFromSmarts(token_reaction)
                    l = [(atom.GetSymbol(), atom.GetAtomMapNum(), atom.GetIdx()) for atom in token_mol.GetAtoms()]
                    atom_map_num = l[0][1]
                    atom_idx = atom_info[atom_map_num][1]
                    reaction_center_list[atom_idx] = True

        return torch.from_numpy(np.array(reaction_center_list))
        
    def smi_tokenizer(self, smi):
        """Tokenize a SMILES sequence or reaction"""
        pattern = "(\[[^\]]+]|Bi|Br?|Ge|Te|Mo|K|Ti|Zr|Y|Na|125I|Al|Ce|Cr|Cl?|Ni?|O|S|Pd?|Fe?|I|b|c|Mn|n|o|s|<unk>|>>|Li|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
        regex = re.compile(pattern)
        tokens = [token for token in regex.findall(smi)]
        assert smi == ''.join(tokens)
        return tokens   