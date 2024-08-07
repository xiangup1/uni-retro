# Copyright (c) DP Techonology, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from unittest.mock import NonCallableMagicMock
from functools import lru_cache
from unicore.data import BaseWrapperDataset
from unicore.data import data_utils
import numpy as np
from rdkit import Chem
import re
import random
import logging
logger = logging.getLogger(__name__)
from .data_utils import smi_tokenizer


class InferMultiReactionSmilesDataset(BaseWrapperDataset):
    def __init__(self, pro_dataset, rec_dataset, seed, times = 10):
        self.rec_dataset = rec_dataset
        self.dataset = pro_dataset
        self.seed = seed
        self.set_epoch(None)
        self.epoch = 0
        self.times = times

    def __len__(self):
        return len(self.dataset)

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch

    def get_epoch(self):
        return self.epoch

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        pro_smis, rec_smi = self.get_multi_infer_smiles(self.dataset[idx], self.rec_dataset[idx])
        return pro_smis, rec_smi

    def randomize_multi_smiles(self, smiles):
        """Perform a randomization of a SMILES string
        must be RDKit sanitizable"""
        try:
            res_smis = []
            for i in range(self.times-1):
                aug_smi = self.random_smi(smiles)
                res_smis.append(aug_smi)
            res_smis = [smiles] + res_smis
        except:
            res_smis = [smiles] * self.times
            print('Warning: illegal smiles: ', res_smis)
        return res_smis

    def random_smi(self, smiles):
        smile_list = smiles.split('.')
        res_smi_list = []
        for smi in smile_list:
            am = Chem.MolFromSmiles(smi)
            aug_smi = Chem.MolToSmiles(am, doRandom=True)
            res_smi_list.append(aug_smi)
        rsmi = ".".join([r_smi for r_smi in res_smi_list])
        return rsmi

    def get_multi_infer_smiles(self, product, reactant):
        product =  ".".join([i for i in product])
        reactant =  ".".join([i for i in reactant])

        product = self.clear_map_canonical_smiles(product)
        rec_smi = self.clear_map_canonical_smiles(reactant)
        pro_smis = self.randomize_multi_smiles(product)
        return pro_smis, rec_smi

    def clear_map_canonical_smiles(self, smi):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            for atom in mol.GetAtoms():
                if atom.HasProp('molAtomMapNumber'):
                    atom.ClearProp('molAtomMapNumber')
            return Chem.MolToSmiles(mol, canonical=False)
        else:
            return smi

