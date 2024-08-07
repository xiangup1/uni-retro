import logging
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import re
import pickle
from unicore.data import BaseWrapperDataset
from functools import lru_cache

logger = logging.getLogger(__name__)


class InferSmilesDataset(BaseWrapperDataset):

    def __init__(self, dataset, infer_type, seed=42):
        self.dataset = dataset
        self.infer_type = infer_type
        self.seed = seed

    def get_infer_smiles(self, idx):
        dict_ = dict()
        if self.infer_type == 'retrosynthetic':
            dict_["smiles_mapnumber_target_list"] = self.dataset[idx].split('.')
            dict_["smiles_mapnumber_reactant_list"] = self.dataset[idx].split('.')
            dict_["smiles_mapnumber_precursor_list"] = ['']
            dict_["class"] = 0
        elif self.infer_type == 'synthetic':
            dict_["smiles_mapnumber_target_list"] = self.dataset[idx].split('.')
            dict_["smiles_mapnumber_reactant_list"] = self.dataset[idx].split('.')
            dict_["smiles_mapnumber_precursor_list"] = ['']
            dict_["class"] = 0
        elif self.infer_type == 'value_fn':         # RMK: modified here for different infer_type
            dict_["smiles"] = self.dataset[idx].split('.')
            dict_["value"] = 0
        else:
            print('It is illegal inference task. ')
            raise NotImplementedError
        return dict_

    def __len__(self):
        return len(self.dataset)

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        return self.get_infer_smiles(idx)
