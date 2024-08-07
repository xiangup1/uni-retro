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
from .data_utils import smi_tokenizer, canonical_smiles_with_am, remove_am_without_canonical, canonical_smiles, rooted_smiles_with_am, clear_map_number_base


class RRandomReactionSmilesAugDataset(BaseWrapperDataset):
    # mainly from https://github.com/otori-bird/retrosynthesis/blob/main/preprocessing/get_R-SMILES.py
    def __init__(self, pro_dataset, rec_dataset, seed, times = 10, prob=1.0, epoch_t = 0):
        self.rec_dataset = rec_dataset
        self.dataset = pro_dataset
        self.prob = prob
        self.seed = seed
        self.set_epoch(None)
        self.epoch = 0
        self.epoch_t = epoch_t
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
        try:
            pro_smi, rec_smi, pro_map_num_smi, reactant_map_num_smi = self.get_rooted_alignment_smiles(self.dataset[idx], self.rec_dataset[idx])
        except:
            print('wz test wrong reactant and product: ', self.dataset[idx], self.rec_dataset[idx])
            pro_smi, rec_smi, pro_map_num_smi, reactant_map_num_smi = self.get_smiles_base_res(self.dataset[idx], self.rec_dataset[idx])
        return pro_smi, rec_smi, pro_map_num_smi, reactant_map_num_smi

    def get_smiles_base_res(self, product, reactant):
        product =  ".".join([i for i in product])
        reactant =  ".".join([i for i in reactant])
        # print('wz test smi pro rea: ', product, reactant)
        product = canonical_smiles_with_am(product)
        reactant = canonical_smiles_with_am(reactant)
        return clear_map_number_base(product), ".".join([clear_map_number_base(i) for i in reactant]), product,  reactant


    def get_rooted_alignment_smiles(self, product, reactant):
        product =  ".".join([i for i in product])
        reactant =  ".".join([i for i in reactant])
        # print('wz test smi pro rea: ', product, reactant)
        product = canonical_smiles_with_am(product)
        reactant = canonical_smiles_with_am(reactant)

        pt = re.compile(r':(\d+)]')
        """checking data quality"""
        rids = sorted(re.findall(pt, reactant))
        pids = sorted(re.findall(pt, product))

        pro_mol = Chem.MolFromSmiles(product)
        rea_mol = Chem.MolFromSmiles(reactant)
        pro_atom_map_numbers = list(map(int, re.findall(r"(?<=:)\d+", product)))
        reactant = reactant.split(".")
        with data_utils.numpy_seed(self.seed, self.epoch):
            try:
                pro_root_atom_map = random.sample(pro_atom_map_numbers, 1)[0]
            except:
                pro_root_atom_map = -1
            pro_root = self.get_root_id(pro_mol, root_map_number=pro_root_atom_map)
            cano_atom_map = self.get_cano_map_number(product, root=pro_root)

        if cano_atom_map is None or self.prob == 0.0:
            # 返回未加处理的 smile
            return clear_map_number_base(product), ".".join([clear_map_number_base(i) for i in reactant]), product,  reactant

        pro_smi = self.clear_map_canonical_smiles(product, root=pro_root, canonical=True)
        pro_map_num_smi = rooted_smiles_with_am(product, canonical=True, root=pro_root)
        aligned_reactants = []
        aligned_map_num_reactants = []
        aligned_reactants_order = []
        rea_atom_map_numbers = [list(map(int, re.findall(r"(?<=:)\d+", rea))) for rea in reactant]
        used_indices = []
        for i, rea_map_number in enumerate(rea_atom_map_numbers):
            for j, map_number in enumerate(cano_atom_map):
                if map_number in rea_map_number:
                    rea_root = self.get_root_id(Chem.MolFromSmiles(reactant[i]), root_map_number=map_number)
                    rea_smi = self.clear_map_canonical_smiles(reactant[i], canonical=True, root=rea_root)
                    rea_map_num_smi = rooted_smiles_with_am(reactant[i], canonical=True, root=rea_root)
                    aligned_reactants.append(rea_smi)
                    aligned_map_num_reactants.append(rea_map_num_smi)
                    aligned_reactants_order.append(j)
                    used_indices.append(i)
                    break
        sorted_reactants = sorted(list(zip(aligned_reactants, aligned_reactants_order)), key=lambda x: x[1])
        aligned_reactants = [item[0] for item in sorted_reactants]

        sorted_map_num_reactants = sorted(list(zip(aligned_map_num_reactants, aligned_reactants_order)), key=lambda x: x[1])
        reactant_smi = ".".join(aligned_reactants)

        aligned_map_num_reactants = [item[0] for item in sorted_map_num_reactants]        
        reactant_map_num_smi = ".".join(aligned_map_num_reactants)
        return pro_smi, reactant_smi, pro_map_num_smi, reactant_map_num_smi

    def get_root_id(self, mol, root_map_number):
        root = -1
        try:
            for i, atom in enumerate(mol.GetAtoms()):
                if atom.GetAtomMapNum() == root_map_number:
                    root = i
                    break
        except:
            logger.info('something error in get_root_id')
            pass
        return root

    def get_cano_map_number(self, smi, root=-1):
        atommap_mol = Chem.MolFromSmiles(smi)
        try:
            canonical_mol = Chem.MolFromSmiles(rooted_smiles_with_am(smi,root=root))
            cano2atommapIdx = atommap_mol.GetSubstructMatch(canonical_mol)
            correct_mapped = [canonical_mol.GetAtomWithIdx(i).GetSymbol() == atommap_mol.GetAtomWithIdx(index).GetSymbol() for i,index in enumerate(cano2atommapIdx)]
            atom_number = len(canonical_mol.GetAtoms())
            if np.sum(correct_mapped) < atom_number or len(cano2atommapIdx) < atom_number:
                cano2atommapIdx = [0] * atom_number
                atommap2canoIdx = canonical_mol.GetSubstructMatch(atommap_mol)
                if len(atommap2canoIdx) != atom_number:
                    return None
                for i, index in enumerate(atommap2canoIdx):
                    cano2atommapIdx[index] = i
            id2atommap = [atom.GetAtomMapNum() for atom in atommap_mol.GetAtoms()]
            return [id2atommap[cano2atommapIdx[i]] for i in range(atom_number)]
        except:
            logger.info('something error in get_cano_map_number')
            return None
    

    def clear_map_canonical_smiles(self, smi, canonical=True, root=-1):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            for atom in mol.GetAtoms():
                if atom.HasProp('molAtomMapNumber'):
                    atom.ClearProp('molAtomMapNumber')
            mol = Chem.RemoveHs(mol)
            return Chem.MolToSmiles(mol, isomericSmiles=True, rootedAtAtom=root, canonical=canonical)
        else:
            return smi

