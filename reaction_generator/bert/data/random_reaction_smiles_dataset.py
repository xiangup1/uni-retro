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


class RandomReactionSmilesDataset(BaseWrapperDataset):
    # mainly from https://github.com/otori-bird/retrosynthesis/blob/main/preprocessing/get_R-SMILES.py
    def __init__(self, pro_dataset, rec_dataset, seed, prob=1.0, epoch_t = 0):
        self.rec_dataset = rec_dataset
        self.dataset = pro_dataset
        self.prob = prob
        self.seed = seed
        self.set_epoch(None)
        self.epoch = 0
        self.epoch_t = epoch_t

    def __len__(self):
        return len(self.dataset)

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch

    def get_epoch(self):
        return self.epoch

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        pro_smi, rec_smi, ro_map_num_smi, reactant_map_num_smi = self.get_rooted_alignment_smiles(self.dataset[idx], self.rec_dataset[idx])
        return pro_smi, rec_smi, ro_map_num_smi, reactant_map_num_smi 

    def get_rooted_alignment_smiles(self, product, reactant):
        product =  ".".join([i for i in product])
        reactant =  ".".join([i for i in reactant])
        pt = re.compile(r':(\d+)]')
        pro_mol = Chem.MolFromSmiles(product)
        rea_mol = Chem.MolFromSmiles(reactant)
        """checking data quality"""
        rids = sorted(re.findall(pt, reactant))
        pids = sorted(re.findall(pt, product))

        pro_atom_map_numbers = list(map(int, re.findall(r"(?<=:)\d+", product)))
        reactant = reactant.split(".")
        with data_utils.numpy_seed(self.seed, self.epoch):
            try:
                pro_root_atom_map = random.sample(pro_atom_map_numbers, 1)[0]
            except:
                pro_root_atom_map = -1
            pro_root = self.get_root_id(pro_mol, root_map_number=pro_root_atom_map)
            cano_atom_map = self.get_cano_map_number(product, root=pro_root)
        if cano_atom_map is None:
            # 返回未加处理的 smile
            return self.clear_map_canonical_smiles(product), ".".join([self.clear_map_canonical_smiles(i) for i in reactant]), self.get_smiles(product),  ".".join([self.get_smiles(i) for i in reactant])

        pro_smi = self.clear_map_canonical_smiles(product, canonical=True, root=pro_root)
        pro_map_num_smi = self.get_smiles(product, canonical=True, root=pro_root)
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
                    rea_map_num_smi = self.get_smiles(reactant[i], canonical=True, root=rea_root)
                    aligned_reactants.append(rea_smi)
                    aligned_map_num_reactants.append(rea_map_num_smi)
                    aligned_reactants_order.append(j)
                    used_indices.append(i)
                    break
        sorted_reactants = sorted(list(zip(aligned_reactants, aligned_reactants_order)), key=lambda x: x[1])
        aligned_reactants = [item[0] for item in sorted_reactants]
        reactant_smi = ".".join(aligned_reactants)

        sorted_map_num_reactants = sorted(list(zip(aligned_map_num_reactants, aligned_reactants_order)), key=lambda x: x[1])
        aligned_map_num_reactants = [item[0] for item in sorted_map_num_reactants]        
        reactant_map_num_smi = ".".join(aligned_map_num_reactants)
        return pro_smi, reactant_smi, pro_map_num_smi, reactant_map_num_smi

    def get_root_id(self, mol, root_map_number):
        root = -1
        for i, atom in enumerate(mol.GetAtoms()):
            if atom.GetAtomMapNum() == root_map_number:
                root = i
                break
        return root

    def get_cano_map_number(self, smi, root=-1):
        atommap_mol = Chem.MolFromSmiles(smi)
        canonical_mol = Chem.MolFromSmiles(self.clear_map_canonical_smiles(smi,root=root))
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

    def clear_map_canonical_smiles(self, smi, canonical=True, root=-1):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            for atom in mol.GetAtoms():
                if atom.HasProp('molAtomMapNumber'):
                    atom.ClearProp('molAtomMapNumber')
            return Chem.MolToSmiles(mol, isomericSmiles=True, rootedAtAtom=root, canonical=canonical) 
        else:
            return smi
    
    def get_smiles(self, smi, canonical=True, root=-1):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            return Chem.MolToSmiles(mol, isomericSmiles=True, rootedAtAtom=root, canonical=canonical)
        else:
            return smi        
