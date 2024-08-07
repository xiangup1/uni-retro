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
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


class ReactionSmilesSyncNumberAugDataset(BaseWrapperDataset):

    def __init__(self, pro_dataset, rec_dataset, seed, times = 10, prob=1.0, epoch_t = 0, aug_strategy = 'root_aug'):
        self.rec_dataset = rec_dataset
        self.dataset = pro_dataset
        self.prob = prob
        self.seed = seed
        self.set_epoch(None)
        self.epoch = 0
        self.epoch_t = epoch_t
        self.times = times
        self.sample_p = 0.0
        self.aug_strategy = aug_strategy

    def __len__(self):
        return len(self.dataset)

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch

    def get_epoch(self):
        return self.epoch

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        pro_smi, rec_smi, pro_map_num_smi, reactant_map_num_smi = self.align_smiles(self.dataset[idx], self.rec_dataset[idx])
        return pro_smi, rec_smi, pro_map_num_smi, reactant_map_num_smi

    def align_smiles(self, nproduct, nreactant):

        with data_utils.numpy_seed(self.seed, self.epoch):
                self.sample_p = random.random()
        try:

            if isinstance(nproduct, str):
                nproduct = [nproduct]

            random.shuffle(nreactant)
            product =  ".".join([i for i in nproduct])
            reactant =  ".".join([i for i in nreactant]) 

            product = self.canonical_smiles_with_am_zero(product)
            reactant = self.canonical_smiles_with_am_zero(reactant)

            if self.prob != 0 and self.sample_p >= self.prob:
                new_product_smi_clear = self.clear_map_canonical_smiles(product)
                new_reactant_smi_clear = self.clear_map_canonical_smiles(reactant)
                return new_product_smi_clear, new_reactant_smi_clear, product, reactant

            if self.prob == 0:
                new_product_smi_clear = self.clear_map_canonical_smiles(product)
                new_reactant_smi_clear = self.clear_map_canonical_smiles(reactant)
                # print('wz test new_product_smi_clear: ', new_product_smi_clear, new_reactant_smi_clear)
                return new_product_smi_clear, new_reactant_smi_clear, product, reactant
                
            ### 根据 reactant的划分 把 product 的 atom 分成两个部分，两部分分别 random, 然后拼起来, 未标注的原子就放到最后,生成 smiles;
            rea_id_list = self.get_id_list(reactant)
            pro_id_list = self.get_id_list(product)
            total_p_id_list_res = self.split_product_id_list(pro_id_list, rea_id_list)
            new_product_res = []
            for s_p_id_list_res in total_p_id_list_res:
                pro_id_list = [item for sublist in s_p_id_list_res for item in sublist]
                pro_id_list = self.split_id_list(pro_id_list)
                new_product = self.renumberSmile(product, pro_id_list)
                new_product_res.append(new_product)

            new_product_smi = '.'.join(new_product_res)
            new_product_l = self.get_id_list(new_product_smi)
            new_product_l = [item for sublist in new_product_l for item in sublist]
            ### product 生成之后再取出来，然后根据原子顺序修改 reactant 的原子顺序，然后生成 smiles;
            new_reactant_l = self.sync_total_reactant_id(new_product_l, rea_id_list)

            new_reactant_res = []
            reactant_list = reactant.split('.')
            for i1, sub_rea_id_list in enumerate(new_reactant_l):
                sub_rea_id_list = self.split_id_list(sub_rea_id_list)
                new_reactant = self.renumberSmile(reactant_list[i1], sub_rea_id_list)
                new_reactant_res.append(new_reactant)

            ### A + B or B + A ?       
            new_reactant_smi = '.'.join(new_reactant_res)
            new_product_smi_clear = self.clear_map_canonical_smiles(new_product_smi)
            new_reactant_smi_clear = self.clear_map_canonical_smiles(new_reactant_smi)
            # print('wz test new_product_smi_clear: ', new_product_smi_clear, new_reactant_smi_clear, new_product_smi, new_reactant_smi)
            return new_product_smi_clear, new_reactant_smi_clear, new_product_smi, new_reactant_smi

        except:
            print('wrong product: ', nproduct)   
            print('wrong reactant: ', nreactant)   
            except_product = ['[CH3:1][CH2:2][CH2:3][N:4]1[CH2:5][CH2:6][O:7][CH:8]([c:10]2[cH:11][cH:12][c:13]([Cl:17])[c:14]([OH:16])[cH:15]2)[CH2:9]1']
            except_reactant = ['C[O:16][c:14]1[c:13]([Cl:17])[cH:12][cH:11][c:10]([CH:8]2[O:7][CH2:6][CH2:5][N:4]([CH2:3][CH2:2][CH3:1])[CH2:9]2)[cH:15]1']
            new_product =  ".".join([i for i in except_product])
            new_reactant =  ".".join([i for i in except_reactant]) 
            new_product_smi = self.clear_map_canonical_smiles(new_product)
            new_reactant_smi = self.clear_map_canonical_smiles(new_reactant) 
            return new_product_smi, new_reactant_smi, new_product, new_reactant


    def sync_total_reactant_id(self, pro_id_list, reactant_id_list):
        new_reactant_id_list = []
        for i, single_rea_id_list in enumerate(reactant_id_list):
            tmp_rea_id_list = self.sync_single_reactant_id(pro_id_list, single_rea_id_list)
            new_reactant_id_list.append(tmp_rea_id_list)
        return new_reactant_id_list
            
    def sync_single_reactant_id(self, pro_id_list, single_rea_id_list):
        tmp_rea_id_list = []
        single_rea_map_id_list = [it[0] for it in single_rea_id_list]
        for j, item in enumerate(pro_id_list):
            if item[0] in single_rea_map_id_list and item[0] > 0:
                item_index = single_rea_map_id_list.index(item[0])
                tmp_rea_id_list.append(single_rea_id_list[item_index])
        for k, itemk in enumerate(single_rea_id_list): 
            if itemk not in tmp_rea_id_list:
                tmp_rea_id_list.append(itemk)
        return tmp_rea_id_list

    def renumberSmile(self, smiles, id_list):
        m = Chem.MolFromSmiles(smiles)
        nm = Chem.RenumberAtoms(m,id_list)
        return Chem.MolToSmiles(nm, canonical=False)


    def split_id_list(self, tuple_list):
        res_list = [x[1] for x in tuple_list]
        return res_list
        
    def get_id_list(self, merge_smi):
        smi_list = merge_smi.split('.')
        total_smi_list = []
        for smi in smi_list:
            c_mol = Chem.MolFromSmiles(smi)
            c_id_list = [(atom.GetAtomMapNum(), atom.GetIdx()) for atom in c_mol.GetAtoms()] 
            total_smi_list.append(c_id_list)
        return total_smi_list


    def split_product_id_list(self, product_id_list, reactant_id_list):
        total_p_id_list_res = []
        for p_id_list in product_id_list:
            single_p_list = self.split_single_product_list(p_id_list, reactant_id_list)
            total_p_id_list_res.append(single_p_list)
        return total_p_id_list_res

    def split_single_product_list(self, single_product_id_list, reactant_id_list):
        single_p_list = [[] for i in range(len(reactant_id_list)+1)]
        for k, it_pair in enumerate(single_product_id_list):
            al_flag = False
            if it_pair[0] == 0:
                single_p_list[len(reactant_id_list)].append(it_pair) 
                continue          
            for j, rea_pair in enumerate(reactant_id_list):
                rea_pair_map_num = [rep[0] for rep in rea_pair]
                if it_pair[0] in rea_pair_map_num:
                    single_p_list[j].append(it_pair)
                    al_flag = True
            if not al_flag:
                single_p_list[len(reactant_id_list)].append(it_pair)
        if len(single_p_list[-1]) <= 0:
            single_p_list_m = single_p_list[:-1]
        else:
            single_p_list_m = single_p_list
        for i, sub_list in enumerate(single_p_list_m):
            single_p_list_m[i] = self.shuffle_smi(sub_list)
        return single_p_list_m

    def shuffle_smi(self, smi_id_list):
        np.random.shuffle(smi_id_list)
        return smi_id_list

    def clear_map_canonical_smiles(self, smi, canonical=False, root=-1):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            for atom in mol.GetAtoms():
                if atom.HasProp('molAtomMapNumber'):
                    atom.ClearProp('molAtomMapNumber')
            mol = Chem.RemoveHs(mol)
            return Chem.MolToSmiles(mol, isomericSmiles=True, rootedAtAtom=root, canonical=canonical)
        else:
            return smi

    def canonical_smiles_with_am_zero(self, smi):
        mol = Chem.MolFromSmiles(smi)
        for atom in mol.GetAtoms():
            if not atom.HasProp('molAtomMapNumber'):
                atom.SetIntProp('molAtomMapNumber', 0)
        return Chem.MolToSmiles(mol, canonical=False) 