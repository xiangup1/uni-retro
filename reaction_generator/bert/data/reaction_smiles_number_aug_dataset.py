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


class ReactionSmilesNumberAugDataset(BaseWrapperDataset):

    def __init__(self, pro_dataset, rec_dataset, seed, times = 10, prob=1.0, epoch_t = 0, aug_strategy = 'number'):
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
        pro_smi, rec_smi, pro_map_num_smi, reactant_map_num_smi = self.get_number_alignment_smiles(self.dataset[idx], self.rec_dataset[idx])
        return pro_smi, rec_smi, pro_map_num_smi, reactant_map_num_smi

    def get_number_alignment_smiles(self, product, reactant):

        with data_utils.numpy_seed(self.seed, self.epoch):
                self.sample_p = random.random()

        product =  ".".join([i for i in product])
        reactant =  ".".join([i for i in reactant])

        product = self.canonical_smiles_with_am_zero(product)
        reactant = self.canonical_smiles_with_am_zero(reactant)
        try:
            # shuffle reactant
            if self.aug_strategy == 'root_aug':
                rea_id_list_merge, rea_max_map_num = self.get_id_list_root(reactant)
            elif self.aug_strategy == 'fragment_index_aug':
                rea_id_list_merge, rea_max_map_num = self.get_id_list_sample(reactant)                
            else:
                rea_id_list_merge, rea_max_map_num = self.get_id_list_sample(reactant)

            pro_id_list_merge, pro_max_map_num = self.get_id_list_sample(product)
            
            if rea_max_map_num != pro_max_map_num:
                # print('provide wrong map num reaction: ', reactant, product)
                rea_max_map_num = min(rea_max_map_num, pro_max_map_num)
                pro_max_map_num = rea_max_map_num         

            # sync shuffle product
            pro_id_dict = self.product_map_num_index_dict(pro_id_list_merge)
            rea_list, pro_list = self.sync_product(rea_id_list_merge, pro_id_list_merge, rea_max_map_num, pro_id_dict)
            rea_id_list = self.split_id_list(rea_list)
            pro_id_list = self.split_id_list(pro_list)
            
            new_product = self.renumberSmile(product, pro_id_list)
            new_reactant = self.renumberSmile(reactant, rea_id_list)
            new_product_smi = self.clear_map_canonical_smiles(new_product)
            new_reactant_smi = self.clear_map_canonical_smiles(new_reactant)

        except:
            new_product =  ".".join([i for i in product])
            new_reactant =  ".".join([i for i in reactant]) 
            print('wrong product: ', new_product)   
            print('wrong reactant: ', new_reactant)   
            except_product = ['[CH3:1][CH2:2][CH2:3][N:4]1[CH2:5][CH2:6][O:7][CH:8]([c:10]2[cH:11][cH:12][c:13]([Cl:17])[c:14]([OH:16])[cH:15]2)[CH2:9]1']
            except_reactant = ['C[O:16][c:14]1[c:13]([Cl:17])[cH:12][cH:11][c:10]([CH:8]2[O:7][CH2:6][CH2:5][N:4]([CH2:3][CH2:2][CH3:1])[CH2:9]2)[cH:15]1']
            new_product =  ".".join([i for i in except_product])
            new_reactant =  ".".join([i for i in except_reactant]) 
            new_product_smi = self.clear_map_canonical_smiles(new_product)
            new_reactant_smi = self.clear_map_canonical_smiles(new_reactant) 
            print('s product: ', new_product, new_product_smi)   
            print('s reactant: ', new_reactant, new_reactant_smi)   
            print('s new_smi_length: ', len(new_reactant_smi), len(new_product_smi)) 
                               
        return new_product_smi, new_reactant_smi, new_product, new_reactant

    def get_smile_atom_num_flag(self, smi):
        mol = Chem.MolFromSmiles(smi)
        atom_map_num_flag = False
        if mol is not None:
            for atom in mol.GetAtoms():
                if atom.HasProp('molAtomMapNumber'):
                    atom_map_num_flag = True
        return atom_map_num_flag

    def get_id_list_root(self, smi):
        smi = self.m_root_aug_smile(smi)
        c_mol = Chem.MolFromSmiles(smi)
        c_id_list = [(atom.GetAtomMapNum(), atom.GetIdx()) for atom in c_mol.GetAtoms()] 
        c_max_atom_num = max(c_id_list, key= lambda x: x[0])
        c_max_atom_num = c_max_atom_num[0]
        max_map_num = c_max_atom_num
        smi_l = list(smi.split('.'))
        c_id_list_merge = []
        id_start = 0
        for it in smi_l:
            sub_id_list, max_atom_num, id_start = self.label_smi(it, c_max_atom_num, max_map_num, id_start)
            c_max_atom_num = max_atom_num
            c_id_list_merge.append(sub_id_list)

        # list shuffle
        if self.prob != 0 and self.sample_p < self.prob:
            # self.epoch_t > 0 and self.epoch < self.epoch_t
            random.shuffle(c_id_list_merge)

        new_c_id_list_merge = []
        for it in c_id_list_merge:
            new_c_id_list_merge += it 
        return new_c_id_list_merge, max_map_num

    def m_root_aug_smile(self, smi):
        res = []
        smi_list = smi.split('.')
        for it in smi_list:
            res.append(self.root_aug_smile(it))
        res_smi = '.'.join(res)
        return res_smi

    def root_aug_smile(self, smi):

        pro_atom_map_numbers = list(map(int, re.findall(r"(?<=:)[0-9]*[1-9][0-9]*", smi)))
        c_mol = Chem.MolFromSmiles(smi)
        with data_utils.numpy_seed(self.seed, self.epoch):
            try:
                pro_root_atom_map = random.sample(pro_atom_map_numbers, 1)[0]
            except:
                pro_root_atom_map = -1
        smi_root  = self.get_root_id(c_mol, root_map_number=pro_root_atom_map)
        aug_smi = Chem.MolToSmiles(c_mol, rootedAtAtom=int(smi_root), canonical=True)
        return aug_smi

    def get_root_id(self, mol, root_map_number):
        root = -1
        for i, atom in enumerate(mol.GetAtoms()):
            if atom.GetAtomMapNum() == root_map_number:
                root = i
                break
        return root


    def get_id_list_sample(self, smi):
        c_mol = Chem.MolFromSmiles(smi)
        c_id_list = [(atom.GetAtomMapNum(), atom.GetIdx()) for atom in c_mol.GetAtoms()] 
        c_max_atom_num = max(c_id_list, key= lambda x: x[0])
        c_max_atom_num = c_max_atom_num[0]
        max_map_num = c_max_atom_num
        smi_l = list(smi.split('.'))
        c_id_list_merge = []
        id_start = 0
        for it in smi_l:
            sub_id_list, max_atom_num, id_start = self.label_shuffle_smi(it, c_max_atom_num, max_map_num, id_start)
            c_max_atom_num = max_atom_num
            c_id_list_merge.append(sub_id_list)

        # list shuffle
        if self.prob != 0 and self.sample_p < self.prob:
            # self.epoch_t > 0 and self.epoch < self.epoch_t
            random.shuffle(c_id_list_merge)

        new_c_id_list_merge = []
        for it in c_id_list_merge:
            new_c_id_list_merge += it 
        return new_c_id_list_merge, max_map_num


    def sync_product(self, rea_list, pro_list, rea_max_map_num, pro_id_dict):
        id_p = 0
        # try:
        for i, item in enumerate(rea_list):
            if item[0] <= rea_max_map_num and id_p < len(pro_list):
                if pro_list[id_p][0] > rea_max_map_num:
                    id_p += 1
                else:
                    pro_atom_now_item = pro_list[id_p]
                    match_item_id = pro_id_dict[item[0]]
                    pro_atom_match_item = pro_list[match_item_id]
                    if pro_atom_now_item != pro_atom_match_item:
                        pro_list[id_p], pro_list[match_item_id] = self.swap_pair(pro_atom_now_item, pro_atom_match_item)
                        pro_id_dict = self.product_map_num_index_dict(pro_list)
                    id_p += 1
        # except:
        #     print('wrong list: ', rea_list)
        #     print('wrong list: ', pro_list)
        #     pass
        return rea_list, pro_list

    def label_smi(self, smi, max_atom_num, max_map_num, id_start = 0):
        r_mol = Chem.MolFromSmiles(smi)
        id_list = [(atom.GetAtomMapNum(), atom.GetIdx() + id_start) for atom in r_mol.GetAtoms()]  
        for i, item in enumerate(id_list):
            if item[0] == 0:
                id_list[i] = (max_atom_num + 1, item[1])
                max_atom_num = max_atom_num + 1

        id_start += len(id_list)
        return id_list, max_atom_num, id_start

    def label_shuffle_smi(self, smi, max_atom_num, max_map_num, id_start = 0):
        r_mol = Chem.MolFromSmiles(smi)
        id_list = [(atom.GetAtomMapNum(), atom.GetIdx() + id_start) for atom in r_mol.GetAtoms()]  
        for i, item in enumerate(id_list):
            if item[0] == 0:
                id_list[i] = (max_atom_num + 1, item[1])
                max_atom_num = max_atom_num + 1
        if self.aug_strategy == 'fragment_index_aug':
            id_list = self.frag_shuffle_smi(id_list, max_map_num)           
        else:
            id_list = self.shuffle_smi(id_list, max_map_num)
        id_start += len(id_list)
        return id_list, max_atom_num, id_start

    def frag_shuffle_smi(self, smi_id_list, max_map_num):

        # shuffle smile
        if self.prob != 0 and self.sample_p < self.prob:
            smi_id_list = self.multi_point_swap_list(smi_id_list, 3)
        smi_id_list = self.rerank_num_list(smi_id_list, max_map_num)
        return smi_id_list

    def multi_point_swap_list(self, smi_id_list, times = 2, delta = 3):
        ### 搞个类似于策略，前几个 id 不动，后面的 id 才做 shuflle， 后面 shuffle 也不弄的太夸张，只转大块的片段。
        try:
            smi_id_list_left = smi_id_list[:delta]
            smi_id_list_right = smi_id_list[delta:]
            for i in range(times):
                smi_id_list_right = self.shuffle_frag(smi_id_list_right)
            smi_id_list = smi_id_list_left + smi_id_list_right
        except:
            smi_id_list = smi_id_list
        return smi_id_list

    def shuffle_frag(self, id_mol_list):
        shuffle_index = [i for i in range(len(id_mol_list))]
        shuflle_point = random.sample(shuffle_index, 1)[0]
        with data_utils.numpy_seed(self.seed, self.epoch+100):
            sample_p = random.random()
        if sample_p > 0.5:
            shuffle_index_left = shuffle_index[:shuflle_point]
        else:
            shuffle_index_left = shuffle_index[:shuflle_point][::-1]
            
        shuffle_index_right = shuffle_index[shuflle_point:]
        shuffle_index_new = shuffle_index_right+shuffle_index_left
        id_mol_list_new = np.array(id_mol_list)[shuffle_index_new]
        return id_mol_list_new.tolist()


    def shuffle_smi(self, smi_id_list, max_map_num):

        # shuffle smile
        if self.prob != 0 and self.sample_p < self.prob:
            np.random.shuffle(smi_id_list)
        smi_id_list = self.rerank_num_list(smi_id_list, max_map_num)
        return smi_id_list

    def rerank_num_list(self, smi_id_list, max_map_num):
        i, j = 0, len(smi_id_list) - 1
        for i, item in enumerate(smi_id_list):
            if(item[0] > max_map_num):
                if i>=j:
                    break
                while(j >= i and smi_id_list[j][0] > max_map_num):
                    j -= 1
                smi_id_list[i], smi_id_list[j] = smi_id_list[j], smi_id_list[i]
                j -= 1
        return smi_id_list

    def swap_pair(self, a, b):
        return b, a

    def renumberSmile(self, smiles, id_list):
        m = Chem.MolFromSmiles(smiles)
        nm = Chem.RenumberAtoms(m,id_list)
        return Chem.MolToSmiles(nm, canonical=False)
    
    def split_id_list(self, tuple_list):
        res_list = [x[1] for x in tuple_list]
        return res_list

    def product_map_num_index_dict(self, pro_id_list_merge):
        pro_dict = {}
        for i, item in enumerate(pro_id_list_merge):
            pro_dict[item[0]] = i
        return pro_dict

    def clear_map_canonical_smiles(self, smi, canonical=False):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            for atom in mol.GetAtoms():
                if atom.HasProp('molAtomMapNumber'):
                    atom.ClearProp('molAtomMapNumber')
            mol = Chem.RemoveHs(mol)
            return Chem.MolToSmiles(mol, isomericSmiles=True, canonical=canonical)
        else:
            return smi

    def canonical_smiles_with_am_zero(self, smi):
        mol = Chem.MolFromSmiles(smi)
        for atom in mol.GetAtoms():
            if not atom.HasProp('molAtomMapNumber'):
                atom.SetIntProp('molAtomMapNumber', 0)
        return Chem.MolToSmiles(mol, canonical=False) 

    def add_exception_data(self, product, reactant):
        new_product =  ".".join([i for i in product])
        new_reactant =  ".".join([i for i in reactant]) 
        print('wrong product: ', new_product)   
        print('wrong reactant: ', new_reactant)   
        except_product = ['[CH3:1][CH2:2][CH2:3][N:4]1[CH2:5][CH2:6][O:7][CH:8]([c:10]2[cH:11][cH:12][c:13]([Cl:17])[c:14]([OH:16])[cH:15]2)[CH2:9]1']
        except_reactant = ['C[O:16][c:14]1[c:13]([Cl:17])[cH:12][cH:11][c:10]([CH:8]2[O:7][CH2:6][CH2:5][N:4]([CH2:3][CH2:2][CH3:1])[CH2:9]2)[cH:15]1']
        new_product =  ".".join([i for i in except_product])
        new_reactant =  ".".join([i for i in except_reactant]) 
        new_product_smi = self.clear_map_canonical_smiles(new_product)
        new_reactant_smi = self.clear_map_canonical_smiles(new_reactant) 
        print('s product: ', new_product, new_product_smi)   
        print('s reactant: ', new_reactant, new_reactant_smi)   
        print('s new_smi_length: ', len(new_reactant_smi), len(new_product_smi))                    
        return new_product_smi, new_reactant_smi, new_product, new_reactant

