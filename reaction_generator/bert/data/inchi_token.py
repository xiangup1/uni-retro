# Copyright (c) DP Techonology, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from unittest.mock import NonCallableMagicMock
from functools import lru_cache
from unicore.data import BaseWrapperDataset
from rdkit import Chem
import random
import re
import logging
logger = logging.getLogger(__name__)


class InchiTokenizerDataset(BaseWrapperDataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch

    def split_form(self, form):
        string = ''
        for i in re.findall(r"[A-Z][^A-Z]*", form):
            elem = re.match(r"\D+", i).group()
            num = i.replace(elem, "")
            if num == "":
                string += f"{elem} "
            else:
                string += f"{elem} {str(num)} "
        return string.rstrip(' ')

    def split_form2(self, form):
        string = ''
        for i in re.findall(r"[a-z][^a-z]*", form):
            elem = i[0]
            num = i.replace(elem, "").replace('/', "")
            num_string = ''
            for j in re.findall(r"[0-9]+[^0-9]*", num):
                num_list = list(re.findall(r'\d+', j))
                assert len(num_list) == 1, f"len(num_list) != 1"
                _num = num_list[0]
                if j == _num:
                    num_string += f"{_num} "
                else:
                    extra = j.replace(_num, "")
                    num_string += f"{_num} {' '.join(list(extra))} "
            string += f"/{elem} {num_string}"
        return string.rstrip(' ')

    def get_inchi_token(self, inchi_key):
        test_inchi_t = inchi_key.split('/')[1]
        test_inchi_t2 = self.split_form(test_inchi_t) + ' ' + self.split_form2('/'.join(inchi_key.split('/')[2:]))
        return test_inchi_t2.split(' ')       

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        return self.get_inchi_token(self.dataset[idx])
