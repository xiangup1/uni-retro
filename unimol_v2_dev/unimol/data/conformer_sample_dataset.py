# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from functools import lru_cache
from unicore.data import BaseWrapperDataset
from . import data_utils


class ConformerPCQSampleDataset(BaseWrapperDataset):
    def __init__(
        self,
        dataset,
        seed,
        atoms,
        coordinates,
        target_coordinates,
        target="target",
        id="id",
    ):
        self.dataset = dataset
        self.seed = seed
        self.atoms = atoms
        self.coordinates = coordinates
        self.target_coordinates = target_coordinates
        self.target = target
        self.id = id
        self.set_epoch(None)

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch

    @lru_cache(maxsize=16)
    def __cached_item__(self, index: int, epoch: int):
        data = self.dataset[index]
        atoms = data[self.atoms]
        id = data[self.id]
        assert len(atoms) > 0
        size = len(data[self.coordinates])
        with data_utils.numpy_seed(self.seed, epoch, index):
            sample_idx = np.random.randint(size)
        coordinates = data[self.coordinates][sample_idx]
        if isinstance(data[self.target_coordinates], list):
            target_coordinates = data[self.target_coordinates][-1]
        else:
            target_coordinates = data[self.target_coordinates]
        assert len(atoms) == len(coordinates)
        assert len(atoms) == len(target_coordinates)
        target = data[self.target]
        return {
            "atoms": np.array(atoms),
            "coordinates": np.array(coordinates),
            "target_coordinates": np.array(target_coordinates),
            "target": target,
            "id": id,
            "node_attr": data["node_attr"],
            "edge_index": data["edge_index"],
            "edge_attr": data["edge_attr"],
        }

    def __getitem__(self, index: int):
        return self.__cached_item__(index, self.epoch)


class ConformerPCQTTASampleDataset(BaseWrapperDataset):
    def __init__(
        self,
        dataset,
        seed,
        atoms,
        coordinates,
        target_coordinates,
        target="target",
        id="id",
        num_replica=6,
    ):
        self.dataset = dataset
        self.seed = seed
        self.atoms = atoms
        self.coordinates = coordinates
        self.target_coordinates = target_coordinates
        self.target = target
        self.id = id
        self.num_replica = num_replica
        self._init_idx()
        self.set_epoch(None)

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch

    def _init_idx(self):
        self.idx2key = {}
        cnt = 0
        for i in range(len(self.dataset)):
            size = len(self.dataset[i][self.coordinates])
            assert size == 1 or size == 2
            for _ in range(self.num_replica // size):
                for j in range(size):
                    self.idx2key[cnt] = (i, j)
                    cnt += 1
        self.cnt = cnt

    def __len__(self):
        return self.cnt

    @lru_cache(maxsize=16)
    def __cached_item__(self, index: int, epoch: int):
        key_idx, conf_idx = self.idx2key[index]
        data = self.dataset[key_idx]
        atoms = data[self.atoms]
        id = data[self.id]
        coordinates = data[self.coordinates][conf_idx]
        if isinstance(data[self.target_coordinates], list):
            target_coordinates = data[self.target_coordinates][-1]
        else:
            target_coordinates = data[self.target_coordinates]
        assert len(atoms) == len(coordinates)
        assert len(atoms) == len(target_coordinates)
        target = data[self.target]

        return {
            "atoms": np.array(atoms),
            "coordinates": np.array(coordinates),
            "target_coordinates": np.array(target_coordinates),
            "target": target,
            "id": id,
            "node_attr": data["node_attr"],
            "edge_index": data["edge_index"],
            "edge_attr": data["edge_attr"],
        }

    def __getitem__(self, index: int):
        return self.__cached_item__(index, self.epoch)
