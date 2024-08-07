from unicore.data import BaseWrapperDataset
from . import data_utils
from functools import lru_cache


class FixPadDataset(BaseWrapperDataset):
    def __init__(self, dataset, pad_idx, length_pad = 180):
        super().__init__(dataset)
        self.dataset = dataset
        self.pad_idx = pad_idx
        self.length_pad = length_pad

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        sample = self.dataset[idx]
        assert len(sample) <= self.length_pad
        sample_pad = sample.new(self.length_pad).fill_(self.pad_idx)
        sample_pad[:len(sample)] = sample
        return sample_pad
    