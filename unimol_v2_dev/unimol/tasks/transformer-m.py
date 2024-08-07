# Copyright (c) DP Techonology, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os

import numpy as np
from unicore.data import (
    Dictionary,
    LMDBDataset,
    RawLabelDataset,
    RawArrayDataset,
    RawNumpyDataset,
    AppendTokenDataset,
    PrependTokenDataset,
    RightPadDataset,
    TokenizeDataset,
    RightPadDataset2D,
    FromNumpyDataset,
    NestedDictionaryDataset,
    EpochShuffleDataset,
)
from unimol.data import (
    KeyDataset,
    LMDBPCQDataset,
    DistanceDataset,
    EdgeTypeDataset,
    NormalizeDataset,
    CoordNoiseDataset,
    RightPadDatasetCoord,
    ConformerPCQSampleDataset,
    ConformerPCQTTASampleDataset,
    GraphFeatures,
    ShortestPathDataset,
    DegreeDataset,
    AtomFeatDataset,
    BondDataset,
    graph_features,
)
from unicore.tasks import UnicoreTask, register_task
from unicore import checkpoint_utils

logger = logging.getLogger(__name__)


@register_task("Transformer-M")
class TransformerMTask(UnicoreTask):
    """Task for training transformer auto-encoder models."""

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument("data", help="downstream data path")
        parser.add_argument(
            "--noise-scale",
            default=0.2,
            type=float,
            help="coordinate noise for masked atoms",
        )

    def __init__(self, args):
        super().__init__(args)
        self.seed = args.seed

    @classmethod
    def setup_task(cls, args, **kwargs):
        return cls(args)

    def load_dataset(self, split, **kwargs):
        """Load a given dataset split.
        Args:
            split (str): name of the data scoure (e.g., bppp)
        """
        split_path = os.path.join(self.args.data, split + ".lmdb")
        dataset = LMDBPCQDataset(split_path)
        if split == "train":
            sample_dataset = ConformerPCQSampleDataset(
                dataset,
                self.seed,
                "atoms",
                "input_pos",
                "label_pos",
                "target",
                "id",
            )
        else:
            sample_dataset = ConformerPCQTTASampleDataset(
                dataset,
                self.seed,
                "atoms",
                "input_pos",
                "label_pos",
                "target",
                "id",
                num_replica=6,
            )
        sample_dataset = NormalizeDataset(sample_dataset, "coordinates")
        sample_dataset = NormalizeDataset(sample_dataset, "target_coordinates")

        tgt_dataset = KeyDataset(sample_dataset, "target")
        # id_dataset = KeyDataset(sample_dataset, "id")
        # src_dataset = KeyDataset(sample_dataset, "atoms")
        # src_dataset = TokenizeDataset(
        #     src_dataset, self.dictionary, max_seq_len=self.args.max_seq_len
        # )
        raw_coord_dataset = KeyDataset(sample_dataset, "coordinates")
        tgt_coord_dataset = KeyDataset(sample_dataset, "target_coordinates")

        graph_features = GraphFeatures(
            sample_dataset,
            tgt_coord_dataset if split in ["train", "valid_our"] else None,
        )

        # src_coord_noise_dataset = CoordNoiseDataset(
        #     raw_coord_dataset,
        #     tgt_coord_dataset,
        #     self.args.coord_gen_prob,
        #     1.0,
        #     src_noise=self.args.src_noise,
        #     tgt_noise=self.args.tgt_noise,
        #     seed=self.seed + 18,
        # )

        # def PrependAndAppend(dataset, pre_token, app_token):
        #     dataset = PrependTokenDataset(dataset, pre_token)
        #     return AppendTokenDataset(dataset, app_token)

        # src_coord_noise_dataset = KeyDataset(src_coord_noise_dataset, "coordinates")
        # src_coord_noise_dataset = RawNumpyDataset(src_coord_noise_dataset)
        # src_coord_noise_dataset = PrependAndAppend(src_coord_noise_dataset, 0.0, 0.0)

        # tgt_coord_dataset = RawNumpyDataset(tgt_coord_dataset)
        # tgt_coord_dataset = PrependAndAppend(tgt_coord_dataset, 0.0, 0.0)
        # tgt_distance_dataset = DistanceDataset(tgt_coord_dataset)

        # src_token_dataset = PrependAndAppend(
        #     src_dataset, self.dictionary.bos(), self.dictionary.eos()
        # )
        # tgt_token_dataset = PrependAndAppend(
        #     src_dataset, self.dictionary.pad(), self.dictionary.pad()
        # )
        # edge_type = EdgeTypeDataset(src_dataset, len(self.dictionary))

        nest_dataset = NestedDictionaryDataset(
            {
                "net_input": {
                    # "src_tokens": RightPadDataset(
                    #     src_dataset,
                    #     pad_idx=self.dictionary.pad(),
                    # ),
                    # "src_coord": RightPadDatasetCoord(
                    #     src_coord_noise_dataset,
                    #     pad_idx=0,
                    # ),
                    # "src_edge_type": RightPadDataset2D(
                    #     edge_type,
                    #     pad_idx=0,
                    # ),
                    "batched_data": graph_features,
                },
                "target": {
                    # "tokens_target": RightPadDataset(
                    #     tgt_dataset,
                    #     pad_idx=self.dictionary.pad(),
                    # ),
                    # "coord_target": RightPadDatasetCoord(
                    #     tgt_coord_dataset,
                    #     pad_idx=0,
                    # ),
                    # "distance_target": RightPadDataset2D(
                    #     tgt_distance_dataset,
                    #     pad_idx=0,
                    # ),
                    "gap_target": RawLabelDataset(
                        tgt_dataset,
                    ),
                },
                # "id": id_dataset,
            },
        )
        if split in ["train", "train.small"]:
            nest_dataset = EpochShuffleDataset(
                nest_dataset, len(nest_dataset), self.seed
            )
        self.datasets[split] = nest_dataset

    def build_model(self, args):
        from unicore import models

        model = models.build_model(args, self)
        return model
