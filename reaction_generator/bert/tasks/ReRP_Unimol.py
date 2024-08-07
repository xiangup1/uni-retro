import logging
import os
from unicore.data import (
    # Dictionary,
    MaskTokensDataset,
    NestedDictionaryDataset,
    NumelDataset,
    NumSamplesDataset,
    LMDBDataset,
    AppendTokenDataset,
    PrependTokenDataset,
    RightPadDataset,
    EpochShuffleDataset,
    RawLabelDataset,
    TokenizeDataset,
    data_utils,
    RightPadDataset2D,
    SortDataset,
    FromNumpyDataset,
    RawArrayDataset

)

from unimol.data import (
    KeyDataset,
    #ConformerSampleDataset,
    DistanceDataset,
    EdgeTypeDataset,
    MaskPointsDataset,
    RemoveHydrogenDataset,
    AtomTypeDataset,
    NormalizeDataset,
    CroppingDataset,
    RightPadDatasetCoord,
    #Add2DConformerDataset,
    #LMDBDataset,
)
#from unimol.data.tta_dataset import TTADataset
from ..data import Dictionary
from unicore import checkpoint_utils

from unicore.tasks import register_task
from .ReRP import ReRPTask

logger = logging.getLogger(__name__)


def freeze_network(network):
    for p in network.parameters():
        p.requires_grad = False


@register_task("ReRP_Unimol")
class ReRPTask_Unimol(ReRPTask):
    """Task for training transformer auto-encoder models."""

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        ReRPTask.add_other_args(parser)
        ReRPTask_Unimol.add_unimol_encoder_args(parser)

    @staticmethod
    def add_unimol_encoder_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument(
            "--encoder-dict-name",
            default="encoder_dict.txt",
            help="dictionary file",
        )
        parser.add_argument(
            "--conf-size",
            default=10,
            type=int,
            help="number of conformers generated with each molecule",
        )
        parser.add_argument(
            "--remove-hydrogen",
            action="store_true",
            help="remove hydrogen atoms",
        )
        parser.add_argument(
            "--remove-polar-hydrogen",
            action="store_true",
            help="remove polar hydrogen atoms",
        )
        parser.add_argument(
            "--max-atoms",
            type=int,
            default=256,
            help="selected maximum number of atoms in a molecule",
        )
        parser.add_argument(
            "--only-polar",
            default=1,
            type=int,
            help="1: only polar hydrogen ; -1: all hydrogen ; 0: remove all hydrogen ",
        )

        parser.add_argument(
            "--unimol-path",
            default="None",
            help="unimol-path",
        )

        parser.add_argument(
            "--freeze-unimol",
            action="store_true",
            help="freeze-unimol",
        )

    def __init__(self, args, dictionary, encoder_dictionary):
        super().__init__(args, dictionary)
        self.encoder_dictionary = encoder_dictionary
        self.encoder_mask_idx = encoder_dictionary.add_symbol(
            "[MASK]", is_special=True)
        if self.args.only_polar > 0:
            self.args.remove_polar_hydrogen = True
        elif args.only_polar < 0:
            self.args.remove_polar_hydrogen = False
        else:
            self.args.remove_hydrogen = True

    @classmethod
    def setup_task(cls, args, **kwargs):
        dictionary = Dictionary.load(os.path.join(args.data, args.dict_name))
        logger.info("dictionary: {} types".format(len(dictionary)))
        encoder_dictionary = Dictionary.load(
            os.path.join(args.data, args.encoder_dict_name))
        logger.info("encoder_dictionary: {} types".format(
            len(encoder_dictionary)))
        return cls(args, dictionary, encoder_dictionary)

    def one_dataset(self, raw_dataset, coord_seed, mask_seed, dictionary, **kwargs):
        dataset = raw_dataset
        if True:  # or split == "train":
            sample_dataset = ConformerSampleDataset(
                dataset, self.args.seed, "target_atoms", "target_coordinates"
            )
            dataset = AtomTypeDataset(dataset, sample_dataset)
        else:
            dataset = TTADataset(
                dataset, self.args.seed, "atoms", "coordinates", self.args.conf_size
            )
            dataset = AtomTypeDataset(dataset, dataset)
        dataset = RemoveHydrogenDataset(
            dataset,
            "atoms",
            "coordinates",
            self.args.remove_hydrogen,
            self.args.remove_polar_hydrogen,
        )

        dataset = CroppingDataset(
            dataset, self.seed, "atoms", "coordinates", self.args.max_atoms
        )
        dataset = NormalizeDataset(
            dataset, "coordinates", normalize_coord=True)
        src_dataset = KeyDataset(dataset, "atoms")
        src_dataset = TokenizeDataset(
            src_dataset, dictionary, max_seq_len=self.args.max_seq_len
        )
        coord_dataset = KeyDataset(dataset, "coordinates")

        def PrependAndAppend(dataset, pre_token, app_token):
            dataset = PrependTokenDataset(dataset, pre_token)
            return AppendTokenDataset(dataset, app_token)

        src_dataset = PrependAndAppend(
            src_dataset, dictionary.bos(), dictionary.eos()
        )
        edge_type = EdgeTypeDataset(src_dataset, len(dictionary))
        coord_dataset = FromNumpyDataset(coord_dataset)
        coord_dataset = PrependAndAppend(coord_dataset, 0.0, 0.0)
        distance_dataset = DistanceDataset(coord_dataset)

        result = {
            "src_tokens": RightPadDataset(
                src_dataset,
                pad_idx=dictionary.pad(),
            ),
            "src_coord": RightPadDatasetCoord(
                coord_dataset,
                pad_idx=0,
            ),
            "src_distance": RightPadDataset2D(
                distance_dataset,
                pad_idx=0,
            ),
            "src_edge_type": RightPadDataset2D(
                edge_type,
                pad_idx=0,
            ),
        }, {}

        return result

    def load_dataset(self, split, combine=False, **kwargs):
        """Load a given dataset split.
        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        split_path = os.path.join(self.args.data, split + ".lmdb")
        raw_dataset = LMDBDataset(split_path)

        if not self.args.use_old_dataset:
            net_input, target = super().one_dataset(
                raw_dataset, self.args.seed, self.args.seed, split=split)
        else:
            net_input, target = super().one_dataset_old(
                raw_dataset, self.args.seed, self.args.seed)
        net_input["smiles_src_tokens"] = net_input.pop("src_tokens")
        
        
        net_input_new, target_new = self.one_dataset(
            raw_dataset, self.args.seed, self.args.seed, self.encoder_dictionary)
        net_input.update(net_input_new)
        target.update(target_new)
        
        dataset = {"net_input": net_input, "target": target}
        dataset = NestedDictionaryDataset(dataset)
        if split in ["train", "train.small"]:
            dataset = EpochShuffleDataset(
                dataset, len(dataset), self.args.seed)
        self.datasets[split] = dataset

    def load_empty_dataset(self, combine=False, **kwargs):

        raw_dataset = LMDBDataset(split_path)

        if not self.args.use_old_dataset:
            net_input, target = super().one_dataset(
                raw_dataset, self.args.seed, self.args.seed, split=split)
        else:
            net_input, target = super().one_dataset_old(
                raw_dataset, self.args.seed, self.args.seed)
                
        net_input["smiles_src_tokens"] = net_input.pop("src_tokens")
        
        net_input_new, target_new = self.one_dataset(
            raw_dataset, self.args.seed, self.args.seed, self.encoder_dictionary)
        net_input.update(net_input_new)
        target.update(target_new)
        
        dataset = {"net_input": net_input, "target": target}
        dataset = NestedDictionaryDataset(dataset)
        if split in ["train", "train.small"]:
            dataset = EpochShuffleDataset(
                dataset, len(dataset), self.args.seed)
        self.datasets[split] = dataset

    def build_model(self, args):
        from unicore import models
        model = models.build_model(args, self)
        if args.unimol_path != "None":
            state = checkpoint_utils.load_checkpoint_to_cpu(args.unimol_path)
            if args.encoder_type == "unimol":
                model.encoder.load_state_dict(state["model"], strict=False)
            elif args.encoder_type == "default_and_unimol":
                model.encoder[1].load_state_dict(state["model"], strict=False)
            logger.info("load weight unimol")
        if args.freeze_unimol:
            freeze_network(model.encoder)
        return model


@register_task("ReRPTask_RS_Unimol")
class ReRPTask_RS_Unimol(ReRPTask_Unimol):
    def __init__(self, args, dictionary, encoder_dictionary):
        super().__init__(args, dictionary, encoder_dictionary)
        self.task_type = "retrosynthetic"
