# Copyright (c) DP Techonology, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from unimol import __version__
from unicore import distributed_utils
from ..utils import save_config

if __version__ in ["1.5.0", "2.0.0", "2.5.0"]:
    import logging
    import os
    import torch
    import numpy as np
    from unicore.data import (
        Dictionary,
        LMDBDataset,
        RightPadDataset,
        # TokenizeDataset,
        RightPadDataset2D,
        NestedDictionaryDataset,
        EpochShuffleDataset,
    )
    from unimol.data import (
        KeyDataset,
    )
    from ..data import (
        CsvGraphormerDataset,
        SmilesDataset,
        GraphormerDataset,
        SeqGraphormerDataset,
        RightPadDataset3D,
        ReorderGraphormerDataset,
        GraphFeatures,
        RandomSmilesDataset,
        ReorderSmilesDataset,
        EMPTY_SMILES_Dataset_G2GT,
        BpeTokenizeDataset,
        TokenizeDataset,
    )
    from ..utils.chemutils import add_chirality
    from ..utils.G2GT_cal import get_smiles, gen_map
    from unicore.tasks import UnicoreTask, register_task
    from unicore import checkpoint_utils

    logger = logging.getLogger(__name__)

    @register_task("G2GT")
    class G2GTMTask(UnicoreTask):
        """Task for training transformer auto-encoder models."""

        @staticmethod
        def add_args(parser):
            """Add task-specific arguments to the parser."""
            parser.add_argument("data", help="downstream data path")

            parser.add_argument(
                "--dict-name",
                default="dict.txt",
                help="dict name",
            )

            parser.add_argument(
                "--laplacian_pe_dim",
                default=30,
                type=int,
                help="laplacian_pe_dim",
            )

            parser.add_argument(
                "--use_reorder", action="store_true", help="use-reorder"
            )

            parser.add_argument(
                "--decoder_attn_from_loader",
                action="store_true",
                help="decoder_attn_from_loader",
            )
            parser.add_argument(
                "--not_want_edge_input",
                action="store_true",
                help="not_want_edge_input",
            )
            parser.add_argument(
                "--want_charge_h",
                action="store_true",
                help="want_charge_h",
            )
            parser.add_argument(
                "--shufflegraph",
                default="none",
                help="shufflegraph",
            )
            parser.add_argument(
                "--not_sumto2",
                action="store_true",
                help="not_sumto2",
            )
            parser.add_argument(
                "--add_len",
                default=0,
                type=int,
                help="GraphFeatures add_len",
            )

            parser.add_argument(
                "--N_left",
                default=0,
                type=int,
                help="N_left",
            )

            parser.add_argument(
                "--infer_save_name",
                default="smi.txt",
                help="infer_save_name",
            )

            parser.add_argument(
                "--use_sep2",
                action="store_true",
                help="use_sep2",
            )

            parser.add_argument(
                "--want_h_degree",
                action="store_true",
                help="want_h_degree",
            )

            parser.add_argument(
                "--want_bond_attn",
                action="store_true",
                help="want_bond_attn",
            )

            parser.add_argument(
                "--use_class",
                action="store_true",
                help="use_class",
            )

            parser.add_argument(
                "--infer_step",
                action="store_true",
                help="infer_step",
            )

            parser.add_argument(
                "--idx_type",
                default=0,
                type=int,
                help="idx_type",
            )

            parser.add_argument(
                "--want_decoder_attn",
                action="store_true",
                help="want_decoder_attn",
            )

            parser.add_argument(
                "--init_train_path",
                default="none",
                help="init_train_path",
            )

            parser.add_argument(
                "--bpe_tokenizer_path",
                default="none",
                help="bpe_tokenizer_path",
            )

            parser.add_argument(
                "--charge_h_last",
                action="store_true",
                help="charge_h_last",
            )

            parser.add_argument(
                "--use_class_encoder",
                action="store_true",
                help="use_class_encoder",
            )

            parser.add_argument(
                "--task_type",
                default="retrosynthesis",
                choices=["retrosynthesis", "synthesis"],
                help="noise type in coordinate noise",
            )

            parser.add_argument(
                "--encoder_pe_after",
                action="store_true",
                help="encoder_pe_after",
            )

            parser.add_argument(
                "--weight_encoder",
                default="",
                help="weight_encoder",
            )

            parser.add_argument(
                "--multi_gap",
                action="store_true",
                help="multi_gap",
            )

            parser.add_argument(
                "--random_smiles_prob",
                default=1.0,
                type=float,
                help="random_smiles_prob",
            )

            parser.add_argument(
                "--use_stereoisomerism",
                action="store_true",
                help="use_stereoisomerism",
            )

            save_config.add_config_save_args(parser)

        def __init__(self, args, dictionary):
            if (distributed_utils.get_data_parallel_rank() == 0) and (
                args.infer_step is False
            ):
                save_config.save_config(args)
            super().__init__(args)
            self.seed = args.seed
            self.dictionary = dictionary
            if self.args.bpe_tokenizer_path != "none":
                self.infer_dictionary = Dictionary.load(
                    os.path.join(args.bpe_tokenizer_path, "dict.txt")
                )

        @classmethod
        def setup_task(cls, args, **kwargs):
            dictionary = Dictionary.load(os.path.join(args.data, args.dict_name))
            logger.info("dictionary: {} types".format(len(dictionary)))
            return cls(args, dictionary)

        def load_empty_dataset(self, name="product_smiles", **kwargs):
            split = "test"
            flag_label_wanted = False
            dataset = EMPTY_SMILES_Dataset_G2GT(name=name, **kwargs)
            self.load_dataset_detail(
                dataset=dataset,
                split=split,
                class_dataset=None,
                flag_label_wanted=flag_label_wanted,
                **kwargs
            )
            return dataset

        def load_dataset(self, split, **kwargs):
            flag_label_wanted = True
            data_path = kwargs.get("data_path")
            data_path = data_path if data_path else self.args.data
            split_path = os.path.join(data_path, split + ".csv")
            if os.path.exists(split_path):
                raw_dataset = CsvGraphormerDataset(split_path)
            else:
                split_path = os.path.join(data_path, split + ".lmdb")
                raw_dataset = LMDBDataset(split_path)

            if self.args.use_class:
                class_dataset = KeyDataset(raw_dataset, "class")
            else:
                class_dataset = None
            dataset = KeyDataset(raw_dataset, "rxn_smiles")
            dataset = SmilesDataset(dataset)

            self.load_dataset_detail(
                dataset=dataset,
                split=split,
                class_dataset=class_dataset,
                flag_label_wanted=flag_label_wanted,
                **kwargs
            )

        def load_dataset_detail(
            self, dataset, split, class_dataset=None, flag_label_wanted=True, **kwargs
        ):
            """Load a given dataset split.
            Args:
                split (str): name of the data scoure (e.g., bppp)
            """

            is_train = "train" in split
            flag_aftsep2 = "aftsep2" in self.args.bpe_tokenizer_path

            if self.args.task_type == "retrosynthesis":
                input_name, output_name = "product_smiles", "reactant_smiles"
            elif self.args.task_type == "synthesis":
                input_name, output_name = "reactant_smiles", "product_smiles"
            else:
                print(self.args.task_type)
                raise
            if flag_label_wanted:
                reactant_dataset = KeyDataset(dataset, output_name)
            product_dataset = KeyDataset(dataset, input_name)

            if is_train and self.args.shufflegraph == "randomsmiles":
                product_dataset = RandomSmilesDataset(
                    product_dataset, prob=self.args.random_smiles_prob
                )
                if not self.args.use_reorder:
                    reactant_dataset = RandomSmilesDataset(
                        reactant_dataset, prob=self.args.random_smiles_prob
                    )

            product_smiles_dataset = product_dataset
            product_dataset = GraphormerDataset(
                product_dataset, use_stereoisomerism=self.args.use_stereoisomerism
            )

            if flag_label_wanted:
                reactant_smiles_dataset = reactant_dataset
                if self.args.use_reorder:
                    reactant_dataset = ReorderSmilesDataset(
                        product_smiles_dataset, reactant_dataset
                    )
                    mol_reactant_dataset = KeyDataset(reactant_dataset, "mol")
                    reactant_dataset = KeyDataset(reactant_dataset, "smiles")
                    reactant_smiles_dataset = reactant_dataset
                    reactant_dataset = GraphormerDataset(
                        mol_reactant_dataset,
                        use_stereoisomerism=self.args.use_stereoisomerism,
                    )

                reactant_dataset = SeqGraphormerDataset(
                    reactant_dataset,
                    class_dataset,
                    min_node=self.args.laplacian_pe_dim,
                    want_attn=self.args.decoder_attn_from_loader,
                    want_charge_h=self.args.want_charge_h,
                    # max_seq_len=self.args.max_seq_len,
                    sumto2=not self.args.not_sumto2,
                    use_sep2=self.args.use_sep2 or flag_aftsep2,
                    want_h_degree=self.args.want_h_degree,
                    idx_type=self.args.idx_type,
                    charge_h_last=self.args.charge_h_last,
                    multi_gap=self.args.multi_gap,
                    use_stereoisomerism=self.args.use_stereoisomerism,
                )

                seq_reactant_dataset = KeyDataset(reactant_dataset, "seq")

                seq_reactant_dataset = TokenizeDataset(
                    seq_reactant_dataset,
                    self.dictionary,
                    max_seq_len=self.args.max_seq_len + 1,
                )
                if self.args.bpe_tokenizer_path != "none":
                    seq_reactant_dataset = BpeTokenizeDataset(
                        seq_reactant_dataset,
                        self.args.bpe_tokenizer_path,
                        flag_aftsep2=flag_aftsep2,
                    )
            product_dataset = GraphFeatures(
                product_dataset,
                pos_dataset=None,
                want_edge_input=not self.args.not_want_edge_input,
                add_len=self.args.add_len,
            )

            net_input = {"batched_data": product_dataset}

            if flag_label_wanted:
                net_input["decoder_src_tokens"] = RightPadDataset(
                    seq_reactant_dataset,
                    pad_idx=self.dictionary.pad(),
                )

            if flag_label_wanted and self.args.decoder_attn_from_loader:
                reactant_degree_attn_mask_dataset = KeyDataset(
                    reactant_dataset, "degree_attn_mask"
                )
                net_input["decoder_degree_attn_mask"] = RightPadDataset2D(
                    reactant_degree_attn_mask_dataset,
                    pad_idx=0,
                )
                if self.args.laplacian_pe_dim > 0:
                    reactant_laplacian_attn_mask_dataset = KeyDataset(
                        reactant_dataset, "laplacian_attn_mask"
                    )
                    net_input["decoder_laplacian_attn_mask"] = RightPadDataset3D(
                        reactant_laplacian_attn_mask_dataset,
                        pad_idx=0,
                    )
            target = {"product_smiles": product_smiles_dataset}

            if flag_label_wanted:
                target["reactant_smiles"] = reactant_smiles_dataset

            nest_dataset = NestedDictionaryDataset(
                {"net_input": net_input, "target": target},
            )
            if split in ["train", "train.small"]:
                nest_dataset = EpochShuffleDataset(
                    nest_dataset, len(nest_dataset), self.seed
                )
            self.datasets[split] = nest_dataset

        def build_model(self, args):
            from unicore import models

            model = models.build_model(args, self)
            if args.init_train_path != "none":
                state = checkpoint_utils.load_checkpoint_to_cpu(args.init_train_path)
                model.load_state_dict(state["model"], strict=False)
            return model

        def get_str(self, tokens):
            if self.args.bpe_tokenizer_path == "none":
                dictionary = self.dictionary
            else:
                dictionary = self.infer_dictionary
            pad_idx = dictionary.pad()
            tokens = tokens.tolist()
            if pad_idx in tokens:
                pad_idx = tokens.index(dictionary.pad())
                token = tokens[1:pad_idx]
            else:
                token = tokens[1:]
            if not hasattr(dictionary, "id2word"):
                dictionary.id2word = {v: k for k, v in dictionary.indices.items()}
            strs = " ".join([dictionary.id2word[i] for i in token])
            if self.args.bpe_tokenizer_path != "none":
                strs = strs.replace("?", " ")
            return strs

        def fill_first_step_tokens(self, bsz, beam_size, j, pred_dict):
            max_pred_shape = max(
                [pred_dict[i][j]["tokens"].shape[0] for i in range(bsz)]
            )
            scores = torch.zeros(bsz * beam_size, max_pred_shape).float()
            tokens = (
                torch.zeros(bsz * beam_size, max_pred_shape + 1)
                .long()
                .fill_(self.dictionary.pad())
            )
            for i in range(bsz):
                for k in range(beam_size):
                    pred_shape = pred_dict[i][j]["tokens"].shape[0] + 1
                    tokens[i * beam_size + k, 1:pred_shape] = pred_dict[i][j]["tokens"]
                    scores[i * beam_size + k, : pred_shape - 1] = pred_dict[i][j][
                        "score"
                    ]
            return tokens, scores

        def write_file_res(
            self,
            gt_product_smiles,
            gt_reactant_smiles,
            tgt_tokens,
            pred,
            beam_size,
            file_check,
        ):
            for i in range(len(gt_reactant_smiles)):
                with open(file_check, "a") as w:
                    w.write(gt_product_smiles[i] + " " + "gt_product" + "\n")
                    w.write(gt_reactant_smiles[i] + " " + "target" + "\n")
                    if tgt_tokens is not None:
                        w.write(self.get_str(tgt_tokens[i]) + " " + "target" + "\n")
                    else:
                        w.write(" " + "target" + "\n")

                    for j in range(beam_size):
                        if self.args.search_strategies in [
                            "SequenceGeneratorBeamSearch_test",
                            "SequenceGeneratorBeamSearch",
                            "SequenceScoreGeneratorBeamSearch",
                        ]:
                            pred_piece = pred[i][j]["tokens"].cpu().numpy()
                            score = pred[i][j]["score"].cpu().detach().numpy()
                            pred_piece = np.insert(pred_piece, 0, 1)
                            pred_piece = pred_piece[:-1]
                        elif self.args.search_strategies == "SimpleGenerator":
                            pred_piece = pred[0][j + i * beam_size].cpu().numpy()
                            score = pred[1][j + i * beam_size]
                        pred_piece = self.get_str(pred_piece)
                        w.write(
                            pred_piece
                            + " "
                            + "predicted"
                            + str(j)
                            + " "
                            + str(score)
                            + "\n"
                        )

        def get_one_sample(self, sample, i):
            new_dict = {}
            for k, v in sample["net_input"]["batched_data"].items():
                new_dict[k] = v[i : i + 1]
            dict_ = {"net_input": {"batched_data": new_dict}}
            return dict_

        def test_step(
            self,
            args,
            sample,
            model,
            loss,
            step,
            seed,
            second_beam_size=0,
            second_token_size=0,
            model2=None,
        ):
            gt_reactant_smiles = sample["target"]["reactant_smiles"]
            gt_product_smiles = sample["target"]["product_smiles"]
            beam_size = model.beam_size
            file_path = args.results_path
            if self.args.search_strategies == "SequenceGeneratorBeamSearch_test":
                prefix_tokens = None
                if self.args.use_class:
                    prefix_tokens = sample["net_input"]["decoder_src_tokens"][
                        :, 1
                    ].unsqueeze(1)
                else:
                    prefix_tokens = None
                pred = model(sample=sample, prefix_tokens=prefix_tokens)
                bsz = self.args.batch_size
                if len(model2) > 0:
                    list_bs = []
                    for i in range(bsz):
                        list_beam = []
                        for j in range(len(model2)):
                            sample_tmp = self.get_one_sample(sample, i)
                            pred_piece = pred[i][j]["tokens"]
                            score = pred[i][j]["score"]
                            pred_k = model2[j](
                                sample_tmp, prefix_tokens=pred_piece.unsqueeze(0)
                            )
                            for k in range(len(pred_k[0])):
                                pred_k[0][k]["score"] = score + pred_k[0][k]["score"]
                                list_beam.append(pred_k[0][k])
                        list_bs.append(list_beam)
                    pred = list_bs
                vae_kl_loss = None
            elif self.args.search_strategies in [
                "SequenceGeneratorBeamSearch",
                "SequenceScoreGeneratorBeamSearch",
            ]:
                if self.args.use_class:
                    prefix_tokens = sample["net_input"]["decoder_src_tokens"][
                        :, 1
                    ].unsqueeze(1)
                else:
                    prefix_tokens = None
                pred = model(sample=sample, prefix_tokens=prefix_tokens)
                vae_kl_loss = None
            elif self.args.search_strategies == "SimpleGenerator":
                pred, vae_kl_loss, simple_score = model._generate(sample)
                pred = (pred, simple_score)
            else:
                raise

            if not os.path.exists(file_path):
                os.makedirs(file_path)
            rank = distributed_utils.get_data_parallel_rank()
            file_check = os.path.join(
                file_path,
                self.args.infer_save_name.replace(".txt", "_" + str(rank) + ".txt"),
            )
            if "decoder_src_tokens" in sample["net_input"].keys():
                tgt_tokens = sample["net_input"]["decoder_src_tokens"].cpu().numpy()
            else:
                tgt_tokens = None
            self.write_file_res(
                gt_product_smiles,
                gt_reactant_smiles,
                tgt_tokens,
                pred,
                beam_size,
                file_check,
            )

            return pred, {
                "vae_kl_loss": vae_kl_loss,
                "sample_size": 1,
                "bsz": sample["net_input"]["decoder_src_tokens"].size(0) if "decoder_src_tokens" in sample["net_input"].keys() else 1,
                "seq_len": sample["net_input"]["decoder_src_tokens"].size(1)
                * sample["net_input"]["decoder_src_tokens"].size(0) if "decoder_src_tokens" in sample["net_input"].keys() else 1,
            }

        def infer_step(self, sample, model, **kwargs):
            # RMK: NoneAtom
            gt_product_smiles = sample["target"]["product_smiles"]
            if self.args.search_strategies in [
                "SequenceGeneratorBeamSearch",
                "SequenceScoreGeneratorBeamSearch",
            ]:
                pred = model(sample)
                vae_kl_loss = None
            elif self.args.search_strategies == "SimpleGenerator":
                pred, vae_kl_loss, simple_score = model._generate(sample)
            else:
                raise
            beam_size = model.beam_size
            result_list = []
            for i in range(len(gt_product_smiles)):
                list_ = []
                for j in range(beam_size):
                    if self.args.search_strategies in [
                        "SequenceGeneratorBeamSearch",
                        "SequenceScoreGeneratorBeamSearch",
                    ]:
                        pred_piece = pred[i][j]["tokens"].cpu().numpy()
                        score = pred[i][j]["score"].cpu().detach().numpy()
                        pred_piece = np.insert(pred_piece, 0, 1)
                        pred_piece = pred_piece[:-1]
                    elif self.args.search_strategies == "SimpleGenerator":
                        pred_piece = pred[j + i * beam_size].cpu().numpy()
                        score = simple_score[j + i * beam_size]

                    pred_piece = self.get_str(pred_piece)
                    pred_piece = get_smiles(
                        pred_piece, atom_map=gen_map(gt_product_smiles[i])
                    )
                    try:
                        pred_piece = add_chirality(
                            gt_product_smiles[i], pred_piece, task=self.args.task_type
                        )
                    except:
                        pass
                    list_.append((pred_piece, score.item()))
                result_list.append(list_)
            return result_list
