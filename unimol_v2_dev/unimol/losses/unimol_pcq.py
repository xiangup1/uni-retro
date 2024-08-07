# Copyright (c) DP Techonology, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F
import numpy as np
from unicore import metrics
from unicore.losses import UnicoreLoss, register_loss
from scipy.spatial.transform import Rotation as R
from typing import List, Callable, Any, Dict


@register_loss("unimol_pcq")
class UniMolPCQLoss(UnicoreLoss):
    def __init__(self, task):
        super().__init__(task)
        self.padding_idx = task.dictionary.pad()
        self.eos_idx = task.dictionary.eos()
        self.bos_idx = task.dictionary.bos()
        self.seed = task.seed
        self.relu = torch.nn.ReLU()

    def forward(self, model, sample, reduce=True):
        input_key = "net_input"
        target_key = "target"
        # token_mask = sample[target_key]["tokens_target"].ne(self.padding_idx)
        token_mask = sample[input_key]["src_tokens"].ne(self.padding_idx)
        token_mask &= sample[input_key]["src_tokens"].ne(self.eos_idx)
        # bos is center, also include it in the loss
        # token_mask &= sample[input_key]["src_tokens"].ne(self.bos_idx)
        distance_mask, coord_mask = calc_mask(token_mask)
        (
            logits,
            encoder_distance,
            encoder_coord,
            x_norm,
            delta_encoder_pair_rep_norm,
            pre_coord,
            plddt,
            p_homo_lumo,
            pred_homo_lumo,
        ) = model(**sample[input_key], encoder_masked_tokens=token_mask)

        logging_output = {
            "sample_size": 1,
            "bsz": sample[target_key]["tokens_target"].size(0),
            "seq_len": sample[target_key]["tokens_target"].size(1)
            * sample[target_key]["tokens_target"].size(0),
        }

        if encoder_coord is not None:
            encoder_coord = encoder_coord.float()
            # coord loss
            # real = mask + delta
            coord_target = sample[target_key]["coord_target"]
            masked_coord_loss = F.l1_loss(
                encoder_coord[coord_mask],
                coord_target[coord_mask].float(),
                reduction="mean",
            )
            loss = masked_coord_loss * self.args.masked_coord_loss
            # restore the scale of loss for displaying
            logging_output["masked_coord_loss"] = masked_coord_loss.data

        if encoder_distance is not None:
            # distance loss
            distance_target = sample[target_key]["distance_target"][distance_mask]
            distance_predict = encoder_distance[distance_mask]
            masked_dist_loss = F.l1_loss(
                distance_predict.float(),
                distance_target.float(),
                reduction="mean",
            )
            loss = loss + masked_dist_loss * self.args.masked_dist_loss
            logging_output["masked_dist_loss"] = masked_dist_loss.data

        if pre_coord is not None:
            pre_coord = pre_coord.float()
            regular_distance = torch.norm(
                encoder_coord[token_mask].view(-1, 3)
                - pre_coord[token_mask].view(-1, 3),
                dim=-1,
            )
            regular_distance = self.relu(regular_distance - self.args.max_dist)
            distance_regular_loss = torch.mean(regular_distance)
            loss = loss + distance_regular_loss * self.args.dist_regular_loss
            logging_output["dist_regular_loss"] = distance_regular_loss.data

        if pred_homo_lumo is not None:
            # which loss function to choose
            pred_homo_lumo_loss = F.l1_loss(
                sample[target_key]["gap_target"].float(),
                pred_homo_lumo.view(-1).float(),
                reduction="mean",
            )
            loss = loss + pred_homo_lumo_loss * self.args.homo_lumo_loss
            logging_output["pred_homo_lumo_loss"] = pred_homo_lumo_loss.data

        if p_homo_lumo is not None:
            num_bins = 50
            homo_lumo = torch.abs(
                pred_homo_lumo.float() - sample[target_key]["gap_target"]
            ).detach()
            bin_index = torch.floor(homo_lumo * num_bins).long()
            bin_index = torch.clamp(bin_index, max=(num_bins - 1))
            p_homo_lumo_one_hot = torch.nn.functional.one_hot(
                bin_index, num_classes=num_bins
            )
            p_homo_lumo_loss = self.softmax_cross_entropy(
                p_homo_lumo, p_homo_lumo_one_hot
            )
            p_homo_lumo_loss = torch.mean(p_homo_lumo_loss)
            loss = loss + p_homo_lumo_loss * self.args.p_homo_lumo_loss
            logging_output["p_homo_lumo_loss"] = p_homo_lumo_loss.data

        if plddt is not None:
            coord_target = sample[target_key]["coord_target"]
            all_atom_pred_pos = encoder_coord.float()
            all_atom_positions = coord_target.float()
            all_atom_mask = token_mask.unsqueeze(-1)  # keep dim

            cutoff = 15.0
            num_bins = 50
            eps = 1e-10
            lddt = self.compute_lddt(
                all_atom_pred_pos,
                all_atom_positions,
                all_atom_mask,
                cutoff=cutoff,
                eps=eps,
            ).detach()

            bin_index = torch.floor(lddt * num_bins).long()
            bin_index = torch.clamp(bin_index, max=(num_bins - 1))
            lddt_ca_one_hot = torch.nn.functional.one_hot(
                bin_index, num_classes=num_bins
            )
            errors = self.softmax_cross_entropy(plddt, lddt_ca_one_hot)
            all_atom_mask = all_atom_mask.squeeze(-1)
            plddt_loss = self.masked_mean(all_atom_mask, errors, dim=-1, eps=eps)
            ca_lddt = self.masked_mean(all_atom_mask, lddt, dim=-1, eps=eps)
            plddt_loss = torch.mean(plddt_loss)
            ca_lddt = torch.mean(ca_lddt)

            loss = loss + plddt_loss * self.args.lddt_loss
            logging_output["plddt_loss"] = plddt_loss.data
            logging_output["ca_lddt"] = ca_lddt.data

        if self.args.x_norm_loss > 0 and x_norm is not None:
            loss = loss + self.args.x_norm_loss * x_norm
            logging_output["x_norm_loss"] = x_norm.data

        if (
            self.args.delta_pair_repr_norm_loss > 0
            and delta_encoder_pair_rep_norm is not None
        ):
            loss = (
                loss + self.args.delta_pair_repr_norm_loss * delta_encoder_pair_rep_norm
            )
            logging_output[
                "delta_pair_repr_norm_loss"
            ] = delta_encoder_pair_rep_norm.data

        logging_output["loss"] = loss.data
        return loss, 1, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs, split="valid") -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        bsz = sum(log.get("bsz", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        seq_len = sum(log.get("seq_len", 0) for log in logging_outputs)
        metrics.log_scalar("loss", loss_sum / sample_size, sample_size, round=3)
        metrics.log_scalar("seq_len", seq_len / bsz, 1, round=3)

        masked_loss = sum(log.get("masked_token_loss", 0) for log in logging_outputs)
        if masked_loss > 0:
            metrics.log_scalar(
                "masked_token_loss", masked_loss / sample_size, sample_size, round=3
            )
        masked_coord_loss = sum(
            log.get("masked_coord_loss", 0) for log in logging_outputs
        )
        if masked_coord_loss > 0:
            metrics.log_scalar(
                "masked_coord_loss",
                masked_coord_loss / sample_size,
                sample_size,
                round=3,
            )

        masked_dist_loss = sum(
            log.get("masked_dist_loss", 0) for log in logging_outputs
        )
        if masked_dist_loss > 0:
            metrics.log_scalar(
                "masked_dist_loss", masked_dist_loss / sample_size, sample_size, round=3
            )

        x_norm_loss = sum(log.get("x_norm_loss", 0) for log in logging_outputs)
        if x_norm_loss > 0:
            metrics.log_scalar(
                "x_norm_loss", x_norm_loss / sample_size, sample_size, round=3
            )

        delta_pair_repr_norm_loss = sum(
            log.get("delta_pair_repr_norm_loss", 0) for log in logging_outputs
        )
        if delta_pair_repr_norm_loss > 0:
            metrics.log_scalar(
                "delta_pair_repr_norm_loss",
                delta_pair_repr_norm_loss / sample_size,
                sample_size,
                round=3,
            )

        plddt_loss = sum(log.get("plddt_loss", 0) for log in logging_outputs)
        if plddt_loss > 0:
            metrics.log_scalar(
                "plddt_loss", plddt_loss / sample_size, sample_size, round=3
            )
        ca_lddt = sum(log.get("ca_lddt", 0) for log in logging_outputs)
        if ca_lddt > 0:
            metrics.log_scalar("ca_lddt", ca_lddt / sample_size, sample_size, round=3)

        p_homo_lumo_loss = sum(
            log.get("p_homo_lumo_loss", 0) for log in logging_outputs
        )
        if p_homo_lumo_loss > 0:
            metrics.log_scalar(
                "p_homo_lumo_loss", p_homo_lumo_loss / sample_size, sample_size, round=3
            )

        pred_homo_lumo_loss = sum(
            log.get("pred_homo_lumo_loss", 0) for log in logging_outputs
        )
        if pred_homo_lumo_loss > 0:
            metrics.log_scalar(
                "pred_homo_lumo_loss",
                pred_homo_lumo_loss / sample_size,
                sample_size,
                round=3,
            )

        dist_regular_loss = sum(
            log.get("dist_regular_loss", 0) for log in logging_outputs
        )
        if dist_regular_loss > 0:
            metrics.log_scalar(
                "dist_regular_loss",
                dist_regular_loss / sample_size,
                sample_size,
                round=3,
            )

    @staticmethod
    def logging_outputs_can_be_summed(is_train) -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return is_train

    def compute_lddt(
        self,
        all_atom_pred_pos: torch.Tensor,
        all_atom_positions: torch.Tensor,
        all_atom_mask: torch.Tensor,
        cutoff: float = 15.0,
        eps: float = 1e-10,
    ) -> torch.Tensor:
        n = all_atom_mask.shape[-2]
        dmat_true = torch.sqrt(
            eps
            + torch.sum(
                (all_atom_positions[..., None, :] - all_atom_positions[..., None, :, :])
                ** 2,
                dim=-1,
            )
        )

        dmat_pred = torch.sqrt(
            eps
            + torch.sum(
                (all_atom_pred_pos[..., None, :] - all_atom_pred_pos[..., None, :, :])
                ** 2,
                dim=-1,
            )
        )
        dists_to_score = (
            (dmat_true < cutoff)
            * all_atom_mask
            * self.permute_final_dims(all_atom_mask, (1, 0))
            * (1.0 - torch.eye(n, device=all_atom_mask.device))
        )

        dist_l1 = torch.abs(dmat_true - dmat_pred)

        score = (
            (dist_l1 < 0.1).type(dist_l1.dtype)
            + (dist_l1 < 0.2).type(dist_l1.dtype)
            + (dist_l1 < 0.4).type(dist_l1.dtype)
            + (dist_l1 < 0.8).type(dist_l1.dtype)
        )
        score = score * 0.25

        norm = 1.0 / (eps + torch.sum(dists_to_score, dim=-1))
        score = norm * (eps + torch.sum(dists_to_score * score, dim=-1))

        return score

    def permute_final_dims(self, tensor: torch.Tensor, inds: List[int]):
        zero_index = -1 * len(inds)
        first_inds = list(range(len(tensor.shape[:zero_index])))
        return tensor.permute(first_inds + [zero_index + i for i in inds])

    def masked_mean(self, mask, value, dim, eps=1e-10, keepdim=False):
        mask = mask.expand(*value.shape)
        return torch.sum(mask * value, dim=dim, keepdim=keepdim) / (
            eps + torch.sum(mask, dim=dim, keepdim=keepdim)
        )

    def softmax_cross_entropy(self, logits, labels):
        loss = -1 * torch.sum(
            labels * torch.nn.functional.log_softmax(logits.float(), dim=-1),
            dim=-1,
        )
        return loss


def calc_mask(token_mask):
    distance_mask = token_mask.unsqueeze(-1) & token_mask.unsqueeze(1)
    coord_mask = token_mask.unsqueeze(-1).expand(-1, -1, 3)
    return distance_mask, coord_mask
