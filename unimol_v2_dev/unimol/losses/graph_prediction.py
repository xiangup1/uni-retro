from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
import numpy as np
from unicore import metrics
from unicore.losses import UnicoreLoss, register_loss
from scipy.spatial.transform import Rotation as R
from typing import List, Callable, Any, Dict
import os


@register_loss("Transformer-M")
class GraphPredictionLoss(UnicoreLoss):
    """
    Implementation for the loss used in masked graph model (MGM) training.
    """

    def __init__(self, task):
        super().__init__(task)
        self.noise_scale = task.args.noise_scale

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """

        with torch.no_grad():
            sample_size = sample["net_input"]["batched_data"]["x"].shape[0]
            natoms = sample["net_input"]["batched_data"]["x"].shape[1]

        # add gaussian noise
        ori_pos = sample["net_input"]["batched_data"]["pos"]
        noise = torch.randn(ori_pos.shape).to(ori_pos) * self.noise_scale
        noise_mask = (ori_pos == 0.0).all(dim=-1, keepdim=True)
        noise = noise.masked_fill_(noise_mask, 0.0)
        sample["net_input"]["batched_data"]["pos"] = ori_pos + noise

        model_output = model(**sample["net_input"])
        logits, node_output = model_output[0], model_output[1]
        logits = logits[:, 0, :].view(-1).float()
        targets = sample["target"]["gap_target"].float().view(-1)

        loss = torch.nn.L1Loss(reduction="sum")(logits, targets)

        if node_output is not None:
            node_mask = (node_output == 0.0).all(dim=-1).all(dim=-1)[
                :, None, None
            ] + noise_mask
            node_output = node_output.masked_fill_(node_mask, 0.0)

            node_output_loss = 1.0 - torch.nn.CosineSimilarity(dim=-1)(
                node_output.to(torch.float32),
                noise.masked_fill_(node_mask, 0.0).to(torch.float32),
            )
            node_output_loss = node_output_loss.masked_fill_(
                node_mask.squeeze(-1), 0.0
            ).sum(dim=-1)

            tgt_count = (~node_mask).squeeze(-1).sum(dim=-1).to(node_output_loss)
            tgt_count = tgt_count.masked_fill_(tgt_count == 0.0, 1.0)
            node_output_loss = (node_output_loss / tgt_count).sum() * 1
        else:
            node_output_loss = (noise - noise).sum()

        logging_output = {
            "loss": loss.data,
            "node_output_loss": node_output_loss.data,
            "total_loss": loss.data + node_output_loss.data,
            "sample_size": sample_size,
            "nsentences": sample_size,
            "ntokens": natoms,
            "bsz": sample_size,
            "n_atoms": natoms * sample_size,
        }
        return loss + node_output_loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs, split="valid") -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        n_atoms = sum(log.get("n_atoms", 0) for log in logging_outputs)
        node_output_loss_sum = sum(
            log.get("node_output_loss", 0) for log in logging_outputs
        )
        total_loss_sum = sum(log.get("total_loss", 0) for log in logging_outputs)

        metrics.log_scalar("loss", loss_sum / sample_size, sample_size, round=6)
        metrics.log_scalar(
            "node_output_loss", node_output_loss_sum / sample_size, sample_size, round=6
        )
        metrics.log_scalar(
            "total_loss", total_loss_sum / sample_size, sample_size, round=6
        )
        metrics.log_scalar("n_atoms", n_atoms / sample_size, sample_size, round=6)

    @staticmethod
    def logging_outputs_can_be_summed(is_train) -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
