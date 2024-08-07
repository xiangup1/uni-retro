import math
import torch
import torch.nn.functional as F
from unicore import metrics, utils
from unicore.losses import UnicoreLoss, register_loss
import torch.nn as nn
import torch.distributed as dist
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score, recall_score, f1_score


def get_loss(logits_decoder, decoder_target, padding_idx):
    decoder_target = decoder_target[:, 1:]
    logits_decoder = logits_decoder[:, :-1]
    decode_tokens = decoder_target.ne(padding_idx)
    decoder_sample_size = decode_tokens.long().sum()

    decoder_loss = F.nll_loss(
        F.log_softmax(
            logits_decoder[decode_tokens], dim=-1, dtype=torch.float32),
        decoder_target[decode_tokens].view(-1),
        ignore_index=padding_idx,
        reduction='mean',
    )

    decoder_pred = torch.argmax(
        logits_decoder[decode_tokens], dim=-1)
    # print('test decoder_pred: ', decoder_pred)
    decoder_hit = (decoder_pred == decoder_target[decode_tokens]).long().sum()
    decoder_cnt = decoder_sample_size

    acc_sentence_count = []
    for i in range(decoder_target.shape[0]):
        decoder_cnt_per_sen = decode_tokens[i].long().sum()
        decoder_pred_per_sen = torch.argmax(
            logits_decoder[i][decode_tokens[i]], dim=-1)
        decoder_hit_per_sen = (decoder_pred_per_sen ==
                               decoder_target[i][decode_tokens[i]]).long().sum()
        acc_sentence_count.append(decoder_hit_per_sen == decoder_cnt_per_sen)
    acc_sentence_count = (sum(acc_sentence_count), len(acc_sentence_count))
    return decoder_loss, decoder_hit, decoder_cnt, acc_sentence_count

def get_align_loss(logits, labels):
    log_logits =torch.log(logits+1e-6)
    align_loss = -torch.mean(log_logits * labels)
    return align_loss


@register_loss("ReRPBart")
class ReRPBartLoss(UnicoreLoss):
    def __init__(self, task):
        super().__init__(task)
        self.padding_idx = task.dictionary.pad()
        self.mask_id = -1
        # self.epoch = task.epoch

    def forward(self, model, sample, reduce=True):
        def inner_forward(input_key='net_input', target_key='target'):
            class_label = sample['net_input']['reaction_type']
            src_tokens_dataset = sample['net_input']['src_tokens']

            decoder_target = sample['net_input']['decoder_src_tokens']
            # torch.set_printoptions(profile = "full")
            
            logits_decoder, extra = model(
                src_tokens = src_tokens_dataset, prev_output_tokens = decoder_target,)

            loss = torch.tensor(0.0)

            logging_output = {
                "sample_size": 1,
                "bsz": sample[input_key]['src_tokens'].size(0),
                "seq_len": sample[input_key]['src_tokens'].size(1) * sample[input_key]['src_tokens'].size(0),
            }

            if logits_decoder is not None:

                decoder_target = sample[input_key]['decoder_src_tokens']

                decoder_loss, decoder_hit, decoder_cnt, acc_sentence_count = get_loss(
                    logits_decoder, decoder_target, self.padding_idx)
                loss = decoder_loss * self.args.decoder_loss
                logging_output = {
                    "sample_size": 1,
                    "bsz": sample[input_key]['src_tokens'].size(0),
                    "seq_len": sample[input_key]['src_tokens'].size(1) * sample[input_key]['src_tokens'].size(0),
                    "decoder_loss": decoder_loss.data,
                    "decoder_hit": decoder_hit.data,
                    "decoder_cnt": decoder_cnt.data,
                    "acc_sentence_hit": acc_sentence_count[0],
                    "acc_sentence_cnt": acc_sentence_count[1],
                }


            logging_output['loss'] = loss.data
            return loss, 1, logging_output

        loss, sample_size, logging_output = inner_forward()
        return loss, sample_size, logging_output

    # def accuracy(self):
    #     """ compute accuracy """
    #     return 100 * (self.n_correct / self.n_words)
    # def cross_entropy(self):
    #     """ compute cross entropy """
    #     return self.loss / self.n_words
    # def ppl(self):
    #     """ compute perplexity """
    #     return math.exp(min(self.loss / self.n_words, 100))

    @staticmethod
    def reduce_metrics(logging_outputs, split='valid') -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        bsz = sum(log.get("bsz", 1) for log in logging_outputs)     # RMK: 0->1 modified here
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        seq_len = sum(log.get("seq_len", 0) for log in logging_outputs)
        metrics.log_scalar(
            "loss", loss_sum / sample_size, sample_size, round=5
        )
        metrics.log_scalar(
            "seq_len", seq_len / bsz, 1, round=3
        )
        decoder_loss = sum(log.get('decoder_loss', 0)
                           for log in logging_outputs)
        if decoder_loss > 0:
            metrics.log_scalar('decoder_loss', decoder_loss /
                               sample_size, sample_size, round=5)
            decoder_acc = sum(log.get('decoder_hit', 0) for log in logging_outputs) / \
                sum(log.get('decoder_cnt', 1) for log in logging_outputs)
            if decoder_acc > 0:
                metrics.log_scalar(
                    'decoder_acc', decoder_acc, sample_size, round=5)

            decoder_cnt_t = sum(log.get('decoder_cnt', 1)
                                for log in logging_outputs)
            decoder_ppl = math.exp(min(decoder_loss / decoder_cnt_t, 100))
            if decoder_ppl > 0:
                metrics.log_scalar(
                    'decoder_ppl', decoder_ppl, sample_size, round=5)

            acc_sentence_count = sum(log.get('acc_sentence_hit', 0) for log in logging_outputs)
            acc_sentence_count = acc_sentence_count / \
                sum(log.get('acc_sentence_cnt', 0) for log in logging_outputs)
            metrics.log_scalar('acc_sentence_percentage',
                               acc_sentence_count, sample_size, round=5)
                               


    @staticmethod
    def logging_outputs_can_be_summed(is_train) -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True

    def get_matrix_label(self, alignment_matrix_dataset, bsz):

        align_matrix_mask_padding = alignment_matrix_dataset.ne(self.mask_id)
        align_matrix_position_mask_padding = alignment_matrix_dataset.gt(self.padding_idx)
        is_inferred_align = (align_matrix_position_mask_padding.sum(-1)>0)
        align_matrix_position_mask_padding = is_inferred_align.unsqueeze(-1).repeat_interleave(align_matrix_mask_padding.shape[-1], dim = -1)
        # align_matrix_position_mask_padding = align_matrix_position_mask_padding.any(dim = -1)
        align_matrix_position_label = align_matrix_mask_padding & align_matrix_position_mask_padding
        return align_matrix_position_label

def cal_loss(pred_vals: torch.Tensor, true_vals: torch.Tensor, norm=2):
    pred_vals = pred_vals.flatten()
    if norm == 2:
        loss = torch.nn.MSELoss()   # easy to be divergent
    elif norm == 1:
        loss = torch.nn.L1Loss()
    loss_val = loss(pred_vals.float(), true_vals)
    # print(pred_vals.dtype)
    # print(true_vals.dtype)
    # exit()
    return loss_val

@register_loss("ReRPBartVF")
class ReRPBartVFLoss(ReRPBartLoss):
    def __init__(self, task):
        super().__init__(task)

    def forward(self, model, sample, reduce=True):
        def inner_forward(input_key='net_input', target_key='target'):
            src_tokens_dataset = sample['net_input']['src_tokens']
            pre_src_tokens_dataset = sample['net_input']['pre_src_tokens']
            value_fn = sample['net_input']['value_fn']
            # print(sample)
            # exit()
            # print('wz test src_tokens_dataset:', src_tokens_dataset.shape, pre_src_tokens_dataset.shape, value_fn.shape)
            # print('wz test src_tokens_dataset2:', src_tokens_dataset[0])
            # RMK: here!
            # BARTModel()
            # try:
            #     value_fn = sample['net_input']['decoder_src_tokens']
            # except KeyError:
            #     print(type(model))
            #     print(sample)
            #     exit()
            # torch.set_printoptions(profile = "full")
            
            logits_decoder, extra = model(
                src_tokens = src_tokens_dataset, prev_output_tokens = pre_src_tokens_dataset,)

            loss = torch.tensor(0.0)
            # print('wz test loss2. ')
            logging_output = {
                "sample_size": 1,
                "bsz": sample[input_key]['src_tokens'].size(0),
                "seq_len": sample[input_key]['src_tokens'].size(1) * sample[input_key]['src_tokens'].size(0),
            }

            if logits_decoder is not None:
                # print('wz test src_tokens_dataset2:', logits_decoder.shape, value_fn.shape, logits_decoder[0])
                value_fn = sample[input_key]['value_fn']# .to(torch.float16)
                loss = cal_loss(logits_decoder.view(-1), value_fn, norm=1)

                # decoder_loss, decoder_hit, decoder_cnt, acc_sentence_count = get_loss(
                #     logits_decoder, value_fn, self.padding_idx)
                # loss = decoder_loss * self.args.decoder_loss
                logging_output = {
                    "sample_size": 1,
                    "bsz": sample[input_key]['src_tokens'].size(0),
                    "seq_len": sample[input_key]['src_tokens'].size(1) * sample[input_key]['src_tokens'].size(0),
                    "loss": loss.data,
                    # "decoder_hit": decoder_hit.data,
                    # "decoder_cnt": decoder_cnt.data,
                    # "acc_sentence_hit": acc_sentence_count[0],
                    # "acc_sentence_cnt": acc_sentence_count[1],
                }


            logging_output['loss'] = loss.data
            return loss, 1, logging_output

        loss, sample_size, logging_output = inner_forward()
        return loss, sample_size, logging_output
