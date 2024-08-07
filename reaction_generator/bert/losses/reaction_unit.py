import math
import torch
import torch.nn.functional as F
from unicore import metrics, utils
from unicore.losses import UnicoreLoss, register_loss
import torch.nn as nn
import torch.distributed as dist


@register_loss("reaction_unit")
class ReactionUnitLoss(UnicoreLoss):
    def __init__(self, task):
        super().__init__(task)
        self.padding_idx = task.dictionary.pad()
        # 6.312581655060595 3.3899264663911888
        self.dist_mean = 6.312581655060595
        self.dist_std = 3.3899264663911888

    def forward(self, model, sample, reduce=True):
        def inner_forward(input_key='net_input', target_key='target'):
            # masked_tokens = sample[target_key]['encoder'].ne(self.padding_idx)
            masked_tokens = sample['net_input']['reverse_src_dataset'].ne(self.padding_idx)
            sample_size = masked_tokens.long().sum()
            src_tokens = sample['net_input']['src_tokens']
            reverse_src_tokens = sample['net_input']['reverse_src_dataset']            
            decoder_src_tokens = sample['target']['reverse_tgt_dataset']
            class_label = sample['net_input']['reaction_type']

            # reaction_type = sample['net_input']['reaction_type']
            # logits_encoder, logits_decoder, cl_out, vae_kl_loss = model(src_tokens, decoder_src_tokens, reaction_type, masked_tokens=masked_tokens, )
            classifier_logits, logits_decoder, vq1, vq2 = model(src_tokens, reverse_src_tokens, decoder_src_tokens, masked_tokens=masked_tokens, )
            # print('test class_label: ', class_label, classifier_logits)

            loss = torch.tensor(0.0)
            # logging_output = {
            #     "sample_size": 1,
            #     "bsz": sample[target_key]['encoder'].size(0),
            #     "seq_len": sample[target_key]['encoder'].size(1) * sample[target_key]['encoder'].size(0),
            # }
            logging_output = {
                "sample_size": 1,
                "bsz": sample['net_input']['reverse_src_dataset'].size(0),
                "seq_len": sample['net_input']['reverse_src_dataset'].size(1) * sample['net_input']['reverse_src_dataset'].size(0),
            }

            # decoder loss 
            if logits_decoder is not None:
                # decoder_target = sample[target_key]["decoder"]
                decoder_target = sample['target']['reverse_tgt_dataset']
                # decode_tokens2 = decoder_target.ne(self.padding_idx)
                # print('test logits_decoder: ', decoder_target[decode_tokens2].shape, logits_decoder.shape)
                decoder_target = decoder_target[:, 1:]
                logits_decoder = logits_decoder[:, :-1]
                decode_tokens = decoder_target.ne(self.padding_idx)
                decoder_sample_size = decode_tokens.long().sum()
 
                decoder_loss = F.nll_loss(
                    F.log_softmax(logits_decoder[decode_tokens],dim=-1, dtype=torch.float32),
                    decoder_target[decode_tokens].view(-1),
                    ignore_index=self.padding_idx,
                    reduction='mean',
                )
                decoder_pred = torch.argmax(logits_decoder[decode_tokens], dim=-1)
                # print('test decoder_target: ', src_tokens, reverse_src_tokens, decoder_target)               
                decoder_hit = (decoder_pred == decoder_target[decode_tokens]).long().sum()
                decoder_cnt = decoder_sample_size
                # print('test decoder_sample_size: ', decoder_sample_size)
                loss = decoder_loss * self.args.decoder_loss
                # logging_output = {
                #     "sample_size": 1,
                #     "bsz": sample[target_key]['encoder'].size(0),
                #     "seq_len": sample[target_key]['encoder'].size(1) * sample[target_key]['encoder'].size(0),
                #     "decoder_loss": decoder_loss.data,
                #     "decoder_hit": decoder_hit.data,
                #     "decoder_cnt": decoder_cnt.data,
                # }
                logging_output = {
                    "sample_size": 1,
                    "bsz": sample['net_input']['reverse_src_dataset'].size(0),
                    "seq_len": sample['net_input']['reverse_src_dataset'].size(1) * sample['net_input']['reverse_src_dataset'].size(0),
                    "decoder_loss": decoder_loss.data,
                    "decoder_hit": decoder_hit.data,
                    "decoder_cnt": decoder_cnt.data,
                }

            # category loss 
            if self.args.ce_loss_weight > 0:
                class_label = class_label - 1
                cat_loss = F.nll_loss(
                    F.log_softmax(classifier_logits, dim=-1, dtype=torch.float32),
                    class_label,
                    reduction='mean',
                )
                # cat_loss = F.cross_entropy(classifier_logits, class_label)
                loss += self.args.ce_loss_weight * cat_loss
                logging_output['ce_loss'] = cat_loss.data 
                # print('test self.args.ce_loss_weight: ', cat_loss.data, classifier_logits.shape, class_label.shape, self.args.ce_loss_weight)

            # mse loss 
            if self.args.mse_loss_weight > 0:
                loss_cosine = 1 - F.cosine_similarity(vq1, vq2, dim=1)
                loss_cosine = torch.mean(loss_cosine)
                loss += self.args.mse_loss_weight * loss_cosine
                logging_output['loss_cosine'] = loss_cosine.data 

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

        bsz = sum(log.get("bsz", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        seq_len = sum(log.get("seq_len", 0) for log in logging_outputs)
        metrics.log_scalar(
            "loss", loss_sum / sample_size, sample_size, round=5
        )
        metrics.log_scalar(
            "seq_len", seq_len / bsz, 1, round=3
        )
        decoder_loss = sum(log.get('decoder_loss', 0) for log in logging_outputs)

        if decoder_loss > 0:
            metrics.log_scalar('decoder_loss', decoder_loss / sample_size , sample_size, round=5)
            decoder_acc = sum(log.get('decoder_hit', 0) for log in logging_outputs) / sum(log.get('decoder_cnt', 1) for log in logging_outputs)
            if decoder_acc > 0:
                metrics.log_scalar('decoder_acc', decoder_acc , sample_size, round=5)
            
            decoder_cnt_t = sum(log.get('decoder_cnt', 1) for log in logging_outputs)
            decoder_ppl = math.exp(min(decoder_loss / decoder_cnt_t, 100))
            if decoder_ppl > 0:
                metrics.log_scalar('decoder_ppl', decoder_ppl , sample_size, round=5)

        loss_cosine = sum(log.get('loss_cosine', 0) for log in logging_outputs)
        if loss_cosine > 0:
            metrics.log_scalar('loss_cosine', loss_cosine / sample_size , sample_size, round=5)  

        ce_loss = sum(log.get('ce_loss', 0) for log in logging_outputs)
        if ce_loss > 0:
            metrics.log_scalar('ce_loss', ce_loss / sample_size , sample_size, round=5)  
  
    @staticmethod
    def logging_outputs_can_be_summed(is_train) -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True