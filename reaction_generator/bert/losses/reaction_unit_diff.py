import math
import torch
import torch.nn.functional as F
from unicore import metrics, utils
from unicore.losses import UnicoreLoss, register_loss
import torch.nn as nn
import torch.distributed as dist


@register_loss("reaction_unit_diff")
class ReactionUnitDiffLoss(UnicoreLoss):
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
            reverse_tgt_tokens = sample['target']['reverse_tgt_dataset']
            class_label = sample['net_input']['reaction_type']
            rec_size = sample['net_input']['rec_size']
            tar_size = sample['net_input']['tar_size']
            delta_reaction_size = rec_size - tar_size
            # print('test delta_reaction_size: ', tar_size.shape, delta_reaction_size.shape, tar_size, delta_reaction_size)
            # reaction_type = sample['net_input']['reaction_type']
            # logits_encoder, logits_decoder, cl_out, vae_kl_loss = model(src_tokens, decoder_src_tokens, reaction_type, masked_tokens=masked_tokens, )
            length_class_logits, classifier_logits, logits_decoder, vq1, vq2, mid_x_decoder_list = model(src_tokens, reverse_src_tokens, reverse_tgt_tokens, masked_tokens=masked_tokens, )
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
                # if self.args.auto_regressive:
                # decoder_target = decoder_target[:, 1:]
                # logits_decoder = logits_decoder[:, :-1]
                decoder_target = decoder_target[:, :]
                logits_decoder = logits_decoder[:, :]

                decode_tokens = decoder_target.ne(self.padding_idx)
                decoder_sample_size = decode_tokens.long().sum()
                decoder_loss = F.nll_loss(
                    F.log_softmax(logits_decoder[decode_tokens],dim=-1, dtype=torch.float32),
                    decoder_target[decode_tokens].view(-1),
                    ignore_index=self.padding_idx,
                    reduction='mean',
                )
                # else:
                #     decoder_target = decoder_target
                #     logits_decoder = logits_decoder
                #     decode_tokens = decoder_target.ne(self.padding_idx)
                #     decoder_sample_size = decode_tokens.long().sum()
                #     decoder_loss = F.nll_loss(
                #         F.log_softmax(logits_decoder.view(-1, logits_decoder.shape[-1]),dim=-1, dtype=torch.float32),
                #         decoder_target.view(-1),
                #         ignore_index=self.padding_idx,
                #         reduction='mean',
                #     )
                decoder_pred = torch.argmax(logits_decoder, dim=-1)    
                decoder_hit = (decoder_pred[decode_tokens] == decoder_target[decode_tokens]).long().sum()
                decoder_cnt = decoder_sample_size
                loss = decoder_loss * self.args.decoder_loss
                logging_output = {
                    "sample_size": 1,
                    "bsz": sample['net_input']['reverse_src_dataset'].size(0),
                    "seq_len": sample['net_input']['reverse_src_dataset'].size(1) * sample['net_input']['reverse_src_dataset'].size(0),
                    "decoder_loss": decoder_loss.data,
                    "decoder_hit": decoder_hit.data,
                    "decoder_cnt": decoder_cnt.data,
                }
                # print('test decoder_pred: ', decoder_pred.shape, decoder_target.shape, decoder_pred, decoder_target)
            #length loss

            ###mid_decoder_term attention
            if len(mid_x_decoder_list) > 0:
                mid_decoder_loss = 0.0
                for x_decoer_logits in mid_x_decoder_list:
                    decoder_target_mid = sample['target']['reverse_tgt_dataset']
                    decode_tokens_mid = decoder_target_mid.ne(self.padding_idx)
                    mid_decoder_loss += F.nll_loss(
                        torch.log(x_decoer_logits[decode_tokens_mid]),
                        decoder_target_mid[decode_tokens_mid].view(-1),
                        ignore_index=self.padding_idx,
                        reduction='mean',
                    )
                
                mid_decoer_pred = torch.argmax(x_decoer_logits, dim=-1)   
                mid_decoder_hit = (mid_decoer_pred[decode_tokens_mid] == decoder_target_mid[decode_tokens_mid]).long().sum()
                loss += mid_decoder_loss
                logging_output['mid_decoder_loss'] = mid_decoder_loss.data 
                logging_output['mid_decoder_hit'] = mid_decoder_hit               

            if self.args.length_loss_weight > 0:
                delta_reaction_size2 = delta_reaction_size + 10
                # cat_loss = F.nll_loss(
                #     F.log_softmax(classifier_logits, dim=-1, dtype=torch.float32),
                #     class_label,
                #     reduction='mean',
                # )
                length_loss = F.cross_entropy(length_class_logits, delta_reaction_size2)
                loss += self.args.length_loss_weight * length_loss
                logging_output['length_loss'] = length_loss.data 
                
                delta_length_pred = torch.argmax(length_class_logits, dim=-1) 
                length_batch_num, length_class_num = length_class_logits.shape
                length_list = torch.tensor([[i for i in range(length_class_num)] for j in  range(length_batch_num)], dtype = torch.float32).to(length_class_logits)
                
                delta_length_pred_weight = torch.round(torch.sum(torch.softmax(length_class_logits, dim = -1) * length_list, dim = -1)).type_as(delta_length_pred)
                delta_length_pred_weight = delta_length_pred_weight - 10
                delta_length_pred = delta_length_pred - 10
                pred_rec_size = tar_size + delta_length_pred
                length_hit = (delta_length_pred == delta_reaction_size).long().sum()
                length_weight_hit = (delta_length_pred_weight == delta_reaction_size).long().sum()
                length_cnt = delta_reaction_size.shape[0]              
                logging_output['length_hit'] = length_hit.data 
                logging_output['length_weight_hit'] = length_weight_hit.data 
                logging_output['length_cnt'] = length_cnt 
           
            # zero punishment 

            # category loss 
            if self.args.ce_loss_weight > 0:
                class_label = class_label - 1
                # cat_loss = F.nll_loss(
                #     F.log_softmax(classifier_logits, dim=-1, dtype=torch.float32),
                #     class_label,
                #     reduction='mean',
                # )
                cat_loss = F.cross_entropy(classifier_logits, class_label)
                loss += self.args.ce_loss_weight * cat_loss
                logging_output['ce_loss'] = cat_loss.data 

                classifier_logits_pred = torch.argmax(classifier_logits, dim=-1)    
                class_hit = (classifier_logits_pred == class_label).long().sum()
                class_cnt = class_label.shape[0]              
                logging_output['class_hit'] = class_hit.data 
                logging_output['class_cnt'] = class_cnt 
                # class_pred = torch.argmax(classifier_logits, dim=-1) 
                # print('test class_pred: ', class_pred.shape, class_pred)

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

            # decoder_acc_add_padding = sum(log.get('decoder_hit', 0) for log in logging_outputs) / sum(log.get('seq_len', 1) for log in logging_outputs)
            # if decoder_acc_add_padding > 0:
            #     metrics.log_scalar('decoder_acc_add_padding', decoder_acc_add_padding , sample_size, round=5)

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
            class_acc = sum(log.get('class_hit', 0) for log in logging_outputs) / sum(log.get('class_cnt', 1) for log in logging_outputs)
            if class_acc > 0:
                metrics.log_scalar('class_acc', class_acc , sample_size, round=5)


        mid_decoder_loss = sum(log.get('mid_decoder_loss', 0) for log in logging_outputs)
        if mid_decoder_loss > 0:
            metrics.log_scalar('mid_decoder_loss', mid_decoder_loss / sample_size , sample_size, round=5)             
            mid_layer_acc = sum(log.get('mid_decoder_hit', 0) for log in logging_outputs) / sum(log.get('decoder_cnt', 1) for log in logging_outputs)
            if mid_layer_acc > 0:
                metrics.log_scalar('mid_layer_acc', mid_layer_acc , sample_size, round=5) 


        length_loss = sum(log.get('length_loss', 0) for log in logging_outputs)
        if length_loss > 0:
            metrics.log_scalar('length_loss', length_loss / sample_size , sample_size, round=5)  
            length_acc = sum(log.get('length_hit', 0) for log in logging_outputs) / sum(log.get('length_cnt', 1) for log in logging_outputs)
            length_weight_acc = sum(log.get('length_weight_hit', 0) for log in logging_outputs) / sum(log.get('length_cnt', 1) for log in logging_outputs)
            if length_acc > 0:
                metrics.log_scalar('length_acc', length_acc , sample_size, round=5) 
            if length_weight_acc > 0:
                metrics.log_scalar('length_weight_acc', length_weight_acc , sample_size, round=5) 
    @staticmethod
    def logging_outputs_can_be_summed(is_train) -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True