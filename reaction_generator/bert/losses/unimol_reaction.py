from cgi import test
import math
import torch
import torch.nn.functional as F
from unicore import metrics, utils
from unicore.losses import UnicoreLoss, register_loss
import torch.nn as nn
import torch.distributed as dist

from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score, recall_score, f1_score

@register_loss("unimol_reaction")
class UniMolReactionLoss(UnicoreLoss):
    def __init__(self, task):
        super().__init__(task)
        self.padding_idx = task.dictionary.pad()
        # 6.312581655060595 3.3899264663911888
        self.dist_mean = 6.312581655060595
        self.dist_std = 3.3899264663911888

    def forward(self, model, sample, reduce=True):
        def inner_forward(input_key='net_input', target_key='target'):
            # masked_tokens = sample[target_key]['encoder'].ne(self.padding_idx)
            masked_tokens = sample['net_input']['src_tokens'].ne(self.padding_idx)
            sample_size = masked_tokens.long().sum()
            src_tokens = sample['net_input']['src_tokens']
            decoder_src_tokens = sample['net_input']['decoder_src_tokens']
            # reaction_type = sample['net_input']['reaction_type']

            reg_label = sample['net_input']['reg_label']
            reg_spe_token = sample['net_input']['reg_spe_token']
            reg_num_token = sample['net_input']['reg_num_token']
            tar_spe_token = sample['net_input']['tar_spe_token']
            tar_num_token = sample['net_input']['tar_num_token']
            reaction_point = sample['net_input']['reaction_point']
            rc_product = sample['net_input']['rc_product']
            class_label = sample['net_input']['reaction_type']
            pro_macc_fp = sample['net_input']['pro_macc_fp']
            pro_ecfp_fp = sample['net_input']['pro_ecfp_fp']

            atom_score, classifier_logits, logits_encoder, logits_decoder, cl_out, vae_kl_loss = model(src_tokens, decoder_src_tokens, class_label, masked_tokens=masked_tokens, pro_macc_fp = pro_macc_fp, pro_ecfp_fp = pro_ecfp_fp, 
            reg_label = reg_label, reg_spe_token = reg_spe_token, reg_num_token = reg_num_token, tar_spe_token = tar_spe_token, tar_num_token = tar_num_token, rc_product = rc_product, )
            loss = torch.tensor(0.0)

            logging_output = {
                "sample_size": 1,
                "bsz": sample['net_input']['src_tokens'].size(0),
                "seq_len": sample['net_input']['src_tokens'].size(1) * sample['net_input']['src_tokens'].size(0),
            }
            if logits_decoder is not None:
                # decoder_target = sample[target_key]["decoder"]
                decoder_target = sample['net_input']['decoder_src_tokens']
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
                # print('test decoder_pred: ', decoder_pred)               
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
                    "bsz": sample['net_input']['src_tokens'].size(0),
                    "seq_len": sample['net_input']['src_tokens'].size(1) * sample['net_input']['src_tokens'].size(0),
                    "decoder_loss": decoder_loss.data,
                    "decoder_hit": decoder_hit.data,
                    "decoder_cnt": decoder_cnt.data,
                }
            if self.args.rc_pred_loss_weight > 0:
                atom_center_loss = nn.BCELoss(reduction='mean')
                rc_product = rc_product[:, 1:] 
                atom_score = atom_score[:,1:]
                src_tokens = src_tokens[:,1:]
                encoder_mask = src_tokens.ne(self.padding_idx)
                rc_product = rc_product[encoder_mask]
                atom_score = atom_score[encoder_mask]
                rc_pred_loss = atom_center_loss(atom_score.view(-1), rc_product.view(-1).float())
                thres_set = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
                rc_auc_score = roc_auc_score(rc_product.view(-1).clone().detach().cpu().numpy(), atom_score.view(-1).clone().detach().cpu().numpy())
                loss += rc_pred_loss
                logging_output['rc_pred_loss'] = rc_pred_loss.data 
                logging_output['rc_auc_score'] = rc_auc_score 

                for thr in thres_set:
                    pre_sc, recall_sc, f1_sc = self.get_thre_choice(thr, atom_score, rc_product)
                    logging_output[str(thr)+'_precision_score'] = pre_sc 
                    logging_output[str(thr)+'_recall_score'] = recall_sc 
                    logging_output[str(thr)+'_f1_score'] = f1_sc

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


            logging_output['loss'] = loss.data
            return loss, 1, logging_output, cl_out

        loss, sample_size, logging_output, cls_repr = inner_forward()
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
    def get_thre_choice(self, thres_point, atom_score, rc_product):
        atom_pred = atom_score.gt(thres_point)
        atom_pred = atom_pred.view(-1).clone().detach().cpu().numpy()
        rc_product = rc_product.view(-1).clone().detach().cpu().numpy()
        precision_score_point = precision_score(rc_product, atom_pred)
        recall_score_point = recall_score(rc_product, atom_pred)
        f1_score_point = f1_score(rc_product, atom_pred)
        return precision_score_point, recall_score_point, f1_score_point

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

        ce_loss = sum(log.get('ce_loss', 0) for log in logging_outputs)
        if ce_loss > 0:
            metrics.log_scalar('ce_loss', ce_loss / sample_size , sample_size, round=5) 

            class_acc = sum(log.get('class_hit', 0) for log in logging_outputs) / sum(log.get('class_cnt', 1) for log in logging_outputs)
            if class_acc > 0:
                metrics.log_scalar('class_acc', class_acc , sample_size, round=5)
        rc_pred_loss = sum(log.get('rc_pred_loss', 0) for log in logging_outputs)
        if rc_pred_loss > 0:
            metrics.log_scalar('rc_pred_loss', rc_pred_loss / sample_size , sample_size, round=5) 
        rc_auc_score = sum(log.get('rc_auc_score', 0) for log in logging_outputs)
        if rc_auc_score > 0:
            metrics.log_scalar('rc_auc_score', rc_auc_score / sample_size , sample_size, round=5) 
        
        thres_set = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        for thr in thres_set:
            precision_score = sum(log.get(str(thr)+'_precision_score', 0) for log in logging_outputs)
            recall_score = sum(log.get(str(thr)+'_recall_score', 0) for log in logging_outputs)
            f1_score = sum(log.get(str(thr)+'_f1_score', 0) for log in logging_outputs)
            if precision_score > 0 and split == 'valid':
                metrics.log_scalar(str(thr)+'_precision_score', precision_score / sample_size , sample_size, round=5) 
            if recall_score > 0 and split == 'valid':
                metrics.log_scalar(str(thr)+'_recall_score', recall_score / sample_size , sample_size, round=5) 
            if f1_score > 0 and split == 'valid':
                metrics.log_scalar(str(thr)+'_f1_score', f1_score / sample_size , sample_size, round=5) 

    @staticmethod
    def logging_outputs_can_be_summed(is_train) -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True