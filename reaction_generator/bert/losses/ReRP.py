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


@register_loss("ReRP")
class ReRPLoss(UnicoreLoss):
    def __init__(self, task):
        super().__init__(task)
        self.padding_idx = task.dictionary.pad()
        self.mask_id = -1
        # self.epoch = task.epoch

    def forward(self, model, sample, reduce=True):
        def inner_forward(input_key='net_input', target_key='target'):
            class_label = sample['net_input']['reaction_type']
            # aug_index_dataset = sample['net_input']['aug_index_dataset']
            src_tokens_dataset = sample['net_input']['src_tokens']
            # new_src_tokens_dataset = sample['net_input']['smiles_src_tokens']
            decoder_target = sample['net_input']['decoder_src_tokens']
            # torch.set_printoptions(profile = "full")

            logits_encoder, logits_decoder, cl_out, vae_kl_loss, classifier_logits, atom_score, atten_score_list = model(
                **sample[input_key], features_only=True)


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

            if self.args.atten_score:
                bsz = decoder_target.shape[0]
                alignment_matrix_dataset = sample['net_input']['align_matrix_dataset']

                atten_score_matrix_l = atten_score_list[-1]
                bsz_head, tgt_len, src_len = atten_score_matrix_l.size()
                head_num = bsz_head//bsz
                atten_score_matrix = atten_score_list[-1].view(bsz, head_num, tgt_len, src_len).transpose(0,1).contiguous()
                align_matrix_mask_padding = self.get_matrix_label(alignment_matrix_dataset, bsz)
                attention_align_loss = 0.0

                for i1 in range(head_num//2):
                    att_score_m = atten_score_matrix[i1,:,:,:]
                    align_matrix_loss = nn.BCELoss(reduction='mean')
                    atten_score_m_ = att_score_m[align_matrix_mask_padding].view(-1)
                    atten_label_m_ = alignment_matrix_dataset[align_matrix_mask_padding].view(-1).half()

                    attention_align_loss += get_align_loss(
                                        atten_score_m_,
                                        atten_label_m_)
                loss += attention_align_loss * self.args.attention_align_loss_weight
                logging_output['attention_align_loss'] = attention_align_loss.data 
        
            if self.args.rc_pred_loss_weight > 0:
                atom_center_loss = nn.BCELoss(reduction='mean')
                can_rc_pro_dataset = sample['net_input']['can_rc_pro_dataset']
                rc_product = can_rc_pro_dataset[:, 1:] 
                atom_score = atom_score[:,1:]
                src_tokens = src_tokens_dataset[:,1:]
                encoder_mask = src_tokens.ne(self.padding_idx)
                rc_product = rc_product[encoder_mask]
                atom_score = atom_score[encoder_mask]
                rc_pred_loss = atom_center_loss(atom_score.view(-1), rc_product.view(-1).half())

                loss += rc_pred_loss * self.args.rc_pred_loss_weight
                rc_auc_score = roc_auc_score(rc_product.view(-1).clone().detach().cpu().numpy(), atom_score.view(-1).clone().detach().cpu().numpy())
                logging_output['rc_pred_loss'] = rc_pred_loss.data 
                logging_output['rc_auc_score'] = rc_auc_score 
                
            # category loss 
            if self.args.ce_loss_weight > 0:
                class_label = class_label - 1

                cat_loss = F.cross_entropy(classifier_logits, class_label)
                loss += self.args.ce_loss_weight * cat_loss
                logging_output['ce_loss'] = cat_loss.data 

                classifier_logits_pred = torch.argmax(classifier_logits, dim=-1) 
                value_class, index_class = classifier_logits.topk(2, dim=-1, largest = True, sorted = True)   

                class_hit = (classifier_logits_pred == class_label).long().sum()
                label_hit_topk = torch.zeros_like(class_label)
                for idx, iv in enumerate(label_hit_topk):
                    label_hit_topk[idx] = (class_label[idx] in index_class[idx])
                class_hit_topk = label_hit_topk.long().sum()
    
                class_cnt = class_label.shape[0]              
                logging_output['class_hit'] = class_hit.data 
                logging_output['class_hit_topk'] = class_hit_topk.data 
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
                               
        ce_loss = sum(log.get('ce_loss', 0) for log in logging_outputs)
        if ce_loss > 0:
            metrics.log_scalar('ce_loss', ce_loss / sample_size , sample_size, round=5) 

            class_acc = sum(log.get('class_hit', 0) for log in logging_outputs) / sum(log.get('class_cnt', 1) for log in logging_outputs)
            if class_acc > 0:
                metrics.log_scalar('class_acc', class_acc , sample_size, round=5)

            class_acc_topk = sum(log.get('class_hit_topk', 0) for log in logging_outputs) / sum(log.get('class_cnt', 1) for log in logging_outputs)
            if class_acc_topk > 0:
                metrics.log_scalar('class_acc_topk', class_acc_topk , sample_size, round=5)           

        rc_pred_loss = sum(log.get('rc_pred_loss', 0) for log in logging_outputs)
        if rc_pred_loss > 0:
            metrics.log_scalar('rc_pred_loss', rc_pred_loss / sample_size , sample_size, round=5) 
        rc_auc_score = sum(log.get('rc_auc_score', 0) for log in logging_outputs)
        if rc_auc_score > 0:
            metrics.log_scalar('rc_auc_score', rc_auc_score / sample_size , sample_size, round=5) 
        attention_align_loss = sum(log.get('attention_align_loss', 0) for log in logging_outputs)
        if attention_align_loss > 0:
            metrics.log_scalar('attention_align_loss', attention_align_loss / sample_size , sample_size, round=5) 
        

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