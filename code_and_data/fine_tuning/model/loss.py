import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor as T
import copy

class AttentionLoss(nn.Module):
    def __init__(self, config):
        super(AttentionLoss, self).__init__()
        self.sep_attn_peak = config.getboolean('attention_loss', 'separate_attention_peak')

    def forward(self, scores_list):
        overall_norm = 0.0
        sub_norm = 0.0

        for scores in scores_list:
            overall_norm += sum(torch.norm(scores, dim=-1))

            if self.sep_attn_peak:
                tmp = torch.ones_like(scores[0])
                for i in range(len(scores)):
                    tmp *= scores[i]
                sub_norm += torch.norm(tmp)

        loss1 = 1 - overall_norm / sum([len(s) for s in scores_list])  # maximize the norm of the attention scores

        if self.sep_attn_peak:
            loss2 = sub_norm  # minimize the norm of the product of attention scores for each document.
            loss = loss1 + loss2
            return [loss, loss1, loss2]

        return [loss1]


class ContrastiveLoss(nn.Module):
    def __init__(self, config):
        super(ContrastiveLoss, self).__init__()
        self.config = config
        self.sim_fct = self.cosine_sim if config.get('contra_loss', 'sim_fct') == 'cos' else self.dot_product
        self.loss_fct = nn.CrossEntropyLoss(reduce=False)
        self.temperature = config.getfloat('contra_loss', 'temperature')

    def forward(self, query_list, output_list, value_list, scores_list=None, truth_list = None):

        neg_query_key = self.config.get('contra_loss', 'neg_query_key')
        neg_value_key = self.config.get('contra_loss', 'neg_value_key')

        neg_value_list = value_list[neg_value_key]
        neg_query_list = query_list[neg_query_key]
        query_list = query_list['single']


        if self.config.getboolean('contra_loss', 'unsupervise'):
            value_zero_mask = self.zero_mask_for_value(query_list, neg_value_list)
        else:
            added_supervised_target = self.target_for_supervised_value(query_list, neg_value_list, truth_list)

        # mask out all the attn negatives by default
        score_mask = self.zero_mask_for_attention_output(query_list, mode='all')  # = AP ALL mask
        attn_weights = self.gen_weights_for_attention_output(attention_scores_list=scores_list) # All 1 or weight


        # flatten embeddings
        query_list = torch.cat(query_list, dim=0) # M *768
        output_list = torch.cat(output_list, dim=0)# M *768 attention
        neg_query_list = torch.cat(neg_query_list, dim=0) # 2M*768
        neg_value_list = torch.cat(neg_value_list, dim=0) # 2N*768
        one_hot_target = torch.eye(len(query_list)).to(query_list.device)

        if self.config.getboolean('contra_loss', 'negatives_value'):
            if self.config.getboolean('contra_loss', 'unsupervise'): #unsupervised
                added_zero_target = torch.zeros(len(query_list), len(neg_value_list), device=query_list.device) # M * 2*N
                one_hot_target = torch.cat((one_hot_target, added_zero_target), dim=1) # M * (M+2*N)
                output_list = torch.cat((output_list, neg_value_list), dim=0)  #dim = 0 ;
                score_mask = torch.cat((score_mask, value_zero_mask), dim=1)   #dim = 1 :
                attn_weights = torch.cat((attn_weights,
                                      torch.ones(len(attn_weights), len(neg_value_list), device=query_list.device)),
                                     dim=1)
            else:
                added_zero_mask = torch.zeros(len(query_list), len(neg_value_list)) # M * 2*N
                added_supervised_target = added_supervised_target.to(query_list.device)
                one_hot_target = torch.cat((one_hot_target, added_supervised_target), dim=1) # M * (M+2*N)
                output_list = torch.cat((output_list, neg_value_list), dim=0)  #dim = 0 ;
                score_mask = torch.cat((score_mask, added_zero_mask), dim=1)   #dim = 1 :
                attn_weights = torch.cat((attn_weights,
                                      torch.ones(len(attn_weights), len(neg_value_list), device=query_list.device)),
                                     dim=1)

        sim_score = self.sim_fct(query_list, output_list) / self.temperature # M*(1)*d_v   (1)* (M+2*M+2*N)*d_v

        sim_score += score_mask.to(query_list.device)
        if not self.config.getboolean('positive_weight', 'use'):
            attn_weights = None

        loss = self.one_hot_cross_entropy_loss(sim_score, one_hot_target, weights=attn_weights, reduction='mean')

        return loss

    def target_for_supervised_value(self,query_list, neg_value_list, truth_list):
        q_nums_list = [len(q) for q in query_list]
        v_nums_list = [len(v) for v in neg_value_list]

        if self.config.getboolean('train', 'multi_gpu'):
            print("q",q_nums_list)
            print("v",v_nums_list)
            print(truth_list)

        target_for_supervised_value = torch.zeros(sum(q_nums_list), sum(v_nums_list)) #M * 2*N

        q_start, v_start = 0, 0

        if self.config.getboolean('train', 'multi_gpu'):
            for idx in range(truth_list.shape[0]):
                q_num, v_num = q_nums_list[idx], v_nums_list[idx]
                print("before cutting", truth_list[idx])
                print(v_num)
                print(truth_list[idx])
                #truth_list[idx] = truth_list[idx][:v_num]
                print("after cutting", truth_list[idx])
                target_for_supervised_value[q_start: q_start+q_num, v_start:v_start+v_num] = truth_list[idx].view(torch.ones(q_num,v_num).shape)
                q_start += q_num
                v_start += v_num
        else:
            for idx in range(len(q_nums_list)):
                q_num, v_num = q_nums_list[idx], v_nums_list[idx]
                target_for_supervised_value[q_start: q_start + q_num, v_start:v_start + v_num] = torch.FloatTensor(truth_list[idx]).view(
                    torch.ones(q_num, v_num).shape)
                q_start += q_num
                v_start += v_num

        return target_for_supervised_value

    def gen_weights_for_attention_output(self, attention_scores_list):

        source = self.config.get('positive_weight', 'source')
        scores_list = [score[source] for score in attention_scores_list]

        if self.config.get('positive_weight', 'type') == 'sum':
            weights_list = [torch.sum(scores, dim=-1) for scores in scores_list]
        elif self.config.get('positive_weight', 'type') == 'norm':
            weights_list = [torch.norm(scores, dim=-1) for scores in scores_list] #[M*1]
        else:
            raise NotImplementedError

        # weights_list 1* M

        if self.config.get('positive_weight', 'normalize') == 'none':
            weights = torch.cat(weights_list, dim=0)  # []M

        elif self.config.get('positive_weight', 'normalize') == 'hard':
            if self.config.get('positive_weight', 'range') == 'in_case':
                weights = torch.cat([w/torch.sum(w, dim=-1) for w in weights_list], dim=0)
            else:
                weights = torch.cat(weights_list, dim=0)
                weights = weights / torch.sum(weights, dim=-1)
        elif self.config.get('positive_weight', 'normalize') == 'soft':
            if self.config.get('positive_weight', 'range') == 'in_case':
                weights = torch.cat([torch.softmax(w, dim=-1) for w in weights_list], dim=0)
            else:
                weights = torch.cat(weights_list, dim=0)
                weights = torch.softmax(weights, dim=-1)
        else:
            raise NotImplementedError

        w_nums = [len(w) for w in weights_list] #[[M]

        if self.config.get('positive_weight', 'range') == 'in_case':
            weight_mask = torch.zeros(sum(w_nums), sum(w_nums), device=weights.device)
            start = 0
            for w_num in w_nums:
                weight_mask[start: start + w_num, start: start + w_num] = 1
                start += w_num
            ones_mask = torch.ones_like(weight_mask)
            weights = weights * weight_mask + (ones_mask - weight_mask)
        else:
            weight_mask = torch.ones(sum(w_nums), sum(w_nums), device=weights.device) # M*M
            weights = weights * weight_mask

        if not self.config.getboolean('positive_weight', 'use'):
            weights = torch.ones(sum(w_nums), sum(w_nums), device=weights.device)

        return weights

    @staticmethod
    def dot_product(emb1, emb2):
        return torch.matmul(emb1, emb2.permute(1, 0))

    @staticmethod
    def cosine_sim(emb1, emb2):
        return torch.cosine_similarity(emb1.unsqueeze(1), emb2.unsqueeze(0), dim=-1)

    @staticmethod
    def zero_mask_for_attention_output(query_list, mode='all', use_pos=False) -> torch.Tensor:
        if mode == "all":
            query_embedding = torch.cat(query_list, dim=0) #11, 768
            batch_size = query_embedding.size(0)
            if use_pos:
                non_eyes_mask = torch.ones(batch_size, batch_size) - torch.eye(batch_size)
            else:
                non_eyes_mask = torch.ones(batch_size, batch_size)
            attention_zero_mask = - non_eyes_mask * 1e12
        elif mode == "hard":
            q_nums = [len(q) for q in query_list]
            attention_zero_mask = torch.zeros(sum(q_nums), sum(q_nums))
            q_start = 0
            for q_num in q_nums:
                attention_zero_mask[q_start: q_start+q_num, q_start: q_start+q_num] = 1
                q_start += q_num
            # set diagonals to 0, diagonals are positives, do not mask
            if use_pos:
                for i in range(sum(q_nums)):
                    attention_zero_mask[i, i] = 0
            attention_zero_mask = - attention_zero_mask * 1e12
        elif mode == 'none':
            q_nums = [len(q) for q in query_list]
            attention_zero_mask = torch.zeros(sum(q_nums), sum(q_nums))
            if not use_pos:
                for i in range(sum(q_nums)):
                    attention_zero_mask[i, i] = 1
                attention_zero_mask = - attention_zero_mask * 1e12
        else:
            raise NotImplementedError

        return attention_zero_mask

    @staticmethod
    def zero_mask_for_value(query_list, value_list) -> torch.Tensor: # M 768 ;2*N 768
        q_nums = [len(q) for q in query_list]
        v_nums = [len(v) for v in value_list]
        value_zero_mask = torch.zeros(sum(q_nums), sum(v_nums)) #M * 2*N

        q_start, v_start = 0, 0
        for idx in range(len(q_nums)):
            q_num, v_num = q_nums[idx], v_nums[idx]
            value_zero_mask[q_start: q_start+q_num, v_start:v_start+v_num] = 1
            q_start += q_num
            v_start += v_num

        return - value_zero_mask * 1e12

    def zero_mask_for_query(self, query_list, neg_query_list, use_pos=True) -> torch.Tensor:
        q_nums = [len(q) for q in query_list]
        nq_nums = [len(nq) for nq in neg_query_list]
        if self.config.get('contra_loss', 'neg_query_key') == 'single':
            query_zero_mask = - torch.eye(sum(q_nums)) * 1e12
        else:
            query_zero_mask = torch.zeros(sum(q_nums), sum(nq_nums))
            for idx in range(len(query_zero_mask)):
                query_zero_mask[idx, 2 * idx] = 1
            if not use_pos:
                for idx in range(len(query_zero_mask)):
                    query_zero_mask[idx, 2 * idx + 1] = 1

            query_zero_mask = - query_zero_mask * 1e12

        return query_zero_mask

    @staticmethod

    def one_hot_cross_entropy_loss(logits, target, weights=None, reduction='mean'):

        if weights is None:
            weights = torch.ones_like(logits)
        logits_exp = torch.exp(logits)
        logits_exp_w = logits_exp * weights

        pos = torch.sum(logits_exp_w * target, dim=1)
        neg = torch.sum(logits_exp_w * (1 - target), dim=1)
        loss_list = - torch.log(pos / (pos + neg))

        if reduction is None:
            loss = loss_list
        elif reduction == "mean":
            loss = torch.sum(loss_list) / len(loss_list)
        elif reduction == "sum":
            loss = torch.sum(loss_list)
        else:
            raise NotImplementedError

        return loss

class SimcseLoss(nn.Module):
    def __init__(self, config):
        super(SimcseLoss, self).__init__()
        self.config = config
        self.loss_fct = nn.CrossEntropyLoss()
        self.sim_fct = self.cosine_sim if config.get('simcse_loss', 'sim_fct') == 'cos' else self.dot_product
        self.temperature = config.getfloat('simcse_loss', 'temperature')

    def forward(self, query_list, value_list):
        loss = 0.0
        if self.config.getboolean('simcse_loss', 'negatives_cross'):
            sep = len(query_list) // 2

            loss += self.vanilla_simcse(torch.cat(query_list[:sep] + value_list[sep:], dim=0))
            loss += self.vanilla_simcse(torch.cat(query_list[sep:] + value_list[:sep], dim=0))

        if self.config.getboolean('simcse_loss', 'negatives_parallel'):
            query_list = torch.cat(query_list, dim=0)
            value_list = torch.cat(value_list, dim=0)

            if not self.config.getboolean('simcse_loss', 'negatives_parallel_single'):
                loss += self.vanilla_simcse(query_list)

            loss += self.vanilla_simcse(value_list)

        return loss

    def vanilla_simcse(self, batch_embeddings):

        batch_size = batch_embeddings.size(0)
        ground_truth = self.generate_positive_label(batch_size).to(batch_embeddings.device)

        sim_score = self.sim_fct(batch_embeddings, batch_embeddings) / self.temperature
        zero_mask = - torch.eye(batch_size).to(sim_score.device) * 1e12  # mask out the diagonal scores (self-self)
        sim_score += zero_mask

        loss = self.loss_fct(sim_score, ground_truth)
        return loss

    @staticmethod
    def dot_product(emb1, emb2):
        return torch.matmul(emb1, emb2.permute(1, 0))

    @staticmethod
    def cosine_sim(emb1, emb2):
        return torch.cosine_similarity(emb1.unsqueeze(1), emb2.unsqueeze(0), dim=-1)


    def generate_positive_label(self, batch_size):
        even = torch.arange(0, batch_size, 2, dtype=torch.long).unsqueeze(1)
        odd = torch.arange(1, batch_size, 2, dtype=torch.long).unsqueeze(1) #103254

        # if self.config.getboolean('train', 'multi_gpu'):
        #     if even.shape[0] == odd.shape[0]:
        #         label = torch.cat([odd, even], dim=1).view(-1)
        #     elif even.shape[0] > odd.shape[0]:
        #         label = torch.cat([odd, even[:-1]], dim=1).view(-1)
        #         temp = even[-1].view(-1)
        #         label = torch.cat([label,temp])
        #     else:
        #         label = torch.cat([odd[:-1], even], dim=1).view(-1)
        #         temp = odd[-1].view(-1)
        #         label = torch.cat([label, temp])
        # else:
        label = torch.cat([odd, even], dim=1).view(-1)


        return label
