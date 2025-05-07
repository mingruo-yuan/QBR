import random
import numpy as np
from time import time
import pickle
import torch
import copy
from tqdm import tqdm
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from model.utils import init_metric, init_tokenizer

def test_document(test_data, encoder_f, encoder_e=None, config=None, device=None, name=None, is_baseline=False, discrete=False):
    # initialize evaluate metrics
    k_list = [int(k) for k in config.get('test', 'k_list').split(',')]
    metric_list = [m.strip() for m in config.get('test', 'metric_list').split(',')]
    pos_score = config.getint('test', 'pos_score')
    tokenizer = init_tokenizer(config.get('encoder', 'backbone'))

    # evaluate result for all sentences.
    eval_result_all = defaultdict(list)
    sentence_samples = []

    # load model and evaluate
    if not discrete:
        if encoder_e is None:
            encoder_e = encoder_f
        encoder_e.eval()
        encoder_f.eval()
        encoder_e.to(device)
        encoder_f.to(device)

    test_data_ = copy.deepcopy(test_data)
    with torch.no_grad():
        for case_ in tqdm(test_data_, desc='Evaluating {}'.format(name)):
            case = copy.deepcopy(case_)
            sent_result = copy.deepcopy(case['sent_result'])



            # get fact and evidence list for a single case document.
            fact_list, evidence_list = [], []
            for result in sent_result:
                #print(result)
                assert not all([c['score'] < pos_score for c in result['candidate_scope']])   # all facts have positives
                fact_list.append(result['query'])
                for e in result['candidate_scope']:
                    evidence = e['candidate_scope']
                    #if evidence not in evidence_list:
                    evidence_list.append(evidence)
            assert len(fact_list) != 0 and len(evidence_list) != 0

            # compute similarity scores
            if is_baseline:  # test baselines
                print(len(evidence_list))
                sim_scores = encoder_f.search(fact_list, evidence_list)

            else:
                print(len(fact_list))
                print(len(fact_list[0]))
                fact_list = fact_list[0].split("********")
                fact_list = fact_list[:1000]
                print("pickle.start")
                print(len(fact_list))

                all_embedding_query = []
                for i in fact_list:
                    inputs_f = tokenizer(i, padding='longest', truncation=True, max_length=128, return_tensors='pt')
                    for ipt_key in inputs_f.keys():
                        inputs_f[ipt_key] = inputs_f[ipt_key].to(device)
                    embedding_f = encoder_f(**inputs_f)
                    all_embedding_query.append(embedding_f)
                file = open(f'query_148.pickle', 'wb')
                pickle.dump(all_embedding_query, file)
                file.close()
                print(len(all_embedding_query))


                print("pickle.start")
                print(len(evidence_list))

                all_embedding = []
                for i in evidence_list:
                    inputs_e = tokenizer(i, padding='longest', truncation=True, max_length=128,
                                     return_tensors='pt')
                    for ipt_key in inputs_e.keys():
                        inputs_e[ipt_key] = inputs_e[ipt_key].to(device)
                    embedding_e = encoder_e(**inputs_e)
                    all_embedding.append(embedding_e)
                file = open(f'string_scope_qbr_148.pickle', 'wb')
                pickle.dump(all_embedding, file)
                file.close()
                print(len(all_embedding))
                print("pickle.done")

                sim_scores = cosine_similarity(embedding_f.cpu(), embedding_e.cpu())

            for result_ in case['sent_result']:
                result = copy.deepcopy(result_)
                # get pred ranking
                pred_ranking = copy.deepcopy(result['candidate_scope'])

                fct_id = fact_list.index(result['query'])
                for idx, evidence in enumerate(pred_ranking):
                    evi_id = evidence_list.index(evidence['candidate_scope'])
                    evidence['pred_score'] = sim_scores[fct_id][evi_id].item()

                # sort evidence list by pred_score, implementing in this way to prevent data leaky.
                pred_indices = [(e['pred_score'], idx) for idx, e in enumerate(pred_ranking)]
                pred_indices = sorted(pred_indices, key=lambda e: e[0], reverse=True)
                pred_ranking = [pred_ranking[idx] for _, idx in pred_indices]

                # compute evaluate metric values
                eval_result = dict()
                for m_name in metric_list:
                    metric = init_metric(m_name)
                    if 'k' in m_name:
                        for k in k_list:
                            score = metric(pred_ranking, k=k, pos_score=pos_score)
                            eval_result[m_name.replace('k', str(k))] = score
                            eval_result_all[m_name.replace('k', str(k))].append(score)  # {"NDCG@5": [...],"P@5": [...]}
                    else:
                        score = metric(pred_ranking, pos_score=pos_score)
                        eval_result[m_name] = score
                        eval_result_all[m_name].append(score)     # {"NDCG@5": [...], "P@5": [...]}

                sentence_samples.append({'fact': result['query'],
                                         'eval_result': eval_result,
                                         'pred_ranking': pred_ranking})

        for m_name in eval_result_all:
            eval_result_all[m_name] = sum(eval_result_all[m_name]) / len(eval_result_all[m_name])

        eval_result_all['avg'] = sum(eval_result_all.values()) / len(eval_result_all.keys())

    return {'sent_avg': eval_result_all,
            'sentence_pred': sentence_samples}

def test(test_data, encoder_f, encoder_e=None, config=None, device=None, name=None, is_baseline=False, discrete=False):
    # initialize evaluate metrics
    k_list = [int(k) for k in config.get('test', 'k_list').split(',')]
    metric_list = [m.strip() for m in config.get('test', 'metric_list').split(',')]
    pos_score = config.getint('test', 'pos_score')
    tokenizer = init_tokenizer(config.get('encoder', 'backbone'))

    # evaluate result for all sentences.
    eval_result_all = defaultdict(list)
    sentence_samples = []

    # load model and evaluate
    if not discrete:
        if encoder_e is None:
            encoder_e = encoder_f
        encoder_e.eval()
        encoder_f.eval()
        encoder_e.to(device)
        encoder_f.to(device)

    test_data_ = copy.deepcopy(test_data)
    with torch.no_grad():
        for case_ in tqdm(test_data_, desc='Evaluating {}'.format(name)):
            case = copy.deepcopy(case_)
            sent_result = copy.deepcopy(case['sent_result'])

            fact_list, evidence_list = [], []
            for result in sent_result:
                assert not all([c['score'] < pos_score for c in result['candidate_scope']])   # all facts have positives
                fact_list.append(result['query'])
                for e in result['candidate_scope']:
                    evidence = e['candidate_scope']
                    if evidence not in evidence_list:
                        evidence_list.append(evidence)
            assert len(fact_list) != 0 and len(evidence_list) != 0

            # compute similarity scores
            if is_baseline:  # test baselines
                sim_scores = encoder_f.search(fact_list, evidence_list)
            else:
                inputs_f = tokenizer(fact_list, padding='longest', truncation=True, max_length=128, return_tensors='pt')
                inputs_e = tokenizer(evidence_list, padding='longest', truncation=True, max_length=128,
                                     return_tensors='pt')

                for ipt_key in inputs_f.keys():
                    inputs_f[ipt_key] = inputs_f[ipt_key].to(device)
                    inputs_e[ipt_key] = inputs_e[ipt_key].to(device)

                embedding_f = encoder_f(**inputs_f)
                embedding_e = encoder_e(**inputs_e)

                sim_scores = cosine_similarity(embedding_f.cpu(), embedding_e.cpu())

            for result_ in case['sent_result']:
                result = copy.deepcopy(result_)
                # get pred ranking
                pred_ranking = copy.deepcopy(result['candidate_scope'])

                fct_id = fact_list.index(result['query'])
                for idx, evidence in enumerate(pred_ranking):
                    evi_id = evidence_list.index(evidence['candidate_scope'])
                    evidence['pred_score'] = sim_scores[fct_id][evi_id].item()

                # sort evidence list by pred_score, implementing in this way to prevent data leaky.
                pred_indices = [(e['pred_score'], idx) for idx, e in enumerate(pred_ranking)]
                pred_indices = sorted(pred_indices, key=lambda e: e[0], reverse=True)
                pred_ranking = [pred_ranking[idx] for _, idx in pred_indices]

                # compute evaluate metric values
                eval_result = dict()
                for m_name in metric_list:
                    metric = init_metric(m_name)
                    if 'k' in m_name:
                        for k in k_list:
                            score = metric(pred_ranking, k=k, pos_score=pos_score)
                            eval_result[m_name.replace('k', str(k))] = score
                            eval_result_all[m_name.replace('k', str(k))].append(score)  # {"NDCG@5": [...],"P@5": [...]}
                    else:
                        score = metric(pred_ranking, pos_score=pos_score)
                        eval_result[m_name] = score
                        eval_result_all[m_name].append(score)     # {"NDCG@5": [...], "P@5": [...]}

                sentence_samples.append({'fact': result['query'],
                                         'eval_result': eval_result,
                                         'pred_ranking': pred_ranking})

        for m_name in eval_result_all:
            eval_result_all[m_name] = sum(eval_result_all[m_name]) / len(eval_result_all[m_name])

        eval_result_all['avg'] = sum(eval_result_all.values()) / len(eval_result_all.keys())

    return {'sent_avg': eval_result_all,
            'sentence_pred': sentence_samples}
