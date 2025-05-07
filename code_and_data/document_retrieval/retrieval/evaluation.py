# Final Hit Document
import pandas as pd
import json
import os

# def write_to_file(data, path):
#     with open(path, "w", encoding="utf-8") as f:
#         f.write(json.dumps(data, ensure_ascii=False, indent=2))
# def read_from_json(path_of_file):
#     with open(path_of_file, 'r' ,encoding='utf-8') as f:
#         results = json.load(f)
#     return results
#
# string_scope_lqb = read_from_json("output/retrieval/retrieval_data_w_dedup_scope_test_index.json")


## Precision
def precision_k(mpnet_w_dedup, k, variable):
    # step 1: get top-k result
    new_mpnet = {}
    for i in mpnet_w_dedup:
        new_mpnet[i] = mpnet_w_dedup[i][:k]
    precision_num = []
    # step 2: get precision number
    for i in new_mpnet:
        page_ground_truth = i.split("-")[1]
        page_list = []
        for index, j in enumerate(new_mpnet[i]):
            retrieved = j.split("-")[variable]
            if page_ground_truth == retrieved:
                page_list.append(1)
            else:
                page_list.append(0)
        # print((page_list))
        precision_num.append(sum(page_list) / k)
    return "{:.4f}".format(sum(precision_num) / len(mpnet_w_dedup))


## Recall
def recall_k(mpnet_w_dedup, k, variable):
    # step 1: get top-k result
    new_mpnet = {}
    for i in mpnet_w_dedup:
        new_mpnet[i] = mpnet_w_dedup[i][:k]

    recall_num = []
    # step 2: get recall number
    for i in new_mpnet:
        signal = False
        page_ground_truth = i.split("-")[1]
        for index, j in enumerate(new_mpnet[i]):
            retrieved = j.split("-")[variable]
            if page_ground_truth == retrieved:
                signal = True
                break
        if signal:
            recall_num.append(1)
        else:
            recall_num.append(0)

    return "{:.4f}".format(sum(recall_num) / len(mpnet_w_dedup))


# MRR
def mrr_k(mpnet_w_dedup, k, variable):
    # step 1: get top-k result
    new_mpnet = {}
    for i in mpnet_w_dedup:
        new_mpnet[i] = mpnet_w_dedup[i][:k]
    # step 2: get mrr number
    mrr_num = []
    for i in new_mpnet:
        page_ground_truth = i.split("-")[1]
        for index, j in enumerate(new_mpnet[i]):
            retrieved = j.split("-")[variable]
            if page_ground_truth == retrieved:
                mrr_num.append(1.0 / (index + 1))
                break

    return "{:.4f}".format(sum(mrr_num) / len(mpnet_w_dedup))

