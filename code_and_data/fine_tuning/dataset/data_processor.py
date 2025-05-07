import copy
import os.path
import random

from torch.utils.data import Dataset
import jsonlines
from tqdm import tqdm
import torch
from collections import defaultdict
import json


class UserQueryExample(object):
    def __init__(self, candidate_scopes, query, truth = None):
        self.candidate_scopes_list = candidate_scopes
        self.query_list = query
        self.truth = truth


class InputFeature(object):
    def __init__(self, inputs_query, inputs_candidate_scopes, inputs_truth):
        self.inputs_query = inputs_query
        self.inputs_candidate_scopes = inputs_candidate_scopes
        self.inputs_truth = inputs_truth


class DataProcessor(Dataset):
    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self.input_examples = None
        self.input_features = None
    def read_example(self, input_file):
        raise NotImplementedError

    def convert_examples_to_features(self):
        raise NotImplementedError

    def __len__(self):
        return len(self.input_features)

    def __getitem__(self, index):
        features = self.input_features[index]

        return features


class CLProcessor(DataProcessor):
    def __init__(self, config, tokenizer, input_file):
        super(CLProcessor, self).__init__(config, tokenizer)
        self.config = config
        self.double_feature = config.getboolean('simcse_loss', 'use')
        self.input_examples = self.read_example(input_file)
        self.input_features = self.convert_examples_to_features()

    def read_example(self, input_file):
        examples = []
        for case in jsonlines.open(input_file):
            example = UserQueryExample(candidate_scopes=case['candidate_scope'], query=case['query'], truth=case['meta_info_ground_scope'])
            examples.append(example)
        return examples

    def convert_examples_to_features(self):
        feature_num = 'double' if self.double_feature else 'single'
        token_type = self.config.get('encoder', 'backbone') + "_" + feature_num + "_" + self.config.get('encoder', 'embedding_size') #比如说是Roberta double
        data_path = self.config.get('data', 'train_data')
        cache_path = data_path.replace('train/', 'train_cache/').replace('.jsonl', '-{}.jsonl'.format(token_type))

        features = []
        if os.path.exists(cache_path):
            inputs_list = jsonlines.open(cache_path)
            for inputs in tqdm(inputs_list, desc='Loading from {}'.format(cache_path)):
                feature = InputFeature(inputs_query=inputs['query'], inputs_candidate_scopes=inputs['candidate_scope'], inputs_truth=inputs['truth'])
                features.append(feature)
        else:
            with jsonlines.open(cache_path, 'w') as f:
                for example in tqdm(self.input_examples, desc='Converting examples to features'):
                    query_list = copy.deepcopy(example.query_list)
                    candidate_scopes_list = copy.deepcopy(example.candidate_scopes_list)
                    truth_list = copy.deepcopy(example.truth)

                    if self.double_feature:
                        # double the inputs for simcse here
                        query_list = sum([list(f) for f in zip(query_list, query_list)], [])
                        truth_list = sum([list(t) for t in zip(truth_list, truth_list)], [])
                        for e_id, evidence_doc in enumerate(candidate_scopes_list):
                            candidate_scopes_list[e_id] = sum([list(e) for e in zip(evidence_doc, evidence_doc)], [])

                    inputs_query = self.tokenizer(query_list, padding='max_length', truncation=True, max_length=int(self.config.get('encoder', 'embedding_size')))
                    inputs_scopes = defaultdict(list)
                    for evi_doc in candidate_scopes_list:
                        evi_inputs = self.tokenizer(evi_doc, padding='max_length', truncation=True, max_length=int(self.config.get('encoder', 'embedding_size')))    # bert 最大 512
                        for key in evi_inputs:
                            inputs_scopes[key].append(evi_inputs[key])

                    if 'token_type_ids' in inputs_query and 'token_type_ids' in inputs_scopes:
                        del inputs_query['token_type_ids']
                        del inputs_scopes['token_type_ids']

                    jsonlines.Writer.write(f, dict(query=dict(inputs_query), candidate_scope=dict(inputs_scopes),truth = truth_list))
                    features.append(InputFeature(inputs_query=inputs_query, inputs_candidate_scopes=inputs_scopes, inputs_truth = truth_list))

            print('finished caching.')
        return features

    def collate_fn(self, batch):
        batch_copy = copy.deepcopy(batch)

        output_batch = dict()

        inputs_query = {'input_ids': [], 'attention_mask': []}
        inputs_candidate_scopes = {'input_ids': [], 'attention_mask': []}
        offset = {'fact': [], 'evidence': [], 'evidence_inner': []}
        truth = []

        st1, st2 = 0, 0
        ed1, ed2 = 0, 0
        for i, b in enumerate(batch_copy):
            #print("len(b)",len(b.inputs_candidate_scopes['input_ids']))
            if isinstance(b.inputs_candidate_scopes['input_ids'][0], list):

                record_num = len(b.inputs_candidate_scopes['input_ids'])
                sample_num = self.config.getint('contra_loss', 'value_sample_num')
                indices = random.sample(range(record_num), min(sample_num, record_num))

                tmp = defaultdict(list)
                for key in b.inputs_candidate_scopes:                   #input_ids attention mask
                    for idx in indices:
                        tmp[key] += b.inputs_candidate_scopes[key][idx]
                evidence = tmp

            else:
                print("b.inputs_candidate_scopes['input_ids'][0] not list")
                evidence = b.inputs_candidate_scopes

            ed1 = st1 + len(b.inputs_query['input_ids'])
            ed2 = st2 + len(evidence['input_ids'])
            offset['fact'].append([st1, ed1])
            offset['evidence'].append([st2, ed2])
            # print(offset)
            st1, st2 = ed1, ed2

            for key in ['input_ids', 'attention_mask']:
                inputs_query[key].extend(b.inputs_query[key])
                inputs_candidate_scopes[key].extend(evidence[key])
            truth.append(b.inputs_truth)

        if self.config.getboolean('train', 'multi_gpu') and (ed2/2) % 2 != 0 and self.config.getboolean('simcse_loss', 'use'):
            offset['evidence'][-1][1] -= 2
            truth[-1] = truth[-1][:-2]
            inputs_candidate_scopes["input_ids"] = inputs_candidate_scopes["input_ids"][:-2]
            inputs_candidate_scopes["attention_mask"]= inputs_candidate_scopes["attention_mask"][:-2]
        if self.config.getboolean('train', 'multi_gpu') and (ed2) % 2 != 0 and (not self.config.getboolean('simcse_loss','use')):
            offset['evidence'][-1][1] -= 1
            truth[-1] = truth[-1][:-1]
            inputs_candidate_scopes["input_ids"] = inputs_candidate_scopes["input_ids"][:-1]
            inputs_candidate_scopes["attention_mask"] = inputs_candidate_scopes["attention_mask"][:-1]


        for key in ['input_ids', 'attention_mask']:
            inputs_query[key] = torch.tensor(inputs_query[key], dtype=torch.long)
            inputs_candidate_scopes[key] = torch.tensor(inputs_candidate_scopes[key], dtype=torch.long)

        output_batch['inputs_query'] = inputs_query
        output_batch['inputs_candidate_scopes'] = inputs_candidate_scopes
        output_batch['offset'] = offset
        output_batch['truth'] = truth

        return output_batch
