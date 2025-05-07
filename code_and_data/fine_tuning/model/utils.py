from .metric import ndcg_at_k, precision_at_k, recall_at_k, mrr, mean_average_precision
from transformers import AutoTokenizer, BertTokenizerFast


backbone_plm_dict = {
    'bert':'sentence-transformers/msmarco-bert-base-dot-v5',
    'bert-tiny': 'sentence-transformers/paraphrase-TinyBERT-L6-v2',
    'roberta': 'sentence-transformers/msmarco-roberta-base-v2',
    'ance':'sentence-transformers/msmarco-roberta-base-ance-firstp',
    'dpr':'sentence-transformers/facebook-dpr-ctx_encoder-multiset-base',
    'tas-b':'sentence-transformers/msmarco-distilbert-base-tas-b',
    'sbert': 'sentence-transformers/all-MiniLM-L6-v2',
    'mpnet': 'sentence-transformers/all-mpnet-base-v2',


    'albert': 'albert-base-v2',
    'llama': 'meta-llama/Llama-2-7b'
}

special_tokenizer = {
    'bert-tiny': BertTokenizerFast,
    'albert': BertTokenizerFast
}


metric_dict = {
    "MRR": mrr,
    "NDCG@k": ndcg_at_k,
    "P@k": precision_at_k,
    "R@k": recall_at_k,
    "MAP": mean_average_precision,

}


def init_tokenizer(name, *args, **params):
    plm_path = backbone_plm_dict[name]
    if name in special_tokenizer:
        return special_tokenizer[name].from_pretrained(plm_path, cache_dir='./data/ckpt/')
    else:
        return AutoTokenizer.from_pretrained(plm_path, cache_dir='./data/ckpt/')


def init_metric(name):
    if name in metric_dict:
        return metric_dict[name]
    else:
        raise NotImplementedError
