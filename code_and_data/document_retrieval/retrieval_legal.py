from retrieval.data_loader import GenericDataLoader,DataLoader
from retrieval.search import DenseRetrievalExactSearch as DRES
from retrieval.model import SentenceBERT
from retrieval.search import EvaluateRetrieval
from retrieval.evaluation import precision_k, recall_k, mrr_k
import retrieval.util
from retrieval.logging import LoggingHandler
from time import time
import logging, json

#### Output
def output_result_w_dedup_scope(results,corpus,queries,dataset_path,dataset,output_signal = False):
    output_result = {}
    output_result_index = {}
    for query_id, scores_dict in list(results.items()):
        single = []
        single_index = []
        scores = sorted(scores_dict.items(), key=lambda item: item[1], reverse=True)
        for rank in range(len(scores)):
            doc_id = scores[rank][0]
            single_index.append(doc_id)
            single.append("Doc %d: %s [%s] - %s\n" % (rank + 1, doc_id, corpus[doc_id].get("title"), corpus[doc_id].get("text")))
        output_result[query_id + " # "+ queries[query_id]] = single
        output_result_index[query_id] = single_index
    if output_signal:
        with open(f"output/retrieval/{dataset_path}_w_dedup_scope_{dataset}_index.json", "w",encoding= 'utf-8') as f:
            f.write(json.dumps(output_result_index,ensure_ascii= False,indent = 2))
    return output_result_index



if __name__ == '__main__':

    #### Data
    dataset_path = "data"
    dataset = "legal"
    data_path = f"{dataset_path}/{dataset}"
    corpus, queries = DataLoader(data_folder=data_path).load(split="test")


    #### Dense Retrieval
    model = DRES(SentenceBERT("all-mpnet-base-v2"))
    retriever = EvaluateRetrieval(model, k_values = [1,3,5,10,200])
    print(retriever.k_values)

    start_time = time()
    results = retriever.retrieve(corpus, queries)
    end_time = time()
    print("Time taken to retrieve: {:.7f} seconds".format(end_time - start_time))
    logging.info("Retriever evaluation for k in: {}".format(retriever.k_values))

    #### Results with LQB #scope deduplication
    logging.info("Retriever evaluation with LQB for k in: {}".format(retriever.k_values[:-1]))
    start_time = time()
    results_w_lqb = retriever.evaluate_w_lqb(results, corpus, retriever.k_values[:-1])
    end_time = time()
    print("Time taken to retrieve: {:.7f} seconds".format(end_time - start_time))
    ## Output
    output_result_index = output_result_w_dedup_scope(results_w_lqb,corpus,queries,dataset_path,dataset)

    evaluation_result = {"P@1":precision_k(output_result_index, 1, 1),"P@3":recall_k(output_result_index, 3, 1),"P@5":recall_k(output_result_index, 5, 1),"MRR":mrr_k(output_result_index, 10, 1)}
    for i in evaluation_result:
        print(i,":",evaluation_result[i])