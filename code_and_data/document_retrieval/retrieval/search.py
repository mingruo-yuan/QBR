from .util import cos_sim, dot_score
import torch
import pytrec_eval
import logging
from time import time
from typing import Type, List, Dict, Union, Tuple
from .util import mrr, recall_cap, hole, top_k_accuracy

logger = logging.getLogger(__name__)

# Parent class for any dense model
class DenseRetrievalExactSearch:

    def __init__(self, model, batch_size: int = 128, corpus_chunk_size: int = 50000, **kwargs):
        # model is class that provides encode_corpus() and encode_queries()
        self.model = model
        self.batch_size = batch_size
        self.score_functions = {'cos_sim': cos_sim, 'dot': dot_score}
        self.score_function_desc = {'cos_sim': "Cosine Similarity", 'dot': "Dot Product"}
        self.corpus_chunk_size = corpus_chunk_size
        self.show_progress_bar = True  # TODO: implement no progress bar if false
        self.convert_to_tensor = True
        self.results = {}

    def search(self,
               corpus: Dict[str, Dict[str, str]],
               queries: Dict[str, str],
               top_k: List[int],
               score_function: str,
               return_sorted: bool = False, **kwargs) -> Dict[str, Dict[str, float]]:
        # Create embeddings for all queries using model.encode_queries()
        # Runs semantic search against the corpus embeddings
        # Returns a ranked list with the corpus ids
        if score_function not in self.score_functions:
            raise ValueError(
                "score function: {} must be either (cos_sim) for cosine similarity or (dot) for dot product".format(
                    score_function))

        logger.info("Encoding Queries...")
        query_ids = list(queries.keys())
        self.results = {qid: {} for qid in query_ids}
        queries = [queries[qid] for qid in queries]
        start_encode_q_time = time()
        query_embeddings = self.model.encode_queries(
            queries, batch_size=self.batch_size, show_progress_bar=self.show_progress_bar,
            convert_to_tensor=self.convert_to_tensor)
        end_encode_q_time = time()
        logger.info("Time taken to encoding Q: {:.7f} seconds".format(end_encode_q_time - start_encode_q_time))

        logger.info("Sorting Corpus by document length (Longest first)...")

        corpus_ids = sorted(corpus, key=lambda k: len(corpus[k].get("title", "") + corpus[k].get("text", "")),
                            reverse=True)
        corpus = [corpus[cid] for cid in corpus_ids]

        logger.info("Encoding Corpus in batches... Warning: This might take a while!")
        logger.info("Scoring Function: {} ({})".format(self.score_function_desc[score_function], score_function))

        itr = range(0, len(corpus), self.corpus_chunk_size)

        for batch_num, corpus_start_idx in enumerate(itr):
            logger.info("Encoding Batch {}/{}...".format(batch_num + 1, len(itr)))
            corpus_end_idx = min(corpus_start_idx + self.corpus_chunk_size, len(corpus))  # end_idx: 3484

            # Encode chunk of corpus
            sub_corpus_embeddings = self.model.encode_corpus(
                corpus[corpus_start_idx:corpus_end_idx],  # 0-3484
                batch_size=self.batch_size,
                show_progress_bar=self.show_progress_bar,
                convert_to_tensor=self.convert_to_tensor
            )
            start_match_q_time = time()
            # Compute similarites using either cosine-similarity or dot product
            cos_scores = self.score_functions[score_function](query_embeddings, sub_corpus_embeddings)
            cos_scores[torch.isnan(cos_scores)] = -1
            end_match_q_time = time()
            logger.info("Time taken to Compute similarites Q: {:.7f} seconds".format(end_match_q_time - start_match_q_time))

            # Get top-k values
            cos_scores_top_k_values, cos_scores_top_k_idx = torch.topk(cos_scores, min(top_k + 1, len(cos_scores[1])),
                                                                       dim=1, largest=True, sorted=return_sorted)
            cos_scores_top_k_values = cos_scores_top_k_values.cpu().tolist()
            cos_scores_top_k_idx = cos_scores_top_k_idx.cpu().tolist()

            for query_itr in range(len(query_embeddings)):
                query_id = query_ids[query_itr]
                for sub_corpus_id, score in zip(cos_scores_top_k_idx[query_itr], cos_scores_top_k_values[query_itr]):
                    corpus_id = corpus_ids[corpus_start_idx + sub_corpus_id]
                    if corpus_id != query_id:
                        self.results[query_id][corpus_id] = score

        return self.results

    def search_w_weight(self,
                        corpus: Dict[str, Dict[str, str]],
                        queries: Dict[str, str],
                        top_k: List[int],
                        score_function: str,
                        index_to_page,
                        cluster,
                        weight_num=0.1,
                        return_sorted: bool = False, **kwargs) -> Dict[str, Dict[str, float]]:
        # Create embeddings for all queries using model.encode_queries()
        # Runs semantic search against the corpus embeddings
        # Returns a ranked list with the corpus ids
        if score_function not in self.score_functions:
            raise ValueError(
                "score function: {} must be either (cos_sim) for cosine similarity or (dot) for dot product".format(
                    score_function))

        logger.info("Encoding Queries...")
        query_ids = list(queries.keys())
        self.results = {qid: {} for qid in query_ids}
        queries = [queries[qid] for qid in queries]
        query_embeddings = self.model.encode_queries(
            queries, batch_size=self.batch_size, show_progress_bar=self.show_progress_bar,
            convert_to_tensor=self.convert_to_tensor)

        logger.info("Sorting Corpus by document length (Longest first)...")

        corpus_ids = sorted(corpus, key=lambda k: len(corpus[k].get("title", "") + corpus[k].get("text", "")),
                            reverse=True)
        corpus = [corpus[cid] for cid in corpus_ids]

        logger.info("Encoding Corpus in batches... Warning: This might take a while!")
        logger.info("Scoring Function: {} ({})".format(self.score_function_desc[score_function], score_function))

        itr = range(0, len(corpus), self.corpus_chunk_size)

        for batch_num, corpus_start_idx in enumerate(itr):
            logger.info("Encoding Batch {}/{}...".format(batch_num + 1, len(itr)))
            corpus_end_idx = min(corpus_start_idx + self.corpus_chunk_size, len(corpus))

            # Encode chunk of corpus
            sub_corpus_embeddings = self.model.encode_corpus(
                corpus[corpus_start_idx:corpus_end_idx],
                batch_size=self.batch_size,
                show_progress_bar=self.show_progress_bar,
                convert_to_tensor=self.convert_to_tensor
            )

            print(len(sub_corpus_embeddings))  # [3484,768]

            weighted_corpus_embeddings = sub_corpus_embeddings
            for index, i in enumerate(sub_corpus_embeddings):
                temp = torch.zeros([1, len(sub_corpus_embeddings[0])], dtype=torch.float, device='cuda')

                page = index_to_page[str(index)]  # int -> str
                group_embed = cluster[page]  # str -> list
                for j in group_embed:
                    if j != index:
                        temp = temp + (weight_num) * sub_corpus_embeddings[j]
                weighted_corpus_embeddings[index] = (1 - (weight_num) * (len(group_embed) - 1)) * i + temp

            sub_corpus_embeddings = weighted_corpus_embeddings

            # Compute similarites using either cosine-similarity or dot product
            cos_scores = self.score_functions[score_function](query_embeddings, sub_corpus_embeddings)
            cos_scores[torch.isnan(cos_scores)] = -1

            # Get top-k values
            cos_scores_top_k_values, cos_scores_top_k_idx = torch.topk(cos_scores, min(top_k + 1, len(cos_scores[1])),
                                                                       dim=1, largest=True, sorted=return_sorted)
            cos_scores_top_k_values = cos_scores_top_k_values.cpu().tolist()
            cos_scores_top_k_idx = cos_scores_top_k_idx.cpu().tolist()

            for query_itr in range(len(query_embeddings)):
                query_id = query_ids[query_itr]
                for sub_corpus_id, score in zip(cos_scores_top_k_idx[query_itr], cos_scores_top_k_values[query_itr]):
                    corpus_id = corpus_ids[corpus_start_idx + sub_corpus_id]
                    # if corpus_id != query_id:
                    self.results[query_id][corpus_id] = score

        return self.results

class EvaluateRetrieval:

    def __init__(self, retriever,
                 k_values, score_function: str = "cos_sim"):
        self.k_values = k_values
        self.top_k = max(k_values)
        self.retriever = retriever
        self.score_function = score_function

    # original
    def retrieve(self, corpus: Dict[str, Dict[str, str]], queries: Dict[str, str], **kwargs) -> Dict[
        str, Dict[str, float]]:
        if not self.retriever:
            raise ValueError("Model/Technique has not been provided!")
        return self.retriever.search(corpus, queries, self.top_k, self.score_function, **kwargs)

    # with weight
    def retrieve_w_weight(self, corpus: Dict[str, Dict[str, str]], queries: Dict[str, str], index_to_page, cluster,
                          weight_num=0.1, **kwargs) -> Dict[str, Dict[str, float]]:
        if not self.retriever:
            raise ValueError("Model/Technique has not been provided!")
        return self.retriever.search_w_weight(corpus, queries, self.top_k, self.score_function, index_to_page, cluster,
                                              weight_num, **kwargs)

    def rerank(self,
               corpus: Dict[str, Dict[str, str]],
               queries: Dict[str, str],
               results: Dict[str, Dict[str, float]],
               top_k: int) -> Dict[str, Dict[str, float]]:

        new_corpus = {}

        for query_id in results:
            if len(results[query_id]) > top_k:
                for (doc_id, _) in sorted(results[query_id].items(), key=lambda item: item[1], reverse=True)[:top_k]:
                    new_corpus[doc_id] = corpus[doc_id]
            else:
                for doc_id in results[query_id]:
                    new_corpus[doc_id] = corpus[doc_id]

        return self.retriever.search(new_corpus, queries, top_k, self.score_function)

    @staticmethod
    def evaluate(qrels: Dict[str, Dict[str, int]],
                 results: Dict[str, Dict[str, float]],
                 k_values: List[int],
                 ignore_identical_ids: bool = False) -> Tuple[
        Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, float]]:

        if ignore_identical_ids:
            logging.info(
                'For evaluation, we ignore identical query and document ids (default), please explicitly set ``ignore_identical_ids=False`` to ignore this.')
            popped = []
            for qid, rels in results.items():
                for pid in list(rels):
                    if qid == pid:
                        results[qid].pop(pid)
                        popped.append(pid)

        ndcg = {}
        _map = {}
        recall = {}
        precision = {}

        for k in k_values:
            ndcg[f"NDCG@{k}"] = 0.0
            _map[f"MAP@{k}"] = 0.0
            recall[f"Recall@{k}"] = 0.0
            precision[f"P@{k}"] = 0.0

        map_string = "map_cut." + ",".join([str(k) for k in k_values])
        ndcg_string = "ndcg_cut." + ",".join([str(k) for k in k_values])
        recall_string = "recall." + ",".join([str(k) for k in k_values])
        precision_string = "P." + ",".join([str(k) for k in k_values])
        evaluator = pytrec_eval.RelevanceEvaluator(qrels, {map_string, ndcg_string, recall_string, precision_string})
        scores = evaluator.evaluate(results)

        for query_id in scores.keys():
            for k in k_values:
                ndcg[f"NDCG@{k}"] += scores[query_id]["ndcg_cut_" + str(k)]
                _map[f"MAP@{k}"] += scores[query_id]["map_cut_" + str(k)]
                recall[f"Recall@{k}"] += scores[query_id]["recall_" + str(k)]
                precision[f"P@{k}"] += scores[query_id]["P_" + str(k)]

        for k in k_values:
            ndcg[f"NDCG@{k}"] = round(ndcg[f"NDCG@{k}"] / len(scores), 5)
            _map[f"MAP@{k}"] = round(_map[f"MAP@{k}"] / len(scores), 5)
            recall[f"Recall@{k}"] = round(recall[f"Recall@{k}"] / len(scores), 5)
            precision[f"P@{k}"] = round(precision[f"P@{k}"] / len(scores), 5)

        for eval in [ndcg, _map, recall, precision]:
            logging.info("\n")
            for k in eval.keys():
                logging.info("{}: {:.4f}".format(k, eval[k]))

        return ndcg, _map, recall, precision

    @staticmethod
    def evaluate_custom(qrels: Dict[str, Dict[str, int]],
                        results: Dict[str, Dict[str, float]],
                        k_values: List[int], metric: str) -> Tuple[Dict[str, float]]:

        if metric.lower() in ["mrr", "mrr@k", "mrr_cut"]:
            return mrr(qrels, results, k_values)

        elif metric.lower() in ["recall_cap", "r_cap", "r_cap@k"]:
            return recall_cap(qrels, results, k_values)

        elif metric.lower() in ["hole", "hole@k"]:
            return hole(qrels, results, k_values)

        elif metric.lower() in ["acc", "top_k_acc", "accuracy", "accuracy@k", "top_k_accuracy"]:
            return top_k_accuracy(qrels, results, k_values)

    @staticmethod
    def evaluate_w_lqb(
                       results: Dict[str, Dict[str, float]],
                       corpus,
                       k_values: List[int],
                       ignore_identical_ids: bool = False) -> Tuple[
        Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, Dict[str, float]]]:

        if ignore_identical_ids:
            logging.info(
                'For evaluation, we ignore identical query and document ids (default), please explicitly set ``ignore_identical_ids=False`` to ignore this.')
            popped = []
            for qid, rels in results.items():
                for pid in list(rels):
                    if qid == pid:
                        results[qid].pop(pid)
                        popped.append(pid)
        temp = {}
        for qid, rels in results.items():  # qid + {}
            scores = sorted(rels.items(), key=lambda item: item[1], reverse=True)  #
            # if qid == "1529-45":
            #    print(scores)
            sorted_rels = {}
            for i in scores:
                sorted_rels[i[0]] = i[1]
            scope = []
            page = set()
            new_rels = {}
            for pid in list(sorted_rels):
                if corpus[pid]["metadata"] in scope:
                    continue
                # elif len(scope) == 10:
                #     break
                elif len(page) == k_values[-1]:
                    break
                else:
                    scope.append(corpus[pid]["metadata"])
                    page.add(pid.split("-")[1])
                    new_rels[pid] = rels[pid]
            # print(len(new_rels))
            # print(page)
            temp[qid] = new_rels

        results = temp
        return results

