from typing import List, Any, Dict, Iterable, Tuple
from math import log2
from collections import defaultdict

def ndcg(ranked_list: List[Any], ground_truth: List[Any], recommend_size: int) -> float:
    """
    Computes the normalized Discounted Cumulative Gain (nDCG).

    Formula:
        DCG@k = sum(1 / log2(i + 2) * rel(i)) for i in range(k)
        nDCG@k = DCG@k / IDCG@k

    where rel(i) = 1 if the item at rank i is relevant, else 0.

    Parameters:
        ranked_list: List of recommended items.
        ground_truth: List of relevant items.
        recommend_size: Number of items to recommend.

    Returns:
        nDCG score as a float.
    """
    k = min(len(ranked_list), recommend_size)
    dcg = sum(1 / log2(i + 2) for i in range(k) if ranked_list[i] in ground_truth)
    ideal_k = min(len(ground_truth), recommend_size)
    ideal_dcg = sum(1 / log2(i + 2) for i in range(ideal_k))
    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0

def precision(ranked_list: List[Any], ground_truth: List[Any], recommend_size: int) -> float:
    """
    Computes Precision@k.

    Formula:
        Precision@k = |relevant ∩ recommended| / k

    Parameters:
        ranked_list: List of recommended items.
        ground_truth: List of relevant items.
        recommend_size: Number of items to recommend.

    Returns:
        Precision score as a float.
    """
    if not ground_truth:
        return 1.0 if not ranked_list else 0.0

    k = min(len(ranked_list), recommend_size)
    true_positive = sum(1 for i in range(k) if ranked_list[i] in ground_truth)
    return true_positive / k if k > 0 else 0.0

def recall(ranked_list: List[Any], ground_truth: List[Any], recommend_size: int) -> float:
    """
    Computes Recall@k.

    Formula:
        Recall@k = |relevant ∩ recommended| / |relevant|

    Parameters:
        ranked_list: List of recommended items.
        ground_truth: List of relevant items.
        recommend_size: Number of items to recommend.

    Returns:
        Recall score as a float.
    """
    if not ground_truth:
        return 1.0 if not ranked_list else 0.0

    tp = true_positives(ranked_list, ground_truth, recommend_size)
    return tp / len(ground_truth)

def true_positives(ranked_list: List[Any], ground_truth: List[Any], recommend_size: int) -> int:
    """
    Counts the number of true positives.

    Formula:
        True Positives@k = |relevant ∩ recommended|

    Parameters:
        ranked_list: List of recommended items.
        ground_truth: List of relevant items.
        recommend_size: Number of items to recommend.

    Returns:
        Number of true positives as an integer.
    """
    k = min(len(ranked_list), recommend_size)
    return sum(1 for i in range(k) if ranked_list[i] in ground_truth)

def f1_score(ranked_list: List[Any], ground_truth: List[Any], recommend_size: int) -> float:
    """
    Computes F1-score@k, the harmonic mean of precision and recall.

    Formula:
        F1@k = 2 * (Precision@k * Recall@k) / (Precision@k + Recall@k)

    Parameters:
        ranked_list: List of recommended items.
        ground_truth: List of relevant items.
        recommend_size: Number of items to recommend.

    Returns:
        F1-score as a float.
    """
    if not ground_truth and not ranked_list:
        return 1.0 # Both empty lists

    prec = precision(ranked_list, ground_truth, recommend_size)
    rec = recall(ranked_list, ground_truth, recommend_size)
    return 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

def hit(ranked_list: List[Any], ground_truth: List[Any], recommend_size: int) -> float:
    """
    Computes Hit@k, indicating if at least one relevant item is recommended.

    Formula:
        Hit@k = 1 if |relevant ∩ recommended| > 0 else 0

    Parameters:
        ranked_list: List of recommended items.
        ground_truth: List of relevant items.
        recommend_size: Number of items to recommend.

    Returns:
        Hit score as a float.
    """
    return 1.0 if any(item in ground_truth for item in ranked_list[:recommend_size]) else 0.0

def reciprocal_rank(ranked_list: List[Any], ground_truth: List[Any], recommend_size: int) -> float:
    """
    Computes Reciprocal Rank (RR), the reciprocal of the rank of the first relevant item.

    Formula:
        RR@k = 1 / rank of first relevant item

    Parameters:
        ranked_list: List of recommended items.
        ground_truth: List of relevant items.
        recommend_size: Number of items to recommend.

    Returns:
        Reciprocal Rank as a float.
    """
    k = min(len(ranked_list), recommend_size)
    for i in range(k):
        if ranked_list[i] in ground_truth:
            return 1.0 / (i + 1)
    return 0.0

def mrr(ranked_lists: Iterable[List[Any]], ground_truths: Iterable[List[Any]], recommend_size: int) -> float:
    """
    Computes Mean Reciprocal Rank (MRR) across multiple queries.

    Formula:
        MRR = (1 / |Q|) * sum(RR@k(q) for q in Q)

    Parameters:
        ranked_lists: List of recommended lists for each query.
        ground_truths: List of relevant items for each query.
        recommend_size: Number of items to recommend.

    Returns:
        MRR score as a float.
    """
    rr_sum = sum(reciprocal_rank(r, g, recommend_size) for r, g in zip(ranked_lists, ground_truths))
    return rr_sum / len(ranked_lists) if ranked_lists else 0.0

def auc(ranked_list: List[Any], ground_truth: List[Any], recommend_size: int) -> float:
    """
    Computes the Area Under the Curve (AUC) for ROC.

    Formula:
        AUC@k = #correct positive-negative pairs / #total positive-negative pairs

    Parameters:
        ranked_list: List of recommended items.
        ground_truth: List of relevant items.
        recommend_size: Number of items to recommend.

    Returns:
        AUC score as a float.
    """
    if not ground_truth:
        return 1.0 if not ranked_list else 0.0
    if not ranked_list:
        # no recommendations while ground truth exists
        return 0.0

    k = min(len(ranked_list), recommend_size)
    true_positives, correct_pairs = 0, 0

    # count # of pairs of items that are ranked in the correct order (i.e. TP > FP)
    for item in ranked_list[:k]:
        if item in ground_truth:
            true_positives += 1
        else:
            correct_pairs += true_positives

    false_positives = k - true_positives
    if true_positives == 0:
        return 0.0 # all pairs are incorrect
    if false_positives == 0:
        return 1.0 # all pairs are correct

    # the number of all possible <TP, FP> pairs
    n_pairs = true_positives * false_positives
    return correct_pairs / n_pairs

def average_precision(ranked_list: List[Any], ground_truth: List[Any], recommend_size: int) -> float:
    """
    Computes the Average Precision (AP) for a ranked list of items.

    Formula:
        AP = (1/R) * Σ (P(k) * rel(k))
    where:
        - R is the number of relevant items
        - P(k) is the precision at rank k
        - rel(k) is 1 if the item at rank k is relevant, else 0

    Args:
        ranked_list (list): The list of ranked items.
        ground_truth (list): The list of relevant items.
        recommend_size (int): The number of recommendations to consider.

    Returns:
        float: The Average Precision score.
    """
    if not ground_truth:
        return 1.0 if not ranked_list else 0.0

    ap_sum = 0.0
    tp_sum = 0

    # Iterate through the top-k ranked list and compute precision for relevant items
    k = min(len(ranked_list), recommend_size)
    for i, item in enumerate(ranked_list[:k]):
        if item in ground_truth:
            tp_sum += 1
            ap_sum += tp_sum / (i + 1)

    # follows replay and https://arxiv.org/abs/2206.12858
    x = min(len(ground_truth), recommend_size)
    # Avoid division by zero if no relevant items exist
    return ap_sum / x if x > 0 else 0.0

def map_score(ranked_lists: Iterable[List[Any]], ground_truths: Iterable[List[Any]], recommend_size: int) -> float:
    """
    Computes the Mean Average Precision (MAP) across multiple queries.

    Formula:
        MAP = (1 / |Q|) * sum(AP@k(q) for q in Q)

    Parameters:
        ranked_lists: List of recommended lists for each query.
        ground_truths: List of relevant items for each query.
        recommend_size: Number of items to recommend.

    Returns:
        MAP score as a float.
    """
    ap_sum = sum(average_precision(r, g, recommend_size) for r, g in zip(ranked_lists, ground_truths))
    return ap_sum / len(ranked_lists) if ranked_lists else 0.0

def compute_scores(
    evaluation_pairs: Iterable[Tuple[List[Any], List[Any]]], recommend_size: int
) -> Dict[str, float]:
    """
    Computes batch evaluation metrics across multiple queries.

    Parameters:
        evaluation_pairs: Iterable of tuples containing recommended and ground truth items for each query.
        recommend_size: Number of items to recommend.

    Returns:
        Dictionary with averaged scores across all queries.
    """
    precision_sum = recall_sum = f1_sum = ndcg_sum = rr_sum = ap_sum = auc_sum = 0.0
    hit_sum = tp_sum = 0
    num_queries = 0  # Total number of queries processed

    for ranked_list, ground_truth in evaluation_pairs:
        num_queries += 1

        # Compute individual metrics for this query
        precision_sum += precision(ranked_list, ground_truth, recommend_size)
        recall_sum += recall(ranked_list, ground_truth, recommend_size)
        f1_sum += f1_score(ranked_list, ground_truth, recommend_size)
        ndcg_sum += ndcg(ranked_list, ground_truth, recommend_size)
        hit_sum += hit(ranked_list, ground_truth, recommend_size)
        rr_sum += reciprocal_rank(ranked_list, ground_truth, recommend_size)
        auc_sum += auc(ranked_list, ground_truth, recommend_size)
        tp_sum += true_positives(ranked_list, ground_truth, recommend_size)
        ap_sum += average_precision(ranked_list, ground_truth, recommend_size)

    # Avoid division by zero if no queries are provided
    if num_queries == 0:
        return defaultdict(float)

    # Calculate averages
    return {
        "precision": precision_sum / num_queries,
        "recall": recall_sum / num_queries,
        "f1": f1_sum / num_queries,
        "ndcg": ndcg_sum / num_queries,
        "hit_rate": hit_sum / num_queries,
        "mrr": rr_sum / num_queries,
        "map": ap_sum / num_queries,
        "tp": tp_sum,
        "auc": auc_sum / num_queries,
    }
