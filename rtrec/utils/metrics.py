from typing import List, Any, Dict, Iterable, Tuple
from math import log2

def relevance(ranked_list: List[Any], ground_truth: List[Any], recommend_size: int) -> Tuple[List[int], int]:
    """
    Computes the relevance vector for the ranked list and the total number of relevant items.

    Args:
        ranked_list (List[Any]): List of recommended items.
        ground_truth (List[Any]): List of relevant items.
        recommend_size (int): Number of top items to consider.

    Returns:
        Tuple[List[int], int]: A binary relevance vector indicating relevant items and the total number of relevant items.

    Formula:
        rel[i] = 1 if ranked_list[i] in ground_truth else 0
    """
    k = min(len(ranked_list), recommend_size)
    ground_truth_set = set(ground_truth)  # Use a set for efficient membership checking
    rel = [1 if item in ground_truth_set else 0 for item in ranked_list[:k]]
    total_relevant = len(ground_truth_set)  # Total relevant items
    return rel, total_relevant

def precision(rel: List[int], k: int) -> float:
    """
    Computes Precision@k.

    Args:
        rel (List[int]): Binary relevance vector.
        k (int): Number of top items considered.

    Returns:
        float: Precision score.

    Formula:
        Precision@k = (Number of relevant items in top-k) / k
    """
    return sum(rel) / k if k > 0 else 0.0

def recall(rel: List[int], total_relevant: int) -> float:
    """
    Computes Recall.

    Args:
        rel (List[int]): Binary relevance vector.
        total_relevant (int): Total number of relevant items.

    Returns:
        float: Recall score.

    Formula:
        Recall = (Number of relevant items retrieved) / (Total relevant items)
    """
    return sum(rel) / total_relevant if total_relevant > 0 else 1.0

def true_positives(rel: List[int]) -> int:
    """
    Computes the number of true positives (tp).

    Args:
        rel (List[int]): Binary relevance vector.

    Returns:
        int: Number of true positives.

    Formula:
        TP = Sum of rel
    """
    return sum(rel)

def f1(precision_value: float, recall_value: float) -> float:
    """
    Computes the F1 score.

    Args:
        precision_value (float): Precision score.
        recall_value (float): Recall score.

    Returns:
        float: F1 score.

    Formula:
        F1 = 2 * (Precision * Recall) / (Precision + Recall)
    """
    return 2 * precision_value * recall_value / (precision_value + recall_value) if (precision_value + recall_value) > 0 else 0.0

def ndcg(rel: List[int], total_relevant: int) -> float:
    """
    Computes Normalized Discounted Cumulative Gain (nDCG).

    Args:
        rel (List[int]): Binary relevance vector.
        total_relevant (int): Total number of relevant items.

    Returns:
        float: nDCG score.

    Formula:
        DCG = Σ (rel[i] / log2(i + 2))
        nDCG = DCG / IDCG

        where IDCG is the ideal DCG, computed for the perfect ranking.
    """
    dcg = sum(r / log2(i + 2) for i, r in enumerate(rel))
    ideal_dcg = sum(1 / log2(i + 2) for i in range(min(total_relevant, len(rel))))
    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0

def hit_rate(rel: List[int]) -> float:
    """
    Computes the Hit Rate.

    Args:
        rel (List[int]): Binary relevance vector.

    Returns:
        float: Hit rate.

    Formula:
        Hit Rate = 1 if any(rel) else 0
    """
    return 1.0 if any(rel) else 0.0

def reciprocal_rank(rel: List[int]) -> float:
    """
    Computes the Reciprocal Rank.

    Args:
        rel (List[int]): Binary relevance vector.

    Returns:
        float: Reciprocal rank.

    Formula:
        Reciprocal Rank = 1 / (Rank of the first relevant item)
    """
    for i, r in enumerate(rel):
        if r == 1:
            return 1.0 / (i + 1)
    return 0.0

def average_precision(rel: List[int], total_relevant: int) -> float:
    """
    Computes Average Precision (AP).

    Args:
        rel (List[int]): Binary relevance vector.
        total_relevant (int): Total number of relevant items.

    Returns:
        float: Average precision.

    Formula:
        AP = Σ (Precision@i * rel[i]) / Total relevant items
    """
    precision_sum, true_positive = 0.0, 0
    for i, r in enumerate(rel):
        if r == 1:
            true_positive += 1
            precision_sum += true_positive / (i + 1)
    return precision_sum / total_relevant if total_relevant > 0 else 0.0

def compute_scores(
    ranked_lists: Iterable[List[Any]], ground_truths: Iterable[List[Any]], recommend_size: int
) -> Dict[str, float]:
    """
    Computes aggregate metrics over multiple queries, including MRR, MAP, and tp.

    Args:
        ranked_lists (Iterable[List[Any]]): Iterable of recommended item lists.
        ground_truths (Iterable[List[Any]]): Iterable of ground truth item lists.
        recommend_size (int): Number of top items to consider.

    Returns:
        Dict[str, float]: Dictionary with average scores across queries.
    """
    precision_sum = recall_sum = f1_sum = ndcg_sum = hit_sum = 0.0
    rr_sum = ap_sum = tp_sum = 0
    num_queries = 0

    for ranked_list, ground_truth in zip(ranked_lists, ground_truths):
        rel, total_relevant = relevance(ranked_list, ground_truth, recommend_size)

        precision_value = precision(rel, len(ranked_list))
        recall_value = recall(rel, total_relevant)
        f1_value = f1(precision_value, recall_value)
        ndcg_value = ndcg(rel, total_relevant)
        hit = hit_rate(rel)
        rr = reciprocal_rank(rel)
        ap = average_precision(rel, total_relevant)
        tp = true_positives(rel)

        precision_sum += precision_value
        recall_sum += recall_value
        f1_sum += f1_value
        ndcg_sum += ndcg_value
        hit_sum += hit
        rr_sum += rr
        ap_sum += ap
        tp_sum += tp
        num_queries += 1

    if num_queries == 0:
        return {
            "precision": 0.0, "recall": 0.0, "f1": 0.0, "ndcg": 0.0,
            "hit": 0.0, "mrr": 0.0, "map": 0.0, "tp": 0
        }

    return {
        "precision": precision_sum / num_queries,
        "recall": recall_sum / num_queries,
        "f1": f1_sum / num_queries,
        "ndcg": ndcg_sum / num_queries,
        "hit": hit_sum / num_queries,
        "mrr": rr_sum / num_queries,
        "map": ap_sum / num_queries,
        "tp": tp_sum / num_queries
    }
