from typing import List, Any
from math import log

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
    def log2(x: int) -> float:
        return log(x, 2)

    def idcg(n: int) -> float:
        return sum(1 / log2(i + 2) for i in range(n))

    k = min(len(ranked_list), recommend_size)
    dcg = sum(1 / log2(i + 2) for i in range(k) if ranked_list[i] in ground_truth)
    ideal_dcg = idcg(min(len(ground_truth), k))
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
    k = min(len(ranked_list), recommend_size)
    true_positive = sum(1 for i in range(k) if ranked_list[i] in ground_truth)
    return true_positive / len(ground_truth) if ground_truth else 1.0

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
    k = min(len(ranked_list), recommend_size)
    return 1.0 if any(ranked_list[i] in ground_truth for i in range(k)) else 0.0

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

def mrr(ranked_lists: List[List[Any]], ground_truths: List[List[Any]], recommend_size: int) -> float:
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

def average_precision(ranked_list: List[Any], ground_truth: List[Any], recommend_size: int) -> float:
    """
    Computes Average Precision (AP)@k.

    Formula:
        AP@k = (1 / |relevant|) * sum(Precision@i * rel(i)) for i in range(k)

    where rel(i) = 1 if the item at rank i is relevant, else 0.

    Parameters:
        ranked_list: List of recommended items.
        ground_truth: List of relevant items.
        recommend_size: Number of items to recommend.

    Returns:
        Average Precision as a float.
    """
    k = min(len(ranked_list), recommend_size)
    relevant_count = 0
    ap_sum = 0.0

    for i in range(k):
        if ranked_list[i] in ground_truth:
            relevant_count += 1
            ap_sum += relevant_count / (i + 1)

    return ap_sum / len(ground_truth) if ground_truth else 0.0

def map_score(ranked_lists: List[List[Any]], ground_truths: List[List[Any]], recommend_size: int) -> float:
    """
    Computes Mean Average Precision (MAP) across multiple queries.

    Formula:
        MAP@k = (1 / |Q|) * sum(AP@k(q) for q in Q)

    Parameters:
        ranked_lists: List of recommended lists for each query.
        ground_truths: List of relevant items for each query.
        recommend_size: Number of items to recommend.

    Returns:
        MAP score as a float.
    """
    ap_sum = sum(average_precision(r, g, recommend_size) for r, g in zip(ranked_lists, ground_truths))
    return ap_sum / len(ranked_lists) if ranked_lists else 0.0

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
    k = min(len(ranked_list), recommend_size)
    true_positive, correct_pairs = 0, 0

    for i in range(k):
        if ranked_list[i] in ground_truth:
            true_positive += 1
        else:
            correct_pairs += true_positive

    n_pairs = true_positive * (k - true_positive)
    return correct_pairs / n_pairs if n_pairs > 0 else 0.5

def coverage(ranked_list: List[Any], all_items: List[Any]) -> float:
    """
    Computes Coverage@k, the proportion of all items that appear in recommendations.

    Formula:
        Coverage@k = |unique recommended items| / |all items|

    Parameters:
        ranked_list: List of recommended items.
        all_items: List of all possible items.

    Returns:
        Coverage score as a float.
    """
    unique_recommended = set(ranked_list)
    return len(unique_recommended) / len(all_items) if all_items else 0.0

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
