from typing import List, Any, Dict, Iterable, Tuple
from math import log2

def relevance(ranked_list: List[Any], ground_truth: List[Any], recommend_size: int) -> Tuple[List[int], int]:
    """
    Determine the relevance of items in the ranked list against the ground truth.

    Parameters:
        ranked_list (List[Any]): The list of ranked items.
        ground_truth (List[Any]): The ground truth relevant items.
        recommend_size (int): The number of recommendations to consider.

    Returns:
        Tuple[List[int], int]: A list indicating relevance (1 for relevant, 0 for non-relevant)
                                and the total number of relevant items.

    Formula:
        rel[i] = 1 if ranked_list[i] ∈ ground_truth else 0
        total_relevant = |ground_truth|
    """
    k = min(len(ranked_list), recommend_size)
    ground_truth_set = set(ground_truth)
    rel = [1 if item in ground_truth_set else 0 for item in ranked_list[:k]]
    total_relevant = len(ground_truth_set)
    return rel, total_relevant

def precision(rel: List[int], k: int) -> float:
    """Calculate precision at k.

    Parameters:
        rel (List[int]): Relevance list for the ranked items.
        k (int): The number of items considered.

    Returns:
        float: The precision score.

    Formula:
        Precision = TP / k
    """
    return sum(rel) / k if k > 0 else 0.0

def recall(rel: List[int], total_relevant: int) -> float:
    """Calculate recall.

    Parameters:
        rel (List[int]): Relevance list for the ranked items.
        total_relevant (int): Total number of relevant items.

    Returns:
        float: The recall score.

    Formula:
        Recall = TP / total_relevant
    """
    return sum(rel) / total_relevant if total_relevant > 0 else 0.0

def true_positives(rel: List[int]) -> int:
    """Count true positives.

    Parameters:
        rel (List[int]): Relevance list for the ranked items.

    Returns:
        int: The number of true positive items.

    Formula:
        TP = sum(rel)
    """
    return sum(rel)

def f1(precision_value: float, recall_value: float) -> float:
    """Calculate F1 score.

    Parameters:
        precision_value (float): Precision score.
        recall_value (float): Recall score.

    Returns:
        float: The F1 score.

    Formula:
        F1 = 2 * (Precision * Recall) / (Precision + Recall)
    """
    return 2 * precision_value * recall_value / (precision_value + recall_value) if (precision_value + recall_value) > 0 else 0.0

def ndcg(rel: List[int], total_relevant: int) -> float:
    """Calculate Normalized Discounted Cumulative Gain (NDCG).

    Parameters:
        rel (List[int]): Relevance list for the ranked items.
        total_relevant (int): Total number of relevant items.

    Returns:
        float: The NDCG score.

    Formula:
        DCG = sum(rel[i] / log2(i + 2) for i in range(len(rel)))
        IDCG = sum(1 / log2(i + 2) for i in range(min(total_relevant, len(rel))))
        NDCG = DCG / IDCG
    """
    dcg = sum(r / log2(i + 2) for i, r in enumerate(rel))
    ideal_dcg = sum(1 / log2(i + 2) for i in range(min(total_relevant, len(rel))))
    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0

def hit_rate(rel: List[int]) -> float:
    """Calculate hit rate.

    Parameters:
        rel (List[int]): Relevance list for the ranked items.

    Returns:
        float: The hit rate.

    Formula:
        Hit Rate = 1 if any(rel) else 0
    """
    return 1.0 if any(rel) else 0.0

def reciprocal_rank(rel: List[int]) -> float:
    """Calculate reciprocal rank.

    Parameters:
        rel (List[int]): Relevance list for the ranked items.

    Returns:
        float: The reciprocal rank.

    Formula:
        Reciprocal Rank = 1 / (first relevant rank)
    """
    for i, r in enumerate(rel):
        if r == 1:
            return 1.0 / (i + 1)
    return 0.0

def average_precision(rel: List[int], total_relevant: int) -> float:
    """Calculate average precision.

    Parameters:
        rel (List[int]): Relevance list for the ranked items.
        total_relevant (int): Total number of relevant items.

    Returns:
        float: The average precision score.

    Formula:
        Average Precision = sum(P(i) for each relevant i) / total_relevant
        where P(i) = TP(i) / (i + 1)
    """
    precision_sum = 0.0
    true_positive = 0
    for i, r in enumerate(rel):
        if r == 1:
            true_positive += 1
            precision_sum += true_positive / (i + 1)
    return precision_sum / total_relevant if total_relevant > 0 else 0.0

def auc(rel: List[int]) -> float:
    """Calculate Area Under the Curve (AUC).

    Parameters:
        rel (List[int]): Relevance list for the ranked items.

    Returns:
        float: The AUC score.

    Formula:
        AUC = (number of correct pairs) / (total pairs)
    """
    n_true_positive = 0   # Count of true positives
    correct_pairs = 0     # Count of correct positive-negative pairs

    for item in rel:
        if item == 1:
            n_true_positive += 1
        else:
            # For each false positive, add the current count of true positives
            correct_pairs += n_true_positive

    # Total pairs are the product of true positives and false positives
    total_pairs = n_true_positive * (len(rel) - n_true_positive)

    return correct_pairs / total_pairs if total_pairs > 0 else 0.0

def compute_scores(
    ranked_lists: Iterable[List[Any]], ground_truths: Iterable[List[Any]], recommend_size: int
) -> Dict[str, float]:
    """
    Compute all evaluation scores for the given ranked lists and ground truths.

    Parameters:
        ranked_lists (Iterable[List[Any]]): Iterable of ranked lists.
        ground_truths (Iterable[List[Any]]): Iterable of ground truth lists.
        recommend_size (int): The number of recommendations to consider.

    Returns:
        Dict[str, float]: A dictionary containing computed scores.

    Each score in the returned dictionary represents an average over all queries:
        precision, recall, F1, NDCG, hit rate, mean reciprocal rank (MRR),
        mean average precision (MAP), total true positives (TP), and area under the curve (AUC).

    Note:
        `ranked_lists` and `ground_truths` must have the same length.
    """
    precision_sum = recall_sum = f1_sum = ndcg_sum = hit_sum = rr_sum = ap_sum = auc_sum = 0.0
    tp_sum = 0
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
        auc_value = auc(rel)

        precision_sum += precision_value
        recall_sum += recall_value
        f1_sum += f1_value
        ndcg_sum += ndcg_value
        hit_sum += hit
        rr_sum += rr
        ap_sum += ap
        tp_sum += tp
        auc_sum += auc_value
        num_queries += 1

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
