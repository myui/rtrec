from collections import defaultdict
from typing import Dict, List

import numpy as np

def minmax_normalize(scores: np.ndarray) -> np.ndarray:
    """
    Normalize scores using min-max scaling along a specific axis.

    Args:
        scores (ndarray): The scores to normalize.

    Returns:
        ndarray: The normalized scores with values between 0 and 1.
    """
    min_val = min(scores)
    max_val = max(scores)
    # Add a small epsilon to avoid division by zero
    return (scores - min_val) / (max_val - min_val + 1e-8)

def rank_normalize(ranked_list: List[int]) -> Dict[int, float]:
    N = len(ranked_list)
    return {
        item: 1 - (rank / (N - 1)) if N > 1 else 1.0 for rank, item in enumerate(ranked_list)
    }

def weighted_borda(rankings: List[List[int]], weights: List[float]) -> Dict[int, float]:
    """
    Weighted Borda Count rank aggregation for multiple recommendation lists.

    Uses rank_normalize to convert rankings to scores, then applies weights.

    Parameters:
        rankings (List[List[int]]): List of ranked item lists (first item is most preferred)
        weights (List[float]): List of weights for each ranking

    Returns:
        Dict[int, float]: Dictionary of items with their aggregated Borda scores
    """
    if len(rankings) != len(weights):
        raise ValueError("Number of rankings must match number of weights")

    # Initialize scores dictionary
    borda_scores = defaultdict(float)

    # Process each ranking list
    for ranking, weight in zip(rankings, weights):
        if not ranking:  # Skip empty rankings
            continue

        # Convert ranking to normalized scores
        normalized_scores = rank_normalize(ranking)

        # Apply weight and add to total scores
        for item_id, score in normalized_scores.items():
            borda_scores[item_id] += score * weight

    return borda_scores
