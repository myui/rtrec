import pytest
from math import log2
from rtrec.utils.metrics import (
    precision,
    recall,
    f1_score,
    ndcg,
    hit,
    reciprocal_rank,
    average_precision,
    auc,
    true_positives,
)

@pytest.mark.parametrize("ranked_list, ground_truth, k, expected", [
    ([1, 3, 2, 6], [1, 2, 4], 4, 0.7039),
    ([1, 3, 2, 6], [1, 2, 4], 2, 0.6131),
    ([3, 2, 1, 6], [1], 2, 0.0),
    ([3, 2, 1, 6], [1], 3, 0.5),
    ([1, 2, 3, 4], [1, 2, 3], 3, 1.0), # optimal ranking
    ([1, 3, 2, 4], [1, 2], 3, (1/log2(1+1) + 1/log2(3+1)) / (1/log2(1+1) + 1/log2(2+1))), # suboptimal ranking
    ([5, 6, 7], [1, 2, 3], 3, 0.0), # no relevant items
])
def test_ndcg(ranked_list, ground_truth, k, expected):
    assert ndcg(ranked_list, ground_truth, k) == pytest.approx(expected, rel=1e-4)

@pytest.mark.parametrize("ranked_list, ground_truth, k, expected", [
    ([1, 3, 2, 6], [1, 2, 4], 4, 0.66666),
    ([1, 3, 2, 6], [1, 2, 4], 2, 0.33333),
    ([], [], 2, 1.0),
    ([1, 3, 2], [], 2, 0.0)
])
def test_recall(ranked_list, ground_truth, k, expected):
    assert recall(ranked_list, ground_truth, k) == pytest.approx(expected, rel=1e-4)

@pytest.mark.parametrize("ranked_list, ground_truth, recommend_size, expected", [
    ([1, 2, 3], [1, 2, 3], 3, 1.0),            # Perfect match
    ([1, 2, 3], [2, 3, 4], 3, 0.66666),        # Partial match
    ([1, 2, 3], [4, 5, 6], 3, 0.0),            # No match
    ([], [1, 2, 3], 3, 0.0),                   # Empty recommendation
    ([1, 2, 3], [], 3, 0.0),                   # Empty ground truth
    ([], [], 3, 1.0)                           # Both empty
])
def test_f1_score(ranked_list, ground_truth, recommend_size, expected):
    """Test the f1_score function with various cases."""
    assert pytest.approx(f1_score(ranked_list, ground_truth, recommend_size), rel=1e-4) == expected

@pytest.mark.parametrize("ranked_list, ground_truth, k, expected", [
    ([1, 3, 2, 6], [1, 2, 4], 4, 0.5),
    ([1, 3, 2, 6], [1, 2, 4], 2, 0.5),
    ([], [], 2, 1.0),
    ([1, 3, 2], [], 2, 0.0)
])
def test_precision(ranked_list, ground_truth, k, expected):
    assert precision(ranked_list, ground_truth, k) == pytest.approx(expected, rel=1e-4)

@pytest.mark.parametrize("ranked_list, ground_truth, k, expected", [
    ([1, 3, 2, 6], [1, 2, 4], 4, 1.0),
    ([6, 2, 3, 1], [1, 2, 4], 4, 0.5),
    ([6, 2, 3, 1], [1, 2, 4], 1, 0.0)
])
def test_reciprocal_rank(ranked_list, ground_truth, k, expected):
    assert reciprocal_rank(ranked_list, ground_truth, k) == pytest.approx(expected, rel=1e-4)

@pytest.mark.parametrize("ranked_list, ground_truth, k, expected", [
    ([1, 3, 2, 6], [1, 2, 4], 4, 1.0),
    ([1, 3, 2, 6], [1, 2, 4], 2, 1.0),
    ([5, 6], [1, 2, 4], 2, 0.0)
])
def test_hit(ranked_list, ground_truth, k, expected):
    assert hit(ranked_list, ground_truth, k) == pytest.approx(expected, rel=1e-4)

@pytest.mark.parametrize("ranked_list, ground_truth, k, expected", [
    ([1, 3, 2, 6], [1, 2, 4], 4, (1/1 + 2/3) / 3),
    ([1, 3, 2, 6], [1, 2, 4], 3, (1/1 + 2/3) / 3),
    ([1, 3, 2, 6], [1, 2, 4], 2, (1/1) / 2),
    ([3, 1, 2, 6], [1, 2, 4], 2, (1/2) / 2),
    ([3, 1], [1, 2], 1, 0),
    ([3, 1], [1, 2], 2, (1/2) / 2),
    ([1 ,3, 2, 6, 4, 5], [1, 2, 4], 6, (1/1 + 2/3 + 3/5) / 3),
    ([1 ,3, 2, 4, 6, 5], [1, 2, 4], 6, (1/1 + 2/3 + 3/4) / 3),
    ([1, 2, 3, 4, 5], [1, 3, 2, 4], 3, (1/1 + 2/2 + 3/3) / 3),
    ([1, 2, 3, 4, 5], [1, 3, 4, 2], 3, (1/1 + 2/2 + 3/3) / 3),
    ([1, 2, 3, 4, 5], [1, 3, 4], 3, (1/1 + 2/3) / 3),
    ([1, 2], [1, 2, 3], 3, (1/1 + 2/2) / 3),
    ([1, 2, 3, 4, 5], [2, 4, 5, 10], 5, (1/2 + 2/4 + 3/5) / 4),
    ([1, 2, 3, 4, 5], [2, 4, 5, 10], 4, (1/2 + 2/4) / 4),
    ([10, 5, 2, 4, 3], [2, 4, 5, 10], 5, 1.0),
    ([1, 3, 6, 7, 8], [2, 4, 5, 10], 4, 0.0),
    ([11, 12, 13, 14, 15, 16, 2, 4, 5, 10], [2, 4, 5, 10], 10, (1/7 + 2/8 + 3/9 + 4/10) / 4),
    ([11, 12, 13, 14, 15, 16, 2, 4, 5, 10], [2, 4, 5, 10], 4, 0.0),
    ([2, 11, 12, 13, 14, 15, 4, 5, 10, 16], [2, 4, 5, 10], 11, (1/1 + 2 / 7 + 3 / 8 + 4 / 9) / 4),
    ([2, 11, 12, 13, 14, 15, 4, 5, 10, 16], [2, 4, 5, 10], 4, (1/1) / 4),
])
def test_average_precision(ranked_list, ground_truth, k, expected):
    assert average_precision(ranked_list, ground_truth, k) == pytest.approx(expected, rel=1e-4)

@pytest.mark.parametrize("ranked_list, ground_truth, k, expected", [
    # Perfect prediction with full overlap
    ([1, 2, 3, 4, 5], [1, 2, 3, 4, 5], 5, 1.0),
    # Relevant items: [1, 2, 3]
    # Irrelevant items: [6, 7]
    # Pairs to Compare: (1, 6), (1, 7), (2, 6), (2, 7), (3, 6), (3, 7)
    ([1, 2, 3, 6, 7], [1, 2, 3, 4, 5], 5, 6 / 6),
    # Relevant Items in ranked_list: [1, 2, 6, 7]
    # Irrelevant Items in ranked_list: [3]
    # Pairs to Compare: (1, 3), (2, 3), (6, 3), (7, 3)
    ([1, 2, 3, 6, 7], [1, 2, 6, 4, 7], 5, 2 / 4),
    ([1, 2, 3, 4, 5], [1, 3, 5], 5, 0.5),
    # Non-overlapping ranked list
    ([6, 7, 8, 9, 10], [1, 2, 3, 4, 5], 5, 0.0),
    # Perfect prediction but with k < total items
    ([1, 2, 3, 4, 5], [1, 2, 3, 4, 5], 3, 1.0),
    # Empty ground truth
    ([1, 2, 3, 4, 5], [], 5, 0.0),
    # Empty ranked list
    ([], [1, 2, 3, 4, 5], 5, 0.0),
    # Larger k than both lists
    ([1, 2, 3], [1, 2, 3], 10, 1.0),
    # Pairs: (2, 1), (2, 3), (2, 5), (4, 1), (4, 3), (4, 5)
    ([1, 2, 3, 4, 5], [2, 4], 5, 3 / 6),
    # Relevant items: [b, d]
    # Irrelevant items: [a, c, e]
    # Pairs: (b, a), (b, c), (b, e), (d, a), (d, c), (d, e)
    (['a', 'b', 'c', 'd', 'e'], ['b', 'd'], 5, 3 / 6),
    # Relevant items: [b]
    # Irrelevant items: [a]
    # Pair: (b, a)
    (['a', 'b', 'c', 'd', 'e'], ['b', 'd'], 2, 0 / 1),
    # No irrelevant items in ranked list
    (['b', 'd'], ['a', 'b', 'c', 'd', 'e'], 5, 1.0),
    (['b', 'd'], ['a', 'b', 'c', 'd', 'e'], 2, 1.0),
    # more tests
    ([1, 3, 2, 6], [1, 2, 4], 4, 0.75),
    ([1, 3, 2, 6], [1, 2, 4], 2, 1.0),
    ([1, 3, 2, 6], [1, 3, 2, 6], 4, 1.0),   # optimal ranking
    ([1, 3, 2, 6], [7, 8, 9, 10], 4, 0.0),  # worst ranking
    ([1, 2, 3, 4, 5], [1, 3, 5], 5, 3 / 6),
    ([], [], 4, 1.0),                       # both empty
    ([1, 2, 3], [], 3, 0.0),                # no grand truth
    ([], [1, 2, 3], 3, 0.0),                # no recommendations
])
def test_auc(ranked_list, ground_truth, k, expected):
    assert auc(ranked_list, ground_truth, k) == pytest.approx(expected, rel=1e-4)

@pytest.mark.parametrize("ranked_list, ground_truth, recommend_size, expected", [
    ([1, 2, 3], [1, 2, 3], 3, 3),             # All recommended items are true positives
    ([1, 2, 3], [2, 3, 4], 3, 2),             # Two true positives
    ([1, 2, 3], [4, 5, 6], 3, 0),             # No true positives
    ([1, 2, 3], [1], 3, 1),                   # One true positive
    ([1, 2, 3], [], 3, 0),                    # No ground truth
    ([], [1, 2, 3], 3, 0),                    # No recommendations
    ([], [], 3, 0),                           # Both empty
])
def test_true_positives(ranked_list, ground_truth, recommend_size, expected):
    """Test the true_positives function with various cases."""
    assert true_positives(ranked_list, ground_truth, recommend_size) == expected

if __name__ == "__main__":
    pytest.main()
