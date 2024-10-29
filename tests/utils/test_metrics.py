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
    ([1, 3, 2, 6], [1, 2, 4], 4, (1/1 + 2/3) / 2),
    ([1, 3, 2, 6], [1, 2, 4], 3, (1/1 + 2/3) / 2),
    ([1, 3, 2, 6], [1, 2, 4], 2, (1/1) / 1),
    ([3, 1, 2, 6], [1, 2, 4], 2, (1/2) / 1),
    ([3, 1], [1, 2], 1, 0),
    ([3, 1], [1, 2], 2, (1/1) / 2),
    ([1 ,3, 2, 6, 4, 5], [1, 2, 4], 6, (1/1 + 2/3 + 3/5) / 3),
    ([1 ,3, 2, 4, 6, 5], [1, 2, 4], 6, (1/1 + 2/3 + 3/4) / 3),
])
def test_average_precision(ranked_list, ground_truth, k, expected):
    assert average_precision(ranked_list, ground_truth, k) == pytest.approx(expected, rel=1e-4)

@pytest.mark.parametrize("ranked_list, ground_truth, k, expected", [
    ([1, 3, 2, 6], [1, 2, 4], 4, 0.75),
    ([1, 3, 2, 6], [1, 2, 4], 2, 1.0),
    ([1, 3, 2, 6], [1, 3, 2, 6], 4, 0.5),  # meaningless case: all TPs
    ([1, 3, 2, 6], [7, 8, 9, 10], 4, 0.5), # meaningless case: all FPs
    ([1, 2, 3, 4, 5], [1, 3, 5], 5, 3 / 6),
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
