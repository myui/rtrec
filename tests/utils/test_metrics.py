import pytest
from rtrec.utils.metrics import (
    relevance,
    precision,
    recall,
    true_positives,
    f1,
    ndcg,
    hit_rate,
    reciprocal_rank,
    average_precision,
    auc,
    compute_scores,
)


@pytest.mark.parametrize("ranked_list, ground_truth, recommend_size, expected_rel, expected_total_relevant", [
    ([1, 2, 3], [2, 3], 3, [0, 1, 1], 2),
    ([1, 4, 3], [2, 3], 3, [0, 0, 0], 2),
    ([1, 2, 3, 4], [2, 4], 2, [0, 1], 2),
])
def test_relevance(ranked_list, ground_truth, recommend_size, expected_rel, expected_total_relevant):
    rel, total_relevant = relevance(ranked_list, ground_truth, recommend_size)
    assert rel == expected_rel
    assert total_relevant == expected_total_relevant

@pytest.mark.parametrize("rel, k, expected_precision", [
    ([1, 1, 0, 1], 3, 2/3),
    ([1, 0, 0, 0], 1, 1.0),
    ([0, 0, 0, 0], 4, 0.0),
])
def test_precision(rel, k, expected_precision):
    assert precision(rel, k) == expected_precision

@pytest.mark.parametrize("rel, total_relevant, expected_recall", [
    ([1, 1, 0, 1], 3, 2/3),
    ([1, 0, 0, 0], 1, 1.0),
    ([0, 0, 0, 0], 0, 0.0),
])
def test_recall(rel, total_relevant, expected_recall):
    assert recall(rel, total_relevant) == expected_recall

@pytest.mark.parametrize("rel, expected_tp", [
    ([1, 1, 0, 1], 3),
    ([0, 0, 0, 0], 0),
])
def test_true_positives(rel, expected_tp):
    assert true_positives(rel) == expected_tp

@pytest.mark.parametrize("precision_value, recall_value, expected_f1", [
    (1.0, 1.0, 1.0),
    (0.0, 1.0, 0.0),
    (0.5, 0.5, 0.5),
])
def test_f1(precision_value, recall_value, expected_f1):
    assert f1(precision_value, recall_value) == expected_f1

@pytest.mark.parametrize("rel, total_relevant, expected_ndcg", [
    ([1, 1, 0, 1], 3, 0.5),
    ([1, 0, 0, 0], 1, 1.0),
    ([0, 0, 0, 0], 0, 0.0),
])
def test_ndcg(rel, total_relevant, expected_ndcg):
    assert ndcg(rel, total_relevant) == expected_ndcg

@pytest.mark.parametrize("rel, expected_hit_rate", [
    ([1, 1, 0, 1], 1.0),
    ([0, 0, 0, 0], 0.0),
])
def test_hit_rate(rel, expected_hit_rate):
    assert hit_rate(rel) == expected_hit_rate

@pytest.mark.parametrize("rel, expected_reciprocal_rank", [
    ([1, 0, 0, 1], 0.5),
    ([0, 0, 0, 0], 0.0),
])
def test_reciprocal_rank(rel, expected_reciprocal_rank):
    assert reciprocal_rank(rel) == expected_reciprocal_rank

@pytest.mark.parametrize("rel, total_relevant, expected_ap", [
    ([1, 1, 0, 1], 3, 0.5),
    ([1, 0, 0, 0], 1, 1.0),
    ([0, 0, 0, 0], 0, 0.0),
])
def test_average_precision(rel, total_relevant, expected_ap):
    assert average_precision(rel, total_relevant) == expected_ap

@pytest.mark.parametrize("rel, expected_auc", [
    ([1, 0, 1, 0], 0.5),
    ([1, 1, 0, 0], 1.0),
])
def test_auc(rel, expected_auc):
    assert auc(rel) == expected_auc

def test_compute_scores():
    ranked_lists = [
        [1, 2, 3],
        [1, 4, 5]
    ]
    ground_truths = [
        [2, 3],
        [1, 5]
    ]
    recommend_size = 3

    scores = compute_scores(ranked_lists, ground_truths, recommend_size)

    assert isinstance(scores, dict)
    assert "precision" in scores
    assert "recall" in scores
    assert "f1" in scores
    assert "ndcg" in scores
    assert "hit_rate" in scores
    assert "mrr" in scores
    assert "map" in scores
    assert "tp" in scores
    assert "auc" in scores

    assert scores["tp"] == 4  # 4 total true positives across both queries
    assert scores["precision"] >= 0.0
    assert scores["recall"] >= 0.0
    assert scores["f1"] >= 0.0
    assert scores["ndcg"] >= 0.0
    assert scores["hit_rate"] in [0.0, 1.0]  # hit rate can only be 0 or 1
    assert scores["mrr"] >= 0.0
    assert scores["map"] >= 0.0
    assert scores["auc"] >= 0.0

# Run the tests
if __name__ == "__main__":
    pytest.main()
