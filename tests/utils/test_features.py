import numpy as np
import pytest
from scipy.sparse import csr_matrix
from rtrec.utils.features import FeatureStore

# Test adding user features
def test_put_user_feature():
    features = FeatureStore()
    features.put_user_feature(1, ["tag1", "tag2"])
    assert 1 in features.user_feature_map
    assert set(features.user_feature_map[1]) == {0, 1}  # Assuming tag IDs start at 0

def test_put_user_feature_replacement():
    features = FeatureStore()
    features.put_user_feature(1, ["tag1", "tag2"])
    features.put_user_feature(1, ["tag3", "tag4"])
    assert 1 in features.user_feature_map
    assert set(features.user_feature_map[1]) == {2, 3}  # Updated IDs for new tags

# Test adding item features
def test_put_item_feature():
    features = FeatureStore()
    features.put_item_feature(1, ["item_tag1", "item_tag2"])
    assert 1 in features.item_feature_map
    assert set(features.item_feature_map[1]) == {0, 1}

def test_put_item_feature_replacement():
    features = FeatureStore()
    features.put_item_feature(1, ["item_tag1", "item_tag2"])
    features.put_item_feature(1, ["item_tag3", "item_tag4"])
    assert 1 in features.item_feature_map
    assert set(features.item_feature_map[1]) == {2, 3}

# Test getting user feature representation
def test_get_user_feature_repr():
    features = FeatureStore()
    features.put_user_feature(1, ["tag1", "tag2"])
    user_repr = features.get_user_feature_repr(["tag1", "tag2"])
    expected_matrix = csr_matrix(([1, 1], ([0, 0], [0, 1])), shape=(1, 2))
    assert (user_repr != expected_matrix).nnz == 0  # Check equality

def test_get_user_feature_repr_non_existent_tag():
    features = FeatureStore()
    features.put_user_feature(1, ["tag1", "tag2"])
    user_repr = features.get_user_feature_repr(["tag3"])
    expected_matrix = csr_matrix(([], ([], [])), shape=(1, 2))
    assert (user_repr != expected_matrix).nnz == 0

# Test getting item feature representation
def test_get_item_feature_repr():
    features = FeatureStore()
    features.put_item_feature(1, ["item_tag1", "item_tag2"])
    item_repr = features.get_item_feature_repr(["item_tag1", "item_tag2"])
    expected_matrix = csr_matrix(([1, 1], ([0, 0], [0, 1])), shape=(1, 2))
    assert (item_repr != expected_matrix).nnz == 0

def test_get_item_feature_repr_non_existent_tag():
    features = FeatureStore()
    features.put_item_feature(1, ["item_tag1", "item_tag2"])
    item_repr = features.get_item_feature_repr(["item_tag3"])
    expected_matrix = csr_matrix(([], ([], [])), shape=(1, 2))
    assert (item_repr != expected_matrix).nnz == 0

# Test building user features matrix
def test_build_user_features_matrix():
    features = FeatureStore()
    features.put_user_feature(0, ["tag1", "tag2"])
    features.put_user_feature(1, ["tag2", "tag3"])
    user_matrix = features.build_user_features_matrix()
    expected_matrix = csr_matrix(np.array([[1, 1, 0], [0, 1, 1]]))
    assert (user_matrix != expected_matrix).nnz == 0

def test_build_user_features_matrix_with_user_id():
    features = FeatureStore()
    features.put_user_feature(0, ["tag1", "tag2"])
    features.put_user_feature(1, ["tag2", "tag3"])
    user_matrix = features.build_user_features_matrix(0)
    assert user_matrix.nnz == 2
    expected_matrix = csr_matrix(np.array([[1, 1, 0], [0, 0, 0]]), shape=(2, 3))
    assert (user_matrix != expected_matrix).nnz == 0
    # not exist user id
    user_matrix = features.build_user_features_matrix(2)
    assert user_matrix.nnz == 0
    expected_matrix = csr_matrix(np.array([[0, 0, 0], [0, 0, 0]]), shape=(2, 3))
    assert (user_matrix != expected_matrix).nnz == 0

# Test building item features matrix
def test_build_item_features_matrix():
    features = FeatureStore()
    features.put_item_feature(0, ["item_tag1", "item_tag2"])
    features.put_item_feature(1, ["item_tag2", "item_tag3"])
    item_matrix = features.build_item_features_matrix()
    expected_matrix = csr_matrix(np.array([[1, 1, 0], [0, 1, 1]]))
    assert (item_matrix != expected_matrix).nnz == 0

def test_build_item_features_matrix_with_item_id():
    features = FeatureStore()
    features.put_item_feature(0, ["item_tag1", "item_tag2"])
    features.put_item_feature(1, ["item_tag2", "item_tag3"])
    item_matrix = features.build_item_features_matrix(0)
    assert item_matrix.nnz == 2
    expected_matrix = csr_matrix(np.array([[1, 1, 0], [0, 0, 0]]), shape=(2, 3))
    assert (item_matrix != expected_matrix).nnz == 0
    # not exist item id
    item_matrix = features.build_item_features_matrix(2)
    assert item_matrix.nnz == 0
    expected_matrix = csr_matrix(np.array([[0, 0, 0], [0, 0, 0]]), shape=(2, 3))
    assert (item_matrix != expected_matrix).nnz == 0

# Run tests using pytest if this file is executed directly
if __name__ == "__main__":
    pytest.main()
