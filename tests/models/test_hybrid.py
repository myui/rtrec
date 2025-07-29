import pytest
import numpy as np

from rtrec.models.hybrid import HybridSlimFM


@pytest.fixture
def model():
    model = HybridSlimFM()
    return model


@pytest.fixture  
def model_with_data():
    model = HybridSlimFM()
    interactions = [
        (0, 0, 1.0, 1.0), 
        (0, 1, 2.0, 1.0), 
        (1, 1, 3.0, 1.0),
        (1, 2, 4.0, 1.0),
        (2, 0, 5.0, 1.0),
        (2, 2, 6.0, 1.0)
    ]
    model.fit(interactions)
    return model


def test_hybrid_batch_recommendation_no_features(model_with_data):
    """Test batch recommendations when no features are registered (SLIM-only path)."""
    user_ids = [0, 1, 2]
    top_k = 2
    
    # Get batch recommendations
    batch_results = model_with_data.recommend_batch(user_ids, top_k=top_k)
    
    # Verify we get results for all users
    assert len(batch_results) == len(user_ids)
    
    # Verify each user gets up to top_k recommendations
    for user_result in batch_results:
        assert len(user_result) <= top_k
        # Verify all results are integers (item IDs)
        assert all(isinstance(item_id, int) for item_id in user_result)


def test_hybrid_batch_vs_individual_consistency(model_with_data):
    """Test that batch and individual recommendations produce identical results."""
    user_ids = [0, 1, 2]
    top_k = 2
    
    # Get batch recommendations
    batch_results = model_with_data.recommend_batch(user_ids, top_k=top_k)
    
    # Get individual recommendations
    individual_results = []
    for user_id in user_ids:
        individual_result = model_with_data.recommend(user_id, top_k=top_k)
        individual_results.append(individual_result)
    
    # Verify they match
    assert batch_results == individual_results


def test_hybrid_batch_with_features():
    """Test batch recommendations with user features registered."""
    model = HybridSlimFM()
    
    # Register features using the add method
    model.feature_store.user_features.add('young')
    model.feature_store.user_features.add('old')
    model.feature_store.item_features.add('movie')
    model.feature_store.item_features.add('book')
    
    # Add interactions with string IDs to test feature path
    interactions = [
        ('user0', 'item0', 1.0, 1.0), 
        ('user0', 'item1', 2.0, 1.0), 
        ('user1', 'item1', 3.0, 1.0),
        ('user1', 'item2', 4.0, 1.0),
        ('user2', 'item0', 5.0, 1.0),
        ('user2', 'item2', 6.0, 1.0)
    ]
    model.fit(interactions)
    
    # Test batch recommendations with user tags
    users = ['user0', 'user1', 'user2']
    users_tags = [['young'], ['old'], ['young']]
    top_k = 2
    
    batch_results = model.recommend_batch(users, users_tags=users_tags, top_k=top_k)
    
    # Verify we get results for all users
    assert len(batch_results) == len(users)
    
    # Verify each user gets recommendations
    for user_result in batch_results:
        assert len(user_result) <= top_k



def test_hybrid_batch_single_user(model_with_data):
    """Test batch recommendations with single user."""
    user_ids = [0]
    top_k = 2
    
    batch_results = model_with_data.recommend_batch(user_ids, top_k=top_k)
    individual_result = model_with_data.recommend(user_ids[0], top_k=top_k)
    
    assert len(batch_results) == 1
    assert batch_results[0] == individual_result


def test_hybrid_batch_with_candidate_items(model_with_data):
    """Test batch recommendations with candidate item filtering."""
    user_ids = [0, 1]
    candidate_items = [0, 2]  # Limit to only items 0 and 2
    top_k = 2
    
    batch_results = model_with_data.recommend_batch(
        user_ids, 
        candidate_items=candidate_items,  # Use correct parameter name
        top_k=top_k
    )
    
    # Verify we get results for all users
    assert len(batch_results) == len(user_ids)
    
    # Verify recommendations only contain candidate items
    for user_result in batch_results:
        for item_id in user_result:
            assert item_id in candidate_items


def test_hybrid_batch_filter_interacted(model_with_data):
    """Test batch recommendations with interaction filtering."""
    user_ids = [0, 1]
    top_k = 3
    
    # Test with filter_interacted=True (default)
    batch_results_filtered = model_with_data.recommend_batch(
        user_ids, 
        top_k=top_k, 
        filter_interacted=True
    )
    
    # Test with filter_interacted=False
    batch_results_unfiltered = model_with_data.recommend_batch(
        user_ids, 
        top_k=top_k, 
        filter_interacted=False
    )
    
    # Verify we get results for all users in both cases
    assert len(batch_results_filtered) == len(user_ids)
    assert len(batch_results_unfiltered) == len(user_ids)
    
    # With filter_interacted=False, we might get more or different recommendations
    # The exact behavior depends on the model, but both should return valid results
    for user_result in batch_results_filtered + batch_results_unfiltered:
        assert all(isinstance(item_id, int) for item_id in user_result)


def test_hybrid_batch_unknown_users(model_with_data):
    """Test batch recommendations with unknown users."""
    # Mix of known and unknown users
    user_ids = [0, 999, 1]  # 999 is unknown
    top_k = 2
    
    batch_results = model_with_data.recommend_batch(user_ids, top_k=top_k)
    
    # Should still get results for all users (unknown users handled by cold start)
    assert len(batch_results) == len(user_ids)


def test_hybrid_model_initialization():
    """Test HybridSlimFM model initialization with custom parameters."""
    model = HybridSlimFM(
        epochs=20,
        n_threads=2,
        use_bias=False,
        similarity_weight_factor=3.0
    )
    
    assert model.epochs == 20
    assert model.n_threads == 2
    assert model.use_bias is False
    assert model.similarity_weight_factor == 3.0