

import pytest

from rtrec.models.slim import SLIM


@pytest.fixture
def model():
    model = SLIM()
    return model

def test_register_user_feature(model):
    user_id = 'user_1'
    user_tags = ['tag1', 'tag2']
    user_index = model.register_user_feature(user_id, user_tags)
    # Verify that the user feature is registered correctly
    assert user_index == model.user_ids.identify(user_id)
    assert model.feature_store.build_user_features_matrix(user_ids=[user_index]).nnz == 2

def test_register_item_feature(model):
    item_id = 'item_1'
    item_tags = ['tagA', 'tagB']
    item_index = model.register_item_feature(item_id, item_tags)
    # Verify that the item feature is registered correctly
    assert item_index == model.item_ids.identify(item_id)
    assert model.feature_store.build_item_features_matrix(item_ids=[item_index]).nnz == 2

def test_add_interactions(model):
    interactions = [('user_1', 'item_1', 1622470427.0, 5.0)]
    model.add_interactions(interactions)
    # Verify that the interaction is added correctly
    user_index = model.user_ids.identify('user_1')
    item_index = model.item_ids.identify('item_1')
    assert model.interactions.get_user_item_rating(user_index, item_index) == 5.0

def test_recommend_no_interactions(model):
    recommendations = model.recommend('user_1', top_k=5)
    # Since there are no interactions, recommendations should be empty
    assert recommendations == []

def test_similar_items_no_data(model):
    import time
    current_unixtime = time.time()
    interactions = [('user_1', 'item_1', 1622470427.0, 5.0), ('user_2', 'item_2', current_unixtime, -2.0)]
    model.fit(interactions)

    def yield_interactions():
        for interaction in interactions:
            yield interaction

    model.fit(yield_interactions())

    similar_items = model.similar_items('item_1', top_k=5)
    # Since there are no items, similar items should be empty
    assert similar_items == []

def test_similar_items(model):
    import time
    current_unixtime = time.time()
    interactions = [('user_1', 'item_1', current_unixtime, 5.0),
                   ('user_1', 'item_3', current_unixtime, 4.0),
                   ('user_1', 'item_4', current_unixtime, 3.0),
                   ('user_2', 'item_1', current_unixtime, 3.0),
                   ('user_2', 'item_2', current_unixtime, -2.0),
                   ('user_2', 'item_4', current_unixtime, 3.0),
                   ('user_3', 'item_1', current_unixtime, 4.0),
                   ('user_3', 'item_3', current_unixtime, 2.0),
                   ('user_3', 'item_4', current_unixtime, 4.0)]
    model.fit(interactions)

    similar_items = model.similar_items('item_1', top_k=5)
    # Verify that the similar items are correct
    assert similar_items == ["item_4", "item_3"]

    results = model.similar_items('item_1', top_k=5, ret_scores=True)
    similar_items, scores = map(list, zip(*results)) # Unzip the results
    assert similar_items == ["item_4", "item_3"]
    assert scores[0] > scores[1]

def test_fit_and_recommend(model):
    import time
    current_unixtime = time.time()
    interactions = [('user_1', 'item_1', current_unixtime, 5.0),
                   ('user_2', 'item_2', current_unixtime, -2.0),
                   ('user_2', 'item_1', current_unixtime, 3.0),
                   ('user_2', 'item_4', current_unixtime, 3.0),
                   ('user_1', 'item_3', current_unixtime, 4.0)]
    model.fit(interactions)

    def yield_interactions():
        for interaction in interactions:
            yield interaction
    model.fit(yield_interactions())

    recommendations = model.recommend('user_1', top_k=5)
    # Verify that the recommendations are correct
    assert recommendations == ["item_4", "item_2"]

def test_get_users_by_items(model):
    import time
    current_unixtime = time.time()
    interactions = [('user_1', 'item_1', current_unixtime, 5.0),
                   ('user_2', 'item_1', current_unixtime, 3.0),
                   ('user_2', 'item_2', current_unixtime, 4.0),
                   ('user_3', 'item_2', current_unixtime, 2.0)]
    model.fit(interactions)
    
    # Test single item
    users = model.get_users_by_items(['item_1'])
    assert set(users) == {'user_1', 'user_2'}
    
    # Test multiple items
    users = model.get_users_by_items(['item_1', 'item_2'])
    assert set(users) == {'user_1', 'user_2', 'user_3'}
    
    # Test non-existent item
    users = model.get_users_by_items(['item_99'])
    assert users == []

def test_recommend_batch(model):
    import time
    current_unixtime = time.time()
    interactions = [('user_1', 'item_1', current_unixtime, 5.0),
                   ('user_1', 'item_3', current_unixtime, 4.0),
                   ('user_2', 'item_2', current_unixtime, 3.0),
                   ('user_2', 'item_4', current_unixtime, 4.0),
                   ('user_3', 'item_1', current_unixtime, 2.0),
                   ('user_3', 'item_2', current_unixtime, 3.0)]
    model.fit(interactions)
    
    # Test batch recommendations for multiple users
    users = ['user_1', 'user_2', 'user_3']
    batch_recommendations = model.recommend_batch(users, top_k=2)
    
    # Verify that we get recommendations for all users
    assert len(batch_recommendations) == 3
    assert all(isinstance(recs, list) for recs in batch_recommendations)
    
    # Test with candidate items
    candidate_items = ['item_1', 'item_2', 'item_3']
    batch_recommendations_filtered = model.recommend_batch(
        users, 
        candidate_items=candidate_items, 
        top_k=2
    )
    
    # Verify filtered recommendations
    assert len(batch_recommendations_filtered) == 3
    for user_recs in batch_recommendations_filtered:
        for item in user_recs:
            assert item in candidate_items
    
    # Test with empty user list
    empty_batch = model.recommend_batch([], top_k=2)
    assert empty_batch == []
    
    # Test with single user (should still return list of lists)
    single_user_batch = model.recommend_batch(['user_1'], top_k=2)
    assert len(single_user_batch) == 1
    assert isinstance(single_user_batch[0], list)
    
    # Test with filter_interacted=False
    batch_with_interacted = model.recommend_batch(
        ['user_1'], 
        top_k=3, 
        filter_interacted=False
    )
    assert len(batch_with_interacted) == 1
    # Should include items that user_1 has interacted with
    user_1_interacted_items = ['item_1', 'item_3']
    recommendations = batch_with_interacted[0]
    # At least one of the interacted items should be in recommendations
    assert any(item in user_1_interacted_items for item in recommendations)

def test_recommend_batch_cold_start(model):
    import time
    current_unixtime = time.time()
    interactions = [('user_1', 'item_1', current_unixtime, 5.0),
                   ('user_1', 'item_2', current_unixtime, 4.0)]
    model.fit(interactions)
    
    # Test with mix of existing and new users
    users = ['user_1', 'new_user']
    batch_recommendations = model.recommend_batch(users, top_k=2)
    
    # Should get recommendations for both users
    assert len(batch_recommendations) == 2
    
    # First user should get personalized recommendations
    assert len(batch_recommendations[0]) <= 2
    
    # New user should get popular items (cold start handling)
    assert len(batch_recommendations[1]) <= 2

if __name__ == "__main__":
    pytest.main()
