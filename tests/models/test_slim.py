

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
    assert model.feature_store.build_user_features_matrix(user_id=user_index).nnz == 2

def test_register_item_feature(model):
    item_id = 'item_1'
    item_tags = ['tagA', 'tagB']
    item_index = model.register_item_feature(item_id, item_tags)
    # Verify that the item feature is registered correctly
    assert item_index == model.item_ids.identify(item_id)
    assert model.feature_store.build_item_features_matrix(item_id=item_index).nnz == 2

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
    interactions = [('user_1', 'item_1', 1622470427.0, 5.0)]
    model.fit(interactions)
    similar_items = model.similar_items('item_1', top_k=5)
    # Since there are no items, similar items should be empty
    assert similar_items == []

if __name__ == "__main__":
    pytest.main()