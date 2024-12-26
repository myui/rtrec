import pytest

from rtrec.models.lightfm import LightFM

@pytest.fixture
def model():
    model = LightFM()
    return model

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
    assert similar_items == ['item_2'] # Unlike SLIM, LightFM is intented to return item_2 while it is not interacted with item_1

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

if __name__ == "__main__":
    pytest.main()
