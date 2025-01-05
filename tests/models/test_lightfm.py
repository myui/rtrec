import pytest

from rtrec.models.lightfm import LightFM
import time
import random

@pytest.fixture
def model():
    model = LightFM(random_state=42, epochs=10)
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
    assert similar_items == ["item_4", "item_3", "item_2"]

    results = model.similar_items('item_1', top_k=5, ret_scores=True)
    similar_items, scores = map(list, zip(*results)) # Unzip the results
    assert similar_items == ["item_4", "item_3", "item_2"]
    assert scores[2] < 0  # item_2 should have a negative score

def test_fit_and_recommend_batch(model):
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

    recommendations = model.recommend_batch(['user_1', 'user_2'], top_k=5)
    # Verify that the recommendations are correct
    assert recommendations == [["item_4", "item_2"], ["item_3"]]

    # more user 3 interactions
    interactions = [('user_3', 'item_1', current_unixtime, 5.0),
                   ('user_3', 'item_1', current_unixtime, 3.0),
                   ('user_3', 'item_4', current_unixtime, 3.0),
                   ('user_3', 'item_3', current_unixtime, 4.0)]
    model.fit(interactions)

    recommendations = model.recommend_batch(['user_2', 'user_3'], top_k=5)
    # Verify that the recommendations are correct
    assert recommendations == [["item_3"], ["item_2"]]

def test_register_user_feature_and_recommend(model):
    rng = random.Random(42)  # Create a non-global RNG with a fixed seed
    current_unixtime = time.time()
    jitter_range = 0.01  # Maximum jitter of 10 milliseconds (positive)
    interactions = [
        ('user_1', 'item_1', current_unixtime + rng.uniform(0, jitter_range), 5.0),
        ('user_2', 'item_2', current_unixtime + 1 + rng.uniform(0, jitter_range), 3.0),
        ('user_1', 'item_3', current_unixtime + 2 + rng.uniform(0, jitter_range), 4.0),
        ('user_3', 'item_4', current_unixtime + 3 + rng.uniform(0, jitter_range), 2.0),
        ('user_3', 'item_5', current_unixtime + 4 + rng.uniform(0, jitter_range), 1.0),
        ('user_4', 'item_6', current_unixtime + 5 + rng.uniform(0, jitter_range), 4.5),
        ('user_5', 'item_7', current_unixtime + 6 + rng.uniform(0, jitter_range), 3.5),
        ('user_1', 'item_8', current_unixtime + 7 + rng.uniform(0, jitter_range), 5.0),
        ('user_2', 'item_9', current_unixtime + 8 + rng.uniform(0, jitter_range), 2.0),
        ('user_5', 'item_10', current_unixtime + 9 + rng.uniform(0, jitter_range), 4.0),
    ]

    # Register user features
    user_features = {
        'user_1': {'age': 25, 'location': 'US'},
        'user_2': {'age': 30, 'location': 'UK'},
        'user_3': {'age': 35, 'location': 'CA'},
        'user_4': {'age': 40, 'location': 'AU'},
        'user_5': {'age': 45, 'location': 'IN'},
    }
    for user, features in user_features.items():
        model.register_user_feature(user, features)

    model.fit(interactions)

    recommendations = model.recommend('user_1', top_k=5)
    assert recommendations  # Ensure recommendations are not empty
    assert all(item not in ['item_1', 'item_3', 'item_8'] for item in recommendations)  # Avoid already interacted items

def test_register_item_feature_and_similar_items(model):
    rng = random.Random(42)  # Create a non-global RNG with a fixed seed
    current_unixtime = time.time()
    jitter_range = 0.01  # Maximum jitter of 10 milliseconds (positive)
    interactions = [
        ('user_1', 'item_1', current_unixtime + rng.uniform(0, jitter_range), 5.0),
        ('user_2', 'item_2', current_unixtime + 1 + rng.uniform(0, jitter_range), 3.0),
        ('user_1', 'item_3', current_unixtime + 2 + rng.uniform(0, jitter_range), 4.0),
        ('user_3', 'item_4', current_unixtime + 3 + rng.uniform(0, jitter_range), 2.0),
        ('user_3', 'item_5', current_unixtime + 4 + rng.uniform(0, jitter_range), 1.0),
        ('user_4', 'item_6', current_unixtime + 5 + rng.uniform(0, jitter_range), 4.5),
        ('user_5', 'item_7', current_unixtime + 6 + rng.uniform(0, jitter_range), 3.5),
        ('user_2', 'item_8', current_unixtime + 7 + rng.uniform(0, jitter_range), 5.0),
        ('user_4', 'item_9', current_unixtime + 8 + rng.uniform(0, jitter_range), 2.5),
        ('user_5', 'item_10', current_unixtime + 9 + rng.uniform(0, jitter_range), 4.0),
    ]

    # Register item features
    item_features = {
        'item_1': {'category': 'electronics', 'price': 100},
        'item_2': {'category': 'books', 'price': 20},
        'item_3': {'category': 'electronics', 'price': 200},
        'item_4': {'category': 'clothing', 'price': 50},
        'item_5': {'category': 'clothing', 'price': 30},
        'item_6': {'category': 'furniture', 'price': 500},
        'item_7': {'category': 'furniture', 'price': 300},
        'item_8': {'category': 'electronics', 'price': 150},
        'item_9': {'category': 'books', 'price': 25},
        'item_10': {'category': 'clothing', 'price': 40},
    }
    for item, features in item_features.items():
        model.register_item_feature(item, features)

    model.fit(interactions)

    similar_items = model.similar_items('item_1', top_k=5)
    assert similar_items  # Ensure similar items are not empty
    assert 'item_1' not in similar_items  # Similar items should not include the item itself
    assert all(isinstance(item, str) for item in similar_items)  # Ensure item IDs are returned

def test_register_user_and_item_features_and_recommend(model):
    rng = random.Random(42)  # Create a non-global RNG with a fixed seed
    current_unixtime = time.time()
    jitter_range = 0.01  # Maximum jitter of 10 milliseconds (positive)

    # Define interactions with jitter, where users have 0-4 interactions
    interactions = [
        ('user_1', 'item_1', current_unixtime + rng.uniform(0, jitter_range), 5.0),
        ('user_1', 'item_3', current_unixtime + 1 + rng.uniform(0, jitter_range), 4.0),
        ('user_1', 'item_8', current_unixtime + 2 + rng.uniform(0, jitter_range), 5.0),
        ('user_2', 'item_2', current_unixtime + 3 + rng.uniform(0, jitter_range), 3.0),
        ('user_2', 'item_9', current_unixtime + 4 + rng.uniform(0, jitter_range), 2.0),
        ('user_2', 'item_8', current_unixtime + 5 + rng.uniform(0, jitter_range), 4.0),
        ('user_3', 'item_4', current_unixtime + 6 + rng.uniform(0, jitter_range), 2.0),
        ('user_3', 'item_5', current_unixtime + 7 + rng.uniform(0, jitter_range), 1.0),
        ('user_4', 'item_6', current_unixtime + 8 + rng.uniform(0, jitter_range), 4.5),
        ('user_5', 'item_7', current_unixtime + 9 + rng.uniform(0, jitter_range), 3.5),
        ('user_5', 'item_10', current_unixtime + 10 + rng.uniform(0, jitter_range), 4.0),
    ]

    # Register user features with additional dimensions like gender
    user_features = {
        'user_1': {'age': 25, 'gender': 'male', 'location': 'US'},
        'user_2': {'age': 30, 'gender': 'female', 'location': 'UK'},
        'user_3': {'age': 35, 'gender': 'male', 'location': 'CA'},
        'user_4': {'age': 40, 'gender': 'female', 'location': 'AU'},
        'user_5': {'age': 45, 'gender': 'male', 'location': 'IN'},
    }
    for user, features in user_features.items():
        model.register_user_feature(user, features)

    # Register item features
    item_features = {
        'item_1': {'category': 'electronics', 'price': 100},
        'item_2': {'category': 'books', 'price': 20},
        'item_3': {'category': 'electronics', 'price': 200},
        'item_4': {'category': 'clothing', 'price': 50},
        'item_5': {'category': 'clothing', 'price': 30},
        'item_6': {'category': 'furniture', 'price': 500},
        'item_7': {'category': 'furniture', 'price': 300},
        'item_8': {'category': 'electronics', 'price': 150},
        'item_9': {'category': 'books', 'price': 25},
        'item_10': {'category': 'clothing', 'price': 40},
    }
    for item, features in item_features.items():
        model.register_item_feature(item, features)

    # Fit model with interactions
    model.fit(interactions)

    # Test recommendations based on user and item features
    recommendations = model.recommend('user_1', top_k=5)
    assert recommendations  # Ensure recommendations are not empty
    assert all(item not in ['item_1', 'item_3', 'item_8'] for item in recommendations)  # Avoid already interacted items

    # Test similar items for an item with item features taken into account
    similar_items = model.similar_items('item_1', top_k=5)
    assert similar_items  # Ensure similar items are not empty
    assert 'item_1' not in similar_items  # Similar items should not include the item itself
    assert all(isinstance(item, str) for item in similar_items)  # Ensure item IDs are returned
    assert all(item in item_features for item in similar_items)  # All similar items should have corresponding item features

def test_recommend_and_similar_items(model):
    # Create a non-global RNG with a fixed seed
    rng = random.Random(42)  # Create a non-global RNG with a fixed seed
    current_unixtime = 1609459200  # Arbitrary timestamp for current time
    jitter_range = 5.0  # Jitter range for random time offset

    # Define 20 interactions with randomized jitter and ratings
    interactions = [
        ('user_1', 'item_1', current_unixtime + rng.uniform(0, jitter_range), 5.0),
        ('user_2', 'item_2', current_unixtime + 1 + rng.uniform(0, jitter_range), 3.0),
        ('user_1', 'item_3', current_unixtime + 2 + rng.uniform(0, jitter_range), 4.0), 
        ('user_5', 'item_10', current_unixtime + 9 + rng.uniform(0, jitter_range), 4.0),
        ('user_5', 'item_1', current_unixtime + 9 + rng.uniform(0, jitter_range), 4.0),
        ('user_3', 'item_5', current_unixtime + 3 + rng.uniform(0, jitter_range), 5.0),
        ('user_4', 'item_4', current_unixtime + 4 + rng.uniform(0, jitter_range), 2.0),
        ('user_6', 'item_6', current_unixtime + 5 + rng.uniform(0, jitter_range), 3.5),
        ('user_7', 'item_7', current_unixtime + 6 + rng.uniform(0, jitter_range), 4.5),
        ('user_8', 'item_8', current_unixtime + 7 + rng.uniform(0, jitter_range), 3.0),
        ('user_9', 'item_9', current_unixtime + 8 + rng.uniform(0, jitter_range), 2.5),
        ('user_1', 'item_4', current_unixtime + 10 + rng.uniform(0, jitter_range), 4.0),
        ('user_2', 'item_8', current_unixtime + 11 + rng.uniform(0, jitter_range), 3.0),
        ('user_3', 'item_2', current_unixtime + 12 + rng.uniform(0, jitter_range), 5.0),
        ('user_4', 'item_1', current_unixtime + 13 + rng.uniform(0, jitter_range), 3.5),
        ('user_5', 'item_3', current_unixtime + 14 + rng.uniform(0, jitter_range), 4.5),
        ('user_6', 'item_2', current_unixtime + 15 + rng.uniform(0, jitter_range), 3.0),
        ('user_7', 'item_10', current_unixtime + 16 + rng.uniform(0, jitter_range), 4.0),
        ('user_8', 'item_9', current_unixtime + 17 + rng.uniform(0, jitter_range), 2.0),
        ('user_9', 'item_7', current_unixtime + 18 + rng.uniform(0, jitter_range), 5.0),
        ('user_10', 'item_6', current_unixtime + 19 + rng.uniform(0, jitter_range), 3.0)
    ]

    # Register user features
    user_features = {
        'user_1': {'age': "20's", 'location': 'US', 'gender': 'M'},
        'user_2': {'age': "30's", 'location': 'UK', 'gender': 'F'},
        'user_3': {'age': "30's", 'location': 'JP', 'gender': 'M'},
        'user_4': {'age': "40's", 'location': 'FR', 'gender': 'M'},
        'user_5': {'age': "40's", 'location': 'CA', 'gender': 'F'},
        'user_6': {'age': "20's", 'location': 'US', 'gender': 'F'},
        'user_7': {'age': "30's", 'location': 'UK', 'gender': 'M'},
        'user_8': {'age': "20's", 'location': 'JP', 'gender': 'F'},
        'user_9': {'age': "30's", 'location': 'FR', 'gender': 'M'},
        'user_10': {'age': "20's", 'location': 'CA', 'gender': 'F'},
    }
    # Register item features
    item_features = {
        'item_1': {'category': 'electronics', 'price': "$100-$200"},
        'item_2': {'category': 'books', 'price': "$0-$50"},
        'item_3': {'category': 'electronics', 'price': "$200-$300"},
        'item_4': {'category': 'clothing', 'price': "$50-$100"},
        'item_5': {'category': 'clothing', 'price': "$0-$50"},
        'item_6': {'category': 'furniture', 'price': "$500+"},
        'item_7': {'category': 'furniture', 'price': "$200-$500"},
        'item_8': {'category': 'electronics', 'price': "$100-$200"},
        'item_9': {'category': 'books', 'price': "$0-$50"},
        'item_10': {'category': 'clothing', 'price': "$0-$50"},
    }

    # Assuming the model object has methods to register user/item features and fit the data
    for user, features in user_features.items():
        model.register_user_feature(user, features)

    for item, features in item_features.items():
        model.register_item_feature(item, features)

    # Fit the model with interactions
    model.fit(interactions)

    # Test: Recommend top-k items for 'user_1'
    recommendations = model.recommend('user_1', top_k=5)
    assert recommendations  # Ensure recommendations are not empty
    assert all(item not in ['item_1', 'item_3', 'item_4'] for item in recommendations)  # Avoid already interacted items

    # Test: Recommend top-k items for 'user_2'
    recommendations_user_2 = model.recommend('user_2', top_k=5)
    assert recommendations_user_2  # Ensure recommendations for user_2 are not empty
    assert all(item not in ['item_2', 'item_8'] for item in recommendations_user_2)  # Avoid already interacted items

    # Test: Get similar items for 'item_1'
    similar_items = model.similar_items('item_1', top_k=5)
    assert similar_items  # Ensure similar items are not empty
    assert 'item_1' not in similar_items  # Similar items should not include the item itself

    # Validate that the similar items are consistent with our expected similarity table
    assert 'item_3' in similar_items[:2]  # item_3 is co-occurring with item_1
    assert 'item_2' in similar_items[:2]  # item_2 is co-occurring with item_1
    assert 'item_4' in similar_items  # item_4 might be co-occurring based on interactions

    # Test: Validate that co-occurrence affects similarity
    similar_items = model.similar_items('item_3', top_k=5)
    assert similar_items
    assert 'item_3' not in similar_items  # Similar items should not include the item itself
    assert similar_items[0] == 'item_1'  # item_1 should be one of the top similar items
    assert 'item_10' in similar_items  # item_10 should appear due to co-occurrence with item_3
    assert 'item_2' in similar_items  # item_2 should also be in the top due to shared interactions with item_3

if __name__ == "__main__":
    pytest.main()
