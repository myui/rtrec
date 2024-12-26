import pytest
from collections import Counter
import time
from time import sleep
from rtrec.utils.interactions import UserItemInteractions
from scipy.sparse import csc_matrix, coo_matrix

@pytest.fixture
def interactions():
    # Create an instance of UserItemInteractions without decay
    return UserItemInteractions(min_value=-5, max_value=10)

@pytest.fixture
def interactions_with_decay():
    # Create an instance of UserItemInteractions with a decay rate applied over 7 days
    return UserItemInteractions(min_value=-5, max_value=10, decay_in_days=7)

def test_add_interaction(interactions):
    tstamp = time.time()  # Use float timestamp
    interactions.add_interaction(1, 10, tstamp, 5.0)
    assert interactions.get_user_item_rating(1, 10) == 5.0

def test_add_multiple_interactions(interactions):
    tstamp = time.time()  # Use float timestamp
    interactions.add_interaction(1, 10, tstamp, 5.0)
    interactions.add_interaction(1, 10, tstamp, 3.0)
    assert interactions.get_user_item_rating(1, 10) == 8.0

def test_add_sub_interaction(interactions):
    tstamp = time.time()  # Use float timestamp
    interactions.add_interaction(1, 10, tstamp, 5.0)
    interactions.add_interaction(1, 10, tstamp, -5.0)
    assert interactions.get_user_item_rating(1, 10) == 0.0
    assert 10 in interactions.get_user_items(1)

def test_non_interacted_items(interactions):
    tstamp = time.time()  # Use float timestamp
    interactions.add_interaction(1, 10, tstamp, 5.0)
    interactions.add_interaction(1, 20, tstamp, 3.0)
    interactions.add_interaction(2, 15, tstamp, 3.0)
    interactions.add_interaction(3, 9, tstamp, 3.0)
    assert set(interactions.get_all_non_interacted_items(1)) == {15, 9}

def test_non_negative_items(interactions):
    tstamp = time.time()  # Use float timestamp
    interactions.add_interaction(1, 10, tstamp, 5.0)
    interactions.add_interaction(1, 20, tstamp, -2.0)
    assert set(interactions.get_all_non_negative_items(1)) == {10}

def test_get_all_users(interactions):
    tstamp = time.time()  # Use float timestamp
    interactions.add_interaction(1, 10, tstamp, 5.0)
    interactions.add_interaction(2, 20, tstamp, 3.0)
    assert set(interactions.get_all_users()) == {1, 2}

def test_get_user_items(interactions):
    tstamp = time.time()  # Use float timestamp
    interactions.add_interaction(1, 10, tstamp, 5.0)
    interactions.add_interaction(1, 20, tstamp, 3.0)
    assert set(interactions.get_user_items(1)) == {10, 20}

def test_empty_user(interactions):
    assert interactions.get_user_items(99) == []
    assert interactions.get_user_item_rating(99, 10) == 0.0

def test_no_decay_rate(interactions):
    tstamp = time.time()  # Use float timestamp
    interactions.add_interaction(1, 10, tstamp, 5.0)
    time.sleep(1)  # Wait for 1 second
    assert interactions.get_user_item_rating(1, 10) == 5.0  # No decay should be applied

def test_decay_functionality(interactions_with_decay):
    tstamp = time.time()  # Use float timestamp
    interactions_with_decay.add_interaction(1, 10, tstamp, 5.0)
    time.sleep(1)  # Wait for 1 second to allow decay
    decayed_rating = interactions_with_decay.get_user_item_rating(1, 10)

    # Calculate expected decayed value
    elapsed_days = 1 / 86400.0  # 1 second in days
    decay_factor = interactions_with_decay.decay_rate ** elapsed_days
    assert abs(1.0 - decay_factor) < 0.001  # Allow some tolerance for floating-point comparison
    expected_rating = 5.0 * decay_factor

    assert abs(decayed_rating - 5.0) < 0.001  # Allow some tolerance for floating-point comparison
    assert abs(decayed_rating - expected_rating) < 0.001  # Allow some tolerance for floating-point comparison

def test_decayed_rating_after_7days(interactions_with_decay):
    # Current timestamp
    current_tstamp = time.time()

    # Timestamp 7 days ago
    tstamp_7_days_ago = current_tstamp - (7 * 86400)  # 7 days in seconds

    # Add interaction with a timestamp from 7 days ago
    interactions_with_decay.add_interaction(1, 10, tstamp_7_days_ago, 5.0)

    # workaround to advance the last update time by another user as the decay is based on the last update time
    interactions_with_decay.add_interaction(2, 10, time.time(), 5.0)

    # Get the rating after 1 second to check the decay
    sleep(1)
    decayed_rating = interactions_with_decay.get_user_item_rating(1, 10)

    # Calculate expected decayed value after 7 days
    elapsed_days = 7 + 1 / 86400.0  # 7 days + 1 second in days
    decay_factor = interactions_with_decay.decay_rate ** elapsed_days
    expected_rating = 5.0 * decay_factor

    # print("decay_factor: {}, expected: {}, actual: {}".format(decay_factor, expected_rating, decayed_rating))
    assert abs(0.5 - decay_factor) < 0.02  # Allow some tolerance for floating-point comparison
    assert abs(5.0 / 2.0 - decayed_rating) < 0.1  # Allow some tolerance for floating-point comparison

    # Check if the decayed rating is as expected, allowing some tolerance for floating-point comparison
    assert abs(decayed_rating - expected_rating) < 0.001

def test_to_csc():
    interactions = UserItemInteractions(min_value=-5, max_value=10, decay_in_days=None)
    interactions.add_interaction(0, 0, tstamp=12345, delta=5)
    interactions.add_interaction(0, 1, tstamp=12346, delta=3)
    interactions.add_interaction(1, 1, tstamp=12347, delta=4)
    interactions.add_interaction(1, 2, tstamp=12348, delta=2)
    interactions.add_interaction(2, 0, tstamp=12349, delta=1)
    interactions.add_interaction(2, 2, tstamp=12350, delta=3)

    # Full matrix
    csc = interactions.to_csc()
    expected = csc_matrix((
        [5, 3, 4, 2, 1, 3],
        ([0, 0, 1, 1, 2, 2], [0, 1, 1, 2, 0, 2])
    ), shape=(3, 3))
    assert (csc != expected).nnz == 0

    # Filter by items
    csc_filtered = interactions.to_csc(select_items=[1, 2])
    expected_filtered = csc_matrix((
        [3, 4, 2, 3],
        ([0, 1, 1, 2], [1, 1, 2, 2])
    ), shape=(3, 3))
    assert (csc_filtered != expected_filtered).nnz == 0

def test_to_coo():
    interactions = UserItemInteractions(min_value=-5, max_value=10, decay_in_days=None)
    interactions.add_interaction(0, 0, tstamp=12345, delta=5)
    interactions.add_interaction(0, 1, tstamp=12346, delta=3)
    interactions.add_interaction(1, 1, tstamp=12347, delta=4)
    interactions.add_interaction(1, 2, tstamp=12348, delta=2)
    interactions.add_interaction(2, 0, tstamp=12349, delta=1)
    interactions.add_interaction(2, 2, tstamp=12350, delta=3)

    # Full matrix
    coo = interactions.to_coo()
    expected = coo_matrix((
        [5, 3, 4, 2, 1, 3],
        ([0, 0, 1, 1, 2, 2], [0, 1, 1, 2, 0, 2])
    ), shape=(3, 3))
    assert (coo != expected).nnz == 0

    # Filter by users
    coo_filtered_users = interactions.to_coo(select_users=[1, 2])
    expected_filtered_users = coo_matrix((
        [4, 2, 1, 3],
        ([1, 1, 2, 2], [1, 2, 0, 2])
    ), shape=(3, 3))
    assert (coo_filtered_users != expected_filtered_users).nnz == 0

    # Filter by users and items
    coo_filtered_users_items = interactions.to_coo(select_users=[1, 2], select_items=[1, 2])
    expected_filtered_users_items = coo_matrix((
        [4, 2, 3],
        ([1, 1, 2], [1, 2, 2])
    ), shape=(3, 3))
    assert (coo_filtered_users_items != expected_filtered_users_items).nnz == 0

if __name__ == "__main__":
    pytest.main()
