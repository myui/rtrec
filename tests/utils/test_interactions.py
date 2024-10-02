import pytest
from collections import Counter
from rtrec.utils.interactions import UserItemInteractions

@pytest.fixture
def interactions():
    return UserItemInteractions()

def test_add_interaction(interactions):
    interactions.add_interaction(1, 10, 5.0)
    assert interactions.get_user_item_rating(1, 10) == 5.0

def test_add_multiple_interactions(interactions):
    interactions.add_interaction(1, 10, 5.0)
    interactions.add_interaction(1, 10, 3.0)
    assert interactions.get_user_item_rating(1, 10) == 8.0

def test_add_sub_interaction(interactions):
    interactions.add_interaction(1, 10, 5.0)
    interactions.add_interaction(1, 10, -5.0)
    assert interactions.get_user_item_rating(1, 10) == 0.0
    assert 10 in interactions.get_user_items(1)

def test_non_interacted_items(interactions):
    interactions.add_interaction(1, 10, 5.0)
    interactions.add_interaction(1, 20, 3.0)
    interactions.add_interaction(2, 15, 3.0)
    interactions.add_interaction(3, 9, 3.0)
    assert set(interactions.get_all_non_interacted_items(1)) == {15, 9}

def test_non_negative_items(interactions):
    interactions.add_interaction(1, 10, 5.0)
    interactions.add_interaction(1, 20, -2.0)
    assert set(interactions.get_all_non_negative_items(1)) == {10}

def test_get_all_users(interactions):
    interactions.add_interaction(1, 10, 5.0)
    interactions.add_interaction(2, 20, 3.0)
    assert set(interactions.get_all_users()) == {1, 2}

def test_get_all_items_for_user(interactions):
    interactions.add_interaction(1, 10, 5.0)
    interactions.add_interaction(1, 20, 3.0)
    assert set(interactions.get_all_items_for_user(1)) == {10, 20}

def test_empty_user(interactions):
    assert interactions.get_user_items(99) == interactions.empty
    assert interactions.get_user_item_rating(99, 10) == 0.0

# Add more tests as needed

if __name__ == "__main__":
    pytest.main()
