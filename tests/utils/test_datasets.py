import pytest
from collections import Counter
from rtrec.utils.datasets import UserItemInteractions

def test_add_interaction():
    ui = UserItemInteractions()
    
    # Add interactions
    ui.add_interaction(1, 101, 3)
    ui.add_interaction(1, 102, 1)
    ui.add_interaction(2, 101, 2)

    # Check interactions
    assert ui.get_user_item_count(1, 101) == 3
    assert ui.get_user_item_count(1, 102) == 1
    assert ui.get_user_item_count(2, 101) == 2
    assert ui.get_user_item_count(2, 102) == 0  # Item not interacted with

def test_remove_interaction():
    ui = UserItemInteractions()

    # Add and reduce interaction counts
    ui.add_interaction(1, 101, 3)
    ui.add_interaction(1, 101, -2)
    assert ui.get_user_item_count(1, 101) == 1

    # Remove interaction completely
    ui.add_interaction(1, 101, -1)
    assert ui.get_user_item_count(1, 101) == 0

def test_remove_user():
    ui = UserItemInteractions()

    # Add interaction and remove it
    ui.add_interaction(1, 101, 3)
    ui.add_interaction(1, 101, -3)  # This should remove the interaction and the user
    assert ui.get_user_item_count(1, 101) == 0
    assert 1 not in ui.get_all_users()

def test_get_all_users():
    ui = UserItemInteractions()

    # Add interactions
    ui.add_interaction(1, 101, 3)
    ui.add_interaction(2, 102, 1)

    assert set(ui.get_all_users()) == {1, 2}

def test_get_all_items_for_user():
    ui = UserItemInteractions()

    # Add interactions
    ui.add_interaction(1, 101, 3)
    ui.add_interaction(1, 102, 1)
    
    assert set(ui.get_all_items_for_user(1)) == {101, 102}
    assert ui.get_all_items_for_user(2) == []

def test_edge_case_negative_count():
    ui = UserItemInteractions()

    # Add interaction with a negative count that should be ignored
    ui.add_interaction(1, 101, -1)
    
    assert ui.get_user_item_count(1, 101) == 0  # No interaction should be recorded
    assert 1 not in ui.get_all_users()

if __name__ == "__main__":
    pytest.main()