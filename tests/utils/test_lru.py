from typing import List
import pytest
from rtrec.utils.lru import LRUFreqSet

@pytest.fixture
def lru_set():
    """Fixture to create a new LRUFreqSet instance for each test."""
    return LRUFreqSet(capacity=3)

@pytest.fixture
def lru_set_with_entries():
    lru_set = LRUFreqSet(capacity=5)
    # Add keys with varying access frequencies
    lru_set.add("A")  # Frequency 1
    lru_set.add("B")  # Frequency 1
    lru_set.add("C")  # Frequency 1
    lru_set.add("A")  # Frequency 2
    lru_set.add("D")  # Frequency 1
    lru_set.add("E")  # Frequency 1
    lru_set.add("A")  # Frequency 3
    lru_set.add("B")  # Frequency 2
    return lru_set

def test_add_and_contains(lru_set: LRUFreqSet):
    """Test adding items and checking their existence."""
    lru_set.add("a")
    lru_set.add("b")
    lru_set.add("c")

    assert "a" in lru_set
    assert "b" in lru_set
    assert "c" in lru_set

def test_eviction_on_capacity(lru_set: LRUFreqSet):
    """Test eviction behavior when the set exceeds capacity."""
    lru_set.add("a")
    lru_set.add("b")
    lru_set.add("c")
    lru_set.add("d")

    # After adding the 4th item, the least recently used ("a") should be evicted.
    assert "a" not in lru_set
    assert "b" in lru_set
    assert "c" in lru_set
    assert "d" in lru_set

def test_update_frequency(lru_set: LRUFreqSet):
    """Test that frequency is updated when an item is accessed."""
    lru_set.add("a")
    lru_set.add("b")
    lru_set.add("c")

    # Update the frequency of "a" by accessing it multiple times
    lru_set.add("a")
    lru_set.add("a")

    # "b" should be evicted because "a" has been accessed more frequently
    lru_set.add("d")

    assert "b" not in lru_set
    assert "a" in lru_set
    assert "c" in lru_set

def test_get_freq_items(lru_set: LRUFreqSet):
    """Test retrieval of the most frequently used items."""
    lru_set.add("a")
    lru_set.add("b")
    lru_set.add("c")

    # Access some items to update their frequencies
    lru_set.add("a")
    lru_set.add("a")

    # "b" should be evicted because "a" has been accessed more frequently
    lru_set.add("d")

    # Access "c" to update its frequency
    lru_set.add("c")

    # Get top 2 most frequently used items
    freq_items = list(lru_set.get_freq_items(2))
    assert freq_items == ["a", "c"]

    # Get all items sorted by frequency
    all_items = list(lru_set.get_freq_items())
    assert all_items == ["a", "c", "d"]

def test_remove_item(lru_set: LRUFreqSet):
    """Test removing items from the set."""
    lru_set.add("a")
    lru_set.add("b")

    lru_set.remove("a")

    assert "a" not in lru_set
    assert "b" in lru_set

    with pytest.raises(KeyError):
        lru_set.remove("a")  # Trying to delete a non-existing item should raise KeyError.

def test_len_and_contains(lru_set: LRUFreqSet):
    """Test the __len__ and __contains__ methods."""
    lru_set.add("a")
    lru_set.add("b")

    assert len(lru_set) == 2
    assert "a" in lru_set
    assert "b" in lru_set
    assert "non_existing" not in lru_set

def test_get_freq_items_all(lru_set_with_entries):
    lru_set = lru_set_with_entries
    result = list(lru_set.get_freq_items())
    expected = ["A", "B", "C", "D", "E"]  # Sorted by frequency, descending
    assert result == expected, f"Expected {expected}, got {result}"

def test_get_freq_items_top_n(lru_set_with_entries):
    lru_set = lru_set_with_entries
    result = list(lru_set.get_freq_items(n=3))
    expected = ["A", "B", "C"]  # Top 3 by frequency
    assert result == expected, f"Expected {expected}, got {result}"

def test_get_freq_items_exclude_items(lru_set_with_entries):
    lru_set = lru_set_with_entries
    result = list(lru_set.get_freq_items(exclude_items=["A", "B"]))
    expected = ["C", "D", "E"]  # Excludes "A" and "B"
    assert result == expected, f"Expected {expected}, got {result}"

def test_get_freq_items_top_n_exclude_items(lru_set_with_entries):
    lru_set = lru_set_with_entries
    result = list(lru_set.get_freq_items(n=2, exclude_items=["A"]))
    expected = ["B", "C"]  # Top 2 excluding "A"
    assert result == expected, f"Expected {expected}, got {result}"

def test_get_freq_items_exclude_all(lru_set_with_entries):
    lru_set = lru_set_with_entries
    result = list(lru_set.get_freq_items(exclude_items=["A", "B", "C", "D", "E"]))
    expected: List[str] = []  # Excludes all items
    assert result == expected, f"Expected {expected}, got {result}"

def test_get_freq_items_empty_set():
    lru_set = LRUFreqSet(capacity=3)
    result = list(lru_set.get_freq_items())
    expected: List[str] = []  # No items in the set
    assert result == expected, f"Expected {expected}, got {result}"

def test_get_freq_items_no_exclusions(lru_set_with_entries):
    lru_set = lru_set_with_entries
    result = list(lru_set.get_freq_items(n=4, exclude_items=[]))
    expected = ["A", "B", "C", "D"]  # Top 4 with no exclusions
    assert result == expected, f"Expected {expected}, got {result}"
