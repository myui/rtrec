import pytest

from rtrec.utils.cache import LRUFreqCache

# Assuming the LRUFreqCache class is already defined as given in previous implementation

@pytest.fixture
def cache():
    """Fixture to create a new LRUFreqCache instance for each test."""
    return LRUFreqCache(capacity=3)

def test_set_and_get(cache):
    """Test adding and retrieving items from the cache."""
    cache["a"] = "apple"
    cache["b"] = "banana"
    cache["c"] = "cherry"

    assert cache["a"] == "apple"
    assert cache["b"] == "banana"
    assert cache["c"] == "cherry"

def test_eviction_on_capacity(cache):
    """Test eviction behavior when the cache exceeds capacity."""
    cache["a"] = "apple"
    cache["b"] = "banana"
    cache["c"] = "cherry"
    cache["d"] = "date"

    # After adding the 4th item, the least recently used ("a") should be evicted.
    assert "a" not in cache
    assert "b" in cache
    assert "c" in cache
    assert "d" in cache

def test_update_frequency(cache):
    """Test that frequency is updated when an item is accessed."""
    cache["a"] = "apple"
    cache["b"] = "banana"

    # Access "a" to update its frequency
    cache["a"]

    # Add a third item to force eviction
    cache["c"] = "cherry"

    # "b" should be evicted because "a" has been accessed more frequently
    assert "b" not in cache
    assert "a" in cache
    assert "c" in cache

def test_get_freq_items(cache):
    """Test retrieval of the most frequently used items."""
    cache["a"] = "apple"
    cache["b"] = "banana"
    cache["c"] = "cherry"
    cache["d"] = "date"

    # Access some items to update their frequencies
    cache["a"]
    cache["a"]
    cache["c"]

    # Get top 2 most frequently used items
    freq_items = list(cache.get_freq_items(2))
    assert freq_items == [("a", 3), ("c", 1)]

    # Get all items sorted by frequency
    all_items = list(cache.get_freq_items())
    assert all_items == [("a", 3), ("c", 1), ("b", 1), ("d", 1)]

def test_get_with_default(cache):
    """Test the get method with a default value."""
    cache["a"] = "apple"
    cache["b"] = "banana"

    assert cache.get("a") == "apple"
    assert cache.get("non_existing", "default_value") == "default_value"

def test_remove_item(cache):
    """Test removing items from the cache."""
    cache["a"] = "apple"
    cache["b"] = "banana"

    del cache["a"]

    assert "a" not in cache
    assert "b" in cache

    with pytest.raises(KeyError):
        del cache["a"]  # Trying to delete a non-existing item should raise KeyError.

def test_len_and_contains(cache):
    """Test the __len__ and __contains__ methods."""
    cache["a"] = "apple"
    cache["b"] = "banana"

    assert len(cache) == 2
    assert "a" in cache
    assert "b" in cache
    assert "non_existing" not in cache
