import os
import pytest
from tempfile import NamedTemporaryFile

from rtrec.utils.diskcache import PersistentCache

# Test class for PersistentCache
@pytest.fixture
def cache():
    # Create a temporary file to store the cache
    with NamedTemporaryFile(delete=False) as temp_file:
        filename = temp_file.name
    cache = PersistentCache(filename, cache_size=3)  # Set cache size to 3 for testing
    yield cache
    cache.close()
    os.remove(filename)  # Cleanup the file after tests

def test_set_get(cache):
    # Test setting and getting values from the cache
    cache.set("key1", "value1")
    cache.set("key2", "value2")

    assert cache.get("key1") == "value1"
    assert cache.get("key2") == "value2"
    assert cache.get("key3") is None  # Key does not exist

def test_eviction(cache):
    # Test eviction of least recently used items when cache exceeds size
    cache.set("key1", "value1")
    cache.set("key2", "value2")
    cache.set("key3", "value3")
    assert len(cache.lru_cache) == 3  # Cache should have 3 items in memory

    # Adding another item should evict the least recently used item
    cache.set("key4", "value4")
    assert "key1" not in cache.lru_cache  # "key1" should be evicted
    assert len(cache.lru_cache) == 3  # Cache should still have 3 items in memory
    assert cache.get("key1") == "value1"  # It should still be accessible from disk

def test_delete(cache):
    # Test deleting an item from the cache and disk
    cache.set("key1", "value1")
    cache.set("key2", "value2")
    cache.delete("key1")

    assert "key1" not in cache.lru_cache
    assert "key1" not in cache.store  # Should also be deleted from disk
    assert cache.get("key1") is None  # Should return None after deletion

def test_flush(cache):
    # Test flushing the LRU cache to disk
    cache.set("key1", "value1")
    cache.set("key2", "value2")
    cache.flush(clear_lru_cache=True)

    # Verify that items are saved to disk
    assert "key1" in cache.store
    assert "key2" in cache.store
    assert len(cache.lru_cache) == 0  # Cache should be cleared

def test_clear(cache):
    # Test clearing the in-memory cache
    cache.set("key1", "value1")
    cache.set("key2", "value2")
    cache.flush(clear_lru_cache=False)
    cache.clear(memory_only=True)

    assert "key1" not in cache.lru_cache
    assert "key2" not in cache.lru_cache
    assert len(cache.lru_cache) == 0  # Cache should be cleared
    assert cache.get("key1") == "value1"  # Should still be in disk storage
    # Ensure key1 moved to cache
    assert len(cache.lru_cache) == 1
    assert cache.lru_cache.get("key1") == "value1"
    assert cache.get("key2") == "value2"  # Should still be in disk storage
    # Ensure key2 moved to cache
    assert len(cache.lru_cache) == 2
    assert cache.lru_cache.get("key2") == "value2"
    assert len(cache.store) == 2 # Disk storage should be empty
    assert len(cache) == 2  # Disk storage should still have 2 items

    # Clear everything including disk storage
    cache.clear(memory_only=False)
    assert "key1" not in cache.store
    assert "key2" not in cache.store
    assert len(cache) == 0  # Disk storage should be empty

def test_persistent_cache_dict_access(cache):
    # Test dict-like access to the cache
    cache["key1"] = "value1"
    cache["key2"] = "value2"

    assert cache["key1"] == "value1"
    assert cache["key2"] == "value2"

    del cache["key1"]
    assert "key1" not in cache
    assert cache["key1"] is None  # Should return None after deletion

def test_size(cache):
    # Test the size of the cache (LRU cache + disk storage)
    cache.set("key1", "value1")
    cache.set("key2", "value2")
    cache.set("key3", "value3")
    cache.set("key4", "value4")

    assert len(cache) == 4  # Total size should be 4 (in LRU cache + disk)

def test_get_after_flush(cache):
    # Test getting an item after it was flushed to disk
    cache.set("key1", "value1")
    cache.set("key2", "value2")
    cache.flush(clear_lru_cache=False)

    # Verify that flushed items are retrievable from disk
    assert cache.get("key1") == "value1"
    assert cache.get("key2") == "value2"
