import shelve
from collections import OrderedDict
from typing import Any

class PersistentCache:
    def __init__(self, filename: str, cache_size: int = 10_000, pickle_protocol: int = 5):
        """
        Initializes the disk-backed storage with an in-memory LRU cache.

        :param filename: Path to the shelve file.
        :param cache_size: Maximum size of the in-memory LRU cache.
        :param pickle_protocol: Protocol version for pickling.
        """
        self.filename = filename
        self.cache_size = cache_size
        self.pickle_protocol = pickle_protocol
        self.lru_cache = OrderedDict()  # In-memory cache
        # flag='n' always create a new, empty database, open for reading and writing.
        self.store = shelve.open(self.filename, flag='n', protocol=pickle_protocol, writeback=False)
        self.size = 0  # Total size, counting both LRU cache and disk storage

    def get(self, key: str) -> Any:
        """
        Retrieves a value from the cache or the disk storage.

        :param key: Key to retrieve the value.
        :return: Value associated with the key, or None if the key is not found.
        """
        if key in self.lru_cache:
            # Move key to the end to mark it as recently used
            self.lru_cache.move_to_end(key)
            return self.lru_cache[key]

        # If not in cache, check on disk
        if key in self.store:
            value = self.store[key]
            self.lru_cache[key] = value
            self._evict_if_needed()
            return value

        return None

    def set(self, key: str, value: Any):
        """
        Sets a value in the cache. If the cache exceeds its size, the least recently used item is evicted.

        :param key: Key to associate with the value.
        :param value: Value to store.
        """
        self.lru_cache[key] = value
        self.lru_cache.move_to_end(key)
        self._evict_if_needed()
        self.size += 1

    def _evict_if_needed(self):
        """
        Evicts the least recently used item from the cache if it exceeds the cache size.
        The evicted item is saved to disk storage.
        """
        if len(self.lru_cache) > self.cache_size:
            oldest_key, oldest_value = self.lru_cache.popitem(last=False)
            self.store[oldest_key] = oldest_value  # Save to disk

    def delete(self, key: str):
        """
        Deletes a key-value pair from the cache and the disk storage.

        :param key: Key to delete.
        """
        if key in self.lru_cache:
            del self.lru_cache[key]
        if key in self.store:
            del self.store[key]
        self.size -= 1

    def flush(self, clear_lru_cache: bool = False):
        """
        Force flush all items in the LRU cache to disk.

        :param clear_lru_cache: If True, clears the LRU cache after flushing.
        """

        # Save all in-memory cache items to disk
        for key, value in self.lru_cache.items():
            self.store[key] = value
        self.store.sync()

        if clear_lru_cache:
            self.lru_cache.clear()

    def clear(self, memory_only: bool = True):
        """
        Clears in-memory cache.

        :param memory_only: Whether to clear only the in-memory cache.
        """
        if memory_only:
            for key in self.lru_cache:
                if key not in self.store:
                    self.size -= 1
            self.lru_cache.clear()
        else:
            self.lru_cache.clear()
            self.store.clear()
            self.size = 0

    def close(self):
        """
        Flushes the cache to disk and closes the storage.
        """
        for key, value in self.lru_cache.items():
            self.store[key] = value
        self.store.close() # Note close() calls sync() internally

    def __getitem__(self, key: str) -> Any:
        """
        Dict-like access interface for getting an item.

        :param key: Key to retrieve the value.
        """
        return self.get(key)

    def __setitem__(self, key: str, value: Any):
        """
        Dict-like access interface for setting an item.

        :param key: Key to associate with the value.
        :param value: Value to store.
        """
        self.set(key, value)

    def __delitem__(self, key: str):
        """
        Dict-like access interface for deleting an item.

        :param key: Key to delete.
        """
        self.delete(key)

    def __contains__(self, key: str) -> bool:
        """
        Dict-like access interface for checking if a key exists.

        :param key: Key to check.
        :return: True if the key exists, False otherwise.
        """
        return key in self.lru_cache or key in self.store

    def __len__(self) -> int:
        """
        Returns the total size of the cache (LRU cache + disk storage).

        :return: Total number of items in the cache.
        """
        return self.size

