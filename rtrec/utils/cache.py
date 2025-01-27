from collections import OrderedDict
from collections.abc import MutableMapping
from typing import Any, Tuple, Iterator, Optional

class LRUFreqCache(MutableMapping):
    def __init__(self, capacity: int):
        """
        Initialize the LRU with frequency cache.

        Parameters:
            capacity (int): Maximum number of entries in the cache.
        """
        if capacity <= 0:
            raise ValueError("Capacity must be greater than 0.")

        self.capacity = capacity
        self.data: OrderedDict[Any, Tuple[Any, int]] = OrderedDict()  # key -> (value, frequency)

    def __getitem__(self, key: Any) -> Any:
        """
        Retrieve the value associated with the key and update its frequency.

        Parameters:
            key (Any): The key to retrieve.

        Returns:
            Any: The value associated with the key.

        Raises:
            KeyError: If the key is not found in the cache.
        """
        if key not in self.data:
            raise KeyError(key)

        value, freq = self.data.pop(key)
        freq += 1
        self.data[key] = (value, freq)

        return value

    def get(self, key: Any, default_value: Optional[Any] = None) -> Optional[Any]:
        """
        Retrieve the value associated with the key, or return the default value if the key is not found.

        Parameters:
            key (Any): The key to retrieve.
            default_value (Optional[Any]): The value to return if the key is not found.

        Returns:
            Optional[Any]: The value associated with the key, or the default value.
        """
        if key not in self.data:
            return default_value

        value, freq = self.data.pop(key)
        freq += 1
        self.data[key] = (value, freq)

        return value

    def __setitem__(self, key: Any, value: Any) -> None:
        """
        Add a key-value pair to the cache or update the value and frequency if the key already exists.

        Parameters:
            key (Any): The key to add or update.
            value (Any): The value to associate with the key.
        """
        if key in self.data:
            _, freq = self.data.pop(key)
            self.data[key] = (value, freq + 1)
        else:
            if len(self.data) >= self.capacity:
                # Inline eviction of the least recently used item
                self.data.popitem(last=False)  # Evict the least recently used item
            self.data[key] = (value, 1)

    def __delitem__(self, key: Any) -> None:
        """
        Remove a key-value pair from the cache.

        Parameters:
            key (Any): The key to remove.

        Raises:
            KeyError: If the key is not found in the cache.
        """
        if key not in self.data:
            raise KeyError(key)

        self.data.pop(key, None)

    def __len__(self) -> int:
        """
        Get the number of items in the cache.

        Returns:
            int: The number of items in the cache.
        """
        return len(self.data)

    def __iter__(self) -> Iterator[Any]:
        """
        Iterate over the keys in the cache.

        Returns:
            Iterator[Any]: An iterator over the keys.
        """
        return iter(self.data)

    def __contains__(self, key: Any) -> bool:
        """
        Check if a key exists in the cache.

        Parameters:
            key (Any): The key to check.

        Returns:
            bool: True if the key exists, False otherwise.
        """
        return key in self.data

    def keys(self) -> Iterator[Any]:
        """
        Get an iterator over the keys in the cache.

        Returns:
            Iterator[Any]: An iterator over the keys.
        """
        return iter(self.data.keys())

    def items(self) -> Iterator[Tuple[Any, Any]]:
        """
        Get an iterator over the key-value pairs in the cache.

        Returns:
            Iterator[Tuple[Any, Any]]: An iterator over the key-value pairs.
        """
        for key, (value, _) in self.data.items():
            yield key, value

    def values(self) -> Iterator[Any]:
        """
        Get an iterator over the values in the cache.

        Returns:
            Iterator[Any]: An iterator over the values.
        """
        return (value for value, _ in self.data.values())

    def __repr__(self) -> str:
        """
        Get a string representation of the LRU with frequency cache.

        Returns:
            str: A string representation of the cache.
        """
        return f"LRUFreqCache(capacity={self.capacity}, size={len(self.data)})"

    def get_freq_items(self, n: Optional[int] = None) -> Iterator[Tuple[Any, Any]]:
        """
        Retrieve the top `n` most frequently used items in the cache. If `n` is None,
        return all items sorted by frequency.

        Parameters:
            n (Optional[int]): The number of top items to retrieve. Defaults to None.

        Returns:
            Iterator[Tuple[Any, int]]: An iterator of (key, frequency) pairs, sorted by frequency in descending order.
        """
        # If n is not specified, return all items sorted by frequency
        sorted_items = sorted(self.data.items(), key=lambda item: item[1][1], reverse=True)
        for key, (value, _) in (sorted_items if n is None else sorted_items[:n]):
            yield key, value
