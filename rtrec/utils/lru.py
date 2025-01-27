from collections import OrderedDict
from collections.abc import MutableSet
from typing import Any, Iterator, List, Optional

class LRUFreqSet(MutableSet):
    def __init__(self, capacity: int):
        """
        Initialize the LRU with frequency set.

        Parameters:
            capacity (int): Maximum number of entries in the set.
        """
        if capacity <= 0:
            raise ValueError("Capacity must be greater than 0.")

        self.capacity = capacity
        self.data: OrderedDict[Any, int] = OrderedDict()  # key -> frequency

    def add(self, key: Any) -> None:
        """
        Add a key to the set or update its frequency if it already exists.

        Parameters:
            key (Any): The key to add or update.
        """
        if key in self.data:
            # Update frequency and mark as recently used
            freq = self.data.pop(key)
            self.data[key] = freq + 1
        else:
            # Evict the least recently used item if at capacity
            if len(self.data) >= self.capacity:
                self.data.popitem(last=False)
            self.data[key] = 1

    def discard(self, key: Any) -> None:
        """
        Remove a key from the set if it exists.

        Parameters:
            key (Any): The key to remove.

        Raises:
            KeyError: If the key is not found in the set.
        """
        if key not in self.data:
            raise KeyError(key)
        self.data.pop(key)

    def __contains__(self, key: Any) -> bool:
        """
        Check if a key exists in the set.

        Parameters:
            key (Any): The key to check.

        Returns:
            bool: True if the key exists, False otherwise.
        """
        return key in self.data

    def __iter__(self) -> Iterator[Any]:
        """
        Iterate over the keys in the set.

        Returns:
            Iterator[Any]: An iterator over the keys.
        """
        return iter(self.data)

    def __len__(self) -> int:
        """
        Get the number of items in the set.

        Returns:
            int: The number of items in the set.
        """
        return len(self.data)

    def __repr__(self) -> str:
        """
        Get a string representation of the LRU with frequency set.

        Returns:
            str: A string representation of the set.
        """
        return f"LRUFreqSet(capacity={self.capacity}, size={len(self.data)})"

    def get_freq_items(self, n: Optional[int] = None, exclude_items: List[Any] = []) -> Iterator[Any]:
        """
        Retrieve the top `n` most frequently used keys in the set, excluding specified items. 
        If `n` is None, return all keys sorted by frequency.

        Parameters:
            n (Optional[int]): The number of top keys to retrieve. Defaults to None.
            exclude_items (List[Any]): List of keys to exclude from the result.

        Returns:
            Iterator[Any]: An iterator of keys, sorted by frequency in descending order.
        """
        # If n is not specified, return all keys sorted by frequency
        sorted_items = sorted(self.data.items(), key=lambda item: item[1], reverse=True)

        if len(exclude_items) > 0:
            count = 0
            for key, _ in sorted_items:
                if key in exclude_items:
                    continue
                if n is not None and count >= n:
                    break
                yield key
                count += 1
        else:
            for key, _ in (sorted_items if n is None else sorted_items[:n]):
                yield key
