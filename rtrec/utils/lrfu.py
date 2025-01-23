import time
from collections.abc import MutableMapping
from typing import Any, Dict, Iterator, Tuple

class LRFUCache(MutableMapping):
    """
    A cache that combines Least Recently Used (LRU) and Least Frequently Used (LFU) strategies.

    Attributes:
        capacity (int): The maximum number of items the cache can hold.
        lambda_factor (float): The decay factor for the cache.
        cache (Dict[int, Tuple[Any, int, float, float]]): The internal storage for the cache.
    """

    def __init__(self, capacity: int, lambda_factor: float = 0.5) -> None:
        """
        Initializes the LRFUCache with a capacity and a lambda factor.

        Args:
            capacity (int): The maximum number of items the cache can hold.
            lambda_factor (float): The decay factor for the cache.
        """
        self.capacity = capacity
        self.lambda_factor = lambda_factor
        self.cache: Dict[int, Tuple[Any, int, float, float]] = {}

    def _calculate_score(self, item_id: int) -> float:
        """
        Calculates the combined recency and frequency score for an item.

        Args:
            item_id (int): The ID of the item.

        Returns:
            float: The combined recency and frequency score.
        """
        current_time = time.time()
        value, freq, last_access_time, _ = self.cache[item_id]
        score = freq * (self.lambda_factor ** (current_time - last_access_time))
        return score

    def _update_scores(self) -> None:
        """
        Updates the scores for all items in the cache.
        """
        for item_id in self.cache:
            value, freq, last_access_time, _ = self.cache[item_id]
            self.cache[item_id] = (value, freq, last_access_time, self._calculate_score(item_id))

    def __getitem__(self, item_id: int) -> Any:
        """
        Retrieves an item from the cache and updates its recency and frequency.

        Args:
            item_id (int): The ID of the item to retrieve.

        Returns:
            Any: The value associated with the item ID.

        Raises:
            KeyError: If the item ID is not found in the cache.
        """
        if item_id in self.cache:
            value, freq, last_access_time, _ = self.cache[item_id]
            freq += 1
            last_access_time = time.time()
            score = self._calculate_score(item_id)
            self.cache[item_id] = (value, freq, last_access_time, score)
            return value
        raise KeyError(f"Item ID {item_id} not found in cache")

    def __setitem__(self, item_id: int, value: Any) -> None:
        """
        Adds an item to the cache or updates an existing item.

        Args:
            item_id (int): The ID of the item to add or update.
            value (Any): The value of the item to add or update.
        """
        if item_id in self.cache:
            _, freq, last_access_time, _ = self.cache[item_id]
            freq += 1
            last_access_time = time.time()
            score = self._calculate_score(item_id)
            self.cache[item_id] = (value, freq, last_access_time, score)
        else:
            if len(self.cache) >= self.capacity:
                self._update_scores()
                lrfu_item = min(self.cache, key=lambda k: self.cache[k][3])
                del self.cache[lrfu_item]
            self.cache[item_id] = (value, 1, time.time(), self._calculate_score(item_id))

    def __delitem__(self, item_id: int) -> None:
        """
        Deletes an item from the cache.

        Args:
            item_id (int): The ID of the item to delete.

        Raises:
            KeyError: If the item ID is not found in the cache.
        """
        if item_id in self.cache:
            del self.cache[item_id]
        else:
            raise KeyError(f"Item ID {item_id} not found in cache")

    def __len__(self) -> int:
        """
        Returns the number of items in the cache.

        Returns:
            int: The number of items in the cache.
        """
        return len(self.cache)

    def __iter__(self) -> Iterator[int]:
        """
        Returns an iterator over the item IDs in the cache.

        Returns:
            Iterator[int]: An iterator over the item IDs in the cache.
        """
        return iter(self.cache)

    def keys(self) -> Iterator[int]:
        """
        Returns an iterator over the item IDs in the cache.

        Returns:
            Iterator[int]: An iterator over the item IDs in the cache.
        """
        return iter(self.cache.keys())

    def __contains__(self, item_id: int) -> bool:
        """
        Checks if an item ID is in the cache.

        Args:
            item_id (int): The ID of the item to check.

        Returns:
            bool: True if the item ID is in the cache, False otherwise.
        """
        return item_id in self.cache

    def __repr__(self) -> str:
        """
        Returns a string representation of the cache.

        Returns:
            str: A string representation of the cache.
        """
        return repr(self.cache)