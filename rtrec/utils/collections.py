from bisect import bisect_left
from typing import Iterable, Optional, Iterator, List, Set, TypeVar

T = TypeVar("T")  # A generic type for elements in the SortedSet

class SortedSet:
    def __init__(self, iterable: Optional[Iterable[T]] = None) -> None:
        self._list: List[T] = []
        self._set: Set[T] = set()
        if iterable is not None:
            for item in iterable:
                self.add(item)

    def add(self, item: T) -> int:
        """
        Adds an item to the sorted set if it's not already present and returns its index.
        If the item is already present, returns its existing index.
        """

        if item in self._set:
            return self.index(item)  # Return the existing index
        else:
            index = bisect_left(self._list, item)
            self._set.add(item)
            self._list.insert(index, item)
            return index  # Return the inserted index

    def index(self, item: T) -> int:
        """
        Returns the index of the item in the sorted set.

        Returns:
            int: Index of the item in the sorted set. -(insertion point+1) if the item is not found.
        """
        left, right = 0, len(self._list)
        while left < right:
            mid = (left + right) // 2
            mid_value = self._list[mid]  # Cache self._list[mid] access
            if mid_value < item:
                left = mid + 1
            elif mid_value > item:
                right = mid
            else:
                return mid  # Item found
        return -(left + 1)  # Item not found, return insertion point

    def __len__(self) -> int:
        return len(self._list)

    def __contains__(self, item: T) -> bool:
        return item in self._set

    def __iter__(self) -> Iterator[T]:
        return iter(self._list)

    def __repr__(self) -> str:
        return f"SortedSet({self._list})"
