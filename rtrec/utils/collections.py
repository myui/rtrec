from typing import Iterable, Optional, TypeVar

T = TypeVar("T")  # A generic type for elements in the IndexedSet

class IndexedSet:
    """
    A data structure that combines the features of a set and a list,
    allowing fast addition of unique keys and retrieval of their indices.
    """
    def __init__(self, iterable: Optional[Iterable[T]] = None) -> None:
        self._key_to_index = {}  # Maps keys to their indices
        self._index_to_key = []  # Maintains the order of insertion
        if iterable is not None:
            for item in iterable:
                self.add(item)

    def add(self, key: T) -> int:
        """
        Adds a unique key to the set and returns its index.

        Args:
            key: The key to add.

        Returns:
            int: The index of the key.
        """
        if key not in self._key_to_index:
            # Assign the next available index to the key
            index = len(self._index_to_key)
            self._key_to_index[key] = index
            self._index_to_key.append(key)
        return self._key_to_index[key]

    def index(self, key: T, default: int = -1) -> int:
        """
        Returns the index of a key.

        Args:
            key: The key whose index is to be retrieved.

        Returns:
            int: The index of the key.

        Raises:
            KeyError: If the key is not found in the set.
        """
        return self._key_to_index.get(key, default)

    def __len__(self):
        """Returns the number of keys in the IndexedSet."""
        return len(self._index_to_key)

    def __contains__(self, key):
        """Checks if a key exists in the IndexedSet."""
        return key in self._key_to_index

    def __iter__(self):
        """Iterates over the keys in the IndexedSet."""
        return iter(self._index_to_key)

    def __getitem__(self, index):
        """
        Gets the key at a given index.

        Args:
            index: The index to retrieve the key from.

        Returns:
            The key at the specified index.
        """
        return self._index_to_key[index]

