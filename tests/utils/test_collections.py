import pytest

from rtrec.utils.collections import IndexedSet

def test_add():
    indexed_set = IndexedSet()
    assert indexed_set.add("apple") == 0  # First entry should have index 0
    assert indexed_set.add("banana") == 1  # Second entry should have index 1
    assert indexed_set.add("apple") == 0  # Duplicate entry should return the existing index

def test_index():
    indexed_set = IndexedSet()
    indexed_set.add("apple")
    indexed_set.add("banana")
    assert indexed_set.index("apple") == 0
    assert indexed_set.index("banana") == 1

    assert indexed_set.index("cherry") == -1
    assert indexed_set.index("cherry", default=-2) == -2

def test_contains():
    indexed_set = IndexedSet()
    indexed_set.add("apple")
    assert "apple" in indexed_set
    assert "banana" not in indexed_set

def test_length():
    indexed_set = IndexedSet()
    assert len(indexed_set) == 0  # Initially empty
    indexed_set.add("apple")
    indexed_set.add("banana")
    assert len(indexed_set) == 2
    indexed_set.add("apple")  # Duplicate entry
    assert len(indexed_set) == 2  # Length should remain the same

def test_iteration():
    indexed_set = IndexedSet()
    indexed_set.add("apple")
    indexed_set.add("banana")
    assert list(indexed_set) == ["apple", "banana"]  # Order should match insertion

def test_getitem():
    indexed_set = IndexedSet()
    indexed_set.add("apple")
    indexed_set.add("banana")
    assert indexed_set[0] == "apple"
    assert indexed_set[1] == "banana"

    with pytest.raises(IndexError):
        _ = indexed_set[2]  # Accessing out of range index should raise IndexError

# Run tests
if __name__ == "__main__":
    pytest.main()
