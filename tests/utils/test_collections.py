import pytest

from rtrec.utils.collections import SortedSet

def test_sorted_set_init_empty():
    """Test initialization of an empty SortedSet."""
    s = SortedSet()
    assert len(s) == 0
    assert list(s) == []
    assert repr(s) == "SortedSet([])"

def test_sorted_set_init_with_iterable():
    """Test initialization with an iterable."""
    s = SortedSet([3, 1, 4, 1, 5, 9])
    assert len(s) == 5
    assert list(s) == [1, 3, 4, 5, 9]
    assert repr(s) == "SortedSet([1, 3, 4, 5, 9])"

def test_sorted_set_add():
    """Test adding elements to the SortedSet."""
    s = SortedSet([3, 1, 4])
    s.add(2)
    assert list(s) == [1, 2, 3, 4]
    assert len(s) == 4
    s.add(3)  # Adding a duplicate should not change the set
    assert list(s) == [1, 2, 3, 4]

def test_sorted_set_index_found():
    """Test finding the index of elements that are present in the set."""
    s = SortedSet([3, 1, 4])
    assert s.index(1) == 0
    assert s.index(3) == 1
    assert s.index(4) == 2

def test_sorted_set_index_not_found():
    """Test the behavior when searching for an index of a non-existent element."""
    s = SortedSet([3, 1, 4])
    # Verify that the index method returns the correct insertion point
    assert s.index(2) == -(1 + 1)  # Expected insertion point: 1
    assert s.index(5) == -(3 + 1)  # Expected insertion point: 3
    assert s.index(0) == -(0 + 1)  # Expected insertion point: 0

def test_sorted_set_len():
    """Test the length of the SortedSet."""
    s = SortedSet()
    assert len(s) == 0
    s.add(1)
    assert len(s) == 1
    s.add(2)
    assert len(s) == 2
    s.add(2)  # Adding a duplicate should not change the length
    assert len(s) == 2

def test_sorted_set_contains():
    """Test the 'in' operation for checking membership."""
    s = SortedSet([3, 1, 4])
    assert 1 in s
    assert 2 not in s
    assert 4 in s

def test_sorted_set_iter():
    """Test iteration over the SortedSet."""
    s = SortedSet([3, 1, 4])
    assert list(iter(s)) == [1, 3, 4]

def test_sorted_set_repr():
    """Test the string representation of the SortedSet."""
    s = SortedSet([3, 1, 4])
    assert repr(s) == "SortedSet([1, 3, 4])"
