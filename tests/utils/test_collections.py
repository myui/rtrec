import pytest

from rtrec.utils.collections import SortedSet
from typing import List

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

def test_add_unique_elements():
    sset = SortedSet()
    assert sset.add(10) == 0  # Adding first element
    assert sset.add(5) == 0   # Adding element at the start
    assert sset.add(15) == 2  # Adding element at the end
    assert len(sset) == 3
    assert list(sset) == [5, 10, 15]

def test_add_duplicate_elements():
    sset = SortedSet([10, 20, 30])
    assert sset.add(20) == 1  # Element already exists, return its index
    assert len(sset) == 3  # Length should not increase
    assert list(sset) == [10, 20, 30]

def test_index_existing_elements():
    sset = SortedSet([10, 20, 30])
    assert sset.index(10) == 0  # Check index of first element
    assert sset.index(20) == 1  # Check index of middle element
    assert sset.index(30) == 2  # Check index of last element

def test_index_non_existing_elements():
    sset = SortedSet([10, 20, 30])
    assert sset.index(15) == -2  # Not found, insertion point is 1
    assert sset.index(5) == -1   # Not found, insertion point is 0
    assert sset.index(35) == -4  # Not found, insertion point is 3

def test_add_and_index_combined():
    sset = SortedSet()
    assert sset.add(50) == 0
    assert sset.index(50) == 0
    assert sset.add(30) == 0
    assert sset.index(30) == 0
    assert sset.add(70) == 2
    assert sset.index(70) == 2
    assert len(sset) == 3
    assert list(sset) == [30, 50, 70]

def test_len_and_contains():
    sset = SortedSet([5, 10, 15])
    assert len(sset) == 3
    assert 10 in sset
    assert 20 not in sset

def test_iter():
    sset = SortedSet([10, 20, 30])
    assert list(iter(sset)) == [10, 20, 30]
