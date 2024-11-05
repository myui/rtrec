import pytest
from rtrec.utils.matrix import SparseMatrix

def test_setitem_and_getitem():
    matrix = SparseMatrix()
    matrix[1, 2] = 3.5
    assert matrix[1, 2] == 3.5, "Failed to set and get a non-zero value"

    matrix[1, 2] = 0
    assert matrix[1, 2] == 0.0, "Failed to remove element by setting it to zero"

def test_nnz_increment_and_decrement():
    matrix = SparseMatrix()
    matrix[0, 0] = 1.0
    matrix[1, 1] = 2.0
    assert matrix.nnz == 2, "NNZ did not increment correctly"

    matrix[1, 1] = 0
    assert matrix.nnz == 1, "NNZ did not decrement correctly when setting value to zero"

def test_delitem():
    matrix = SparseMatrix()
    matrix[0, 0] = 1.0
    matrix[1, 1] = 2.0
    del matrix[1, 1]
    assert matrix[1, 1] == 0.0, "Failed to delete item"
    assert matrix.nnz == 1, "NNZ did not decrement correctly on deletion"

def test_get():
    matrix = SparseMatrix()
    matrix[1, 2] = 5.0
    assert matrix.get((1, 2)) == 5.0, "Failed to retrieve existing value"
    assert matrix.get((2, 3), 0.0) == 0.0, "Failed to return default for non-existing key"

def test_len():
    matrix = SparseMatrix()
    assert matrix.len() == 0, "Length is incorrect for empty matrix"
    matrix[0, 0] = 1.0
    matrix[1, 1] = 2.0
    assert matrix.len() == 2, "Length is incorrect after adding items"

def test_row_sum():
    matrix = SparseMatrix()
    matrix[0, 1] = 2.0
    matrix[0, 2] = 3.0
    matrix[1, 0] = 4.0
    assert matrix.row_sum(0) == 5.0, "Row sum is incorrect for row 0"
    assert matrix.row_sum(1) == 4.0, "Row sum is incorrect for row 1"
    assert matrix.row_sum(2) == 0.0, "Row sum is incorrect for empty row"

if __name__ == "__main__":
    pytest.main()