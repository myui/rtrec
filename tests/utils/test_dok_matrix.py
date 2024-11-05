import pytest
from rtrec.utils.matrix import DoKMatrix

def test_set_and_get_item():
    matrix = DoKMatrix()
    matrix[1, 2] = 5.0
    assert matrix[1, 2] == 5.0
    assert matrix[0, 0] == 0.0

def test_set_and_get_item_zero():
    matrix = DoKMatrix()
    matrix[1, 2] = 5.0
    matrix[1, 2] = 0.0
    assert matrix[1, 2] == 0.0

def test_delete_item():
    matrix = DoKMatrix()
    matrix[1, 2] = 5.0
    assert matrix.len() == 1
    del matrix[1, 2]
    assert matrix.len() == 0
    del matrix[1, 3]
    assert matrix.len() == 0
    assert matrix[1, 2] == 0.0

def test_max_dimensions():
    matrix = DoKMatrix()
    matrix[4, 5] = 1.0
    assert matrix.max_row == 5
    assert matrix.max_col == 6

def test_get_row():
    matrix = DoKMatrix()
    matrix[1, 2] = 3.0
    matrix[1, 3] = 4.0
    assert matrix.max_row == 2
    assert matrix.max_col == 4
    assert matrix.get_row(1) == [0, 0, 3.0, 4.0]

def test_get_column():
    matrix = DoKMatrix()
    matrix[2, 1] = 5.0
    matrix[4, 1] = 6.0
    assert matrix.max_row == 5
    assert matrix.max_col == 2
    assert matrix.get_column(1) == [0, 0, 5.0, 0, 6.0]

if __name__ == "__main__":
    pytest.main()
