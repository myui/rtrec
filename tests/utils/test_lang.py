from typing import Dict, Any

import pytest
from rtrec.utils.lang import extract_func_args

def test_extract_func_args():
    def sample_function(a, b, c):
        return a + b + c

    kwargs: Dict[str, Any] = {'a': 1, 'b': 2, 'c': 3, 'd': 4}
    expected_args = {'a': 1, 'b': 2, 'c': 3}

    result = extract_func_args(sample_function, kwargs)
    assert result == expected_args

def test_extract_func_args_missing_keys():
    def sample_function(a, b, c):
        return a + b + c

    kwargs: Dict[str, Any] = {'a': 1, 'b': 2}
    expected_args = {'a': 1, 'b': 2}

    result = extract_func_args(sample_function, kwargs)
    assert result == expected_args

def test_extract_func_args_no_matching_keys():
    def sample_function(a, b, c):
        return a + b + c

    kwargs: Dict[str, Any] = {'x': 10, 'y': 20}
    expected_args = {}

    result = extract_func_args(sample_function, kwargs)
    assert result == expected_args

if __name__ == "__main__":
    pytest.main()
