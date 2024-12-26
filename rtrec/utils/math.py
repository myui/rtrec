import numpy as np

def sigmoid(x):
    """
    sigmoid function

    Parameters
    ----------
    x : array_like, shape=(n_data), dtype=float
        arguments of function

    Returns
    -------
    sig : array, shape=(n_data), dtype=float
        1.0 / (1.0 + exp(- x))
    """

    # restrict domain of sigmoid function within [1e-15, 1 - 1e-15]
    sigmoid_range = 34.538776394910684
    x = np.clip(x, -sigmoid_range, sigmoid_range)

    return 1.0 / (1.0 + np.exp(-x))

def calc_norm(factors: np.ndarray, avoid_zeros: bool = False) -> np.ndarray:
    norm = np.linalg.norm(factors, axis=1)
    # don't divide by zero in similar_items, replace with small value
    if avoid_zeros:
        norm[norm == 0] = 1e-10
    return norm
