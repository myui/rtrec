import numpy as np

@staticmethod
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
