import numpy as np
from abc import ABC, abstractmethod
from typing import Any, Optional

class Optimizer(ABC):
    def __init__(self):
        pass

    @property
    def name(self) -> str:
        """
        Get the name of the optimizer.
        :return: The name of the optimizer
        """
        return self.__class.__name__.lower()

    @abstractmethod
    def update_gradients(self, item_idx: int, grad: float) -> float:
        """
        Update gradient accumulation.
        :param item_idx: Index of the item
        :param grad: Gradient to accumulate
        :return: Updated gradient
        """
        pass

class AdaGrad(Optimizer):
    def __init__(self, eps: float=1.0):
        """
        AdaGrad Constructor
        :param num_items: Number of items in the dataset (optional)
        """
        super().__init__()
        self.eps = eps
        self.grad_squared_accum: dict[int, float] = {}

    def update_gradients(self, item_idx: int, grad: float) -> float:
        """
        Update gradient accumulation for AdaGrad.
        :param item_idx: Index of the item
        :param grad: Gradient to accumulate
        :return: Updated gradient
        """

        old_gg = self.grad_squared_accum.get(item_idx, 0.0)
        new_gg = old_gg + grad ** 2
        self.grad_squared_accum[item_idx] = new_gg
        return grad / (np.sqrt(old_gg) + self.eps)


class FTRL(Optimizer):
    def __init__(self, alpha: float = 0.5, beta: float = 1.0, lambda1: float = 0.0002, lambda2: float = 0.0001):
        """
        FTRL Constructor
        :param alpha: Learning rate
        :param beta: Parameter for adaptive learning rate
        :param lambda1: L1 regularization parameter
        :param lambda2: L2 regularization parameter
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.z: dict[int, float] = {}
        self.n: dict[int, float] = {}
        self.W: dict[int, float] = {}

    def update_gradients(self, item_idx: int, grad: float) -> float:
        """
        Update gradient accumulation for FTRL.
        :param item_idx: Index of the item
        :param grad: Gradient to accumulate
        :return: Updated gradient
        """
        # Retrieve or initialize the accumulated values
        z_val = self.z.get(item_idx, 0.0)
        n_val = self.n.get(item_idx, 0.0)

        # Update the accumulation values
        n_new = n_val + grad ** 2
        sigma = (np.sqrt(n_new) - np.sqrt(n_val)) / self.alpha
        z_new = z_val + grad - sigma * self.W.get(item_idx, 0.0)

        if abs(z_new) <= self.lambda1:
            self.W.pop(item_idx, None)
            return 0.0

        # Update the parameter with L1 and L2 regularization
        self.z[item_idx] = z_new
        self.n[item_idx] = n_new
        
        # Compute the weight update
        if np.abs(z_new) <= self.lambda1:
            weight_update = 0.0
            self.W.pop(item_idx, None)
        else:
            weight_update = - (z_new - np.sign(z_new) * self.lambda1) / ((self.beta + np.sqrt(n_new)) / self.alpha + self.lambda2)
            self.W[item_idx] = weight_update

        # ensure weight_update is finite
        if not np.isfinite(weight_update):
            raise ValueError(f"Weight update is not finite: {weight_update}")

        # return 0.0 if weight_update is close to zero
        if abs(weight_update) < 1e-8:
            self.W.pop(item_idx, None)
            return 0.0

        return weight_update

def get_optimizer(name: str, **kwargs: Any) -> Optimizer:
    """
    Create an instance of an Optimizer based on the provided name and keyword arguments.

    :param name: The type of optimizer to create ("adagrad" or "ftrl").
    :param kwargs: Additional parameters required for the optimizer initialization.
    :return: An instance of the specified Optimizer subclass.
    :raises ValueError: If an unknown optimizer name is provided or required parameters are missing.
    """
    name = name.lower()

    if name == "adagrad":
        eps = kwargs.get("eps", 1.0)
        return AdaGrad(eps)

    elif name == "ftrl":
        alpha = kwargs.get("alpha", 0.5)
        beta = kwargs.get("beta", 1.0)
        lambda1 = kwargs.get("lambda1", 0.0002)
        lambda2 = kwargs.get("lambda2", 0.0001)
        return FTRL(alpha, beta, lambda1, lambda2)

    else:
        raise ValueError(f"Unknown optimizer type '{name}'. Available options: 'adagrad', 'ftrl'.")
