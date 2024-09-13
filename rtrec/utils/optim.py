import numpy as np
from abc import ABC, abstractmethod
from typing import Optional

class Optimizer(ABC):
    def __init__(self):
        pass

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
    def __init__(self, alpha: float, beta: float, lambda1: float, lambda2: float):
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
        z_new = z_val + grad - (np.sqrt(n_new) - np.sqrt(n_val)) / self.alpha * z_val

        # Update the parameter with L1 and L2 regularization
        self.z[item_idx] = z_new
        self.n[item_idx] = n_new
        
        # Compute the weight update
        if np.abs(z_new) <= self.lambda1:
            weight_update = 0.0
        else:
            weight_update = - (z_new - np.sign(z_new) * self.lambda1) / ((self.beta + np.sqrt(n_new)) / self.alpha + self.lambda2)
        
        return weight_update