from abc import ABC, abstractmethod
from typing import Union

class Regularization(ABC):
    def __init__(self, lambda_: float = 0.0001):
        self.lambda_ = lambda_

    def regularize(self, weight: float, gradient: float) -> float:
        """Apply the regularization to the gradient."""
        return gradient - self.lambda_ * self.regularization_term(weight)

    @abstractmethod
    def regularization_term(self, weight: float) -> float:
        """Compute the regularization term based on the weight."""
        pass

class PassThrough(Regularization):
    def regularization_term(self, weight: float) -> float:
        return 0.0

    def regularize(self, weight: float, gradient: float) -> float:
        return gradient


class L1(Regularization):
    def regularization_term(self, weight: float) -> float:
        return 1.0 if weight > 0 else -1.0


class L2(Regularization):
    def regularization_term(self, weight: float) -> float:
        return weight


class ElasticNet(Regularization):
    def __init__(self, lambda_: float = 0.0001, l1_ratio: float = 0.5):
        super().__init__(lambda_)
        self.l1 = L1(lambda_)
        self.l2 = L2(lambda_)
        self.l1_ratio = l1_ratio
        if not (0.0 <= self.l1_ratio <= 1.0):
            raise ValueError(f"L1 ratio should be in [0.0, 1.0], but got {self.l1_ratio}")

    def regularization_term(self, weight: float) -> float:
        l1_reg = self.l1.regularization_term(weight)
        l2_reg = self.l2.regularization_term(weight)
        return self.l1_ratio * l1_reg + (1.0 - self.l1_ratio) * l2_reg


def get_regularization(reg_type: str = 'pass_through', **kwargs) -> Regularization:
    """
    Get an instance of the Regularization class based on the provided regularization type and additional parameters.

    Parameters:
        reg_type (str): Type of regularization ('pass_through', 'l1', 'l2', 'elastic_net').
        **kwargs: Additional keyword arguments for specific regularization types:
            - lambda_ (float): The regularization parameter (default is 0.0001).
            - l1_ratio (float): The ratio of L1 regularization in ElasticNet (default is 0.5).

    Returns:
        Regularization: An instance of the corresponding Regularization subclass.

    Raises:
        ValueError: If the regularization type is unknown or if l1_ratio is out of bounds.
    """
    # Extract additional arguments with defaults
    lambda_ = kwargs.get('lambda_', 0.0001)
    l1_ratio = kwargs.get('l1_ratio', 0.5)

    if reg_type == 'pass_through':
        return PassThrough()
    elif reg_type == 'l1':
        return L1(lambda_)
    elif reg_type == 'l2':
        return L2(lambda_)
    elif reg_type == 'elastic_net':
        if not (0.0 <= l1_ratio <= 1.0):
            raise ValueError(f"L1 ratio should be in [0.0, 1.0], but got {l1_ratio}")
        return ElasticNet(lambda_, l1_ratio)
    else:
        raise ValueError(f"Unknown regularization type: {reg_type}")
