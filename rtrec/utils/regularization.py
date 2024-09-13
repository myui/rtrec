from abc import ABC, abstractmethod
from typing import Union

class Regularization(ABC):
    def __init__(self, lambda_: float = 0.0001):
        self.lambda_ = lambda_

    def regularize(self, weight: float, gradient: float) -> float:
        """Apply the regularization to the gradient."""
        return gradient + self.lambda_ * self.regularization_term(weight)

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
