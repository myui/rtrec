from typing import Any, List

from .base import ExplictFeedbackRecommender

import numpy as np

class SLIM_MSE(ExplictFeedbackRecommender):
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

        self.optimizer = FTRL(**kwargs)
        self.W = self.optimizer.W

        self.cumulative_loss = 0.0
        self.steps = 0

    def get_emprical_loss(self) -> float:
        if self.steps == 0:
            return 0.0
        return self.cumulative_loss / self.steps

    def _predict(self, items: List[int]) -> List[float]:
        """
        Predict scores for a list of items.
        """
        return [self.W.row_sum(item) for item in items]

    def _update(self, user: int, item_id: int, rating: float) -> None:
        """
        Incremental weight update based on SLIM loss.
        :param user: User index
        :param item_id: Item index
        :param rating: Rating value
        """

        user_items = self.get_interacted_items(user)

        # Compute the gradient (MSE)        
        dloss = self._predict_rating(user, item_id, rating, user_items) - rating
        self.cumulative_loss += dloss**2
        self.steps += 1

        # update item similarity matrix
        for user_item in user_items:
            # Note diagonal elements are not updated for item-item similarity matrix
            if user_item == item_id:
                continue

            grad = dloss * self.get_rating(user, user_item)
            self.ftrl.update_gradients((user_item, item_id), grad)

    def _predict_rating(self, user: int, item_id: int, rating: float, user_items: List[int]) -> float:
        """
        Compute the derivative of the loss function.
        """
        predicted = 0.0

        for user_item in self.get_interacted_items(user):
            predicted += self.W[user_item, item_id] * self.get_rating(user, user_item)

        return predicted

class FTRL():
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
        self.z: dict[tuple[int, int], float] = {}
        self.n: dict[tuple[int, int], float] = {}
        self.W: dict[tuple[int, int], float] = {}

    def update_gradients(self, key: tuple[int, int], grad: float) -> float:
        """
        Update gradient accumulation for FTRL.
        :param key: Tuple of two integers representing the indices (e.g., (i, j)).
        :param grad: Gradient to accumulate.
        :return: Updated weight.
        """
        # Retrieve or initialize the accumulated values
        z_val = self.z.get(key, 0.0)
        n_val = self.n.get(key, 0.0)

        # Update the accumulation values
        n_new = n_val + grad ** 2
        sigma = (np.sqrt(n_new) - np.sqrt(n_val)) / self.alpha
        z_new = z_val + grad - sigma * self.W.get(key, 0.0)

        # Apply L1 regularization
        if abs(z_new) <= self.lambda1:
            self.W.pop(key, None)
            return 0.0

        # Update the parameter with L1 and L2 regularization
        self.z[key] = z_new
        self.n[key] = n_new
        
        # Compute the weight update
        weight_update = - (z_new - np.sign(z_new) * self.lambda1) / ((self.beta + np.sqrt(n_new)) / self.alpha + self.lambda2)

        # Ensure weight_update is finite
        if not np.isfinite(weight_update):
            raise ValueError(f"Weight update is not finite: {weight_update}")

        # Return 0.0 if weight_update is close to zero
        if abs(weight_update) < 1e-8:
            self.W.pop(key, None)
            return 0.0
        
        self.W[key] = weight_update
        return weight_update
