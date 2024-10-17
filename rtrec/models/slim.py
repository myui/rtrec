from collections import defaultdict
from typing import Any, List

from .base import ExplictFeedbackRecommender

import numpy as np

class SLIM_MSE(ExplictFeedbackRecommender):
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

        self.ftrl = FTRL(**kwargs)
        self.W = self.ftrl.W

        self.cumulative_loss = 0.0
        self.steps = 0

    def get_empirical_loss(self) -> float:
        if self.steps == 0:
            return 0.0
        return self.cumulative_loss / self.steps

    def _predict(self, user_id: int, item_ids: List[int]) -> List[float]:
        """
        Predict scores for a list of items.
        """
        return [
            self._predict_rating(user_id, item_id, bypass_prediction=False)
            for item_id in item_ids
        ]

    def _update(self, user_id: int, item_id: int) -> None:
        """
        Incremental weight update based on SLIM loss.
        :param user: User index
        :param item_id: Item index
        """

        user_item_ids = self._get_interacted_items(user_id)

        # Compute the gradient (MSE)        
        dloss = self._predict_rating(user_id, item_id) - self._get_rating(user_id, item_id)
        self.cumulative_loss += dloss**2
        self.steps += 1

        # update item similarity matrix
        for user_item_id in user_item_ids:
            # Note diagonal elements are not updated for item-item similarity matrix
            if user_item_id == item_id:
                continue

            grad = dloss * self._get_rating(user_id, user_item_id)
            self.ftrl.update_gradients((user_item_id, item_id), grad)

    def _predict_rating(self, user_id: int, item_id: int, bypass_prediction: bool=False) -> float:
        user_item_ids = self._get_interacted_items(user_id)
        if bypass_prediction:
            if len(user_item_ids) == 1 and user_item_ids[0] == item_id:
                # return raw rating if user has only interacted with the item
                return self._get_rating(user_id, item_id)

        predicted = 0.0
        for user_item_id in user_item_ids:
            if user_item_id == item_id:
                continue # diagonal elements are not updated for item-item similarity matrix
            predicted += self.W[user_item_id, item_id] * self._get_rating(user_id, user_item_id)

        return predicted

class FTRL():
    def __init__(self, alpha: float = 0.5, beta: float = 1.0, lambda1: float = 0.0002, lambda2: float = 0.0001, **kwargs: Any):
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
        self.W: dict[tuple[int, int], float] = defaultdict(float)

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
