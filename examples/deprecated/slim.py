import numpy as np

from typing import Any, Optional
from math import inf

from ...rtrec.models.base import BaseModel

class SLIM_MSE(BaseModel):
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

        self.ftrl = FTRL(**kwargs)

        self.n_recent = kwargs.get('n_recent', 100)

        # item-item similarity matrix
        # target_item_id, base_item_id -> similarity
        self.W = self.ftrl.W

        self.cumulative_loss = 0.0
        self.steps = 0

    def get_empirical_error(self) -> float:
        if self.steps == 0:
            return 0.0
        return self.cumulative_loss / self.steps

    def _get_similarity(self, target_item_id: int, base_item_id: int) -> float:
        """
        Get the similarity between two items.
        :param target_item_id: Target item index
        :param base_item_id: Item index
        :return: Similarity between the two items
        """
        return self.W.get((target_item_id, base_item_id), -inf)

    def _update(self, user_id: int, item_id: int) -> None:
        """
        Incremental weight update based on SLIM loss.
        :param user: User index
        :param item_id: Item index
        """
        # Compute the gradient (MSE)
        dloss = self._predict_rating(user_id, item_id, self.n_recent) - self._get_rating(user_id, item_id)
        self.cumulative_loss += abs(dloss)
        self.steps += 1

        # update item similarity matrix
        # Note: Change applied to the original SLIM algorithm;only update the similarity matrix
        # where the user has (recently) interacted with the item.
        #
        # No interaction implies no update; grad = dloss * rating = 0
        # see discussions in https://github.com/MaurizioFD/RecSys_Course_AT_PoliMi/issues/22
        user_item_ids = self._get_interacted_items(user_id, n_recent=self.n_recent)
        for user_item_id in user_item_ids:
            # Note diagonal elements are not updated for item-item similarity matrix
            if user_item_id == item_id:
                continue

            grad = dloss * self._get_rating(user_id, user_item_id)
            self.ftrl.update_gradients((user_item_id, item_id), grad)

    def _predict_rating(self, user_id: int, item_id: int, bypass_prediction: bool=False, n_recent: Optional[int] = None) -> float:
        user_item_ids = self._get_interacted_items(user_id, n_recent=n_recent)
        if bypass_prediction:
            if len(user_item_ids) == 1 and user_item_ids[0] == item_id:
                # return raw rating if user has only interacted with the item
                return self._get_rating(user_id, item_id)

        predicted = 0.0
        for user_item_id in user_item_ids:
            if user_item_id == item_id:
                continue # diagonal elements are not updated for item-item similarity matrix
            predicted += self.W.get((user_item_id, item_id), 0.0) * self._get_rating(user_id, user_item_id)

        return predicted

class FTRL():
    def __init__(self, alpha: float = 0.01, beta: float = 1.0, lambda1: float = 0.0002, lambda2: float = 0.0001, **kwargs: Any):
        """
        FTRL Constructor
        :param alpha: Learning rate
        :param beta: A kind of smoothing parameter of AdaGrad (denominator)
        :param lambda1: L1 regularization parameter
        :param lambda2: L2 regularization parameter

        Ref for the FTRL algorithm:
        - https://www.eecs.tufts.edu/~dsculley/papers/ad-click-prediction.pdf
        - https://arxiv.org/abs/1403.3465
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
