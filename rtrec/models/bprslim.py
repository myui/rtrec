from scipy.special import expit
from typing import Any, List, Tuple
from math import inf

from .base import ImplicitFeedbackRecommender
from ..utils.optim import get_optimizer
from ..utils.regularization import get_regularization
from ..utils.eta import get_eta_estimator
from ..utils.matrix import SparseMatrix

class BPR_SLIM(ImplicitFeedbackRecommender):
    """
    Bayesian Personalized Ranking (BPR) for SLIM.
    Reference: https://arxiv.org/abs/1205.2618
    https://github.com/recsyspolimi/RecSys_Course_AT_PoliMi/blob/master/Practice%2009%20-%20BPR%20for%20SLIM%20and%20MF.ipynb
    """

    def __init__(self, **kwargs: Any):
        """
        Initialize the BPRSLIM model.
        :param kwargs: Additional keyword arguments for the model
        """
        self.eta = get_eta_estimator(**kwargs)
        self.regularization = get_regularization(**kwargs)
        self.optimizer = get_optimizer(**kwargs)

        # Initialize item-to-item similarity matrix
        self.W = SparseMatrix() # target_item_id, base_item_id -> similarity
        self.cumulative_loss = 0.0
        self.steps = 0

    def get_empirical_error(self, reset: bool=False) -> float:
        if self.steps == 0:
            return 0.0
        err = self.cumulative_loss / self.steps
        if reset:
            self.cumulative_loss = 0.0
            self.steps = 0
        return err

    def _get_similarity(self, target_item_id: int, base_item_id: int) -> float:
        """
        Get the similarity between two items.
        :param target_item_id: Target item index
        :param base_item_id: Item index
        :return: Similarity between the two items
        """
        return self.W.get((target_item_id, base_item_id), -inf)

    def _predict(self, user_id: int, item_ids: List[int]) -> List[float]:
        """
        Predict scores for a list of items.
        """
        return [self.W.row_sum(iid) for iid in item_ids]

    def _update(self, user_id: int, positive_item_id: int, negative_item_id: int) -> None:
        """
        Incremental weight update based on BPR loss.
        :param user: User index
        :param positive_item: Item index for the positive sample
        :param negative_item: Item index for the negative sample
        """

        self.eta.update()
    
        user_items = self._get_interacted_items(user_id)

        grad = self._bpr_loss(user_items, positive_item_id, negative_item_id)
        self.cumulative_loss += abs(grad)
        self.steps += 1

        # get updated gradients
        pos_grad = self.optimizer.update_gradients(positive_item_id, grad)
        neg_grad = self.optimizer.update_gradients(negative_item_id, grad)

        # update item similarity matrix
        if self.optimizer.name == "ftrl":
            for user_item in user_items:
                # Note diagonal elements are not updated for item-item similarity matrix
                if user_item != positive_item_id:
                    if abs(pos_grad) < 1e-8:
                        del self.W[positive_item_id, user_item]
                    else:
                        # similarity value is increased for positive item
                        self.W[positive_item_id, user_item] += pos_grad
                if user_item != negative_item_id:
                    if abs(neg_grad) < 1e-8:
                        del self.W[negative_item_id, user_item]
                    else:
                        # similarity value is decreased for negative item
                        self.W[negative_item_id, user_item] -= neg_grad
        else:
            for user_item in user_items:
                # Note diagonal elements are not updated for item-item similarity matrix
                if user_item != positive_item_id:
                    # similarity value is increased for positive item
                    delta = self.eta.value * self.regularization.regularize(self.W[positive_item_id, user_item], pos_grad)
                    if abs(delta) < 1e-8:
                        del self.W[positive_item_id, user_item]
                    else:
                        self.W[positive_item_id, user_item] += delta

                if user_item != negative_item_id:
                    # similarity value is decreased for negative item
                    delta = self.eta.value * self.regularization.regularize(self.W[negative_item_id, user_item], neg_grad)
                    if abs(delta) < 1e-8:
                        del self.W[positive_item_id, user_item]
                    else:
                        self.W[negative_item_id, user_item] -= delta

    def _bpr_loss(self, user_item_ids: List[int], positive_item_id: int, negative_item_id: int) -> float:
        """
        BPR Loss function
        :param user_items: user interacted items
        :param positive_item: Item index for the positive sample
        :param negative_item: Item index for the negative sample
        :return: BPR loss value
        """

        # Calculate the difference in scores between positive and negative items
        diff = sum(self.W[positive_item_id, uid] - self.W[negative_item_id, uid] for uid in user_item_ids)

        # minus in order for the exponent of the exponential to be positive
        return expit(-diff) # a.k.a. logistic sigmoid function
