from scipy.special import expit
from typing import Any, List, Tuple

from .base import ImplicitFeedbackRecommender
from ..utils.optim import get_optimizer
from ..utils.regularization import get_regularization
from ..utils.eta import get_eta_estimator
from ..utils.matrix import DoKMatrix

class BPR_SLIM(ImplicitFeedbackRecommender):
    def __init__(self, **kwargs: Any):
        """
        Initialize the BPRSLIM model.
        :param kwargs: Additional keyword arguments for the model
        """
        self.eta = get_eta_estimator(**kwargs)
        self.regularization = get_regularization(**kwargs)
        self.optimizer = get_optimizer(**kwargs)

        # Initialize item-to-item similarity matrix as DoK matrix
        self.W = DoKMatrix()

    def _predict(self, items: List[int]) -> List[float]:
        """
        Predict scores for a list of items.
        """
        return [self.W.row_sum(item) for item in items]

    def _update(self, user: int, positive_item: int, negative_item: int) -> None:
        """
        Incremental weight update based on BPR loss.
        :param user: User index
        :param positive_item: Item index for the positive sample
        :param negative_item: Item index for the negative sample
        """

        self.eta.update()
    
        user_items = self.get_interacted_items(user)

        grad = self._bpr_loss(user_items, positive_item, negative_item)

        # get updated gradients
        pos_grad = self.optimizer.update_gradients(positive_item, grad)
        neg_grad = self.optimizer.update_gradients(negative_item, grad)

        # update item similarity matrix
        if self.optimizer.name == "ftrl":
            for user_item in user_items:
                # Note diagonal elements are not updated for item-item similarity matrix
                if user_item != positive_item:
                    if abs(pos_grad) < 1e-8:
                        del self.W[user_item, positive_item]
                    else:
                        # similarity value is increased for positive item
                        self.W[user_item, positive_item] += pos_grad
                if user_item != negative_item:
                    if abs(neg_grad) < 1e-8:
                        del self.W[user_item, negative_item]
                    else:
                        # similarity value is decreased for negative item
                        self.W[user_item, negative_item] -= neg_grad
        else:
            for user_item in user_items:
                # Note diagonal elements are not updated for item-item similarity matrix
                if user_item != positive_item:
                    # similarity value is increased for positive item
                    delta = self.eta.value * self.regularization.regularize(self.W[user_item, positive_item], pos_grad)
                    if abs(delta) < 1e-8:
                        del self.W[user_item, positive_item]
                    else:
                        self.W[user_item, positive_item] += delta

                if user_item != negative_item:
                    # similarity value is decreased for negative item
                    delta = self.eta.value * self.regularization.regularize(self.W[user_item, negative_item], neg_grad)
                    if abs(delta) < 1e-8:
                        del self.W[user_item, positive_item]
                    else:
                        self.W[user_item, negative_item] -= delta

    def _bpr_loss(self, user_items: List[int], positive_item: int, negative_item: int) -> float:
        """
        BPR Loss function
        :param user_items: user interacted items
        :param positive_item: Item index for the positive sample
        :param negative_item: Item index for the negative sample
        :return: BPR loss value
        """

        # Calculate the difference in scores between positive and negative items
        diff = sum(self.W[user_item, positive_item] - self.W[user_item, negative_item] for user_item in user_items)

        # minus in order for the exponent of the exponential to be positive
        return expit(-diff) # a.k.a. logistic sigmoid function
