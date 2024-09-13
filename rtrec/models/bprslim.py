import numpy as np
import hnswlib
from scipy.sparse import dok_matrix
from scipy.special import expit
from typing import List, Tuple

from ..utils.optim import AdaGrad
from ..utils.regularization import L1
from ..utils.eta import InvscalingEtaEstimator
from ..utils.datasets import UserItemInteractions
from ..utils.matrix import DoKMatrix

class BPRSLIM:
    def __init__(self, initial_learning_rate: float = 0.05, lambda_reg: float = 0.001, num_neighbors: int = 10, ef: int = 50):
        """
        BPRSLIM Constructor
        :param lambda_reg: Regularization parameter for weight updates
        :param initial_learning_rate: Initial learning rate for AdaGrad
        :param num_neighbors: Number of nearest neighbors for hnswlib index
        :param ef: Parameter to control speed/accuracy tradeoff in HNSW search
        """
        self.eta = InvscalingEtaEstimator(initial_eta=initial_learning_rate)
        self.regularization = L1(lambda_=lambda_reg)
        
        # Initialize user-item interactions
        self.interactions = UserItemInteractions()

        # Initialize item-to-item similarity matrix as DoK matrix
        self.W = DoKMatrix()
        
        # Initialize AdaGrad optimizer
        self.optimizer = AdaGrad()

        # HNSWLib index for fast nearest neighbor search using inner product

        # max_elements - the maximum number of elements, should be known beforehand
        # ef_construction - controls index search speed/build speed tradeoff
        # M - is tightly connected with internal dimensionality of the data
        #     strongly affects the memory consumption

        self.index = hnswlib.Index(space='ip', dim=0)
        self.index.init_index(max_elements=0, ef_construction=num_neighbors, M=16, allow_replace_deleted=True)

        # Controlling the recall by setting ef:
        # higher ef leads to better accuracy, but slower search
        self.index.set_ef(ef)

    def fit(self, user_interactions: List[Tuple[int, int, int]]) -> None:
        """
        Incrementally fit the BPRSLIM model with user interactions.
        :param user_interactions: List of (user, positive_item, negative_item) tuples
        """
        for user, positive_item, negative_item in user_interactions:
            # Update user-item interactions
            self.interactions.add_interaction(user, positive_item, count=1)
            self.interactions.add_interaction(user, negative_item, count=-1)
            self._update(user, positive_item, negative_item)

    def predict(self, user: int, top_k: int = 10) -> np.ndarray:
        """
        Predict top-K recommended items for a given user using HNSW index.
        :param user: User index
        :param top_k: Number of top similar items to keep for each item
        :return: List of top-K item indices recommended for the user
        """

        # Calculate the inner product between user items and the item-item similarity matrix
        # and return top-k items with the highest scores
        user_items = self.interactions.get_all_items_for_user(user)

        # Calculate the inner product between user items and the item-item similarity matrix
        # and return top-k items with the highest scores:
        # 
        #   scores = self.W[user_items].sum(axis=0)
        #   return np.argsort(-scores)[:k]

        # Use the HNSW index to get the top-K most similar items by inner product search
        distances, indices = self.index.knn_query(user_items, k=top_k)
        return indices.flatten()

    def _update(self, user: int, positive_item: int, negative_item: int) -> None:
        """
        Incremental weight update based on BPR loss.
        :param user: User index
        :param positive_item: Item index for the positive sample
        :param negative_item: Item index for the negative sample
        """
    
        user_items = self.interactions.get_all_items_for_user(user)

        grad = self._bpr_loss(user_items, positive_item, negative_item)

        # Update gradient accumulation and AdaGrad learning rates
        pos_grad = self.optimizer.update_gradients(positive_item, grad)
        neg_grad = self.optimizer.update_gradients(negative_item, grad)

        # update item similarity matrix
        for item in user_items:
            # Note diagonal elements are not updated for item-item similarity matrix
            if item != positive_item:
                self.W[positive_item, item] += self.eta.value * self.regularization.regularize(self.W[positive_item, item], pos_grad)

            if item != negative_item:
                self.W[negative_item, item] -= self.eta.value * self.regularization.regularize(self.W[negative_item, item], neg_grad)

        # Update the HNSW index incrementally for the modified items
        self._update_index(positive_item)
        self._update_index(negative_item)

    def _bpr_loss(self, user_items: List[int], positive_item: int, negative_item: int) -> float:
        """
        BPR Loss function
        :param user_items: user interacted items
        :param positive_item: Item index for the positive sample
        :param negative_item: Item index for the negative sample
        :return: BPR loss value
        """

        # Calculate the difference in scores between positive and negative items
        diff = sum(self.W[positive_item, item] - self.W[negative_item, item] for item in user_items)

        # minus in order for the exponent of the exponential to be positive
        return expit(-diff) # a.k.a. logistic sigmoid function

    def _update_index(self, item_idx: int) -> None:
        """
        Update the HNSW index for a specific item after its weights are modified.
        :param item_idx: Index of the item whose weights have been updated
        """
        if item_idx >= self.index.max_elements:
            self.index.init_index(max_elements=item_idx + 1, ef_construction=self.index.ef_construction)
        self.index.mark_deleted(item_idx)
        item_vector = self.W[item_idx, :].toarray().flatten()
        self.index.add_items(item_vector.reshape(1, -1), ids=[item_idx], replace_deleted=True)
