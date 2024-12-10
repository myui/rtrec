import bisect
import numpy as np
from typing import List, Optional
from scipy.sparse import csr_matrix
from tqdm import tqdm

from rtrec.utils.math import sigmoid

class ImplicitFactorizationMachines:
    def __init__(self,
                 num_factors: int = 10,
                 learning_rate: float = 0.05,
                 reg: float = 0.0001,
                 loss: str = 'warp',
                 random_state: Optional[int] = None,
                 epsilon: float = 1.0,
                 max_sampled: int = 10,
                 max_loss: float = 10.0):
        """
        Factorization Machines for implicit feedback.

        Args:
            num_factors (int): Number of latent factors.
            learning_rate (float): Learning rate for updates.
            reg (float): Regularization parameter.
            loss (str): Loss function to use ('bpr' or 'warp').
            random_state (Optional[int]): Random seed for reproducibility.
            epsilon (float): Smoothing term for AdaGrad.
        """
        self.num_factors = num_factors
        self.learning_rate = learning_rate
        self.reg = reg
        self.loss = loss.lower()
        assert self.loss in {'bpr', 'warp'}, "Loss must be 'bpr' or 'warp'"
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)
        self.max_sampled = max_sampled
        self.max_loss = max_loss

        self.user_factors = None
        self.item_factors = None
        # User and item biases
        self.user_biases = None
        self.item_biases = None

        # AdaGrad caches
        self.adagrad_cache_user = None
        self.adagrad_cache_item = None
        self.adagrad_cache_user_bias = None
        self.adagrad_cache_item_bias = None

        self.epsilon = epsilon

    def fit(self, interactions: csr_matrix, epochs: int = 10) -> None:
        """
        Fits the model to the provided interaction data.

        Args:
            interactions (csr_matrix): User-item interaction matrix (implicit feedback).
            epochs (int): Number of training epochs.
        """
        num_users, num_items = interactions.shape
        if self.user_factors is None:
            self.user_factors = self.rng.normal(0, 0.01, (num_users, self.num_factors))
        if self.item_factors is None:
            self.item_factors = self.rng.normal(0, 0.01, (num_items, self.num_factors))
        if self.user_biases is None:
            self.user_biases = np.zeros(num_users)
        if self.item_biases is None:
            self.item_biases = np.zeros(num_items)

        # Initialize AdaGrad caches
        self.adagrad_cache_user = np.zeros_like(self.user_factors)
        self.adagrad_cache_item = np.zeros_like(self.item_factors)
        self.adagrad_cache_user_bias = np.zeros(num_users)
        self.adagrad_cache_item_bias = np.zeros(num_items)

        with tqdm(total = num_users * epochs) as pbar:
            for epoch in range(epochs):
                for user in range(num_users):
                    user_interactions = interactions[user].indices
                    if len(user_interactions) == 0:
                        pbar.update(1)
                        continue

                    for pos_item in user_interactions:
                        if self.loss == 'bpr':
                            self._update_bpr(interactions, user, pos_item)
                        elif self.loss == 'warp':
                            self._update_warp(interactions, user, pos_item)
                    pbar.update(1)

    def partial_fit(self, interactions: csr_matrix, users: np.ndarray, items: np.ndarray) -> None:
        """
        Incrementally fits the model to new interaction data, considering new users and items.

        Args:
            interactions (csr_matrix): User-item interaction matrix (implicit feedback).
            users (np.ndarray): Array of user indices to update.
            items (np.ndarray): Array of item indices to update.
        """
        num_users, num_items = interactions.shape

        for user, pos_item in zip(users, items):
            # Ensure user and item factors are initialized for new users/items
            while user >= self.user_factors.shape[0]:
                new_user_factors = self.rng.normal(0, 0.01, (1, self.num_factors))
                self.user_factors = np.vstack((self.user_factors, new_user_factors))
                self.adagrad_cache_user = np.vstack((self.adagrad_cache_user, np.zeros((1, self.num_factors))))

            while pos_item >= self.item_factors.shape[0]:
                new_item_factors = self.rng.normal(0, 0.01, (1, self.num_factors))
                self.item_factors = np.vstack((self.item_factors, new_item_factors))
                self.adagrad_cache_item = np.vstack((self.adagrad_cache_item, np.zeros((1, self.num_factors))))

            if self.loss == 'bpr':               
                self._update_bpr(interactions, user, pos_item)
            elif self.loss == 'warp':
                self._update_warp(interactions, user, pos_item)

    def _update_bpr(self, interactions: csr_matrix, user: int, pos_item: int) -> None:
        def _sample_negative(interactions: csr_matrix, user: int) -> int:
            """Samples a random negative item for a given user using binary search and random number generation."""
            num_items = interactions.shape[1]
            positives = interactions[user].indices  # Already sorted as per the csr_matrix format
            while True:
                # Generate a random item index
                item = self.rng.integers(0, num_items)
                # Check if the item is not in the positives list using binary search
                idx = bisect.bisect_left(positives, item)
                if idx == len(positives) or positives[idx] != item:
                    return item

        """Performs an update using the BPR loss with AdaGrad."""
        neg_item = _sample_negative(interactions, user)

        user_factors = self.user_factors[user]
        pos_factors = self.item_factors[pos_item]
        neg_factors = self.item_factors[neg_item]
        user_bias = self.user_biases[user]
        pos_bias = self.item_biases[pos_item]
        neg_bias = self.item_biases[neg_item]

        x_uij = (user_factors @ (pos_factors - neg_factors)) + (pos_bias - neg_bias) + user_bias
        dloss = sigmoid(x_uij)

        # Compute gradients
        grad_user = (pos_factors - neg_factors) * dloss + self.reg * user_factors
        grad_pos = user_factors * dloss + self.reg * pos_factors
        grad_neg = -user_factors * dloss + self.reg * neg_factors
        grad_user_bias = dloss + self.reg * user_bias
        grad_pos_bias = dloss + self.reg * pos_bias
        grad_neg_bias = -dloss + self.reg * neg_bias
        
        # Update AdaGrad cache for user and items
        self.adagrad_cache_user[user] += grad_user**2
        self.adagrad_cache_item[pos_item] += grad_pos**2
        self.adagrad_cache_item[neg_item] += grad_neg**2
        self.adagrad_cache_user_bias[user] += grad_user_bias**2
        self.adagrad_cache_item_bias[pos_item] += grad_pos_bias**2
        self.adagrad_cache_item_bias[neg_item] += grad_neg_bias**2

        # Calculate learning rates
        user_lr = self.learning_rate / (np.sqrt(self.adagrad_cache_user[user] + self.epsilon))
        pos_item_lr = self.learning_rate / (np.sqrt(self.adagrad_cache_item[pos_item] + self.epsilon))
        neg_item_lr = self.learning_rate / (np.sqrt(self.adagrad_cache_item[neg_item] + self.epsilon))
        user_bias_lr = self.learning_rate / (np.sqrt(self.adagrad_cache_user_bias[user]) + self.epsilon)
        pos_bias_lr = self.learning_rate / (np.sqrt(self.adagrad_cache_item_bias[pos_item]) + self.epsilon)
        neg_bias_lr = self.learning_rate / (np.sqrt(self.adagrad_cache_item_bias[neg_item]) + self.epsilon)

        # Apply updates
        self.user_factors[user] += user_lr * grad_user
        self.item_factors[pos_item] += pos_item_lr * grad_pos
        self.item_factors[neg_item] += neg_item_lr * grad_neg
        self.user_biases[user] += user_bias_lr * grad_user_bias
        self.item_biases[pos_item] += pos_bias_lr * grad_pos_bias
        self.item_biases[neg_item] += neg_bias_lr * grad_neg_bias

    def _update_warp(self, interactions: csr_matrix, user: int, pos_item: int) -> None:
        """Performs an update using the WARP (Weighted Approximate-Rank Pairwise) loss with AdaGrad."""
        num_items = interactions.shape[1]  # Total number of items
        pos_item_vector = self.item_factors[pos_item]  # Vector for the positive item
        user_vector = self.user_factors[user]  # Vector for the user

        # Compute the prediction for the positive item
        positive_prediction = user_vector @ pos_item_vector + self.user_biases[user] + self.item_biases[pos_item]

        negative_items = np.setdiff1d(np.arange(num_items), interactions[user].indices)
        self.rng.shuffle(negative_items)
        
        # Initialize rank and loss weight
        sampled = 0
        loss = 0.0

        for neg_item in negative_items[:self.max_sampled]:
            sampled += 1

            # Compute the negative item vector
            neg_item_vector = self.item_factors[neg_item]

            # Compute the prediction for the negative item
            negative_prediction = user_vector @ neg_item_vector + self.user_biases[user] + self.item_biases[neg_item]
            
            # Negative items are sampled until a "violator" is found.
            # A violator is a negative item where the positive item's score is not sufficiently higher
            if positive_prediction - negative_prediction < 1:
                # Approx warp loss used in LightFM
                # see https://building-babylon.net/2016/03/18/warp-loss-for-implicit-feedback-recommendation/ for WARP loss derivation
                # loss = np.log(num_items - 1 // sampled) # LightFM's WARP loss
                # Non approximated WARP loss is slightly better in my experiments         
                loss = sum(1.0 / k for k in range(1, sampled + 1)) # WARP loss
                self._warp_update(loss, user, pos_item, neg_item, user_vector, pos_item_vector, neg_item_vector)
                break

    def _warp_update(self, loss: float, user: int, pos_item: int, neg_item: int, user_vector: np.ndarray, pos_item_vector: np.ndarray, neg_item_vector: np.ndarray) -> None:
        """Performs a WARP update with AdaGrad."""
        # Compute the gradient
        grad_user = loss * (pos_item_vector - neg_item_vector) + self.reg * user_vector
        grad_pos = loss * user_vector + self.reg * pos_item_vector
        grad_neg = -loss * user_vector + self.reg * neg_item_vector
        grad_user_bias = loss + self.reg * self.user_biases[user]
        grad_pos_item_bias = loss + self.reg * self.item_biases[pos_item]
        grad_neg_item_bias = -loss + self.reg * self.item_biases[neg_item]

        # Update the AdaGrad cache
        self.adagrad_cache_user[user] += grad_user**2
        self.adagrad_cache_item[pos_item] += grad_pos**2
        self.adagrad_cache_item[neg_item] += grad_neg**2
        self.adagrad_cache_user_bias[user] += grad_user_bias**2
        self.adagrad_cache_item_bias[pos_item] += grad_pos_item_bias**2
        self.adagrad_cache_item_bias[neg_item] += grad_neg_item_bias**2

        # Compute the learning rates
        user_lr = self.learning_rate / (np.sqrt(self.adagrad_cache_user[user]) + self.epsilon)
        pos_item_lr = self.learning_rate / (np.sqrt(self.adagrad_cache_item[pos_item]) + self.epsilon)
        neg_item_lr = self.learning_rate / (np.sqrt(self.adagrad_cache_item[neg_item]) + self.epsilon)
        user_bias_lr = self.learning_rate / (np.sqrt(self.adagrad_cache_user_bias[user]) + self.epsilon)
        pos_item_bias_lr = self.learning_rate / (np.sqrt(self.adagrad_cache_item_bias[pos_item]) + self.epsilon)
        neg_item_bias_lr = self.learning_rate / (np.sqrt(self.adagrad_cache_item_bias[neg_item]) + self.epsilon)

        # Update the user and item factors
        self.user_factors[user] += user_lr * (grad_user + self.reg * user_vector)
        self.item_factors[pos_item] += pos_item_lr * (grad_pos + self.reg * pos_item_vector)
        self.item_factors[neg_item] += neg_item_lr * (grad_neg + self.reg * neg_item_vector)
        self.user_biases[user] += user_bias_lr * grad_user_bias
        self.item_biases[pos_item] += pos_item_bias_lr * grad_pos_item_bias
        self.item_biases[neg_item] += neg_item_bias_lr * grad_neg_item_bias

    def predict(self, user: int, items: np.ndarray) -> np.ndarray:
        """
        Predicts the scores for a user and a set of items.

        Args:
            user (int): User index.
            items (np.ndarray): Array of item indices.

        Returns:
            np.ndarray: Predicted scores for the items.
        """
        user_factors = self.user_factors[user]
        item_factors = self.item_factors[items]
        user_bias = self.user_biases[user]
        item_bias = self.item_biases[items]
        return user_factors @ item_factors.T + user_bias + item_bias

    def predict_all(self, user: int) -> np.ndarray:
        """
        Predicts the scores for all items for a given user.

        Args:
            user (int): User index.

        Returns:
            np.ndarray: Predicted scores for all items.
        """
        return self.user_factors[user] @ self.item_factors.T + self.user_biases[user] + self.item_biases

    def recommend(self, user_id: int, interactions: csr_matrix, top_k: int = 10, exclude_seen: bool = True) -> List[int]:
        """
        Recommend top-K items for a given user.

        Args:
            user_id (int): ID of the user (row index in interactions).
            interactions (csr_matrix): User-item interaction matrix.
            top_k (int): Number of recommendations to return.
            exclude_seen (bool): Whether to exclude items the user has already interacted with.

        Returns:
            List of recommended item indices.
        """
        if exclude_seen:
            num_items = interactions.shape[1]
            seen_items = interactions[user_id].indices
            candidate_items = np.setdiff1d(np.arange(num_items), seen_items)
            scores = self.predict(user_id, candidate_items)
            # Get the top-K items by sorting the predicted scores in descending order
            top_items = candidate_items[np.argsort(-scores)][:top_k]
        else:
            scores = self.predict_all(user_id)
            top_items = np.argsort(-scores)[:top_k]

        return top_items
