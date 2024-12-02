import bisect
import numpy as np
from typing import List, Optional
from scipy.sparse import csr_matrix

class ImplicitFactorizationMachines:
    def __init__(self,
                 num_factors: int = 10,
                 learning_rate: float = 0.01,
                 reg: float = 0.001,
                 loss: str = 'bpr',
                 random_state: Optional[int] = None,
                 epsilon: float = 1.0):
        """
        Factorization Machines for implicit feedback, inspired by LightFM.

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

        self.user_factors = None
        self.item_factors = None

        # AdaGrad caches
        self.adagrad_cache_user = None
        self.adagrad_cache_item = None
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

        # Initialize AdaGrad caches
        self.adagrad_cache_user = np.zeros_like(self.user_factors)
        self.adagrad_cache_item = np.zeros_like(self.item_factors)

        for epoch in range(epochs):
            for user in range(num_users):
                user_interactions = interactions[user].indices
                if len(user_interactions) == 0:
                    continue

                for pos_item in user_interactions:
                    if self.loss == 'bpr':
                        self._update_bpr(interactions, user, pos_item)
                    elif self.loss == 'warp':
                        self._update_warp(interactions, user, pos_item)

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
        def _sample_negative(self, interactions: csr_matrix, user: int) -> int:
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
        neg_item = self._sample_negative(interactions, user)

        user_factors = self.user_factors[user]
        pos_factors = self.item_factors[pos_item]
        neg_factors = self.item_factors[neg_item]

        x_uij = user_factors @ (pos_factors - neg_factors)
        dloss = 1 / (1 + np.exp(-x_uij)) # Sigmoid

        # Compute gradients
        grad_user = (pos_factors - neg_factors) * dloss + self.reg * user_factors
        grad_pos = user_factors * dloss + self.reg * pos_factors
        grad_neg = -user_factors * dloss + self.reg * neg_factors

        # Update AdaGrad cache for user and items
        self.adagrad_cache_user[user] += grad_user**2
        self.adagrad_cache_item[pos_item] += grad_pos**2
        self.adagrad_cache_item[neg_item] += grad_neg**2

        # Calculate learning rates
        user_lr = self.learning_rate / (np.sqrt(self.adagrad_cache_user[user] + self.epsilon))
        pos_item_lr = self.learning_rate / (np.sqrt(self.adagrad_cache_item[pos_item] + self.epsilon))
        neg_item_lr = self.learning_rate / (np.sqrt(self.adagrad_cache_item[neg_item] + self.epsilon))

        # Apply updates
        self.user_factors[user] += user_lr * grad_user
        self.item_factors[pos_item] += pos_item_lr * grad_pos
        self.item_factors[neg_item] += neg_item_lr * grad_neg

    def _update_warp(self, interactions: csr_matrix, user: int, pos_item: int) -> None:
        """Performs an update using the WARP (Weighted Approximate-Rank Pairwise) loss with AdaGrad."""
        num_items = interactions.shape[1]  # Total number of items
        pos_item_vector = self.item_factors[pos_item]  # Vector for the positive item
        user_vector = self.user_factors[user]  # Vector for the user

        # Compute the prediction for the positive item
        positive_prediction = user_vector @ pos_item_vector

        # Set the number of negative samples to try and initialize the sampled count
        sampled = 0
        while sampled < self.max_sampled:
            sampled += 1
            # Sample a random negative item ID
            neg_item = self.rng.randint(0, num_items)

            # Ensure the sampled negative item isn't already interacted with by the user
            if neg_item in interactions[user].indices:
                continue

            # Compute the negative item vector
            neg_item_vector = self.item_factors[neg_item]

            # Compute the prediction for the negative item
            negative_prediction = user_vector @ neg_item_vector

            # If the negative prediction is greater than the positive prediction minus a margin, continue
            if negative_prediction > positive_prediction - 1:
                # Calculate the loss and apply a numerical stability cap
                loss = self.weight * np.log(max(1.0, (num_items - 1) // sampled))

                # Cap loss to avoid numerical overflow
                loss = min(loss, self.max_loss)

                # Perform the WARP update
                self._warp_update(loss, user, pos_item, neg_item, user_vector, pos_item_vector, neg_item_vector)
                break

    def _warp_update(self, loss: float, user: int, pos_item: int, neg_item: int, user_repr: np.ndarray, pos_item_repr: np.ndarray, neg_item_repr: np.ndarray) -> None:
        """Performs a gradient update step for WARP with AdaGrad."""
        # Compute gradients for user and item factors
        grad_user = (pos_item_repr - neg_item_repr)
        grad_pos = user_repr - neg_item_repr
        grad_neg = -user_repr

        # Update AdaGrad caches
        self.adagrad_cache_user[user] += grad_user**2
        self.adagrad_cache_item[pos_item] += grad_pos**2
        self.adagrad_cache_item[neg_item] += grad_neg**2

        user_lr = self.learning_rate / (np.sqrt(self.adagrad_cache_user[user] + self.epsilon))
        pos_item_lr = self.learning_rate / (np.sqrt(self.adagrad_cache_item[pos_item] + self.epsilon))
        neg_item_lr = self.learning_rate / (np.sqrt(self.adagrad_cache_item[neg_item] + self.epsilon))

        # Apply parameter updates
        self.user_factors[user] -= user_lr * grad_user
        self.item_factors[pos_item] -= pos_item_lr * grad_pos
        self.item_factors[neg_item] -= neg_item_lr * grad_neg

    def predict(self, user: int, items: np.ndarray) -> np.ndarray:
        """
        Predicts the scores for a user and a set of items.

        Args:
            user (int): User index.
            items (np.ndarray): Array of item indices.

        Returns:
            np.ndarray: Predicted scores for the items.
        """
        return self.user_factors[user] @ self.item_factors[items].T

    def predict_all(self, user: int) -> np.ndarray:
        """
        Predicts the scores for all items for a given user.

        Args:
            user (int): User index.

        Returns:
            np.ndarray: Predicted scores for all items.
        """
        return self.user_factors[user] @ self.item_factors.T

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
        else:
            scores = self.predict_all(user_id)

        # Get the top-K items by sorting the predicted scores in descending order
        top_items = candidate_items[np.argsort(-scores)][:top_k]
        return top_items

