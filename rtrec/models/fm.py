import numpy as np
from typing import Dict, Any, Iterable, Tuple, List

from .base import ExplicitFeedbackRecommender

class FactorizationMachines(ExplicitFeedbackRecommender):
    def __init__(self, n_factors: int, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        self.ftrl = FTRL(**kwargs)

        # Initialize parameters
        self.n_factors: int = n_factors  # Number of latent factors
        self.feature_map: Dict[Any, int] = {}  # Maps feature keys to indices
        self.w: List[float] = [0.0]  # Linear weights with w[0] as global bias
        self.V: List[np.ndarray] = []  # Factor matrix (list of arrays)

        self.cumulative_loss = 0.0
        self.steps = 0

    def _get_or_create_index(self, key: Any) -> int:
        """Get the index for a feature key, creating a new one if it doesn't exist."""
        index = self.feature_map.get(key, None)

        if index is None:
            index = len(self.feature_map)  # Start from 0 for linear weights
            self.feature_map[key] = index

            # Ensure the list is long enough to accommodate this new index
            while len(self.w) <= index:
                self.w.append(0.0)  # Initialize linear weights as 0.0

            # Initialize the factor vector for the new feature
            while len(self.V) <= index:  # Factor matrix index matches directly
                self.V.append(np.random.normal(0, 0.1, self.n_factors))  # Random factor initialization

        return index

    def predict_rating(self, user: int, item: int) -> float:
        """Predict the rating for a user-item pair."""
        user_idx = self._get_or_create_index(f'u{user}')
        item_idx = self._get_or_create_index(f'i{item}')

        # Linear term (includes global bias and feature-specific biases)
        linear_term: float = self.w[0]  # Start with the global bias
        linear_term += self.w[user_idx + 1]  # Use idx+1 for user bias
        linear_term += self.w[item_idx + 1]  # Use idx+1 for item bias

        # Interaction term
        interaction_term: float = 0.0
        for f in range(self.n_factors):
            sum_vx: float = 0.0
            sum_vx_sq: float = 0.0
            for idx in [user_idx, item_idx]:
                Vi = self.V[idx]
                if Vi is None:
                    continue
                Vif = self.Vi[f]
                sum_vx += Vif
                sum_vx_sq += Vif * Vif
            interaction_term += 0.5 * (sum_vx ** 2 - sum_vx_sq)

        return linear_term + interaction_term

    def _update(self, user: int, item: int) -> None:
        """Perform a single update for the given user-item pair."""
        y = self._get_rating(user, item)  # True rating
        y_pred = self.predict_rating(user, item)  # Predicted rating
        dloss = y - y_pred  # Prediction error

        self.steps += 1
        self.cumulative_loss += abs(dloss)

        if abs(dloss) <= 1e-6:
            return

        # Update linear terms for non-zero features
        user_idx = self._get_or_create_index(f'u{user}')
        item_idx = self._get_or_create_index(f'i{item}')

        grad = dloss  # Gradient is the error for this simple regression task as feature value is 1.0
        self.w[0] += self.ftrl.update(0, grad)  # Update global bias
        self.w[user_idx + 1] += self.ftrl.update(user_idx + 1, grad)  # Update user bias
        self.w[item_idx + 1] += self.ftrl.update(item_idx + 1, grad)  # Update item bias

        # Update interaction factors (latent factors for user-item pair)
        for f in range(self.n_factors):
            sum_vx = self.V[user_idx][f] + self.V[item_idx][f]
            for idx in [user_idx, item_idx]:
                v_i_f = self.V[idx][f]
                gradient = dloss * (sum_vx - v_i_f)
                if abs(gradient) <= 1e-6:
                    continue
                self.V[idx][f] += self.ftrl.update((idx, f), gradient)

class FTRL:
    def __init__(self, alpha: float = 0.1, beta: float = 1.0, L1: float = 0.1, L2: float = 1.0, decay_rate: float = 0.9) -> None:
        self.alpha = alpha  # learning rate
        self.beta = beta    # scaling factor for regularization
        self.L1 = L1        # L1 regularization
        self.L2 = L2        # L2 regularization
        self.decay_rate = decay_rate

        self.z: Dict[Any, float] = {}  # accumulation of gradients
        self.n: Dict[Any, float] = {}  # accumulation of squared gradients

    def update(self, feature: Any, gradient: float) -> float:
        """Update weights with FTRL update rule."""
        old_z = self.z.get(feature, 0.0)
        self.z[feature] = self.z.get(feature, 0.0) + gradient - (self.L1 * np.sign(self.z.get(feature, 0.0)) + self.L2 * self.z.get(feature, 0.0))
        self.n[feature] = self.n.get(feature, 0.0) + gradient ** 2

        if np.abs(self.z[feature]) < self.L1:
            self.z[feature] = 0.0

        return (self.z[feature] - old_z) / (self.beta + np.sqrt(self.n[feature]))
