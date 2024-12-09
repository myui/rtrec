from collections import defaultdict
import numpy as np
from typing import Dict, Any, Iterable, Tuple, List

from .base import ExplicitFeedbackRecommender, inv_scaling

class FactorizationMachines(ExplicitFeedbackRecommender):
    def __init__(self, n_factors: int, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        # Initialize parameters
        self.alpha = kwargs.get('alpha', 0.01)  # Learning rate
        self.power_t = kwargs.get('power_t', 0.1)  # Power for inv-scaling learning rate
        self.lambda2 = kwargs.get('lambda2', 0.0001)  # L2 regularization for factors

        self.n_factors: int = n_factors  # Number of latent factors
        self.feature_map: Dict[str, int] = {}  # Maps feature keys to indices
        self.w: List[float] = [0.0]  # Linear weights with w[0] as global bias
        self.V: List[np.ndarray] = []  # Factor matrix (list of arrays)

        self.cumulative_loss = 0.0
        self.steps = 0

    def get_empirical_error(self) -> float:
        if self.steps == 0:
            return 0.0
        return self.cumulative_loss / self.steps

    def _get_or_create_index(self, key: str) -> int:
        """Get the index for a feature key, creating a new one if it doesn't exist."""
        index = self.feature_map.get(key, None)

        if index is None:
            index = len(self.feature_map)  # Start from 0 for linear weights
            self.feature_map[key] = index

            # Ensure the list is long enough to accommodate this new index
            self.w.append(0.0)  # Initialize linear weights as 0.0

            # Initialize the factor vector for the new feature
            self.V.append(np.random.normal(0, 0.1, self.n_factors))  # Random factor initialization 

        return index

    def _predict_rating(self, user_id: int, item_id: int, bypass_prediction: bool=False) -> float:
        user_idx = self._get_or_create_index(f'u{user_id}')
        item_idx = self._get_or_create_index(f'i{item_id}')

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
                Vif = Vi[f]
                sum_vx += Vif
                sum_vx_sq += Vif * Vif
            interaction_term += 0.5 * (sum_vx ** 2 - sum_vx_sq)

        return linear_term + interaction_term

    def _update(self, user_id: int, item_id: int) -> None:
        """Perform a single update for the given user-item pair."""
        # Update linear terms for non-zero features
        y = self._get_rating(user_id, item_id)  # True rating
        y_pred = self._predict_rating(user_id, item_id)  # Predicted rating
        dloss = y_pred - y  # Prediction error

        self.steps += 1
        self.cumulative_loss += abs(dloss)

        if abs(dloss) <= 1e-6:
            return

        grad = dloss  # Gradient is the error for this simple regression task as feature value is 1.0

        user_idx = self._get_or_create_index(f'u{user_id}')
        item_idx = self._get_or_create_index(f'i{item_id}')

        adjusted_learning_rate = inv_scaling(self.alpha, self.steps, self.power_t)
        self.w[0] -= adjusted_learning_rate * grad  # Update global bias
        self.w[user_idx + 1] -= adjusted_learning_rate * (grad + self.lambda2 * self.w[user_idx + 1])  # Update user bias
        self.w[item_idx + 1] -= adjusted_learning_rate * (grad + self.lambda2 * self.w[item_idx + 1])  # Update item bias

        # Update interaction factors (latent factors for user-item pair)
        for f in range(self.n_factors):
            sum_vx = self.V[user_idx][f] + self.V[item_idx][f]
            for idx in [user_idx, item_idx]:
                v_if = self.V[idx][f]
                gradient = dloss * (sum_vx - self.V[idx][f])
                if abs(gradient) <= 1e-6:
                    continue
                self.V[idx][f] -= adjusted_learning_rate * (gradient + self.lambda2 * v_if)

    def _get_similarity(self, target_item_id: int, base_item_id: int) -> float:
        """Compute the cosine similarity between two items."""
        target_item_idx = self._get_or_create_index(f'i{target_item_id}')
        base_item_idx = self._get_or_create_index(f'i{base_item_id}')

        target_item_factors = self.V[target_item_idx]
        base_item_factors = self.V[base_item_idx]

        dot_product = np.dot(target_item_factors, base_item_factors)
        target_norm = np.linalg.norm(target_item_factors)
        base_norm = np.linalg.norm(base_item_factors)

        # Avoid division by zero
        return dot_product / (target_norm * base_norm + 1e-6) # cosine similarity

@DeprecationWarning
class AdaGrad:

    def __init__(self, alpha: float = 0.01, lambda1 = 0.0002, lambda2 = 0.0001, epsilon: float = 1e-6, **kwargs: Any) -> None:
        """
        AdaGrad Constructor
        :param alpha: Learning rate
        :param epsilon: Small constant to avoid division by zero
        """
        self.alpha = alpha
        self.epsilon = epsilon
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.G = defaultdict(lambda: defaultdict(float))

    def update(self, feature_idx: int, factor_idx: int, grad: float, V: List[np.ndarray]) -> None:
        """
        Update gradient accumulation for AdaGrad.
        :param key: Tuple of two integers representing the indices (e.g., (i, j)).
        :param grad: Gradient to accumulate.
        :return: Updated weight.
        """
        # Update the sum of squared gradients
        G_val = self.G[feature_idx][factor_idx]
        G_new = G_val + np.clip(grad ** 2, 1e-8, 1e8)
        self.G[feature_idx][factor_idx] = G_new

        # Update the weight
        current_v = V[feature_idx][factor_idx]
        adaptive_lr = self.alpha / (np.sqrt(G_new) + self.epsilon)
        l1_penalty = np.sign(current_v) * self.lambda1
        l2_penalty = current_v * self.lambda2
        V[feature_idx][factor_idx] -= adaptive_lr * (grad + l1_penalty + l2_penalty)

        # Ensure weight_update is finite
        if not np.isfinite(V[feature_idx][factor_idx]):
            raise ValueError(f"Weight update is not finite: {V[feature_idx][factor_idx]}")
