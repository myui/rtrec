import numpy as np
from abc import ABC, abstractmethod
from typing import Any, List, Tuple

from rtrec.utils.datasets import UserItemInteractions

class BaseRecommender(ABC):
    def __init__(self, **kwargs: Any):
        """
        Initialize the recommender model.
        :param kwargs: Additional keyword arguments for the model
        """
        # Initialize user-item interactions
        self.interactions = UserItemInteractions()

    def get_interacted_items(self, user_id: int) -> List[int]:
        """
        Get a list of all items a user has interacted with.
        """
        return self.interactions.get_all_items_for_user(user_id)
    
    def get_rating(self, user_id: int, item_id: int) -> float:
        """
        Get the rating for a specific user-item pair.
        """
        return self.interactions.get_user_item_count(user_id, item_id, default_count=0.0)

    def recommend(self, user: int, top_k: int = 10, filter_interacted: bool = True) -> List[int]:
        """
        Recommend top-K items for a given user.
        :param user: User index
        :param top_k: Number of top items to recommend
        :param filter_interacted: Whether to filter out items the user has already interacted with
        :return: List of top-K item indices recommended for the user
        """

        if filter_interacted:
            # Get all items the user has not interacted with
            candidate_items = self.interactions.get_all_non_interacted_items(user)
        else:
            # Get all non-negative items (non-interacted or positively interacted)
            candidate_items = self.interactions.get_all_non_negative_items(user)

        # Predict scores for candidate items
        scores = self._predict(candidate_items)

        # Return top-K items with the highest scores
        return sorted(candidate_items, key=lambda i: scores[i], reverse=True)[:top_k]

    @abstractmethod
    def _predict(self, items: List[int]) -> List[float]:
        """
        Predict scores for a list of items.
        """
        raise NotImplementedError("The _predict method must be implemented by the subclass.")

class ImplicitFeedbackRecommender(BaseRecommender):

    def __init__(self, **kwargs: Any):
        """
        Initialize the implicit feedback recommender model.
        :param kwargs: Additional keyword arguments for the model
        """
        super().__init__(**kwargs)

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

    @abstractmethod
    def _update(self, user: int, positive_item: int, negative_item: int) -> None:
        """
        Incremental weight update based on BPR loss.
        :param user: User index
        :param positive_item: Item index for the positive sample
        :param negative_item: Item index for the negative sample
        """
        raise NotImplementedError("The _update method must be implemented by the subclass.")
    

class ExplictFeedbackRecommender(BaseRecommender):

    def __init__(self, **kwargs: Any):
        """
        Initialize the implicit feedback recommender model.
        :param kwargs: Additional keyword arguments for the model
        """
        super().__init__(**kwargs)

    def fit(self, user_interactions: List[Tuple[int, int, float]]) -> None:
        for user, item_id, rating in user_interactions:
            self.interactions.add_interaction(user, item_id, rating)
            self._update(user, item_id, rating)

    @abstractmethod
    def _update(self, user: int, item_id: int, rating: float) -> None:
        raise NotImplementedError("The _update method must be implemented by the subclass.")