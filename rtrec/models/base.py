import numpy as np
from abc import ABC, abstractmethod
from typing import Any, List, Tuple

from rtrec.utils.identifiers import Identifier
from rtrec.utils.interactions import UserItemInteractions

class BaseRecommender(ABC):
    def __init__(self, **kwargs: Any):
        """
        Initialize the recommender model.
        :param kwargs: Additional keyword arguments for the model
        """
        # Initialize user-item interactions
        self.interactions = UserItemInteractions()

        # Initialize user and item ID mappings
        self.user_ids = Identifier()
        self.item_ids = Identifier()

    def _get_interacted_items(self, user_id: int) -> List[int]:
        """
        Get a list of all items a user has interacted with.
        """
        return self.interactions.get_all_items_for_user(user_id)
    
    def _get_rating(self, user_id: int, item_id: int) -> float:
        """
        Get the rating for a specific user-item pair.
        """
        return self.interactions.get_user_item_rating(user_id, item_id, default_rating=0.0)

    def recommend(self, user: Any, top_k: int = 10, filter_interacted: bool = True) -> List[Any]:
        """
        Recommend top-K items for a given user.
        :param user: User index
        :param top_k: Number of top items to recommend
        :param filter_interacted: Whether to filter out items the user has already interacted with
        :return: List of top-K item indices recommended for the user
        """

        user_id = self.user_ids.get_id(user)
        if user_id is None:
            return [] # TODO: return popoular items?

        if filter_interacted:
            # Get all items the user has not interacted with
            candidate_item_ids = self.interactions.get_all_non_interacted_items(user_id)
        else:
            # Get all non-negative items (non-interacted or positively interacted)
            candidate_item_ids = self.interactions.get_all_non_negative_items(user_id)

        # Predict scores for candidate items
        scores = self._predict(candidate_item_ids)

        # Map item IDs back to original items
        candidate_items = [self.item_ids[id] for id in candidate_item_ids]

        # Return top-K items with the highest scores
        return sorted(candidate_items, key=dict(zip(candidate_items, scores)).get, reverse=True)[:top_k]

    @abstractmethod
    def _predict(self, item_ids: List[int]) -> List[float]:
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

    def fit(self, user_interactions: List[Tuple[int, float, Any, Any]]) -> None:
        """
        Incrementally fit the BPRSLIM model with user interactions.
        :param user_interactions: List of (user, positive_item, negative_item) tuples
        """
        for user, tstamp, positive_item, negative_item in user_interactions:
            # Update user-item interactions
            user_id = self.user_ids.identify(user)
            positive_item_id = self.item_ids.identify(positive_item)
            negative_item_id = self.item_ids.identify(negative_item)
            self.interactions.add_interaction(user_id, positive_item_id, tstamp, delta=1)
            self.interactions.add_interaction(user_id, negative_item_id, tstamp, delta=-1)
            self._update(user_id, positive_item_id, negative_item_id)

    @abstractmethod
    def _update(self, user: int, positive_item_id: int, negative_item_id: int) -> None:
        """
        Incremental weight update based on BPR loss.
        :param user: User index
        :param positive_item_id: Item index for the positive sample
        :param negative_item_id: Item index for the negative sample
        """
        raise NotImplementedError("The _update method must be implemented by the subclass.")
    

class ExplictFeedbackRecommender(BaseRecommender):

    def __init__(self, **kwargs: Any):
        """
        Initialize the implicit feedback recommender model.
        :param kwargs: Additional keyword arguments for the model
        """
        super().__init__(**kwargs)

    def fit(self, user_interactions: List[Tuple[Any, Any, float, float]]) -> None:
        for user, item, tstamp, rating in user_interactions:
            user_id = self.user_ids.identify(user)
            item_id = self.item_ids.identify(item)
            self.interactions.add_interaction(user_id, item_id, tstamp, rating)
            self._update(user_id, item_id)

    @abstractmethod
    def _update(self, user_id: int, item_id: int, rating: float) -> None:
        raise NotImplementedError("The _update method must be implemented by the subclass.")
