import logging

from math import inf
from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple, Iterable

from rtrec.utils.features import FeatureStore
from rtrec.utils.identifiers import Identifier
from rtrec.utils.interactions import UserItemInteractions

class BaseModel(ABC):
    def __init__(self, **kwargs: Any):
        """
        Initialize the recommender model.
        :param kwargs: Additional keyword arguments for the model
        """
        # Initialize user-item interactions
        self.interactions = UserItemInteractions(**kwargs)

        # Initialize user and item ID mappings
        self.user_ids = Identifier()
        self.item_ids = Identifier()

        self.feature_store = FeatureStore()

    def register_user_feature(self, user_id: Any, user_tags: List[str]) -> int:
        """
        Register user features in the feature store.
        :param user_id: User identifier
        :param user_tags: List of user features
        :return: User index
        """
        user_id = self.user_ids.identify(user_id)
        self.feature_store.put_user_feature(user_id, user_tags)
        return user_id

    def register_item_feature(self, item_id: Any, item_tags: List[str]) -> int:
        """
        Register item features in the feature store.
        :param item_id: Item identifier
        :param item_tags: List of item features
        :return: Item index
        """
        item_id = self.item_ids.identify(item_id)
        self.feature_store.put_item_feature(item_id, item_tags)
        return item_id

    def fit(self, user_interactions: Iterable[Tuple[Any, Any, float, float]], update_interaction: bool=False) -> None: 
        user_ids, item_ids = [], []
        for user, item, tstamp, rating in user_interactions:
            try:
                user_id = self.user_ids.identify(user)
                item_id = self.item_ids.identify(item)
                self.interactions.add_interaction(user_id, item_id, tstamp, rating, upsert=update_interaction)
                user_ids.append(user_id)
                item_ids.append(item_id)
            except Exception as e:
                logging.warning(f"Error processing interaction: {e}")
                continue

        return self._fit(user_ids, item_ids)

    def _fit(self, user_ids: List[int], item_ids: List[int]) -> None:
        """
        Fit the recommender model.
        :param user_ids: List of user indices
        :param item_ids: List of item indices
        """
        raise NotImplementedError("_fit method must be implemented in the derived class")

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

        # Get candidate items for recommendation
        candidate_item_ids = self.interactions.get_all_non_interacted_items(user_id) if filter_interacted else self.interactions.get_all_non_negative_items(user_id)

        # Get top-K recommendations
        recommended_item_ids = self._recommend(user_id, candidate_item_ids, top_k=top_k, filter_interacted=filter_interacted)

        # Resolve item indices to original item values
        return [self.item_ids.get(item_id) for item_id in recommended_item_ids]

    @abstractmethod
    def _recommend(self, user_id: int, candidate_item_ids: List[int], top_k: int = 10, filter_interacted: bool = True) -> List[int]:
        """
        Recommend top-K items for a given user.
        :param user_id: User index
        :param candidate_item_ids: List of candidate item indices
        :param top_k: Number of top items to recommend
        :param filter_interacted: Whether to filter out items the user has already interacted with
        :return: List of top-K item indices recommended for the user
        """
        raise NotImplementedError("_recommend method must be implemented in the derived class")

    def similar_items(self, query_item: Any, top_k: int = 10) -> List[Any]:
        """
        Find similar items for a list of query items.
        :param query_item: List of query item indices
        :param top_k: Number of top similar items to return for each query item
        :return: List of top-K similar items for each query item
        """
        query_item_id = self.item_ids.identify(query_item)
        if query_item_id is None:
            return []

        # Get top-K similar items
        similar_item_ids = self._similar_items(query_item_id, top_k=top_k)

        # Resolve item indices to original item values
        return [self.item_ids.get(item_id) for item_id in similar_item_ids]

    @abstractmethod
    def _similar_items(self, query_item_id: int, top_k: int = 10) -> List[int]:
        """
        Find similar items for a list of query items.
        :param query_item_id: item id to find similar items for
        :param top_k: Number of top similar items to return for each query item
        :param filter_query_items: Whether to filter out items in the query_items list
        :return: List of top-K similar items for each query item
        """
        raise NotImplementedError("_similar_items method must be implemented in the derived class")
