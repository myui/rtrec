import logging

from math import inf
from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple, Iterable, Self
from scipy.sparse import csc_matrix

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

    def add_interactions(
            self,
            interactions: Iterable[Tuple[Any, Any, float, float]],
            update_interaction: bool = False,
            record_interactions: bool = False
    ) -> None:
        """
        Add user-item interactions to the model.
        :param user_interactions: List of user-item interactions
        :param update_interaction: Whether to update existing interactions
        :param record_interactions: Whether to record user-item interactions
        :return: Tuple of user and item indices
        """
        for user, item, tstamp, rating in interactions:
            try:
                user_id = self.user_ids.identify(user)
                item_id = self.item_ids.identify(item)
                self.interactions.add_interaction(user_id, item_id, tstamp, rating, upsert=update_interaction)
                if record_interactions:
                    self._record_interactions(user_id, item_id, tstamp, rating)
            except Exception as e:
                logging.warning(f"Error processing interaction: {e}")
                continue

    @abstractmethod
    def _record_interactions(self, user_id: int, item_id: int, tstamp: float, rating: float) -> None:
        """
        Record user-item interactions.
        :param user_id: User index
        :param item_id: Item index
        :param tstamp: Interaction timestamp
        :param rating: Interaction rating
        """
        raise NotImplementedError("_record_interactions method must be implemented in the derived class")

    def fit(self, interactions: Iterable[Tuple[Any, Any, float, float]], update_interaction: bool=False) -> Self:
        """
        Fit the recommender model on the given user-item interactions.
        :param user_interactions: List of user-item interactions
        :param update_interaction: Whether to update existing interactions
        """
        self.add_interactions(interactions, update_interaction=update_interaction, record_interactions=True)
        return self._fit_recorded()

    @abstractmethod
    def _fit_recorded(self) -> Self:
        """
        Fit the recommender model on the recorded user-item interactions.
        """
        raise NotImplementedError("_fit_recorded method must be implemented in the derived class")

    @abstractmethod
    def bulk_fit(self, progress_bar: bool=True, parallel: bool=True) -> Self:
        """
        Fit the recommender model on the given interaction matrix.
        :param progress_bar: Whether to display a progress bar. Defaults to True.
        :param parallel: Whether to run the fitting process in parallel. Defaults to True.
        """
        raise NotImplementedError("bulk_fit method must be implemented in the derived class")

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
        # candidate_item_ids = self.interactions.get_all_non_interacted_items(user_id) if filter_interacted else self.interactions.get_all_non_negative_items(user_id)

        # Get top-K recommendations
        recommended_item_ids = self._recommend(user_id, top_k=top_k, filter_interacted=filter_interacted)

        # Resolve item indices to original item values
        return [self.item_ids.get(item_id) for item_id in recommended_item_ids]

    @abstractmethod
    def _recommend(self, user_id: int, top_k: int = 10, filter_interacted: bool = True) -> List[int]:
        """
        Recommend top-K items for a given user.
        :param user_id: User index
        :param top_k: Number of top items to recommend
        :param filter_interacted: Whether to filter out items the user has already interacted with
        :return: List of top-K item indices recommended for the user
        """
        raise NotImplementedError("_recommend method must be implemented in the derived class")

    def recommend_batch(self, users: List[Any], top_k: int = 10, filter_interacted: bool = True) -> List[List[Any]]:
        """
        Recommend top-K items for a list of users.
        :param users: List of user indices
        :param top_k: Number of top items to recommend
        :param filter_interacted: Whether to filter out items the user has already interacted with
        :return: List of top-K item indices recommended for each user
        """
        user_ids = [self.user_ids.get_id(user) for user in users]
        interaction_matrix = self.interactions.to_csr(select_users=user_ids)

        results = []
        for user_id in user_ids:
            if user_id is None:
                results.append([]) # TODO: return popoular items?
                continue
            recommended_item_ids = self._recommend(user_id, interaction_matrix, top_k=top_k, filter_interacted=filter_interacted)
            results.append([self.item_ids.get(item_id) for item_id in recommended_item_ids])
        return results

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
