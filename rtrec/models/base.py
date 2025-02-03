import logging

from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple, Iterable, Self

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

    def register_user_feature(self, user: Any, user_tags: List[str]) -> int:
        """
        Register user features in the feature store.
        :param user: User to register features for
        :param user_tags: List of user features
        :return: User index
        """
        user_id = self.user_ids.identify(user)
        self.feature_store.put_user_features(user_id, user_tags)
        return user_id

    def clear_user_features(self, user_ids: Optional[List[int]] = None) -> None:
        """
        Clear user features.
        :param user_ids: List of user indices to clear features for. If None, clear all user features.
        """
        self.feature_store.clear_user_features(user_ids)

    def register_item_feature(self, item: Any, item_tags: List[str]) -> int:
        """
        Register item features in the feature store.
        :param item_id: Item to register features for
        :param item_tags: List of item features
        :return: Item index
        """
        item_id = self.item_ids.identify(item)
        self.feature_store.put_item_features(item_id, item_tags)
        return item_id

    def clear_item_features(self, item_ids: Optional[List[int]] = None) -> None:
        """
        Clear item features.
        :param item_ids: List of item indices to clear features for. If None, clear all item features.
        """
        self.feature_store.clear_item_features(item_ids)

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

    def fit(self, interactions: Iterable[Tuple[Any, Any, float, float]], update_interaction: bool=False, progress_bar: bool=True) -> Self:
        """
        Fit the recommender model on the given user-item interactions.
        :param user_interactions: List of user-item interactions
        :param update_interaction: Whether to update existing interactions
        :param progress_bar: Whether to display a progress bar
        """
        self.add_interactions(interactions, update_interaction=update_interaction, record_interactions=True)
        return self._fit_recorded(progress_bar=progress_bar)

    @abstractmethod
    def _fit_recorded(self, parallel: bool=False, progress_bar: bool=True) -> Self:
        """
        Fit the recommender model on the recorded user-item interactions.
        :param parallel: Whether to run the fitting process in parallel. Defaults to False.
        :param progress_bar: Whether to display a progress bar. Defaults to True.
        """
        raise NotImplementedError("_fit_recorded method must be implemented in the derived class")

    @abstractmethod
    def bulk_fit(self, parallel: bool=True, progress_bar: bool=True) -> Self:
        """
        Fit the recommender model on the given interaction matrix.
        :param parallel: Whether to run the fitting process in parallel. Defaults to True.
        :param progress_bar: Whether to display a progress bar. Defaults to True.
        """
        raise NotImplementedError("bulk_fit method must be implemented in the derived class")

    def recommend(self, user: Any, candidate_items: Optional[List[Any]] = None, user_tags: Optional[List[str]] = None, top_k: int = 10, filter_interacted: bool = True) -> List[Any]:
        """
        Recommend top-K items for a given user.
        :param user: User to recommend items for
        :param candidate_items: List of candidate items to recommend from
        :param user_tags: List of user tags
        :param top_k: Number of top items to recommend
        :param filter_interacted: Whether to filter out items the user has already interacted with
        :return: List of top-K items recommended for the user
        """

        candidate_item_ids = None
        if candidate_items is not None:
            candidate_item_ids = [
                item_id for item in candidate_items
                if (item_id := self.item_ids.get_id(item)) is not None
                and (not self.item_ids.pass_through or item_id <= self.interactions.max_item_id)
            ]
            if len(candidate_item_ids) == 0:
                candidate_item_ids = None

        user_id = self.user_ids.get_id(user)
        if self.user_ids.pass_through and user_id > self.interactions.max_user_id:
            user_id = None
        if user_id is None:
            hot_item_ids = self.interactions.get_hot_items(top_k, filter_interacted=False)
            if candidate_item_ids is not None:
                # take intersection between hot items and candidate items
                hot_item_ids = [item_id for item_id in hot_item_ids if item_id in candidate_item_ids]
            return hot_item_ids

        # Get top-K recommendations
        recommended_item_ids = self._recommend(user_id, candidate_item_ids=candidate_item_ids, user_tags=user_tags, top_k=top_k, filter_interacted=filter_interacted)

        # Resolve item indices to original item values
        return [self.item_ids.get(item_id) for item_id in recommended_item_ids]

    @abstractmethod
    def _recommend(self, user_id: int, candidate_item_ids: Optional[List[int]] = None, user_tags: Optional[List[str]] = None, top_k: int = 10, filter_interacted: bool = True) -> List[int]:
        """
        Recommend top-K items for a given user.
        :param user_id: User index
        :param candidate_item_ids: List of candidate item indices to recommend from
        :param user_tags: List of user tags
        :param top_k: Number of top items to recommend
        :param filter_interacted: Whether to filter out items the user has already interacted with
        :return: List of top-K item indices recommended for the user
        """
        raise NotImplementedError("_recommend method must be implemented in the derived class")

    def recommend_batch(self, users: List[Any], candidate_items: Optional[List[Any]] = None, users_tags: Optional[List[List[str]]] = None, top_k: int = 10, filter_interacted: bool = True) -> List[List[Any]]:
        """
        Recommend top-K items for a list of users.
        :param users: List of users to recommend items for
        :param candidate_items: List of candidate items to recommend from
        :param users_tags: List of user tags
        :param top_k: Number of top items to recommend
        :param filter_interacted: Whether to filter out items the user has already interacted with
        :return: List of top-K items recommended for each user
        """
        user_ids = []
        for user in users:
            uid = self.user_ids.get_id(user)
            if self.user_ids.pass_through and uid > self.interactions.max_user_id:
                user_ids.append(None)
            else:
                user_ids.append(uid)
        candidate_item_ids = None
        if candidate_items is not None:
            candidate_item_ids = [
                item_id for item in candidate_items
                if (item_id := self.item_ids.get_id(item)) is not None
                and (not self.item_ids.pass_through or item_id <= self.interactions.max_item_id)
            ]
            if len(candidate_item_ids) == 0:
                candidate_item_ids = None

        results = self._recommend_batch(user_ids, candidate_item_ids=candidate_item_ids, users_tags=users_tags, top_k=top_k, filter_interacted=filter_interacted)
        return [[self.item_ids.get(item_id) for item_id in internal_ids] for internal_ids in results]

    def _recommend_batch(self, user_ids: List[int], candidate_item_ids: Optional[List[int]] = None, users_tags: Optional[List[List[str]]] = None, top_k: int = 10, filter_interacted: bool = True) -> List[List[int]]:
        """
        Recommend top-K items for a list of users.
        :param user_ids: List of user indices
        :param candidate_item_ids: List of candidate item indices to recommend from
        :param users_tags: List of user tags
        :param interaction_matrix: User-item interaction matrix
        :param top_k: Number of top items to recommend
        :param filter_interacted: Whether to filter out items the user has already interacted with
        :return: List of top-K item indices recommended for each user
        """
        results = []
        hot_items = None
        if users_tags: # If user tags are provided
            assert len(user_ids) == len(users_tags), f"Number of user tags must match the number of users. Got {len(user_ids)} users and {len(users_tags)} user tags."
            for user_id, user_tags in zip(user_ids, users_tags):
                if user_id is None:
                    # Return popular items if user is not found
                    if hot_items is None:
                        hot_items = self.interactions.get_hot_items(top_k, filter_interacted=False)
                        if candidate_item_ids is not None:
                            # take intersection between hot items and candidate items
                            hot_items = [item_id for item_id in hot_items if item_id in candidate_item_ids]
                    results.append(hot_items)
                else:
                    recommended_item_ids = self._recommend(user_id, candidate_item_ids=candidate_item_ids, user_tags=user_tags, top_k=top_k, filter_interacted=filter_interacted)
                    results.append(recommended_item_ids)
        else:
            for user_id in user_ids:
                if user_id is None:
                    # Return popular items if user is not found
                    if hot_items is None:
                        hot_items = self.interactions.get_hot_items(top_k, filter_interacted=False)
                        if candidate_item_ids is not None:
                            # take intersection between hot items and candidate items
                            hot_items = [item_id for item_id in hot_items if item_id in candidate_item_ids]
                    results.append(hot_items)
                else:
                    recommended_item_ids = self._recommend(user_id, candidate_item_ids=candidate_item_ids, top_k=top_k, filter_interacted=filter_interacted)
                    results.append(recommended_item_ids)
        return results

    def similar_items(self, query_item: Any, query_item_tags: Optional[List[str]] = None, top_k: int = 10, ret_scores: bool=False) -> List[Tuple[Any, float]] | List[Any]:
        """
        Find similar items for a list of query items.
        :param query_item: List of query items
        :param query_item_tags: List of query item tags
        :param top_k: Number of top similar items to return for each query item
        :param ret_scores: Whether to return similarity scores. Defaults to False.
        :return: List of top-K similar items for each query item with similarity scores. If ret_scores is False, only return similar items.
        """
        query_item_id = self.item_ids.identify(query_item)
        if query_item_id is None:
            return []

        # Get top-K similar items
        similar_item_ids = self._similar_items(query_item_id, query_item_tags=query_item_tags, top_k=top_k)

        # Resolve item indices to original item values
        if ret_scores:
            return [(self.item_ids.get(item_id), score) for item_id, score in similar_item_ids]
        else:
            return [self.item_ids.get(item_id) for item_id, _ in similar_item_ids]

    @abstractmethod
    def _similar_items(self, query_item_id: int,  query_item_tags: Optional[List[str]] = None, top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Find similar items for a list of query items.
        :param query_item_id: item id to find similar items for
        :param query_item_tags: List of query item tags
        :param top_k: Number of top similar items to return for each query item
        :param filter_query_items: Whether to filter out items in the query_items list
        :return: List of top-K similar items for each query item with similarity scores
        """
        raise NotImplementedError("_similar_items method must be implemented in the derived class")
