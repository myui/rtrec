import logging
import pickle
from abc import ABC, abstractmethod
from io import BytesIO
from typing import Any, Iterable, List, Optional, Tuple, Union, TYPE_CHECKING

if TYPE_CHECKING:  # static analyzers
    from typing import Self
else:  # runtime on 3.10
    try:
        from typing import Self  # 3.11+
    except ImportError:
        from typing_extensions import Self

from rtrec.utils.features import FeatureStore
from rtrec.utils.identifiers import Identifier
from rtrec.utils.interactions import UserItemInteractions

FileLike = Union[BytesIO, Any]

class BaseModel(ABC):
    def __init__(self, **kwargs: Any):
        """
        Initialize the recommender model.
        :param kwargs: Additional keyword arguments for the model
        """
        # Initialize user-item interactions
        self.interactions = UserItemInteractions(**kwargs)

        # Initialize user and item ID mappings
        self.user_ids = Identifier(**kwargs)
        self.item_ids = Identifier(**kwargs)

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
        if user_id is not None:
            if self.user_ids.pass_through and user_id > self.interactions.max_user_id:
                # If user_id is greater than max_user_id, treat it as a cold-start user
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
        # Track indices of cold-start users
        cold_user_ids: List[Optional[int]] = []
        cold_indices = []
        # Track indices of existing users
        hot_user_ids: List[int] = []
        hot_indices = []

        for i, user in enumerate(users):
            uid = self.user_ids.get_id(user)
            if uid is None or (self.user_ids.pass_through and uid > self.interactions.max_user_id):
                cold_user_ids.append(self.handle_unknown_user(user))
                cold_indices.append(i)
            else:
                hot_user_ids.append(uid)
                hot_indices.append(i)

        candidate_item_ids = None
        if candidate_items is not None:
            candidate_item_ids = [
                item_id for item in candidate_items
                if (item_id := self.item_ids.get_id(item)) is not None
                and (not self.item_ids.pass_through or item_id <= self.interactions.max_item_id)
            ]
            if len(candidate_item_ids) == 0:
                candidate_item_ids = None

        if len(cold_user_ids) == 0:
            # If there are no cold-start users, proceed with batch recommendation
            batch_results = self._recommend_hot_batch(hot_user_ids, candidate_item_ids=candidate_item_ids, users_tags=users_tags, top_k=top_k, filter_interacted=filter_interacted)
            return [[self.item_ids.get(item_id) for item_id in internal_ids] for internal_ids in batch_results]

        # Initialize results list
        results: list[list[int]] = [[] for _ in range(len(users))]

        # Handle cold-start users
        if cold_indices:
            cold_user_tags = None
            if users_tags:
                cold_user_tags = [users_tags[i] for i in cold_indices]

            # Get recommendations for cold-start user
            cold_results = self._recommend_cold_batch(
                cold_user_ids,
                candidate_item_ids=candidate_item_ids,
                users_tags=cold_user_tags,
                top_k=top_k
            )

            # Map results back to original positions and convert item ids to original items
            for i, orig_idx in enumerate(cold_indices):
                results[orig_idx] = [self.item_ids.get(item_id) for item_id in cold_results[i]]

        # Handle existing users
        if hot_indices:
            hot_users_tags = None
            if users_tags:
                hot_users_tags = [users_tags[i] for i in hot_indices]

            # Get recommendations for hot users
            hot_results = self._recommend_hot_batch(
                hot_user_ids,
                candidate_item_ids=candidate_item_ids,
                users_tags=hot_users_tags,
                top_k=top_k,
                filter_interacted=filter_interacted
            )

            # Map results back to original positions and convert item ids to original items
            for i, orig_idx in enumerate(hot_indices):
                results[orig_idx] = [self.item_ids.get(item_id) for item_id in hot_results[i]]

        return results # type: ignore

    def handle_unknown_user(self, user: Any) -> Optional[int]:
        """
        Handle the case when a user is not found in the model.
        :param user: User to handle
        :return: User index
        """
        return None

    def _recommend_cold_batch(self, user_ids: List[Optional[int]], candidate_item_ids: Optional[List[int]] = None, users_tags: Optional[List[List[str]]] = None, top_k: int = 10) -> List[List[int]]:
        """
        Get recommendations for cold-start users (users not in the model).
        By default, it returns the most popular items.

        Args:
        :param user_ids: List of user indices
        :param candidate_item_ids: List of candidate item indices to recommend from
        :param users_tags: List of user tags
        :param top_k: Number of top items to recommend
        :return: List of top-K item indices recommended for each user
        """
        hot_items = self.interactions.get_hot_items(top_k, filter_interacted=False)
        if candidate_item_ids is not None:
            # take intersection between hot items and candidate items
            hot_items = [item_id for item_id in hot_items if item_id in candidate_item_ids]
        return [hot_items for _ in user_ids]

    def _recommend_hot_batch(self, user_ids: List[int], candidate_item_ids: Optional[List[int]] = None, users_tags: Optional[List[List[str]]] = None, top_k: int = 10, filter_interacted: bool = True) -> List[List[int]]:
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
        if users_tags: # If user tags are provided
            assert len(user_ids) == len(users_tags), f"Number of user tags must match the number of users. Got {len(user_ids)} users and {len(users_tags)} user tags."
            for user_id, user_tags in zip(user_ids, users_tags):
                recommended_item_ids = self._recommend(user_id, candidate_item_ids=candidate_item_ids, user_tags=user_tags, top_k=top_k, filter_interacted=filter_interacted)
                results.append(recommended_item_ids)
        else:
            for user_id in user_ids:
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
    def _similar_items(self, query_item_id: int, query_item_tags: Optional[List[str]] = None, top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Find similar items for a list of query items.
        :param query_item_id: item id to find similar items for
        :param query_item_tags: List of query item tags
        :param top_k: Number of top similar items to return for each query item
        :param filter_query_items: Whether to filter out items in the query_items list
        :return: List of top-K similar items for each query item with similarity scores
        """
        raise NotImplementedError("_similar_items method must be implemented in the derived class")

    def save(self, f: FileLike) -> int:
        """
        Save the model to a file-like object.
        :param f: File-like object to save the model to
        :return: Number of bytes written
        """
        data = self._serialize()
        serialized_data = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
        return f.write(serialized_data)

    @classmethod
    def load(cls, f: FileLike) -> Self:
        """
        Load the model from a file-like object.
        :param f: File-like object to load the model from
        :return: Loaded model instance
        """
        serialized_data = f.read()
        data = pickle.loads(serialized_data)
        return cls._deserialize(data)

    @classmethod
    def loads(cls, data: bytes) -> Self:
        """
        Load the model from serialized bytes.
        :param data: Serialized model data
        :return: Loaded model instance
        """
        buffer = BytesIO(data)
        return cls.load(buffer)

    @abstractmethod
    def _serialize(self) -> dict:
        """
        Serialize the model state to a dictionary.
        :return: Dictionary containing model state
        """
        raise NotImplementedError("_serialize method must be implemented in the derived class")

    @classmethod
    @abstractmethod
    def _deserialize(cls, data: dict) -> Self:
        """
        Deserialize the model state from a dictionary.
        :param data: Dictionary containing model state
        :return: Model instance
        """
        raise NotImplementedError("_deserialize method must be implemented in the derived class")
