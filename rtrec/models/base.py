import logging

from math import inf
from abc import ABC, abstractmethod
from typing import Any, List, Tuple, Iterable

from rtrec.utils.identifiers import Identifier
from rtrec.utils.interactions import UserItemInteractions

class BaseRecommender(ABC):
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

    @abstractmethod
    def get_empirical_error(self) -> float:
        """
        Get the empirical error of the model.
        The empirical error is the average loss over all user-item interactions.
        see https://en.wikipedia.org/wiki/Generalization_error
        """
        raise NotImplementedError("The get_empirical_loss method must be implemented by the subclass.")

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

        # Get candidate items for recommendation
        candidate_item_ids = self.interactions.get_all_non_interacted_items(user_id) if filter_interacted else self.interactions.get_all_non_negative_items(user_id)

        # Predict scores for candidate items
        scores = self._predict(user_id, candidate_item_ids)

        # Map item IDs back to original items
        candidate_items = [self.item_ids[id] for id in candidate_item_ids]

        # Return top-K items with the highest scores
        assert len(candidate_items) == len(scores), "Number of items and scores must match"
        # return sorted(candidate_items, key=dict(zip(candidate_items, scores)).get, reverse=True)[:top_k]
        return [k for k, v in sorted(zip(candidate_items, scores), key=lambda x: x[1], reverse=True)[:top_k]]

    def similar_items(self, query_items: List[Any], top_k: int = 10, filter_query_items: bool = True) -> List[List[Any]]:
        """
        Find similar items for a list of query items.
        :param query_items: List of query items
        :param top_k: Number of top similar items to return
        :param filter_interacted: Whether to filter out items in the query_items list
        :return: List of top-K similar items for each query item
        """
        query_item_ids = [self.item_ids.get_id(item) for item in query_items]
        target_item_ids = self.interactions.get_all_item_ids()

        return self._similar_items(target_item_ids, query_item_ids, top_k=top_k, filter_query_items=filter_query_items)

    def _similar_items(self, target_item_ids: List[int], query_item_ids: List[int], top_k: int = 10, filter_query_items: bool = True) -> List[List[int]]:
        """
        Find similar items for a list of query item indices.
        :param target_item_ids: List of target item indices
        :param query_item_ids: List of query item indices
        :param top_k: Number of top similar items to return
        :param filter_query_items: Whether to filter out items in the query_items list
        :return: List of top-K similar items for each query item
        """
        similar_items = []
        neg_inf = -inf
        for query_item_id in query_item_ids:
            if query_item_id is None:
                similar_items.append([])
            else:
                similarity_scores = [
                    (self.item_ids[target_item_id], self._get_similarity(target_item_id, query_item_id))
                    for target_item_id in target_item_ids
                    if filter_query_items is False or target_item_id != query_item_id
                ]
                similar_items.append(
                    [k for k, v in sorted(similarity_scores, key=lambda x: x[1], reverse=True)[:top_k]]
                )

        return similar_items

    @abstractmethod
    def _get_similarity(self, target_item_id: int, base_item_id: int) -> float:
        """
        Get the similarity between two items.
        :param target_item_id: Target item index
        :param base_item_id: Item index
        :return: Similarity between the two items
        """
        raise NotImplementedError(f"The _get_similarity method must be implemented by the subclass.")

    @abstractmethod
    def _predict(self, user_id: int, item_ids: List[int]) -> List[float]:
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

    def fit(self, user_interactions: Iterable[Tuple[int, float, Any, Any]], update_interaction: bool=False) -> None:
        """
        Incrementally fit the BPRSLIM model with user interactions.
        :param user_interactions: List of (user, positive_item, negative_item) tuples
        :param epoch: Current training epoch number (default is 0)
        """
        for user, tstamp, positive_item, negative_item in user_interactions:
            # Update user-item interactions
            try:
                user_id = self.user_ids.identify(user)
                positive_item_id = self.item_ids.identify(positive_item)
                negative_item_id = self.item_ids.identify(negative_item)
                self.interactions.add_interaction(user_id, positive_item_id, tstamp, delta=1, upsert=update_interaction)
                self.interactions.add_interaction(user_id, negative_item_id, tstamp, delta=-1, upsert=update_interaction)
                self._update(user_id, positive_item_id, negative_item_id)
            except Exception as e:
                logging.warning(f"Error processing interaction: {e}")
                continue


    @abstractmethod
    def _update(self, user: int, positive_item_id: int, negative_item_id: int) -> None:
        """
        Incremental weight update based on BPR loss.
        :param user: User index
        :param positive_item_id: Item index for the positive sample
        :param negative_item_id: Item index for the negative sample
        """
        raise NotImplementedError("The _update method must be implemented by the subclass.")
    

class ExplicitFeedbackRecommender(BaseRecommender):

    def __init__(self, **kwargs: Any):
        """
        Initialize the implicit feedback recommender model.
        :param kwargs: Additional keyword arguments for the model
        """
        super().__init__(**kwargs)

    def fit(self, user_interactions: Iterable[Tuple[Any, Any, float, float]], update_interaction: bool=False) -> None:
        for user, item, tstamp, rating in user_interactions:
            try:
                user_id = self.user_ids.identify(user)
                item_id = self.item_ids.identify(item)
                self.interactions.add_interaction(user_id, item_id, tstamp, rating, upsert=update_interaction)
                self._update(user_id, item_id)
            except Exception as e:
                logging.warning(f"Error processing interaction: {e}")
                continue

    @abstractmethod
    def _update(self, user_id: int, item_id: int, rating: float) -> None:
        raise NotImplementedError("The _update method must be implemented by the subclass.")

    def _predict(self, user_id: int, item_ids: List[int]) -> List[float]:
        """
        Predict scores for a list of items.
        """
        return [
            self._predict_rating(user_id, item_id, bypass_prediction=False)
            for item_id in item_ids
        ]

    def predict_rating(self, user: Any, item: Any) -> float:
        user_id = self.user_ids.get_id(user)
        item_id = self.item_ids.get_id(item)
        if user_id is None or item_id is None:
            return 0.0
        return self._predict_rating(user_id, item_id, bypass_prediction=True)

    @abstractmethod
    def _predict_rating(self, user_id: int, item_id: int, bypass_prediction: bool=False) -> float:
        """
        Compute the derivative of the loss function.
        :param user_id: User index
        :param item_id: Item index
        :param bypass_prediction: Flag to bypass prediction if user has only interacted with the item (default: False)
        """
        raise NotImplementedError("The _predict_rating method must be implemented by the subclass.")
