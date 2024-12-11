import logging
from typing import Any, Iterable, List, Optional, Tuple, override
from scipy.sparse import csc_matrix

from ..models.internal.slim_elastic import SLIMElastic
from .base import BaseModel

class SLIM(BaseModel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = SLIMElastic(kwargs)

    @override
    def fit(self, user_interactions: Iterable[Tuple[Any, Any, float, float]], update_interaction: bool=False) -> None:
        item_ids = []
        for user, item, tstamp, rating in user_interactions:
            try:
                user_id = self.user_ids.identify(user)
                item_id = self.item_ids.identify(item)
                self.interactions.add_interaction(user_id, item_id, tstamp, rating, upsert=update_interaction)
                item_ids.append(item_id)
            except Exception as e:
                logging.warning(f"Error processing interaction: {e}")
                continue
        interaction_matrix = self.interactions.to_csc(item_ids)
        self.model.partial_fit_items(interaction_matrix, item_ids)

    def _bulk_fit(self, interaction_matrix: csc_matrix) -> None:
        """
        Fit the recommender model on the given interaction matrix.
        :param interaction_matrix: Sparse interaction matrix
        """
        self.model.fit(interaction_matrix, progress_bar=True)

    def _recommend(self, user_id: int, candidate_item_ids: List[int], top_k: int = 10, filter_interacted: bool = True) -> List[int]:
        """
        Recommend top-K items for a given user.
        :param user_id: User index
        :param candidate_item_ids: List of candidate item indices
        :param top_k: Number of top items to recommend
        :param filter_interacted: Whether to filter out items the user has already interacted with
        :return: List of top-K item indices recommended for the user
        """
        interaction_matrix = self.interactions.to_csr(select_users=[user_id])
        return self.model.recommend(user_id, interaction_matrix, candidate_item_ids, top_k=top_k, filter_interacted=filter_interacted)

    def _recommend_batch(self, user_id: int, interaction_matrix: csc_matrix, candidate_item_ids: Optional[List[int]]=None, top_k: int = 10, filter_interacted: bool = True) -> List[int]:
        """
        Recommend top-K items for a list of users.
        :param user_id: User index
        :param interaction_matrix: Sparse user-item interaction matrix
        :param candidate_item_ids: List of candidate item indices
        :param top_k: Number of top items to recommend
        :param filter_interacted: Whether to filter out items the user has already interacted with
        :return: List of top-K item indices recommended for each user
        """
        return self.model.recommend(user_id, interaction_matrix, candidate_item_ids, top_k=top_k, filter_interacted=filter_interacted)

    def _similar_items(self, query_item_id: int, top_k: int = 10) -> List[int]:
        """
        Find similar items for a list of query items.
        :param query_item_ids: List of query item indices
        :param top_k: Number of top similar items to return for each query item
        :param filter_query_items: Whether to filter out items in the query_items list
        :return: List of top-K similar items for each query item
        """
        return self.model.similar_items(query_item_id, top_k=top_k)