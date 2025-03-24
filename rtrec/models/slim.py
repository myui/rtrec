import logging
from typing import Any, Iterable, List, Optional, Tuple
# require typing-extensions >= 4.5
# from typing import override

from ..models.internal.slim_elastic import SLIMElastic
from .base import BaseModel

class SLIM(BaseModel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = SLIMElastic(kwargs)
        self.recorded_item_ids = set()

    #@override
    def fit(self, interactions: Iterable[Tuple[Any, Any, float, float]], update_interaction: bool=False, progress_bar: bool=True) -> None:
        item_id_set = set()
        for user, item, tstamp, rating in interactions:
            try:
                user_id = self.user_ids.identify(user)
                item_id = self.item_ids.identify(item)
                self.interactions.add_interaction(user_id, item_id, tstamp, rating, upsert=update_interaction)
                item_id_set.add(item_id)
            except Exception as e:
                logging.warning(f"Error processing interaction: {e}")
                continue
        item_ids = list(item_id_set)
        interaction_matrix = self.interactions.to_csc(item_ids)
        self.model.partial_fit_items(interaction_matrix, item_ids, progress_bar=progress_bar)

    def _record_interactions(self, user_id: int, item_id: int, tstamp: float, rating: float) -> None:
        self.recorded_item_ids.add(item_id)

    def _fit_recorded(self, parallel: bool=False, progress_bar: bool=True) -> None:
        item_ids = list(self.recorded_item_ids)
        interaction_matrix = self.interactions.to_csc(item_ids)
        self.model.partial_fit_items(interaction_matrix, item_ids, parallel=parallel, progress_bar=progress_bar)
        self.recorded_item_ids.clear()

    def bulk_fit(self, parallel: bool=False, progress_bar: bool=True) -> None:
        """
        Fit the recommender model on the given interaction matrix.
        :param interaction_matrix: Sparse interaction matrix
        :param parallel: Whether to run the fitting process in parallel. Defaults to False
        :param progress_bar: Whether to display a progress bar
        """
        interaction_matrix = self.interactions.to_csc()
        self.model.fit(interaction_matrix, parallel=parallel, progress_bar=progress_bar)

    def _recommend(self, user_id: int, candidate_item_ids: Optional[List[int]] = None, user_tags: Optional[List[str]] = None, top_k: int = 10, filter_interacted: bool = True) -> List[int]:
        """
        Recommend top-K items for a given user.
        :param user_id: User index
        :param candidate_item_ids: List of candidate item indices
        :param user_tags: List of user tags
        :param candidate_item_ids: List of candidate item indices
        :param top_k: Number of top items to recommend
        :param filter_interacted: Whether to filter out items the user has already interacted with
        :return: List of top-K item indices recommended for the user
        """
        interaction_matrix = self.interactions.to_csr(select_users=[user_id])
        dense_output = not self.item_ids.pass_through
        return self.model.recommend(user_id, interaction_matrix, candidate_item_ids=candidate_item_ids, top_k=top_k, filter_interacted=filter_interacted, dense_output=dense_output)

    def _similar_items(self, query_item_id: int, query_item_tags: Optional[List[str]] = None, top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Find similar items for a list of query items.
        :param query_item_ids: List of query item indices
        :param query_item_tags: List of tags for each query item
        :param top_k: Number of top similar items to return for each query item
        :param filter_query_items: Whether to filter out items in the query_items list
        :return: List of top-K similar items for each query item with similarity scores
        """
        return self.model.similar_items(query_item_id, top_k=top_k)
