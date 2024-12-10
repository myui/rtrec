import logging
from typing import Any, Iterable, List, Tuple, override

from ..models.internal.slim_elastic import SLIMElastic
from .base import BaseModel

class SLIM(BaseModel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = SLIMElastic(**kwargs)

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
        interaction_matirx = self.interactions.to_csc()
        self.model.partial_fit_items(interaction_matirx, item_ids)

    def _recommend(self, user_id: int, candidate_item_ids: List[int], top_k: int = 10, filter_interacted: bool = True) -> List[int]:
        """
        Recommend top-K items for a given user.
        :param user_id: User index
        :param candidate_item_ids: List of candidate item indices
        :param top_k: Number of top items to recommend
        :param filter_interacted: Whether to filter out items the user has already interacted with
        :return: List of top-K item indices recommended for the user
        """
        interaction_matrix = self.interactions.to_csr()
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