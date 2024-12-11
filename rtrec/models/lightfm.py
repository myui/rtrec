import logging
from typing import Any, Iterable, List, Tuple
from .base import BaseModel
from scipy.sparse import csc_matrix
from .internal.lightfm_wrapper import LightFMWrapper

class LightFM(BaseModel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = LightFMWrapper(**kwargs)
        # TODO

        self.recorded_user_ids = set()
        self.recorded_item_ids = set()

    def fit(self, user_interactions: Iterable[Tuple[Any, Any, float, float]], update_interaction: bool=False) -> None:        
        pass # TODO

    def _record_interactions(self, user_id: int, item_id: int, tstamp: float, rating: float) -> None:
        self.recorded_user_ids.add(user_id)
        self.recorded_item_ids.add(item_id)

    def _fit_recorded(self) -> None:
        interactions = self.interactions.to_coo(select_users=list(self.recorded_user_ids), select_items=list(self.recorded_item_ids))
        # TODO
        self.recorded_user_ids.clear()
        self.recorded_item_ids.clear()

    def bulk_fit(self) -> None:
        interactions = self.interactions.to_coo()

        pass # TODO

    def _recommend(self, user_id: int, top_k: int = 10, filter_interacted: bool = True) -> List[int]:
        pass # TODO

    def _similar_items(self, query_item_id: int, top_k: int = 10) -> List[int]:
        pass # TODO
