import logging
from typing import Any, Iterable, List, Optional, Tuple, override, TYPE_CHECKING

if TYPE_CHECKING:  # static analyzers
    from typing import Self
else:  # runtime on 3.10
    try:
        from typing import Self  # 3.11+
    except ImportError:
        from typing_extensions import Self

import implicit.cpu.topk as implicit
import numpy as np
from scipy import sparse
from scipy.sparse import csr_matrix

from ..utils.math import calc_norm
from .base import BaseModel
from .internal.lightfm_wrapper import LightFMWrapper


class LightFM(BaseModel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.epochs = kwargs.get("epochs", 10)
        self.n_threads = kwargs.get("n_threads", 1)
        self.use_bias = kwargs.get("use_bias", True)
        self.model = LightFMWrapper(**kwargs)
        self.recorded_user_ids = set()
        self.recorded_item_ids = set()

    @override
    def fit(self, interactions: Iterable[Tuple[Any, Any, float, float]], update_interaction: bool=False, progress_bar: bool=True) -> Self:
        item_id_set, user_id_set = set(), set()
        for user, item, tstamp, rating in interactions:
            try:
                user_id = self.user_ids.identify(user)
                item_id = self.item_ids.identify(item)
                self.interactions.add_interaction(user_id, item_id, tstamp, rating, upsert=update_interaction)
                item_id_set.add(item_id)
                user_id_set.add(user_id)
            except Exception as e:
                logging.warning(f"Error processing interaction: {e}")
                continue

        item_ids = list(item_id_set)
        user_ids = list(user_id_set)
        user_features = self._create_user_features(user_ids=user_ids)
        item_features = self._create_item_features(item_ids=item_ids)
        ui_coo = self.interactions.to_coo(select_users=user_ids, select_items=item_ids)
        num_users, num_items = ui_coo.shape # type: ignore
        assert user_features.shape[0] == num_users # type: ignore
        assert item_features.shape[0] == num_items # type: ignore
        sample_weights = ui_coo if self.model.loss == "warp-kos" else None
        self.model.fit_partial(ui_coo, user_features, item_features, sample_weight=sample_weights, epochs=self.epochs, num_threads=self.n_threads, verbose=progress_bar)
        return self

    def _record_interactions(self, user_id: int, item_id: int, tstamp: float, rating: float) -> None:
        self.recorded_user_ids.add(user_id)
        self.recorded_item_ids.add(item_id)

    def _fit_recorded(self, parallel: bool=False, progress_bar: bool=True) -> Self:
        user_ids = list(self.recorded_user_ids)
        item_ids = list(self.recorded_item_ids)
        user_features = self._create_user_features(user_ids=user_ids)
        item_features = self._create_item_features(item_ids=item_ids)
        ui_coo = self.interactions.to_coo(select_users=user_ids, select_items=item_ids)
        num_users, num_items = ui_coo.shape # type: ignore
        assert user_features.shape[0] == num_users # type: ignore
        assert item_features.shape[0] == num_items # type: ignore
        sample_weights = ui_coo if self.model.loss == "warp-kos" else None
        self.model.fit_partial(ui_coo, user_features, item_features, sample_weight=sample_weights, epochs=self.epochs, num_threads=self.n_threads, verbose=progress_bar)

        # Clear recorded user and item IDs
        self.recorded_user_ids.clear()
        self.recorded_item_ids.clear()
        return self

    def bulk_fit(self, parallel: bool=False, progress_bar: bool=True) -> Self:
        user_features = self._create_user_features()
        item_features = self._create_item_features()
        ui_coo = self.interactions.to_coo()
        num_users, num_items = ui_coo.shape # type: ignore
        assert user_features.shape[0] == num_users # type: ignore
        assert item_features.shape[0] == num_items # type: ignore
        sample_weights = ui_coo if self.model.loss == "warp-kos" else None
        self.model.fit_partial(ui_coo, user_features, item_features, sample_weight=sample_weights, epochs=self.epochs, num_threads=self.n_threads, verbose=progress_bar)
        return self

    def _recommend(self, user_id: int, candidate_item_ids: Optional[List[int]] = None, user_tags: Optional[List[str]] = None, top_k: int = 10, filter_interacted: bool = True) -> List[int]:
        users_tags = [user_tags] if user_tags is not None else None
        user_features = self._create_user_features(user_ids=[user_id], users_tags=users_tags, slice=True)
        item_features = self._create_item_features(item_ids=candidate_item_ids, slice=False)

        user_biases, user_embeddings = self.model.get_user_representations(user_features)
        item_biases, item_embeddings = self.model.get_item_representations(item_features)
        if self.use_bias:
            # Note np.ones for dot product with item biases
            user_vector = np.hstack((user_biases[:, np.newaxis], np.ones((user_biases.size, 1)), user_embeddings), dtype=np.float32) # type: ignore
            # Note np.ones for dot product with user biases
            item_vector = np.hstack((np.ones((item_biases.size, 1)), item_biases[:, np.newaxis], item_embeddings), dtype=np.float32) # type: ignore
        else:
            user_vector = user_embeddings
            item_vector = item_embeddings

        filter_items = None
        if filter_interacted:
            ui_csr = self.interactions.to_csr(select_users=[user_id], include_weights=False)
            filter_items = ui_csr[user_id:].indices # type: ignore

        ids, scores = implicit.topk(items=item_vector, query=user_vector, k=top_k, filter_items=filter_items, num_threads=self.n_threads)
        ids = ids.ravel()
        scores = scores.ravel()

        # implicit assigns negative infinity to the scores to be fitered out
        # see https://github.com/benfred/implicit/blob/v0.7.2/implicit/cpu/topk.pyx#L54
        # the largest possible negative finite value in float32, which is approximately -3.4028235e+38.
        min_score = -np.finfo(np.float32).max
        for i in range(len(ids)):
            if candidate_item_ids and ids[i] not in candidate_item_ids:
                # remove ids not exist in candidate_item_ids
                ids = ids[:i]
                break
            elif scores[i] <= min_score:
                # remove ids less than or equal to min_score
                ids = ids[:i]
                break

        return ids.tolist() # ndarray to list

    @override
    def handle_unknown_user(self, user: Any) -> Optional[int]:
        """
        Handle unknown user in recommend_batch() method.
        """
        return 0 # workaround for a cold user problem

    @override
    def _recommend_cold_batch(self, user_ids: List[Optional[int]], candidate_item_ids: Optional[List[int]] = None, users_tags: Optional[List[List[str]]] = None, top_k: int = 10) -> List[List[int]]:
        if candidate_item_ids is None:
            candidate_item_ids = self.interactions.get_hot_items(filter_interacted=False)

        if users_tags is None:
            users_tags = [[] for _ in range(len(user_ids))]

        user_features = self._cold_user_features(users_tags=users_tags)
        item_features = self._create_item_features(item_ids=candidate_item_ids, slice=False)

        user_biases, user_embeddings = self.model.get_user_representations(user_features)
        item_biases, item_embeddings = self.model.get_item_representations(item_features)
        if self.use_bias:
            # Note np.ones for dot product with item biases
            user_vector = np.hstack((user_biases[:, np.newaxis], np.ones((user_biases.size, 1)), user_embeddings), dtype=np.float32) # type: ignore
            # Note np.ones for dot product with user biases
            item_vector = np.hstack((np.ones((item_biases.size, 1)), item_biases[:, np.newaxis], item_embeddings), dtype=np.float32) # type: ignore
        else:
            user_vector = user_embeddings
            item_vector = item_embeddings

        ids_array, scores_array = implicit.topk(items=item_vector, query=user_vector, k=top_k, num_threads=self.n_threads)
        assert len(ids_array) == len(user_ids)

        results = []
        # implicit assigns negative infinity to the scores to be fitered out
        # see https://github.com/benfred/implicit/blob/v0.7.2/implicit/cpu/topk.pyx#L54
        # the largest possible negative finite value in float32, which is approximately -3.4028235e+38.
        min_score = -np.finfo(np.float32).max
        for ids, scores in zip(ids_array, scores_array):
            for i in range(len(ids)):
                if candidate_item_ids and ids[i] not in candidate_item_ids:
                    # remove ids not exist in candidate_item_ids
                    ids = ids[:i]
                    break
                elif scores[i] <= min_score:
                    # remove ids less than or equal to min_score
                    ids = ids[:i]
                    break
            results.append(ids.tolist())
        return results

    def _cold_user_features(self, users_tags: List[List[str]]) -> csr_matrix:
        num_rows = len(users_tags)
        num_hot_users = self.interactions.shape[0]

        # Create zero matrix for identity since cold users have no history
        users = sparse.csr_matrix((num_rows, num_hot_users), dtype="float32")

        num_user_features = self.model.user_embeddings.shape[0] - num_hot_users # type: ignore
        assert num_user_features > 0, f"num_user_features should be greater than 0, but got {num_user_features}"

        # create user features matrix of shape (num_rows, num_hot_items) from users_tags
        rows, cols, data = [], [], []
        user_features = self.feature_store.user_features
        for row_id, user_tags in enumerate(users_tags):
            for tag in user_tags:
                tag_id = user_features.index(tag)
                if tag_id < 0:
                    continue
                if tag_id >= num_user_features:
                    continue # ignore not learned features
                rows.append(row_id)
                cols.append(tag_id)
                data.append(1)

        features = csr_matrix((data, (rows, cols)), shape=(num_rows, num_user_features), dtype="float32") # Shape: (num_rows, num_hot_items)

        # Horizontal stack the identity matrix and the features matrix
        return sparse.hstack((users, features), format="csr") # type: ignore

    @override
    def _recommend_hot_batch(self, user_ids: List[int], candidate_item_ids: Optional[List[int]] = None, users_tags: Optional[List[List[str]]] = None, top_k: int = 10, filter_interacted: bool = True) -> List[List[int]]:
        user_features = self._create_user_features(user_ids=user_ids, users_tags=users_tags, slice=True)
        item_features = self._create_item_features(item_ids=candidate_item_ids, slice=False)

        user_biases, user_embeddings = self.model.get_user_representations(user_features)
        item_biases, item_embeddings = self.model.get_item_representations(item_features)
        if self.use_bias:
            # Note np.ones for dot product with item biases
            user_vector = np.hstack((user_biases[:, np.newaxis], np.ones((user_biases.size, 1)), user_embeddings), dtype=np.float32) # type: ignore
            # Note np.ones for dot product with user biases
            item_vector = np.hstack((np.ones((item_biases.size, 1)), item_biases[:, np.newaxis], item_embeddings), dtype=np.float32) # type: ignore
        else:
            user_vector = user_embeddings
            item_vector = item_embeddings

        filter_query_items = None
        if filter_interacted:
            # see https://github.com/benfred/implicit/blob/v0.7.2/implicit/cpu/topk.pyx#L54
            filter_query_items = self.interactions.to_csr(select_users=user_ids, include_weights=False)
            filter_query_items = filter_query_items[user_ids,:]

        ids_array, scores_array = implicit.topk(items=item_vector, query=user_vector, k=top_k, filter_query_items=filter_query_items, num_threads=self.n_threads)
        assert len(ids_array) == len(user_ids)

        results = []
        # implicit assigns negative infinity to the scores to be fitered out
        # see https://github.com/benfred/implicit/blob/v0.7.2/implicit/cpu/topk.pyx#L54
        # the largest possible negative finite value in float32, which is approximately -3.4028235e+38.
        min_score = -np.finfo(np.float32).max
        for ids, scores in zip(ids_array, scores_array):
            for i in range(len(ids)):
                if candidate_item_ids and ids[i] not in candidate_item_ids:
                    # remove ids not exist in candidate_item_ids
                    ids = ids[:i]
                    break
                elif scores[i] <= min_score:
                    # remove ids less than or equal to min_score
                    ids = ids[:i]
                    break
            results.append(ids.tolist())
        return results

    def _similar_items(self, query_item_id: int, query_item_tags: Optional[List[str]] = None, top_k: int = 10) -> List[Tuple[int, float]]:
        items_tags = [query_item_tags] if query_item_tags is not None else None
        query_features = self._create_item_features(item_ids=[query_item_id], items_tags=items_tags, slice=True)
        target_features = self._create_item_features()

        query_biases, query_embeddings = self.model.get_item_representations(query_features)
        target_biases, target_embeddings = self.model.get_item_representations(target_features)
        if self.use_bias:
            query_vector = np.hstack((query_biases[:, np.newaxis], query_embeddings), dtype=np.float32) # type: ignore
            target_vector = np.hstack((target_biases[:, np.newaxis], target_embeddings), dtype=np.float32) # type: ignore
        else:
            query_vector = query_embeddings
            target_vector = target_embeddings

        if target_vector is None:
            raise ValueError("target_vector is None, cannot compute norm.")
        target_norm = calc_norm(target_vector)

        ids_raw, scores_raw = implicit.topk(items=target_vector, query=query_vector, k=top_k, item_norms=target_norm, filter_items=np.array([query_item_id], dtype="int32"), num_threads=self.n_threads)
        ids: np.ndarray = ids_raw.ravel()
        scores: np.ndarray = scores_raw.ravel()

        # implicit assigns negative infinity to the scores to be fitered out
        # see https://github.com/benfred/implicit/blob/v0.7.2/implicit/cpu/topk.pyx#L54
        # the largest possible negative finite value in float32, which is approximately -3.4028235e+38.
        min_score = -np.finfo(np.float32).max
        # remove ids less than or equal to min_score
        for i in range(len(ids)):
            if scores[i] <= min_score:
                ids = ids[:i]
                scores = scores[:i]
                break

        # Convert back to cosine similarity from dot product scores
        if query_vector is None:
            raise ValueError("query_vector is None, cannot compute norm.")
        query_norm = calc_norm(query_vector)
        scores = scores / query_norm

        # exclude the query item itself
        valid_mask = ids != query_item_id
        ids = ids[valid_mask].tolist() # type: ignore
        scores = scores[valid_mask].tolist() # type: ignore
        return list(zip(ids, scores))

    def _serialize(self) -> dict:
        """
        Serialize the LightFM model state to a dictionary.
        :return: Dictionary containing model state
        """
        return {
            'model': self.model,
            'interactions': self.interactions,
            'user_ids': self.user_ids,
            'item_ids': self.item_ids,
            'feature_store': self.feature_store,
            'epochs': self.epochs,
            'n_threads': self.n_threads,
            'use_bias': self.use_bias,
            # 'recorded_user_ids': list(self.recorded_user_ids),
            # 'recorded_item_ids': list(self.recorded_item_ids)
        }

    @classmethod
    def _deserialize(cls, data: dict) -> Self:
        """
        Deserialize the LightFM model state from a dictionary.
        :param data: Dictionary containing model state
        :return: LightFM model instance
        """
        # Create instance with the right configuration
        kwargs = {
            'epochs': data['epochs'],
            'n_threads': data['n_threads'],
            'use_bias': data['use_bias']
        }
        instance = cls(**kwargs)

        # Restore model state
        instance.model = data['model']
        instance.interactions = data['interactions']
        instance.user_ids = data['user_ids']
        instance.item_ids = data['item_ids']
        instance.feature_store = data['feature_store']
        # instance.recorded_user_ids = set(data['recorded_user_ids'])
        # instance.recorded_item_ids = set(data['recorded_item_ids'])

        return instance

    def _create_user_features(self, user_ids: Optional[List[int]]=None, users_tags: Optional[List[List[str]]] = None, slice: bool=False) -> csr_matrix:
        """
        Create user features matrix for the given users.

        Parameters:
            user_ids (Optional[List[int]]): List of User IDs to create the user features matrix for.
            users_tags (Optional[List[List[str]]): List of user tags for each user.
            slice (bool): Whether to slice the user features matrix to the given user IDs.
        Returns:
            csr_matrix: User features matrix of shape (num_users, num_users + num_features) for the given users.
        """
        if slice and user_ids is not None:
            num_users = max(user_ids) + 1
        else:
            num_users = self.interactions.shape[0]

        user_features = self.feature_store.build_user_features_matrix(user_ids, users_tags=users_tags, num_users=num_users)  # Shape: (num_users, num_features)

        # Create user identity matrix of shape (len(user_ids), num_users) for the given users
        if user_ids is None:
            user_identity = sparse.identity(num_users, dtype="float32", format="csr")
        else:
            rows = user_ids
            cols = user_ids
            data = [1] * len(user_ids)
            user_identity = csr_matrix((data, (rows, cols)), dtype="float32", shape=(num_users, self.interactions.shape[0]))

        if user_features is None:
            user_matrix = user_identity
        else:
            user_matrix = sparse.hstack((user_identity, user_features), format="csr")

        if slice and user_ids is not None:
            user_matrix = user_matrix[np.array(user_ids),:] # type: ignore
        return user_matrix # type: ignore

    def _create_item_features(self, item_ids: Optional[List[int]]=None, items_tags: Optional[List[List[str]]] = None, slice: bool=False) -> csr_matrix:
        """
        Create item features matrix for the given items.

        Parameters:
            item_ids (Optional[List[int]]): List of Item IDs to create the item features matrix for.
            items_tags (Optional[List[List[str]]): List of item tags for each item
            slice (bool): Whether to slice the item features matrix to the given item IDs.
        Returns:
            csr_matrix: Item features matrix of shape (num_items, num_items + num_features) for the given items.
        """
        if slice and item_ids is not None:
            num_items = max(item_ids) + 1
        else:
            num_items = self.interactions.shape[1]

        item_features = self.feature_store.build_item_features_matrix(item_ids, items_tags=items_tags, num_items=num_items)

        # Create item identity matrix of shape (len(item_ids), num_items) for the given items
        if item_ids is None:
            item_identity = sparse.identity(num_items, dtype="float32", format="csr")
        else:
            rows = item_ids
            cols = item_ids
            data = [1] * len(item_ids)
            item_identity = csr_matrix((data, (rows, cols)), dtype="float32", shape=(num_items, self.interactions.shape[1]))

        if item_features is None:
            item_matrix = item_identity
        else:
            item_matrix = sparse.hstack((item_identity, item_features), format="csr")

        if slice and item_ids is not None:
            item_matrix = item_matrix[np.array(item_ids),:] # type: ignore
        return item_matrix # type: ignore
