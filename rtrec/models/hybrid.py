from collections import defaultdict
import logging
from typing import Any, Dict, Iterable, List, Optional, Tuple
from typing import override

from rtrec.models.internal.slim_elastic import SLIMElastic
from rtrec.utils.scoring import minmax_normalize

from ..utils.math import calc_norm
from .base import BaseModel
from .internal.lightfm_wrapper import LightFMWrapper

from scipy import sparse
from scipy.sparse import csr_matrix
import numpy as np
import implicit.cpu.topk as implicit

def compute_similarity_weight(num_contacts: int) -> float:
    """
    Compute the similarity weighting factor based on the number of user-item interactions.

    Args:
        num_contacts (int): Number of user-item interactions.

    Returns:
        float: Similarity weighting factor between 0 and 2.
        The factor approaches 2 as the number of contacts increases.
    """
    return 2.0 * num_contacts / (num_contacts + 1.0)

def comb_sum(fm_ids: np.ndarray, fm_scores: np.ndarray,
                slim_ids: List[int], slim_scores: np.ndarray) -> Dict[int, float]:
    """
    CombSUM rank aggregation for two recommendation lists.

    Parameters:
        fm_ids (np.ndarray): Item IDs from the FM model.
        fm_scores (np.ndarray): Scores from the FM model.
        slim_ids (List[int]): Item IDs from the SLIM model.
        slim_scores (np.ndarray): Scores from the SLIM model.

    Returns:
        Dict[int, float]: Dictionary containing the aggregated scores for each item.
    """
    summed_scores = defaultdict(float)

    # Process FM scores
    for item_id, score in zip(fm_ids, fm_scores):
        iid = int(item_id)
        summed_scores[iid] += float(score)

    # Process SLIM scores
    for item_id, score in zip(slim_ids, slim_scores):
        iid = int(item_id)
        summed_scores[iid] += float(score)

    return summed_scores

def comb_mnz(fm_ids: np.ndarray, fm_scores: np.ndarray,
             slim_ids: List[int], slim_scores: np.ndarray) -> Dict[int, float]:
    """
    CombMNZ rank aggregation for two recommendation lists.

    Parameters:
        fm_ids (np.ndarray): Item IDs from the FM model.
        fm_scores (np.ndarray): Scores from the FM model.
        slim_ids (List[int]): Item IDs from the SLIM model.
        slim_scores (np.ndarray): Scores from the SLIM model.

    Returns:
        Dict[int, float]: Dictionary containing the aggregated scores for each item.
    """
    summed_scores = defaultdict(float)
    item_counts = defaultdict(int)

    # Process FM scores
    for item_id, score in zip(fm_ids, fm_scores):
        iid = int(item_id)
        summed_scores[iid] += float(score)
        item_counts[iid] += 1

    # Process SLIM scores
    for item_id, score in zip(slim_ids, slim_scores):
        iid = int(item_id)
        summed_scores[iid] += float(score)
        item_counts[iid] += 1

    # Calculate CombMNZ scores
    return {
        item_id: score * item_counts[item_id]
        for item_id, score in summed_scores.items()
    }

class HybridSlimFM(BaseModel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.epochs = kwargs.get("epochs", 10)
        self.n_threads = kwargs.get("n_threads", 1)
        self.use_bias = kwargs.get("use_bias", True)
        self.model = LightFMWrapper(**kwargs)
        self.recorded_user_ids = set()
        self.recorded_item_ids = set()
        # Initialize SLIM model
        self.slim_model = SLIMElastic(kwargs)
        # Initialize interaction counts (> 1); hack to reduce memory usage
        self.interaction_counts: Dict[int, Dict[int, int]] = {} # item_id -> {user_id -> num_contacts}

    @override
    def fit(self, interactions: Iterable[Tuple[Any, Any, float, float]], update_interaction: bool=False, progress_bar: bool=True) -> None:
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
        num_users, num_items = ui_coo.shape
        assert user_features.shape[0] == num_users
        assert item_features.shape[0] == num_items
        sample_weights = ui_coo if self.model.loss == "warp-kos" else None
        self.model.fit_partial(ui_coo, user_features, item_features, sample_weight=sample_weights, epochs=self.epochs, num_threads=self.n_threads, verbose=progress_bar)
        # Fit SLIM model
        self.slim_model.partial_fit_items(ui_coo.tocsc(copy=False), item_ids, progress_bar=progress_bar)

    @override
    def add_interactions(self, interactions, update_interaction = False, record_interactions = False):
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
                # Increment interaction counts for this user-item pair
                self._incr_interaction_counts(user_id, item_id)
                # Add interaction to the model
                self.interactions.add_interaction(user_id, item_id, tstamp, rating, upsert=update_interaction)
                if record_interactions:
                    self._record_interactions(user_id, item_id, tstamp, rating)
            except Exception as e:
                logging.warning(f"Error processing interaction: {e}")
                continue

    def _record_interactions(self, user_id: int, item_id: int, tstamp: float, rating: float) -> None:
        self.recorded_user_ids.add(user_id)
        self.recorded_item_ids.add(item_id)

    def _fit_recorded(self, parallel: bool=False, progress_bar: bool=True) -> None:
        user_ids = list(self.recorded_user_ids)
        item_ids = list(self.recorded_item_ids)
        user_features = self._create_user_features(user_ids=user_ids)
        item_features = self._create_item_features(item_ids=item_ids)
        ui_coo = self.interactions.to_coo(select_users=user_ids, select_items=item_ids)
        num_users, num_items = ui_coo.shape
        assert user_features.shape[0] == num_users
        assert item_features.shape[0] == num_items
        sample_weights = ui_coo if self.model.loss == "warp-kos" else None
        self.model.fit_partial(ui_coo, user_features, item_features, sample_weight=sample_weights, epochs=self.epochs, num_threads=self.n_threads, verbose=progress_bar)

        # Fit SLIM model
        self.slim_model.partial_fit_items(ui_coo.tocsc(copy=False), item_ids, parallel=parallel, progress_bar=progress_bar)

        # Clear recorded user and item IDs
        self.recorded_user_ids.clear()
        self.recorded_item_ids.clear()

    def bulk_fit(self, parallel: bool=False, progress_bar: bool=True) -> None:
        user_features = self._create_user_features()
        item_features = self._create_item_features()
        ui_coo = self.interactions.to_coo()
        num_users, num_items = ui_coo.shape
        assert user_features.shape[0] == num_users
        assert item_features.shape[0] == num_items
        sample_weights = ui_coo if self.model.loss == "warp-kos" else None
        self.model.fit_partial(ui_coo, user_features, item_features, sample_weight=sample_weights, epochs=self.epochs, num_threads=self.n_threads, verbose=progress_bar)
        # Fit SLIM model
        ui_csc = ui_coo.tocsc(copy=False)
        self.slim_model.fit(ui_csc, parallel=parallel, progress_bar=progress_bar)

    def _recommend(self, user_id: int, candidate_item_ids: Optional[List[int]] = None, user_tags: Optional[List[str]] = None, top_k: int = 10, filter_interacted: bool = True) -> List[int]:
        if len(self.feature_store.user_features) == 0 and len(self.feature_store.item_features) == 0:
            # No features registered, use SLIM
            interaction_matrix = self.interactions.to_csr(select_users=[user_id])
            dense_output = not self.item_ids.pass_through
            result = self.slim_model.recommend(user_id, interaction_matrix, candidate_item_ids=candidate_item_ids, top_k=top_k, filter_interacted=filter_interacted, dense_output=dense_output)
            return result

        users_tags = [user_tags] if user_tags is not None else None
        user_features = self._create_user_features(user_ids=[user_id], users_tags=users_tags, slice=True)
        item_features = self._create_item_features(item_ids=candidate_item_ids, slice=False)

        user_biases, user_embeddings = self.model.get_user_representations(user_features) # user_biases: shape (1, num_users), user_embeddings: shape (1, num_features)
        item_biases, item_embeddings = self.model.get_item_representations(item_features) # item_biases: shape (num_items, 1), item_embeddings: shape (num_items, num_features)
        if self.use_bias:
            # Note np.ones for dot product with item biases
            user_vector = np.hstack((user_biases[:, np.newaxis], np.ones((user_biases.size, 1)), user_embeddings), dtype=np.float32)
            # Note np.ones for dot product with user biases
            item_vector = np.hstack((np.ones((item_biases.size, 1)), item_biases[:, np.newaxis], item_embeddings), dtype=np.float32)
        else:
            user_vector = user_embeddings
            item_vector = item_embeddings

        ui_csr = self.interactions.to_csr(select_users=[user_id], include_weights=True)
        filter_items = None
        if filter_interacted:
            filter_items = ui_csr[user_id:].indices

        fm_ids, fm_scores = implicit.topk(items=item_vector, query=user_vector, k=top_k, filter_items=filter_items, num_threads=self.n_threads)
        fm_ids = fm_ids.ravel()
        fm_scores = fm_scores.ravel()

        # implicit assigns negative infinity to the scores to be fitered out
        # see https://github.com/benfred/implicit/blob/v0.7.2/implicit/cpu/topk.pyx#L54
        # the largest possible negative finite value in float32, which is approximately -3.4028235e+38.
        min_score = -np.finfo(np.float32).max
        for i in range(len(fm_ids)):
            if candidate_item_ids and fm_ids[i] not in candidate_item_ids:
                # remove ids not exist in candidate_item_ids
                fm_ids = fm_ids[:i]
                break
            elif fm_scores[i] <= min_score:
                # remove ids less than or equal to min_score
                fm_ids = fm_ids[:i]
                break

        dense_output = not self.item_ids.pass_through
        slim_ids, slim_scores = self.slim_model.recommend(user_id, ui_csr, candidate_item_ids=candidate_item_ids, top_k=top_k, filter_interacted=filter_interacted, dense_output=dense_output, ret_scores=True)

        return self._ensemble_by_scores(user_id, fm_ids, fm_scores, slim_ids, slim_scores, top_k)

    @override
    def handle_unknown_user(self, user: Any) -> Optional[int]:
        """
        Handle unknown user in recommend_batch() method.
        """
        return 0 # workaround for a cold user problem

    @override
    def _recommend_batch(self,
                         user_ids: List[int],
                         candidate_item_ids: Optional[List[int]] = None,
                         users_tags: Optional[List[List[str]]] = None,
                         top_k: int = 10,
                         filter_interacted: bool = True) -> List[List[int]]:
        assert len(user_ids) > 0, "user_ids should not be empty"

        if len(self.feature_store.user_features) == 0 and len(self.feature_store.item_features) == 0:
            # No features registered, use SLIM
            interaction_matrix = self.interactions.to_csr(select_users=user_ids)
            dense_output = not self.item_ids.pass_through
            result = [
                self.slim_model.recommend(user_id, interaction_matrix, candidate_item_ids=candidate_item_ids, top_k=top_k, filter_interacted=filter_interacted, dense_output=dense_output)
                for user_id in user_ids
            ]
            return result

        user_features = self._create_user_features(user_ids=user_ids, users_tags=users_tags, slice=True)
        item_features = self._create_item_features(item_ids=candidate_item_ids, slice=False)

        user_biases, user_embeddings = self.model.get_user_representations(user_features)
        item_biases, item_embeddings = self.model.get_item_representations(item_features)
        if self.use_bias:
            # Note np.ones for dot product with item biases
            user_vector = np.hstack((user_biases[:, np.newaxis], np.ones((user_biases.size, 1)), user_embeddings), dtype=np.float32)
            # Note np.ones for dot product with user biases
            item_vector = np.hstack((np.ones((item_biases.size, 1)), item_biases[:, np.newaxis], item_embeddings), dtype=np.float32)
        else:
            user_vector = user_embeddings
            item_vector = item_embeddings

        ui_csr = self.interactions.to_csr(select_users=user_ids, include_weights=True)
        filter_query_items = None
        if filter_interacted:
            # see https://github.com/benfred/implicit/blob/v0.7.2/implicit/cpu/topk.pyx#L54
            filter_query_items = ui_csr[user_ids,:]

        ids_array, scores_array = implicit.topk(items=item_vector, query=user_vector, k=top_k, filter_query_items=filter_query_items, num_threads=self.n_threads)
        assert len(ids_array) == len(user_ids)

        results = []
        # implicit assigns negative infinity to the scores to be fitered out
        # see https://github.com/benfred/implicit/blob/v0.7.2/implicit/cpu/topk.pyx#L54
        # the largest possible negative finite value in float32, which is approximately -3.4028235e+38.
        min_score = -np.finfo(np.float32).max
        dense_output = not self.item_ids.pass_through
        for user_id, fm_ids, fm_scores in zip(user_ids, ids_array, scores_array):
            for i in range(len(fm_ids)):
                if candidate_item_ids and fm_ids[i] not in candidate_item_ids:
                    # remove ids not exist in candidate_item_ids
                    fm_ids = fm_ids[:i]
                    break
                elif fm_scores[i] <= min_score:
                    # remove ids less than or equal to min_score
                    fm_ids = fm_ids[:i]
                    break

            slim_ids, slim_scores = self.slim_model.recommend(user_id, ui_csr, candidate_item_ids=candidate_item_ids, top_k=top_k, filter_interacted=filter_interacted, dense_output=dense_output, ret_scores=True)
            # Combine scores from both models and get top-k item ids
            top_items = self._ensemble_by_scores(user_id, fm_ids, fm_scores, slim_ids, slim_scores, top_k)
            results.append(top_items)
        return results

    def _similar_items(self, query_item_id: int, query_item_tags: Optional[List[str]] = None, top_k: int = 10) -> List[Tuple[int, float]]:
        items_tags = [query_item_tags] if query_item_tags is not None else None
        query_features = self._create_item_features(item_ids=[query_item_id], items_tags=items_tags, slice=True)
        target_features = self._create_item_features()

        query_biases, query_embeddings = self.model.get_item_representations(query_features)
        target_biases, target_embeddings = self.model.get_item_representations(target_features)
        if self.use_bias:
            query_vector = np.hstack((query_biases[:, np.newaxis], query_embeddings), dtype=np.float32)
            target_vector = np.hstack((target_biases[:, np.newaxis], target_embeddings), dtype=np.float32)
        else:
            query_vector = query_embeddings
            target_vector = target_embeddings

        target_norm = calc_norm(target_vector)

        fm_ids, fm_scores = implicit.topk(items=target_vector, query=query_vector, k=top_k, item_norms=target_norm, filter_items=np.array([query_item_id], dtype="int32"), num_threads=self.n_threads)
        fm_ids: np.ndarray = fm_ids.ravel()
        fm_scores: np.ndarray  = fm_scores.ravel()

        # implicit assigns negative infinity to the scores to be fitered out
        # see https://github.com/benfred/implicit/blob/v0.7.2/implicit/cpu/topk.pyx#L54
        # the largest possible negative finite value in float32, which is approximately -3.4028235e+38.
        min_score = -np.finfo(np.float32).max
        # remove ids less than or equal to min_score
        for i in range(len(fm_ids)):
            if fm_scores[i] <= min_score:
                fm_ids = fm_ids[:i]
                fm_scores = fm_scores[:i]
                break

        # Convert back to cosine similarity from dot product scores
        query_norm = calc_norm(query_vector)
        fm_scores = fm_scores / query_norm

        # exclude the query item itself
        valid_mask = fm_ids != query_item_id
        fm_ids = fm_ids[valid_mask]
        fm_scores = fm_scores[valid_mask]

        # Get SLIM similar items
        slim_ids, slim_scores = self.slim_model.similar_items(query_item_id, top_k=top_k, ret_ndarrays=True)

        # Combine scores from both models
        fm_scores = minmax_normalize(fm_scores)
        slim_scores = minmax_normalize(slim_scores)
        comb_scores = comb_sum(fm_ids, fm_scores, slim_ids, slim_scores)
        # Get top-k item ids
        sorted_items = sorted(comb_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return sorted_items

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
            user_matrix = user_matrix[np.array(user_ids),:]
        return user_matrix

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
            item_matrix = item_matrix[np.array(item_ids),:]
        return item_matrix

    def _incr_interaction_counts(self, user_id: int, item_id: int):
        if self.interactions.has_interaction(user_id, item_id):
            interactions_for_item = self.interaction_counts.get(item_id, {})
            cur_count = interactions_for_item.get(user_id, 1)
            interactions_for_item[user_id] = cur_count + 1 # note start from 2
            self.interaction_counts[item_id] = interactions_for_item

    def _get_interaction_counts(self, user_id: int, item_id: int) -> int:
        # First check if we have cached a count > 1
        interactions_for_item = self.interaction_counts.get(item_id, {})
        if user_id in interactions_for_item:
            return interactions_for_item[user_id]

        # Otherwise, check if there's at least one interaction
        if self.interactions.has_interaction(user_id, item_id):
            return 1

        # No interactions
        return 0

    def _ensemble_by_scores(self, user_id: int,
                    fm_ids: np.ndarray, fm_scores: np.ndarray,
                    slim_ids: List[int], slim_scores: np.ndarray,
                    topk: int) -> List[int]:
        """
        Combine scores from factorization machine (FM) and SLIM models using weighted ensemble.
        Returns top-k items based on the combined scores.

        Args:
            user_id (int): The user ID for which recommendations are being made.
            fm_ids (List[int]): Item IDs from the FM model.
            fm_scores (np.ndarray): Scores from the FM model.
            slim_ids (List[int]): Item IDs from the SLIM model.
            slim_scores (np.ndarray): Scores from the SLIM model.
            topk (int, optional): Number of top items to return.

        Returns:
            List[int]: top-k score item ids
        """
        # Normalize scores from both models
        fm_scores = minmax_normalize(fm_scores)
        slim_scores = minmax_normalize(slim_scores)

        # Create a combined scores initialized with FM scores
        combined_dict = dict(zip(fm_ids, fm_scores))

        # Process SLIM scores
        for i, item_id in enumerate(slim_ids):
            # Get interaction count
            num_contacts = self._get_interaction_counts(user_id, item_id)

            # Apply similarity weighting
            weight = compute_similarity_weight(num_contacts)

            # Add weighted SLIM score
            combined_dict[item_id] = combined_dict.get(item_id, 0.0) + weight * slim_scores[i]

        # Sort items by score in descending order and get top-k
        sorted_items = sorted(combined_dict.items(), key=lambda x: x[1], reverse=True)[:topk]

        # Get top-k itemm ids
        top_ids, _ = zip(*sorted_items)
        return list(top_ids)
