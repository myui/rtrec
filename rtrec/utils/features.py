from typing import List, Optional, Set
from scipy.sparse import csr_matrix
import numpy as np
from .collections import SortedSet

class FeatureStore:

    def __init__(self):
        self.user_features: SortedSet[str] = SortedSet()
        self.item_features: SortedSet[str] = SortedSet()
        self.user_feature_map: dict[int, List[int]] = {}
        self.item_feature_map: dict[int, List[int]] = {}

    def put_user_feature(self, user_id: int, user_tags: List[str]) -> None:
        """
        Add a list of user features to the user features set.
        Replace the existing user features if the user ID already exists.

        Parameters:
            user_id (int): User ID
            user_tags (List[str]): List of user features
        """
        user_feature_ids = []
        for tag in user_tags:
            tag_id = self.user_features.add(tag)
            if tag_id not in user_feature_ids:
                user_feature_ids.append(tag_id)
        self.user_feature_map[user_id] = user_feature_ids

    def put_item_feature(self, item_id: int, item_tags: List[str]) -> None:
        """
        Add a list of item features to the item features set.
        Replace the existing item features if the item ID already exists.

        Parameters:
            item_id (int): Item ID
            item_tags (List[str]): List of item features
        """
        item_feature_ids = []
        for tag in item_tags:
            tag_id = self.item_features.add(tag)
            if tag_id not in item_feature_ids:
                item_feature_ids.append(tag_id)
        self.item_feature_map[item_id] = item_feature_ids

    def get_user_feature_repr(self, user_tags: List[str]) -> csr_matrix:
        """
        Get the user feature representation matrix for the given user tags.

        Parameters:
            user_tags (List[str]): List of user tags
        Returns:
            csr_matrix: User feature representation of shape (1, n_features)
        """
        user_feature_ids = []
        for tag in user_tags:
            tag_id = self.user_features.index(tag)
            if tag_id >= 0:
                user_feature_ids.append(tag_id)

        cols = np.array(user_feature_ids)
        rows = np.zeros(len(user_feature_ids))
        data = np.ones(len(user_feature_ids))
        return csr_matrix((data, (rows, cols)), shape=(1, len(self.user_features)), dtype=np.float32)

    def get_item_feature_repr(self, item_tags: List[str]) -> csr_matrix:
        """
        Get the item feature representation for the given item tags.

        Parameters:
            item_tags (List[str]): List of item tags
        Returns:
            csr_matrix: Item feature representation of shape (1, n_features)
        """
        item_feature_ids = []
        for tag in item_tags:
            tag_id = self.item_features.index(tag)
            if tag_id >= 0:
                item_feature_ids.append(tag_id)

        cols = np.array(item_feature_ids)
        rows = np.zeros(len(item_feature_ids))
        data = np.ones(len(item_feature_ids))
        return csr_matrix((data, (rows, cols)), shape=(1, len(self.item_features)), dtype=np.float32)

    def build_user_features_matrix(self, user_ids: Optional[List[int]]=None, num_users: Optional[int]=None) -> csr_matrix | None:
        """
        Parameters:
            user_ids (Optional[List[int]]): List of user IDs to build the user features matrix for.
            num_users (Optional[int]): Number of users to build the user features matrix for.
        Returns:
            csr_matrix: User features matrix of shape (n_users, n_features). If no user features are registered, return None.
        """
        # If no user features are registered, return None
        if len(self.user_features) == 0:
            return None

        rows, cols, data = [], [], []
        max_user_id = 0
        if user_ids is None:
            for user_id, feature_ids in self.user_feature_map.items():
                for feature_id in feature_ids:
                    rows.append(user_id)
                    cols.append(feature_id)
                    data.append(1)
                    max_user_id = max(max_user_id, user_id)
        else:
            for user_id in user_ids:
                for feature_id in self.user_feature_map.get(user_id, []):
                    rows.append(user_id)
                    cols.append(feature_id)
                    data.append(1)
                    max_user_id = max(max_user_id, user_id)

        if num_users is None:
            num_users = max_user_id + 1
        return csr_matrix((data, (rows, cols)), shape=(num_users, len(self.user_features)), dtype=np.float32)

    def build_item_features_matrix(self, item_ids: Optional[int]=None, num_items: Optional[int]=None) -> csr_matrix | None:
        """
        Parameters:
            item_ids (Optional[List[int]]): List of item IDs to build the item features matrix for
            num_items (Optional[int]): Number of items to build the item features matrix for
        Returns:
            csr_matrix: Item features matrix of shape (n_items, n_features). If no item features are registered, return None.
        """
        # If no item features are registered, return None
        if len(self.item_features) == 0:
            return None

        rows, cols, data = [], [], []
        max_item_id = 0
        if item_ids is None:
            for item_id, feature_ids in self.item_feature_map.items():
                for feature_id in feature_ids:
                    rows.append(item_id)
                    cols.append(feature_id)
                    data.append(1)
                    max_item_id = max(max_item_id, item_id)
        else:
            for item_id in item_ids:
                for feature_id in self.item_feature_map.get(item_id, []):
                    rows.append(item_id)
                    cols.append(feature_id)
                    data.append(1)
                    max_item_id = max(max_item_id, item_id)

        if num_items is None:
            num_items = max_item_id + 1
        return csr_matrix((data, (rows, cols)), shape=(num_items, len(self.item_features)), dtype=np.float32)
