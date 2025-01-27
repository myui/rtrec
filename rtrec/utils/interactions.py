from collections import defaultdict
from typing import List, Optional, Any
import time, math
import logging
from datetime import datetime, timezone

import numpy as np
from scipy.sparse import csr_matrix, csc_matrix, coo_matrix
from .lru import LRUFreqSet

class UserItemInteractions:
    def __init__(self, min_value: int = -5, max_value: int = 10, decay_in_days: Optional[int] = None, **kwargs: Any) -> None:
        """
        Initializes the UserItemInteractions class.

        Args:
            min_value (int): Minimum allowable value for interactions.
            max_value (int): Maximum allowable value for interactions.
            decay_rate (Optional[float]): Rate at which interactions decay over time.
                                          If None, no decay is applied.
        """
        # Store interactions as a dictionary of dictionaries in shape {user_id: {item_id: (value, timestamp)}}
        self.interactions: defaultdict[int, dict[int, tuple[float, float]]] = defaultdict(dict)
        self.all_item_ids = set()
        n_recent_hot = kwargs.get("n_recent_hot", 100_000)
        self.hot_items = LRUFreqSet(capacity=n_recent_hot)
        assert max_value > min_value, f"max_value should be greater than min_value {max_value} > {min_value}"
        self.min_value = min_value
        self.max_value = max_value
        if decay_in_days is None:
            self.decay_rate = None
        else:
            # Follow the way in "Time Weight collaborative filtering" in the paper
            # Half-life decay in time: decay_rate = 1 - ln(2) / decay_in_days
            # https://dl.acm.org/doi/10.1145/1099554.1099689
            self.decay_rate = 1.0 - (math.log(2) / decay_in_days)
        self.max_user_id = 0
        self.max_item_id = 0
        self.max_timestamp = 0.0

    def get_decay_rate(self) -> Optional[float]:
        """
        Retrieves the decay rate for interactions.

        Returns:
            Optional[float]: The decay rate for interactions.
        """
        return self.decay_rate

    def set_decay_rate(self, decay_rate: Optional[float]) -> None:
        """
        Sets the decay rate for interactions.

        Args:
            decay_rate (Optional[float]): The decay rate for interactions.
        """
        self.decay_rate = decay_rate

    def _apply_decay(self, value: float, last_timestamp: float) -> float:
        """
        Applies decay to a given value based on the elapsed time since the last interaction.

        Args:
            value (float): The original interaction value.
            last_timestamp (float): The timestamp of the last interaction.

        Returns:
            float: The decayed interaction value.
        """
        if self.decay_rate is None:
            return value

        elapsed_seconds = self.max_timestamp - last_timestamp
        elapsed_days = elapsed_seconds / 86400.0

        return value * self.decay_rate ** elapsed_days # approximated exponential decay in time e^(-ln(2)/decay_in_days * elapsed_days)

    def add_interaction(self, user_id: int, item_id: int, tstamp: float, delta: float = 1.0, upsert: bool = False) -> None:
        """
        Adds or updates an interaction count for a user-item pair.

        Args:
            user_id (int): ID of the user.
            item_id (int): ID of the item.
            delta (float): Change in interaction count (default is 1.0).
            upsert (bool): Flag to update the interaction count if it already exists (default is False).
        """
        # Validate the timestamp
        current_unix_time = time.time()
        if tstamp > current_unix_time + 180.0:  # Allow for a 180-second buffer
            current_rfc3339 = datetime.fromtimestamp(current_unix_time, tz=timezone.utc).isoformat() + "Z"
            tstamp_rfc3339 = datetime.fromtimestamp(tstamp, tz=timezone.utc).isoformat() + "Z"
            logging.warning(f"Timestamp {tstamp_rfc3339} is in the future. Current time is {current_rfc3339}")

        # Update the maximum timestamp to avoid conflicts
        self.max_timestamp = max(self.max_timestamp, tstamp + 1.0)

        if upsert:
            self.interactions[user_id][item_id] = (delta, tstamp)
        else:
            current = self.get_user_item_rating(user_id, item_id, default_rating=0.0)
            new_value = current + delta

            # Clip the new value within the defined bounds
            new_value = max(self.min_value, min(new_value, self.max_value))

            # Store the updated value with the current timestamp
            self.interactions[user_id][item_id] = (new_value, tstamp)
        # Track all unique item IDs
        self.all_item_ids.add(item_id)
        # Update the hot items cache
        if delta > 0:
            self.hot_items.add(item_id)
        # Update maximum user and item IDs
        self.max_user_id = max(self.max_user_id, user_id)
        self.max_item_id = max(self.max_item_id, item_id)

    def get_user_item_rating(self, user_id: int, item_id: int, default_rating: float = 0.0) -> float:
        """
        Retrieves the interaction count for a specific user-item pair, applying decay if necessary.

        Args:
            user_id (int): ID of the user.
            item_id (int): ID of the item.
            default_rating (float): Default rating to return if no interaction exists (default is 0.0).

        Returns:
            float: The decayed interaction value for the specified user-item pair.
        """
        current, last_timestamp = self.interactions[user_id].get(item_id, (default_rating, 0.0))
        if current == default_rating:
            return default_rating  # Return default if no interaction exists
        return self._apply_decay(current, last_timestamp)

    def get_user_items(self, user_id: int, n_recent: Optional[int] = None) -> List[int]:
        """
        Retrieves the dictionary of item IDs and their interaction counts for a given user,
        applying decay to each interaction.

        Args:
            user_id (int): ID of the user.
            n_recent (Optional[int]): Number of most recent items to consider (default is None).

        Returns:
            List[int]: List of item IDs that the user has interacted with.
        """
        user_interactions = self.interactions.get(user_id)
        if user_interactions is None:
            return []

        # Use top-k recent items for the user
        if n_recent is not None and len(self.interactions) > n_recent:
            # Sort by timestamp in descending order
            sorted_items = sorted(
                user_interactions.items(), key=lambda x: x[1][1], reverse=True
            )
            return [item_id for item_id, _ in sorted_items[:n_recent]]
        else:
            return list(user_interactions.keys())

    def get_all_item_ids(self) -> List[int]:
        """
        Retrieves a list of all unique item IDs.

        Returns:
            List[int]: List of unique item IDs.
        """
        return list(self.all_item_ids)

    def get_all_users(self) -> List[int]:
        """
        Retrieves a list of all user IDs.

        Returns:
            List[int]: List of user IDs.
        """
        return list(self.interactions.keys())

    def get_all_non_interacted_items(self, user_id: int) -> List[int]:
        """
        Retrieves a list of all items a user has not interacted with.

        Args:
            user_id (int): ID of the user.

        Returns:
            List[int]: List of item IDs the user has not interacted with.
        """
        interacted_items = self.get_user_items(user_id)
        if len(interacted_items) == 0:
            return list(self.all_item_ids)
        return list(self.all_item_ids.difference(interacted_items))

    def get_all_non_negative_items(self, user_id: int) -> List[int]:
        """
        Retrieves a list of all items with non-negative interaction counts, applying decay to each interaction.

        Args:
            user_id (int): ID of the user.

        Returns:
            List[int]: List of item IDs with non-negative interaction counts.
        """
        # Return all items with non-negative interaction counts after applying decay
        return [item_id for item_id in self.all_item_ids
                if self.get_user_item_rating(user_id, item_id, default_rating=0.0) >= 0.0]

    def get_hot_items(self, n: int, user_id: Optional[int]=None, filter_interacted: bool = True) -> List[int]:
        """
        Retrieves the top N most interacted items.

        Args:
            n (int): Number of items to retrieve.

        Returns:
            List[int]: List of item IDs of the top N most interacted items.
        """
        interacted_items = []
        if filter_interacted:
            assert user_id is not None, "User ID must be provided to filter interacted items."
            interacted_items = self.get_user_items(user_id)
        return list(self.hot_items.get_freq_items(n, exclude_items=interacted_items))

    def to_csr(self, select_users: List[int] = None, include_weights: bool = True) -> csr_matrix:
        rows, cols = [], []

        if include_weights:
            data = []
            if select_users:
                for user in select_users:
                    for item, (rating, tstamp) in self.interactions.get(user, {}).items():
                        rows.append(user)
                        cols.append(item)
                        data.append(self._apply_decay(rating, tstamp))
            else:
                for user, inner_dict in self.interactions.items():
                    for item, (rating, tstamp) in inner_dict.items():
                        rows.append(user)
                        cols.append(item)
                        data.append(self._apply_decay(rating, tstamp))
            return csr_matrix((data, (rows, cols)), shape=(self.max_user_id + 1, self.max_item_id + 1), dtype="float32")
        else:
            if select_users:
                for user in select_users:
                    for item in self.interactions.get(user, {}):
                        rows.append(user)
                        cols.append(item)
            else:
                for user, inner_dict in self.interactions.items():
                    for item in inner_dict:
                        rows.append(user)
                        cols.append(item)
            data = np.ones(len(rows), dtype="int32")
            return csr_matrix((data, (rows, cols)), shape=(self.max_user_id + 1, self.max_item_id + 1), dtype="int32")

    def to_csc(self, select_items: List[int] = None) -> csc_matrix:
        rows, cols, data = [], [], []

        for user, inner_dict in self.interactions.items():
            for item, (rating, tstamp) in inner_dict.items():
                if select_items is not None and item not in select_items:
                    continue
                rows.append(user)
                cols.append(item)
                data.append(self._apply_decay(rating, tstamp))

        # Create the csc_matrix
        return csc_matrix((data, (rows, cols)), shape=(self.max_user_id + 1, self.max_item_id + 1), dtype="float32")

    def to_coo(self, select_users: List[int] = None, select_items: List[int] = None) -> coo_matrix:
        rows, cols, data = [], [], []

        if select_users is None:
            if select_items is None:
                for user, inner_dict in self.interactions.items():
                    for item, (rating, tstamp) in inner_dict.items():
                        rows.append(user)
                        cols.append(item)
                        data.append(self._apply_decay(rating, tstamp))
            else:
                for user, inner_dict in self.interactions.items():
                    for item in select_items:
                        if item not in inner_dict:
                            continue
                        rating, tstamp = inner_dict[item]
                        rows.append(user)
                        cols.append(item)
                        data.append(self._apply_decay(rating, tstamp))
        else:
            if select_items is None:
                for user in select_users:
                    for item, (rating, tstamp) in self.interactions.get(user, {}).items():
                        rows.append(user)
                        cols.append(item)
                        data.append(self._apply_decay(rating, tstamp))
            else:
                for user in select_users:
                    inner_dict = self.interactions.get(user, {})
                    for item in select_items:
                        if item not in inner_dict:
                            continue
                        rating, tstamp = inner_dict[item]
                        rows.append(user)
                        cols.append(item)
                        data.append(self._apply_decay(rating, tstamp))

        # Create the coo_matrix
        return coo_matrix((data, (rows, cols)), shape=(self.max_user_id + 1, self.max_item_id + 1), dtype="float32")

    @property
    def shape(self) -> tuple[int, int]:
        """
        Retrieves the shape of the interaction matrix.

        Returns:
            tuple[int, int]: The shape of the interaction matrix of the form (n_users, n_items).
        """
        return self.max_user_id + 1, self.max_item_id + 1
