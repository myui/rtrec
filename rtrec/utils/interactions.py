from collections import defaultdict
from typing import List, Optional
import time

class UserItemInteractions:
    def __init__(self, min_value: int = -5, max_value: int = 10, decay_rate: Optional[float] = None) -> None:
        """
        Initializes the UserItemInteractions class.

        Args:
            min_value (int): Minimum allowable value for interactions.
            max_value (int): Maximum allowable value for interactions.
            decay_rate (Optional[float]): Rate at which interactions decay over time.
                                          If None, no decay is applied.
        """
        self.interactions: defaultdict[int, dict[int, tuple[float, float]]] = defaultdict(dict)
        self.empty = {}
        self.all_item_ids = set()
        assert max_value > min_value, f"max_value should be greater than min_value {max_value} > {min_value}"
        self.min_value = min_value
        self.max_value = max_value
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

        elapsed_time = time.time() - last_timestamp
        return value * (1.0 - self.decay_rate) ** elapsed_time

    def add_interaction(self, user_id: int, item_id: int, delta: float = 1.0) -> None:
        """
        Adds or updates an interaction count for a user-item pair.

        Args:
            user_id (int): ID of the user.
            item_id (int): ID of the item.
            delta (float): Change in interaction count (default is 1.0).
        """
        current = self.get_user_item_rating(user_id, item_id, default_rating=0.0)
        new_value = current + delta

        # Clip the new value within the defined bounds
        new_value = max(self.min_value, min(new_value, self.max_value))

        # Store the updated value with the current timestamp
        self.interactions[user_id][item_id] = (new_value, time.time())
        self.all_item_ids.add(item_id)

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
        current, last_timestamp = self.interactions[user_id].get(item_id, (default_rating, time.time()))
        if current == default_rating:
            return default_rating  # Return default if no interaction exists
        return self._apply_decay(current, last_timestamp)

    def get_user_items(self, user_id: int) -> dict[int, float]:
        """
        Retrieves the dictionary of item IDs and their interaction counts for a given user,
        applying decay to each interaction.

        Args:
            user_id (int): ID of the user.

        Returns:
            dict[int, float]: Dictionary of item IDs and their decayed interaction values.
        """
        return {item_id: self._apply_decay(value, timestamp)
                for item_id, (value, timestamp) in self.interactions.get(user_id, self.empty).items()}

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

    def get_all_items_for_user(self, user_id: int) -> List[int]:
        """
        Retrieves a list of all items a user has interacted with, applying decay to each interaction.

        Args:
            user_id (int): ID of the user.

        Returns:
            List[int]: List of item IDs that the user has interacted with.
        """
        return list(self.get_user_items(user_id).keys())

    def get_all_non_interacted_items(self, user_id: int) -> List[int]:
        """
        Retrieves a list of all items a user has not interacted with.

        Args:
            user_id (int): ID of the user.

        Returns:
            List[int]: List of item IDs the user has not interacted with.
        """
        interacted_items = set(self.get_user_items(user_id).keys())
        return [item_id for item_id in self.all_item_ids if item_id not in interacted_items]

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
