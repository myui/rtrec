from collections import defaultdict, Counter
from typing import Dict, List

class UserItemInteractions:
    def __init__(self) -> None:
        self.interactions: defaultdict[int, Counter[float]] = defaultdict(Counter)
        self.empty = Counter()
        self.all_item_ids = set()

    def add_interaction(self, user_id: int, item_id: int, delta: float = 1.0) -> None:
        """
        Add or update an interaction count for a user-item pair.
        If the count becomes 0 or less, remove the interaction.
        """
        current = self.get_user_items(user_id).get(item_id, 0.0)
        self.interactions[user_id][item_id] = current + delta

        self.all_item_ids.add(item_id)

    def get_all_item_ids(self) -> List[int]:
        """
        Get a list of all unique item IDs.
        """
        return list(self.all_item_ids)

    def get_user_items(self, user_id: int) -> Counter[float]:
        """
        Get the dictionary of item_id -> interaction_count for a given user.
        """
        return self.interactions.get(user_id, self.empty)

    def get_user_item_count(self, user_id: int, item_id: int, default_count: float = 0.0) -> float:
        """
        Get the interaction count for a specific user-item pair.
        """
        return self.get_user_items(user_id).get(item_id, default_count)

    def get_all_users(self) -> List[int]:
        """
        Get a list of all users.
        """
        return list(self.interactions.keys())

    def get_all_items_for_user(self, user_id: int) -> List[int]:
        """
        Get a list of all items a user has interacted with.
        """
        return list(self.get_user_items(user_id).keys())

    def get_all_non_interacted_items(self, user_id: int) -> List[int]:
        """
        Get a list of all items a user has not interacted with.
        """
        interacted_items = set(self.get_user_items(user_id).keys())
        return [item_id for item_id in self.all_item_ids if item_id not in interacted_items]

    def get_all_non_negative_items(self, user_id: int) -> List[int]:
        """
        Get a list of all non-negative items.
        """
        interacted_items = set(self.get_user_items(user_id).keys())
        # Return all items with non-negative interaction counts
        return [item_id for item_id in self.all_item_ids if self.get_user_item_count(user_id, item_id, default_count=-1.0) > 0.0]
