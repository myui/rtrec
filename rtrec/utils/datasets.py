from collections import defaultdict, Counter
from typing import Dict, List

class UserItemInteractions:
    def __init__(self) -> None:
        # defaultdict of Counters to store user interactions
        self.interactions: defaultdict[int, Counter[int]] = defaultdict(Counter)
        self.empty = Counter()

    def add_interaction(self, user_id: int, item_id: int, count: int = 1) -> None:
        """
        Add or update an interaction count for a user-item pair.
        If the count becomes 0 or less, remove the interaction.
        """
        current_count = self.get_user_items(user_id).get(item_id, 0)
        new_count = current_count + count

        if new_count > 0:
            self.interactions[user_id][item_id] = new_count
        else:
            if user_id in self.interactions:
                # Remove the item if the count is zero or less
                del self.interactions[user_id][item_id]
                # Remove the user if they have no more interactions
                del self.interactions[user_id]

    def get_user_items(self, user_id: int) -> Counter[int]:
        """
        Get the dictionary of item_id -> interaction_count for a given user.
        """
        return self.interactions.get(user_id, self.empty)

    def get_user_item_count(self, user_id: int, item_id: int) -> int:
        """
        Get the interaction count for a specific user-item pair.
        """
        return self.get_user_items(user_id).get(item_id, 0)

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
