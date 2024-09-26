use std::collections::{HashMap, HashSet};

pub struct UserItemInteractions {
    interactions: HashMap<i32, HashMap<i32, f32>>,
    all_item_ids: HashSet<i32>,
    min_value: f32,
    max_value: f32,
}

impl UserItemInteractions {
    pub fn new(min_value: f32, max_value: f32) -> Self {
        assert!(max_value > min_value, "max_value should be greater than min_value");

        UserItemInteractions {
            interactions: HashMap::new(),
            all_item_ids: HashSet::new(),
            min_value,
            max_value,
        }
    }

    pub fn add_interaction(&mut self, user_id: i32, item_id: i32, delta: f32) {
        // Use the entry API to get a mutable reference to the user-item interactions
        let user_items = self.interactions.entry(user_id).or_insert_with(HashMap::new);

        // Update the interaction value, clamping it within min_value and max_value
        let new_value = user_items
            .entry(item_id)
            .and_modify(|e| *e += delta)
            .or_insert(delta); // Start with delta if the entry doesn't exist

        // Clamp the new value between min_value and max_value
        *new_value = new_value.clamp(self.min_value, self.max_value);

        // Keep track of all unique item IDs
        self.all_item_ids.insert(item_id);
    }

    pub fn get_user_item_rating(&self, user_id: i32, item_id: i32) -> f32 {
        self.interactions
            .get(&user_id)
            .and_then(|item_map| item_map.get(&item_id))
            .copied()
            .unwrap_or_default()
    }

    pub fn get_all_items_for_user(&self, user_id: i32) -> Vec<i32> {
        self.interactions
            .get(&user_id)
            .map(|item_map| item_map.keys().cloned().collect())
            .unwrap_or_default()
    }

    pub fn get_all_non_interacted_items(&self, user_id: i32) -> Vec<i32> {
        let interacted_items: HashSet<_> = self.get_all_items_for_user(user_id).into_iter().collect();
        self.all_item_ids.difference(&interacted_items).cloned().collect()
    }

    pub fn get_all_non_negative_items(&self, user_id: i32) -> Vec<i32> {
        self.all_item_ids.iter()
            .filter(|&&item_id| self.get_user_item_rating(user_id, item_id) > 0.0)
            .cloned()
            .collect()
    }

}