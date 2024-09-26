use std::collections::{HashMap, HashSet};

pub struct UserItemInteractions {
    interactions: HashMap<i32, HashMap<i32, f32>>,
    all_item_ids: HashSet<i32>,
}

impl UserItemInteractions {
    pub fn new() -> Self {
        UserItemInteractions {
            interactions: HashMap::new(),
            all_item_ids: HashSet::new(),
        }
    }

    pub fn add_interaction(&mut self, user_id: i32, item_id: i32, rating: f32) {
        self.interactions
            .entry(user_id)
            .or_default()
            .entry(item_id)
            .and_modify(|e| *e += rating)
            .or_insert(rating);

        // Keep track of all unique item IDs
        self.all_item_ids.insert(item_id);
    }

    pub fn get_user_item_count(&self, user_id: i32, item_id: i32) -> f32 {
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
            .filter(|&&item_id| self.get_user_item_count(user_id, item_id) > 0.0)
            .cloned()
            .collect()
    }

}