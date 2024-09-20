use std::collections::HashMap;

pub struct UserItemInteractions {
    interactions: HashMap<i32, HashMap<i32, f32>>,
    max_item_id: i32,
}

impl UserItemInteractions {
    pub fn new() -> Self {
        UserItemInteractions {
            interactions: HashMap::new(),
            max_item_id: 0,
        }
    }

    pub fn max_item_id(&self) -> i32 {
        self.max_item_id
    }

    pub fn add_interaction(&mut self, user_id: i32, item_id: i32, rating: f32) {
        self.interactions
            .entry(user_id)
            .or_default()
            .entry(item_id)
            .and_modify(|e| *e += rating)
            .or_insert(rating);
        self.max_item_id = self.max_item_id.max(item_id);
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
        let interacted_items = self.get_all_items_for_user(user_id);
        let interacted_items_set: std::collections::HashSet<_> = interacted_items.into_iter().collect();

        (0..=self.max_item_id)
            .filter(|&item_id| !interacted_items_set.contains(&item_id))
            .collect()
    }

    pub fn get_all_non_negative_items(&self, user_id: i32) -> Vec<i32> {
        (0..=self.max_item_id)
            .filter(|&item_id| {
                self.get_user_item_count(user_id, item_id) >= 0.0
            })
            .collect()
    }

}