use std::collections::{HashMap, HashSet};
use std::time::SystemTime;
use std::f32::consts::E;
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct UserItemInteractions {
    interactions: HashMap<i32, HashMap<i32, (f32, f32)>>, // Store interaction value and timestamp
    all_item_ids: HashSet<i32>,
    min_value: f32,
    max_value: f32,
    decay_rate: Option<f32>, // Optional decay rate
}

impl UserItemInteractions {
    pub fn new(min_value: f32, max_value: f32, decay_in_days: Option<f32>) -> Self {
        assert!(max_value > min_value, "max_value should be greater than min_value");

        let decay_rate = decay_in_days.map(|days| 1.0 - (E.ln() / days));

        UserItemInteractions {
            interactions: HashMap::new(),
            all_item_ids: HashSet::new(),
            min_value,
            max_value,
            decay_rate,
        }
    }

    fn _apply_decay(&self, value: f32, last_timestamp: f32) -> f32 {
        if let Some(decay_rate) = self.decay_rate {
            // Calculate elapsed time since the last interaction
            let elapsed_secs = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap().as_secs() as f32 - last_timestamp;
            let elapsed_days = elapsed_secs / 86400.0;
            return value * decay_rate.powf(elapsed_days); // Apply exponential decay
        }
        value // Return original value if no decay rate is set
    }

    pub fn add_interaction(&mut self, user_id: i32, item_id: i32, tstamp: f32, delta: f32) {
        // Get the current rating for the user-item pair, applying decay if necessary
        let current_value = self.get_user_item_rating(user_id, item_id, 0.0);

        // Calculate the new value by adding the delta
        let new_value = (current_value + delta).clamp(self.min_value, self.max_value);

        // Store the updated value with the current timestamp
        self.interactions
            .entry(user_id)
            .or_insert_with(HashMap::new)
            .insert(item_id, (new_value, tstamp)); // Update the timestamp

        // Track all unique item IDs
        self.all_item_ids.insert(item_id);
    }

    pub fn get_user_item_rating(&self, user_id: i32, item_id: i32, default_rating: f32) -> f32 {
        if let Some(item_map) = self.interactions.get(&user_id) {
            if let Some(&(current_value, last_timestamp)) = item_map.get(&item_id) {
                return self._apply_decay(current_value, last_timestamp);
            }
        }
        default_rating // Return default rating if no interaction exists
    }

    pub fn get_user_items(&self, user_id: i32) -> HashMap<i32, f32> {
        self.interactions.get(&user_id)
            .map(|item_map| {
                item_map.iter()
                    .map(|(&item_id, &(value, timestamp))| {
                        (item_id, self._apply_decay(value, timestamp))
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    pub fn get_all_item_ids(&self) -> Vec<i32> {
        self.all_item_ids.iter().cloned().collect()
    }

    pub fn get_all_users(&self) -> Vec<i32> {
        self.interactions.keys().cloned().collect()
    }

    pub fn get_all_items_for_user(&self, user_id: i32) -> Vec<i32> {
        self.get_user_items(user_id).keys().cloned().collect()
    }

    pub fn get_all_non_interacted_items(&self, user_id: i32) -> Vec<i32> {
        let interacted_items: HashSet<_> = self.get_user_items(user_id).keys().cloned().collect();
        self.all_item_ids.difference(&interacted_items).cloned().collect()
    }

    pub fn get_all_non_negative_items(&self, user_id: i32) -> Vec<i32> {
        self.all_item_ids.iter()
            .filter(|&&item_id| self.get_user_item_rating(user_id, item_id, 0.0) >= 0.0)
            .cloned()
            .collect()
    }
}
