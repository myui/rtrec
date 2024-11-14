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

    /// Apply exponential decay to the value based on the elapsed time since the last interaction.
    /// Half-life decay is used to calculate the decay rate.
    /// Reference: https://dl.acm.org/doi/10.1145/1099554.1099689
    fn _apply_decay(&self, value: f32, last_timestamp: f32) -> f32 {
        if let Some(decay_rate) = self.decay_rate {
            // Calculate elapsed time since the last interaction
            let elapsed_secs = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap().as_secs_f32() - last_timestamp;
            let elapsed_days = elapsed_secs / 86400.0;
            return value * decay_rate.powf(elapsed_days); // Apply exponential decay
        }
        value // Return the original value if decay rate is not set
    }

    pub fn add_interaction(&mut self, user_id: i32, item_id: i32, tstamp: f32, delta: f32, upsert: bool) {
        if upsert {
            self.interactions
                .entry(user_id)
                .or_insert_with(HashMap::new)
                .insert(item_id, (delta, tstamp)); // Update the timestamp
        } else {
            // Get the current rating for the user-item pair, applying decay if necessary
            let current_value = self.get_user_item_rating(user_id, item_id, 0.0);

            // Calculate the new value by adding the delta
            let new_value = (current_value + delta).clamp(self.min_value, self.max_value);

            // Store the updated value with the current timestamp
            self.interactions
                .entry(user_id)
                .or_insert_with(HashMap::new)
                .insert(item_id, (new_value, tstamp)); // Update the timestamp
        }

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

    pub fn get_user_items(&self, user_id: i32, n_recent: Option<usize>) -> HashMap<i32, f32> {
        // Get the most recent n_recent items interacted by the user
        // If n_recent is None, return all items interacted by the user
        if let Some(n_recent) = n_recent {
            // Get all items, sort by timestamp, and select the n most recent ones
            self.interactions.get(&user_id).map(|items| {
                let mut sorted_items = items.iter().collect::<Vec<_>>();
                // Sort by timestamp in descending order
                sorted_items.sort_by(|&(_, &(_, ts1)), &(_, &(_, ts2))| ts2.partial_cmp(&ts1).unwrap());

                // Apply decay to the top n items and collect them into a HashMap
                sorted_items.iter()
                    .take(n_recent)
                    .map(|(&item_id, &(value, timestamp))| {
                        (item_id, self._apply_decay(value, timestamp))
                    })
                    .collect()
            }).unwrap_or_default()
        } else {
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
    }

    pub fn get_all_item_ids(&self) -> Vec<i32> {
        self.all_item_ids.iter().cloned().collect()
    }

    pub fn get_all_users(&self) -> Vec<i32> {
        self.interactions.keys().cloned().collect()
    }

    pub fn get_all_items_for_user(&self, user_id: i32, n_recent: Option<usize>) -> Vec<i32> {
        self.get_user_items(user_id, n_recent).keys().cloned().collect()
    }

    pub fn get_all_non_interacted_items(&self, user_id: i32) -> Vec<i32> {
        let interacted_items: HashSet<_> = self.get_user_items(user_id, None).keys().cloned().collect();
        self.all_item_ids.difference(&interacted_items).cloned().collect()
    }

    pub fn get_all_non_negative_items(&self, user_id: i32) -> Vec<i32> {
        self.all_item_ids.iter()
            .filter(|&&item_id| self.get_user_item_rating(user_id, item_id, 0.0) >= 0.0)
            .cloned()
            .collect()
    }
}
