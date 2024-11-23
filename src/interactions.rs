use std::time::SystemTime;
use std::f32::consts::E;
use chrono::{TimeZone, Utc};
use hashbrown::{HashMap, HashSet};
use log::warn;
use rayon::slice::ParallelSliceMut;
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct UserItemInteractions {
    interactions: HashMap<i32, HashMap<i32, (f32, f32)>>, // Store interaction value and timestamp
    all_item_ids: HashSet<i32>,
    min_value: f32,
    max_value: f32,
    decay_rate: Option<f32>, // Optional decay rate
    max_timestamp: f32,
}

impl UserItemInteractions {
    pub fn new(min_value: f32, max_value: f32, decay_in_days: Option<f32>) -> Self {
        assert!(max_value > min_value, "max_value should be greater than min_value");

        let decay_rate = decay_in_days.map(|days| 1.0 - (E.ln() / days));

        UserItemInteractions {
            interactions: HashMap::with_capacity(8192),
            all_item_ids: HashSet::with_capacity(8192),
            min_value,
            max_value,
            decay_rate,
            max_timestamp: 0.0,
        }
    }

    pub fn get_decay_rate(&self) -> Option<f32> {
        self.decay_rate
    }

    pub fn set_decay_rate(&mut self, decay_rate: Option<f32>) {
        self.decay_rate = decay_rate;
    }

    /// Apply exponential decay to the value based on the elapsed time since the last interaction.
    /// Half-life decay is used to calculate the decay rate.
    /// Reference: https://dl.acm.org/doi/10.1145/1099554.1099689
    #[inline]
    fn _apply_decay(&self, value: f32, last_timestamp: f32) -> f32 {
        if let Some(decay_rate) = self.decay_rate {
            // Calculate elapsed time since the last interaction
            let elapsed_secs = self.max_timestamp - last_timestamp;
            let elapsed_days = elapsed_secs / 86400.0;
            return value * decay_rate.powf(elapsed_days); // Apply exponential decay
        }
        value // Return the original value if decay rate is not set
    }

    pub fn add_interaction(&mut self, user_id: i32, item_id: i32, tstamp: f32, delta: f32, upsert: bool) {
        let current_unix_time = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap().as_secs_f32();
        if tstamp > current_unix_time + 180.0 {
            // add some buffer (180 secs) as system clock may not be in sync with the server
            warn!("Timestamp {} is in the future. Current time is {}", fmt_unix_time(tstamp), fmt_unix_time(current_unix_time));
        }
        self.max_timestamp = self.max_timestamp.max(tstamp + 1.0); // Add 1 second to avoid timestamp conflicts
        let new_value = if upsert {
            delta
        } else {
            // Get the current rating for the user-item pair, applying decay if necessary
            let current_value = self.get_user_item_rating(user_id, item_id, 0.0);
            // Calculate the new value by adding the delta
            (current_value + delta).clamp(self.min_value, self.max_value)
        };
        self.interactions.entry(user_id).or_default().insert(item_id, (new_value, tstamp));

        // Track all unique item IDs
        self.all_item_ids.insert(item_id);
    }

    #[inline]
    pub fn get_user_item_rating(&self, user_id: i32, item_id: i32, default_rating: f32) -> f32 {
        if let Some(item_map) = self.interactions.get(&user_id) {
            if let Some(&(current_value, last_timestamp)) = item_map.get(&item_id) {
                return self._apply_decay(current_value, last_timestamp);
            }
        }
        default_rating // Return default rating if no interaction exists
    }

    /// Return the n most recent items for the user, or all items if n_recent is None
    pub fn get_user_items(&self, user_id: i32, n_recent: Option<usize>) -> Vec<i32> {
        if let Some(item_map) = self.interactions.get(&user_id) {
            if let Some(n) = n_recent {
                let capacity = item_map.len();
                let mut items: Vec<_> = Vec::with_capacity(capacity);
                items.extend(item_map.iter());
                if capacity > 20 {
                    items.par_sort_unstable_by(|&(_, &(_, ts1)), &(_, &(_, ts2))| ts2.partial_cmp(&ts1).unwrap());
                } else {
                    items.sort_unstable_by(|&(_, &(_, ts1)), &(_, &(_, ts2))| ts2.partial_cmp(&ts1).unwrap());
                }
                items.iter().take(n).map(|(&item_id, _)| item_id).collect()
            } else {
                item_map.keys().copied().collect()
            }
        } else {
            Vec::new()
        }
    }

    #[inline]
    pub fn get_all_item_ids(&self) -> Vec<i32> {
        self.all_item_ids.iter().copied().collect()
    }

    #[inline]
    pub fn get_all_users(&self) -> Vec<i32> {
        self.interactions.keys().copied().collect()
    }

    #[inline]
    pub fn get_all_non_interacted_items(&self, user_id: i32) -> Vec<i32> {
        let interacted_items: HashSet<i32> = self.get_user_items(user_id, None).into_iter().collect();
        self.all_item_ids.difference(&interacted_items).copied().collect()
    }

    #[inline]
    pub fn get_all_non_negative_items(&self, user_id: i32) -> Vec<i32> {
        self.all_item_ids.iter()
            .filter(|&&item_id| self.get_user_item_rating(user_id, item_id, 0.0) >= 0.0)
            .copied()
            .collect()
    }
}

#[inline]
fn fmt_unix_time(timestamp: f32) -> String {
    // Convert to integer and fractional seconds
    let secs = timestamp as i64;
    let nanos = ((timestamp - secs as f32) * 1_000_000_000.0) as u32;

    // Create DateTime from seconds and nanoseconds
    if let Some(datetime) = Utc.timestamp_opt(secs, nanos).single() {
        // Return the formatted datetime string
        datetime.to_rfc3339()
    } else {
        // Handle invalid timestamp
        "Invalid timestamp".to_string()
    }
}