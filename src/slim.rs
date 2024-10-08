use pyo3::prelude::*;

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use serde::{Serialize, Deserialize};

use crate::ftrl::FTRL;
use crate::interactions::UserItemInteractions;

#[pyclass]
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SlimMSE {
    interactions: UserItemInteractions,
    ftrl: FTRL,
    weights: HashMap<(i32, i32), f32>, // Direct reference to FTRL's weights
    cumulative_loss: f32,
    steps: usize,
}

#[pymethods]
impl SlimMSE {
    #[new]
    #[pyo3(signature = (alpha = 0.5, beta = 1.0, lambda1 = 0.0002, lambda2 = 0.0001, min_value = -5.0, max_value = 10.0, decay_in_days = None))]
    pub fn new(alpha: f32, beta: f32, lambda1: f32, lambda2: f32, min_value: f32, max_value: f32, decay_in_days: Option<f32>) -> Self {
        let ftrl = FTRL::new(alpha, beta, lambda1, lambda2);
        let weights = ftrl.get_weights().clone(); // Get the weights reference

        SlimMSE {
            interactions: UserItemInteractions::new(min_value, max_value, decay_in_days),
            ftrl,
            weights,
            cumulative_loss: 0.0,
            steps: 0,
        }
    }

    pub fn fit(&mut self, user_interactions: Vec<(i32, i32, f32, f32)>) {
        for (user_id, item_id, tstamp, rating) in user_interactions {
            self.interactions.add_interaction(user_id, item_id, tstamp, rating);
            self.update_weights(user_id, item_id);
        }
    }

    fn update_weights(&mut self, user_id: i32, item_id: i32) {
        let user_items = self.interactions.get_all_items_for_user(user_id);
        let predicted = self._predict_rating(user_id, item_id);
        let dloss = predicted - self.interactions.get_user_item_rating(user_id, item_id, 0.0);

        self.cumulative_loss += dloss.powi(2);
        self.steps += 1;

        for &ui in &user_items {
            if ui != item_id {
                let grad = dloss * self.interactions.get_user_item_rating(user_id, ui, 0.0);
                self.ftrl.update_gradients((ui, item_id), grad);
            }
        }
    }

    fn _predict_rating(&self, user_id: i32, item_id: i32) -> f32 {
        let user_items = self.interactions.get_all_items_for_user(user_id);
        user_items.iter()
            .map(|&ui| self.weights.get(&(ui, item_id)).unwrap_or(&0.0) * self.interactions.get_user_item_rating(user_id, ui, 0.0))
            .sum()
    }

    pub fn recommend(&self, user_id: i32, top_k: usize, filter_interacted: Option<bool>) -> Vec<i32> {
        // Use `unwrap_or` to set `filter_interacted` to `true` by default
        let filter_interacted = filter_interacted.unwrap_or(true);

        // Get the candidate items based on the filtering condition
        let candidate_items = if filter_interacted {
            self.interactions.get_all_non_interacted_items(user_id)
        } else {
            self.interactions.get_all_non_negative_items(user_id)
        };

        // Predict scores for the candidate items
        let mut scores: Vec<(i32, f32)> = candidate_items
            .iter()
            .map(|&item_id| {
                let score = self._predict_rating(user_id, item_id);
                (item_id, score)
            })
            .collect();

        // Sort items by score in descending order
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Take the top-k items and return their IDs
        scores.iter().take(top_k).map(|&(id, _)| id).collect()
    }

    pub fn get_empirical_loss(&self) -> f32 {
        if self.steps == 0 {
            0.0
        } else {
            self.cumulative_loss / self.steps as f32
        }
    }

    /// Save the SlimMSE model to a file using MessagePack.
    pub fn save(&self, file_path: &str) -> PyResult<()> {
        let file = File::create(file_path).map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("Failed to create file: {}", e)))?;
        let mut writer = BufWriter::new(file);

        // Serialize the SlimMSE struct to MessagePack
        rmp_serde::encode::write(&mut writer, &self).map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("Failed to serialize: {}", e)))?;
        Ok(())
    }

    /// Load the SlimMSE model from a file using MessagePack.
    #[staticmethod]
    pub fn load(file_path: &str) -> PyResult<Self> {
        let file = File::open(file_path).map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("Failed to open file: {}", e)))?;
        let reader = BufReader::new(file);

        // Deserialize the SlimMSE struct from MessagePack
        let slim: SlimMSE = rmp_serde::decode::from_read(reader).map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("Failed to deserialize: {}", e)))?;
        Ok(slim)
    }

}
