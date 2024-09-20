use pyo3::prelude::*;

use std::collections::HashMap;

use crate::ftrl::FTRL;
use crate::datasets::UserItemInteractions;

#[pyclass]
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
    pub fn new(alpha: f32, beta: f32, lambda1: f32, lambda2: f32) -> Self {
        let ftrl = FTRL::new(alpha, beta, lambda1, lambda2);
        let weights = ftrl.get_weights().clone(); // Get the weights reference

        SlimMSE {
            interactions: UserItemInteractions::new(),
            ftrl,
            weights,
            cumulative_loss: 0.0,
            steps: 0,
        }
    }

    pub fn fit(&mut self, user_interactions: Vec<(i32, i32, f32)>) {
        for (user_id, item_id, rating) in user_interactions {
            self.interactions.add_interaction(user_id, item_id, rating);
            self.update_weights(user_id, item_id, rating);
        }
    }

    fn update_weights(&mut self, user_id: i32, item_id: i32, rating: f32) {
        let user_items = self.interactions.get_all_items_for_user(user_id);
        let predicted = self._predict_rating(user_id, item_id);
        let dloss = predicted - rating;

        self.cumulative_loss += dloss.powi(2);
        self.steps += 1;

        for &ui in &user_items {
            if ui != item_id {
                let grad = dloss * self.interactions.get_user_item_count(user_id, ui);
                self.ftrl.update_gradients((ui, item_id), grad);
            }
        }
    }

    fn _predict_rating(&self, user_id: i32, item_id: i32) -> f32 {
        let user_items = self.interactions.get_all_items_for_user(user_id);
        user_items.iter()
            .map(|&ui| self.weights.get(&(ui, item_id)).unwrap_or(&0.0) * self.interactions.get_user_item_count(user_id, ui))
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
}

