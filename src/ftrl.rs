use serde::{Serialize, Deserialize};
use hashbrown::HashMap;

/// FTRL (Follow The Regularized Leader) optimizer implementation.
/// This is a variant of FTRL-Proximal algorithm for online learning.
///
/// Reference:
/// - https://www.eecs.tufts.edu/~dsculley/papers/ad-click-prediction.pdf
/// - https://arxiv.org/abs/1403.3465
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct FTRL {
    alpha: f32,
    beta: f32,
    lambda1: f32,
    lambda2: f32,
    z: HashMap<(i32, i32), f32>,
    n: HashMap<(i32, i32), f32>,
    /// similarity matrix of (target_item_id, base_item_id) -> coefficient
    weights: HashMap<(i32, i32), f32>,
}

impl FTRL {
    pub fn new(alpha: f32, beta: f32, lambda1: f32, lambda2: f32) -> Self {
        FTRL {
            alpha,
            beta,
            lambda1,
            lambda2,
            z: HashMap::new(),
            n: HashMap::new(),
            weights: HashMap::new(),
        }
    }

    pub fn get_weights(&self) -> &HashMap<(i32, i32), f32> {
        &self.weights
    }

    #[allow(dead_code)]
    #[deprecated(note = "Use `vectorized_update_gradients` instead")]
    pub fn update_gradients(&mut self, key: (i32, i32), grad: f32) -> f32 {
        let z_val = self.z.get(&key).cloned().unwrap_or_default();
        let n_val = self.n.get(&key).cloned().unwrap_or_default();

        let n_new = n_val + grad.powi(2);
        let sigma = (n_new.sqrt() - n_val.sqrt()) / self.alpha;
        let z_new = z_val + grad - sigma * self.weights.get(&key).cloned().unwrap_or_default();

        if z_new.abs() <= self.lambda1 {
            self.weights.remove(&key);
            return 0.0;
        }

        self.z.insert(key, z_new);
        self.n.insert(key, n_new);

        let weight_update = -(z_new - z_new.signum() * self.lambda1) / ((self.beta + n_new.sqrt()) / self.alpha + self.lambda2);

        if weight_update.abs() < 1e-8 {
            self.weights.remove(&key);
            return 0.0;
        }

        self.weights.insert(key, weight_update);
        weight_update
    }

    pub fn vectorized_update_gradients(&mut self, item_id: i32, updates: &[(i32, f32)]) {
        // Preallocate collections to store updates
        let size: usize = updates.len();
        let mut z_updates = Vec::with_capacity(size);
        let mut n_updates = Vec::with_capacity(size);
        let mut weight_updates = Vec::with_capacity(size);

        // Record each update
        for &(ui, grad) in updates {
            let key = (ui, item_id);
            let z_val = self.z.get(&key).cloned().unwrap_or_default();
            let n_val = self.n.get(&key).cloned().unwrap_or_default();

            let n_new = n_val + grad.powi(2);
            let sigma = (n_new.sqrt() - n_val.sqrt()) / self.alpha;
            let z_new = z_val + grad - sigma * self.weights.get(&key).cloned().unwrap_or_default();

            if z_new.abs() > self.lambda1 {
                z_updates.push((key, z_new));
                n_updates.push((key, n_new));

                let weight_update = -(z_new - z_new.signum() * self.lambda1)
                    / ((self.beta + n_new.sqrt()) / self.alpha + self.lambda2);
                if weight_update.abs() >= 1e-8 {
                    weight_updates.push((key, weight_update));
                } else {
                    self.weights.remove(&key);
                }
            } else {
                self.weights.remove(&key);
            }
        }

        // Apply the updates in bulk
        for (key, z_new) in z_updates {
            self.z.insert(key, z_new);
        }
        for (key, n_new) in n_updates {
            self.n.insert(key, n_new);
        }
        for (key, weight_update) in weight_updates {
            self.weights.insert(key, weight_update);
        }
    }

}