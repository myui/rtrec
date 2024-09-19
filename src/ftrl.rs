use std::collections::HashMap;

pub struct FTRL {
    alpha: f32,
    beta: f32,
    lambda1: f32,
    lambda2: f32,
    z: HashMap<(i32, i32), f32>,
    n: HashMap<(i32, i32), f32>,
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
}