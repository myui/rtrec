use std::{fmt::Debug, sync::LazyLock};

use log::debug;
use hashbrown::HashMap;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use serde_flexitos::{MapRegistry, Registry, serialize_trait_object};

pub trait Optimizer : erased_serde::Serialize + Debug + Send + Sync {
    // Gets the ID uniquely identifying the concrete type of this value.
    fn id(&self) -> &'static str;
    fn get_weights(&self) -> &HashMap<i64, f32>;
    fn update_gradients(&mut self, ui: i32, item_id: i32, grad: f32, step: u32);
}

#[derive(Serialize, Deserialize, Debug)]
pub struct OptimizerObject {
    pub name: String,
    pub instance: Box<dyn Optimizer>,
}

impl OptimizerObject {
    pub fn new(name: String, instance: Box<dyn Optimizer>) -> Self {
        OptimizerObject {
            name,
            instance,
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SGD {
    weights: HashMap<i64, f32>,
    alpha: f32, // learning rate
    power_t: f32, // power_t
    lambda1: f32, // L1 regularization
    lambda2: f32, // L2 regularization
}

impl SGD {
    const ID: &'static str = "SGD";
    pub fn new(alpha: f32, power_t: f32, lambda1: f32, lambda2: f32) -> Self {
        SGD {
            weights: HashMap::with_capacity(8192),
            alpha,
            power_t,
            lambda1,
            lambda2,
        }
    }
}

impl Optimizer for SGD {

    fn id(&self) -> &'static str {
        Self::ID
    }

    #[inline]
    fn get_weights(&self) -> &HashMap<i64, f32> {
        &self.weights
    }

    fn update_gradients(&mut self, ui: i32, item_id: i32, grad: f32, _step: u32) {
        let key = compound_key(ui, item_id);
        let eta = invoscaling_eta(self.alpha, _step, self.power_t);
        let new_weight = self.weights.entry(key).and_modify(|e| *e -= eta * grad).or_insert(-eta * grad);
        if new_weight.abs() <= 1e-7 {
            self.weights.remove(&key);
        } else {
            let l1_penalty = self.lambda1 * new_weight.signum();
            let l2_penalty = self.lambda2 * *new_weight;
            *new_weight -= self.alpha * (l1_penalty + l2_penalty);
        }
    }

}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct AdaGrad {
    weights: HashMap<i64, f32>,
    sum_sq_grad: HashMap<i64, f32>,
    alpha: f32, // initial learning rate
    lambda1: f32, // L1 regularization
    lambda2: f32, // L2 regularization
}

impl AdaGrad {
    const ID: &'static str = "AdaGrad";

    pub fn new(alpha: f32, lambda1: f32, lambda2: f32) -> Self {
        AdaGrad {
            weights: HashMap::with_capacity(8192),
            sum_sq_grad: HashMap::with_capacity(8192),
            alpha,
            lambda1,
            lambda2,
        }
    }
}

impl Optimizer for AdaGrad {

    fn id(&self) -> &'static str {
        Self::ID
    }

    #[inline]
    fn get_weights(&self) -> &HashMap<i64, f32> {
        &self.weights
    }

    fn update_gradients(&mut self, ui: i32, item_id: i32, grad: f32, _step: u32) {
        let key = compound_key(ui, item_id);

        let gg = grad.powi(2).clamp(1e-8, 1e8); // clip squared gradient
        let sum_gg =* self.sum_sq_grad.entry(key).and_modify(|e| *e += gg).or_insert(gg);

        let eta = self.alpha / (1.0 + sum_gg.sqrt());
        let delta = eta * grad;
        let new_weight = self.weights.entry(key).and_modify(|e| *e -= delta).or_insert(-delta);
        if new_weight.abs() <= 1e-7 {
            self.weights.remove(&key);
        } else {
            let l1_penalty = self.lambda1 * new_weight.signum();
            let l2_penalty = self.lambda2 * *new_weight;
            *new_weight -= eta * (l1_penalty + l2_penalty);
        }
    }
}

// AdaGradRDA optimizer
// This optimizer is a variant of AdaGrad that uses a proximal gradient step with
// adaptive learning rate and regularization.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct AdaGradRDA {
    weights: HashMap<i64, f32>,
    sum_grad: HashMap<i64, f32>,
    sum_sq_grad: HashMap<i64, f32>,
    alpha: f32, // initial learning rate
    lambda1: f32, // L1 regularization
}

impl AdaGradRDA {
    const ID: &'static str = "AdaGradRDA";

    pub fn new(alpha: f32, lambda1: f32) -> Self {
        AdaGradRDA {
            weights: HashMap::with_capacity(8192),
            sum_grad: HashMap::with_capacity(8192),
            sum_sq_grad: HashMap::with_capacity(8192),
            alpha,
            lambda1,
        }
    }
}

impl Optimizer for AdaGradRDA {

    fn id(&self) -> &'static str {
        Self::ID
    }

    #[inline]
    fn get_weights(&self) -> &HashMap<i64, f32> {
        &self.weights
    }

    fn update_gradients(&mut self, ui: i32, item_id: i32, grad: f32, step: u32) {
        let key = compound_key(ui, item_id);

        let new_sum_grad = self.sum_grad.get(&key).copied().unwrap_or_default() + grad;
        let sign = new_sum_grad.signum();
        let mean_grad = (sign * new_sum_grad / (step as f32)) - self.lambda1;
        if mean_grad < 0.0 {
            self.weights.remove(&key);
        } else {
            let gg = grad.powi(2).clamp(1e-8, 1e8); // clip squared gradient
            let sum_gg =* self.sum_sq_grad.entry(key).and_modify(|e| *e += gg).or_insert(gg);
            let new_weight = -sign * self.alpha * step as f32 * mean_grad / (1.0 + sum_gg).sqrt();
            self.weights.insert(key, new_weight);
            self.sum_grad.insert(key, new_sum_grad);
        }
    }
}

/// FTRL (Follow The Regularized Leader) optimizer implementation.
/// This is a variant of FTRL-Proximal algorithm for online learning.
///
/// Reference:
/// - https://www.eecs.tufts.edu/~dsculley/papers/ad-click-prediction.pdf
/// - https://arxiv.org/abs/1403.3465
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct FTRL {
    beta: f32,
    alpha: f32,
    lambda1: f32,
    lambda2: f32,
    z: HashMap<i64, f32>,
    n: HashMap<i64, f32>,
    /// similarity matrix of (target_item_id, base_item_id) -> coefficient
    weights: HashMap<i64, f32>,
}

impl FTRL {
    const ID: &'static str = "FTRL";

    pub fn new(alpha: f32, beta: f32, lambda1: f32, lambda2: f32) -> Self {
        FTRL {
            alpha,
            beta,
            lambda1,
            lambda2,
            z: HashMap::with_capacity(8192),
            n: HashMap::with_capacity(8192),
            weights: HashMap::with_capacity(8192),
        }
    }

    #[deprecated="Use update_gradients instead"]
    pub fn vectorized_update_gradients(&mut self, item_id: i32, updates: &[(i32, f32)]) {
        // Preallocate collections to store updates
        let size: usize = updates.len();
        let mut z_updates = Vec::with_capacity(size);
        let mut n_updates = Vec::with_capacity(size);
        let mut weight_updates = Vec::with_capacity(size);

        // Record each update
        for &(ui, grad) in updates {
            let key = compound_key(ui, item_id);
            let z_val = self.z.get(&key).copied().unwrap_or_default();
            let n_val = self.n.get(&key).copied().unwrap_or_default();

            let n_new = n_val + grad.powi(2);
            let sigma = (n_new.sqrt() - n_val.sqrt()) / self.alpha;
            let z_new = z_val + grad - sigma * self.weights.get(&key).copied().unwrap_or_default();

            if z_new.abs() > self.lambda1 {
                z_updates.push((key, z_new));
                n_updates.push((key, n_new));

                let weight_update = -(z_new - z_new.signum() * self.lambda1)
                    / ((self.beta + n_new.sqrt()) / self.alpha + self.lambda2);
                if weight_update.abs() >= 1e-7 {
                    weight_updates.push((key, weight_update));
                } else {
                    debug!("Weight update is too small at ui:item_id is ({}:{}) => {}", ui, item_id, weight_update);
                    self.weights.remove(&key);
                }
            } else {
                debug!("Weight update is too small at ui:item_id is ({}:{}) => {}", ui, item_id, z_new);
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

impl Optimizer for FTRL {

    fn id(&self) -> &'static str {
        Self::ID
    }

    #[inline]
    fn get_weights(&self) -> &HashMap<i64, f32> {
        &self.weights
    }

    fn update_gradients(&mut self, ui: i32, item_id: i32, grad: f32, _step: u32) {
        let key = compound_key(ui, item_id);
        let z_val = self.z.get(&key).copied().unwrap_or_default();
        let n_val = self.n.get(&key).copied().unwrap_or_default();

        let n_new = n_val + grad.powi(2);
        let sigma = (n_new.sqrt() - n_val.sqrt()) / self.alpha;
        let z_new = z_val + grad - sigma * self.weights.get(&key).copied().unwrap_or_default();

        if z_new.abs() <= self.lambda1 {
            self.weights.remove(&key);
            return;
        }

        self.z.insert(key, z_new);
        self.n.insert(key, n_new);

        let weight_update = -(z_new - z_new.signum() * self.lambda1) / ((self.beta + n_new.sqrt()) / self.alpha + self.lambda2);

        if weight_update.abs() < 1e-7 {
            self.weights.remove(&key);
            return;
        }

        self.weights.insert(key, weight_update);
    }

}

// Create registry for `Example` and register all concrete types with it. Store in static with
// `LazyLock` to lazily initialize it once while being able to create global references to it.
static SERDE_REGISTRY: LazyLock<MapRegistry<dyn Optimizer>> = LazyLock::new(|| {
    let mut registry = MapRegistry::<dyn Optimizer>::new("Optimizer");
    registry.register(SGD::ID, |d| Ok(Box::new(erased_serde::deserialize::<SGD>(d)?)));
    registry.register(AdaGrad::ID, |d| Ok(Box::new(erased_serde::deserialize::<AdaGrad>(d)?)));
    registry.register(AdaGradRDA::ID, |d| Ok(Box::new(erased_serde::deserialize::<AdaGradRDA>(d)?)));
    registry.register(FTRL::ID, |d| Ok(Box::new(erased_serde::deserialize::<FTRL>(d)?)));
    registry
});

// (De)serialize implementations
impl<'a> Serialize for dyn Optimizer + 'a {
    fn serialize<S: Serializer >(&self, serializer: S) -> Result<S::Ok, S::Error> {
        // Check that `Example` has `erased_serde::Serialize` as a supertrait, preventing infinite
        // recursion at runtime.
        const fn __check_erased_serialize_supertrait<T: ?Sized + Optimizer>() {
            serde_flexitos::ser::require_erased_serialize_impl::<T>();
        }
        serialize_trait_object(serializer, self.id(), self)
    }
}

impl<'de> Deserialize<'de> for Box<dyn Optimizer> {
    fn deserialize<D: Deserializer<'de> >(deserializer: D) -> Result<Self, D::Error> {
        SERDE_REGISTRY.deserialize_trait_object(deserializer)
    }
}

pub fn create_optimizer(name: &str, alpha: f32, lambda1: f32, lambda2: f32) -> OptimizerObject {
    let power_t = 0.1;
    let beta = 1.0;
    let optimizer = match name.to_lowercase().as_str() {
        "sgd" => Box::new(SGD::new(alpha, power_t, lambda1, lambda2)) as Box<dyn Optimizer>,
        "adagrad" => Box::new(AdaGrad::new(alpha, lambda1, lambda2)) as Box<dyn Optimizer>,
        "adagrad_rda" => Box::new(AdaGradRDA::new(alpha, lambda1)) as Box<dyn Optimizer>,
        "ftrl" => Box::new(FTRL::new(alpha, beta, lambda1, lambda2)) as Box<dyn Optimizer>,
        _ => panic!("Unknown optimizer: {}", name),
    };
    OptimizerObject::new(name.to_string(), optimizer)
}

#[inline(always)]
pub fn compound_key(k1: i32, k2: i32) -> i64 {
    ((k1 as i64) << 32) | (k2 as u32 as i64)
}

#[inline(always)]
fn invoscaling_eta(alpha: f32, step: u32, power_t: f32) -> f32 {
    alpha / (step as f32).powf(power_t)
}
