use std::fs::File;
use std::io::{BufReader, BufWriter};

use pyo3::prelude::*;

use log::{debug, warn};
use env_logger;
use serde::{Serialize, Deserialize};
use rusoto_core::Region;
use rusoto_s3::{PutObjectRequest, GetObjectRequest, S3Client, S3};
use tokio::runtime::Runtime;
use tokio::io::AsyncReadExt;
use rayon::prelude::*;

use crate::optimizers::{compound_key, create_optimizer, OptimizerObject};
use crate::interactions::UserItemInteractions;
use crate::identifiers::{Identifier, SerializableValue};

#[pyclass]
#[derive(Serialize, Deserialize, Debug)]
pub struct SlimMSE {
    interactions: UserItemInteractions,
    optim: OptimizerObject,
    cumulative_loss: f32,
    steps: u32,
    user_ids: Identifier,
    item_ids: Identifier,
    n_recent: Option<usize>,
}

#[pymethods]
impl SlimMSE {
    #[new]
    #[pyo3(signature = (optimizer = "adagrad_rda", alpha = 0.001, lambda1 = 0.0002, lambda2 = 0.0001, rating_range = (-5.0, 10.0), decay_in_days = None, n_recent = 50))]
    pub fn new(optimizer: &str, alpha: f32, lambda1: f32, lambda2: f32, rating_range: (f32, f32), decay_in_days: Option<f32>, n_recent: Option<usize>) -> Self {
        env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).try_init().ok();

        let optim = create_optimizer(optimizer, alpha, lambda1, lambda2);
        SlimMSE {
            interactions: UserItemInteractions::new(rating_range.0, rating_range.1, decay_in_days),
            optim,
            cumulative_loss: 0.0,
            steps: 0,
            user_ids: Identifier::new("user"),
            item_ids: Identifier::new("item"),
            n_recent,
        }
    }

    pub fn fit(&mut self, user_interactions: Vec<(SerializableValue, SerializableValue, f32, f32)>, update_interaction: Option<bool>) {
        for (user, item, tstamp, rating) in user_interactions {
            if let Err(e) = {
                let user_id = self.identify_user(user);
                let item_id = self.identify_item(item);
                self.interactions.add_interaction(user_id, item_id, tstamp, rating, update_interaction.unwrap_or(false));
                self.update_weights(user_id, item_id);
                Ok::<(), Box<dyn std::error::Error>>(())
            } {
                warn!("Failed to fit interaction: {}", e);
            }
        }
    }

    pub fn fit_identified(&mut self, user_interactions: Vec<(i32, i32, f32, f32)>, add_interaction: Option<bool>, update_interaction: Option<bool>) {
        for (user_id, item_id, tstamp, rating) in user_interactions {
            if let Err(e) = {
                if add_interaction.unwrap_or(true) {
                    self.interactions.add_interaction(user_id, item_id, tstamp, rating, update_interaction.unwrap_or(false));
                }
                self.update_weights(user_id, item_id);
                Ok::<(), Box<dyn std::error::Error>>(())
            } {
                warn!("Failed to fit interaction: {}", e);
            }
        }
    }

    /// Bulk identify users and items from the provided interactions.
    #[inline]
    pub fn bulk_identify(&mut self, user_interactions: Vec<(SerializableValue, SerializableValue)>) -> Vec<(i32, i32)> {
        user_interactions.into_iter().map(|(user, item)| {
            let user_id = self.identify_user(user);
            let item_id = self.identify_item(item);
            (user_id, item_id)
        }).collect()
    }

    #[inline(always)]
    fn identify_user(&mut self, user: SerializableValue) -> i32 {
        self.user_ids.identify(user).unwrap_or_else(|_| panic!("Failed to identify user"))
    }

    #[inline(always)]
    fn identify_item(&mut self, item: SerializableValue) -> i32 {
        self.item_ids.identify(item).unwrap_or_else(|_| panic!("Failed to identify item"))
    }

    fn update_weights(&mut self, user_id: i32, item_id: i32) {
        let n_recent = self.n_recent;
        let predicted = self._predict_rating(user_id, item_id, false, n_recent);
        let actual = self.interactions.get_user_item_rating(user_id, item_id, 0.0);
        let dloss = predicted - actual;

        // test if dloss is finite
        if !dloss.is_finite() {
            debug!("dloss is not finite for user_id: {}, item_id: {}, dloss: {}, predicted: {}, actual: {}", user_id, item_id, dloss, predicted, actual);
            return;
        }

        self.cumulative_loss += dloss.abs();
        self.steps = self.steps.saturating_add(1);

        // update item similarity matrix
        // Note: Change applied to the original SLIM algorithm; only update the similarity matrix
        // where the user has (recently) interacted with the item.
        //
        // No interaction implies no update; grad = dloss * rating = 0
        let optimizer = &mut self.optim.instance;
        let user_items = self.interactions.get_user_items(user_id, n_recent);
        user_items.iter()
            .filter(|&&ui| ui != item_id) // Skip diagonal elements
            .for_each(|&ui| {
                let rating = self.interactions.get_user_item_rating(user_id, ui, 0.0);
                let grad = dloss * rating;
                if grad.abs() < 1e-7 { // Skip very small gradients
                    return;
                }
                optimizer.update_gradients(ui, item_id, grad, self.steps);
            });
    }

    pub fn predict_rating_batch(&self, users: Vec<SerializableValue>, items: Vec<SerializableValue>) -> Vec<f32> {
        users.iter().zip(items.iter()).map(|(user, item)| {
            let user_id = match self.user_ids.get_id(user).unwrap() {
                Some(id) => id,
                None => return 0.0, // Return 0 if user ID not found
            };
            let item_id = match self.item_ids.get_id(item).unwrap() {
                Some(id) => id,
                None => return 0.0, // Return 0 if item ID not found
            };
            self._predict_rating(user_id, item_id, true, None)
        }).collect()
    }

    pub fn predict_rating(&self, user: SerializableValue, item: SerializableValue) -> f32 {
        let user_id = match self.user_ids.get_id(&user).unwrap() {
            Some(id) => id,
            None => return 0.0, // Return 0 if user ID not found
        };
        let item_id = match self.item_ids.get_id(&item).unwrap() {
            Some(id) => id,
            None => return 0.0, // Return 0 if item ID not found
        };
        self._predict_rating(user_id, item_id, true, None)
    }

    fn _predict_rating(&self, user_id: i32, item_id: i32, bypass_prediction: bool, n_recent: Option<usize>) -> f32 {
        let user_items = self.interactions.get_user_items(user_id, n_recent);

        if bypass_prediction && user_items.len() == 1 && user_items[0] == item_id {
            // Return raw rating if user has only interacted with the item
            return self.interactions.get_user_item_rating(user_id, item_id, 0.0);
        }

        let weights = self.optim.instance.get_weights();
        if user_items.len() > 20 {
            user_items.par_chunks(20)
                .map(|chunk| {
                    chunk.iter()
                        .filter(|&&ui| ui != item_id) // Skip diagonal elements
                        .map(|&ui| {
                            let weight = weights.get(&compound_key(ui, item_id)).unwrap_or(&0.0);
                            let rating = self.interactions.get_user_item_rating(user_id, ui, 0.0);
                            weight * rating
                        })
                        .sum::<f32>()
                })
                .sum()
        } else {
            user_items.iter()
                .filter(|&&ui| ui != item_id) // Skip diagonal elements
                .map(|&ui| {
                    let weight = weights.get(&compound_key(ui, item_id)).unwrap_or(&0.0);
                    let rating = self.interactions.get_user_item_rating(user_id, ui, 0.0);
                    weight * rating
                })
                .sum()
        }
    }

    #[pyo3(signature = (users, top_k = 10, filter_interacted = true))]
    pub fn recommend_batch(&self, users: Vec<SerializableValue>, top_k: usize, filter_interacted: Option<bool>) -> Vec<Vec<SerializableValue>> {
        // avoid nested parallelism (into_par_iter) for CPU cache locality
        users.into_iter().map(|user| self.recommend(user, top_k, filter_interacted)).collect()
    }

    #[pyo3(signature = (user, top_k = 10, filter_interacted = true))]
    pub fn recommend(&self, user: SerializableValue, top_k: usize, filter_interacted: Option<bool>) -> Vec<SerializableValue> {
        let user_id = match self.user_ids.get_id(&user).unwrap() {
            Some(id) => id,
            None => return vec![], // TODO: Return empty list if user ID not found
        };

        // Get the candidate items based on the filtering condition
        let candidate_item_ids = if filter_interacted.unwrap_or(true) {
            self.interactions.get_all_non_interacted_items(user_id)
        } else {
            self.interactions.get_all_non_negative_items(user_id)
        };

        // Predict scores for the candidate items
        let mut scores: Vec<(SerializableValue, f32)> = candidate_item_ids
            .par_chunks(1024)
            .flat_map(|chunk| {
                chunk.iter().map(|&item_id| {
                    let score = self._predict_rating(user_id, item_id, false, None);
                    let item = self.item_ids.get(item_id).unwrap();
                    (item, score)
                }).collect::<Vec<_>>()
            })
            .collect();

        // Sort items by score in descending order
        scores.par_sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Take the top-k items and return their them
        scores.iter().take(top_k).map(|(item, _)| item.clone()).collect()
    }

    pub fn similar_items(
        &self,
        query_items: Vec<SerializableValue>, // assuming item IDs are strings
        top_k: usize,
        filter_query_items: bool,
    ) -> Vec<Vec<SerializableValue>> {
        // Convert query items to internal IDs
        let query_item_ids: Vec<Option<i32>> = query_items
            .iter()
            .map(|item| self.item_ids.get_id(item).unwrap())
            .collect();

        // Get all target item IDs from interactions
        let target_item_ids = self.interactions.get_all_item_ids();

        let weights = self.optim.instance.get_weights();

        // Loop over each query item and find similar items
        // Use parallel iterator for better performance
        let similar_items = query_item_ids
            .iter() // Do not use par_iter() here to avoid nested parallelism for CPU cache locality
            .map(|&query_item_id_opt| {
                if let Some(query_item_id) = query_item_id_opt {
                    let mut item_scores: Vec<(i32, f32)> = target_item_ids
                        .par_iter()
                        .filter_map(|&target_item_id| {
                            if !filter_query_items || target_item_id != query_item_id {
                                // Retrieve similarity score from weights or use NEG_INFINITY as default
                                let similarity_score: f32 =
                                    *weights.get(&compound_key(target_item_id, query_item_id))
                                    .unwrap_or(&f32::NEG_INFINITY);

                                Some((target_item_id, similarity_score))
                            } else {
                                None
                            }
                        })
                        .collect();

                    // Sort by similarity score in descending order and keep the top_k items
                    item_scores.par_sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                    item_scores
                        .iter()
                        .take(top_k)
                        .filter_map(|&(item_id, _)| self.item_ids.get(item_id).ok())
                        .collect()
                } else {
                    // If the query item ID is None, add an empty list
                    Vec::new()
                }
            })
            .collect();
        similar_items
    }

    pub fn get_empirical_error(&mut self) -> f32 {
        if self.steps == 0 {
            return 0.0; // Avoid division by zero
        }
        self.cumulative_loss / self.steps as f32
    }

    /// Save the SlimMSE model to a specified path using MessagePack.
    /// Supports saving to a local file or an S3 path (e.g., s3://bucket-name/path/to/file).
    pub fn save(&self, path: &str) -> PyResult<()> {
        if path.starts_with("s3://") {
            // Delegate saving to S3
            save_to_s3(path, &self)
        } else {
            // Save to local file system
            save_to_file(path, &self)
        }
    }

    /// Load the SlimMSE model from a specified path using MessagePack.
    /// Supports loading from a local file or an S3 path (e.g., s3://bucket-name/path/to/file).
    #[staticmethod]
    pub fn load(path: &str) -> PyResult<Self> {
        if path.starts_with("s3://") {
            // Delegate loading from S3
            load_from_s3(path)
        } else {
            // Load from local file system
            load_from_file(path)
        }
    }

}

/// Save the given object to a local file using the specified file path.
fn save_to_file<T>(file_path: &str, object: &T) -> PyResult<()>
where
    T: Serialize,
{
    let path = if let Some(stripped) = file_path.strip_prefix("file://") {
        stripped
    } else {
        file_path // Use the path as is
    };

    let file = File::create(path)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("Failed to create file: {}", e)))?;
    let mut writer = BufWriter::new(file);

    // Serialize the object to MessagePack format
    rmp_serde::encode::write(&mut writer, object)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("Failed to serialize: {}", e)))?;

    Ok(())
}

/// Load an object of type `T` from a local file using the specified file path.
fn load_from_file<T>(file_path: &str) -> PyResult<T>
where
    T: for<'de> Deserialize<'de>,
{
    let path = if let Some(stripped) = file_path.strip_prefix("file://") {
        stripped
    } else {
        file_path // Use the path as is
    };

    let file = File::open(path)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("Failed to open file: {}", e)))?;
    let reader = BufReader::new(file);

    // Deserialize the object from MessagePack format
    let object: T = rmp_serde::decode::from_read(reader)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("Failed to deserialize: {}", e)))?;

    Ok(object)
}

/// Save the given object to S3 using the specified S3 path.
/// The S3 path should be of the form `s3://bucket-name/path/to/file`.
fn save_to_s3<T>(s3_path: &str, object: &T) -> PyResult<()>
where
    T: Serialize,
{
    let (bucket_name, object_key) = parse_s3_path(s3_path);

    // Serialize the object to MessagePack format
    let serialized_data = rmp_serde::to_vec(object)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("Failed to serialize: {}", e)))?;

    // Create a new Tokio runtime for async S3 operations
    let rt = Runtime::new().unwrap();
    rt.block_on(async {
        // Initialize S3 client
        let client = S3Client::new(Region::default());

        // Create PutObjectRequest
        let put_request = PutObjectRequest {
            bucket: bucket_name.to_string(),
            key: object_key.to_string(),
            body: Some(serialized_data.into()),
            ..Default::default()
        };

        // Upload the serialized object to S3
        client.put_object(put_request).await.map_err(|e| {
            pyo3::exceptions::PyIOError::new_err(format!("Failed to upload to S3: {:?}", e))
        })
    })?;

    Ok(())
}

/// Load an object of type `SlimMSE` from S3 using the specified S3 path.
/// The S3 path should be of the form `s3://bucket-name/path/to/file`.
fn load_from_s3(s3_path: &str) -> PyResult<SlimMSE> {
    let (bucket_name, object_key) = parse_s3_path(s3_path);

    // Create a new Tokio runtime for async S3 operations
    let rt = Runtime::new().unwrap();
    let data = rt.block_on(async {
        // Initialize S3 client
        let client = S3Client::new(Region::default());

        // Create GetObjectRequest
        let get_request = GetObjectRequest {
            bucket: bucket_name.to_string(),
            key: object_key.to_string(),
            ..Default::default()
        };

        // Download the object from S3
        match client.get_object(get_request).await {
            Ok(output) => {
                let mut stream = output.body.unwrap().into_async_read();
                let mut body = Vec::new();
                stream.read_to_end(&mut body).await.unwrap();
                Ok(body)
            }
            Err(e) => Err(pyo3::exceptions::PyIOError::new_err(format!("Failed to download from S3: {:?}", e))),
        }
    })?;

    // Deserialize the data into a SlimMSE instance
    let slim: SlimMSE = rmp_serde::from_slice(&data)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("Failed to deserialize: {}", e)))?;
    Ok(slim)
}

/// Helper function to parse S3 paths into bucket name and object key.
/// S3 paths are of the form: s3://bucket-name/path/to/file
fn parse_s3_path(s3_path: &str) -> (&str, &str) {
    let path_without_prefix = &s3_path[5..]; // Remove "s3://"
    let mut split = path_without_prefix.splitn(2, '/');
    let bucket_name = split.next().expect("Invalid S3 path: No bucket name found");
    let object_key = split.next().unwrap_or(""); // If no '/' found, object_key is empty
    (bucket_name, object_key)
}
