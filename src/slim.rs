use pyo3::prelude::*;

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use serde::{Serialize, Deserialize};

use rusoto_core::Region;
use rusoto_s3::{PutObjectRequest, GetObjectRequest, S3Client, S3};
use tokio::runtime::Runtime;
use tokio::io::AsyncReadExt;

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
    let path = if file_path.starts_with("file://") {
        &file_path[7..] // Remove "file://" prefix
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
    let path = if file_path.starts_with("file://") {
        &file_path[7..] // Remove "file://" prefix
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