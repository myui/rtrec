use serde::{Serialize, Deserialize};
use std::any::Any;
use std::collections::HashMap;
use std::error::Error;
use std::fmt;

/// Custom error for identifier-related issues.
#[derive(Debug)]
struct IdentifierError {
    id_name: String,
    obj_id: usize,
}

impl fmt::Display for IdentifierError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Identifier not found for {}: {}", self.id_name, self.obj_id)
    }
}

impl Error for IdentifierError {}

#[derive(Debug)]
struct MixedTypeError {
    name: String,
    obj: SerializableValue,
}

impl fmt::Display for MixedTypeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Mixed types detected for {}: {:?}", self.name, self.obj)
    }
}

impl Error for MixedTypeError {}

/// Wrapper for serializable values that can be converted to/from `Any`.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq, Hash)]
pub enum SerializableValue {
    Integer(i32),
    Text(String),
}

impl SerializableValue {
    /// Converts to `&dyn Any`.
    #[allow(dead_code)]
    fn as_any(&self) -> &dyn Any {
        match self {
            SerializableValue::Integer(v) => v,
            SerializableValue::Text(v) => v,
        }
    }

    #[allow(dead_code)]
    fn from_any(value: &dyn Any) -> Option<Self> {
        if let Some(v) = value.downcast_ref::<i32>() {
            Some(SerializableValue::Integer(*v))
        } else if let Some(v) = value.downcast_ref::<String>() {
            Some(SerializableValue::Text(v.clone()))
        } else {
            None
        }
    }
}

/// Identifier struct to manage object-to-ID mappings.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Identifier {
    name: String,
    pass_through: Option<bool>,
    obj_to_id: HashMap<SerializableValue, usize>,
    id_to_obj: Vec<SerializableValue>,
}

impl Identifier {
    /// Creates a new Identifier.
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            pass_through: None,
            obj_to_id: HashMap::new(),
            id_to_obj: Vec::new(),
        }
    }

    /// Identifies an object and returns its ID.
    pub fn identify(&mut self, obj: SerializableValue) -> Result<usize, Box<dyn Error>> {
        if let Some(id) = self.as_integer(&obj) {
            if self.pass_through == Some(false) {
                return Err(Box::new(MixedTypeError {
                    name: self.name.clone(),
                    obj,
                }));
            }
            self.pass_through = Some(true);
            return Ok(id);
        }

        if self.pass_through == Some(true) {
            return Err(Box::new(MixedTypeError {
                name: self.name.clone(),
                obj,
            }));
        }

        if let Some(&id) = self.obj_to_id.get(&obj) {
            return Ok(id);
        }

        let new_id = self.id_to_obj.len();
        self.obj_to_id.insert(obj.clone(), new_id);
        self.id_to_obj.push(obj);
        self.pass_through = Some(false);
        Ok(new_id)
    }

    /// Retrieves the ID of an object if it exists.
    pub fn get_id(&self, obj: &SerializableValue) -> Result<Option<usize>, Box<dyn Error>> {
        if let Some(id) = self.as_integer(obj) {
            if self.pass_through == Some(false) {
                return Err(Box::new(MixedTypeError {
                    name: self.name.clone(),
                    obj: obj.clone(),
                }));
            }
            return Ok(Some(id));
        }

        Ok(self.obj_to_id.get(obj).copied())
    }

    /// Retrieves an object by its ID.
    pub fn get(&self, obj_id: usize) -> Result<SerializableValue, Box<dyn Error>> {
        if self.pass_through == Some(true) {
            return Ok(SerializableValue::Integer(obj_id as i32));
        }

        self.id_to_obj.get(obj_id).cloned().ok_or_else(|| {
            Box::new(IdentifierError {
                id_name: self.name.clone(),
                obj_id,
            }) as Box<dyn Error>
        })
    }

    /// Retrieves an object by its ID or returns a default value.
    pub fn get_or_default(
        &self,
        obj_id: usize,
        default: Option<SerializableValue>,
    ) -> SerializableValue {
        if self.pass_through == Some(true) {
            return SerializableValue::Integer(obj_id as i32);
        }
        self.id_to_obj.get(obj_id).cloned().unwrap_or_else(|| default.unwrap())
    }

    /// Helper function to treat integer-like values as their IDs.
    fn as_integer(&self, obj: &SerializableValue) -> Option<usize> {
        if let SerializableValue::Integer(v) = obj {
            Some(*v as usize)
        } else {
            None
        }
    }
}

