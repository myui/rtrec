use std::any::Any;
use std::error::Error;
use std::fmt;

use hashbrown::HashMap;
use serde::{Serialize, Deserialize};

use pyo3::types::PyString;
use pyo3::{FromPyObject, PyAny, PyObject, PyResult, Python};
use pyo3::conversion::IntoPy;

#[derive(Debug)]
pub enum IdentifierError {
    NotFound { id_name: String, obj_id: i32 },
    MixedTypes { name: String, obj: SerializableValue },
}

impl fmt::Display for IdentifierError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IdentifierError::NotFound { id_name, obj_id } => {
                write!(f, "Identifier not found for {}: {}", id_name, obj_id)
            }
            IdentifierError::MixedTypes { name, obj } => {
                write!(f, "Mixed types detected for {}: {:?}", name, obj)
            }
        }
    }
}

impl Error for IdentifierError {}

/// Wrapper for serializable values that can be converted to/from `Any`.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq, Hash)]
pub enum SerializableValue {
    Integer(i32),
    Text(String),
}

impl SerializableValue {
    /// Converts to `&dyn Any`.
    pub fn as_any(&self) -> &dyn Any {
        match self {
            SerializableValue::Integer(v) => v,
            SerializableValue::Text(v) => v,
        }
    }

    pub fn from_any(value: &dyn Any) -> Self {
        if let Some(v) = value.downcast_ref::<i32>() {
            SerializableValue::Integer(*v)
        } else if let Some(v) = value.downcast_ref::<String>() {
            SerializableValue::Text(v.clone())
        } else {
            panic!("Unsupported type for SerializableValue");
        }
    }

    #[inline]
    pub fn into_py(self) -> PyObject {
        Python::with_gil(|py| match self {
            SerializableValue::Integer(v) => v.into_py(py),
            SerializableValue::Text(v) => PyString::new_bound(py, &v).into_py(py),
        })
    }
}

impl<'source> FromPyObject<'source> for SerializableValue {
    #[inline]
    fn extract(ob: &'source PyAny) -> PyResult<Self> {
        if let Ok(val) = ob.extract::<i32>() {
            Ok(SerializableValue::Integer(val))
        } else if let Ok(val) = ob.extract::<String>() {
            Ok(SerializableValue::Text(val))
        } else {
            Err(pyo3::exceptions::PyTypeError::new_err("Unsupported type for SerializableValue"))
        }
    }
}

// Implement conversion for SerializableValue to PyObject
impl IntoPy<PyObject> for SerializableValue {
    #[inline]
    fn into_py(self, py: Python<'_>) -> PyObject {
        match self {
            SerializableValue::Integer(v) => v.into_py(py),
            SerializableValue::Text(v) => PyString::new_bound(py, &v).into_py(py),
        }
    }
}

/// Identifier struct to manage object-to-ID mappings.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Identifier {
    name: String,
    pass_through: Option<bool>,
    obj_to_id: HashMap<SerializableValue, i32>,
    id_to_obj: Vec<SerializableValue>,
}

impl Identifier {
    /// Creates a new Identifier.
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            pass_through: None,
            obj_to_id: HashMap::with_capacity(8192),
            id_to_obj: Vec::with_capacity(8192),
        }
    }

    /// Identifies an object and returns its ID.
    pub fn identify(&mut self, obj: SerializableValue) -> Result<i32, IdentifierError> {
        if let Some(id) = self.as_integer(&obj) {
            if self.pass_through == Some(false) {
                return Err(IdentifierError::MixedTypes {
                    name: self.name.clone(),
                    obj,
                });
            }
            self.pass_through = Some(true);
            return Ok(id);
        }

        if self.pass_through == Some(true) {
            return Err(IdentifierError::MixedTypes {
                name: self.name.clone(),
                obj,
            });
        }

        if let Some(&id) = self.obj_to_id.get(&obj) {
            return Ok(id);
        }

        let new_id = self.id_to_obj.len() as i32;
        self.obj_to_id.insert(obj.clone(), new_id);
        self.id_to_obj.push(obj);
        self.pass_through = Some(false);
        Ok(new_id)
    }

    /// Retrieves the ID of an object if it exists.
    #[inline]
    pub fn get_id(&self, obj: &SerializableValue) -> Result<Option<i32>, IdentifierError> {
        if let Some(id) = self.as_integer(obj) {
            if self.pass_through == Some(false) {
                return Err(IdentifierError::MixedTypes{
                    name: self.name.clone(),
                    obj: obj.clone(),
                });
            }
            return Ok(Some(id));
        }

        Ok(self.obj_to_id.get(obj).copied())
    }

    /// Retrieves an object by its ID.
    #[inline]
    pub fn get(&self, obj_id: i32) -> Result<SerializableValue, IdentifierError> {
        if self.pass_through == Some(true) {
            return Ok(SerializableValue::Integer(obj_id));
        }

        self.id_to_obj
            .get(obj_id as usize)
            .cloned()
            .ok_or_else(|| IdentifierError::NotFound {
                id_name: self.name.clone(),
                obj_id,
            })
    }

    /// Retrieves an object by its ID or returns a default value.
    #[inline]
    pub fn get_or_default(
        &self,
        obj_id: i32,
        default: Option<SerializableValue>,
    ) -> SerializableValue {
        if self.pass_through == Some(true) {
            return SerializableValue::Integer(obj_id);
        }

        self.id_to_obj.get(obj_id as usize).cloned().unwrap_or_else(|| {
            default.unwrap_or_else(|| {
                panic!("Identifier not found for {}: {}", self.name, obj_id);
            })
        })
    }

    /// Helper function to treat integer-like values as their IDs.
    #[inline(always)]
    fn as_integer(&self, obj: &SerializableValue) -> Option<i32> {
        if let SerializableValue::Integer(v) = obj {
            Some(*v)
        } else {
            None
        }
    }
}

