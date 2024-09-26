pub mod slim;
pub mod datasets;
pub mod ftrl;

use slim::SlimMSE;
use pyo3::prelude::*;

#[pymodule]
fn _lowlevel(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<SlimMSE>()?;
    Ok(())
}
