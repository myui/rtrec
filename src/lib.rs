mod slim;
mod datasets;
mod ftrl;

use slim::SlimMSE;
use pyo3::prelude::*;

#[pymodule]
fn _lowlevel(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<SlimMSE>()?;
    Ok(())
}
