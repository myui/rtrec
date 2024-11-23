pub mod slim;
pub mod interactions;
pub mod optimizers;
pub mod identifiers;

use slim::SlimMSE;
use pyo3::prelude::*;

#[pyfunction]
fn set_notebook_mode(toggle: bool) -> PyResult<()> {
    kdam::set_notebook(toggle);
    Ok(())
}

#[pymodule]
fn _lowlevel(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<SlimMSE>()?;
    m.add_function(wrap_pyfunction!(set_notebook_mode, m)?)?;
    Ok(())
}
