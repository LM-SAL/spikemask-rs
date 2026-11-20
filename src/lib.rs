//! Candidate cosmic-ray spike masks for two-dimensional detector exposures.
mod api;
mod detect;

#[cfg(test)]
mod tests;

use pyo3::prelude::*;

#[pymodule(gil_used = false)]
fn _spikes_rs(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(api::spike_mask, module)?)?;
    Ok(())
}
