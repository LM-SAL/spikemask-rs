//! Candidate cosmic-ray spike masks for two-dimensional detector exposures.
mod api;
mod detect;
mod workers;

#[cfg(test)]
mod tests;

use pyo3::prelude::*;

#[pymodule(gil_used = false)]
fn _spikemask_rs(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<workers::WorkerPool>()?;
    module.add_function(wrap_pyfunction!(api::spike_mask, module)?)?;
    Ok(())
}
