//! Python binding and validation.
use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::{exceptions::PyValueError, prelude::*};

use crate::detect::Detector;

#[pyfunction]
pub(crate) fn spike_mask<'py>(
    py: Python<'py>,
    exposure: PyReadonlyArray2<'py, f64>,
    window: (usize, usize),
    sigma_upper: f64,
    max_iters: usize,
) -> PyResult<Bound<'py, PyArray2<bool>>> {
    if window.0.is_multiple_of(2) || window.1.is_multiple_of(2) {
        return Err(PyValueError::new_err(
            "window dimensions must be positive odd integers",
        ));
    }
    if !sigma_upper.is_finite() || sigma_upper <= 0. {
        return Err(PyValueError::new_err(
            "sigma_upper must be positive and finite",
        ));
    }
    if max_iters == 0 {
        return Err(PyValueError::new_err("max_iters must be positive"));
    }
    if !exposure.is_c_contiguous() {
        return Err(PyValueError::new_err("exposure must be C-contiguous"));
    }
    let (height, width) = (exposure.shape()[0], exposure.shape()[1]);
    let detector = Detector {
        height,
        width,
        window,
        sigma_upper,
    };
    // Later passes mark their private working copy, so the caller keeps every original measurement.
    let image = exposure.as_slice()?.to_vec();
    let mask = py.detach(move || detector.mask(image, max_iters));
    PyArray1::from_vec(py, mask).reshape([height, width])
}
