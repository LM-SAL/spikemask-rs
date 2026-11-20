//! Python binding and validation.
use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::{exceptions::PyValueError, prelude::*};

use crate::detect::Detector;
use crate::workers::{self, WorkerPool};

#[derive(FromPyObject)]
pub(crate) enum Exposure<'py> {
    Int32(PyReadonlyArray2<'py, i32>),
    Float32(PyReadonlyArray2<'py, f32>),
    Float64(PyReadonlyArray2<'py, f64>),
}

fn working_copy<T: numpy::Element + Copy + Into<f64>>(
    exposure: PyReadonlyArray2<'_, T>,
) -> PyResult<([usize; 2], Vec<f64>)> {
    if !exposure.is_c_contiguous() {
        return Err(PyValueError::new_err("exposure must be C-contiguous"));
    }
    let shape = [exposure.shape()[0], exposure.shape()[1]];
    // Fill the mutable clipping buffer directly from the detector's native dtype.
    let image = exposure
        .as_slice()?
        .iter()
        .copied()
        .map(Into::into)
        .collect();
    Ok((shape, image))
}

#[pyfunction]
#[pyo3(signature = (exposure, window, sigma_upper, max_iters, pool=None))]
pub(crate) fn spike_mask<'py>(
    py: Python<'py>,
    exposure: Exposure<'py>,
    window: (usize, usize),
    sigma_upper: f64,
    max_iters: usize,
    pool: Option<&WorkerPool>,
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
    let ([height, width], image) = match exposure {
        Exposure::Int32(a) => working_copy(a)?,
        Exposure::Float32(a) => working_copy(a)?,
        Exposure::Float64(a) => working_copy(a)?,
    };
    let detector = Detector {
        height,
        width,
        window,
        sigma_upper,
    };
    let mask = py.detach(move || workers::run(pool, || detector.mask(image, max_iters)));
    PyArray1::from_vec(py, mask).reshape([height, width])
}
