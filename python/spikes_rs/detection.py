"""
Upper sigma clipping for detector pixels.
"""

from __future__ import annotations

from operator import index
from typing import TYPE_CHECKING

import numpy as np

from ._spikes_rs import spike_mask as _spike_mask

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

__all__ = ["spike_mask"]

_DETECTOR_NDIM = 2


def spike_mask(
    exposure: ArrayLike,
    *,
    window: tuple[int, int] = (5, 3),
    sigma_upper: float = 2.65,
    max_iters: int = 10,
) -> NDArray[np.bool_]:
    """
    Flag candidate positive spikes in one two-dimensional exposure.

    Parameters
    ----------
    exposure : array_like
        Real-valued detector image. Nonfinite samples are ignored.
    window : tuple of int, optional
        Odd, positive sizes in (rows, columns) order. By default (5, 3).
    sigma_upper : float, optional
        Flag a pixel when it exceeds the local median by more than this many
        population standard deviations. By default 2.65.
    max_iters : int, optional
        Maximum number of passes. Each pass excludes previously flagged pixels.
        By default 10.

    Returns
    -------
    mask : numpy.ndarray
        Boolean array with the shape of the exposure and True at suspected spikes.
    """
    image = np.asarray(exposure)
    if image.ndim != _DETECTOR_NDIM or image.dtype.kind not in "iuf":
        msg = "exposure must be a real numeric 2-D array"
        raise ValueError(msg)
    rows, columns = map(index, window)
    if min(rows, columns) <= 0:
        msg = "window dimensions must be positive odd integers"
        raise ValueError(msg)
    max_iters = index(max_iters)
    if max_iters <= 0:
        msg = "max_iters must be positive"
        raise ValueError(msg)
    dtype = image.dtype.newbyteorder("=")
    if dtype not in (np.int32, np.float32):
        dtype = np.float64
    return _spike_mask(
        np.ascontiguousarray(image, dtype=dtype),
        (rows, columns),
        sigma_upper,
        max_iters,
    )
