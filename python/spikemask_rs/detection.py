"""
Upper sigma clipping for detector pixels.
"""

from __future__ import annotations

from operator import index
from typing import TYPE_CHECKING

import numpy as np

from ._spikemask_rs import spike_mask as _spike_mask
from ._workers import resolve_pool

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

__all__ = ["spike_mask"]

_DETECTOR_NDIM = 2


# Defaults measured on mocked MUSE frames (design notes); the IDL demo uses (5, 3) and 2.65.
def spike_mask(
    exposure: ArrayLike,
    *,
    window: tuple[int, int] = (7, 5),
    sigma_upper: float = 3.0,
    max_iters: int = 10,
    workers: int | None = None,
) -> NDArray[np.bool_]:
    """
    Flag candidate positive spikes in one two-dimensional exposure.

    Parameters
    ----------
    exposure : array_like
        Real-valued detector image. Nonfinite samples are ignored.
    window : tuple of int, optional
        Odd, positive sizes in (rows, columns) order. By default (7, 5).
    sigma_upper : float, optional
        Flag a pixel when it exceeds the local median by more than this many
        population standard deviations. By default 3.0.
    max_iters : int, optional
        Maximum number of passes. Each pass excludes previously flagged pixels.
        By default 10.
    workers : int, optional
        Number of Rayon threads for this call. By default use the active
        worker context or Rayon's default pool.

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
        resolve_pool(workers),
    )
