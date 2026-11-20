"""
Compare the public mask with the C-backed clipping kernels in ftoolss.
"""

import numpy as np
import pytest

from spikes_rs import spike_mask

ftoolss = pytest.importorskip("ftoolss", exc_type=ModuleNotFoundError)


def ftoolss_mask(image, window, max_iters):
    # ftoolss writes NaNs, so integer counts need a floating type that represents every int32.
    data = image.astype(np.float64) if image.dtype == np.int32 else image
    clipped = ftoolss.sigma_clip(
        data,
        window[::-1],  # ftoolss specifies (columns, rows).
        sigma_upper=2.65,
        sigma_lower=1000,
        maxiters=max_iters,
    )
    return np.isfinite(image) & np.isnan(clipped)


@pytest.mark.parametrize("dtype", [np.int32, np.float32, np.float64])
@pytest.mark.parametrize("window", [(5, 3), (3, 5)])
@pytest.mark.parametrize("max_iters", [1, 10])
@pytest.mark.parametrize("workers", [1, 4])
def test_mask_matches_ftoolss(dtype, window, max_iters, workers):
    image = np.random.default_rng(2026).normal(100, 3, (35, 49)).astype(dtype)
    image[0, 8] = 700
    image[16:18, 24] = [500, 10000]
    image[27, 38] = -9999
    if dtype != np.int32:
        image[5, ::3] = np.nan

    expected = ftoolss_mask(image, window, max_iters)
    assert expected[0, 8]
    assert expected[17, 24]
    assert expected[16, 24] == (max_iters > 1)  # The stronger neighbor hides it on pass one.
    assert not expected[27, 38]
    actual = spike_mask(image, window=window, max_iters=max_iters, workers=workers)
    np.testing.assert_array_equal(actual, expected, strict=True)


def test_int32_single_count_above_float32_precision_matches_ftoolss():
    image = np.full((13, 27), 2**27, dtype=np.int32)
    image[6, 12] += 1
    expected = ftoolss_mask(image, (5, 3), 10)
    assert np.argwhere(expected).tolist() == [[6, 12]]
    np.testing.assert_array_equal(spike_mask(image), expected, strict=True)
