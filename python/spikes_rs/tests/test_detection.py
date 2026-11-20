from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from spikes_rs import spike_mask


def reference(image, window, sigma, passes):
    data = np.array(image, dtype=float, copy=True)
    rows, cols = np.array(window) // 2
    mask = np.zeros(image.shape, dtype=bool)
    for _ in range(passes):
        added = np.zeros_like(mask)
        for y, x in zip(*np.nonzero(np.isfinite(data)), strict=True):
            patch = data[max(y - rows, 0) : y + rows + 1, max(x - cols, 0) : x + cols + 1]
            values = patch[np.isfinite(patch)]
            added[y, x] = data[y, x] - np.median(values) > sigma * np.std(values)
        if not added.any():
            break
        data[added] = np.nan
        mask |= added
    return mask


@pytest.mark.parametrize("window", [(5, 3), (3, 5), (3, 3), (1, 9), (7, 1), (31, 33), (1, 1)])
@pytest.mark.parametrize(("sigma", "passes"), [(2.65, 10), (1.75, 1), (0.9, 3)])
def test_matches_a_full_rescan(window, sigma, passes):
    rng = np.random.default_rng(5)
    data = rng.normal(size=(12, 23))
    data[6:8, 10] = [25, 600]
    data[4, 12] = -9999
    data[2, :4] = [np.nan, np.inf, -np.inf, np.nan]
    expected = reference(data, window, sigma, passes)
    original = data.copy()
    actual = spike_mask(data, window=window, sigma_upper=sigma, max_iters=passes)
    assert actual.dtype == bool
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(data, original)


def test_candidate_mask_preserves_data_and_leaves_missing_values_unflagged():
    data = np.full((30, 32), 100.0)
    data[1, 1] = np.nan
    data[8, 15] = -10000
    data[12, 15] = 10000
    data.setflags(write=False)
    assert np.argwhere(spike_mask(data)).tolist() == [[12, 15]]
    assert data[12, 15] == 10000


@pytest.mark.parametrize("dtype", [np.float32, np.float64, ">f4", ">f8", np.int16, np.uint32])
def test_detector_layout_and_dtype(dtype):
    image = np.full((16, 24), 10, dtype=dtype)
    image[7, 13] = 10000
    for layout in [image, np.asfortranarray(image), image[:, ::-1], image.T]:
        np.testing.assert_array_equal(spike_mask(layout), layout == 10000)


@pytest.mark.parametrize("shape", [(0, 8), (8, 0), (1, 1)])
def test_empty_or_single_pixel(shape):
    np.testing.assert_array_equal(spike_mask(np.zeros(shape)), np.zeros(shape, dtype=bool))


@pytest.mark.parametrize(
    "options",
    [
        {"window": (4, 3)},
        {"window": (1, -1)},
        {"window": (3, 0)},
        {"window": (3,)},
        {"window": (3.5, 5)},
        {"sigma_upper": 0},
        {"sigma_upper": np.inf},
        {"sigma_upper": np.nan},
        {"max_iters": -1},
        {"max_iters": 0},
        {"max_iters": 1.5},
    ],
)
def test_invalid_settings(options):
    with pytest.raises((ValueError, TypeError)):
        spike_mask(np.ones((3, 3)), **options)


@pytest.mark.parametrize("bad", [np.zeros((1, 2, 3)), np.ones((3, 3), complex)])
def test_invalid_detector_data(bad):
    with pytest.raises(ValueError, match="real numeric 2-D"):
        spike_mask(bad)


def test_no_fractional_rounding_at_the_threshold():
    image = np.array([[0.0, 0.0, 1.0]])
    threshold = 1 / np.std(image)
    for sigma in [threshold, np.nextafter(threshold, 0)]:
        np.testing.assert_array_equal(
            spike_mask(image, window=(1, 5), sigma_upper=sigma),
            reference(image, (1, 5), sigma, 10),
        )


def test_independent_calls_from_threads():
    images = [np.random.default_rng(seed).normal(size=(18, 21)) for seed in range(4)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for parallel, serial in zip(
            pool.map(spike_mask, images), map(spike_mask, images), strict=True
        ):
            np.testing.assert_array_equal(parallel, serial)


def test_ftoolss_parity():
    ftoolss = pytest.importorskip("ftoolss")
    image = np.random.default_rng(2026).normal(size=(65, 75))
    image[::7, ::5] += 2.5
    image[1, ::3] = np.nan
    for window in [(5, 3), (3, 5)]:
        clipped = ftoolss.sigma_clip(
            image, window[::-1], sigma_lower=np.inf, sigma_upper=2.65, maxiters=10
        )
        np.testing.assert_array_equal(
            spike_mask(image, window=window), np.isfinite(image) & np.isnan(clipped)
        )
