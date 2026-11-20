"""
Measure the end-to-end time to flag one detector image after warming the Rust worker pool.
"""

import argparse
from pathlib import Path
from statistics import median
from time import perf_counter_ns

import numpy as np

from spikemask_rs import spike_mask


def main() -> None:
    """
    Time the public API on a two-dimensional NumPy file.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="detector image saved with np.save")
    parser.add_argument("--window", type=int, nargs=2, default=(7, 5), metavar=("ROWS", "COLS"))
    args = parser.parse_args()
    image = np.load(args.image, allow_pickle=False)
    expected = spike_mask(image, window=args.window)
    times = []
    for _ in range(41):
        start = perf_counter_ns()
        result = spike_mask(image, window=args.window)
        times.append((perf_counter_ns() - start) / 1_000_000)
        np.testing.assert_array_equal(result, expected)
    print(f"{image.shape}, {image.dtype}, {expected.sum()} flagged, median {median(times):.3f} ms")  # noqa: T201


if __name__ == "__main__":
    main()
