Design notes
============

Detection rule
--------------

The MUSE IDL cosmic-ray demo calls ``ftoolss.sigma_clip`` with a three-column,
five-row window, an upper limit of 2.65 standard deviations, a lower limit of
1,000, and up to ten passes. This detector tests the upper tail only. The Rust
implementation was written independently for a mask-only API. The iterative rule
also appears in Alfred Voyeux's `rsliding <https://git.ias.u-psud.fr/avoyeux/rsliding/-/blob/68f28a8b5efbf6567f62eb88434d73c594f71209/rust/src/core/sliding_sigma_clipping.rs>`_.

The median and the population standard deviation use the same finite samples,
including the tested pixel. Standard deviation is calculated around the mean in two
passes. The strict upper cut is applied simultaneously across the image before any
flags from that pass affect the next one. Original missing values remain unflagged.
The tests compare the mask with a complete NumPy rescan. In addition,
``test_c_parity.py`` compares every flag with the newly clipped pixels from
``ftoolss.sigma_clip``, which calls the C median and standard-deviation kernels.
The comparison covers detector dtypes, both window orientations and successive passes.

Default window
--------------

The complete five-row, three-column window contains 15 samples. Its median is the
eighth ordered value. Exactly the same upper cut can be evaluated by comparing
the center pixel with all 15 samples and counting how often the difference exceeds
the cut. Eight successes are sufficient. This lets Rust evaluate adjacent columns
together. An x86 CPU with AVX2 uses a runtime-selected implementation. Edges,
missing samples and other window shapes use the median of the available samples
directly, averaging the two middle values when the count is even.

Later passes
------------

Only windows that overlap a newly flagged sample need another visit. For sparse hits
the detector collects these candidate pixels and removes duplicates. When that list
could be as large as the image, scanning the image again bounds temporary storage.
Both paths use the same detection rule.

Measured throughput
-------------------

On a Ryzen 9 9950X with a pool reused across 41 calls, the 328 by 1,024
MUSE demo exposure took 5.09 ms with one worker and 1.15 ms with 16 workers
(median, float32 input). Rounding that exposure to int32 gave 5.00 ms and
1.08 ms respectively. Both masks matched ``ftoolss`` on the corresponding
input. The timings include preparing the clipping buffer and allocating the mask.
An earlier comparison on the float64 demo found 28.7 ms for stock ``rsliding``
at revision ``68f28a8b`` with 32 workers and 153 ms for ``ftoolss`` 6.1.21
(medians of seven warm calls). Runtime depends on the window, image size,
density of hits and CPU.

Worker pools
------------

A call with ``workers=n`` uses a local Rayon pool of that size.
If the active pool already has that size, the call reuses it.
``set_workers(n)`` creates one pool for a block of calls in the current Python
context. The binding runs the parallel kernel inside the selected pool.
Calls outside either setting use Rayon's default. Separate processes must enter
their own context. The pool and context behavior are covered by the Rust
worker-pool test and the Python worker tests.


Detector dtypes
---------------

The binding accepts native int32 and float32 exposures directly. It widens
each sample into its mutable float64 clipping buffer in one pass; all int32
counts, including single-DN differences above 2²⁴, are exactly representable.
This also preserves the clipping decisions for float32 images near a threshold.
