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
The comparison covers the defaults, the demo's settings in both window orientations,
detector dtypes and successive passes. Two differences are known: infinite samples
are ignored here like NaN, while ``ftoolss`` keeps them in its statistics; and an
``ftoolss`` compiled with fused multiply-add (``-march=native``) can differ by a pixel
that lies exactly on the cut.

Default settings
----------------

The defaults are a seven-row, five-column window, a cut of 3.0 standard deviations
and up to ten passes. The IDL demo uses five rows, three columns and 2.65;
``spike_mask(exposure, window=(5, 3), sigma_upper=2.65)`` reproduces it.

The defaults were chosen from a sweep of 855 settings on mocked Level 1 frames
carrying the cosmic-ray hits of an IRIS dark from the South Atlantic Anomaly.
Percent of hit pixels found, and of pixels without a hit that were flagged:

======  ==============  ===========  ================  =============
Frame   Found, default  Found, demo  Flagged, default  Flagged, demo
======  ==============  ===========  ================  =============
SG 108  93.1            88.9         0.23              0.52
SG 171  79.7            78.6         0.20              0.43
SG 284  82.3            80.6         0.10              0.31
CI 195  31.0            30.4         0.07              0.25
======  ==============  ===========  ================  =============

On IRIS sit-and-stare Si IV spectra of 10 September 2014, with the neighboring
exposures as the truth, the defaults found 89 percent of the hits on the dark half
of the detector and the demo's settings 63 percent.

Counting the cut
----------------

The cut needs only whether the median passes it. A larger sample leaves a smaller
difference from the center pixel, so the median passes exactly when more than half
of the samples do, and counting replaces sorting. Complete windows are tested for
eight adjacent centers at once; an x86 CPU with AVX2 uses a runtime-selected
implementation. Edges and windows with missing samples are counted one pixel at a
time. When exactly half of an even count passes, the detector averages the two
middle samples; ``an_even_window_split_by_the_cut_is_decided_by_its_median``
covers that case.

Later passes
------------

Only windows that overlap a newly flagged sample can change, so each pass retests
the neighbors of the new hits and keeps a flag found twice once. When those
neighborhoods hold as many pixels as the image, the image is scanned again instead.
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
