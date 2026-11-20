Design notes
============

Detection rule
--------------

Each pass flags pixels more than ``sigma_upper`` population standard deviations
above the local median. Both statistics use the finite samples in the window,
including the tested pixel. The detector computes the mean before summing squared
deviations. It excludes flagged samples on the next pass and leaves missing
values unflagged.

The MUSE IDL demo uses ``ftoolss.sigma_clip`` with a lower limit of 1,000 standard
deviations. This detector tests only the upper tail. Its Rust implementation is
independent of Alfred Voyeux's `rsliding <https://git.ias.u-psud.fr/avoyeux/rsliding/-/blob/68f28a8b5efbf6567f62eb88434d73c594f71209/rust/src/core/sliding_sigma_clipping.rs>`_,
which uses the same iterative rule.

``test_c_parity.py`` compares the mask with pixels newly clipped by ``ftoolss``
across window orientations, detector dtypes and pass counts. The implementations
differ on infinities, which ``ftoolss`` includes in its statistics. Compiling
``ftoolss`` with fused multiply-add can also change a flag at the clipping threshold.

Default settings
----------------

The default window is ``(7, 5)`` with a 3.0-sigma cut. The IDL demo uses
``(5, 3)`` and 2.65 sigma. Both default to a maximum of ten passes.

The defaults were chosen by testing 855 settings on mock MUSE Level 1 frames
with cosmic-ray hits from an IRIS dark exposure taken in the South Atlantic
Anomaly. “Found” is the percentage of hit pixels detected. “Flagged” is the
percentage of pixels without a hit that were flagged.

======  ==============  ===========  ================  =============
Frame   Found, default  Found, demo  Flagged, default  Flagged, demo
======  ==============  ===========  ================  =============
SG 108  93.1            88.9         0.23              0.52
SG 171  79.7            78.6         0.20              0.43
SG 284  82.3            80.6         0.10              0.31
CI 195  31.0            30.4         0.07              0.25
======  ==============  ===========  ================  =============

On the dark half of IRIS sit-and-stare Si IV spectra from 10 September 2014,
the defaults detected 89 percent of hits identified from neighboring exposures.
The demo settings detected 63 percent.

.. _counting-the-cut:

Testing the median
------------------

``Detector::is_hit`` counts the samples more than the threshold below the center
pixel. A strict majority flags the center. If exactly half qualify, the two
middle values straddle the cut. Their average decides whether to flag the pixel.
``an_even_window_split_by_the_cut_is_decided_by_its_median`` tests both outcomes.

``Detector::interior`` tests eight adjacent centers at once, using AVX2 on x86
when available. Edges and windows with missing samples use ``Detector::is_hit``.
``test_matches_a_full_rescan`` compares both paths with NumPy for eight window shapes.

Later passes
------------

Only windows containing a newly flagged sample can change. ``Detector::rescan``
retests their centers and removes duplicate hits. It scans the whole image when
the estimated number of visits, counting overlaps, reaches the image size.
``a_nearby_spike_appears_on_the_second_pass`` and ``test_matches_a_full_rescan``
check successive passes.

Measured throughput
-------------------

Median times on a Ryzen 9 9950X for mock int32 Level 1 frames. The detector
reused its worker pool. Times include preparing the clipping buffer and
allocating the mask.

========================  ========  ============  ============
Implementation            Settings  SG 1024×1024  CI 2048×4096
========================  ========  ============  ============
spikemask-rs, 1 worker    Default   14 ms         94 ms
spikemask-rs, 16 workers  Default   1.9 ms        12 ms
spikemask-rs, 1 worker    Demo      7.1 ms        56 ms
ftoolss 6.1.21            Default   1.2 s         6.2 s
ftoolss 6.1.21            Demo      240 ms        1.9 s
========================  ========  ============  ============

Masks matched ``ftoolss`` at both settings. On the float64 demo exposure,
``rsliding`` revision ``68f28a8b`` took 28.7 ms with 32 workers and ``ftoolss``
took 153 ms (medians of seven warm calls).

Worker pools
------------

``set_workers(n)`` keeps a Rayon pool in the current Python context.
``resolve_pool`` reuses it for calls requesting the same count and creates a pool
for a different count. Calls with neither use Rayon's default pool.
``test_call_and_scoped_workers_reach_the_kernel`` checks pool reuse, and
``the_kernel_inherits_the_selected_pool`` checks which pool executes the kernel.

Detector dtypes
---------------

``working_copy`` widens native int32 and float32 samples directly into the
float64 clipping buffer. This preserves single-count differences above 2²⁴ in
int32 exposures and avoids float32 rounding near a clipping threshold.
``test_int32_counts_resolve_a_single_dn_at_large_offsets`` checks the one-count case.
