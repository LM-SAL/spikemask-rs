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
The Python parity tests compare the output with a complete rescan using NumPy.

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

On a Ryzen 9 9950X, the default detector took 1.20 ms per 328 by 1,024 float32
MUSE demo exposure using 16 Rayon workers. This is the median of 61 warm Python
calls on Linux including input conversion and mask allocation. The float64 input
took 1.12 ms with 16 workers and 1.36 ms with the default worker count. All
2,103 flagged pixels matched the demo's ``ftoolss`` mask. On the same float64 image,
stock ``rsliding`` at revision ``68f28a8b`` took 28.7 ms with 32 workers and
``ftoolss`` 6.1.21 took 153 ms (medians of seven warm calls). Runtime depends on
the window, image size, density of hits and CPU.
