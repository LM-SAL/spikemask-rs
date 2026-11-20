0.1.0 (2026-11-20)
==================

New Features
------------

- Added ``spikes_rs.spike_mask``, which flags candidate cosmic-ray hits in a two-dimensional detector exposure by iterative upper sigma clipping. It defaults to a seven-row, five-column window and a cut of 3.0 standard deviations; ``window=(5, 3)`` and ``sigma_upper=2.65`` reproduce the MUSE IDL demo. (#2)
- Added the ``workers`` option of ``spikes_rs.spike_mask`` and ``spikes_rs.set_workers``, which set the number of Rayon threads for one call and for a block of calls. (#2)
