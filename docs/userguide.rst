Flagging an exposure
====================

A flag identifies a suspected hit at the same row and column as the original measurement.

>>> import numpy as np
>>> from spikes_rs import spike_mask
>>> image = np.full((20, 30), 100.0, dtype=np.float32)
>>> image[10, 15] = 10_000
>>> image[1, 1] = np.nan
>>> mask = spike_mask(image)
>>> np.argwhere(mask).tolist()
[[10, 15]]
>>> float(image[10, 15])
10000.0

Use ``window=(rows, columns)`` to change the local neighborhood. Both sizes must be
positive and odd. Edges use only the available pixels. Each pass recomputes local
statistics after excluding newly flagged pixels, which can reveal a weaker adjacent hit.
``window=(5, 3)`` and ``sigma_upper=2.65`` reproduce the MUSE IDL demo.

.. _stage-worker-counts:

Stage worker counts
-------------------

Pass a positive integer as ``workers=`` to :func:`~spikes_rs.spike_mask` for one call.
For a stage with repeated calls, :func:`~spikes_rs.set_workers` lets them reuse one Rayon pool.
An explicit count overrides the stage setting for that call. For example, these calls
use four workers and one worker respectively.

>>> from spikes_rs import set_workers
>>> with set_workers(4):
...     mask = spike_mask(image)
...     single_worker_mask = spike_mask(image, workers=1)
>>> np.testing.assert_array_equal(mask, single_worker_mask)

Leaving the block restores the previous setting. Each package manages its own pool.
A thread started inside the block inherits the setting only where Python copies the
context into new threads, the default on free-threaded builds; elsewhere, pass
``workers=`` or enter the block in that thread.
If stages run concurrently, budget workers across the active pools. For a stage running
in another process, enter the context in that process. Start it with ``spawn`` or
``forkserver``: a call in a process forked after the first call never returns.

With ``workers=None`` (the default), a call uses the active context or falls back to
Rayon's default pool. Set ``RAYON_NUM_THREADS`` before starting Python to configure
that fallback, for example ``RAYON_NUM_THREADS=4 python pipeline.py``. When unset,
Rayon chooses the default size from the available CPUs.
