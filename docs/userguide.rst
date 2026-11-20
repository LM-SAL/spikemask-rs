Flagging an exposure
====================

``spike_mask`` returns a Boolean mask and leaves the exposure intact.

>>> import numpy as np
>>> from spikemask_rs import spike_mask
>>> image = np.full((20, 30), 100.0, dtype=np.float32)
>>> image[10, 15] = 10_000
>>> image[1, 1] = np.nan
>>> mask = spike_mask(image)
>>> np.argwhere(mask).tolist()
[[10, 15]]
>>> float(image[10, 15])
10000.0

Set ``window=(rows, columns)`` to choose the neighborhood. Both sizes must be
positive and odd. At the edges the detector uses the available pixels. Each pass
excludes earlier hits, which can reveal a weaker spike next to a strong one.

.. _stage-worker-counts:

Choosing a worker count
-----------------------

Use :func:`~spikemask_rs.set_workers` to share a thread pool across calls.
Pass ``workers`` to :func:`~spikemask_rs.spike_mask` to override the count for one call.

>>> from spikemask_rs import set_workers
>>> with set_workers(4):
...     mask = spike_mask(image)
...     single_worker_mask = spike_mask(image, workers=1)
>>> np.testing.assert_array_equal(mask, single_worker_mask)

Leaving the block restores the previous setting. Outside a worker context,
calls with ``workers=None`` use Rayon's default pool, sized from the available
CPUs. Set ``RAYON_NUM_THREADS`` before starting Python to size that pool.

When running calls concurrently, budget threads across their pools. A new Python
thread inherits the worker setting if Python copies the context, the default on
free-threaded builds. Otherwise, set the count in the new thread.

For multiprocessing, use ``spawn`` or ``forkserver`` and set the count in each
child process. Forking after the first parallel call can hang the child.
