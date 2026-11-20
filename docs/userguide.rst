Flagging an exposure
====================

A flag identifies a suspected hit at the same row and column as the original measurement.

>>> import numpy as np
>>> from spikes_rs import spike_mask
>>> image = np.full((20, 30), 100.0)
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

Stage worker counts
-------------------

Use a positive ``workers`` argument on ``spike_mask`` for one call.
For repeated calls, create the pool once in the stage that runs them::

    from spikes_rs import spike_mask, set_workers

    with set_workers(4):
        masks = [spike_mask(image) for image in exposures]

The pool applies to ``spikes_rs`` calls made in that context.
