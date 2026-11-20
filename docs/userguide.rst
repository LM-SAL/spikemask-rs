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
