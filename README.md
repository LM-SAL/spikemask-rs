# spikes-rs

Detect candidate cosmic-ray hits in a two-dimensional detector exposure using local upper sigma clipping. The Python function returns a Boolean mask and preserves every measured value.

```python
from spikes_rs import spike_mask

suspected_hits = spike_mask(exposure, window=(7, 5), sigma_upper=3.0, max_iters=10)
```

Window dimensions are in rows and columns. The detector skips nonfinite pixels and truncates the window at image edges.

Build from this checkout with `python -m pip install .`. See [installation](docs/installation.rst) for development commands and [design notes](docs/design-notes.rst) for the algorithm and its relationship to the MUSE clipping demo and rsliding.
