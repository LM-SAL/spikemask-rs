# spikemask-rs

Flag candidate cosmic-ray hits in a two-dimensional detector exposure using local upper sigma clipping. The function returns a Boolean mask and leaves the exposure intact.

```python
from spikemask_rs import spike_mask

suspected_hits = spike_mask(exposure)
```

Install with `python -m pip install spikemask-rs`.
See the [user guide](https://spikemask-rs.readthedocs.io/en/latest/userguide.html) for examples,
the [installation guide](https://spikemask-rs.readthedocs.io/en/latest/installation.html#development) for development commands,
and the [design notes](https://spikemask-rs.readthedocs.io/en/latest/design-notes.html) for the detection rule and choice of defaults.
