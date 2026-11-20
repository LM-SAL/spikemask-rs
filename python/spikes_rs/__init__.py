"""
Candidate cosmic-ray spike masks for detector exposures.
"""

from importlib.metadata import version as _get_version

from ._workers import set_workers
from .detection import spike_mask

__version__ = _get_version("spikes-rs")
__all__ = ["__version__", "set_workers", "spike_mask"]
