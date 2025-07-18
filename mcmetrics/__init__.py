"""
Monte Carlo Metrics

Bayesian resampling utilities for confusion matrix based model evaluation.
"""

# public objects
from importlib.metadata import version, PackageNotFoundError
from .main import MCMetrics                                  # <- your class

# Optional: convenience factory so users can write mcm(...)
def mcmetrics(*args, **kwargs):
    """Shorthand factory: returns `MCMetrics(*args, **kwargs)`."""
    return MCMetrics(*args, **kwargs)

# Alias people seem to try
mc = mcmetrics

# exported by `from mcmetrics import *`
__all__ = ["MCMetrics", "mcmetrics", "mc"]

# version string
try:
    # if installed from a wheel
    __version__ = version(__name__)
except PackageNotFoundError:
    # running from a source tree / editable install
    __version__ = "0.0.0+dev"
