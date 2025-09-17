import numpy as np

from metrics import Metrics
from data import Metadata
from sampler import Sampler
from utils import Utilities


class MCMetrics:
    
    def check_cm(self, cm):
        """
        Checks that the confusion matrix is square and non-negative.
        """
        cm = np.array(cm)
        if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
            raise ValueError("Confusion matrix must be a square matrix.")
        if np.any(cm < 0):
            raise ValueError("Confusion matrix must have non-negative entries.")
        return cm

    def __init__(self, **kwargs):
        
        if 'cm' in kwargs:
            self.cm = kwargs.get('cm')

