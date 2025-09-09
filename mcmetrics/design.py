"""
This module contains classes for the implementation of 
Bayesian study design when pertaining to classification and
regression tasks. 
"""

import numpy as np
from data import Metadata

class Assurance:

    """
    Bayesian assurance
    """

    def __init__(self, n, **kwargs):
        
        for key, value in kwargs.items():
            
            if key in Metadata.metric_metadata.keys():
                setattr(self, key, value)

                # check if a prior is sp





class Stratified:

    """
    Stratified case-control design
    """

    def __init__(self):
        pass


class Sequential:

    """
    Sequential design
    """

    def __init__(self):
        pass