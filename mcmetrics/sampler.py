"""
Sampler class for performing posterior sampling via Markov Chain Monte Carlo.
"""

import numpy as np
import multiprocessing


class Sampler:
    """
    Sampler class for performing posterior sampling via Markov Chain Monte Carlo.
    """

    def __init__(self, size, **kwargs):
        self.size = size
        self.threads = kwargs.get('threads', 1)

    
    ############# HELPER FUNCTIONS ################

    def division_of_labor(self, samples):
        """
        Determines the number of samples to be assigned to each thread.
        """

        min_samples = samples // self.threads
        thread_spool = [min_samples] * self.threads

        for i in range(0, samples % self.threads):
            thread_spool[i] += 1

        return thread_spool
    
    
    

