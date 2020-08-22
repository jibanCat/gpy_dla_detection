"""
Parameter prior for zQSO estimation. 
"""

import numpy as np

from .zqso_set_parameters import ZParameters


class ZSamples:
    """
    A class to generate random samples for zQSOs.
    
    In the paper we used QMC samples. For convenience,
    we simply use linearly spacing sequence for the python code.
    
    TODO: in case we prefer a data driven prior, we can write another
    class to do that.
    """

    def __init__(self, params: ZParameters):
        self.params = params

        self.num_zqso_samples = params.num_zqso_samples

    def sample_z_qsos(
        self, z_qso_min: float = 2.14, z_qso_max: float = 6.16
    ) -> np.ndarray:
        return np.linspace(z_qso_min, z_qso_max, self.num_zqso_samples)
