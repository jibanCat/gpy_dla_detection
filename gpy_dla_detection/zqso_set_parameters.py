"""
z_qso_set_parameters.py : sets various parameters for the zQSO estimation
pipeline.

See the original code here for mode details:
https://github.com/sbird/gp_qso_redshift/blob/master/set_parameters.m

Note:
----
Physical constants are handled as class attrs
Pipeline parameters are handled as instance attrs
Lambda functions are handled as instance methods
"""
import numpy as np

from .set_parameters import Parameters


class ZParameters(Parameters):
    def __init__(
        self,
        # normalization parameters
        # I use 1216 is basically because I want integer in my saved filenames
        normalization_min_lambda: float = 1216.0
        - 40.0,  # range of rest wavelengths to use   Å
        normalization_max_lambda=1216.0 + 40.0,  # for flux normalization
        # null model parameters
        min_lambda: float = 910.0,  # range of rest wavelengths to       Å
        max_lambda: float = 3000.0,  #   model
        dlambda: float = 0.25,  # separation of wavelength grid      Å
        k: int = 20,  # rank of non-diagonal contribution
        max_noise_variance: float = 4.0
        ** 2,  # maximum pixel noise allowed during model training
        num_zqso_samples: int = 10000,  # number of parameter samples
        minFunc_options: dict = {  # optimization options for model fitting
            "MaxIter": 4000,
            "MaxFunEvals": 8000,
        },
    ):
        self.normalization_min_lambda = normalization_min_lambda
        self.normalization_max_lambda = normalization_max_lambda

        self.min_lambda = min_lambda
        self.max_lambda = max_lambda

        self.dlambda = dlambda

        self.k = k

        self.max_noise_variance = max_noise_variance

        self.num_zqso_samples = num_zqso_samples

        self.minFunc_options = minFunc_options
