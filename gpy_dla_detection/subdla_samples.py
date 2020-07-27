"""
subdla_samples.py : generate the Quasi-Monte Carlo samples for
calculating log model evidence of subDLA model
"""
import numpy as np
import h5py
from .set_parameters import Parameters
from .model_priors import PriorCatalog
from .dla_samples import DLASamples


class SubDLASamples(DLASamples):
    """
    A class to generate and store the QMC samples for DLAs:
    theta = (z_dla, logNHI) = (redshift of DLA, column density of DLA)
    
    :attr offset_samples: used for z_lls samples
    :attr nhi_samples: nhi samples for subDLAs

    Note: lls here means subDLAs.
    """

    def __init__(
        self, params: Parameters, prior: PriorCatalog, extrapolate_min_log_nhi: float
    ):
        # for DLA logNHI, we have a combination of a data-driven prior and an uniform prior.
        # we extrapolate the peak value of the DLA logNHI prior (around logNHI = 20.03)
        # uniformly to logNHI = extrapolate_min_log_nhi.
        self.extrapolate_min_log_nhi = extrapolate_min_log_nhi

        super().__init__(params, prior)

    def log_nhi_prior(self):
        NotImplementedError

    def z_dla_prior(self):
        NotImplementedError

    @property
    def offset_samples(self):
        NotImplementedError

    @property
    def log_nhi_samples(self):
        NotImplementedError

    @property
    def nhi_samples(self):
        NotImplementedError

    @property
    def Z_lls(self):
        """
        the normalization factor (partition function) of subDLA logNHI prior. 
        """
        NotImplementedError

    @property
    def Z_dla(self):
        """
        the normalization factor (partition function) of DLA logNHI prior. 
        """
        NotImplementedError


class SubDLASamplesMAT(SubDLASamples):
    """
    Load subDLA samples from .mat file, which generated from
    Ho-Bird-Garnett's multi_dlas/set_lls_parameters.m
    """

    def __init__(
        self,
        params: Parameters,
        prior: PriorCatalog,
        sub_dla_samples_file: str = "subdla_samples.mat",
    ):
        sub_dla_samples = h5py.File(sub_dla_samples_file, "r")

        super().__init__(
            params, prior, sub_dla_samples["extrapolate_min_log_nhi"][0, 0]
        )

        assert self.alpha == sub_dla_samples["alpha"][0, 0]
        assert self.num_dla_samples == sub_dla_samples["num_dla_samples"][0, 0]

        self._offset_samples = sub_dla_samples["offset_samples"][:, 0]
        self._log_nhi_samples = sub_dla_samples["lls_log_nhi_samples"][:, 0]
        self._nhi_samples = sub_dla_samples["lls_nhi_samples"][:, 0]

        # load normalization factors (partition functions)
        self._Z_dla = sub_dla_samples["Z_dla"][0, 0]
        self._Z_lls = sub_dla_samples["Z_lls"][0, 0]

    @property
    def Z_dla(self) -> float:
        return self._Z_dla

    @property
    def Z_lls(self) -> float:
        return self._Z_lls

    @property
    def offset_samples(self) -> np.ndarray:
        return self._offset_samples

    @property
    def log_nhi_samples(self) -> np.ndarray:
        return self._log_nhi_samples

    @property
    def nhi_samples(self) -> np.ndarray:
        return self._nhi_samples

    def sample_z_lls(self, wavelengths: np.ndarray, z_qso: float) -> np.ndarray:
        sample_z_lls = (
            self.params.min_z_dla(wavelengths, z_qso)
            + (
                self.params.max_z_dla(wavelengths, z_qso)
                - self.params.min_z_dla(wavelengths, z_qso)
            )
            * self._offset_samples
        )

        return sample_z_lls
