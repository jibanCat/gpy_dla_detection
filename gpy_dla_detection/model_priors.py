"""
model_priors.py : model priors are handled by a prior_catalog,
so we build a class to control the catalog.
"""
from typing import Tuple

import numpy as np
import h5py
from .set_parameters import Parameters


class PriorCatalog:
    """
    A prior catalog class, corresponding to prior_catalog in
    Roman's MATLAB code. It holds the catalog of prior used for
    training, so it will give the number of DLAs conditioned on
    a given z_QSO:
    
    % DLA existence prior
    less_ind = (prior.z_qsos < (z_qso + prior_z_qso_increase));

    then the prior is calculated via P(DLA|zQSO) = M / N,
    M =  this_num_dlas    = nnz(prior.dla_ind(less_ind));
    N =  this_num_quasars = nnz(less_ind);
    
    Note: default set to dr9q_concordance

    :param param: default parameters set in the set_parameters.
    :param catalog_name: the catalog.mat built by Roman's build_catalog.m (this basically is
        used to find zQSOs, could be replaced by other files in the future if we want to
        completely depcreate the MATLAB codes)
    :param los_catalog: the line-of-sight searched by this catalog.
    :param dla_catalog: the (thingIDs, zDLAs, logNHIs) searched by the catalog.
    :param prior_ind: the string to indicate the prior ind to be selected, this is meant
        to be similar to Roman's code, but since the hash table is not working in h5py
        so I change the format slightly. Please see in the default value.
    """

    def __init__(
        self,
        params: Parameters,
        catalog_name: str = "catalog.mat",
        los_catalog: str = "los_catalog",
        dla_catalog: str = "dla_catalog",
        dla_catalog_name: str = "dr9q_concordance",
        prior_ind: str = str(
            "self.in_dr9 &" " self.los_ind &" " (self.filter_flags == 0)"
        ),
    ):
        # set_parameters
        self.params = params

        # load the mat file
        with h5py.File(catalog_name, "r") as catalog:
            # prepare the zQSOs and thingIDs
            self.in_dr9 = catalog["in_dr9"][0, :].astype(np.bool)
            self.in_dr10 = catalog["in_dr10"][0, :].astype(np.bool)
            self.z_qsos = catalog["z_qsos"][0, :]
            self.filter_flags = catalog["filter_flags"][0, :]
            self.thing_ids = catalog["thing_ids"][0, :].astype(np.int)

        # load the DLA catalog from separated files
        if dla_catalog_name == "dr9q_concordance":
            thing_ids_los, thing_ids_dla, z_dlas, log_nhis = self.load_concordance(
                los_catalog, dla_catalog
            )
        else:
            self.load_catalog(los_catalog, dla_catalog)

        # prepare the dla_ind and los_ind, which are the same as Roman's MATLAB code
        self.los_ind: np.ndarray = np.isin(self.thing_ids, thing_ids_los)
        self.dla_ind: np.ndarray = np.isin(self.thing_ids, thing_ids_dla)
        assert self.los_ind.shape[0] == self.z_qsos.shape[0]

        # assign z_dlas and log_nhis
        self.z_dlas: np.ndarray = np.empty(self.dla_ind.shape)
        self.log_nhis: np.ndarray = np.empty(self.dla_ind.shape)
        self.z_dlas[:] = np.nan
        self.log_nhis[:] = np.nan
        self.z_dlas[self.dla_ind] = z_dlas[np.isin(thing_ids_dla, self.thing_ids)]
        self.log_nhis[self.dla_ind] = log_nhis[np.isin(thing_ids_dla, self.thing_ids)]

        # reselect prior ind; this is irreversible so put in __init__; we won't be able
        # to re-evaluate this multiple times, so not put into a separated method.
        if type(prior_ind) is str:
            prior_ind = eval(prior_ind)

        self.thing_ids = self.thing_ids[prior_ind]
        self.z_qsos = self.z_qsos[prior_ind]
        self.dla_ind = self.dla_ind[prior_ind]
        self.z_dlas = self.z_dlas[prior_ind]
        self.log_nhis = self.log_nhis[prior_ind]

        # filter out DLAs from prior catalog corresponding to region of spectrum below
        # Ly∞ QSO rest
        self.dla_ind = self.filter_z_dlas(self.dla_ind, self.z_dlas)

    def load_concordance(
        self, los_concordance: str, dla_concordance: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        load catalog from concordance. Need to run Roman's download_catalog.sh first.
        """
        dla_catalog = np.loadtxt(dla_concordance)
        los_catalog = np.loadtxt(los_concordance)

        thing_ids_dla = dla_catalog[:, 0].astype(np.int)
        z_dlas = dla_catalog[:, 1]
        log_nhis = dla_catalog[:, 2]

        thing_ids_los = los_catalog.astype(np.int)
        return thing_ids_los, thing_ids_dla, z_dlas, log_nhis

    def load_catalog(self, los_catalog: str, dla_catalog: str) -> None:
        NotImplementedError

    def filter_z_dlas(self, dla_ind: np.ndarray, z_dlas: np.ndarray) -> np.ndarray:
        """
        filter out DLAs from prior catalog corresponding to region of spectrum below
        Ly∞ QSO rest 
        """
        num_dlas = np.sum(dla_ind)

        z_dlas = self.z_dlas[dla_ind]
        z_qsos = self.z_qsos[dla_ind]

        ind = Parameters.observed_wavelengths(
            Parameters.lya_wavelength, z_dlas
        ) < Parameters.observed_wavelengths(Parameters.lyman_limit, z_qsos)

        # this is not working, interesting. seems like python create a new variable for dla_ind[dla_ind]
        # dla_ind[dla_ind][ind] = False
        real_index = np.where(dla_ind)[0]
        real_index = real_index[ind]
        dla_ind[real_index] = False

        # assert the number of decrease is correct
        assert num_dlas == (np.sum(ind) + np.sum(dla_ind))

        return dla_ind

    def less_ind(self, z_qso: float) -> Tuple[float, float]:
        """
        DLA existence prior: P(DLA | zQSO) = M / N
            M = num_dlas
            N = num_quasars

        :param z_qso: the quasar redshift to be conditioned on.
        :return: (this_num_dlas, this_num_quasars) 
        """
        # use QSOs with z < (z_QSO + x) for prior
        less_ind = self.z_qsos < (z_qso + self.params.prior_z_qso_increase)

        this_num_dlas = np.sum(self.dla_ind[less_ind])
        this_num_quasars = np.sum(less_ind)

        return this_num_dlas, this_num_quasars
