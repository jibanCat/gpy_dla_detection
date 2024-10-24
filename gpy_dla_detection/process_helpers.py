"""
Helper functions for processing the results of the DLA detection algorithm.
"""

import numpy as np
import h5py
from typing import List


def initialize_results(num_spectra: int, max_dlas: int, num_dla_samples: int) -> dict:
    """
    Initialize the results dictionary to store outputs for all spectra.

    This function creates a dictionary that will store various results for each spectrum processed
    by the DLA detection algorithm. It initializes arrays with NaN (or zeros where appropriate)
    for each spectrum's output, which includes priors, likelihoods, posteriors, DLA parameters, and more.

    Parameters:
    ----------
    num_spectra : int
        The number of spectra being processed.
    max_dlas : int
        The maximum number of DLAs to model.
    num_dla_samples : int
        The number of DLA samples to process.

    Returns:
    --------
    results : dict
        A dictionary initialized with NaN or zero arrays to store results for each spectrum.
        Keys in the dictionary represent different computed results from the Bayesian model.

    Keys:
    -----
    min_z_dlas : Minimum redshift of the DLAs detected for each spectrum.
    max_z_dlas : Maximum redshift of the DLAs detected for each spectrum.
    log_priors_no_dla : Log prior probabilities for the no-DLA model.
    log_priors_dla : Log prior probabilities for DLA models (up to `max_dlas` DLAs).
    log_likelihoods_no_dla : Log likelihood for the no-DLA model.
    log_likelihoods_dla : Log likelihoods for DLA models.
    log_posteriors_no_dla : Log posteriors for the no-DLA model.
    log_posteriors_dla : Log posteriors for DLA models.
    sample_log_likelihoods_dla : Sampled log likelihoods for DLA models.
    base_sample_inds : Indices of base samples used in the sampling process.
    MAP_z_dlas : Maximum a posteriori (MAP) redshift estimates for DLAs.
    MAP_log_nhis : Maximum a posteriori (MAP) log N_HI estimates for DLAs.
    model_posteriors : Posterior probabilities for each model.
    p_dlas : Posterior probability of DLA models.
    p_no_dlas : Posterior probability of the no-DLA model.
    """

    results = {
        "min_z_dlas": np.full(
            (num_spectra,), np.nan
        ),  # Minimum DLA redshift for each spectrum
        "max_z_dlas": np.full(
            (num_spectra,), np.nan
        ),  # Maximum DLA redshift for each spectrum
        "log_priors_no_dla": np.full(
            (num_spectra,), np.nan
        ),  # Log prior for no-DLA model
        "log_priors_dla": np.full(
            (num_spectra, max_dlas), np.nan
        ),  # Log priors for DLA models
        "log_likelihoods_no_dla": np.full(
            (num_spectra,), np.nan
        ),  # Log likelihood for no-DLA model
        "log_likelihoods_dla": np.full(
            (num_spectra, max_dlas), np.nan
        ),  # Log likelihoods for DLA models
        "log_posteriors_no_dla": np.full(
            (num_spectra,), np.nan
        ),  # Log posteriors for no-DLA model
        "log_posteriors_dla": np.full(
            (num_spectra, max_dlas), np.nan
        ),  # Log posteriors for DLA models
        # "sample_log_likelihoods_dla": np.full(
        #     (num_spectra, num_dla_samples, max_dlas), np.nan
        # ),  # Sampled log likelihoods for DLA models
        # Correct shape for base_sample_inds: (num_spectra, max_dlas - 1, num_dla_samples)
        "base_sample_inds": np.zeros(
            (num_spectra, max_dlas - 1, num_dla_samples), dtype=np.int32
        ),  # Indices for base samples
        "MAP_z_dlas": np.full(
            (num_spectra, max_dlas), np.nan
        ),  # MAP redshift estimates for DLAs
        "MAP_log_nhis": np.full(
            (num_spectra, max_dlas), np.nan
        ),  # MAP log N_HI estimates for DLAs
        "z_dla_errs": np.full(
            (num_spectra, max_dlas), np.nan
        ),  # 1-sigma errors for DLA redshifts
        "log_nhi_errs": np.full(
            (num_spectra, max_dlas), np.nan
        ),  # 1-sigma errors for log N_HI values
        "model_posteriors": np.full(
            (num_spectra, 1 + 1 + max_dlas), np.nan
        ),  # Model posterior probabilities
        "p_dlas": np.full(
            (num_spectra,), np.nan
        ),  # Posterior probability for DLA models
        "p_no_dlas": np.full(
            (num_spectra,), np.nan
        ),  # Posterior probability for no-DLA model
        "sample_z_dlas": np.full(
            (num_spectra, num_dla_samples), np.nan
        ),  # Sampled redshifts for DLAs
        "log_nhi_samples": np.full(
            (num_spectra, num_dla_samples), np.nan
        ),  # Sampled log N_HI values
    }
    return results


def save_results_to_hdf5(
    filename: str, results: dict, spectrum_ids: List[str], z_qsos: np.ndarray
) -> None:
    """
    Save the results of the DLA detection process into an HDF5 file.

    This function writes the results from Bayesian model selection and DLA detection into an
    HDF5 file. It saves all relevant information, including spectrum IDs, QSO redshifts,
    and computed posteriors, priors, and likelihoods.

    Parameters:
    ----------
    filename : str
        The name of the HDF5 file to save the results.
    results : dict
        The results dictionary containing the processed outputs (e.g., priors, likelihoods, posteriors).
    spectrum_ids : List[str]
        List of spectrum IDs that were processed.
    z_qsos : np.ndarray
        Array of redshift values for each Quasi-Stellar Object (QSO) corresponding to the spectra.

    Keys in `results`:
    -----------------
    Each key in the `results` dictionary corresponds to a specific output of the DLA detection process.
    They include priors, likelihoods, and model estimates for each spectrum.
    """

    with h5py.File(filename, "w") as f:
        # Save spectrum IDs and QSO redshifts
        f.create_dataset(
            "spectrum_ids", data=np.array(spectrum_ids, dtype="S")
        )  # Save spectrum IDs as strings
        f.create_dataset("z_qsos", data=z_qsos)  # Save QSO redshifts

        # Loop through the results dictionary and save each key-value pair as an HDF5 dataset
        for key, value in results.items():
            f.create_dataset(key, data=value)  # Save each result in the HDF5 file
