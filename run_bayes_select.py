"""
Run Bayesian model selection on DESI spectra for Damped Lyman-Alpha systems (DLAs).

This script reads DESI spectra using DESISpectrumReader and processes each spectrum
with Bayesian model selection to detect DLAs.

The results are saved in an HDF5 file, and the script can optionally generate plots.
"""

import os
import time
import numpy as np
import h5py
from typing import List
from matplotlib import pyplot as plt

from gpy_dla_detection.set_parameters import Parameters
from gpy_dla_detection.model_priors import PriorCatalog
from gpy_dla_detection.null_gp import NullGPMAT
from gpy_dla_detection.dla_gp import DLAGPMAT
from gpy_dla_detection.subdla_gp import SubDLAGPMAT
from gpy_dla_detection.dla_samples import DLASamplesMAT
from gpy_dla_detection.subdla_samples import SubDLASamplesMAT
from gpy_dla_detection.bayesian_model_selection import BayesModelSelect
from gpy_dla_detection.plottings.plot_model import plot_dla_model

from gpy_dla_detection.desi_spectrum_reader import (
    DESISpectrumReader,
)
from collections import namedtuple
import argparse

from gpy_dla_detection.process_helpers import initialize_results, save_results_to_hdf5

from gpy_dla_detection.plottings.plot_model import (
    plot_samples_vs_this_mu,
    plot_real_spectrum_space,
)

# Namedtuple to store spectrum data
SpectrumData = namedtuple(
    "SpectrumData", ["wavelengths", "flux", "noise_variance", "pixel_mask"]
)


def process_qso(
    spectra_filename: str,
    zbest_filename: str,
    learned_file: str,
    catalog_name: str,
    los_catalog: str,
    dla_catalog: str,
    dla_samples_file: str,
    sub_dla_samples_file: str,
    min_z_separation: float,
    prev_tau_0: float,
    prev_beta: float,
    max_dlas: int = 4,
    broadening: bool = True,
    plot_figures: bool = False,
    max_workers: int = None,
    batch_size: int = 100,
):
    """
    Process DESI spectra with Bayesian model selection for DLA detection.

    Parameters:
    ----------
    spectra_filename : str
        The filename of the DESI spectra FITS file.
    zbest_filename : str
        The filename of the DESI redshift catalog (zbest-*.fits).
    learned_file : str
        The filename of the learned QSO model file.
    catalog_name : str
        The filename of the catalog file.
    los_catalog : str
        The filename of the line-of-sight catalog.
    dla_catalog : str
        The filename of the DLA catalog.
    dla_samples_file : str
        The filename of the DLA samples file.
    sub_dla_samples_file : str
        The filename of the sub-DLA samples file.
    min_z_separation : float
        Minimum redshift separation for DLA models.
    prev_tau_0 : float
        The previous value of tau_0 for modeling the Lyman forest noise.
    prev_beta : float
        The previous value of beta for modeling the Lyman forest noise.
    max_dlas : int, optional
        Maximum number of DLAs to model (default is 4).
    broadening : bool, optional
        Whether to include instrumental broadening (default is True).
    plot_figures : bool, optional
        If True, generates plots for each processed spectrum (default is False).
    max_workers : int, optional
        The number of workers for parallel processing (default is None).
    batch_size : int, optional
        The batch size for parallel model evidence computation (default is 100).
    """

    # Initialize parameters and priors
    params = Parameters()

    # Load prior catalog data
    prior = PriorCatalog(
        params,
        catalog_name,
        los_catalog,
        dla_catalog,
    )

    # Load DLA and subDLA sample data
    dla_samples = DLASamplesMAT(params, prior, dla_samples_file)
    subdla_samples = SubDLASamplesMAT(params, prior, sub_dla_samples_file)

    # Initialize Bayesian model selection
    bayes = BayesModelSelect([0, 1, max_dlas], 2)

    # Use DESISpectrumReader to read the spectra and redshift catalog
    reader = DESISpectrumReader(spectra_filename, zbest_filename)
    reader.read_spectra()
    reader.read_redshift_catalog()

    # Get redshift data and list of all spectrum IDs
    redshift_data = reader.get_redshift_data()
    all_spectrum_ids = reader.get_all_spectrum_ids()

    num_spectra = len(all_spectrum_ids)

    # Initialize arrays to store results
    results = initialize_results(num_spectra, max_dlas, params.num_dla_samples)

    # Loop through all spectra in the DESI file
    for idx, spectrum_id in enumerate(all_spectrum_ids):
        tic = time.time()

        # Get the redshift for the current spectrum
        z_qso = redshift_data["Z"][idx]

        # Get spectrum data (wavelengths, flux, noise_variance, pixel_mask)
        spectrum_data = reader.get_spectrum_data(spectrum_id)
        wavelengths = spectrum_data.wavelengths
        flux = spectrum_data.flux
        noise_variance = spectrum_data.noise_variance
        pixel_mask = spectrum_data.pixel_mask

        # Convert wavelengths to rest-frame
        rest_wavelengths = params.emitted_wavelengths(wavelengths, z_qso)

        # Instantiate the models once for all spectra
        null_gp = NullGPMAT(
            params,
            prior,
            learned_file=learned_file,
            prev_tau_0=prev_tau_0,
            prev_beta=prev_beta,
        )

        dla_gp = DLAGPMAT(
            params=params,
            prior=prior,
            dla_samples=dla_samples,
            min_z_separation=min_z_separation,
            learned_file=learned_file,
            broadening=broadening,
            prev_tau_0=prev_tau_0,
            prev_beta=prev_beta,
        )

        subdla_gp = SubDLAGPMAT(
            params=params,
            prior=prior,
            dla_samples=subdla_samples,
            min_z_separation=min_z_separation,
            learned_file=learned_file,
            broadening=broadening,
            prev_tau_0=prev_tau_0,
            prev_beta=prev_beta,
        )

        # Process the spectrum using the instantiated models
        process_single_spectrum(
            idx,
            spectrum_id,
            z_qso,
            wavelengths,
            rest_wavelengths,
            flux,
            noise_variance,
            pixel_mask,
            params,
            prior,
            dla_samples,
            subdla_samples,
            bayes,
            results,
            max_dlas,
            broadening,
            null_gp,  # Pass the NullGPMAT object
            dla_gp,  # Pass the DLAGPMAT object
            subdla_gp,  # Pass the SubDLAGPMAT object
            min_z_separation,
            plot_figures,
            max_workers,
            batch_size,
        )

        toc = time.time()
        print(
            f"Processed spectrum {idx + 1}/{num_spectra} (ID: {spectrum_id}), time spent: {(toc - tic) // 60:.0f}m {(toc - tic) % 60:.0f}s"
        )

    # Save all results into an HDF5 file
    save_results_to_hdf5(
        "processed_desi_spectra.h5", results, all_spectrum_ids, redshift_data["Z"]
    )


def process_single_spectrum(
    idx: int,
    spectrum_id: str,
    z_qso: float,
    wavelengths: np.ndarray,
    rest_wavelengths: np.ndarray,
    flux: np.ndarray,
    noise_variance: np.ndarray,
    pixel_mask: np.ndarray,
    params: Parameters,
    prior: PriorCatalog,
    dla_samples: DLASamplesMAT,
    subdla_samples: SubDLASamplesMAT,
    bayes: BayesModelSelect,
    results: dict,
    max_dlas: int,
    broadening: bool,
    gp: NullGPMAT,  # Pass already initialized NullGPMAT
    dla_gp: DLAGPMAT,  # Pass already initialized DLAGPMAT
    subdla_gp: SubDLAGPMAT,  # Pass already initialized SubDLAGPMAT
    min_z_separation: float,
    plot_figures: bool,
    max_workers: int,
    batch_size: int,
):
    """
    Process a single spectrum using pre-initialized Null, DLA, and SubDLA models.

    Parameters:
    ----------
    idx : int
        Index of the spectrum being processed.
    spectrum_id : str
        Identifier of the spectrum being processed.
    z_qso : float
        Redshift of the quasar for the spectrum.
    wavelengths : np.ndarray
        Observed wavelengths of the spectrum.
    rest_wavelengths : np.ndarray
        Rest-frame wavelengths of the spectrum.
    flux : np.ndarray
        Flux values of the spectrum.
    noise_variance : np.ndarray
        Noise variance per pixel in the spectrum.
    pixel_mask : np.ndarray
        Mask indicating which pixels are flagged as bad or good.
    params : Parameters
        Parameters instance for the analysis.
    prior : PriorCatalog
        Prior catalog instance for the analysis.
    dla_samples : DLASamplesMAT
        DLA samples data for the analysis.
    subdla_samples : SubDLASamplesMAT
        SubDLA samples data for the analysis.
    bayes : BayesModelSelect
        Bayesian model selection object for DLA detection.
    results : dict
        Dictionary to store the results.
    max_dlas : int
        Maximum number of DLAs to model.
    broadening : bool
        Whether to include instrumental broadening.
    gp : NullGPMAT
        Pre-initialized NullGPMAT object.
    dla_gp : DLAGPMAT
        Pre-initialized DLAGPMAT object.
    subdla_gp : SubDLAGPMAT
        Pre-initialized SubDLAGPMAT object.
    min_z_separation : float
        Minimum redshift separation for DLA models.
    plot_figures : bool
        If True, generates plots for each processed spectrum.
    max_workers : int
        Number of workers for parallel processing.
    batch_size : int
        Batch size for parallel model evidence computation.
    """

    # Set data for the Null model (no DLAs)
    gp.set_data(
        rest_wavelengths, flux, noise_variance, pixel_mask, z_qso, build_model=True
    )

    # Set data for the DLA model (up to max DLAs)
    dla_gp.set_data(
        rest_wavelengths, flux, noise_variance, pixel_mask, z_qso, build_model=True
    )

    # Set data for the Sub-DLA model
    subdla_gp.set_data(
        rest_wavelengths, flux, noise_variance, pixel_mask, z_qso, build_model=True
    )

    # Run Bayesian model selection with parallelized model evidence computation
    log_posteriors = bayes.model_selection(
        [gp, subdla_gp, dla_gp], z_qso, max_workers=max_workers, batch_size=batch_size
    )

    # Store results
    results["min_z_dlas"][idx] = dla_gp.params.min_z_dla(wavelengths, z_qso)
    results["max_z_dlas"][idx] = dla_gp.params.max_z_dla(wavelengths, z_qso)
    results["log_priors_no_dla"][idx] = bayes.log_priors[0]
    results["log_priors_dla"][idx, :] = bayes.log_priors[-max_dlas:]
    results["log_likelihoods_no_dla"][idx] = bayes.log_likelihoods[0]
    results["log_likelihoods_dla"][idx, :] = bayes.log_likelihoods[-max_dlas:]
    results["log_posteriors_no_dla"][idx] = bayes.log_posteriors[0]
    results["log_posteriors_dla"][idx, :] = bayes.log_posteriors[-max_dlas:]
    results["sample_log_likelihoods_dla"][idx, :, :] = dla_gp.sample_log_likelihoods
    results["base_sample_inds"][idx, :, :] = dla_gp.base_sample_inds

    # MAP results
    MAP_z_dla, MAP_log_nhi = dla_gp.maximum_a_posteriori()
    results["MAP_z_dlas"][idx, :, :] = MAP_z_dla
    results["MAP_log_nhis"][idx, :, :] = MAP_log_nhi

    # Save real-scale model posteriors
    results["model_posteriors"][idx, :] = bayes.model_posteriors[:]
    results["p_dlas"][idx] = bayes.p_dla
    results["p_no_dlas"][idx] = bayes.p_no_dla

    # Optionally generate plots
    if plot_figures:
        title = f"Spectrum {spectrum_id}; zQSO: {z_qso:.2f}"
        out_filename = f"spec-{str(idx).zfill(6)}"
        plot_samples_vs_this_mu(
            dla_gp, bayes, filename=out_filename, sub_dir="images", title=title
        )
        plt.clf()
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process DESI spectra with Bayesian model selection for DLA detection."
    )

    parser.add_argument(
        "--spectra_filename",
        required=True,
        help="DESI spectra FITS filename (e.g., spectra-*.fits).",
    )
    parser.add_argument(
        "--zbest_filename",
        required=True,
        help="DESI redshift catalog filename (zbest-*.fits).",
    )
    parser.add_argument(
        "--learned_file",
        default="data/dr12q/processed/learned_qso_model_lyseries_variance_kim_dr9q_minus_concordance.mat",
        help="Learned QSO model file (default: learned_qso_model_lyseries_variance_kim_dr9q_minus_concordance.mat).",
    )
    parser.add_argument(
        "--catalog_name",
        default="data/dr12q/processed/catalog.mat",
        help="Catalog filename (default: catalog.mat).",
    )
    parser.add_argument(
        "--los_catalog",
        default="data/dla_catalogs/dr9q_concordance/processed/los_catalog",
        help="Line-of-sight catalog filename (default: los_catalog).",
    )
    parser.add_argument(
        "--dla_catalog",
        default="data/dla_catalogs/dr9q_concordance/processed/dla_catalog",
        help="DLA catalog filename (default: dla_catalog).",
    )
    parser.add_argument(
        "--dla_samples_file",
        default="data/dr12q/processed/dla_samples_a03.mat",
        help="DLA samples file (default: dla_samples_a03.mat).",
    )
    parser.add_argument(
        "--sub_dla_samples_file",
        default="data/dr12q/processed/subdla_samples.mat",
        help="Sub-DLA samples file (default: subdla_samples.mat).",
    )
    parser.add_argument(
        "--min_z_separation",
        type=float,
        default=3000.0,
        help="Minimum redshift separation for DLA models (default: 3000.0).",
    )
    parser.add_argument(
        "--prev_tau_0",
        type=float,
        default=0.0023,
        help="Previous value for tau_0 (default: 0.0023).",
    )
    parser.add_argument(
        "--prev_beta",
        type=float,
        default=3.65,
        help="Previous value for beta (default: 3.65).",
    )
    parser.add_argument(
        "--max_dlas",
        type=int,
        default=4,
        help="Maximum number of DLAs to model (default: 4).",
    )
    parser.add_argument(
        "--plot_figures",
        type=int,
        default=0,
        help="Set to 1 to generate plots, 0 otherwise (default: 0).",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=None,
        help="Number of workers for parallel processing (default: None).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=100,
        help="Batch size for parallel model evidence computation (default: 100).",
    )

    args = parser.parse_args()

    process_qso(
        spectra_filename=args.spectra_filename,
        zbest_filename=args.zbest_filename,
        learned_file=args.learned_file,
        catalog_name=args.catalog_name,
        los_catalog=args.los_catalog,
        dla_catalog=args.dla_catalog,
        dla_samples_file=args.dla_samples_file,
        sub_dla_samples_file=args.sub_dla_samples_file,
        min_z_separation=args.min_z_separation,
        prev_tau_0=args.prev_tau_0,
        prev_beta=args.prev_beta,
        max_dlas=args.max_dlas,
        plot_figures=bool(args.plot_figures),
        max_workers=args.max_workers,
        batch_size=args.batch_size,
    )
