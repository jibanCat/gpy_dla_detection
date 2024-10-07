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
)  # Assuming you have DESISpectrumReader implemented
from collections import namedtuple
import argparse


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
    max_dlas: int = 4,
    broadening: bool = True,
    plot_figures: bool = False,
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
    max_dlas : int, optional
        Maximum number of DLAs to model (default is 4).
    broadening : bool, optional
        Whether to include instrumental broadening (default is True).
    plot_figures : bool, optional
        If True, generates plots for each processed spectrum (default is False).
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

        # Process the spectrum using Null, DLA, and subDLA models
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
            learned_file,
            min_z_separation,
            plot_figures,
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
    learned_file: str,
    min_z_separation: float,
    plot_figures: bool,
):
    """
    Process a single spectrum using Null, DLA, and subDLA models.
    """

    # Null model (no DLAs)
    gp = NullGPMAT(
        params,
        prior,
        learned_file,
    )
    gp.set_data(
        rest_wavelengths, flux, noise_variance, pixel_mask, z_qso, build_model=True
    )

    # DLA model (up to max DLAs)
    dla_gp = DLAGPMAT(
        params=params,
        prior=prior,
        dla_samples=dla_samples,
        min_z_separation=min_z_separation,
        learned_file=learned_file,
        broadening=broadening,
    )
    dla_gp.set_data(
        rest_wavelengths, flux, noise_variance, pixel_mask, z_qso, build_model=True
    )

    # Sub-DLA model
    subdla_gp = SubDLAGPMAT(
        params=params,
        prior=prior,
        dla_samples=subdla_samples,
        min_z_separation=min_z_separation,
        learned_file=learned_file,
        broadening=broadening,
    )
    subdla_gp.set_data(
        rest_wavelengths, flux, noise_variance, pixel_mask, z_qso, build_model=True
    )

    # Run Bayesian model selection
    log_posteriors = bayes.model_selection([gp, subdla_gp, dla_gp], z_qso)

    # Store results
    results["min_z_dlas"][idx] = dla_gp.params.min_z_dla(wavelengths, z_qso)
    results["max_z_dlas"][idx] = dla_gp.params.max_z_dla(wavelengths, z_qso)
    results["log_priors_no_dla"][idx] = bayes.log_priors[0]
    results["log_priors_dla"][idx, :] = bayes.log_priors[-max_dlas:]
    results["log_likelihoods_no_dla"][idx] = bayes.log_likelihoods[0]
    results["log_likelihoods_dla"][idx, :] = bayes.log_likelihoods[-max_dlas:]
    results["log_posteriors_no_dla"][idx] = bayes.log_posteriors[0]
    results["log_posteriors_dla"][idx, :] = bayes.log_posteriors[-max_dlas:]

    results["sample_log_likelihoods_dla"][idx, :, :] = dla_gp.sample_log_likelihoods[
        :, :
    ]
    results["base_sample_inds"][idx, :, :] = dla_gp.base_sample_inds[:, :].T

    results["sample_log_likelihoods_lls"][idx, :] = subdla_gp.sample_log_likelihoods[
        :, 0
    ]

    # Maximum a posteriori (MAP) estimation for each model
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
        make_plots(dla_gp, bayes, filename=out_filename, sub_dir="images", title=title)
        plt.clf()
        plt.close()


def save_results_to_hdf5(
    filename: str, results: dict, spectrum_ids: List[str], z_qso_list: List[float]
):
    """
    Save the results of the spectrum processing into an HDF5 file.
    """
    with h5py.File(filename, "w") as f:
        for key, data in results.items():
            f.create_dataset(key, data=data)
        f.create_dataset("z_qsos", data=np.array(z_qso_list))
        f.create_dataset(
            "spectrum_ids",
            data=np.array(spectrum_ids, dtype=h5py.string_dtype(encoding="utf-8")),
        )


def make_plots(
    dla_gp: DLAGPMAT,
    bayes: BayesModelSelect,
    filename: str = "spec.pdf",
    sub_dir: str = "images",
    title: str = "",
):
    """
    Generate plots of the GP mean model and the sample likelihood plots.
    """
    title += f"P(DLA | D) = {', '.join([f'{x:.2g}' for x in bayes.model_posteriors])}"

    if not os.path.exists(sub_dir):
        os.makedirs(sub_dir)

    # Find the model with the highest posterior and plot it
    ind = np.argmax(bayes.model_posteriors)
    nth_dla = ind - bayes.dla_model_ind + 1

    if nth_dla <= 0:
        label = f"P(no DLA | D) = {bayes.p_no_dla:.2g}"
    else:
        label = f"P(DLA({nth_dla}) | D) = {bayes.model_posteriors[ind]:.2g}"

    plot_dla_model(dla_gp=dla_gp, nth_dla=nth_dla, title=title, label=label)
    plt.savefig(os.path.join(sub_dir, f"{filename}.pdf"), format="pdf", dpi=100)


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
        max_dlas=args.max_dlas,
        plot_figures=bool(args.plot_figures),
    )
