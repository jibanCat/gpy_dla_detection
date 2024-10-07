import os
import time
import numpy as np
import h5py
from typing import List
import argparse
from matplotlib import pyplot as plt

from gpy_dla_detection.set_parameters import Parameters
from gpy_dla_detection.model_priors import PriorCatalog
from gpy_dla_detection.null_gp import NullGPMAT
from gpy_dla_detection.dla_gp import DLAGPMAT
from gpy_dla_detection.subdla_gp import SubDLAGPMAT
from gpy_dla_detection.dla_samples import DLASamplesMAT
from gpy_dla_detection.subdla_samples import SubDLASamplesMAT
from gpy_dla_detection.bayesian_model_selection import BayesModelSelect
from gpy_dla_detection.desi_spectrum_reader import DESISpectrumReader
from gpy_dla_detection.process_helpers import initialize_results, save_results_to_hdf5
from gpy_dla_detection.plottings.plot_model import plot_samples_vs_this_mu


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
    gp: NullGPMAT,  # Pre-initialized NullGPMAT
    dla_gp: DLAGPMAT,  # Pre-initialized DLAGPMAT
    subdla_gp: SubDLAGPMAT,  # Pre-initialized SubDLAGPMAT
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
    bayes.model_selection(
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


class DLAProcessor:
    """
    Class to handle Bayesian model selection for Damped Lyman-Alpha (DLA) system detection in DESI spectra.

    Parameters:
    ----------
    spectra_filename : str
        DESI spectra FITS filename.
    zbest_filename : str
        DESI redshift catalog filename.
    learned_file : str
        Learned QSO model file path.
    catalog_name : str
        Catalog file path.
    los_catalog : str
        Line-of-sight catalog file path.
    dla_catalog : str
        DLA catalog file path.
    dla_samples_file : str
        DLA samples file path.
    sub_dla_samples_file : str
        Sub-DLA samples file path.
    params : Parameters
        Parameters object containing various settings and hyperparameters.
    min_z_separation : float
        Minimum redshift separation between DLAs.
    prev_tau_0 : float
        Previous value of the DLA optical depth.
    prev_beta : float
        Previous value of the DLA power-law index.
    max_dlas : int, optional
        Maximum number of DLAs to consider per spectrum (default is 4).
    broadening : bool, optional
        Flag indicating whether to apply broadening to the DLA profiles (default is True).
    plot_figures : bool, optional
        Flag indicating whether to plot diagnostic figures during processing (default is False).
    max_workers : int, optional
        Maximum number of parallel workers to use for processing (default is None).
    batch_size : int, optional
        Batch size for parallel processing (default is 100).
    """

    def __init__(
        self,
        spectra_filename: str,
        zbest_filename: str,
        learned_file: str,
        catalog_name: str,
        los_catalog: str,
        dla_catalog: str,
        dla_samples_file: str,
        sub_dla_samples_file: str,
        params: Parameters,
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
        Initialize the DLAProcessor class with necessary data files and parameters.
        """

        self.spectra_filename = spectra_filename
        self.zbest_filename = zbest_filename
        self.learned_file = learned_file
        self.catalog_name = catalog_name
        self.los_catalog = los_catalog
        self.dla_catalog = dla_catalog
        self.dla_samples_file = dla_samples_file
        self.sub_dla_samples_file = sub_dla_samples_file
        self.min_z_separation = min_z_separation
        self.prev_tau_0 = prev_tau_0
        self.prev_beta = prev_beta
        self.max_dlas = max_dlas
        self.broadening = broadening
        self.plot_figures = plot_figures
        self.max_workers = max_workers
        self.batch_size = batch_size
        self.params = params  # Pass in the Parameters object here

        # Initialize prior catalog and Bayesian model selection
        self.prior = PriorCatalog(self.params, catalog_name, los_catalog, dla_catalog)
        self.dla_samples = DLASamplesMAT(self.params, self.prior, dla_samples_file)
        self.subdla_samples = SubDLASamplesMAT(
            self.params, self.prior, sub_dla_samples_file
        )
        self.bayes = BayesModelSelect([0, 1, max_dlas], 2)

        # Initialize reader for DESI spectra
        self.reader = DESISpectrumReader(spectra_filename, zbest_filename)
        self.reader.read_spectra()
        self.reader.read_redshift_catalog()
        self.redshift_data = self.reader.get_redshift_data()
        self.all_spectrum_ids = self.reader.get_all_spectrum_ids()
        self.results = initialize_results(
            len(self.all_spectrum_ids), max_dlas, self.params.num_dla_samples
        )

    def process_all_spectra(self):
        """
        Process all spectra in the DESI file.
        """
        for idx, spectrum_id in enumerate(self.all_spectrum_ids):
            tic = time.time()

            z_qso = self.redshift_data["Z"][idx]
            spectrum_data = self.reader.get_spectrum_data(spectrum_id)
            wavelengths = spectrum_data.wavelengths
            flux = spectrum_data.flux
            noise_variance = spectrum_data.noise_variance
            pixel_mask = spectrum_data.pixel_mask

            rest_wavelengths = self.params.emitted_wavelengths(wavelengths, z_qso)

            # Initialize the Null and DLA models for this spectrum
            null_gp = NullGPMAT(
                self.params,
                self.prior,
                learned_file=self.learned_file,
                prev_tau_0=self.prev_tau_0,
                prev_beta=self.prev_beta,
            )
            dla_gp = DLAGPMAT(
                self.params,
                self.prior,
                self.dla_samples,
                min_z_separation=self.min_z_separation,
                learned_file=self.learned_file,
                broadening=self.broadening,
                prev_tau_0=self.prev_tau_0,
                prev_beta=self.prev_beta,
            )
            subdla_gp = SubDLAGPMAT(
                self.params,
                self.prior,
                self.subdla_samples,
                min_z_separation=self.min_z_separation,
                learned_file=self.learned_file,
                broadening=self.broadening,
                prev_tau_0=self.prev_tau_0,
                prev_beta=self.prev_beta,
            )

            # Process single spectrum
            process_single_spectrum(
                idx,
                spectrum_id,
                z_qso,
                wavelengths,
                rest_wavelengths,
                flux,
                noise_variance,
                pixel_mask,
                self.params,
                self.prior,
                self.dla_samples,
                self.subdla_samples,
                self.bayes,
                self.results,
                self.max_dlas,
                self.broadening,
                null_gp,
                dla_gp,
                subdla_gp,
                self.min_z_separation,
                self.plot_figures,
                self.max_workers,
                self.batch_size,
            )

            toc = time.time()
            print(
                f"Processed spectrum {idx + 1}/{len(self.all_spectrum_ids)} (ID: {spectrum_id}), time spent: {(toc - tic) // 60:.0f}m {(toc - tic) % 60:.0f}s"
            )

        # Save results to HDF5 file
        save_results_to_hdf5(
            "processed_desi_spectra.h5",
            self.results,
            self.all_spectrum_ids,
            self.redshift_data["Z"],
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process DESI spectra with Bayesian model selection for DLA detection."
    )

    # Spectra and file-related arguments
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
        default="data/dr12q/processed/learned_qso_model_lyseries_variance.mat",
        help="Learned QSO model file path.",
    )
    parser.add_argument(
        "--catalog_name",
        default="data/dr12q/processed/catalog.mat",
        help="Catalog file path.",
    )
    parser.add_argument(
        "--los_catalog",
        default="data/dla_catalogs/processed/los_catalog",
        help="Line-of-sight catalog file path.",
    )
    parser.add_argument(
        "--dla_catalog",
        default="data/dla_catalogs/processed/dla_catalog",
        help="DLA catalog file path.",
    )
    parser.add_argument(
        "--dla_samples_file",
        default="data/dr12q/processed/dla_samples_a03.mat",
        help="DLA samples file path.",
    )
    parser.add_argument(
        "--sub_dla_samples_file",
        default="data/dr12q/processed/subdla_samples.mat",
        help="Sub-DLA samples file path.",
    )

    # DLA-related arguments
    parser.add_argument(
        "--min_z_separation",
        type=float,
        default=3000.0,
        help="Minimum redshift separation for DLA models.",
    )
    parser.add_argument(
        "--prev_tau_0", type=float, default=0.00554, help="Previous value for tau_0."
    )
    parser.add_argument(
        "--prev_beta", type=float, default=3.182, help="Previous value for beta."
    )
    parser.add_argument(
        "--max_dlas", type=int, default=3, help="Maximum number of DLAs to model."
    )
    parser.add_argument(
        "--plot_figures",
        type=int,
        default=0,
        help="Set to 1 to generate plots, 0 otherwise.",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=32,
        help="Number of workers for parallel processing.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=313,
        help="Batch size for parallel model evidence computation.",
    )

    # Parameter-related arguments
    # These are the values used in the trained GP model, don't change them unless you change the trained model
    parser.add_argument(
        "--loading_min_lambda",
        type=float,
        default=800,
        help="Range of rest wavelengths to load (Å).",
    )
    parser.add_argument(
        "--loading_max_lambda",
        type=float,
        default=1550,
        help="Range of rest wavelengths to load (Å).",
    )
    parser.add_argument(
        "--normalization_min_lambda",
        type=float,
        default=1425,
        help="Range of rest wavelengths for flux normalization.",
    )
    parser.add_argument(
        "--normalization_max_lambda",
        type=float,
        default=1475,
        help="Range of rest wavelengths for flux normalization.",
    )
    parser.add_argument(
        "--min_lambda",
        type=float,
        default=850.75,
        help="Range of rest wavelengths to model (Å).",
    )
    parser.add_argument(
        "--max_lambda",
        type=float,
        default=1420.75,
        help="Range of rest wavelengths to model (Å).",
    )
    parser.add_argument(
        "--dlambda", type=float, default=0.25, help="Separation of wavelength grid (Å)."
    )
    parser.add_argument(
        "--k", type=int, default=20, help="Rank of non-diagonal contribution."
    )
    parser.add_argument(
        "--max_noise_variance",
        type=float,
        default=9,
        help="Maximum pixel noise allowed during model training.",
    )

    args = parser.parse_args()

    # Initialize Parameters object with user inputs
    params = Parameters(
        loading_min_lambda=args.loading_min_lambda,
        loading_max_lambda=args.loading_max_lambda,
        normalization_min_lambda=args.normalization_min_lambda,
        normalization_max_lambda=args.normalization_max_lambda,
        min_lambda=args.min_lambda,
        max_lambda=args.max_lambda,
        dlambda=args.dlambda,
        k=args.k,
        max_noise_variance=args.max_noise_variance,
    )

    processor = DLAProcessor(
        spectra_filename=args.spectra_filename,
        zbest_filename=args.zbest_filename,
        learned_file=args.learned_file,
        catalog_name=args.catalog_name,
        los_catalog=args.los_catalog,
        dla_catalog=args.dla_catalog,
        dla_samples_file=args.dla_samples_file,
        sub_dla_samples_file=args.sub_dla_samples_file,
        params=params,
        min_z_separation=args.min_z_separation,
        prev_tau_0=args.prev_tau_0,
        prev_beta=args.prev_beta,
        max_dlas=args.max_dlas,
        plot_figures=bool(args.plot_figures),
        max_workers=args.max_workers,
        batch_size=args.batch_size,
    )

    processor.process_all_spectra()
