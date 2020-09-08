"""
Perturb the zQSO value in the SDSS catalogue and test
the difference in the result of Bayesian model selection.
"""
from typing import List, Any, Tuple

import time, os

from .process_qso import process_qso

from gpy_dla_detection import read_spec

from gpy_dla_detection.bayesian_model_selection import BayesModelSelect

from gpy_dla_detection.zqso_set_parameters import ZParameters
from gpy_dla_detection.zqso_samples import ZSamples
from gpy_dla_detection.zqso_gp import ZGPMAT

from matplotlib import pyplot as plt


def z_qso_correction(
    filename: str,
    z_qso: float,
    dz: float = 0.5,
    learned_file_zestimation: str = "data/dr12q/processed/learned_zqso_only_model_outdata_normout_dr9q_minus_concordance_norm_1176-1256.mat",
    read_spec=read_spec.read_spec,
    catalog_file: str = "data/dr12q/processed/catalog.mat",
    learned_file: str = "data/dr12q/processed/learned_qso_model_lyseries_variance_kim_dr9q_minus_concordance.mat",
    prior_los: str = "data/dla_catalogs/dr9q_concordance/processed/los_catalog",
    prior_dla: str = "data/dla_catalogs/dr9q_concordance/processed/dla_catalog",
    dla_samples_file: str = "data/dr12q/processed/dla_samples_a03.mat",
    subdla_samples_file: str = "data/dr12q/processed/subdla_samples.mat",
    max_dlas: int = 4,
    min_z_separation: float = 3000.0,
    broadening: bool = True,
) -> Tuple[BayesModelSelect, BayesModelSelect, ZGPMAT]:
    # Bayesian model selection on the zQSO in catalog
    bayes, model_list = process_qso(
        filename,
        z_qso,
        read_spec,
        catalog_file,
        learned_file,
        prior_los,
        prior_dla,
        dla_samples_file,
        subdla_samples_file,
        max_dlas,
        min_z_separation,
        broadening,
    )

    params = ZParameters()
    z_qso_samples = ZSamples(params)

    wavelengths, flux, noise_variance, pixel_mask = read_spec(filename)

    z_qso_gp = ZGPMAT(params, z_qso_samples, learned_file=learned_file_zestimation,)

    tic = time.time()

    z_qso_gp.inference_z_qso(
        wavelengths,
        flux,
        noise_variance,
        pixel_mask,
        z_qso_min=z_qso - dz,
        z_qso_max=z_qso + dz,
    )
    print("Z True : {:.3g}".format(z_qso))

    toc = time.time()
    print("spent {} mins; {} seconds".format((toc - tic) // 60, (toc - tic) % 60))

    # corrected the zQSO and re-calculate the model posteriors
    bayes_corrected, model_list_corrected = process_qso(
        filename,
        z_qso_gp.z_map,
        read_spec,
        catalog_file,
        learned_file,
        prior_los,
        prior_dla,
        dla_samples_file,
        subdla_samples_file,
        max_dlas,
        min_z_separation,
        broadening,
    )

    return bayes, bayes_corrected, z_qso_gp, model_list, model_list_corrected


def catalog_z_qso_check(
    plate: int,
    mjd: int,
    fiber_id: int,
    z_qso: float,
    release="dr14q",
    learned_file_zestimation: str = "data/dr12q/processed/learned_zqso_only_model_outdata_normout_dr9q_minus_concordance_norm_1176-1256.mat",
):
    """
    give the spec id and check if the zQSO changed
    """
    filename = "spec-{:d}-{:d}-{:04}.fits".format(plate, mjd, fiber_id)

    if not os.path.exists(filename):
        read_spec.retrieve_raw_spec(plate, mjd, fiber_id, release=release)

    params = ZParameters()
    z_qso_samples = ZSamples(params)

    if release == "dr12q":
        wavelengths, flux, noise_variance, pixel_mask = read_spec.read_spec(filename)
    elif release == "dr14q":
        wavelengths, flux, noise_variance, pixel_mask = read_spec.read_spec_dr14q(
            filename
        )
    else:
        raise Exception("must choose between dr12q or dr14q!")

    z_qso_gp = ZGPMAT(params, z_qso_samples, learned_file=learned_file_zestimation,)

    tic = time.time()

    z_qso_gp.inference_z_qso(wavelengths, flux, noise_variance, pixel_mask)
    print("Z True : {:.3g}".format(z_qso))

    toc = time.time()
    print("spent {} mins; {} seconds".format((toc - tic) // 60, (toc - tic) % 60))

    # some testing output plots to see which prediction make more sense
    if not os.path.isdir("z_qso_check_images/"):
        os.mkdir("z_qso_check_images/")

    sample_z_qsos = z_qso_gp.z_qso_samples.sample_z_qsos()

    # plot the full spectrum
    fig, axs = plt.subplots(3, 1, figsize=(16, 15))

    # 1. catalogue zQSO; also plot the model
    z_qso_gp.set_data(
        wavelengths,
        flux,
        noise_variance,
        pixel_mask,
        z_qso=z_qso,
        normalize=True,
        build_model=True,
    )
    axs[0].plot(z_qso_gp.x, z_qso_gp.y, label="catalogue zQSO")
    axs[0].plot(z_qso_gp.x, z_qso_gp.this_mu, label="mean model")
    axs[0].set_xlabel(r"Rest-frame Wavelength ($\AA$)")
    axs[0].set_ylabel(r"Normalized Flux")
    axs[0].set_ylim(-1, 6)
    axs[0].legend()

    # 2. sample likelihood plot
    axs[1].plot(sample_z_qsos, z_qso_gp.sample_log_likelihoods, label="zQSO inference")
    axs[1].vlines(
        z_qso,
        z_qso_gp.sample_log_likelihoods.min(),
        z_qso_gp.sample_log_likelihoods.max(),
        ls="--",
        color="C1",
        label="catalogue zQSO : {:.4g}".format(z_qso),
    )
    axs[1].vlines(
        z_qso_gp.z_map,
        z_qso_gp.sample_log_likelihoods.min(),
        z_qso_gp.sample_log_likelihoods.max(),
        ls="--",
        color="C2",
        label="zMAP : {:.4g}".format(z_qso_gp.z_map),
    )
    axs[1].set_xlabel("zQSO")
    axs[1].set_ylabel("p(Data | zQSO, M)")
    axs[1].legend()

    # 3. z_map; also plot the model
    z_qso_gp.set_data(
        wavelengths,
        flux,
        noise_variance,
        pixel_mask,
        z_qso=z_qso_gp.z_map,
        normalize=True,
        build_model=True,
    )
    axs[2].plot(z_qso_gp.x, z_qso_gp.y, label="zMAP")
    axs[2].plot(z_qso_gp.x, z_qso_gp.this_mu, label="mean model")
    axs[2].set_xlabel(r"Rest-frame Wavelength ($\AA$)")
    axs[2].set_ylabel(r"Normalized Flux")
    axs[2].set_ylim(-1, 6)
    axs[2].legend()

    plt.savefig(
        "z_qso_check_images/spec-{:d}-{:d}-{:04d}.pdf".format(plate, mjd, fiber_id),
        format="pdf",
        dpi=150,
    )
    plt.clf()
    plt.close()

    # plot the modelling range for DLA
    # plot the full spectrum
    fig, axs = plt.subplots(2, 1, figsize=(16, 10))

    # 1. catalogue zQSO; also plot the model
    z_qso_gp.set_data(
        wavelengths,
        flux,
        noise_variance,
        pixel_mask,
        z_qso=z_qso,
        normalize=True,
        build_model=True,
    )
    ind = (z_qso_gp.x <= params.lya_wavelength) * (z_qso_gp.x >= params.lyman_limit)
    axs[0].plot(z_qso_gp.x[ind], z_qso_gp.y[ind], label="catalogue zQSO : {:.4g}".format(z_qso))
    axs[0].plot(z_qso_gp.x[ind], z_qso_gp.this_mu[ind], label="mean model")
    axs[0].set_xlabel(r"Rest-frame Wavelength ($\AA$)")
    axs[0].set_ylabel(r"Normalized Flux")
    axs[0].legend()

    # 2. z_map; also plot the model
    z_qso_gp.set_data(
        wavelengths,
        flux,
        noise_variance,
        pixel_mask,
        z_qso=z_qso_gp.z_map,
        normalize=True,
        build_model=True,
    )
    ind = (z_qso_gp.x <= params.lya_wavelength) * (z_qso_gp.x >= params.lyman_limit)
    axs[1].plot(z_qso_gp.x[ind], z_qso_gp.y[ind], label="zMAP : {:.4g}".format(z_qso_gp.z_map))
    axs[1].plot(z_qso_gp.x[ind], z_qso_gp.this_mu[ind], label="mean model")
    axs[1].set_xlabel(r"Rest-frame Wavelength ($\AA$)")
    axs[1].set_ylabel(r"Normalized Flux")
    axs[1].legend()

    plt.savefig(
        "z_qso_check_images/spec-{:d}-{:d}-{:04d}_dla_range.pdf".format(
            plate, mjd, fiber_id
        ),
        format="pdf",
        dpi=150,
    )
    plt.clf()
    plt.close()

    return z_qso_gp
