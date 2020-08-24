"""
Perturb the zQSO value in the SDSS catalogue and test
the difference in the result of Bayesian model selection.
"""
from typing import List, Any, Tuple

import time

from .process_qso import process_qso

from gpy_dla_detection import read_spec

from gpy_dla_detection.bayesian_model_selection import BayesModelSelect

from gpy_dla_detection.zqso_set_parameters import ZParameters
from gpy_dla_detection.zqso_samples import ZSamples
from gpy_dla_detection.zqso_gp import ZGPMAT


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

    z_qso_gp = ZGPMAT(
        params,
        z_qso_samples,
        learned_file=learned_file_zestimation,
    )

    tic = time.time()

    z_qso_gp.inference_z_qso(wavelengths, flux, noise_variance, pixel_mask, z_qso_min=z_qso - dz, z_qso_max=z_qso + dz)
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
