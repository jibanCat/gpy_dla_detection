"""
A test file for testing zestimation

The learned file could be downloaded at
[learned_zqso_only_model_outdata_full_dr9q_minus_concordance_norm_1176-1256.mat]
(https://drive.google.com/file/d/1SqAU_BXwKUx8Zr38KTaA_nvuvbw-WPQM/view?usp=sharing)
"""
import os
import re
import time

import numpy as np

from .test_selection import filenames, z_qsos

from gpy_dla_detection.read_spec import read_spec, retrieve_raw_spec
from gpy_dla_detection.zqso_set_parameters import ZParameters
from gpy_dla_detection.zqso_samples import ZSamples
from gpy_dla_detection.zqso_gp import ZGPMAT


def test_zestimation(nspec: int):
    filename = filenames[nspec]

    if not os.path.exists(filename):
        plate, mjd, fiber_id = re.findall(
            r"spec-([0-9]+)-([0-9]+)-([0-9]+).fits", filename,
        )[0]
        retrieve_raw_spec(int(plate), int(mjd), int(fiber_id))

    params = ZParameters()
    z_qso_samples = ZSamples(params)

    wavelengths, flux, noise_variance, pixel_mask = read_spec(filename)

    z_qso_gp = ZGPMAT(
        params,
        z_qso_samples,
        learned_file="data/dr12q/processed/learned_zqso_only_model_outdata_full_dr9q_minus_concordance_norm_1176-1256.mat",
    )

    tic = time.time()

    z_qso_gp.inference_z_qso(wavelengths, flux, noise_variance, pixel_mask)
    print("Z True : {:.3g}".format(z_qsos[nspec]))

    toc = time.time()
    print("spent {} mins; {} seconds".format((toc - tic) // 60, (toc - tic) % 60))

    return z_qso_gp.z_map, z_qsos[nspec]


def test_batch(num_quasars: int = 100):
    all_z_diffs = np.zeros((num_quasars,))

    for nspec in range(num_quasars):
        z_map, z_true = test_zestimation(nspec)

        z_diff = z_map - z_true

        print("[Info] z_diff = z_map - z_true = {:.8g}".format(z_diff))

        all_z_diffs[nspec] = z_diff

    print("[Info] abs(z_diff) < 0.5 = {:.4g}".format(accuracy(all_z_diffs, 0.5)))
    print("[Info] abs(z_diff) < 0.05 = {:.4g}".format(accuracy(all_z_diffs, 0.05)))

    # we got ~99% accuracy in https://arxiv.org/abs/2006.07343
    # so at least we need to ensure ~98% here
    assert accuracy(all_z_diffs, 0.5) > 0.98


def accuracy(z_diff: np.ndarray, z_thresh: float):
    num_quasars = z_diff.shape[0]
    corrects = (np.abs(z_diff) < z_thresh).sum()

    return corrects / num_quasars
