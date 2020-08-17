"""
A test file for testing zestimation
"""
import os
import re
import time

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
        learned_file="data/dr12q/processed/learned_zqso_only_model_outdata_normout_dr9q_minus_concordance_norm_1176-1256.mat",
    )

    tic = time.time()

    z_qso_gp.inference_z_qso(wavelengths, flux, noise_variance, pixel_mask)
    print("Z True : {:.3g}".format(z_qsos[nspec]))

    toc = time.time()
    print("spent {} mins; {} seconds".format((toc - tic) // 60, (toc - tic) % 60))
