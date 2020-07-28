"""
Download spectrum fits files directly from URL links for testing.
"""
import os
import numpy as np
from gpy_dla_detection.read_spec import read_spec, retrieve_raw_spec


def download_ho_2020_spectrum(num_quasars: int = 5):
    """
    Download first N spectra from Ho-Bird-Garnett (2020) catalogue.
    """
    assert num_quasars <= 20

    # first 20 from catalogue
    plates = np.array(
        [
            6173,
            6177,
            4354,
            6498,
            6177,
            4216,
            6182,
            4296,
            7134,
            6877,
            6177,
            4277,
            4415,
            4216,
            4216,
            7167,
            6177,
            4354,
            7144,
            6177,
        ]
    )
    mjds = np.array(
        [
            56238,
            56268,
            55810,
            56565,
            56268,
            55477,
            56190,
            55499,
            56566,
            56544,
            56268,
            55506,
            55831,
            55477,
            55477,
            56604,
            56268,
            55810,
            56564,
            56268,
        ]
    )
    fiber_ids = np.array(
        [
            528,
            595,
            646,
            177,
            608,
            312,
            652,
            364,
            594,
            564,
            648,
            896,
            554,
            302,
            292,
            290,
            384,
            686,
            752,
            640,
        ]
    )

    for plate, mjd, fiber_id in zip(
        plates[:num_quasars], mjds[:num_quasars], fiber_ids[:num_quasars]
    ):

        filename = "spec-{}-{}-{}.fits".format(plate, mjd, str(fiber_id).zfill(4))

        print(filename)

        if not os.path.exists(filename):
            retrieve_raw_spec(plate, mjd, fiber_id)  # the spectrum at paper
