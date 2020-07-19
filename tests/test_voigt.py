"""
test_voigt.py : test the Voigt profile
"""
import numpy as np
from gpy_dla_detection.voigt import voigt_absorption, instrument_profile, width


def test_instrumental_boardening():
    # test 1
    z_qso = 3.15
    wavelengths = np.linspace(911, 1216, 1000) * (1 + z_qso)

    z_dla = 3.1
    nhi = 10 ** 20.3

    raw_profile = voigt_absorption(
        wavelengths, nhi, z_dla, num_lines=3, boardening=False
    )

    # the convolution written in Roman's code
    profile = np.zeros((wavelengths.shape[0] - 2 * width,))
    num_points = len(profile)

    # instrumental broadening
    for i in range(num_points):
        for k, j in enumerate(range(i, i + 2 * width + 1)):
            profile[i] += raw_profile[j] * instrument_profile[k]

    # numpy native convolution
    profile_numpy = np.convolve(raw_profile, instrument_profile, "valid")

    assert np.all(np.abs(profile - profile_numpy) < 1e-4)

    # test 2
    z_qso = 5
    wavelengths = np.linspace(911, 1216) * (1 + z_qso)

    z_dla = 4.5
    nhi = 10 ** 21

    raw_profile = voigt_absorption(
        wavelengths, nhi, z_dla, num_lines=5, boardening=False
    )

    # the convolution written in Roman's code
    profile = np.zeros((wavelengths.shape[0] - 2 * width,))
    num_points = len(profile)

    # instrumental broadening
    for i in range(num_points):
        for k, j in enumerate(range(i, i + 2 * width + 1)):
            profile[i] += raw_profile[j] * instrument_profile[k]

    # numpy native convolution
    profile_numpy = np.convolve(raw_profile, instrument_profile, "valid")

    assert np.all(np.abs(profile - profile_numpy) < 1e-4)
