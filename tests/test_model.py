"""
test_model.py : test the functions related to GP model
"""
import numpy as np
from gpy_dla_detection.effective_optical_depth import effective_optical_depth
from gpy_dla_detection.set_parameters import Parameters


def test_effective_optical_depth():
    z_qso = 4
    rest_wavelengths = np.linspace(911, 1216, 500)
    wavelengths = Parameters.observed_wavelengths(rest_wavelengths, z_qso)

    total_optical_depth = effective_optical_depth(
        wavelengths, 3.65, 0.0023, z_qso, 31, True
    )

    assert 0 < np.exp(-total_optical_depth.sum(axis=1)).min() < 1
    assert 0 < np.exp(-total_optical_depth.sum(axis=1)).max() < 1

    total_optical_depth_2 = effective_optical_depth(
        wavelengths, 3.65, 0.0023, z_qso, 1, True
    )

    assert np.mean(np.exp(-total_optical_depth.sum(axis=1))) < np.mean(
        np.exp(-total_optical_depth_2.sum(axis=1))
    )

    total_optical_depth_3 = effective_optical_depth(
        wavelengths, 3.65, 0.0023, 2.2, 31, True
    )

    assert np.mean(np.exp(-total_optical_depth.sum(axis=1))) < np.mean(
        np.exp(-total_optical_depth_3.sum(axis=1))
    )
