"""
Test set_parameters
"""
from gpy_dla_detection.set_parameters import Parameters


def test_values():
    parameters = Parameters()

    assert abs(parameters.lya_wavelength - 1216) < 1
    assert abs(parameters.lyb_wavelength - 1025) < 1

    assert abs(parameters.kms_to_z(3000) - 0.01) < 1e-4

    wavelength = 1000
    z_qso = 3

    assert (
        abs(
            wavelength
            - parameters.emitted_wavelengths(
                parameters.observed_wavelengths(wavelength, z_qso), z_qso
            )
        )
        < 1e-4
    )
