"""
test read spec
"""
import os
from gpy_dla_detection.read_spec import read_spec, retrieve_raw_spec
import numpy as np


def test_read_spec():
    if not os.path.exists("spec-7340-56825-0576.fits"):
        retrieve_raw_spec(7340, 56825, 576)  # an arbitrary spectrum

    wavelengths, flux, noise_variance, pixel_mask = read_spec(
        "spec-7340-56825-0576.fits"
    )

    assert min(wavelengths) > 1216
    assert len(flux) == len(noise_variance)
    assert type(pixel_mask[0]) is np.bool_
