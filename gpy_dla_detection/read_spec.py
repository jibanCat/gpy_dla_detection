"""
read_spec.py : python version of Roman's read_spec.m

Note:
----
we can add more read_spec functions for different
datasets.
"""
from typing import Tuple

from urllib import request
import numpy as np
from astropy.io import fits


file_loader = lambda plate, mjd, fiber_id: "spec-{:d}-{:d}-{:04d}.fits".format(
    plate, mjd, fiber_id
)


def read_spec(filename: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    loads data from SDSS DR12Q coadded "speclite" FITS file;
    see
    https://data.sdss.org/datamodel/files/BOSS_SPECTRO_REDUX/RUN2D/spectra/PLATE4/spec.html
    for a complete description of the data format

    Returns:
    ----
    wavelengths     : observed wavelengths 
    flux            : coadded calibrated flux 10**-17 erg s**-1 cm**-2 A**-1
    noise_variance  : noise variance per pixel
    pixel_mask      : if 1/noise_variance = 0 and BRIGHTSKY

    TODO: can I read zQSO from spec? I guess it depends on the fits files of different surveys
    """
    with fits.open(filename) as hdu:
        data = hdu["COADD"].data

        # coadded calibrated flux 10**-17 erg s**-1 cm**-2 A**-1
        flux = data["flux"]

        # log_10 wavelength       log (A)
        log_wavelengths = data["loglam"]

        # inverse noise variance
        inverse_noise_variance = data["ivar"]

        # `and` mask
        and_mask = data["and_mask"]

        wavelengths = 10 ** log_wavelengths

        # handle divide by zero
        ind = inverse_noise_variance == 0
        noise_variance = np.zeros(inverse_noise_variance.shape)
        noise_variance[:] = np.nan  # fill zero division with NaNs
        noise_variance[~ind] = 1 / inverse_noise_variance[~ind]

        # derive bad pixel mask, follow the same recipe in Roman's read_spec.m
        BRIGHTSKY = 24
        pixel_mask = (inverse_noise_variance == 0) | np.array(
            [(m >> BRIGHTSKY) & 1 for m in and_mask]
        ).astype("bool")

    return wavelengths, flux, noise_variance, pixel_mask

def read_spec_dr14q(filename: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    loads data from SDSS DR14Q coadded "speclite" FITS file;

    They don't have COADD column, so write the read_spec into another function
    for cleaner design.

    Returns:
    ----
    wavelengths     : observed wavelengths
    flux            : coadded calibrated flux 10**-17 erg s**-1 cm**-2 A**-1
    noise_variance  : noise variance per pixel
    pixel_mask      : if 1/noise_variance = 0 and BRIGHTSKY
    """
    with fits.open(filename) as hdu:
        # the first binary table
        data = hdu[1].data

        # coadded calibrated flux 10**-17 erg s**-1 cm**-2 A**-1
        flux = data["flux"]

        # log_10 wavelength       log (A)
        log_wavelengths = data["loglam"]

        # inverse noise variance
        inverse_noise_variance = data["ivar"]

        # `and` mask
        and_mask = data["and_mask"]

        wavelengths = 10 ** log_wavelengths

        # handle divide by zero
        ind = inverse_noise_variance == 0
        noise_variance = np.zeros(inverse_noise_variance.shape)
        noise_variance[:] = np.nan  # fill zero division with NaNs
        noise_variance[~ind] = 1 / inverse_noise_variance[~ind]

        # derive bad pixel mask, follow the same recipe in Roman's read_spec.m
        BRIGHTSKY = 24
        pixel_mask = (inverse_noise_variance == 0) | np.array(
            [(m >> BRIGHTSKY) & 1 for m in and_mask]
        ).astype("bool")

    return wavelengths, flux, noise_variance, pixel_mask


def retrieve_raw_spec(plate: int, mjd: int, fiber_id: int, release: str = "dr12q"):
    """
    utility function to download a raw spec from SDSS
    """
    filename = file_loader(plate, mjd, fiber_id)

    if release == "dr12q":
        # greedy list all plates at v_5_7_2
        v_5_7_2_plates = [
            7339,
            7340,
            7386,
            7388,
            7389,
            7391,
            7396,
            7398,
            7401,
            7402,
            7404,
            7406,
            7407,
            7408,
            7409,
            7411,
            7413,
            7416,
            7419,
            7422,
            7425,
            7426,
            7428,
            7455,
            7512,
            7513,
            7515,
            7516,
            7517,
            7562,
            7563,
            7564,
            7565,
        ]

        in_5_7_2 = plate in v_5_7_2_plates
        url = "https://data.sdss.org/sas/dr12/boss/spectro/redux/{}/spectra/{:d}/spec-{:d}-{:d}-{:04d}.fits".format(
            ["v5_7_0", "v5_7_2"][in_5_7_2], plate, plate, mjd, fiber_id
        )

    elif release == "dr14q":
        url = "https://data.sdss.org/sas/dr16/eboss/spectro/redux/{}/spectra/lite/{:d}/spec-{:d}-{:d}-{:04d}.fits".format(
            "v5_13_0", plate, plate, mjd, fiber_id
        )
    else:
        raise Exception("must choose between dr12q or dr14q!")


    print("[Info] retrieving {} ...".format(url), end=" ")
    request.urlretrieve(url, filename)
    print("Done.")
