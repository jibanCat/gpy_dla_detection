"""
Plot the raw spectrum from fits file
"""
from typing import Optional

import os
import re

import numpy as np
from matplotlib import pyplot as plt
from ..read_spec import read_spec, read_spec_dr14q, file_loader, retrieve_raw_spec
from ..set_parameters import Parameters

def plot_raw_spectrum(filename: str, release: str = 'dr12q', z_qso: Optional[float] = None):
    '''
    Plot the raw spectrum, the spectrum before normalisation.
    
    :param filename: filename of the fits file. Must follow the convention,
        "spec-{:d}-{:d}-{:04d}.fits".format(plate, mjd, fiber_id)
    :param release: either dr12q or dr14q
    :param z_qso: if known, plot an sub-axis with rest-frame wavelengths.
    '''
    assert release in ("dr12q", "dr14q")

    # must follow the filename rule to extract
    plate, mjd, fiber_id = re.findall(
            r"spec-([0-9]+)-([0-9]+)-([0-9]+).fits", filename,
        )[0]
    
    if not os.path.exists(filename):
        retrieve_raw_spec(int(plate), int(mjd), int(fiber_id), release=release)
        # to prevent some people might tempt to load fits file
        # from other directories, here we re-specify the filename
        print("[Warning] file {} not found, re-download the file.".format(filename))
        filename = file_loader(int(plate), int(mjd), int(fiber_id))

    # read fits file
    if release == "dr12q":
        wavelengths, flux, noise_variance, pixel_mask = read_spec(filename)
    elif release == "dr14q":
        wavelengths, flux, noise_variance, pixel_mask = read_spec_dr14q(filename)
    else:
        raise Exception("must choose between dr12q or dr14q!")

    # plotting config
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(wavelengths, flux,           lw=0.25, label=r"$y(\lambda_{obs})$")
    ax.plot(wavelengths, noise_variance, lw=0.25, label=r"$\sigma^2(\lambda_{obs})$")
    ax.set_xlabel(r" Observed Wavelength [$\AA$]")
    ax.set_ylabel(r"Flux [$10^{-17}{erg}/s/{cm}^{2}/\AA$]")
    ax.set_title(r"{}".format(filename))
    ax.set_ylim( np.quantile(flux, 0.005), np.quantile(flux, 0.995) )
    ax.legend()

    # [z_qso] plot the rest-frame axis if z_qso known
    if z_qso != None:
        assert (0 < z_qso) and (z_qso < 99)

        ax2 = ax.secondary_xaxis('top', functions=(
            lambda x : Parameters.emitted_wavelengths(x, z_qso), 
            lambda x : Parameters.observed_wavelengths(x, z_qso)))
        ax2.set_xlabel(r"Rest Wavelength [$\AA$]")
