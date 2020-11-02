"""
Plot the GP model with sample likelihoods
"""
from typing import Optional

import numpy as np
from matplotlib import pyplot as plt

from ..dla_gp import DLAGP


def plot_dla_model(
    dla_gp: DLAGP,
    nth_dla: int,
    title: Optional[str] = None,
    label: Optional[str] = None,
):
    # plot both mean GP model and the sample
    fig, ax = plt.subplots(2, 1, figsize=(16, 10))

    # [title] e.g., thing_id=, DLA(nth_dla)=?
    ax[0].set_title(title)

    plot_sample_likelihoods(dla_gp=dla_gp, ax=ax[0])
    # [label] e.g., spec-xxx-xxx-xxx
    plot_this_mu(dla_gp=dla_gp, nth_dla=nth_dla, ax=ax[1], label=label)


def plot_this_mu(
    dla_gp: DLAGP,
    nth_dla: int = 1,
    ax: Optional[plt.axes] = None,
    label: Optional[str] = None,
):
    """
    Plot the GP mean model onto data

    :param dla_gp: the DLAGP instance you want to plot
    :param nth_dla: the num of DLA you want to plot. Default 1 DLA.
    :param ax: the matplotlib.pyplot.ax you want to plot on. if None, generate a new one.
    :param label: the label you want to put on the figure.
    """
    # [check] make sure we ran the log_evidence
    assert "sample_log_likelihoods" in dir(dla_gp)

    this_rest_wavelengths = dla_gp.x
    this_flux = dla_gp.y

    this_mu = dla_gp.this_mu

    # [ax] if not given, create a new one
    if not ax:
        fig, ax = plt.subplots(1, 1, figsize=(16, 5))

    # observed data
    ax.plot(this_rest_wavelengths, this_flux, label=label, color="C0")

    # DLA model
    if nth_dla > 0:
        # [MAP] maximum a posteriori values
        # N * (1~k models) * (1~k MAP dlas)
        MAP_z_dla, MAP_log_nhi = dla_gp.maximum_a_posteriori()
        # make them to be 1-D array
        map_z_dlas = MAP_z_dla[nth_dla - 1, :nth_dla]
        map_log_nhis = MAP_log_nhi[nth_dla - 1, :nth_dla]
        # feed in MAP values and get the absorption profile given (z_dlas, nhis)
        dla_mu, dla_M, dla_omega2 = dla_gp.this_dla_gp(map_z_dlas, 10 ** map_log_nhis)

        ax.plot(
            this_rest_wavelengths,
            dla_mu,
            label=r"$\mathcal{M}$"
            + r" DLA({n}); ".format(n=nth_dla)
            + "z_dlas = ({}); ".format(",".join("{:.3g}".format(z) for z in map_z_dlas))
            + "lognhi = ({})".format(
                ",".join("{:.3g}".format(n) for n in map_log_nhis)
            ),
            color="red",
        )
    else:
        ax.plot(
            this_rest_wavelengths,
            this_mu,
            label=r"$\mathcal{M}$" + r" DLA({n})".format(n=0),
            color="red",
        )

    ax.set_xlim(this_rest_wavelengths.min(), this_rest_wavelengths.max())
    ax.set_ylim( this_mu.min() - 2, this_mu.max() + 1 )
    ax.set_xlabel(r"Rest-Wavelength $\lambda_{\mathrm{rest}}$ $\AA$")
    ax.set_ylabel(r"Normalised Flux")
    ax.legend()


def plot_sample_likelihoods(dla_gp: DLAGP, ax: Optional[plt.axes] = None):
    """
    Plot the sample likelihoods in the parameter space
    """
    sample_z_dlas = dla_gp.dla_samples.sample_z_dlas(
        dla_gp.this_wavelengths, dla_gp.z_qso
    )

    # [color sequence] convert sample log likelihoods to values in (0, 1)
    sample_log_likelihoods = dla_gp.sample_log_likelihoods[
        :, 0
    ]  # only query the DLA(1) likelihoods
    # TODO: marginalize over k DLAs
    max_like = np.nanmax(sample_log_likelihoods)
    min_like = np.nanmin(sample_log_likelihoods)

    colours = (sample_log_likelihoods - min_like) / (max_like - min_like)

    # scale to make the colour more visible
    # TODO: make it more reasonable. scatter only takes values between [0, 1].
    colours = colours * 5 - 4
    colours[colours < 0] = 0

    if not ax:
        fig, ax = plt.subplots(1, 1, figsize=(16, 5))

    ax.scatter(
        sample_z_dlas, dla_gp.dla_samples.log_nhi_samples, c=colours,
    )

    # [min max sample zDLAs] instead of using min max from sample_z_dlas
    # using the zDLAs converted from wavelengths will better reflect the
    # range of wavelengths range in the this_mu plot.
    z_dlas = (dla_gp.this_wavelengths / dla_gp.params.lya_wavelength) - 1
    ax.set_xlim(z_dlas.min(), z_dlas.max())
    ax.set_ylim(
        dla_gp.dla_samples.log_nhi_samples.min(),
        dla_gp.dla_samples.log_nhi_samples.max(),
    )
    ax.set_xlabel(r"$z_{DLA}$")
    ax.set_ylabel(r"$log N_{HI}$")
