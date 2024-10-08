"""
Plot the GP model with sample likelihoods
"""

from typing import Optional

import os
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
        dla_mu, dla_M, dla_omega2 = dla_gp.this_dla_gp(map_z_dlas, 10**map_log_nhis)

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
    ax.set_ylim(this_mu.min() - 2, this_mu.max() + 1)
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
        sample_z_dlas,
        dla_gp.dla_samples.log_nhi_samples,
        c=colours,
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


def plot_real_spectrum_space(gp, lya_gp, nth_lya, title=""):
    """
    Plot the real spectrum space with the GP model fits for Lyman-alpha absorption.

    Parameters:
    ----------
    gp : NullGPMAT
        The Null GP model used for the quasar spectrum's continuum.
    lya_gp : DLAGPMAT
        The Lyman-alpha GP model used for the DLA detection.
    nth_lya : int
        The number of absorbers to plot.
    title : str, optional
        Title of the plot (default is an empty string).
    """

    # Extract the maximum a posteriori estimates
    MAP_z_dla, MAP_log_nhi = lya_gp.maximum_a_posteriori()
    map_z_dlas = MAP_z_dla[nth_lya - 1, :nth_lya]
    map_log_nhis = MAP_log_nhi[nth_lya - 1, :nth_lya]

    # Get the absorption profile for the DLA model
    lya_mu, _, _ = lya_gp.this_dla_gp(map_z_dlas, 10**map_log_nhis)

    # Plotting the real spectrum space and the GP models
    plt.figure(figsize=(16, 5))

    # Plot the quasar spectrum's flux (data)
    plt.plot(gp.X, gp.Y, label="Data")

    # Plot the instrumental uncertainty as a fill_between
    plt.fill_between(
        gp.X,
        gp.Y - 2 * np.sqrt(gp.v),
        gp.Y + 2 * np.sqrt(gp.v),
        label="Instrumental Uncertainty (95%)",
        color="C0",
        alpha=0.3,
    )

    # Plot the GP null model's continuum (mu)
    plt.plot(
        gp.rest_wavelengths,
        gp.mu,
        label="GP null model (mu = continuum)",
        color="C3",
        ls="--",
    )

    # Plot the Lyman-alpha GP model's mean function (absorption)
    plt.plot(
        gp.X,
        lya_mu,
        label="GP Lya model (mu = meanflux)",
        color="red",
    )

    # Set plot labels and limits
    plt.xlabel("Rest-frame Wavelengths [$\AA$]")
    plt.ylabel("Normalized Flux")
    plt.legend()
    plt.ylim(-1, 5)
    plt.title(title)

    # Show the plot
    plt.tight_layout()
    plt.show()


def plot_samples_vs_this_mu(dla_gp, bayes, filename, sub_dir="images", title=""):
    """
    Create and save DLA detection plots for a given spectrum.

    Parameters:
    ----------
    dla_gp : DLAGPMAT
        The DLA model (DLAGPMAT object).
    bayes : BayesModelSelect
        Bayesian model selection object used to calculate probabilities.
    filename : str
        Filename for saving the plot (without file extension).
    sub_dir : str, optional
        Sub-directory to save the plots (default is "images").
    title : str, optional
        Title of the plot (default is an empty string).
    """

    # Ensure the output directory exists
    os.makedirs(sub_dir, exist_ok=True)
    file_path = os.path.join(sub_dir, f"{filename}.png")

    # Determine the number of absorbers to plot based on the posterior probability
    nth_lya = 1 + bayes.model_posteriors[2:].argmax() if bayes.p_dla > 0.9 else 0

    # Extract sample z_dlas and sample log likelihoods
    sample_z_dlas = dla_gp.dla_samples.sample_z_dlas(
        dla_gp.this_wavelengths, dla_gp.z_qso
    )
    sample_log_likelihoods = dla_gp.sample_log_likelihoods[:, 0]  # DLA(1) likelihoods

    # Scale sample log likelihoods to color values
    max_like = np.nanmax(sample_log_likelihoods)
    min_like = np.nanmin(sample_log_likelihoods)
    colours = (sample_log_likelihoods - min_like) / (max_like - min_like)
    colours = colours * 5 - 4  # scale for visibility
    colours[colours < 0] = 0

    # Generate the maximum a posteriori estimates
    MAP_z_dla, MAP_log_nhi = dla_gp.maximum_a_posteriori()
    map_z_dlas = MAP_z_dla[nth_lya - 1, :nth_lya]
    map_log_nhis = MAP_log_nhi[nth_lya - 1, :nth_lya]

    # Generate the absorption profile
    lya_mu, _, _ = dla_gp.this_dla_gp(map_z_dlas, 10**map_log_nhis)

    # Only plot the spectrum within the search range
    this_rest_wavelengths = dla_gp.x
    ind = this_rest_wavelengths < dla_gp.params.lya_wavelength
    this_rest_wavelengths = this_rest_wavelengths[ind]
    lya_mu = lya_mu[ind]

    # Create the plot with two panels
    fig, ax = plt.subplots(2, 1, figsize=(16, 10))

    # Plot the real spectrum space
    ax[0].plot(
        (this_rest_wavelengths * (1 + dla_gp.z_qso)) / dla_gp.params.lya_wavelength - 1,
        dla_gp.Y[ind],
    )
    ax[0].plot(
        (this_rest_wavelengths * (1 + dla_gp.z_qso)) / dla_gp.params.lya_wavelength - 1,
        lya_mu,
        label=r"$\mathcal{M}$"
        + r" HCD({n}); ".format(n=nth_lya)
        + "z_dlas = ({}); ".format(",".join("{:.3g}".format(z) for z in map_z_dlas))
        + "lognhi = ({})".format(",".join("{:.3g}".format(n) for n in map_log_nhis)),
        color="red",
    )
    ax[0].fill_between(
        (this_rest_wavelengths * (1 + dla_gp.z_qso)) / dla_gp.params.lya_wavelength - 1,
        dla_gp.Y[ind] - 2 * np.sqrt(dla_gp.v[ind]),
        dla_gp.Y[ind] + 2 * np.sqrt(dla_gp.v[ind]),
        label="Instrumental Uncertainty (95%)",
        color="C0",
        alpha=0.3,
    )
    ax[0].set_xlim(sample_z_dlas.min(), dla_gp.z_qso)
    ax[0].legend()
    ax[0].set_title(title)

    # Plot the posterior space
    ax[1].scatter(
        sample_z_dlas,
        dla_gp.dla_samples.log_nhi_samples,
        c=colours,
        marker="o",
        alpha=0.5,
    )
    ax[1].scatter(map_z_dlas, map_log_nhis, marker="*", s=100, color="C3")
    ax[1].set_xlim(sample_z_dlas.min(), dla_gp.z_qso)
    ax[1].set_ylim(
        dla_gp.dla_samples.log_nhi_samples.min(),
        dla_gp.dla_samples.log_nhi_samples.max(),
    )
    ax[1].set_xlabel(r"$z_{Lya}$")
    ax[1].set_ylabel(r"$log N_{HI}$")

    # Save the plot to the specified directory
    plt.tight_layout()
    plt.savefig(file_path)
    plt.close()

    # Plot and save the real spectrum space plot
    plot_real_spectrum_space(dla_gp, dla_gp, nth_lya, title=title)
    plt.savefig(file_path)
    plt.close()
