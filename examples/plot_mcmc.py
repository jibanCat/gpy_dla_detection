"""
Plot the mcmc results
"""
from typing import List, Optional
import numpy as np
from matplotlib import pyplot as plt
import emcee
import corner

from gpy_dla_detection.dla_gp import DLAGP


def plot_corner(
    sampler: emcee.EnsembleSampler,
    discard: int = 2000,
    min_z: float = 0.0,
    max_z: float = 6.0,
    truths: Optional[List[float]] = None,
):
    """
    :param sampler: the emcee sampler you sampled from DLA model, dla_gp.run_mcmc()
    :param discard: how many samples in the MCMC chain you would like to discard. The first
        few samples should always be discarded due to the sampler won't settle to the right
        peak at first. The discard number varies depending on the problem.
    :param min_z: this is optional. If set, then the chians with zDLA samples lower than
        min_z will be discarded.
    :param max_z: this is optional. If set, then the chians with zDLA samples higher than
        min_z will be discarded.
    """
    samples = sampler.get_chain()

    labels = ["zDLA", "logNHI"]

    fig, axes = plt.subplots(2, figsize=(10, 7), sharex=True)
    for i in range(len(labels)):
        ax = axes[i]
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])
    axes[-1].set_xlabel("set number")

    tau = sampler.get_autocorr_time()
    print("AutoCorrelation analysis", tau)

    # [min_z, max_z] only plot those samples within min_z~max_z range
    selected_ind = np.where(
        (samples[:, :, :].mean(axis=0)[:, 0] > min_z) & 
        (samples[:, :, :].mean(axis=0)[:, 0] < max_z) )[0]
    # flatten multiple chains to one chain
    flat_samples = samples[discard:, selected_ind, :].reshape(
        (samples.shape[0] - discard) * len(selected_ind), samples.shape[-1]
    )

    # a corner plot
    fig = corner.corner(flat_samples, labels=labels, truths=truths)
    fig.set_figheight(10)
    fig.set_figwidth(10)


def plot_sample_this_mu(dla_gp: DLAGP, sampler: emcee.EnsembleSampler, discard: int):
    # discard first 100; it is still multi-modal though
    flat_samples = sampler.get_chain(discard=discard, thin=2, flat=True)

    plt.figure(figsize=(16, 5))

    inds = np.random.randint(len(flat_samples), size=200)
    for ind in inds:
        sample = flat_samples[ind]
        z_dla, log_nhi = sample
        dla_mu, dla_M, dla_omega2 = dla_gp.this_dla_gp(
            np.array([z_dla]), np.array([10 ** log_nhi])
        )

        plt.plot(dla_gp.x, dla_mu, color="C1", alpha=0.1)

    plt.plot(dla_gp.x, dla_gp.y, color="C0")
    plt.xlabel(r"rest-wavelengths $\lambda_{\mathrm{rest}}$ $\AA$")
    plt.ylabel(r"normalised flux")
