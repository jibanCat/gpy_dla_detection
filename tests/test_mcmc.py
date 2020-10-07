"""
Test the MCMC results
"""
from .test_model import prepare_dla_model
from examples.plot_mcmc import plot_corner, plot_sample_this_mu


def test_posterior(nsamples: int = 10000, discard: int = 2000):
    # prepare a DLA model conditioned on a given spectrum
    dla_gp = prepare_dla_model(plate=5309, mjd=55929, fiber_id=362, z_qso=3.166)
    # sampling DLA parameters using Markov Chain Monte Carlo
    sampler = dla_gp.run_mcmc(nwalkers=32, kth_dla=1, nsamples=nsamples)

    plot_corner(sampler, discard=discard)
    plot_sample_this_mu(dla_gp, sampler, discard=discard)
