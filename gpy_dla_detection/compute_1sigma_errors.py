import numpy as np
from scipy.stats import gaussian_kde


def compute_1sigma_errors(
    MAP_z_dlas,
    MAP_log_nhi,
    sample_z_dlas,
    sample_log_nhi_samples,
    sample_log_likelihoods,
):
    """
    Compute the 1-sigma errors for MAP estimates of z_dlas and log_nhi.

    Parameters
    ----------
    MAP_z_dlas : np.ndarray
        The MAP estimates of z_dlas (shape: (number of detected DLAs,))
    MAP_log_nhi : np.ndarray
        The MAP estimates of log_nhi (shape: (number of detected DLAs,))
    sample_z_dlas : np.ndarray
        Samples of z_dlas (shape: (10000,))
    sample_log_nhi_samples : np.ndarray
        Samples of log_nhi (shape: (10000,))
    sample_log_likelihoods : np.ndarray
        Log likelihoods for the samples (shape: (10000,))

    Returns
    -------
    sigma_z_dlas : np.ndarray
        1-sigma errors for z_dlas (shape: (number of detected DLAs,))
    sigma_log_nhi : np.ndarray
        1-sigma errors for log_nhi (shape: (number of detected DLAs,))
    """

    # Convert log-likelihoods to probabilities
    sample_probabilities = np.exp(
        sample_log_likelihoods - np.max(sample_log_likelihoods)
    )

    # Define arrays to store the 1-sigma errors
    sigma_z_dlas = np.zeros_like(MAP_z_dlas)
    sigma_log_nhi = np.zeros_like(MAP_log_nhi)

    # Iterate over each detected DLA to compute the 1-sigma error for each parameter
    for i, (map_z, map_log_nhi) in enumerate(zip(MAP_z_dlas, MAP_log_nhi)):
        # Mask samples based on the current detected DLA
        z_mask = (
            np.abs(sample_z_dlas - map_z) < 0.1
        )  # Adjust this range based on your prior knowledge
        nhi_mask = (
            np.abs(sample_log_nhi_samples - map_log_nhi) < 0.5
        )  # Adjust range similarly

        # Get the intersection of z and nhi masks for this detected DLA
        combined_mask = z_mask & nhi_mask
        print(sum(combined_mask))

        # Filter samples and probabilities for the current DLA
        z_samples_filtered = sample_z_dlas[combined_mask]
        log_nhi_samples_filtered = sample_log_nhi_samples[combined_mask]
        probabilities_filtered = sample_probabilities[combined_mask]

        # Check if we have enough samples to perform KDE
        if len(z_samples_filtered) < 10 or len(log_nhi_samples_filtered) < 10:
            print(f"Warning: Not enough samples for DLA {i+1}. Skipping.")
            sigma_z_dlas[i] = np.nan
            sigma_log_nhi[i] = np.nan
            continue

        # Estimate the density of z_dlas and log_nhi separately using Gaussian KDE
        kde_z = gaussian_kde(z_samples_filtered, weights=probabilities_filtered)
        kde_nhi = gaussian_kde(log_nhi_samples_filtered, weights=probabilities_filtered)

        # Compute the cumulative distribution function (CDF) for z_dlas
        cdf_z = lambda z: kde_z.integrate_box_1d(-np.inf, z)
        lower_z = map_z
        upper_z = map_z

        # Find the lower bound for the 68% confidence interval for z_dlas
        while cdf_z(lower_z) > 0.16:
            lower_z -= 0.001

        # Find the upper bound for the 68% confidence interval for z_dlas
        while cdf_z(upper_z) < 0.84:
            upper_z += 0.001

        # 1-sigma error for z_dlas
        sigma_z_dlas[i] = (upper_z - lower_z) / 2

        # Compute the cumulative distribution function (CDF) for log_nhi
        cdf_nhi = lambda nhi: kde_nhi.integrate_box_1d(-np.inf, nhi)
        lower_nhi = map_log_nhi
        upper_nhi = map_log_nhi

        # Find the lower bound for the 68% confidence interval for log_nhi
        while cdf_nhi(lower_nhi) > 0.16:
            lower_nhi -= 0.001

        # Find the upper bound for the 68% confidence interval for log_nhi
        while cdf_nhi(upper_nhi) < 0.84:
            upper_nhi += 0.001

        # 1-sigma error for log_nhi
        sigma_log_nhi[i] = (upper_nhi - lower_nhi) / 2

    return sigma_z_dlas, sigma_log_nhi


def compute_1sigma_errors_fast(
    MAP_z_dlas,
    MAP_log_nhi,
    sample_z_dlas,
    sample_log_nhi_samples,
    sample_log_likelihoods,
):
    """
    Compute the 1-sigma errors for MAP estimates of z_dlas and log_nhi using a Gaussian approximation.

    Parameters
    ----------
    MAP_z_dlas : np.ndarray
        The MAP estimates of z_dlas (shape: (number of detected DLAs,))
    MAP_log_nhi : np.ndarray
        The MAP estimates of log_nhi (shape: (number of detected DLAs,))
    sample_z_dlas : np.ndarray
        Samples of z_dlas (shape: (10000,))
    sample_log_nhi_samples : np.ndarray
        Samples of log_nhi (shape: (10000,))
    sample_log_likelihoods : np.ndarray
        Log likelihoods for the samples (shape: (10000,))

    Returns
    -------
    sigma_z_dlas : np.ndarray
        1-sigma errors for z_dlas (shape: (number of detected DLAs,))
    sigma_log_nhi : np.ndarray
        1-sigma errors for log_nhi (shape: (number of detected DLAs,))
    """

    # Convert log-likelihoods to probabilities
    sample_probabilities = np.exp(
        sample_log_likelihoods - np.max(sample_log_likelihoods)
    )

    # Define arrays to store the 1-sigma errors
    sigma_z_dlas = np.zeros_like(MAP_z_dlas)
    sigma_log_nhi = np.zeros_like(MAP_log_nhi)

    # Iterate over each detected DLA to compute the 1-sigma error for each parameter
    for i, (map_z, map_log_nhi) in enumerate(zip(MAP_z_dlas, MAP_log_nhi)):
        # Mask samples based on the current detected DLA
        z_mask = np.abs(sample_z_dlas - map_z) < 0.1  # Narrow window around MAP_z
        nhi_mask = (
            np.abs(sample_log_nhi_samples - map_log_nhi) < 0.5
        )  # Narrow window around MAP_log_nhi

        # Get the intersection of z and nhi masks for this detected DLA
        combined_mask = z_mask & nhi_mask

        # Filter samples and probabilities for the current DLA
        z_samples_filtered = sample_z_dlas[combined_mask]
        log_nhi_samples_filtered = sample_log_nhi_samples[combined_mask]

        probabilities_filtered = sample_probabilities[combined_mask]

        # Normalize the probabilities
        probabilities_filtered /= np.sum(probabilities_filtered)

        # Estimate the 1-sigma errors by fitting Gaussian distributions
        sigma_z_dlas[i] = np.sqrt(
            np.average(
                (z_samples_filtered - map_z) ** 2, weights=probabilities_filtered
            )
        )
        sigma_log_nhi[i] = np.sqrt(
            np.average(
                (log_nhi_samples_filtered - map_log_nhi) ** 2,
                weights=probabilities_filtered,
            )
        )

    return sigma_z_dlas, sigma_log_nhi
