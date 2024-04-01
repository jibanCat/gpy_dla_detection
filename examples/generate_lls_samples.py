"""
Generate samples of LLS, SLLS, DLAs, MgII, CIV from the inverse transform sampling of the CDDF.

# run in the terminal
python examples/generate_lls_samples.py
"""
import numpy as np
import os
import h5py

from gpy_dla_detection.cddf_samples import (
    generate_dla_samples,
    generate_civ_samples,
    generate_mgii_samples,
)


# Generate and save into files to the default directory
def generate_vanilla_run_samples(
    num_samples: int = 10000, resolution: int = 10000
) -> None:
    """
    Generate samples from the CDDF using inverse transform sampling.

    Parameters:
    - num_samples: The number of samples to generate.
    - resolution: The resolution of the inverse transform sampling.

    Returns:
    - samples: A NumPy array of samples from the CDDF.
    """
    dla_samples, dla_halton_sequence = generate_dla_samples(
        num_samples=num_samples, resolution=resolution
    )
    civ_samples, civ_halton_sequence = generate_civ_samples(
        num_samples=num_samples, resolution=resolution
    )
    mgii_samples, mgii_halton_sequence = generate_mgii_samples(
        num_samples=num_samples, resolution=resolution
    )

    # base directory
    base_dir = os.path.join("data", "dr12q", "processed")

    # save the samples
    with h5py.File(os.path.join(base_dir, "hi_samples.h5"), "w") as f:
        f.create_dataset("halton_sequence", data=dla_halton_sequence)
        f.create_dataset("samples_log_nhis", data=dla_samples)
    with h5py.File(os.path.join(base_dir, "civ_samples.h5"), "w") as f:
        f.create_dataset("halton_sequence", data=civ_halton_sequence)
        f.create_dataset("samples_log_nhis", data=civ_samples)
    with h5py.File(os.path.join(base_dir, "mgii_samples.h5"), "w") as f:
        f.create_dataset("halton_sequence", data=mgii_halton_sequence)
        f.create_dataset("samples_log_nhis", data=mgii_samples)
