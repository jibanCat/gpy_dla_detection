import os
import glob
from typing import Tuple, List, Dict

import numpy as np
from astropy.io import fits
from astropy.table import Table
import desispec.io
from collections import namedtuple

# Define a namedtuple for storing each spectrum's data
SpectrumData = namedtuple(
    "SpectrumData", ["wavelengths", "flux", "noise_variance", "pixel_mask"]
)


class DESISpectrumReader:
    """
    A class to read and process DESI spectra and redshift catalog.
    """

    def __init__(self, spectra_filename: str, zbest_filename: str):
        """
        Initializes the reader with a spectra file and the corresponding redshift catalog.

        Parameters:
        ----------
        spectra_filename : str
            The filename of the DESI spectra file.
        zbest_filename : str
            The filename of the DESI redshift catalog.
        """
        self.spectra_filename = spectra_filename
        self.zbest_filename = zbest_filename
        self.spectra_data = {}  # Dictionary to store each spectrum's data
        self.redshift_data = None

    def read_spectra(self):
        """
        Reads the DESI spectra file, organizes multiple spectra into a dictionary,
        and sorts wavelengths, flux, noise variance, and pixel mask for each spectrum.
        """
        allcoadds = desispec.io.read_spectra(self.spectra_filename)
        num_spectra = allcoadds.num_spectra()  # Number of spectra in the file
        bands = allcoadds.bands  # Bands available in the spectra file (b, r, z)

        # Loop over each spectrum in the file
        for spec_idx in range(num_spectra):
            all_wavelengths = []
            all_flux = []
            all_noise_variance = []
            all_pixel_mask = []

            # Process each band (b, r, z) for the current spectrum
            for band in bands:
                wave = allcoadds.wave[band]
                flux = allcoadds.flux[band][spec_idx]
                ivar = allcoadds.ivar[band][spec_idx]
                mask = allcoadds.mask[band][spec_idx]

                # Convert inverse variance to variance
                noise_variance = np.zeros(ivar.shape)
                ind = ivar == 0
                noise_variance[:] = np.nan
                noise_variance[~ind] = 1 / ivar[~ind]

                # Collect the data for this band and add to the spectrum data
                all_wavelengths.extend(wave)
                all_flux.extend(flux.flatten())
                all_noise_variance.extend(noise_variance.flatten())
                all_pixel_mask.extend(mask.flatten())

            # Sort all data by wavelength
            sorted_indices = np.argsort(all_wavelengths)
            wavelengths_sorted = np.array(all_wavelengths)[sorted_indices]
            flux_sorted = np.array(all_flux)[sorted_indices]
            noise_variance_sorted = np.array(all_noise_variance)[sorted_indices]
            pixel_mask_sorted = np.array(all_pixel_mask)[sorted_indices]

            # Store the sorted data for this spectrum in a namedtuple
            self.spectra_data[f"spectrum_{spec_idx}"] = SpectrumData(
                wavelengths=wavelengths_sorted,
                flux=flux_sorted,
                noise_variance=noise_variance_sorted,
                pixel_mask=np.bool_(pixel_mask_sorted),
            )

    def read_redshift_catalog(self):
        """
        Reads the redshift data from the zbest file.

        Stores Z, DELTACHI2, ZERR, ZWARN, TARGETID columns.
        """
        allzbest = Table.read(self.zbest_filename)
        self.redshift_data = {
            "Z": allzbest["Z"].data,
            "DELTACHI2": allzbest["DELTACHI2"].data,
            "ZERR": allzbest["ZERR"].data,
            "ZWARN": allzbest["ZWARN"].data,
            "TARGETID": allzbest["TARGETID"].data,
        }

    def get_spectrum_data(self, spectrum_id: str) -> SpectrumData:
        """
        Returns the data for a specific spectrum by its ID.

        Parameters:
        ----------
        spectrum_id : str
            The identifier for the spectrum (e.g., "spectrum_0" for the first spectrum).

        Returns:
        --------
        SpectrumData : namedtuple
            A namedtuple containing wavelengths, flux, noise_variance, and pixel_mask for the spectrum.
        """
        return self.spectra_data.get(spectrum_id, None)

    def get_all_spectrum_ids(self) -> List[str]:
        """
        Returns a list of all spectrum IDs available in the spectra file.

        Returns:
        --------
        List of spectrum IDs.
        """
        return list(self.spectra_data.keys())

    def get_redshift_data(self) -> Dict[str, np.ndarray]:
        """
        Returns the redshift data.

        Returns:
        --------
        redshift_data : dict
            Dictionary with redshift information: Z, DELTACHI2, ZERR, ZWARN, TARGETID.
        """
        return self.redshift_data


# import os
# import glob
# from typing import List, Tuple


def read_single_folder(folder_path: str) -> Tuple[str, str]:
    """
    Reads the two required files (spectra and zbest) from a single folder.

    Parameters:
    ----------
    folder_path : str
        The path to the folder containing the spectra-*.fits and zbest-*.fits files.

    Returns:
    --------
    Tuple containing the paths to the spectra and zbest files.
    """
    # Ensure the folder path exists and contains files
    if not os.path.exists(folder_path):
        print(f"Warning: Folder not found: {folder_path}")
        return None, None

    # Get all files in the folder
    this_filenames = glob.glob(os.path.join(folder_path, "*"))

    # Initialize variables for file paths
    spectra_file = None
    zbest_file = None

    # Loop through the filenames in the folder to identify the spectra and zbest files
    for filename in this_filenames:
        if "spectra-" in os.path.basename(filename):
            spectra_file = filename
        elif "zbest-" in os.path.basename(filename):
            zbest_file = filename

    # Return the file paths if both files are found, otherwise return None
    if spectra_file and zbest_file:
        return spectra_file, zbest_file
    else:
        print(f"Warning: Missing files in {folder_path}")
        return None, None


def process_single_folder(folder_path: str) -> DESISpectrumReader:
    """
    Process the spectra and zbest files in a single folder.

    Parameters:
    ----------
    folder_path : str
        The folder containing spectra-*.fits and zbest-*.fits files.

    Returns:
    --------
    DESISpectrumReader or None
        A DESISpectrumReader instance containing the spectra and redshift data,
        or None if files are missing.
    """
    spectra_file, zbest_file = read_single_folder(folder_path)

    if spectra_file and zbest_file:
        # Process spectra
        reader = DESISpectrumReader(spectra_file, zbest_file)
        reader.read_spectra()
        reader.read_redshift_catalog()

        print(f"Processed spectra from {spectra_file}")
        print(f"Processed redshift catalog from {zbest_file}")

        # Return the DESISpectrumReader instance
        return reader
    else:
        # If files are missing, return None
        print(f"Skipping folder: {folder_path}")
        return None


def read_all_catalogs(base_path: str) -> List[DESISpectrumReader]:
    """
    Reads all spectra and redshift catalogs from the given path.

    Parameters:
    ----------
    base_path : str
        The base path where the spectra and redshift catalogs are located.

    Returns:
    --------
    readers : list of DESISpectrumReader
        A list of initialized DESISpectrumReader objects for each catalog.
    """
    all_filenames = glob.glob(os.path.join(base_path, "*/*"))
    readers = []

    for spectra_path in all_filenames:
        # Process each folder and check for missing files
        reader = process_single_folder(spectra_path)

        # Only append valid readers (those that successfully processed files)
        if reader is not None:
            readers.append(reader)

    return readers
