import os
import ctypes
import numpy as np

# Define the class to wrap the Voigt profile calculation
class VoigtProfile:
    def __init__(self):
        # Get the directory of the current file
        current_dir = os.path.dirname(os.path.abspath(__file__))

        # Load the shared library
        self.voigt_lib = ctypes.CDLL(os.path.join(current_dir, '_voigt.so'))

        # Define the argument and return types for the `compute_voigt` function
        self.voigt_lib.compute_voigt.argtypes = [
            ctypes.POINTER(ctypes.c_double),  # lambdas array
            ctypes.c_double,                 # z (redshift)
            ctypes.c_double,                 # N (column density)
            ctypes.c_int,                    # num_lines
            ctypes.c_int,                    # num_points
            ctypes.POINTER(ctypes.c_double)  # output profile
        ]
        self.voigt_lib.compute_voigt.restype = None  # No return value

    def compute_voigt_profile(self, wavelengths, nhi, z_dla, num_lines=3):
        """
        Compute the Voigt absorption profile using the compiled C function.
        
        Parameters:
            wavelengths (np.ndarray): Wavelengths array.
            z (float): Redshift.
            N (float): Column density.
            num_lines (int): Number of Lyman series lines to consider.
        
        Returns:
            np.ndarray: The computed absorption profile.
        """
        # Ensure wavelengths is a C-contiguous array of doubles
        wavelengths = np.ascontiguousarray(wavelengths, dtype=np.float64)

        # Number of points in the input wavelengths array
        num_points = len(wavelengths)

        # Prepare an output array for the profile (adjust size if necessary)
        profile = np.zeros(num_points - 2 * 3, dtype=np.float64)

        # Call the C function
        self.voigt_lib.compute_voigt(
            wavelengths.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_double(z_dla),
            ctypes.c_double(nhi),
            ctypes.c_int(num_lines),
            ctypes.c_int(num_points),
            profile.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        )

        return profile