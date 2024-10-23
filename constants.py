"""
set constant values used in DLA finder
"""

from scipy.constants import speed_of_light

# set redshift window for quasars
zmin_qso = 2.0
zmax_qso = 4.25
# set the wave window for DLA search
# rest-frame angstroms
search_minlam = 900.0
search_maxlam = 1230.0
# set the log10 column density search range
# for DLAs, units of (cm^-2)
nhimin = 20.1
nhimax = 22.6
# set delta chi2 threshold for detection
detection = 0.01

# set constants for DLA profile and BAL masking
c = speed_of_light / 1000.0  # m/s -> km/s
Lya_line = float(1215.67)  ## Lya wavelength [A]
Lyb_line = float(1025.72)  ## Lyb wavelength [A]
oscillator_strength_Lya = 0.41641
oscillator_strength_Lyb = 0.079142
gamma_Lya = 6.2648e08  # s^-1 damping constant
gamma_Lyb = 1.6725e8  # s^-1 damping constant
gastemp = 5 * 1e4  # K


# constants for masking broad absorption lines
# line centers identical to those defined in igmhub/picca
bal_lines = {
    "CIV": 1549.0,
    "SiIV2": 1403.0,
    "SiIV1": 1394.0,
    "NV": 1240.81,
    "Lya": 1216.1,
    "CIII": 1175.0,
    "PV2": 1128.0,
    "PV1": 1117.0,
    "SIV2": 1074.0,
    "SIV1": 1062.0,
    "OIV": 1031.0,
    "OVI": 1037.0,
    "OI": 1039.0,
    "Lyb": 1025.7,
    "Ly3": 972.5,
    "CIII": 977.0,
    "NIII": 989.9,
    "Ly4": 949.7,
}

Lyman_series = dict()

# optical depth parameters from Kamble et al. (2020)
# used by QSO-HIZv1.1, N>2 are negelected
# arxiv 1904.01110
Lyman_series["kamble20"] = {
    "Lya": {"line": Lya_line, "A": 0.00554, "B": 3.182},
    #'Lyb'     : { 'line':Lyb_line,  'A':0.00554/5.2615,   'B':3.182 },
    #'Ly3'     : { 'line':972.537,  'A':0.00554/14.356,   'B':3.182 },
    #'Ly4'     : { 'line':949.7431, 'A':0.00554/29.85984, 'B':3.182 },
    #'Ly5'     : { 'line':937.8035, 'A':0.00554/53.36202, 'B':3.182 },
}

# optical depth parameters from Turner et al. (2024)
# arxiv 2405.06743
Lyman_series["turner24"] = {
    "Lya": {"line": Lya_line, "A": 0.00246, "B": 3.182},
    "Lyb": {"line": Lyb_line, "A": 0.00246 / 5.2615, "B": 3.62},
    "Ly3": {"line": 972.537, "A": 0.00246 / 14.356, "B": 3.62},
    "Ly4": {"line": 949.7431, "A": 0.00246 / 29.85984, "B": 3.62},
    "Ly5": {"line": 937.8035, "A": 0.00246 / 53.36202, "B": 3.62},
}
