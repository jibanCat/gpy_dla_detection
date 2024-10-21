# GP-DLA finder for DESI quasar spectra, in Python

![rainbow_dlas](https://jibancat.github.io/images/RdBu_dlas.png)

This code repository contains code to completely reproduce the DLA
catalog reported in

> R Garnett, S Ho, S Bird, and J Schnedier. Detecting Damped Lyman-α
> Absorbers with Gaussian Processes. [arXiv:1605.04460
> [astro-ph.CO]](https://arxiv.org/abs/1605.04460),

and

> M-F Ho, S Bird, and R Garnett. Detecting Multiple DLAs per
> Spectrum in SDSS DR12 with Gaussian Processes. [arXiv:2003.11036
> [astro-ph.CO]](https://arxiv.org/abs/2003.11036),

all the intermediate data products including the Gaussian
process null model could be acquired via running the MATLAB version of the code: https://github.com/rmgarnett/gp_dla_detection/.
The design of this repo assumes users already had the learned GP model and the users want to apply the trained model on new quasar spectra.

The parameters are tunable in the `gpy_dla_detection.set_parameters.Parameters` as instance attributes. The provided parameters should
exactly reproduce the catalog in the work of Ho-Bird-Garnett (2020);
however, you may feel free to modify these choices as you see fit.

## Downloading the external DLA catalogues and the learned model

First we download the raw catalog data (requires both `wget` and `gawk`):

    # in shell
    cd data/scripts
    ./download_catalogs.sh
    ./download_gp_files.sh

## Compilation and Installation Guide for C Helper Functions

> Warning: If you don't compile the C voigt function, you automatically fall back to the slower Python version.

The processing code uses a C helper function `gpy_dla_detection/ctypes_voigt.c` to efficiently compute Voigt profiles. This requires the `libcerf` library to be installed. Below are the steps for compiling `libcerf` from source and setting up the environment.

1. Step 1: Compile `libcerf` from Source

Navigate to your home directory:

```bash
cd $HOME
git clone https://jugit.fz-juelich.de/mlz/libcerf.git
cd libcerf
mkdir build
cd build
cmake ..
make
ctest
make install DESTDIR=~/.local/
```

2. Step 2: Compile the C Helper Function

Navigate to the directory where the `desi_gpy_dla_detection` repository is located:

```bash
cd /path/to/desi_gpy_dla_detection/gpy_dla_detection
```

Compile the C file using the installed `libcerf`:

```bash
cc -fPIC -shared -o _voigt.so ctypes_voigt.c -I$HOME/.local/usr/local/include -L$HOME/.local/usr/local/lib64 -lcerf
```

3. Step 3: Set Up the Environment

Update the LD_LIBRARY_PATH to include the path to the compiled libcerf library:

```bash
echo 'export LD_LIBRARY_PATH=$HOME/.local/usr/local/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
```

- Replace `/path/to/desi_gpy_dla_detection` with the actual path where the code repository is located.
- Ensure that the cc compiler and other necessary build tools (like `cmake`) are installed on your system.

## Run GP DLA finder for DESI like .fits file

We provide a simple Python script, `run_bayes_select.py` , to reproduce the MATLAB code `multi_dlas/process_qsos_multiple_dlas_meanflux.m` and create a HDF5 catalogue in the end with the posterior probability of having DLAs in a given spectrum.

To run this Python script, do:

```bash
# Set default arguments if not provided
SPECTRA_FILENAME="${SPECTRA_FILENAME:-/path/to/spectra.fits}"
ZBEST_FILENAME="${ZBEST_FILENAME:-/path/to/zbest.fits}"
LEARNED_FILE="${LEARNED_FILE:-data/dr12q/processed/learned_qso_model_lyseries_variance_wmu_boss_dr16q_minus_dr12q_gp_851-1421.mat}"
CATALOG_NAME="${CATALOG_NAME:-data/dr12q/processed/catalog.mat}"
LOS_CATALOG="${LOS_CATALOG:-data/dla_catalogs/dr9q_concordance/processed/los_catalog}"
DLA_CATALOG="${DLA_CATALOG:-data/dla_catalogs/dr9q_concordance/processed/dla_catalog}"
DLA_SAMPLES_FILE="${DLA_SAMPLES_FILE:-data/dr12q/processed/dla_samples_a03.mat}"
SUB_DLA_SAMPLES_FILE="${SUB_DLA_SAMPLES_FILE:-data/dr12q/processed/subdla_samples.mat}"
MIN_Z_SEPARATION="${MIN_Z_SEPARATION:-3000.0}"
PREV_TAU_0="${PREV_TAU_0:-0.00554}"
PREV_BETA="${PREV_BETA:-3.182}"
MAX_DLAS="${MAX_DLAS:-3}"
PLOT_FIGURES="${PLOT_FIGURES:-0}"
MAX_WORKERS="${MAX_WORKERS:-32}"
BATCH_SIZE="${BATCH_SIZE:-313}"
LOADING_MIN_LAMBDA="${LOADING_MIN_LAMBDA:-800}"
LOADING_MAX_LAMBDA="${LOADING_MAX_LAMBDA:-1550}"
NORMALIZATION_MIN_LAMBDA="${NORMALIZATION_MIN_LAMBDA:-1425}"
NORMALIZATION_MAX_LAMBDA="${NORMALIZATION_MAX_LAMBDA:-1475}"
MIN_LAMBDA="${MIN_LAMBDA:-850.75}"
MAX_LAMBDA="${MAX_LAMBDA:-1420.75}"
DLAMBDA="${DLAMBDA:-0.25}"
K="${K:-20}"
MAX_NOISE_VARIANCE="${MAX_NOISE_VARIANCE:-9}"

# in shell, to make predictions on two QSO spectra.
python run_bayes_select.py \
--spectra_filename "${SPECTRA_FILENAME}" \
--zbest_filename "${ZBEST_FILENAME}" \
--learned_file "${LEARNED_FILE}" \
--catalog_name "${CATALOG_NAME}" \
--los_catalog "${LOS_CATALOG}" \
--dla_catalog "${DLA_CATALOG}" \
--dla_samples_file "${DLA_SAMPLES_FILE}" \
--sub_dla_samples_file "${SUB_DLA_SAMPLES_FILE}" \
--min_z_separation "${MIN_Z_SEPARATION}" \
--prev_tau_0 "${PREV_TAU_0}" \
--prev_beta "${PREV_BETA}" \
--max_dlas "${MAX_DLAS}" \
--plot_figures "${PLOT_FIGURES}" \
--max_workers "${MAX_WORKERS}" \
--batch_size "${BATCH_SIZE}" \
--loading_min_lambda "${LOADING_MIN_LAMBDA}" \
--loading_max_lambda "${LOADING_MAX_LAMBDA}" \
--normalization_min_lambda "${NORMALIZATION_MIN_LAMBDA}" \
--normalization_max_lambda "${NORMALIZATION_MAX_LAMBDA}" \
--min_lambda "${MIN_LAMBDA}" \
--max_lambda "${MAX_LAMBDA}" \
--dlambda "${DLAMBDA}" \
--k "${K}" \
--max_noise_variance "${MAX_NOISE_VARIANCE}"
```

Each DESI `.fits` file has multiple spectra. The output of this script is a DLA catalog for all spectra in that `.fits` file.
You probably can re-design the whole pipeline for multiple `.fits` file, but I am not that familiar with DESI data structure at the moment.

## For developers

There are some customizable features for this GP-DLA model.
For customization, go to this tutorial:

- Number of DLA samples
- Marginalizing over meanflux for purity (Ho 2021 model)
- Resample the DLA column density prior

## Additional feature: Marginalizing over metal lines for DLAs

To improve the purity, one can do the metal line detection alongside the DLAs.

Here I provide some routines but this is unpublished so no guarantee for improvement and any biases introduced.

## Additional feature: Marginalizing over quasar redshift for DLAs

Our GP method could also be used to estimate the quasar redshift. Details could be found in [Leah (2020)](https://arxiv.org/abs/2006.07343). Note that the original MATLAB code is in https://github.com/sbird/gp_qso_redshift. Here we only translated part of the codes without the learning functionality. To use the method, we need to download the trained GP model:

- [learned_zqso_only_model_outdata_full_dr9q_minus_concordance_norm_1176-1256.mat](https://drive.google.com/file/d/1SqAU_BXwKUx8Zr38KTaA_nvuvbw-WPQM/view?usp=sharing)

For how to use the redshift estimation method, please find the notebook in `notebooks/`.

- [Quasar Redshift Estimations.ipynb](https://nbviewer.jupyter.org/github/jibanCat/gpy_dla_detection/blob/zqso_notebooks/notebooks/Quasar%20Redshift%20Estimations.ipynb)

## Python requirements

- Python 3.5+: I use typing
- numpy
- scipy
- matplotlib
- h5py
- astropy: for reading fits file

To prevent users have troubles setting up the environment,
here is the exact packages I used to run the tests (below is my `poetry` .toml file):

```
[tool.poetry]
name = "gpy_dla_detection"
version = "0.1.0"
description = "Detecting damped Lyman alpha absorbers with Gaussian processes, in Python!"
authors = ["mho026@ucr.edu"]

[tool.poetry.dependencies]
python = "^3.8"
numpy = "^1.19.1"
scipy = "^1.5.2"
matplotlib = "^3.3.0"
h5py = "^2.10.0"
astropy = "^4.0.1"

[tool.poetry.dev-dependencies]
pytest = "^5.2"
black = "^19.10b0"
pylint = "^2.5.3"
mypy = "^0.782"
ipython = "^7.17.0"

[build-system]
requires = ["poetry>=0.12"]
build-backend = "poetry.masonry.api"
```

The dependency:

```
astropy 4.0.1.post1 Community-developed python astronomy tools
h5py 2.10.0 Read and write HDF5 files from Python
├── numpy >=1.7
└── six *
matplotlib 3.3.0 Python plotting package
├── cycler >=0.10
│   └── six *
├── kiwisolver >=1.0.1
├── numpy >=1.15
├── pillow >=6.2.0
├── pyparsing >=2.0.3,<2.0.4 || >2.0.4,<2.1.2 || >2.1.2,<2.1.6 || >2.1.6
└── python-dateutil >=2.1
    └── six >=1.5
numpy 1.19.1 NumPy is the fundamental package for array computing with Python.
scipy 1.5.2 SciPy: Scientific Library for Python
└── numpy >=1.14.5
```
