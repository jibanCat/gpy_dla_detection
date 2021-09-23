DLA detection pipleine for BOSS quasar spectra, **in Python**
=============================================================

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
The design of this repo assumes users already had the learned GP model and the users want apply the trained model on new quasar spectra.

The parameters are tunable in the `gpy_dla_detection.set_parameters.Parameters` as instance attributes. The provided parameters should
exactly reproduce the catalog in the work of Ho-Bird-Garnett (2020);
however, you may feel free to modify these choices as you see fit.

Downloading the external DLA catalogues and the learned model
----------------------------------------

First we download the raw catalog data:

    # in shell
    cd data/scripts
    ./download_catalogs.sh

The learned model of Ho-Bird-Garnett (2020) is publicly available here:
http://tiny.cc/multidla_catalog_gp_dr12q
The required `.mat` files for this Python repo are:
- The GP model: [learned_qso_model_lyseries_variance_kim_dr9q_minus_concordance.mat](https://drive.google.com/file/d/16n7cDNyXmwoHOw9jFiF5em1z8Q4hQkED/view?usp=sharing) 
- DLA samples for Quasi-Monte Carlo integration: [dla_samples_a03.mat](https://drive.google.com/file/d/1pE5nFkMvXPmSJimr6uXBRUWNYZhp9h00/view?usp=sharing)
- SubDLA samples for Quasi-Monte Carlo integration: [subdla_samples.mat](https://drive.google.com/file/d/1UFdsFAiYNU8QdGph4UY3B86W-ge-112n/view?usp=sharing)
- The SDSS DR12 QSO catalogue, including the `filter_flags` we used to train the GP model: [catalog.mat](https://drive.google.com/file/d/1-DE6NdFhaEcI0bk-l-GiN2DzxoWoLW-L/view?usp=sharing)

The above files should be placed in `data/dr12q/processed/` (this folder would be created after running `download_catalogs.sh`).


Reproducing Ho-Bird-Garnett (2020) predictions
----------------------------------------------

We provide a simple Python script, `run_bayes_select.py` , to reproduce the MATLAB code `multi_dlas/process_qsos_multiple_dlas_meanflux.m` and create a HDF5 catalogue in the end with the posterior probability of having DLAs in a given spectrum.

**(Note: we tried to make the saved variables in the output HDF5 file to be the same as the MATLAB version of the `.mat` catalogue file, though there are still some differences.)**

All the parameters and filepaths are hard-coded in this script to ensure we can exactly reproduce the results of Ho-Bird-Garnett DLA catalogue.

To run this Python script, do:

    # in shell, to make predictions on two QSO spectra.
    python run_bayes_select.py --qso_list spec1.fits spec2.fits --z_qso_list z_qso1 z_qso2

This is assumed the fits files are SDSS DR12Q spectra.
Users can re-design the spectrum reading function by following same input arguments and outputs.
The default fits file reader is `gpy_dla_detection.read_spec.read_spec`.

Users can run the test function `tests.test_selection.test_p_dlas(10)` to get the first 10 DLA predictions from Ho-Bird-Garnett (2020):

    # Download the test spectra first, in bash:
    python -c "from examples.download_spectra import *; download_ho_2020_spectrum(10)"
    # And run the test:
    python -c "from tests.test_selection import *; test_p_dlas(10)"


For developers
--------------

There are some tunable parameters defined in `gpy_dla_detection.set_parameters.Parameters`.
The following parameters will directly affect the DLA predictions:

- `prior_z_qso_increase: float = 30000.0`: DLA existence prior is defined in Garnett (2017) as (prior.z_qsos < (z_qso + prior_z_qso_increase)).
- `num_lines: int = 3`: The number of members of the Lyman series to use in DLA profiles.
- `max_z_cut: float = 3000.0`: The maximum search range for zDLA sampling is `max z_DLA = z_QSO - max_z_cut`.
- `min_z_cut: float = 3000.0`: The minimum search range for zDLA sampling is `min z_DLA = z_Ly∞ + min_z_cut`.
- `num_forest_lines: int = 31`: The number of Lyman-series forest to suppress the mean GP model to mean-flux. 

The `Prior` class is also modifiable. This class is in charge of model priors for different GP models. Users can write a custom Prior class to reflect their prior belief on their own datasets.

Users can also write their own `DLASamples` class to reflect their own priors on DLA parameters, (zDLA, logNHI).

Additional feature: quasar redshift estimation
----

Our GP method could also be used to estimate the quasar redshift. Details could be found in [Leah (2020)](https://arxiv.org/abs/2006.07343). Note that the original MATLAB code is in https://github.com/sbird/gp_qso_redshift. Here we only translated part of the codes without the learning functionality. To use the method, we need to download the trained GP model:

- [learned_zqso_only_model_outdata_full_dr9q_minus_concordance_norm_1176-1256.mat](https://drive.google.com/file/d/1SqAU_BXwKUx8Zr38KTaA_nvuvbw-WPQM/view?usp=sharing)

For how to use the redshift estimation method, please find the notebook in `notebooks/`.

Python requirements
-------------------

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

Known caveats
-------------

- The Quasi-Monte Carlo sampling in `dla_gp.log_model_evidences` has not parallelized yet.
- The DLA profile `voigt.voigt_absorption` is not very efficient.
- The overall speed of this Python code is roughly 10 times slower than the MATLAB counterpart on a 32 core machine.
