"""
Test Bayesian model selection works the same as the MATLAB code.
"""
import numpy as np
import h5py

from run_bayes_select import process_qso

z_qsos = np.array(
    [
        2.30909729,
        2.49794078,
        2.328,
        2.377,
        3.71199346,
        2.163,
        2.559,
        2.35414815,
        2.16476059,
        2.18012738,
        2.52689695,
        2.79486632,
        2.38407612,
        2.87937236,
        2.66187549,
        2.71465468,
        3.00907731,
        2.27381587,
        3.12379503,
        2.49857759,
    ]
)

p_dlas = np.array(
    [
        3.26688972e-06,
        3.83387245e-01,
        1.96516980e-07,
        3.34011390e-08,
        5.95499579e-09,
        6.28749357e-21,
        1.75249316e-08,
        1.08421388e-09,
        2.50499977e-06,
        2.52358271e-02,
        2.69959991e-07,
        3.95859409e-06,
        3.86508691e-06,
        8.19295955e-06,
        2.08142284e-07,
        9.99998690e-01,
        6.67577612e-06,
        1.42478300e-06,
        2.93785844e-04,
        2.80101943e-08,
    ]
)

map_num_dlas = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0])

filenames = [
    "spec-6173-56238-0528.fits",
    "spec-6177-56268-0595.fits",
    "spec-4354-55810-0646.fits",
    "spec-6498-56565-0177.fits",
    "spec-6177-56268-0608.fits",
    "spec-4216-55477-0312.fits",
    "spec-6182-56190-0652.fits",
    "spec-4296-55499-0364.fits",
    "spec-7134-56566-0594.fits",
    "spec-6877-56544-0564.fits",
    "spec-6177-56268-0648.fits",
    "spec-4277-55506-0896.fits",
    "spec-4415-55831-0554.fits",
    "spec-4216-55477-0302.fits",
    "spec-4216-55477-0292.fits",
    "spec-7167-56604-0290.fits",
    "spec-6177-56268-0384.fits",
    "spec-4354-55810-0686.fits",
    "spec-7144-56564-0752.fits",
    "spec-6177-56268-0640.fits",
]


def test_p_dlas(num_quasars: int = 10):

    process_qso(filenames[:num_quasars], z_qsos[:num_quasars])

    with h5py.File("processed_qsos_multi_meanflux.h5", "r") as f:
        print("P(DLA | D)")
        print("----")
        print("Catalogue:", p_dlas[:num_quasars])
        print("Python code:", f["p_dlas"][()])
        assert np.all(np.abs(p_dlas[:num_quasars] - f["p_dlas"][()]) < 1e-2)

        MAP_num_dlas = np.nanargmax(f["model_posteriors"][()], axis=1)
        MAP_num_dlas = MAP_num_dlas - 1
        MAP_num_dlas[MAP_num_dlas < 0] = 0

        print("Num DLAs")
        print("----")
        print("Catalogue:", map_num_dlas[:num_quasars])
        print("Python code:", MAP_num_dlas)
        assert np.all(MAP_num_dlas == map_num_dlas[:num_quasars])
