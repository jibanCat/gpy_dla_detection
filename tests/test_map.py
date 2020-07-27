"""
Test the maximum a posteriori estimates
"""
import time

import numpy as np
from .test_model import prepare_dla_model


def test_DLA_MAP():
    # test 1
    dla_gp = prepare_dla_model(plate=5309, mjd=55929, fiber_id=362, z_qso=3.166)

    tic = time.time()

    max_dlas = 4
    log_likelihoods_dla = dla_gp.log_model_evidences(max_dlas)

    toc = time.time()
    # very time consuming: ~ 4 mins for a single spectrum without parallelized.
    print("spent {} mins; {} seconds".format((toc - tic) // 60, (toc - tic) % 60))

    catalog_MAP_log_nhis = np.array(
        [
            [22.28420156, np.nan, np.nan, np.nan],
            [20.63417494, 22.28420156, np.nan, np.nan],
            [20.60601572, 22.28420156, 20.63417494, np.nan],
            [20.12721363, 22.28420156, 20.63417494, 20.36967609],
        ]
    )

    catalog_MAP_z_dlas = np.array(
        [
            [3.03175723, np.nan, np.nan, np.nan],
            [2.52182382, 3.03175723, np.nan, np.nan],
            [2.39393537, 3.03175723, 2.52182382, np.nan],
            [2.94786938, 3.03175723, 2.52182382, 2.38944805],
        ]
    )

    mapind = np.nanargmax(log_likelihoods_dla)

    MAP_z_dla, MAP_log_nhi = dla_gp.maximum_a_posteriori()

    nanind = np.isnan(catalog_MAP_z_dlas[mapind])
    assert np.all(
        np.abs(MAP_z_dla[mapind][~nanind] - catalog_MAP_z_dlas[mapind][~nanind]) < 1e-1
    )
    assert np.all(
        np.abs(MAP_log_nhi[mapind][~nanind] - catalog_MAP_log_nhis[mapind][~nanind])
        < 1e-1
    )

    # test 2
    dla_gp = prepare_dla_model(plate=3816, mjd=55272, fiber_id=76, z_qso=3.68457627)

    tic = time.time()

    max_dlas = 4
    log_likelihoods_dla = dla_gp.log_model_evidences(max_dlas)

    toc = time.time()
    # very time consuming: ~ 4 mins for a single spectrum without parallelized.
    print("spent {} mins; {} seconds".format((toc - tic) // 60, (toc - tic) % 60))

    catalog_MAP_log_nhis = np.array(
        [
            [21.05371292, np.nan, np.nan, np.nan],
            [20.0073665, 20.94707037, np.nan, np.nan],
            [20.00838815, 20.94707037, 20.0073665, np.nan],
            [20.20539934, 20.94707037, 20.0073665, 20.0134955],
        ]
    )

    catalog_MAP_z_dlas = np.array(
        [
            [3.42520566, np.nan, np.nan, np.nan],
            [2.69422714, 3.42710284, np.nan, np.nan],
            [3.41452521, 3.42710284, 2.69422714, np.nan],
            [3.43813463, 3.42710284, 2.69422714, 3.41262802],
        ]
    )

    mapind = np.nanargmax(log_likelihoods_dla)

    MAP_z_dla, MAP_log_nhi = dla_gp.maximum_a_posteriori()

    nanind = np.isnan(catalog_MAP_z_dlas[mapind])

    assert np.all(
        np.abs(MAP_z_dla[mapind][~nanind] - catalog_MAP_z_dlas[mapind][~nanind]) < 1e-1
    )
    assert np.all(
        np.abs(MAP_log_nhi[mapind][~nanind] - catalog_MAP_log_nhis[mapind][~nanind])
        < 1e-1
    )
