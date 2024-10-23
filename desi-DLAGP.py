#!/usr/bin/env python

"""
Description of code
"""

from astropy.table import Table, vstack
import numpy as np
from scipy.interpolate import interp1d
import fitsio

import os
import argparse
import time
from concurrent.futures import ProcessPoolExecutor

import dlasearch
import constants

from desiutil.log import log


def parse(options=None):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="""search for DLAs in DESI quasar spectra""",
    )

    parser.add_argument(
        "-q",
        "--qsocat",
        type=str,
        default=None,
        required=True,
        help="path to quasar catalog",
    )

    parser.add_argument(
        "-r",
        "--release",
        type=str,
        default=None,
        required=True,
        help="DESI redux version (e.g. iron)",
    )

    parser.add_argument(
        "-p",
        "--program",
        type=str,
        default="dark",
        required=False,
        help="observing program, default is dark",
    )

    parser.add_argument(
        "-s",
        "--survey",
        type=str,
        default="main",
        required=False,
        help="survey, default is main",
    )

    parser.add_argument(
        "--mocks",
        default=False,
        required=False,
        action="store_true",
        help="is this a mock catalog? Default is False",
    )

    parser.add_argument(
        "--mockdir",
        type=str,
        default=None,
        required=False,
        help="path to mock directory",
    )

    parser.add_argument(
        "--tilebased",
        default=False,
        required=False,
        action="store_true",
        help="use tile based coadds, default is False",
    )

    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="./models/QSO-HIZv1.1_RR.npz",
        required=False,
        help="path to intrinsic flux model",
    )

    parser.add_argument(
        "--varlss",
        type=str,
        default="./lss_variance/jura-var-lss.fits",
        required=False,
        help="path to LSS variance input files",
    )

    parser.add_argument(
        "--balmask",
        default=False,
        required=False,
        action="store_true",
        help="should BALs be masked using AI_CIV? Default is False but recommended setting is True",
    )

    parser.add_argument(
        "-o",
        "--outdir",
        type=str,
        default=None,
        required=True,
        help="output directory for DLA catalog",
    )

    parser.add_argument(
        "-n",
        "--nproc",
        type=int,
        default=64,
        required=False,
        help="number of multiprocressing processes to use, default is 64",
    )

    if options is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(options)

    return args


def main(args=None):

    if isinstance(args, (list, tuple, type(None))):
        args = parse(args)

    # Check is catalog exists
    if not os.path.isfile(args.qsocat):
        log.error(f"{args.qsocat} does not exist")
        exit(1)
    # if catalog is healpix based, we must have program & survey
    if not (args.tilebased) and not (args.mocks):
        log.info(
            f"expecting healpix catalog for redux={args.release}, survey={args.survey}, program={args.program}; confirm this matches the catalog provided!"
        )
    # confirm bal masking choice
    if not (args.balmask):
        log.warning(
            f"BALs will not be masked! The only good reason to do this is if you do not have a BAL catalog, set --balmask to turn on masking."
        )
    # check the model file exits
    if not os.path.isfile(args.model):
        log.error(f"cannot not find flux model file, looking for {args.model}")
        exit(1)
    # check if LSS file exits
    if not os.path.isfile(args.varlss):
        log.error(f"cannot find LSS variance file, looking for {args.varlss}")
        exit(1)
    # check if mock data
    if args.mocks and (args.mockdir is None):
        log.error(f"mocks argument set to true but no mock data path provided")
    elif args.mocks and not (os.path.exists(args.mockdir)):
        log.error(f"{args.mockdir} does not exist")
        exit(1)

    tini = time.time()

    # read in quasar catalog and intrinsic flux model
    if args.mocks:
        catalog = read_mock_catalog(args.qsocat, args.balmask, args.mockdir)
    else:
        catalog = read_catalog(args.qsocat, args.balmask, args.tilebased)

    model = np.load(args.model)
    fluxmodel = dict()
    fluxmodel["PCA_COMP"] = model["PCA_COMP"]
    fluxmodel["PCA_WAVE"] = 10 ** model["LOGLAM"]
    fluxmodel["IGM"] = model["IGM"][0]

    # add lss variance info to dictionary for forest fitting
    fluxmodel = read_varlss(args.varlss, fluxmodel)

    # set up for nested multiprocessing
    nproc_futures = int(os.cpu_count() / args.nproc)

    if not (args.tilebased) and not (args.mocks):

        # TO DO : process in batches to add caching

        datapath = f"/global/cfs/cdirs/desi/spectro/redux/{args.release}/healpix/{args.survey}/{args.program}"

        if nproc_futures == 1:
            results = []
            for hpx in np.unique(catalog["HPXPIXEL"]):
                results.append(
                    dlasearch.dlasearch_hpx(
                        hpx,
                        args.survey,
                        args.program,
                        datapath,
                        catalog[catalog["HPXPIXEL"] == hpx],
                        fluxmodel,
                        args.nproc,
                    )
                )

        if nproc_futures > 1:
            arguments = [
                {
                    "healpix": hpx,
                    "survey": args.survey,
                    "program": args.program,
                    "datapath": datapath,
                    "hpxcat": catalog[catalog["HPXPIXEL"] == hpx],
                    "model": fluxmodel,
                    "nproc": args.nproc,
                }
                for ih, hpx in enumerate(np.unique(catalog["HPXPIXEL"]))
            ]

            with ProcessPoolExecutor(nproc_futures) as pool:
                results = list(pool.map(_dlasearchhpx, arguments))

        results = vstack(results)
        results.meta["EXTNAME"] = "DLACAT"

        # remove extra column from hpx with no detections
        if "col0" in results.columns:
            results.remove_column("col0")

        outfile = os.path.join(
            args.outdir, f"dlacat-{args.release}-{args.survey}-{args.program}.fits"
        )
        if os.path.isfile(outfile):
            log.warning(
                f"dlacat-{args.release}-{args.survey}-{args.program}.fits already exists in {args.outdir}, overwriting"
            )
        results.write(outfile, overwrite=True)

    # place holder until tile-based developed
    elif args.tilebased:
        # TO DO : process in batches to add caching
        log.error("tile based capability does not exist")
        exit(1)

    elif args.mocks:

        # TO DO : process in batches to add caching

        datapath = f"{args.mockdir}/spectra-16"

        speclist = []
        for level1 in os.listdir(f"{datapath}"):
            for level2 in os.listdir(f"{datapath}/{level1}"):
                if os.path.exists(
                    f"{datapath}/{level1}/{level2}/spectra-16-{level2}.fits"
                ):
                    speclist.append(
                        f"{datapath}/{level1}/{level2}/spectra-16-{level2}.fits"
                    )

        if nproc_futures == 1:
            results = []
            for specfile in speclist:
                results.append(
                    dlasearch.dlasearch_mock(specfile, catalog, fluxmodel, args.nproc)
                )

        if nproc_futures > 1:
            arguments = [
                {
                    "specfile": specfile,
                    "catalog": catalog,
                    "model": fluxmodel,
                    "nproc": args.nproc,
                }
                for ih, specfile in enumerate(speclist)
            ]

            with ProcessPoolExecutor(nproc_futures) as pool:
                results = list(pool.map(_dlasearchmock, arguments))

        results = vstack(results)
        results.meta["EXTNAME"] = "DLACAT"

        # remove extra column from spec groups with no detections
        if "col0" in results.columns:
            results.remove_column("col0")

        outfile = os.path.join(args.outdir, f"dlacat-{args.release}-mockcat.fits")
        if os.path.isfile(outfile):
            log.warning(
                f"dlacat-{args.release}-mockcat.fits already exists in {args.outdir}, overwriting"
            )
        results.write(outfile, overwrite=True)

    tfin = time.time()
    total_time = tfin - tini

    print(f"total run time: {np.round(total_time/60,1)} minutes")


def read_catalog(qsocat, balmask, bytile):
    """
    read quasar catalog

    Arguments
    ---------
    qsocat (str) : path to quasar catalog
    balmask (bool) : should BAL attributes from baltools be read in?
    bytile (bool) : catalog is tilebased, default assumption is healpix

    Returns
    -------
    table of relevant attributes for z>2 quasars

    """

    if balmask:
        try:
            # read the following columns from qsocat
            cols = [
                "TARGETID",
                "TARGET_RA",
                "TARGET_DEC",
                "Z",
                "HPXPIXEL",
                "AI_CIV",
                "NCIV_450",
                "VMIN_CIV_450",
                "VMAX_CIV_450",
            ]
            if bytile:
                cols = [
                    "TARGETID",
                    "TARGET_RA",
                    "TARGET_DEC",
                    "Z",
                    "TILEID",
                    "PETAL_LOC",
                    "AI_CIV",
                    "NCIV_450",
                    "VMIN_CIV_450",
                    "VMAX_CIV_450",
                ]
            catalog = Table(fitsio.read(qsocat, ext=1, columns=cols))
        except:
            log.error(f"cannot find {cols} in quasar catalog")
            exit(1)
    else:
        # read the following columns from qsocat
        cols = ["TARGETID", "TARGET_RA", "TARGET_DEC", "Z", "HPXPIXEL"]
        if bytile:
            cols = ["TARGETID", "TARGET_RA", "TARGET_DEC", "Z", "TILEID", "PETAL_LOC"]
        catalog = Table(fitsio.read(qsocat, ext=1, columns=cols))

    log.info(f"Successfully read quasar catalog: {qsocat}")

    # Apply redshift cuts
    zmask = (catalog["Z"] > constants.zmin_qso) & (catalog["Z"] < constants.zmax_qso)
    log.info(f"objects in catalog: {len(catalog)} ")
    log.info(
        f"restricting to {constants.zmin_qso} < z < {constants.zmax_qso}: {np.sum(zmask)} objects remain"
    )

    catalog = catalog[zmask]

    return catalog


def read_mock_catalog(qsocat, balmask, mockpath):
    """
    read quasar catalog

    Arguments
    ---------
    qsocat (str) : path to quasar catalog
    balmask (bool) : should BAL attributes be read in?
    mockpath (str) : path to mock data

    Returns
    -------
    table of relevant attributes for z>2 quasars

    """
    # read the following columns from qsocat
    cols = ["TARGETID", "RA", "DEC", "Z"]
    catalog = Table(fitsio.read(qsocat, ext=1, columns=cols))
    log.info(f"Successfully read mock quasar catalog: {qsocat}")

    # Apply redshift cuts
    zmask = (catalog["Z"] > constants.zmin_qso) & (catalog["Z"] < constants.zmax_qso)
    log.info(f"objects in catalog: {len(catalog)} ")
    log.info(
        f"restricting to {constants.zmin_qso} < z < {constants.zmax_qso}: {np.sum(zmask)} objects remain"
    )

    catalog = catalog[zmask]

    if balmask:
        try:
            # open bal catalog
            balcat = os.path.join(mockpath, "bal_cat.fits")
            cols = ["TARGETID", "AI_CIV", "NCIV_450", "VMIN_CIV_450", "VMAX_CIV_450"]
            balcat = Table(fitsio.read(balcat, ext=1, columns=cols))

            # add columns to catalog
            ai = np.full(len(catalog), 0.0)
            nciv = np.full(len(catalog), 0)
            vmin = np.full((len(catalog), balcat["VMIN_CIV_450"].shape[1]), -1.0)
            vmax = np.full((len(catalog), balcat["VMIN_CIV_450"].shape[1]), -1.0)

            for i, tid in enumerate(catalog["TARGETID"]):
                if np.any(tid == balcat["TARGETID"]):
                    match = balcat[balcat["TARGETID"] == tid]
                    ai[i] = match["AI_CIV"]
                    nciv[i] = match["NCIV_450"]
                    vmin[i] = match["VMIN_CIV_450"]
                    vmax[i] = match["VMAX_CIV_450"]

            catalog.add_columns(
                [ai, nciv, vmin, vmax],
                names=["AI_CIV", "NCIV_450", "VMIN_CIV_450", "VMAX_CIV_450"],
            )

        except:
            log.error(f"cannot find mock bal_cat.fits in {mockpath}")
            exit(1)

    return catalog


def read_varlss(varlss_path, fluxmodel):
    """
    add sigma_lss function to flux model dictionary

    Arguments
    ---------
    varlss_path (str) : path to file containing LSS variance function
    fluxmodel (dict) : dictionary for PCA flux model

    Returns
    -------
    fluxmodel (dict) : flux model dictionary with var_lss entry appended

    """

    log.info(f"reading sigma_lss function from {varlss_path}")

    # read in data
    varlss_lya = Table(fitsio.read(varlss_path, ext="VAR_FUNC_LYA"))
    varlss_lyb = Table(fitsio.read(varlss_path, ext="VAR_FUNC_LYB"))

    # map lambda_obs -> var_lss
    fluxmodel["VAR_FUNC_LYA"] = interp1d(
        10 ** varlss_lya["LOGLAM"],
        varlss_lya["VAR_LSS"],
        bounds_error=False,
        fill_value=0.0,
    )
    fluxmodel["VAR_FUNC_LYB"] = interp1d(
        10 ** varlss_lyb["LOGLAM"],
        varlss_lyb["VAR_LSS"],
        bounds_error=False,
        fill_value=0.0,
    )

    return fluxmodel


# for parallelization over hpx
def _dlasearchhpx(arguments):
    return dlasearch.dlasearch_hpx(**arguments)


# for parallelization over mock spectra files
def _dlasearchmock(arguments):
    return dlasearch.dlasearch_mock(**arguments)


if __name__ == "__main__":
    main()
