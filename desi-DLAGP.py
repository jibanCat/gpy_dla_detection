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

# GP-DLA imports
from run_bayes_select import DLAHolder
from gpy_dla_detection.set_parameters import Parameters


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

    ###======== GP-DLA specific arguments =========###

    # Spectra and file-related arguments
    parser.add_argument(
        "--spectra_filename",
        required=True,
        help="DESI spectra FITS filename (e.g., spectra-*.fits).",
    )
    parser.add_argument(
        "--zbest_filename",
        required=True,
        help="DESI redshift catalog filename (zbest-*.fits).",
    )
    parser.add_argument(
        "--learned_file",
        default="data/dr12q/processed/learned_qso_model_lyseries_variance_wmu_boss_dr16q_minus_dr12q_gp_851-1421.mat",
        help="Learned QSO model file path.",
    )
    parser.add_argument(
        "--catalog_name",
        default="data/dr12q/processed/catalog.mat",
        help="Catalog file path.",
    )
    parser.add_argument(
        "--los_catalog",
        default="data/dla_catalogs/dr9q_concordance/processed/los_catalog",
        help="Line-of-sight catalog file path.",
    )
    parser.add_argument(
        "--dla_catalog",
        default="data/dla_catalogs/dr9q_concordance/processed/dla_catalog",
        help="DLA catalog file path.",
    )
    parser.add_argument(
        "--dla_samples_file",
        default="data/dr12q/processed/dla_samples_a03.mat",
        help="DLA samples file path.",
    )
    parser.add_argument(
        "--sub_dla_samples_file",
        default="data/dr12q/processed/subdla_samples.mat",
        help="Sub-DLA samples file path.",
    )

    # DLA-related arguments
    parser.add_argument(
        "--min_z_separation",
        type=float,
        default=3000.0,
        help="Minimum redshift separation for DLA models.",
    )
    parser.add_argument(
        "--prev_tau_0", type=float, default=0.00554, help="Previous value for tau_0."
    )
    parser.add_argument(
        "--prev_beta", type=float, default=3.182, help="Previous value for beta."
    )
    parser.add_argument(
        "--max_dlas", type=int, default=3, help="Maximum number of DLAs to model."
    )
    parser.add_argument(
        "--plot_figures",
        type=int,
        default=0,
        help="Set to 1 to generate plots, 0 otherwise.",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=32,
        help="Number of workers for parallel processing.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=313,
        help="Batch size for parallel model evidence computation.",
    )

    # Parameter-related arguments
    # These are the values used in the trained GP model, don't change them unless you change the trained model
    parser.add_argument(
        "--loading_min_lambda",
        type=float,
        default=800,
        help="Range of rest wavelengths to load (Å).",
    )
    parser.add_argument(
        "--loading_max_lambda",
        type=float,
        default=1550,
        help="Range of rest wavelengths to load (Å).",
    )
    parser.add_argument(
        "--normalization_min_lambda",
        type=float,
        default=1425,
        help="Range of rest wavelengths for flux normalization.",
    )
    parser.add_argument(
        "--normalization_max_lambda",
        type=float,
        default=1475,
        help="Range of rest wavelengths for flux normalization.",
    )
    parser.add_argument(
        "--min_lambda",
        type=float,
        default=850.75,
        help="Range of rest wavelengths to model (Å).",
    )
    parser.add_argument(
        "--max_lambda",
        type=float,
        default=1420.75,
        help="Range of rest wavelengths to model (Å).",
    )
    parser.add_argument(
        "--dlambda", type=float, default=0.25, help="Separation of wavelength grid (Å)."
    )
    parser.add_argument(
        "--k", type=int, default=20, help="Rank of non-diagonal contribution."
    )
    parser.add_argument(
        "--max_noise_variance",
        type=float,
        default=9,
        help="Maximum pixel noise allowed during model training.",
    )

    # process range
    parser.add_argument(
        "--hpx_start",
        type=int,
        default=0,
        help="start healpix pixel",
    )
    parser.add_argument(
        "--hpx_end",
        type=int,
        default=1,
        help="end healpix pixel",
    )
    parser.add_argument(
        "--level2_start",
        type=int,
        default=0,
        help="start level2 folder",
    )
    parser.add_argument(
        "--level2_end",
        type=int,
        default=1,
        help="end level2 folder",
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
        log.info(f"running in between healpix pixels {args.hpx_start} - {args.hpx_end}")

    # confirm bal masking choice
    if not (args.balmask):
        log.warning(
            f"BALs will not be masked! The only good reason to do this is if you do not have a BAL catalog, set --balmask to turn on masking."
        )

    # check if mock data
    if args.mocks and (args.mockdir is None):
        log.error(f"mocks argument set to true but no mock data path provided")
    elif args.mocks and not (os.path.exists(args.mockdir)):
        log.error(f"{args.mockdir} does not exist")
        exit(1)

    tini = time.time()

    # read in quasar catalog and intrinsic flux model
    # TODO: Get the total number of spectra
    # For real data, count the number of healpix pixels
    # For mock data, count the number of level1 folders
    if args.mocks:
        #  Mock section: count the total number of spectra.fits files
        datapath = f"{args.mockdir}/spectra-16"
        # list of .fits files, each ~ 800 spectra
        speclist = []
        all_level2 = []
        for level1 in os.listdir(f"{datapath}"):
            for level2 in os.listdir(f"{datapath}/{level1}"):
                if os.path.exists(
                    f"{datapath}/{level1}/{level2}/spectra-16-{level2}.fits"
                ):
                    speclist.append(
                        f"{datapath}/{level1}/{level2}/spectra-16-{level2}.fits"
                    )
                    all_level2.append(level2)

        # reorder speclist by level2
        argsortind = np.argsort(list(map(int, all_level2)))
        speclist = np.array(speclist)[
            argsortind
        ]  # these would be by order from 0 - 3071
        all_level2 = np.array(list(map(int, all_level2)))[argsortind]

        # running in between mock level2 folders: level2_start - level2_end
        log.info(
            "running in between mock level2 folders {} - {}; Total: {}".format(
                args.level2_start, args.level2_end, all_level2[-1]
            )
        )
        ind = (all_level2 >= args.level2_start) & (all_level2 < args.level2_end)

        catalog = read_mock_catalog(args.qsocat, args.balmask, args.mockdir)
        # num_spectra  # TODO: count from the number of level2 folders
    else:
        # running in between healpix pixels: hpx_start - hpx_end
        catalog = read_catalog(args.qsocat, args.balmask, args.tilebased)

        all_hpxs = catalog["HPXPIXEL"]
        log.info(
            "running in between healpix pixels {} - {}; Total {}".format(
                args.hpx_start, args.hpx_end, all_hpxs[-1]
            )
        )
        ind = (all_hpxs >= args.hpx_start) & (all_hpxs < args.hpx_end)

        # num_spectra = np.sum(ind)  # count from the number of healpix pixels

    # TODO: Set up the GP-DLA model
    # Initialize Parameters object with user inputs
    params = Parameters(
        loading_min_lambda=args.loading_min_lambda,
        loading_max_lambda=args.loading_max_lambda,
        normalization_min_lambda=args.normalization_min_lambda,
        normalization_max_lambda=args.normalization_max_lambda,
        min_lambda=args.min_lambda,
        max_lambda=args.max_lambda,
        dlambda=args.dlambda,
        k=args.k,
        max_noise_variance=args.max_noise_variance,
    )
    # This is the GP-DLA processor which can be reused for multiple spectra
    model = DLAHolder(
        learned_file=args.learned_file,
        catalog_name=args.catalog_name,
        los_catalog=args.los_catalog,
        dla_catalog=args.dla_catalog,
        dla_samples_file=args.dla_samples_file,
        sub_dla_samples_file=args.sub_dla_samples_file,
        params=params,
        min_z_separation=args.min_z_separation,
        prev_tau_0=args.prev_tau_0,
        prev_beta=args.prev_beta,
        max_dlas=args.max_dlas,
        plot_figures=bool(args.plot_figures),
        max_workers=args.max_workers,
        batch_size=args.batch_size,
    )

    # set up for nested multiprocessing
    nproc_futures = int(os.cpu_count() / args.nproc)

    if not (args.tilebased) and not (args.mocks):

        # TO DO : process in batches to add caching

        datapath = f"/global/cfs/cdirs/desi/spectro/redux/{args.release}/healpix/{args.survey}/{args.program}"

        # Allyson recommend: 1 process per healpix
        # So here if I want to run in chunks, still look through the healpix pixels
        this_hpxs = all_hpxs[ind]

        if nproc_futures == 1:
            results = []
            for hpx in np.unique(this_hpxs):
                results.append(
                    dlasearch.dlasearch_hpx(
                        hpx,
                        args.survey,
                        args.program,
                        datapath,
                        catalog[catalog["HPXPIXEL"] == hpx],
                        model,
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
                    "model": model,
                    "nproc": args.nproc,
                }
                for ih, hpx in enumerate(np.unique(this_hpxs))
            ]

            with ProcessPoolExecutor(nproc_futures) as pool:
                results = list(pool.map(_dlasearchhpx, arguments))

        results = vstack(results)
        results.meta["EXTNAME"] = "DLACAT"

        # remove extra column from hpx with no detections
        if "col0" in results.columns:
            results.remove_column("col0")

        # filename for output include release, survey, program and healpix range
        outfile = os.path.join(
            args.outdir,
            f"dlacat-{args.release}-{args.survey}-{args.program}-hpx-{args.hpx_start}-{args.hpx_end}.fits",
        )
        if os.path.isfile(outfile):
            log.warning(
                f"dlacat-{args.release}-{args.survey}-{args.program}-hpx-{args.hpx_start}-{args.hpx_end}.fits already exists in {args.outdir}, overwriting"
            )
        results.write(outfile, overwrite=True)

    # place holder until tile-based developed
    elif args.tilebased:
        # TO DO : process in batches to add caching
        log.error("tile based capability does not exist")
        exit(1)

    elif args.mocks:

        # TO DO : process in batches to add caching
        if nproc_futures == 1:
            results = []
            for specfile in speclist:
                results.append(
                    dlasearch.dlasearch_mock(specfile, catalog, model, args.nproc)
                )

        if nproc_futures > 1:
            arguments = [
                {
                    "specfile": specfile,
                    "catalog": catalog,
                    "model": model,
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

        # filename for output include release, survey, program and folder range
        outfile = os.path.join(
            args.outdir,
            f"dlacat-{args.release}-mockcat-{args.level2_start}-{args.level2_end}.fits",
        )
        if os.path.isfile(outfile):
            log.warning(
                f"dlacat-{args.release}-mockcat-{args.level2_start}-{args.level2_end}.fits already exists in {args.outdir}, overwriting"
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


# for parallelization over hpx
def _dlasearchhpx(arguments):
    return dlasearch.dlasearch_hpx(**arguments)


# for parallelization over mock spectra files
def _dlasearchmock(arguments):
    return dlasearch.dlasearch_mock(**arguments)


if __name__ == "__main__":
    main()
