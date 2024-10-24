#!/usr/bin/env python

"""
Identify DLAs in data sample using template plus viogt fits
"""

import numpy as np
import os
import fitsio
from astropy.table import Table, vstack

import multiprocessing as mp
from functools import partial
import time

# desi packages - TO DO : remove or isolate desi dependencies
import desispec.io
from desispec.interpolation import resample_flux
from desispec.coaddition import coadd_cameras, resample_spectra_lin_or_log
from desiutil.log import log

import constants

# import dlaprofile
from fitwarning import DLAFLAG

import warnings
from scipy.optimize import OptimizeWarning

from run_bayes_select import DLAHolder
from gpy_dla_detection.set_parameters import Parameters

warnings.simplefilter("error", OptimizeWarning)

#### FOR TESTING ONLY ####
# import matplotlib.pyplot as plt
##########################


def dlasearch_hpx(healpix, survey, program, datapath, hpxcat, model_params, executor):
    """
    Find the best fitting DLA profile(s) for spectra in hpx catalog

    Arguments
    ---------
    healpix (int): N64 healpix
    survey (str): e.g., main, sv1, sv2, etc.
    program (str): e.g., bright, dark, etc.
    datapath (str): path to coadd files
    hpxcat (table): collection of spectra to search for DLAs, all belonging to
                    single healpix
    model_params (dict): dictionary of parameters for the DLAHolder model
    executor: Executor for parallel processing

    Returns
    -------
    fitresults (table): attributes of detected DLAs
    """
    t0 = time.time()

    # Read spectra from healpix
    coaddname = f"coadd-{survey}-{program}-{str(healpix)}.fits"
    coadd = os.path.join(datapath, str(healpix // 100), str(healpix), coaddname)

    if os.path.exists(coadd):
        # Reconstruct the Parameters instance from the dictionary
        params = Parameters(**model_params["params_dict"])

        # Reconstruct the DLAHolder instance using the reconstructed Parameters
        model = DLAHolder(
            learned_file=model_params["learned_file"],
            catalog_name=model_params["catalog_name"],
            los_catalog=model_params["los_catalog"],
            dla_catalog=model_params["dla_catalog"],
            dla_samples_file=model_params["dla_samples_file"],
            sub_dla_samples_file=model_params["sub_dla_samples_file"],
            params=params,
            min_z_separation=model_params["min_z_separation"],
            prev_tau_0=model_params["prev_tau_0"],
            prev_beta=model_params["prev_beta"],
            max_dlas=model_params["max_dlas"],
            plot_figures=model_params["plot_figures"],
            max_workers=model_params["max_workers"],
            batch_size=model_params["batch_size"],
        )

        # Process spectra group
        fitresults = process_spectra_group(coadd, hpxcat, model, executor)

    else:
        log.error(f"could not locate coadd file for healpix {healpix}")
        return ()

    t1 = time.time()
    total = np.round(t1 - t0, 2)
    log.info(
        f"Completed processing of {len(hpxcat)} spectra from healpix {healpix} in {total}s"
    )

    return fitresults

def dlasearch_tile(tileid, datapath, tilecat, model, nproc):
    """
    Find the best fitting DLA profile(s) for spectra in hpx catalog

    Arguments
    ---------
    tileid (int) : tile no.
    datapath (str) : path to coadd files
    tilecat (table) : collection of spectra to search for DLAs, all belonging to
                     single tile
    model (dict) : flux model dictionary containing 'PCA_WAVE', 'PCA_COMP', 'IGM',
                    'VAR_FUNC_LYA', and 'VAR_FUNC_LYB' keys
    nproc (int) : number of multiprocessing processes for solve_DLA, default=64

    Returns
    -------
    fitresults (table) : fit attributes for detected DLAs
    """

    t0 = time.time()

    # do tile based search, will need to save tileid in catalog since targetid is not unique to a tile
    # call process_spectra_group, append tileid and petal id to fitresults
    # loop over petal number

    # e.g. for petal in np.unique(tilecat['PETAL_LOC']):
    #           petcat = tilecat[tilecat['PETAL_LOC'] == petal]
    #           coadd = 'path to tile-petal coadd'
    #           #check if pool should be set up
    #           process_spectr_group(coadd, petcat, model, pool)
    #           # apeend tile and petal columns

    t1 = time.time()
    total = t1 - t0
    log.info(
        f"Completed processing of {len(tilecat)} spectra from tile {tileid} in {total}s"
    )

def dlasearch_mock(specfile, catalog, model_params, executor):
    """
    Find the best fitting DLA profile(s) for spectra in the mock catalog.

    Arguments
    ---------
    specfile (str): Path to the mock spectra file.
    catalog (table): Catalog of spectra to search for DLAs.
    model_params (dict): Dictionary containing parameters for the DLAHolder model.
    executor: Executor for parallel processing.

    Returns
    -------
    fitresults (table): Fit attributes for detected DLAs.
    """
    t0 = time.time()

    if os.path.exists(specfile):
        fm = desispec.io.read_fibermap(specfile)
        tidmask = np.in1d(catalog["TARGETID"], fm["TARGETID"])
        catalog = catalog[tidmask]
        if len(catalog) < 1:
            return ()

        # Reconstruct the Parameters instance from the dictionary
        params = Parameters(**model_params["params_dict"])

        # Reconstruct the DLAHolder instance using the reconstructed Parameters
        model = DLAHolder(
            learned_file=model_params["learned_file"],
            catalog_name=model_params["catalog_name"],
            los_catalog=model_params["los_catalog"],
            dla_catalog=model_params["dla_catalog"],
            dla_samples_file=model_params["dla_samples_file"],
            sub_dla_samples_file=model_params["sub_dla_samples_file"],
            params=params,
            min_z_separation=model_params["min_z_separation"],
            prev_tau_0=model_params["prev_tau_0"],
            prev_beta=model_params["prev_beta"],
            max_dlas=model_params["max_dlas"],
            plot_figures=model_params["plot_figures"],
            max_workers=model_params["max_workers"],
            batch_size=model_params["batch_size"],
        )

        # Process spectra group
        fitresults = process_spectra_group(specfile, catalog, model, executor)
    else:
        log.error(f"could not locate coadd file for {specfile}")
        return ()

    t1 = time.time()
    total = np.round(t1 - t0, 2)
    log.info(
        f"Completed processing of {len(catalog)} spectra from {specfile} in {total}s"
    )

    return fitresults

def process_spectra_group(coaddpath, catalog, model: DLAHolder, executor=None):
    """
    pre-process group of spectra in same file and run DLA searching tools

    Arguments
    ---------
    coaddpath (str) : path to file containing spectra
    catalog (table) : collection of spectra in file to search for DLAs
    model (dict) : flux model containing 'PCA_WAVE', 'PCA_COMP', and 'IGM' keys
    executor : shared mp pool

    Returns
    -------
    fitresults (table) : attributes of detected DLAs
    """

    specobj = desispec.io.read_spectra(
        coaddpath,
        targetids=catalog["TARGETID"],
        skip_hdus=["EXP_FIBERMAP", "SCORES", "EXTRA_CATALOG"],
    )
    try:
        specobj = coadd_cameras(specobj)
    except:
        if specobj.resolution_data is not None:
            # resample on linear grid
            wave_min = np.min(specobj.wave["b"])
            wave_max = np.max(specobj.wave["z"])
            specobj = resample_spectra_lin_or_log(
                specobj,
                linear_step=0.8,
                wave_min=wave_min,
                wave_max=wave_max,
                fast=True,
            )
            specobj = coadd_cameras(specobj)
        else:
            # check if mock truth file exists
            truthfile = coaddpath.replace("spectra-16-", "truth-16-")
            if not (os.path.exists(truthfile)):
                log.error(
                    f"cannot process {coaddpath}; no mock truth file or resolution data"
                )
            specobj.resolution_data = {}
            for cam in ["b", "r", "z"]:
                tres = fitsio.read(truthfile, ext=f"{cam}_RESOLUTION")
                tresdata = np.empty(
                    [
                        specobj.flux[cam].shape[0],
                        tres.shape[0],
                        specobj.flux[cam].shape[1],
                    ],
                    dtype=float,
                )
                for i in range(specobj.flux[cam].shape[0]):
                    tresdata[i] = tres
                specobj.resolution_data[cam] = tresdata
            specobj = resample_spectra_lin_or_log(
                specobj,
                linear_step=0.8,
                wave_min=np.min(specobj.wave["b"]),
                wave_max=np.max(specobj.wave["z"]),
                fast=True,
            )

    # for each entry in passed catalog, fit spectrum with intrinsic model + N DLA
    wave = specobj.wave["brz"]

    # lists shared with Allyson's finder
    tidlist, ralist, declist, zqsolist, snrlist, dlaidlist = [], [], [], [], [], []
    zlist, nhilist, zerrlist, nhierrlist, fitwarnlist = (
        [],
        [],
        [],
        [],
        [],
    )
    # lists for GP-DLA results
    pdlalist = []
    pnulllist = []
    logpdlalist = []
    logpnulllist = []
    modelplist = []

    # set up results dict for GPDLA
    num_spectra = len(catalog)
    model.initialize_results(num_spectra=num_spectra)

    # for each entry in passed catalog, fit spectrum with intrinsic model + N DLA
    for entry in range(len(catalog)):

        tid = catalog["TARGETID"][entry]
        try:
            ra = catalog["TARGET_RA"][entry]
            dec = catalog["TARGET_DEC"][entry]
        except:
            # mock catalog
            ra = catalog["RA"][entry]
            dec = catalog["DEC"][entry]
        zqso = catalog["Z"][entry]

        try:
            idx = np.nonzero(specobj.fibermap["TARGETID"] == tid)[0][0]
        except:
            log.error(f"Targetid {tid} NOT FOUND on healpix {catalog["HPXPIXEL"][entry]}")
            continue

        # TODO: Do the GP finder here

        flux = specobj.flux["brz"][idx]
        ivar = specobj.ivar["brz"][idx]
        wave_rf = wave / (1 + zqso)
        pixel_mask = specobj.mask["brz"][idx].astype(np.bool_)

        # apply mask to BAL features, if available
        if "NCIV_450" in catalog.columns:
            nbal = catalog["NCIV_450"][entry]
            bal_locs = []
            for n in range(nbal):
                # Compute velocity ranges
                v_max = -catalog[entry]["VMAX_CIV_450"][n] / constants.c + 1.0
                v_min = -catalog[entry]["VMIN_CIV_450"][n] / constants.c + 1.0

                for line, lam in constants.bal_lines.items():
                    # Mask wavelengths within the velocity ranges
                    mask = np.logical_and(wave_rf > lam * v_max, wave_rf < lam * v_min)
                    if (line == "Lya") or (line == "NV"):
                        rededge = (lam * v_min) * (1 + zqso)
                        blueedge = (lam * v_max) * (1 + zqso)
                        bal_locs.append((rededge, blueedge))

                    # Update pixel mask
                    pixel_mask[mask] = True

                    ivar[mask] = 0

        # Convert inverse variance to variance
        noise_variance = np.zeros(ivar.shape)
        ind = ivar == 0
        noise_variance[:] = np.nan
        noise_variance[~ind] = 1 / ivar[~ind]

        # This part set by Allyson, leave it as it is to match the final catalog filtering
        # only searching to rest frame 900 A (TODO: make this match GPDLA search range)
        fitmask = wave_rf > constants.search_minlam
        # limit our bestfit comparision w/ and w/o DLAs to search region of spectrum
        searchmask = np.ma.masked_inside(wave_rf[fitmask], constants.search_minlam, constants.search_maxlam).mask
        # check if too much of the spectrum is masked
        if np.sum(ivar[fitmask][searchmask] != 0) / np.sum(searchmask) < 0.2:
            log.warning(f"Targetid {tid} skipped - SEARCH WINDOW >80% MASKED")
            continue

        # resample model to observed wave grid
        model.process_qso(
            entry, tid, wavelengths=wave, flux=flux, noise_variance=noise_variance,
            pixel_mask=pixel_mask, z_qso=zqso, executor=executor,
        )

        # fitmodel = np.zeros([model["PCA_COMP"].shape[0], np.sum(fitmask)])
        # for i in range(model["PCA_COMP"].shape[0]):
        #     fitmodel[i] = resample_flux(
        #         wave[fitmask], model["PCA_WAVE"] * (1 + zqso), model["PCA_COMP"][i]
        #     )

        # TODO: update GPDLA's meanflux model to DESI's, which needs to re-train on Y1
        # # apply mean transmission correction for lyman alpha forest
        # for transition, values in constants.Lyman_series[model["IGM"]].items():
        #     lam_range = wave_rf[fitmask] < values["line"]
        #     zpix = wave[fitmask][lam_range] / values["line"] - 1
        #     T = np.exp(-values["A"] * (1 + zpix) ** values["B"])
        #     fitmodel[:, lam_range] *= T

        # # determine var_lss array
        # lyaregion = (wave_rf < constants.Lya_line) & (wave_rf > constants.Lyb_line)
        # lybregion = (
        #     wave_rf < constants.Lyb_line
        # )  # assuming N>3 transition minimal impact
        # varlss = np.zeros(len(ivar))
        # varlss[lyaregion] = varlss_lya[lyaregion]
        # varlss[lybregion] = varlss_lyb[lybregion]

        # TODO: get zerr and nhierr from GPDLA
        # model w/o DLAs

        log_posteriors_no_dla = model.results["log_posteriors_no_dla"][entry]
        p_no_dla = model.results["p_no_dlas"][entry]
        # coeff_null, chi2dof_null = fit_spectrum(
        #     wave[fitmask],
        #     flux[fitmask],
        #     ivar[fitmask],
        #     fitmodel,
        #     varlss[fitmask],
        #     searchmask,
        # )

        zdla = model.results["MAP_z_dlas"][entry]
        zerr = model.results["z_dla_errs"][entry]
        nhi = model.results["MAP_log_nhis"][entry]
        nhierr = model.results["log_nhi_errs"][entry]
        log_posteriors_dla = model.results["log_posteriors_dla"][entry]
        p_dla = model.results["p_dlas"][entry]
        model_posteriors = model.results["model_posteriors"][entry]

        # replace nan with -1 for Allysion's convention
        zdla[np.isnan(zdla)] = -1
        zerr[np.isnan(zerr)] = -1
        nhi[np.isnan(nhi)] = -1
        nhierr[np.isnan(nhierr)] = -1

        # Allyson's code to get fitwarning
        # TODO: replace this specific to GP
        fitwarn = np.full(3,0)

        # check for potential BAL contamination in solution
        # false positive should only come from Lya and NV - all other lines too weak
        if ("nbal" in locals()) & np.any(zdla != -1):
            lam_center_dla = constants.Lya_line * (1 + zdla)
            for window in bal_locs:
                balflag = (lam_center_dla < window[0]) & (lam_center_dla > window[1])
                fitwarn[balflag] |= DLAFLAG.POTENTIAL_BAL

        ndla = np.sum(zdla != -1)
        for n in range(ndla):
            tidlist.append(tid)
            dlaid = str(tid) + "00" + str(n)
            dlaidlist.append(dlaid)
            ralist.append(ra)
            declist.append(dec)
            zqsolist.append(zqso)

            # DLA parameters
            zlist.append(zdla[n])
            zerrlist.append(zerr[n])
            nhilist.append(nhi[n])
            nhierrlist.append(nhierr[n])
            fitwarnlist.append(fitwarn[n])
            # average signal to noise in search region of unmasked pixels
            mask = ivar[fitmask][searchmask] != 0
            snr = np.mean(
                (flux[fitmask][searchmask] * np.sqrt(ivar[fitmask][searchmask]))[mask]
            )
            snrlist.append(snr)

            # GP-DLA results
            pdlalist.append(p_dla)
            pnulllist.append(p_no_dla)
            logpdlalist.append(log_posteriors_dla[n])
            logpnulllist.append(log_posteriors_no_dla)
            modelplist.append(model_posteriors[2+n])


    if len(tidlist) == 0:
        # avoid vstack error for empty tables
        return ()

    fitresults = Table(
        data=(
            tidlist,
            ralist,
            declist,
            zqsolist,
            snrlist,
            dlaidlist,
            zlist,
            zerrlist,
            nhilist,
            nhierrlist,
            fitwarnlist,
            # GP-DLA results
            pdlalist, # posterior probability of DLA model
            pnulllist, # posterior probability of no DLA model
            logpdlalist, # log posterior probability of DLA model
            logpnulllist, # log posterior probability of no DLA model
            modelplist, # model posterior probabilities
        ),
        names=[
            "TARGETID",
            "RA",
            "DEC",
            "Z",
            "SNR",
            "DLAID",
            "Z_DLA",
            "Z_DLA_ERR",
            "NHI",
            "NHI_ERR",
            "DLAFLAG",
            "P_DLA",
            "P_NULL",
            "LOGP_DLA",
            "LOGP_NULL",
            "MODEL_P",            
        ],
        dtype=(
            "int",
            "float64",
            "float64",
            "float64",
            "float64",
            "str",
            "float64",
            "float64",
            "float64",
            "float64",
            "int",
            "float64",
            "float64",
            "float64",
            "float64",
            "float64",            
        ),
    )

    return fitresults
