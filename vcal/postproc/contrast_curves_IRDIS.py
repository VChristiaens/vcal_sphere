#! /usr/bin/env python
# coding: utf-8
"""Module with the postprocessing routine for SPHERE/IRDIS data."""

__author__ = "V. Christiaens"
__all__ = ["contrast_curves_IRDIS"]

# *Version 1 (2026/09/12)* -- this version

######################### Importations and definitions ########################

import json
import os
import pdb
from multiprocessing import cpu_count
from os.path import isfile, isdir, join

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame as DF
from scipy import interpolate

from vip_hci.fits import open_fits, write_fits
from vip_hci.metrics import snrmap, contrast_curve
from vip_hci.preproc import (
    cube_shift,
    frame_shift,
    cube_crop_frames
)
from vip_hci.var import (
    mask_circle,
    cube_filter_highpass,
    frame_center,
    frame_filter_lowpass,
    get_annulus_segments,
)
from vip_hci.psfsub import (
    median_sub,
    pca,
    pca_annular,
    nmf,
    MEDIAN_SUB_Params,
    PCA_Params,
    PCA_ANNULAR_Params,
    NMF_Params,
)
from vip_hci.psfsub.utils_pca import pca_annulus
from vip_hci.fm import (
    normalize_psf,
    cube_inject_companions,
    cube_planet_free,
    find_nearest,
)

from vip_hci.config import time_ini, timing

# specific to applefy contrast curves
from applefy.detections.contrast import Contrast
from applefy.utils import flux_ratio2mag, mag2flux_ratio
from applefy.wrappers.vip import MultiComponentPCAvip
from applefy.utils.photometry import AperturePhotometryMode
from applefy.statistics import TTest, gaussian_sigma_2_fpf
import seaborn as sns


matplotlib.use("Agg")

# define path where to look for static calibration files
local_path = True
if "VCAL_PATH" in os.environ:
    vcal_path = os.environ["VCAL_PATH"]
else:
    from vcal import __path__ as vcal_path

    vcal_path = vcal_path[0]

    if not isdir(join(vcal_path, "Static/")):
        from astropy.utils.data import download_file

        local_path = False  # triggers downloads from GitHub repo
        url_d = "https://github.com/VChristiaens/vcal_sphere/raw/main/Static/"


def contrast_curves_IRDIS(
    params_cc_name="VCAL_params_cc_IRDIS.json",
    params_preproc_name="VCAL_params_preproc_IRDIS.json",
    params_calib_name="VCAL_params_calib.json",
    planet_parameter=None,
) -> None:
    """
    Calculate Applefy contrast curves for SPHERE/IRDIS data using parameters\
    provided in json file.

    *Note: standard contrast curves can also be calculated with postproc_IRDIS,
    setting the following parameters in the json file: do_pca_sann=False,
    fake_planet=True
    
    Input:
    ******
    params_cc_name: str, opt
        Full path + name of the json file containing contrast curve parameters.
    params_preproc_name: str, opt
        Full path + name of the json file containing preproc parameters.
    params_calib_name: str, opt
        Full path + name of the json file containing calibration parameters.
    planet_parameter: None or numpy 2D ndarray
        If not None, should be a n_planets x 3-element numpy array containing:
            - radial separation (in px),
            - azimuth (in deg, from x=0 axis),
            - and flux (in ADUs);
        for each companion candidate in the dataset. I.e. for 1 companion
        candidate, dimensions of array should be (1, 3); NOT (3,).


    Output:
    *******
    None. All products are written as fits, csv and pdf files.

    """
    plt.style.use("default")
    with open(params_cc_name, "r") as read_file_params_postproc:
        params_postproc = json.load(read_file_params_postproc)
    with open(params_preproc_name, "r") as read_file_params_preproc:
        params_preproc = json.load(read_file_params_preproc)
    with open(params_calib_name, "r") as read_file_params_calib:
        params_calib = json.load(read_file_params_calib)

    if local_path:
        f1 = join(vcal_path, "Static/sphere_filt_spec.json")
        f2 = join(vcal_path, "Static/SPHERE_IRDIS_ALC_transmission_px.fits")
    else:
        try:
            f1 = download_file(
                join(url_d, "sphere_filt_spec.json"), cache=True
            )
            f2 = download_file(
                join(url_d, "SPHERE_IRDIS_ALC_transmission_px.fits"),
                cache=True,
            )
        except:
            msg = "1. VCAL_PATH environment variable is not defined."
            msg += "2. No internet connection to download calibration files."
            msg += "To run the pipeline either solve 1 or 2."
            raise IOError(msg)

    with open(f1) as filt_spec_file:
        filt_spec = json.load(filt_spec_file)[
            params_calib["comb_iflt"]
        ]  # Get infos of current filters combination

    # from calib
    path = params_calib["path"]
    plsc_ori = params_preproc["plsc"]
    filters = filt_spec["filters"]
    if len(filters) == 1:
        filters = [filters[0] + "_l", filters[0] + "_r"]  # CI
        plsc_ori *= 2  # plate scale is the same for each detector half
    path_irdis = path + "IRDIS_reduction/"

    # from preproc
    coro = params_preproc["coro"]
    bin_fac = params_preproc.get("bin_fac", 1)
    distort_corr = params_preproc["distort_corr"]
    if distort_corr:
        distort_corr_labs = ["_DistCorr"]
    else:
        distort_corr_labs = [""]
    final_crop_sz = params_preproc.get("final_crop_sz", 101)
    final_cubename = params_preproc.get("final_cubename", "final_cube")
    final_anglename = params_preproc.get(
        "final_anglename", "final_derot_angles"
    )
    final_psfname = params_preproc.get("final_psfname", "final_psf_med")
    final_fluxname = params_preproc.get("final_fluxname", "final_flux")
    final_fwhmname = params_preproc.get("final_fwhmname", "final_fwhm")
    final_scalefacname = params_preproc.get("final_scalefacname", None)
    if final_scalefacname is None:
        final_scalefacname = params_postproc.get("final_scalefacname", None)
    ## norm output names
    if final_cubename.endswith(".fits"):
        final_cubename = final_cubename[:-5]
    if final_anglename.endswith(".fits"):
        final_anglename = final_anglename[:-5]
    if final_psfname.endswith(".fits"):
        final_psfname = final_psfname[:-5]
    if final_fluxname.endswith(".fits"):
        final_fluxname = final_fluxname[:-5]
    if final_fwhmname.endswith(".fits"):
        final_fwhmname = final_fwhmname[:-5]
    if final_scalefacname is not None:
        if final_scalefacname.endswith(".fits"):
            final_scalefacname = final_scalefacname[:-5]
    label_test_pre = params_preproc.get("label_test", "")
    outpath_2 = path_irdis + "2_preproc_vip{}/".format(label_test_pre)

    # from postproc param file
    sourcename = params_postproc.get("sourcename", "")  # can have spaces
    details = params_postproc.get("details", "")
    label_test = params_postproc.get("label_test", "")
    do_no_crop = params_postproc.get("do_no_crop", False)  # whether to also run post-processing on full frames (slower)

    source = sourcename.replace(" ", "")  # same without space

    ## Options
    verbose = params_postproc.get(
        "verbose", 0
    )  # whether to print(more information during the reduction
    debug = params_postproc.get("debug", False)
    nproc = params_postproc.get(
        "nproc", int(cpu_count() / 2)
    )  # number of processors to use - default set to cpu_count()/2 for efficiency
    overwrite_ADI = params_postproc.get(
        "overwrite_ADI", 1
    )  # whether to overwrite median-ADI results
    overwrite_pp = params_postproc.get(
        "overwrite_pp", 1
    )  # whether to overwrite PCA-ADI results

    ## TO DO?
    do_adi = params_postproc.get("do_adi", 1)
    do_pca_full = params_postproc.get("do_pca_full", 1)
    do_pca_ann = params_postproc.get("do_pca_ann", 1)
    ## Planet?
    planet = params_postproc.get("planet", 0)  # is there a companion?
    planet_pos_crop = params_postproc.get(
        "planet_pos_crop", None
    )  # If so, where is it (or where is it expected)?   (x, y) in cropped frames
    planet_pos_full = params_postproc.get(
        "planet_pos_full", None
    )  # (x, y) in full frames
    if planet_pos_crop and planet_pos_full is not None:
        source_xy = [
            tuple(planet_pos_crop),
            tuple(planet_pos_full),
        ]  # to pass to full frame PCA to trigger a rotation threshold
    else:
        source_xy = [None, None]
    subtract_planet = params_postproc.get(
        "subtract_planet", 0
    )  # this should only be used as a second iteration, after negfc on the companion has enabled to determine its parameters

    ## Inject fake companions? If True => will compute contrast curves
    fcp_pos_r_crop = np.array(
        params_postproc.get("fcp_pos_r_crop", [0.5])
    )  # list of r in arcsec where the fcps are injected in the cropped cube
    fcp_pos_r_full = np.array(
        params_postproc.get("fcp_pos_r_full", [0.5])
    )  # same for the uncropped cube
    injection_fac = params_postproc.get(
        "injection_fac", 1.0
    )  # scaling factor for the injection of fcps with respect to first 5-sigma contrast estimate (e.g. 3/5. to inject at 3 sigma instead of 5 sigma)
    fc_snr = params_postproc.get(
        "fc_snr", 100
    )  # snr of the injected fcp in contrast_curve to compute throughput
    nspi = params_postproc.get(
        "nbr", 9
    )  # number of PAs where the contrast curve is computed
    wedge = tuple(
        params_postproc.get("wedge", [0, 360])
    )  # in which range of PA should the contrast curve be computed

    ## Post-processing
    high_pass_filter_list = params_postproc.get(
        "high_pass_filter_list", [0]
    )  # float or None # If not None nor 0, this is the size of the median filter (in FWHM) used to filter out small spatial frequencies - might be useful to remove large scale noise variations in the image, but risky if extended disk signal is present.
    mask_IWA_px = params_postproc.get(
        "mask_IWA", 5
    )  # just show pca images beyond the provided mask radius (in pixels)
    do_conv = params_postproc.get(
        "do_conv", 0
    )  # whether to smooth final images
    do_snr_map = params_postproc.get(
        "do_snr_map", [0, 0, 0]
    )  # to plot the snr_map (warning: computer intensive); useful only when point-like features are seen in the image
    if not isinstance(do_snr_map, list):
        do_snr_map = [do_snr_map] * 3
    do_stim_map = params_postproc.get(
        "do_stim_map", [0, 0, 0]
    )  # to plot the snr_map (warning: computer intensive); useful only when point-like features are seen in the image
    if not isinstance(do_stim_map, list):
        do_stim_map = [do_stim_map] * 3
    do_color_map = params_postproc.get("do_color_map", [0, 0, 0])
    if not isinstance(do_color_map, list):
        do_color_map = [do_color_map] * 3
    ###RDI
    strategy = params_postproc.get("strategy", "ADI")
    ref_cube_name = params_postproc.get("ref_cube_name", None)
    scaling = params_postproc.get("scaling", None)  # for RDI
    mask_PCA = params_postproc.get("mask_PCA", None)
    ##DBI
    ### PCA options
    delta_rot = params_postproc.get(
        "delta_rot", (1, 3)
    )  # float or tuple expressed in FWHM # Threshold in azimuthal motion to keep frames in the PCA library created by PCA-annular. If a tuple, corresponds to the threshold for the innermost and outermost annuli, respectively.
    #### how is SVD done for PCA:
    svd_mode = params_postproc.get("svd_mode", "lapack")
    #### number of principal components
    n_firstguess_pcs = params_postproc.get(
        "n_firstguess_pcs", 5
    )  # number of pcs geometrically explored between 1 and N_frames/2 

    # contrast curves
    n_br = params_postproc.get("n_br", 6)

    #### min/max number of frames to create PCA library
    min_fr = params_postproc.get("min_fr", 10)
    max_fr = params_postproc.get("max_fr", 200)

    # imlib / interpolation
    imlib = params_postproc.get("imlib", "vip-fft")
    interpolation = params_postproc.get("interpolation", "lanczos4")

    ################ LOADING FILES AND FORMATTING  - don't change #################

    ## Formatting paths
    outpath_4 = path_irdis + "4_contrast_curves_bin{:.0f}" + label_test + "/"
    outpath_5 = outpath_4 + "{}_{}/"
    checkpoint_dir = join(outpath_4,"cc/")

    ref_cube = None
    label_stg = strategy
    if isinstance(ref_cube_name, str):
        if scaling is not None:
            label_stg += "_" + scaling
        if mask_PCA is not None:
            if np.isscalar(mask_PCA):
                label_stg += "_mask{:.1f}".format(mask_PCA)
                mask_PCA = (int(mask_PCA / np.median(plsc_ori)),)
            elif len(mask_PCA) != 2:
                msg = "If mask_PCA is set to a tuple, it can only have 2 "
                msg += " elements."
                raise TypeError(msg)
            else:
                label_stg += "_mask{:.1f}-{:.1f}".format(
                    mask_PCA[0], mask_PCA[1]
                )
                mask_PCA = (
                    int(mask_PCA[0] / np.median(plsc_ori)),
                    int(mask_PCA[1] / np.median(plsc_ori)),
                )

    if coro:
        transmission_name = f2
        transmission = open_fits(transmission_name)
        # transmission = (transmission[1], transmission[0])
    else:
        transmission = None

    if isinstance(delta_rot, list):
        delta_rot = tuple(delta_rot)
        delta_rot_tmp = delta_rot[0]
    else:
        delta_rot_tmp = delta_rot
    label_test = "_thr{:.0f}_mask{:.1f}_maxfr{:.0f}".format(
        delta_rot_tmp, mask_IWA_px, max_fr
    )

    if isinstance(svd_mode, str):
        svd_mode_all = [svd_mode, svd_mode]
    elif isinstance(svd_mode, list):
        svd_mode_all = svd_mode
    
    test_pcs_full = params_postproc.get("pcs_full", [1, 21, 1])
    test_pcs_ann = params_postproc.get("pcs_ann", [1, 11, 1])
    if len(test_pcs_full) == 3:
        test_pcs_full = list(
            range(test_pcs_full[0], test_pcs_full[1], test_pcs_full[2])
            )
    if len(test_pcs_ann) == 3:
        test_pcs_ann = list(
            range(test_pcs_ann[0], test_pcs_ann[1], test_pcs_ann[2])
        )

    # fr_sel_str = "-".join(frame_selection)
    ## Default is post-process twice: 1) crop, 2) no crop
    # crop_list = [final_crop_sz_px,0] # !!! ALREADY CROPPED VERSION OPENED BELOW ! IMPORTANT: always put the case with cropping first (in case you wish to crop)
    if isinstance(final_crop_sz, list):
        ncrop = len(final_crop_sz)
        for i in range(ncrop):
            if final_crop_sz[ncrop - 1 - i] % 2:
                final_crop_sz = final_crop_sz[ncrop - 1 - i]
                break
    final_crop_as = final_crop_sz * np.median(plsc_ori)
    crop_lab_list = ["crop_{:.1f}as".format(final_crop_as)]
    if do_no_crop:
        crop_lab_list.append("no_crop")

    # FULL
    test_pcs_full_crop = test_pcs_full  # [1]
    # for ii,jj in enumerate(range(1,11,1)):
    #    test_pcs_full_crop.append(test_pcs_full_crop[ii]+jj)
    print("test pcs full (crop): ", test_pcs_full_crop)
    test_pcs_full_nocrop = test_pcs_full  # [5] # randsvd
    # for ii,jj in enumerate(range(1,11,1)):
    #    test_pcs_full_nocrop.append(test_pcs_full_nocrop[ii]+jj)
    print("test pcs full (no crop): ", test_pcs_full_nocrop)
    test_pcs_full_all = [test_pcs_full_crop, test_pcs_full_nocrop]
    # ANN
    test_pcs_ann_crop = test_pcs_ann  # [2] # randsvd
    # for ii,jj in enumerate(range(1,11,1)):
    #    test_pcs_ann_crop.append(test_pcs_ann_crop[ii]+jj)
    print("test pcs ann (crop): ", test_pcs_ann_crop)
    test_pcs_ann_nocrop = None  # does not matter, we will never do pca-ann on non-cropped cubes !!!
    test_pcs_ann_all = [test_pcs_ann_crop, test_pcs_ann_nocrop]

    th0 = wedge[0]  # trigonometric angle for the first fcp to be injected
    all_markers = [
        "ro",
        "yo",
        "bo",
        "go",
        "ko",
        "co",
        "mo",
    ] * nspi  # for plotting the snr of the fcps (should contain at least as many elements as fcps)

    ########################## START POST-PROCESSING ##############################

    # DEFINE FUTURE PANDAS DATAFRAMES
    # (3 figures of merit: contrast at 0.15'', snr of the companion (~0.27), contrast at 0.40'')
    n_tests = len(crop_lab_list) * len(filters)
    if final_scalefacname is not None:
        scale_list = open_fits(outpath_2 + final_scalefacname)
    else:
        scale_list = None
    # LOOP ON ALL PARAMETERS
    counter = 0

    for distort_corr_lab in distort_corr_labs:
        # for bb, bin_fac in enumerate(bin_fac_list):
        bin_fac_list = [bin_fac]  # dirty hack to avoid re-writing it all
        if not isdir(outpath_4.format(bin_fac)):
            os.makedirs(outpath_4.format(bin_fac))
        if not isdir(checkpoint_dir.format(bin_fac)):
            os.makedirs(checkpoint_dir.format(bin_fac))
        
        for cc, crop_lab in enumerate(
            crop_lab_list
        ):  # cropped frames, then non-cropped frames
            print(
                "*** TESTING binning x{:.0f} - {} (test {}/{})***".format(
                    bin_fac, crop_lab_list[cc], counter + 1, n_tests
                )
            )
            for high_pass_filter in high_pass_filter_list:
                # 1. regular ADI/RDI per channel for all crops, regardless of scale_list provided
                #if scale_list is None:
                for ff, filt in enumerate(filters):
                    if not isdir(
                        outpath_5.format(bin_fac, filt, crop_lab_list[cc])
                    ):
                        os.makedirs(outpath_5.format(
                                bin_fac, filt, crop_lab_list[cc]
                            )
                        )
                    # fwhm = fwhm_ori[ff]
                    plsc = plsc_ori[ff]

                    if cc == 0 or not isfile(
                        outpath_2
                        + final_cubename
                        + "_full{}.fits".format(filt)
                    ):
                        ADI_cube = open_fits(
                            outpath_2
                            + final_cubename
                            + "{}.fits".format(filt)
                        )
                        lab_full = ""
                    else:
                        ADI_cube = open_fits(
                            outpath_2
                            + final_cubename
                            + "_full{}.fits".format(filt)
                        )
                        lab_full = "_full"
                    if isinstance(ref_cube_name, str):
                        ref_cube = open_fits(ref_cube_name.format(filt))

                    derot_angles = open_fits(
                        outpath_2
                        + final_anglename
                        + "{}.fits".format(filt)
                    )
                    #                    if derot_name == "rotnth":
                    #                        derot_angles*=-1
                    psf = open_fits(
                        outpath_2 + final_psfname + "{}.fits".format(filt)
                    )  # this has all the unsat psf frames

                    # crop ADI_cube if even
                    if not ADI_cube.shape[-1] % 2:
                        ADI_cube = ADI_cube[:, 1:, 1:]
                        ADI_cube = cube_shift(
                            ADI_cube, 0.5, 0.5, nproc=nproc, imlib=imlib
                        )
                    # crop psf if even
                    # psf = np.median(psf_cube, axis=0)
                    # psf=psf_cube[0]
                    # write_fits(outpath_2+psf_name+'_'+filt,psf_cube)
                    if not psf.shape[-1] % 2:
                        psf = psf[1:, 1:]
                        psf = frame_shift(psf, 0.5, 0.5)

                    # measure flux and fwhm
                    psfn, starphot, fwhm = normalize_psf(
                        psf,
                        fwhm="fit",
                        size=19,
                        full_output=True,
                        force_odd=True,
                        mask_core=6,
                    )
                    # mask_IWA_px = int(mask_IWA*fwhm)
                    if starphot < 0 or fwhm < 3:
                        print("There is a problem with the unsat psf")
                        pdb.set_trace()

                    if high_pass_filter:
                        label_filt = label_test+"_hpf"
                        # MODIFY THE IF BELOW
                        if isfile(
                            outpath_4.format(bin_fac)
                            + final_cubename
                            + "{}{}{}.fits".format(
                                lab_full, filt, label_filt
                            )
                        ):
                            ADI_cube = open_fits(
                                outpath_4.format(bin_fac)
                                + final_cubename
                                + "{}{}{}.fits".format(
                                    lab_full, filt, label_filt
                                )
                            )
                        else:
                            ADI_cube = cube_filter_highpass(
                                ADI_cube,
                                "median-subt",
                                median_size=int(high_pass_filter * fwhm),
                            )
                            write_fits(
                                outpath_4.format(bin_fac)
                                + final_cubename
                                + "{}{}{}.fits".format(
                                    lab_full, filt, label_filt
                                ),
                                ADI_cube,
                            )
                            # vip.fits.append_extension(outpath_4.format(bin_fac)+"final_cube_{}{}.fits".format(filt,label_filt), derot_angles)

                    else:
                        label_filt = label_test

                    ## SUBTRACT COMPANION, IF ANY
                    if subtract_planet:
                        ADI_cube = cube_planet_free(
                            planet_parameter,
                            ADI_cube,
                            derot_angles,
                            psfn,
                            imlib,
                        )

                    ######################## 2. Crop the cube #########################

                    # DEPENDING ON THE SITUATION, CROP
                    PCA_ADI_cube_ori = ADI_cube.copy()
                    cy, cx = frame_center(PCA_ADI_cube_ori[0])
                    if planet:
                        if cc == 0:  # use crop cube
                            xx_comp = planet_pos_crop[0]
                            yy_comp = planet_pos_crop[1]
                        else:
                            xx_comp = planet_pos_full[0]
                            yy_comp = planet_pos_full[1]
                        r_pl = np.sqrt(
                            (xx_comp - cx) ** 2 + (yy_comp - cy) ** 2
                        )
                        
                    if cc == 0:  # use crop cube
                        rad_arr = fcp_pos_r_crop / plsc
                    else:
                        rad_arr = fcp_pos_r_full / plsc
                        
                    # adapt positions depending on mask and crop size
                    while rad_arr[0] < mask_IWA_px:
                        rad_arr = rad_arr[1:]
                            
                    while rad_arr[-1] >= PCA_ADI_cube_ori.shape[2] / 2:
                        rad_arr = rad_arr[:-1]
                        
                    nfcp = rad_arr.shape[0]
                    if not do_adi:
                        ADI_cube = None


                    ################# 3. First quick contrast curve ###################
                    # This is to determine the level at which each fcp should be injected
                    if cc == 0:
                        if (
                            not isfile(
                                outpath_5.format(
                                    bin_fac, filt, crop_lab_list[cc]
                                )
                                + "TMP_first_guess_5sig_sensitivity_"
                                + label_stg
                                + label_filt
                                + ".fits"
                            )
                            or not isfile(
                                outpath_5.format(
                                    bin_fac, filt, crop_lab_list[cc]
                                )
                                + "TMP_first_guess_contrast_curve_PCA-{}-full.csv".format(
                                    label_stg
                                )
                            )
                            or overwrite_pp
                        ):
                            df_list = []
                        # CROP ADI / REF CUBE to min size for sizes to match
                        if ref_cube is not None:
                            if (
                                ref_cube.shape[-1]
                                > PCA_ADI_cube_ori.shape[-1]
                            ):
                                ref_cube = cube_crop_frames(
                                    ref_cube, PCA_ADI_cube_ori.shape[-1]
                                )
                            elif (
                                ref_cube.shape[-1]
                                < PCA_ADI_cube_ori.shape[-1]
                            ):
                                PCA_ADI_cube_ori = cube_crop_frames(
                                    PCA_ADI_cube_ori, ref_cube.shape[-1]
                                )
                        nfr = PCA_ADI_cube_ori.shape[0]
                        firstguess_pcs = list(np.geomspace(1, nfr//2, n_firstguess_pcs))
                        firstguess_pcs = [int(firstguess_pcs[i]) for i in range(n_firstguess_pcs)]
                        for nn, npc in enumerate(firstguess_pcs):
                            pn_contr_curve_full_rr = contrast_curve(
                                PCA_ADI_cube_ori,
                                derot_angles,
                                psfn,
                                fwhm,
                                plsc,
                                starphot=starphot,
                                algo=pca,
                                sigma=5,
                                nbranch=n_br,
                                theta=0,
                                inner_rad=1,
                                wedge=(0, 360),
                                cube_ref=ref_cube,
                                scaling=scaling,
                                student=True,
                                transmission=transmission,
                                plot=True,
                                dpi=100,
                                verbose=verbose,
                                ncomp=int(npc),
                                svd_mode=svd_mode_all[0],
                                nproc=nproc,
                            )
                            # DF.to_csv(pn_contr_curve_full_nn, path_or_buf=outpath_4.format(crop_lab_list[cc])+'contrast_curve_PCA-ADI-full_optimal_at_{:.1f}as.csv'.format(rad*plsc), sep=',', na_rep='', float_format=None)
                            df_list.append(pn_contr_curve_full_rr)
                        pn_contr_curve_full_rsvd_opt = (
                            pn_contr_curve_full_rr.copy()
                        )

                        for jj in range(
                            pn_contr_curve_full_rsvd_opt.shape[0]
                        ):
                            sensitivities = []
                            for nn, npc in enumerate(firstguess_pcs):
                                sensitivities.append(
                                    df_list[nn]["sensitivity_student"][jj]
                                )
                            print(
                                "Sensitivities at {}: ".format(
                                    df_list[nn]["distance"][jj]
                                ),
                                sensitivities,
                            )
                            idx_min = np.argmin(sensitivities)
                            pn_contr_curve_full_rsvd_opt[
                                "sensitivity_student"
                            ][jj] = df_list[idx_min][
                                "sensitivity_student"
                            ][
                                jj
                            ]
                            pn_contr_curve_full_rsvd_opt[
                                "sensitivity_gaussian"
                            ][jj] = df_list[idx_min][
                                "sensitivity_gaussian"
                            ][
                                jj
                            ]
                            pn_contr_curve_full_rsvd_opt["throughput"][
                                jj
                            ] = df_list[idx_min]["throughput"][jj]
                            pn_contr_curve_full_rsvd_opt["noise"][jj] = (
                                df_list[idx_min]["noise"][jj]
                            )
                            pn_contr_curve_full_rsvd_opt["sigma corr"][
                                jj
                            ] = df_list[idx_min]["sigma corr"][jj]
                        DF.to_csv(
                            pn_contr_curve_full_rsvd_opt,
                            path_or_buf=outpath_5.format(
                                bin_fac, filt, crop_lab_list[cc]
                            )
                            + "TMP_optimal_contrast_curve_PCA-{}-full_randsvd.csv".format(
                                label_stg
                            ),
                            sep=",",
                            na_rep="",
                            float_format=None,
                        )
                        arr_dist = np.array(
                            pn_contr_curve_full_rsvd_opt["distance"]
                        )
                        arr_contrast = np.array(
                            pn_contr_curve_full_rsvd_opt[
                                "sensitivity_student"
                            ]
                        )

                        sensitivity_5sig_full_rsvd_df = np.zeros(nfcp)
                        for ff in range(nfcp):
                            idx = find_nearest(arr_dist, rad_arr[ff])
                            sensitivity_5sig_full_rsvd_df[ff] = (
                                arr_contrast[idx]
                            )
                        write_fits(
                            outpath_5.format(
                                bin_fac, filt, crop_lab_list[cc]
                            )
                            + "TMP_first_guess_5sig_sensitivity_{}".format(
                                label_stg
                            )
                            + label_filt
                            + ".fits",
                            sensitivity_5sig_full_rsvd_df,
                        )
                    else:
                        sensitivity_5sig_full_rsvd_df = open_fits(
                            outpath_5.format(
                                bin_fac, filt, crop_lab_list[cc]
                            )
                            + "TMP_first_guess_5sig_sensitivity_{}".format(
                                label_stg
                            )
                            + label_filt
                            + ".fits"
                        )
                        
                    med_contrast = np.nanmedian(pn_contr_curve_full_rsvd_opt["sensitivity_student"][:])
                    flux_ratio_mag = -2.5*np.log10(med_contrast)
                    if flux_ratio_mag < 0:
                        flux_ratio_mag *= -1

                    ############### 4. INJECT FAKE PLANETS AT 5-sigma #################
                    PCA_ADI_cube = PCA_ADI_cube_ori.copy()
                    if True:
                        th_step = (wedge[1] - wedge[0]) / nspi
                        for ns in range(nspi):
                            theta0 = th0 + ns * th_step
                            
                            th_stepr = (wedge[1] - wedge[0]) / nfcp
                            for ff in range(nfcp):
                                if (
                                    ff + 1
                                    > sensitivity_5sig_full_rsvd_df.shape[
                                        0
                                    ]
                                ):
                                    flevel = (
                                        np.median(starphot)
                                        * sensitivity_5sig_full_rsvd_df[-1]
                                        * injection_fac
                                        / np.sqrt(
                                            ((rad_arr[ff] * plsc) / 0.5)
                                        )
                                    )
                                else:
                                    flevel = (
                                        np.median(starphot)
                                        * sensitivity_5sig_full_rsvd_df[ff]
                                        * injection_fac
                                    )  # injected at ~3 sigma level instead of 5 sigma (rule is normalized at 0.5'', empirically it seems one has to be more conservative below 1'', hence division by radius)
                                PCA_ADI_cube = cube_inject_companions(
                                    PCA_ADI_cube,
                                    psfn,
                                    derot_angles,
                                    flevel,
                                    plsc=plsc,
                                    rad_dists=rad_arr[ff : ff + 1],
                                    n_branches=1,
                                    theta=(theta0 + ff * th_stepr) % 360,
                                    imlib=imlib,
                                    interpolation=interpolation,
                                    verbose=verbose,
                                    nproc=nproc,
                                )
                            write_fits(
                                outpath_5.format(
                                    bin_fac, filt, crop_lab_list[cc]
                                )
                                + "7_final_crop_PCA_cube"
                                + label_filt
                                + "_fcp_spi{:.0f}.fits".format(ns),
                                PCA_ADI_cube,
                            )
                            # vip.fits.append_extension(outpath_5.format(bin_fac,filt,crop_lab_list[cc])+'7_final_crop_PCA_cube'+label_filt+'_fcp_spi{:.0f}.fits'.format(ns), derot_angles)

                        nfcp_df = range(1, nfcp + 1)
                        if do_adi:
                            sensitivity_5sig_adi_df = np.zeros(nfcp)
                        if do_pca_full:
                            id_npc_full_df = np.zeros(nfcp)
                            sensitivity_5sig_full_df = np.zeros(nfcp)
                        if (
                            do_pca_ann
                            and cc == 0
                            and bin_fac == np.amax(bin_fac_list)
                        ):
                            id_npc_ann_df = np.zeros(nfcp)
                            sensitivity_5sig_ann_df = np.zeros(nfcp)

                    ######################### 5. Simple ADI ###########################
                    if do_adi:
                        if (
                            not isfile(
                                outpath_5.format(
                                    bin_fac, filt, crop_lab_list[cc]
                                )
                                + "final_ADI_simple"
                                + label_filt
                                + ".fits"
                            )
                            or overwrite_ADI
                        ):
                            params = MEDIAN_SUB_Params(
                                cube=ADI_cube,
                                angle_list=derot_angles,
                                fwhm=fwhm,
                                radius_int=mask_IWA_px,
                                delta_rot=delta_rot,
                                full_output=True,
                                verbose=True,
                                nproc=nproc,
                                imlib=imlib,
                            )
                            _, tmp, tmp_tmp = median_sub(
                                algo_params=params
                            )

                            tmp_tmp = mask_circle(
                                tmp_tmp, 0.9 * fwhm
                            )  # we mask the IWA
                            write_fits(
                                outpath_5.format(
                                    bin_fac, filt, crop_lab_list[cc]
                                )
                                + "final_ADI_simple"
                                + label_filt
                                + ".fits",
                                tmp_tmp,
                            )

                        # id_snr_adi_df[counter] = vip.metrics.snr(tmp_tmp, (xx_comp,yy_comp), fwhm)
                        ## Convolution
                        if (
                            not isfile(
                                outpath_5.format(
                                    bin_fac, filt, crop_lab_list[cc]
                                )
                                + "final_ADI_simple"
                                + label_filt
                                + "_conv.fits"
                            )
                            or overwrite_ADI
                        ) and do_conv:
                            tmp = open_fits(
                                outpath_5.format(
                                    bin_fac, filt, crop_lab_list[cc]
                                )
                                + "final_ADI_simple"
                                + label_filt
                                + ".fits"
                            )
                            tmp = frame_filter_lowpass(
                                tmp, mode="gauss", fwhm_size=fwhm / 2
                            )
                            write_fits(
                                outpath_5.format(
                                    bin_fac, filt, crop_lab_list[cc]
                                )
                                + "final_ADI_simple"
                                + label_filt
                                + "_conv.fits",
                                tmp,
                                verbose=False,
                            )
                        ## SNR map
                        if (
                            not isfile(
                                outpath_5.format(
                                    bin_fac, filt, crop_lab_list[cc]
                                )
                                + "final_ADI_simple"
                                + label_filt
                                + "_snrmap.fits"
                            )
                            or overwrite_ADI
                        ) and do_snr_map[0]:
                            tmp = open_fits(
                                outpath_5.format(
                                    bin_fac, filt, crop_lab_list[cc]
                                )
                                + "final_ADI_simple"
                                + label_filt
                                + ".fits"
                            )
                            # rad_in = mask_IWA
                            tmp_tmp = snrmap(
                                tmp, fwhm, plot=False, nproc=nproc
                            )
                            tmp_tmp = mask_circle(
                                tmp_tmp, mask_IWA_px
                            )  # rad_in*fwhm)
                            write_fits(
                                outpath_5.format(
                                    bin_fac, filt, crop_lab_list[cc]
                                )
                                + "final_ADI_simple"
                                + label_filt
                                + "_snrmap.fits",
                                tmp_tmp,
                                verbose=False,
                            )
                        ## Contrast curve ADI
                        # psfn = open_fits(outpath_2+'master_unsat_psf_norm'+'.fits')
                        # starphot = open_fits(outpath_4.format(bin_fac)+'7_norm_fact'+'.fits')
                        pn_contr_curve_adi = contrast_curve(
                            ADI_cube,
                            derot_angles,
                            psfn,
                            fwhm,
                            plsc,
                            starphot=starphot,
                            algo=median_sub,
                            sigma=5,
                            nbranch=n_br,
                            theta=0,
                            inner_rad=1,
                            wedge=(0, 360),
                            fc_snr=fc_snr,
                            student=True,
                            transmission=None,
                            smooth=True,
                            plot=False,
                            dpi=100,
                            debug=False,
                            verbose=verbose,
                            nproc=nproc,
                        )
                        DF.to_csv(
                            pn_contr_curve_adi,
                            path_or_buf=outpath_5.format(
                                bin_fac, filt, crop_lab_list[cc]
                            )
                            + "Optimal_contrast_curve_median-ADI-full.csv",
                            sep=",",
                            na_rep="",
                            float_format=None,
                        )
                        
                        ADI_cube = None

                    # CROP ADI / REF CUBE to min size for sizes to match
                    if ref_cube is not None:
                        if ref_cube.shape[-1] > PCA_ADI_cube.shape[-1]:
                            ref_cube = cube_crop_frames(
                                ref_cube, PCA_ADI_cube.shape[-1]
                            )
                        elif ref_cube.shape[-1] < PCA_ADI_cube.shape[-1]:
                            PCA_ADI_cube = cube_crop_frames(
                                PCA_ADI_cube, ref_cube.shape[-1]
                            )


                    ####################### 6. PCA-ADI full ###########################
                    if do_pca_full:
                        if planet:
                            # SUBTRACT THE PLANET FROM THE CUBE
                            cube_emp = cube_planet_free(
                                planet_parameter,
                                PCA_ADI_cube_ori,
                                derot_angles,
                                psfn,
                                imlib,
                            )
                            PCA_ADI_cube_ori = cube_emp.copy()
                            label_emp = "_empty" + label_filt
                        else:
                            cube_emp = PCA_ADI_cube_ori
                            label_emp = label_filt  # 9.1 Recompute the contrast curve for optimal npcs
                            
                        num_fake_planets = nspi
                        flux_ratio = mag2flux_ratio(flux_ratio_mag)
                        print("{} fake planets will be injected azimuthally at {} flux ratio ({:.1f} mag difference)".format(num_fake_planets,
                                                                                                                             flux_ratio, 
                                                                                                                             flux_ratio_mag))
                        contrast_instance = Contrast(science_sequence=cube_emp,
                                                     psf_template=psfn,
                                                     parang_rad=derot_angles,
                                                     psf_fwhm_radius=fwhm/2,
                                                     dit_psf_template=1.,
                                                     dit_science=starphot,
                                                     scaling_factor=1., # A factor to account e.g. for ND filters
                                                     checkpoint_dir=checkpoint_dir)
                        
                        # PCA_ADI_cube, derot_angles = vip.fits.open_adicube(outpath_5.format(bin_fac,filt,crop_lab_list[cc])+'7_final_crop_PCA_cube'+label_filt+'.fits')
                        # First let's readapt the number of pcs to be tested
                        components = test_pcs_full_all[cc]

                        test_pcs_str_list = [str(x) for x in test_pcs_full]
                        ntest_pcs = len(test_pcs_full)
                        if ntest_pcs < 21:
                            test_pcs_str = "npc" + "-".join(test_pcs_str_list)
                        else:
                            test_pcs_str = "npc{}-{}".format(test_pcs_full[0],
                                                             test_pcs_full[-1])

                        if (
                            mask_PCA is None
                            or strategy == "`ADI"
                        ):
                            mask_rdi = None
                        else:
                            mask_tmp = np.ones_like(
                                PCA_ADI_cube[0]
                            )
                            
                            if len(mask_PCA) == 2:
                                anchor_mask = (
                                    get_annulus_segments(
                                        mask_tmp,
                                        mask_PCA[0],
                                        mask_PCA[1]
                                        - mask_PCA[0],
                                        mode="mask",
                                    )[0]
                                )
                            else:
                                anchor_mask = mask_circle(
                                    mask_tmp,
                                    mask_PCA,
                                    fillwith=0,
                                    mode="in",
                                )
                            if mask_IWA_px > 0:
                                boat_mask = mask_circle(
                                    mask_tmp,
                                    mask_IWA_px,
                                    fillwith=0,
                                    mode="in",
                                )
                            else:
                                boat_mask = mask_tmp
                            mask_rdi = (anchor_mask, boat_mask)
                            
                        kwargs = {'mask_center_px':mask_IWA_px,
                                  'verbose':False,
                                  'cube_ref':ref_cube,
                                  'svd_mode':svd_mode_all[cc],
                                  'scaling':scaling,
                                  'fwhm':fwhm,
                                  'collapse':'median',
                                  'mask_rdi':mask_rdi,
                                  'nproc':1,
                                  'imlib':imlib,
                                  'interpolation':interpolation
                                  }
                            
                        contrast_instance.design_fake_planet_experiments(flux_ratios=flux_ratio,
                                                 num_planets=num_fake_planets,
                                                 overwrite=True)
                        algorithm_function = MultiComponentPCAvip(num_pcas=components, kwarg=kwargs)
                        contrast_instance.run_fake_planet_experiments(algorithm_function=algorithm_function, 
                                                  num_parallel=cpu_count()//2)

                        photometry_mode_planet = AperturePhotometryMode("ASS", # or "AS"
                                                                        psf_fwhm_radius=fwhm/2, 
                                                                        search_area=0.5)

                        photometry_mode_noise = AperturePhotometryMode("AS",psf_fwhm_radius=fwhm/2)

                        contrast_instance.prepare_contrast_results(
                            photometry_mode_planet=photometry_mode_planet,
                            photometry_mode_noise=photometry_mode_noise)
                        statistical_test = TTest()
                        
                        contrast_curves, contrast_errors = contrast_instance.compute_analytic_contrast_curves(
                            statistical_test=statistical_test,
                            confidence_level_fpf=gaussian_sigma_2_fpf(5),
                            num_rot_iter=20,
                            pixel_scale=plsc)
                        
                        overall_best = np.min(contrast_curves.values, axis=1)
                        separations_arcsec = contrast_curves.reset_index(level=0).index
                        separations_FWHM = contrast_curves.reset_index(level=1).index
                        
                        opt_contrast_curve = DF(
                            {
                                "separations_arcsec": separations_arcsec,
                                "separations_FWHM": separations_FWHM,
                                "sensitivity": overall_best
                            })
                            
                        DF.to_csv(
                            opt_contrast_curve,
                            path_or_buf=outpath_5.format(
                                bin_fac, filt, crop_lab_list[cc]
                            )
                            + "Optimal_contrast_curve_PCA-{}-full.csv".format(
                                label_stg
                            ),
                            sep=",",
                            na_rep="",
                            float_format=None,
                        )
                            
                        best_idx = np.argmin(contrast_curves.values, axis=1)
                        best_contrast_errors = contrast_errors.values[np.arange(len(best_idx)), best_idx]

                        # PLOT !
                        colors = sns.color_palette("rocket_r",
                                                   n_colors=len(contrast_curves.columns))
                        colors.append('b')
    
                        # 1.) Create Plot Layout
                        fig = plt.figure(constrained_layout=False, figsize=(12, 8))
                        gs0 = fig.add_gridspec(1, 1)
                        axis_contrast_curvse = fig.add_subplot(gs0[0, 0])
                        
                        
                        # ---------------------- Create the Plot --------------------
                        i = 0 # color picker
                        
                        for tmp_model in contrast_curves.columns:
                        
                            num_components = int(tmp_model[5:9])
                            tmp_flux_ratios = contrast_curves.reset_index(
                                level=0)[tmp_model].values
                            tmp_errors = contrast_errors.reset_index(
                                level=0)[tmp_model].values
                        
                            axis_contrast_curvse.plot(
                                separations_arcsec,
                                tmp_flux_ratios,
                                color = colors[i],
                                label=num_components)
                        
                            axis_contrast_curvse.fill_between(
                                separations_arcsec,
                                tmp_flux_ratios + tmp_errors, 
                                tmp_flux_ratios - tmp_errors,
                                color = colors[i],
                                alpha=0.5)
                            i+=1
                        
                        axis_contrast_curvse.set_yscale("log")
                        # ------------ Plot the overall best -------------------------
                        axis_contrast_curvse.plot(
                            separations_arcsec,
                            overall_best,
                            color = colors[i],
                            lw=3,
                            ls="--",
                            label="Best")
                        
                        # ------------- Double axis and limits -----------------------
                        lim_mag_y = (12.5, 6)
                        lim_arcsec_x = (0.1, 1.3)
                        sep_lambda_arcse = interpolate.interp1d(
                            separations_arcsec, 
                            separations_FWHM, 
                            fill_value='extrapolate')
                        
                        axis_contrast_curvse_mag = axis_contrast_curvse.twinx()
                        axis_contrast_curvse_mag.plot(
                            separations_arcsec,
                            flux_ratio2mag(tmp_flux_ratios),
                            alpha=0.)
                        axis_contrast_curvse_mag.invert_yaxis()
                        
                        axis_contrast_curvse_lambda = axis_contrast_curvse.twiny()
                        axis_contrast_curvse_lambda.plot(
                            separations_FWHM,
                            tmp_flux_ratios,
                            alpha=0.)
                        
                        axis_contrast_curvse.grid(which='both')
                        axis_contrast_curvse_mag.set_ylim(*lim_mag_y)
                        axis_contrast_curvse.set_ylim(
                            mag2flux_ratio(lim_mag_y[0]), 
                            mag2flux_ratio(lim_mag_y[1]))
                        
                        axis_contrast_curvse.set_xlim(
                            *lim_arcsec_x)
                        axis_contrast_curvse_mag.set_xlim(
                            *lim_arcsec_x)
                        axis_contrast_curvse_lambda.set_xlim(
                            *sep_lambda_arcse(lim_arcsec_x))
                        
                        # ----------- Labels and fontsizes --------------------------
                        
                        axis_contrast_curvse.set_xlabel(
                            r"Separation [arcsec]", size=16)
                        axis_contrast_curvse_lambda.set_xlabel(
                            r"Separation [FWHM]", size=16)
                        
                        axis_contrast_curvse.set_ylabel(
                            r"Planet-to-star flux ratio", size=16)
                        axis_contrast_curvse_mag.set_ylabel(
                            r"$\Delta$ Magnitude", size=16)
                        
                        axis_contrast_curvse.tick_params(
                            axis='both', which='major', labelsize=14)
                        axis_contrast_curvse_lambda.tick_params(
                            axis='both', which='major', labelsize=14)
                        axis_contrast_curvse_mag.tick_params(
                            axis='both', which='major', labelsize=14)
                        
                        axis_contrast_curvse_mag.set_title(
                            r"$5 \sigma_{\mathcal{N}}$ Contrast Curves",
                            fontsize=18, fontweight="bold", y=1.1)
                        
                        # --------------------------- Legend -----------------------
                        handles, labels = axis_contrast_curvse.\
                            get_legend_handles_labels()
                        
                        leg1 = fig.legend(handles, labels, 
                                          bbox_to_anchor=(0.12, -0.08), 
                                          fontsize=14, 
                                          title="# PCA components",
                                          loc='lower left', ncol=8)
                        
                        _=plt.setp(leg1.get_title(),fontsize=14)
                        
                        plt.savefig(outpath_5.format(
                            bin_fac, filt, crop_lab_list[cc]
                        )
                        + "Optimal_contrast_curve_PCA-{}-full.pdf".format(
                            label_stg
                        ), bbox_inches='tight')
    
                        counter +=1
            
    return None
