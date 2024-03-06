#! /usr/bin/env python
# coding: utf-8
"""
Utility routines for post-processing of SPHERE/IFS data.
"""

__author__ = 'V. Christiaens'

__all__ = ['postproc_IFS']

# *Version 1 (2019/12)* 
# *Version 2 (2020/04) - this version* 

# How to use this script:
# 1) First pass: set planet=False, fake_planets = False 
#    => check where are the planets, and write coords (x,y) if any
# 2) Second pass: set planet = True; planet_pos = (x,y)
#    => infer optimal npc for pca-annulus in NEGFC
# 3) Then run NEGFC script
#    => infer exact planet parameters
# 4) Third pass: set planet_parameter=np.array([[r,theta,flux]]) with results 
#                from NEGFC; subtract_planet = True, fake_planets = True
#    => build contrast curves

# Implement ADI contrast curve
######################### Importations and definitions ########################

import json
from matplotlib import use as mpl_backend
from matplotlib import pyplot as plt
from multiprocessing import cpu_count
import numpy as np
from os.path import isfile, isdir
import os
from pandas.core.frame import DataFrame as DF

from vip_hci.psfsub import pca, pca_annular, PCA_Params, PCA_ANNULAR_Params
from vip_hci.metrics import stim_map as compute_stim_map
from vip_hci.metrics import inverse_stim_map as compute_inverse_stim_map
from vip_hci.fm import cube_inject_companions, cube_planet_free, find_nearest
from vip_hci.metrics import contrast_curve, snr, snrmap
from vip_hci.preproc import cube_derotate
from vip_hci.fits import open_fits, write_fits
from vip_hci.var import mask_circle, frame_center

from vcal import __path__ as vcal_path
mpl_backend('Agg')


############### PARAMETERS to be adapted to each dataset #####################
#from C_2019_10_J19003645.IFS_reduction.VCAL_2_preproc_IFS import distort_corr_labs # adapt location of 2nd script
#from C_2019_10_J19003645.IFS_reduction.VCAL_2_preproc_IFS import final_cubename, final_anglename, final_psfnname, final_lbdaname
#from C_2019_10_J19003645.IFS_reduction.VCAL_2_preproc_IFS import final_fluxname, final_fwhmname, path_ifs, plsc # adapt location of 2nd script
#from C_2019_10_J19003645.IFS_reduction.VCAL_2_preproc_IFS import outpath as inpath

# Suggestion: run this script several times with the following parameters:
#1. planet = False, fake_planet=False => do_adi= True, do_pca_full=True, do_adi_ann=True
#2. If a blob is found: set planet_pos_crop to the coordinates of the blob. Set planet=True and do_pca_sann=True
#3. If no blob is found: do_pca_sann=False; fake_planet=True => will calculate contrast curves
#4. If a blob is found, infer its (r, theta, flux) using NEGFC (other script) => rerun 3. after setting planet_parameter to the result of NEGFC, and subtract_planet=True    


##################### START POST-PROCESSING - don't  modify below #############

def postproc_IFS(params_postproc_name='VCAL_params_postproc_IFS.json',
                 params_preproc_name='VCAL_params_preproc_IFS.json', 
                 params_calib_name='VCAL_params_calib.json',
                 planet_parameter=None) -> None:
    """
    Postprocessing of SPHERE/IFS data using preproc parameters provided in 
    json file.

    *Suggestion: run this routine several times with the following parameters 
    set in the parameter file:
        #1. planet = False, fake_planet=False 
            => do_adi= True, do_adi_full=True, do_adi_ann=True
        #2. If a blob is found: set planet_pos to the coordinates of the
            blob. Set planet=True and do_pca_sann=True (single annulus PCA)
        #3. If no blob is found: do_pca_sann=False; fake_planet=True => will 
            calculate contrast curves
        #4. If a blob is found, infer its (r, theta, flux) using NEGFC 
            (other script) => rerun 3. after setting planet_parameter to the 
            result of NEGFC, and subtract_planet=True to get contrast curve.

    Input:
    ******
    params_postproc_name: str, opt
        Full path + name of the json file containing postproc parameters.
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
    None. All products are written as fits files in a 3_postproc subfolder.
    
    """
    ##################### 0. Load all parameters ################################
    plt.style.use('default')
    with open(params_postproc_name, 'r') as read_file_params_postproc:
        params_postproc = json.load(read_file_params_postproc)
    with open(params_preproc_name, 'r') as read_file_params_preproc:
        params_preproc = json.load(read_file_params_preproc)
    with open(params_calib_name, 'r') as read_file_params_calib:
        params_calib = json.load(read_file_params_calib)

    # from calib
    path = params_calib['path']
    path_ifs = path+"IFS_reduction/"
    outpath_fig = path_ifs+'outpath_fig/'
    
    # from preproc
    coro = params_preproc['coro']
    plsc = np.array(params_preproc['plsc'])
    final_crop_sz = params_preproc.get('final_crop_sz', 101)
    final_cubename = params_preproc.get('final_cubename', 'final_cube_ASDI')
    final_anglename = params_preproc.get('final_anglename', 'final_derot_angles')
    final_psfname = params_preproc.get('final_psfname', 'final_psf_med')
    final_fluxname = params_preproc.get('final_fluxname','final_flux')
    final_fwhmname = params_preproc.get('final_fwhmname','final_fwhm')  
    final_lbdaname = params_preproc.get('final_lbdaname','lbdas') 
    final_scalefacname = params_preproc.get('final_scalefacname', None)
#    psf_name = final_psfname # possibly change if not PSF (e.g. OBJ is not saturated)
#    fwhm_name = final_fwhmname
    fluxes_name = final_fluxname #just the first row (second corresponds to uncertainties?)
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
    label_test_pre = params_preproc.get('label_test', '')
    inpath = path_ifs+"2_preproc_vip{}/".format(label_test_pre)
        
    # from postproc param file
    source = params_postproc['source']            # should be without space
    sourcename = params_postproc['sourcename']    # can have spaces
    details = params_postproc['details']          # such as instrument and date
    label_test = params_postproc.get('label_test', "")    # manually provide a test label
    
    ## Options
    verbose = params_postproc.get("verbose",0)     # whether to print(more information during the reduction
    debug = params_postproc.get("debug",False)   
    #debug = True                        # whether to print(even more information, helpful for debugging
    #debug_ = True                       # whether to keep all the intermediate fits files of the reduction by the end of the notebook (useful for debugging)
    nproc = params_postproc.get('nproc',int(cpu_count()/2))                          # number of processors to use - can also be set to cpu_count()/2 for efficiency
    overwrite_pp = params_postproc.get('overwrite_pp',1)         # whether to overwrite PCA-ADI results
    
    ## TO DO?
    do_sdi = params_postproc.get('do_sdi', 1)
    do_adi_full = params_postproc.get('do_adi_full', 1)     # PCA-ADI in full frame in each spectral channel
    do_adi_ann = params_postproc.get('do_adi_ann', 0)       # PCA-ADI in annuli in each spectral channel
    do_sadi_full = params_postproc.get('do_sadi_full', 1)   # PCA-SADI in full frame
    do_sadi_ann = params_postproc.get('do_sadi_ann', 0)     # PCA-SADI in annuli

    ## Planet?
    planet = params_postproc.get('planet',0)                      # is there a companion?
    planet_pos = params_postproc.get('planet_pos',None) # If so, where is it (or where is it expected)?   (x, y) in frame
    subtract_planet = params_postproc.get('subtract_planet',0)    # this should only be used as a second iteration, after negfc on the companion has enabled to determine its parameters

    
    ## Inject fake companions? If True => will compute contrast curves
    fake_planet = params_postproc.get('fake_planet',0)                 #  FIRST RUN IT AS FALSE TO CHECK FOR THE PRESENCE OF TRUE COMPANIONS
    fcp_pos_r = np.array(params_postproc.get('fcp_pos_r',[0.2,0.4]))
    fc_snr = params_postproc.get('fc_snr',10.) # snr of the injected fcp in contrast_curve to compute throughput
    nspi = params_postproc.get('nspi',9)        # number of spirals where fcps should be injected - also corresponds to number of PAs where the contrast curve is computed
    wedge = tuple(params_postproc.get('wedge',[0,360])) # in which range of PA should the contrast curve be computed

    ## Post-processing
    sadi_steps  =params_postproc.get('sadi_steps',1)
    mask_IWA_px = params_postproc.get('mask_IWA',10)  # just show pca images beyond the provided mask radius (provide in pixels)
    do_snr_map = params_postproc.get('do_snr_map',[0,0,0]) # to plot the snr_map (warning: computer intensive); useful only when point-like features are seen in the image
    if not isinstance(do_snr_map, list):
        do_snr_map = [do_snr_map]*3
    do_stim_map = params_postproc.get('do_stim_map',[0,0,0]) # to plot the snr_map (warning: computer intensive); useful only when point-like features are seen in the image
    if not isinstance(do_stim_map, list):
        do_stim_map = [do_stim_map]*3
    exclude_negative_lobes = params_postproc.get('exclude_negative_lobes', 1)
    #flux_weights = False # whether to combine residual frames based on original measured fluxes of the star (proxy of AO quality) ## TRY BOTH!
    ###RDI
    ref_cube_name = params_postproc.get('ref_cube_name',None)
    #prep_ref_cube = params_postproc.get('prep_ref_cube',[1,2])
    scaling = params_postproc.get('scaling',None) # for RDI
    mask_PCA = params_postproc.get('mask_PCA',None)
    ##SDI
    start_nz = params_postproc.get('start_nz',0)
    end_nz = params_postproc.get('end_nz',-1)
    adimsdi = params_postproc.get('adimsdi',"double")
    crop_ifs = params_postproc.get('crop_ifs',0)  # whether to crop the IFS frames after rescaling during PCA-SDI. Leave it to False for SINFONI.
    scalings = [params_postproc.get('scaling',None)] # list of pre-PCA cube scaling(s) to be tested for PCA-SADI.

    ### PCA options
    delta_rot = params_postproc.get('delta_rot', 1)  # float or int expressed in FWHM. Threshold in azimuthal motion to keep frames in the PCA library
    delta_rot_ann = params_postproc.get('delta_rot_ann', [1, 3]) # float, int or tuple expressed in FWHM. Threshold in azimuthal motion to keep frames in the PCA library created by PCA-annular. If a tuple, corresponds to the threshold for the innermost and outermost annuli, respectively.
    if type(delta_rot_ann) == list and len(delta_rot_ann) == 2:
        delta_rot_ann = tuple(delta_rot_ann)  # converts to tuple as .json parameter file does not support tuples
    asize=params_postproc.get('asize',3) # width of the annnuli for either pca in concentric annuli or on a single annulus, provided in FWHM
    #### how is SVD done for PCA:
    svd_mode = params_postproc.get('svd_mode','lapack')
    #### number of principal components
    firstguess_pcs = params_postproc.get('firstguess_pcs',[1,21,1])  # explored for first contrast curve 
    test_pcs_sdi = params_postproc.get('pcs_sdi',[1,5,1])
    test_pcs_adi_full = params_postproc.get('pcs_adi_full',[1,21,1]) 
    test_pcs_adi_ann = params_postproc.get('pcs_adi_ann',[1,11,1])
    test_pcs_sadi_full = params_postproc.get('pcs_sadi_full',[1,11,1]) 
    test_pcs_sadi_full_sdi = params_postproc.get('pcs_sadi_full_sdi',[1,4,1]) 
    test_pcs_sadi_full_adi = params_postproc.get('pcs_sadi_full_adi',[1,11,1]) 
    test_pcs_sadi_ann_sdi = params_postproc.get('pcs_sadi_ann_sdi',[1,4,1]) 
    test_pcs_sadi_ann_adi = params_postproc.get('pcs_sadi_ann_adi',[1,4,1])

    #### min/max number of frames to create PCA library
    max_fr_list = params_postproc.get('max_fr_list',[50])
    
    ############### 1. Define variables and load data + FORMATTING ################
    fwhm = open_fits(inpath+final_fwhmname)
    fwhm_med = np.median(fwhm)
    asize*=fwhm_med

    firstguess_pcs = list(range(firstguess_pcs[0],firstguess_pcs[1],firstguess_pcs[2]))
    test_pcs_sdi = list(range(test_pcs_sdi[0],test_pcs_sdi[1],test_pcs_sdi[2]))
    test_pcs_adi_full = list(range(test_pcs_adi_full[0],test_pcs_adi_full[1],test_pcs_adi_full[2]))
    test_pcs_adi_ann = list(range(test_pcs_adi_ann[0],test_pcs_adi_ann[1],test_pcs_adi_ann[2]))
    test_pcs_sadi_full = list(range(test_pcs_sadi_full[0],test_pcs_sadi_full[1],test_pcs_sadi_full[2]))
    test_pcs_sadi_full_sdi = list(range(test_pcs_sadi_full_sdi[0],test_pcs_sadi_full_sdi[1],test_pcs_sadi_full_sdi[2]))
    test_pcs_sadi_full_adi = list(range(test_pcs_sadi_full_adi[0],test_pcs_sadi_full_adi[1],test_pcs_sadi_full_adi[2]))
    test_pcs_sadi_ann_sdi = list(range(test_pcs_sadi_ann_sdi[0],test_pcs_sadi_ann_sdi[1],test_pcs_sadi_ann_sdi[2]))
    test_pcs_sadi_ann_adi = list(range(test_pcs_sadi_ann_adi[0],test_pcs_sadi_ann_adi[1],test_pcs_sadi_ann_adi[2]))

    th0 = wedge[0]  # trigonometric angle for the first fcp to be injected
    
    # Plotting options
    all_markers_shape = ['o','s','d','+','x']
    colors = ['k','g','b','m','r','c','y'] # for plotting of ADI full, ADI ann and SADI ann
    all_markers= ['ro','yo','bo','go','ko','co','mo']*nspi # for plotting the snr of the fcps (should contain at least as many elements as fcps)

    if coro:
        transmission_name = vcal_path[0] + "/../Static/" + "SPHERE_IRDIS_ALC_transmission_px.fits"
        transmission = open_fits(transmission_name)
        transmission = (transmission[1],transmission[0])
    else:
        transmission = None

    if isinstance(final_crop_sz,list):
        ncrop = len(final_crop_sz)
        for i in range(ncrop):
            if final_crop_sz[ncrop-1-i]%2:
                final_crop_sz = final_crop_sz[ncrop-1-i]
                break
    
    # for bb, bin_fac in enumerate(bin_fac_list):
    if not isdir(outpath_fig):
        os.system("mkdir "+outpath_fig)
    
    PCA_ASDI_cube_ori= open_fits(inpath+final_cubename)
    nz, nn, ny, nx = PCA_ASDI_cube_ori.shape

    # for SADI processing, whether to median combine all spectral channels or just certain channels, or loop over channels
    # if single integers for start and end channel is given
    if type(start_nz) and type(end_nz) == int:
        if end_nz < 0:  # in case -1 or -2 etc. is used
            end_nz += nz
        ifs_collapse_range_list = [(start_nz,end_nz)]
        if end_nz - start_nz + 1 == nz:  # if all channels are to be used (+1 for zero-based indexing)
            ifs_collapse_range_lab = ['all_ch']
        else:
            ifs_collapse_range_lab = ['ch{}-{}'.format(start_nz+1, end_nz+1)]
        print('Using spectral channels {}-{} for SADI'.format(start_nz+1, end_nz+1), flush=True)
    elif type(start_nz) and type(end_nz) == list:
        if len(start_nz) == len(end_nz):
            ifs_collapse_range_list = []
            ifs_collapse_range_lab = []
            for i in range(len(end_nz)):
                if end_nz[i] < 0:
                    end_nz[i] += nz
                ifs_collapse_range_list.append((start_nz[i], end_nz[i]))
                if end_nz[i] - start_nz[i] + 1 == nz:
                    ifs_collapse_range_lab.append(('all_ch'))  # don't remove redundant parenthesis
                else:
                    ifs_collapse_range_lab.append(('ch{}-{}'.format(start_nz[i] + 1, end_nz[i] + 1)))
                print('Using spectral channels {}-{} for SADI reduction {}'.format(start_nz[i] + 1, end_nz[i] + 1, i + 1),
                      flush=True)
        else:
            raise TypeError("Lists of start and end spectral channels should be equal length.")
    else:
        raise TypeError("Start and end spectral channel should be ints or lists.")

    derot_angles = open_fits(inpath+final_anglename)
    lbdas = open_fits(inpath+final_lbdaname)
    
    psfn = open_fits(inpath+final_psfname)
    starphot = open_fits(inpath+fluxes_name)[0]
    
    if final_scalefacname is not None:
        scale_list = open_fits(inpath+final_scalefacname, verbose=debug)
        msg = ("\nUsing scaling factors from pre-processing. \n"
               "It is recommended these are checked to ensure that they make sense.\n")
        print(msg, flush=True)
    else:
        scale_list = np.amax(lbdas)/lbdas
        if verbose:
            msg = "\nUsing theoretical scaling factors.\n"
            print(msg, flush=True)

    #mask_IWA_px = int(mask_IWA*fwhm_med)
    mask_IWA = mask_IWA_px/fwhm_med
    ref_cube = [None]*nz
    if ref_cube_name is not None:
        label_stg = "RDI"
        if scaling is not None:
            label_stg += "_"+scaling
        if mask_PCA is not None:
            label_stg += "_mask{:.1f}".format(mask_PCA)
            mask_PCA = int(mask_PCA/np.median(plsc))
    else:
        label_stg = "ADI"
            
    if subtract_planet:
        PCA_ASDI_cube_ori = cube_planet_free(planet_parameter, PCA_ASDI_cube_ori, derot_angles, psfn)
    if planet or fake_planet:
        cy, cx = frame_center(PCA_ASDI_cube_ori[0,0])
    if planet:
        xx_comp = planet_pos[0]
        yy_comp = planet_pos[1]
    #     r_pl = np.sqrt((xx_comp-cx)**2+(yy_comp-cy)**2)
    if fake_planet:
        rad_arr = fcp_pos_r/plsc
        while rad_arr[-1] >= PCA_ASDI_cube_ori.shape[2]:
            rad_arr = rad_arr[:-1]
        nfcp = rad_arr.shape[0]                        
    
    
    for max_fr in max_fr_list:
        label_test_ann = label_test
        if label_test == "":  # if no custom label is provided, make one
            label_test = '_mask{:.1f}_deltarot{:.1f}_maxfr{:.0f}'.format(mask_IWA, delta_rot, max_fr)
            if type(delta_rot_ann) == tuple:
                label_test_ann = '_mask{:.1f}_deltarot{:.1f}-{:.1f}_maxfr{:.0f}'.format(mask_IWA, delta_rot_ann[0], delta_rot_ann[1], max_fr)
            # if delta_rot_ann is an int or a float, show it in the label to one decimal place
            elif type(delta_rot_ann) == int or type(delta_rot_ann) == float:
                label_test_ann = '_mask{:.1f}_deltarot{:.1f}_maxfr{:.0f}'.format(mask_IWA, delta_rot_ann, max_fr)
            else:
                label_test_ann = label_test
        outpath = path_ifs+"3_postproc{}/".format(label_test)
        if not isdir(outpath):
            os.system("mkdir "+outpath)
        ################# 2. First quick contrast curve ###################
        # This is to determine the level at which the fake companions should be injected
        if fake_planet:
            if not isfile(outpath+'TMP_first_guess_5sig_sensitivity_ASDI'+label_test+'.fits') or not isfile(outpath+'TMP_optimal_contrast_curve_PCA-ASDI-full_randsvd.csv') or overwrite_pp:
                df_list = []
                for nn, npc in enumerate(firstguess_pcs):
                    pn_contr_curve_full_rr = contrast_curve(PCA_ASDI_cube_ori, derot_angles, psfn,
                                                            fwhm, plsc, starphot=starphot,
                                                            algo=pca, sigma=5., nbranch=1,
                                                            theta=0, inner_rad=1, wedge=(0,360),
                                                            fc_snr=fc_snr,
                                                            student=True, transmission=transmission,
                                                            plot=True, dpi=100,
                                                            verbose=verbose, ncomp=int(npc), svd_mode=svd_mode,
                                                            scale_list=scale_list, adimsdi='single', nproc=nproc)
                    df_list.append(pn_contr_curve_full_rr)
                pn_contr_curve_full_rsvd_opt = pn_contr_curve_full_rr.copy()
        
                for jj in range(pn_contr_curve_full_rsvd_opt.shape[0]):  
                    sensitivities = []
                    for rr, rad in enumerate(rad_arr):
                        sensitivities.append(df_list[rr]['sensitivity (Student)'][jj])
                    print("Sensitivities at {}: ".format(df_list[rr]['distance'][jj]), sensitivities)
                    idx_min = np.argmin(sensitivities)
                    pn_contr_curve_full_rsvd_opt['sensitivity (Student)'][jj] = df_list[idx_min]['sensitivity (Student)'][jj]
                    pn_contr_curve_full_rsvd_opt['sensitivity (Gauss)'][jj] = df_list[idx_min]['sensitivity (Gauss)'][jj]
                    pn_contr_curve_full_rsvd_opt['throughput'][jj] = df_list[idx_min]['throughput'][jj]
                    pn_contr_curve_full_rsvd_opt['noise'][jj] = df_list[idx_min]['noise'][jj]
                    pn_contr_curve_full_rsvd_opt['sigma corr'][jj] = df_list[idx_min]['sigma corr'][jj]
                DF.to_csv(pn_contr_curve_full_rsvd_opt, path_or_buf=outpath+'TMP_optimal_contrast_curve_PCA-ASDI-full_randsvd.csv', sep=',', na_rep='', float_format=None)
                arr_dist = np.array(pn_contr_curve_full_rsvd_opt['distance'])
                arr_contrast = np.array(pn_contr_curve_full_rsvd_opt['sensitivity (Student)'])
                
                sensitivity_5sig_full_rsvd_df = np.zeros(nfcp)
                for ff in range(nfcp):
                    idx = find_nearest(arr_dist, rad_arr[ff])
                    sensitivity_5sig_full_rsvd_df[ff] = arr_contrast[idx]
                write_fits(outpath+'TMP_first_guess_5sig_sensitivity_ASDI'+label_test+'.fits', sensitivity_5sig_full_rsvd_df)
            else:
                sensitivity_5sig_full_rsvd_df = open_fits(outpath+'TMP_first_guess_5sig_sensitivity_ASDI'+label_test+'.fits')               
                
                
        ############### 3. INJECT FAKE PLANETS AT 5-sigma #################
        if fake_planet:
            th_step = 360./nspi
            for ns in range(nspi):
                theta0 = th0+ns*th_step
                PCA_ASDI_cube = PCA_ASDI_cube_ori.copy()
                for ff in range(nfcp):
                    if ff+1 > sensitivity_5sig_full_rsvd_df.shape[0]:
                        flevel = starphot*sensitivity_5sig_full_rsvd_df[-1]/np.sqrt(((rad_arr[ff]*plsc)/0.5))
                    else:
                        flevel = starphot*sensitivity_5sig_full_rsvd_df[ff] # injected at ~3 sigma level instead of 5 sigma (rule is normalized at 0.5'', empirically it seems one has to be more conservative below 1'', hence division by radius)
                    PCA_ASDI_cube = cube_inject_companions(PCA_ASDI_cube, psfn,
                                                           derot_angles, flevel,
                                                           plsc, rad_dists=rad_arr[ff:ff+1],
                                                           n_branches=1, theta=(theta0+ff*th_step)%360,
                                                           imlib='opencv', verbose=verbose, nproc=nproc)
                write_fits(outpath+'PCA_cube'+label_test+'_fcp_spi{:.0f}.fits'.format(ns), PCA_ASDI_cube)
                #vip.fits.append_extension(outpath_5.format(bin_fac,filt,crop_lab_list[cc])+'7_final_crop_PCA_cube'+label_test+'_fcp_spi{:.0f}.fits'.format(ns), derot_angles)
        
            nfcp_df = range(1,nfcp+1)
            if do_sdi:
                id_npc_sdi_df = np.zeros(nfcp)
                sensitivity_5sig_sdi_df = np.zeros(nfcp)                
            if do_adi_full:
                id_npc_adi_full_df = np.zeros(nfcp)
                #sensitivity_5sig_adi_full_df = np.zeros(nfcp)
            if do_adi_ann:
                id_npc_adi_ann_df = np.zeros([nfcp,2])
                #sensitivity_5sig_adi_ann_df = np.zeros(nfcp)  
            if do_sadi_full:
                id_npc_full_df = np.zeros(nfcp)
                sensitivity_5sig_full_df = np.zeros(nfcp)
            if do_sadi_ann:
                id_npc_ann_df = np.zeros([nfcp,2])
                sensitivity_5sig_ann_df = np.zeros(nfcp)                           
        else:
            PCA_ASDI_cube = PCA_ASDI_cube_ori.copy()
        
        
        ######################### 4. PCA-SDI ###########################
        if do_sdi:
            test_pcs_str_list = [str(x) for x in test_pcs_sdi]
            test_pcs_str = "npc"+"-".join(test_pcs_str_list)
            ntest_pcs = len(test_pcs_sdi)
            if mask_PCA is None:
                mask_rdi = None
            else:
                mask_tmp = np.ones_like(PCA_ASDI_cube[0,0])
                mask_rdi = mask_circle(mask_tmp, mask_PCA, fillwith=0, mode='in')
            if not fake_planet:
                if not isfile(outpath+'final_PCA-SDI_'+test_pcs_str+label_test+'.fits') or overwrite_pp:
                    if planet:                
                        snr_tmp = np.zeros(ntest_pcs)
                    tmp = np.zeros([ntest_pcs,PCA_ASDI_cube.shape[2],PCA_ASDI_cube.shape[3]])
                    if do_stim_map:
                        stim_map = np.zeros([ntest_pcs,PCA_ASDI_cube.shape[2],PCA_ASDI_cube.shape[3]])
                        inv_stim_map = np.zeros([ntest_pcs,PCA_ASDI_cube.shape[2],PCA_ASDI_cube.shape[3]])
                        thr = np.zeros(ntest_pcs)
                    for pp, npc in enumerate(test_pcs_sdi):
                        params_pca = PCA_Params(cube=PCA_ASDI_cube, angle_list=derot_angles, cube_ref=None,
                                               scale_list=scale_list,  ncomp=(int(npc),None), svd_mode=svd_mode,
                                               scaling=None, mask_center_px=mask_IWA_px, adimsdi='double',
                                               crop_ifs=crop_ifs, delta_rot=delta_rot, fwhm=fwhm_med, collapse='median',
                                               check_memory=True, full_output=True, verbose=verbose, mask_rdi=mask_rdi,
                                               nproc=nproc, imlib="opencv", imlib2="opencv")
                        tmp[pp], tmp_tmp, tmp_tmp_der = pca(algo_params=params_pca)
                        if do_stim_map:
                            stim_map[pp] = compute_stim_map(tmp_tmp_der)
                            inv_stim_map[pp] = compute_inverse_stim_map(tmp_tmp,derot_angles, nproc=nproc)
                            thr[pp] = np.amax(inv_stim_map[pp])
                        if planet:
                            snr_tmp[pp] = snr(tmp[pp], (xx_comp,yy_comp), fwhm_med, plot=False, exclude_negative_lobes=exclude_negative_lobes,
                                                          verbose=False)
                    write_fits(outpath+'final_PCA-SDI_'+test_pcs_str+label_test+'.fits', tmp)
                    if do_stim_map:
                        write_fits(outpath+'final_PCA-SDI_'+test_pcs_str+label_test+'_stimmap.fits', stim_map)
                        write_fits(outpath+'final_PCA-SDI_'+test_pcs_str+label_test+'_invstimmap.fits', inv_stim_map)
                        write_fits(outpath+'final_PCA-SDI_'+test_pcs_str+label_test+'_stimthr.fits', thr)
                    if planet:
                        plt.close() 
                        plt.figure()
                        plt.title('SNR for '+sourcename+' b'+details+ '(PCA-ADI full-frame)')
                        plt.ylabel('SNR')
                        plt.xlabel('npc')  
                        for pp, npc in enumerate(test_pcs_sdi):
                            if snr_tmp[pp] > 5:
                                marker = 'go'
                            elif snr_tmp[pp] > 3:
                                marker = 'bo'
                            else:
                                marker = 'ro'
                            plt.plot(npc, snr_tmp[pp], marker)   
                        plt.savefig(outpath+'SNR_'+source+'_PCA-ASDI-full'+'.pdf', format='pdf')
                        write_fits(outpath+'final_PCA-SDI_SNR_'+test_pcs_str+label_test+'.fits', snr_tmp)
                else:
                    tmp_tmp = open_fits(outpath+'final_PCA-SDI_'+test_pcs_str+label_test+'.fits')                    
                ## SNR map  
                if (not isfile(outpath+'final_PCA-SDI_'+test_pcs_str+label_test+'_snrmap.fits') or overwrite_pp) and do_snr_map:
                    tmp = open_fits(outpath+'final_PCA-SDI_'+test_pcs_str+label_test+'.fits')
                    rad_in = mask_IWA
                    for pp in range(tmp.shape[0]):
                        tmp[pp] = snrmap(tmp[pp], fwhm_med, plot=False, nproc=nproc)
                        tmp[pp] = mask_circle(tmp[pp],rad_in*fwhm_med)
                    write_fits(outpath+'final_PCA-SDI_'+test_pcs_str+label_test+'_snrmap.fits', tmp, verbose=False)
                
            else:
                snr_tmp_tmp = np.zeros([nspi,ntest_pcs,nfcp])
                tmp_tmp = np.zeros([ntest_pcs,PCA_ASDI_cube.shape[2],PCA_ASDI_cube.shape[3]])
                for ns in range(nspi):
                    theta0 = th0+ns*th_step
                    PCA_ASDI_cube = open_fits(outpath+'PCA_cube'+label_test+'_fcp_spi{:.0f}.fits'.format(ns))
                    for pp, npc in enumerate(test_pcs_sdi):
                        if svd_mode == 'randsvd':
                            tmp_tmp_tmp = np.zeros([3,PCA_ASDI_cube.shape[2],PCA_ASDI_cube.shape[3]])
                            for nr in range(3):
                                params_pca = PCA_Params(cube=PCA_ASDI_cube, angle_list=derot_angles, cube_ref=None,
                                                       scale_list=scale_list, ncomp=(int(npc),None), svd_mode=svd_mode,
                                                       scaling=None, mask_center_px=mask_IWA_px,crop_ifs=crop_ifs,
                                                       delta_rot=delta_rot, fwhm=fwhm_med, collapse='median',
                                                       check_memory=True, adimsdi='double', full_output=False,
                                                       verbose=verbose, nproc=nproc)
                                tmp_tmp_tmp[nr] = pca(algo_params=params_pca)
                            tmp_tmp[pp] = np.median(tmp_tmp_tmp, axis=0)
                        else:
                            params_pca = PCA_Params(cube=PCA_ASDI_cube, angle_list=derot_angles, cube_ref=None,
                                                   scale_list=scale_list, ncomp=(int(npc),None), svd_mode=svd_mode,
                                                   scaling=None, mask_center_px=mask_IWA_px,crop_ifs=crop_ifs,
                                                   delta_rot=delta_rot, fwhm=fwhm_med, collapse='median',
                                                   check_memory=True, adimsdi='double', full_output=False,
                                                   verbose=verbose, nproc=nproc)
                            tmp_tmp[pp] = pca(algo_params=params_pca)
                                    
                        for ff in range(nfcp):
                            xx_fcp = cx + rad_arr[ff]*np.cos(np.deg2rad(theta0+ff*th_step))
                            yy_fcp = cy + rad_arr[ff]*np.sin(np.deg2rad(theta0+ff*th_step))
                            snr_tmp_tmp[ns,pp,ff] = snr(tmp_tmp[pp], (xx_fcp,yy_fcp), fwhm_med, plot=False, exclude_negative_lobes=exclude_negative_lobes,
                                                                  verbose=True)   
                    write_fits(outpath+'TMP_PCA-SDI_'+test_pcs_str+label_test+'_fcp_spi{:.0f}.fits'.format(ns), tmp_tmp)                                             
                snr_fcp = np.median(snr_tmp_tmp, axis=0)
                plt.close() 
                plt.figure()
                plt.title('SNR for fcps '+details+ '(PCA-SDI)')
                plt.ylabel('SNR')
                plt.xlabel('npc')
                for ff in range(nfcp):
                    marker = all_markers[ff]
                    for pp, npc in enumerate(test_pcs_sdi):
                        plt.plot(npc, snr_fcp[pp,ff], marker)
                try:
                    plt.savefig(outpath_fig+'SNR_fcps_PCA-SDI'+'.pdf', format='pdf')
                except:
                    pass
                write_fits(outpath+'final_PCA-SDI_SNR_fcps_'+test_pcs_str+label_test+'.fits', snr_fcp)   
                
                ## Find best npc for each radius
                for ff in range(nfcp):
                    idx_best_snr = np.argmax(snr_fcp[:,ff])
                    id_npc_sdi_df[ff] = test_pcs_sdi[idx_best_snr]
         
                
                ## 7.3. Final PCA-SDI frames with optimal npcs        
                tmp_tmp = np.zeros([nfcp,PCA_ASDI_cube_ori.shape[1],PCA_ASDI_cube_ori.shape[2]])
                test_pcs_str_list = [str(int(x)) for x in id_npc_sdi_df]                               
                test_pcs_str = "npc_opt"+"-".join(test_pcs_str_list)
                test_rad_str_list = ["{:.1f}".format(x) for x in rad_arr*plsc]                               
                test_rad_str = "rad"+"-".join(test_rad_str_list)
                for pp, npc in enumerate(id_npc_sdi_df):
                    if svd_mode == 'randsvd':
                        tmp_tmp_tmp = np.zeros([3,PCA_ASDI_cube_ori.shape[1],PCA_ASDI_cube_ori.shape[2]])
                        for nr in range(3):
                            params_pca = PCA_Params(cube=PCA_ASDI_cube_ori, angle_list=derot_angles, cube_ref=None,
                                                   scale_list=scale_list, ncomp=(int(npc),None), svd_mode=svd_mode,
                                                   scaling=None, mask_center_px=mask_IWA_px, crop_ifs=crop_ifs,
                                                   delta_rot=delta_rot, fwhm=fwhm_med, collapse='median',
                                                   check_memory=True, adimsdi='double', full_output=False,
                                                   verbose=verbose, nproc=nproc)
                            tmp_tmp_tmp[nr] = pca(algo_params=params_pca)
                        tmp_tmp[pp] = np.median(tmp_tmp_tmp, axis=0)
                    else:
                        params_pca = PCA_Params(cube=PCA_ASDI_cube_ori, angle_list=derot_angles, cube_ref=None,
                                               scale_list=scale_list, ncomp=(int(npc),None), svd_mode=svd_mode,
                                               scaling=None, mask_center_px=mask_IWA_px, crop_ifs=crop_ifs,
                                               delta_rot=delta_rot, fwhm=fwhm_med, collapse='median', check_memory=True,
                                               adimsdi='double', full_output=False, verbose=verbose,
                                               nproc=nproc)
                        tmp_tmp[pp] = pca(algo_params=params_pca)
                        
                write_fits(outpath+'final_PCA-SDI_image_{}_at_{}as'.format(test_pcs_str,test_rad_str)+label_test+'.fits', tmp_tmp)
                write_fits(outpath+'final_PCA-SDI_opt_npc_at_{}as'.format(test_rad_str)+label_test+'.fits', id_npc_sdi_df)           
                
                ### SNR map  
                if (not isfile(outpath+'final_PCA-SDI_snrmap_{}_at_{}as'.format(test_pcs_str,test_rad_str)+label_test+'_snrmap.fits') or overwrite_pp) and do_snr_map:
                    tmp = open_fits(outpath+'final_PCA-SDI_image_{}_at_{}as'.format(test_pcs_str,test_rad_str)+label_test+'.fits')
                    rad_in = mask_IWA # we comment it for a better visualization of the snr map (there are spurious values in the center)
                    #rad_in = 1.5
                    #tmp_tmp = np.ones_like(tmp)
                    for pp in range(tmp.shape[0]):
                        tmp[pp] = snrmap(tmp[pp], fwhm_med, plot=False, nproc=nproc)
                        tmp[pp] = mask_circle(tmp[pp],rad_in*fwhm_med)
                    write_fits(outpath+'final_PCA-SDI_snrmap_{}_at_{}as'.format(test_pcs_str,test_rad_str)+label_test+'.fits', tmp, verbose=False)   
        
        
        ####################### 5. PCA-ADI full ###########################
        if do_adi_full:
            #PCA_ADI_cube, derot_angles = vip.fits.open_adicube(outpath_5.format(bin_fac,filt,crop_lab_list[cc])+'7_final_crop_PCA_cube'+label_test+'.fits')
            # First let's readapt the number of pcs to be tested
            test_pcs_str_list = [str(x) for x in test_pcs_adi_full]
            test_pcs_str = "npc"+"-".join(test_pcs_str_list)                     
            ntest_pcs = len(test_pcs_adi_full)
            if not fake_planet:
                if planet:
                    opt_pp = np.zeros(ntest_pcs)             
                    snr_tmp = np.zeros([ntest_pcs,nz])
                    plt.close() 
                    plt.figure()
                    plt.title('SNR for '+sourcename+' b'+details+ '(PCA-ADI full)')
                    plt.ylabel('SNR')
                    plt.xlabel('npc adi')
                tmp_tmp = np.zeros([ntest_pcs, PCA_ASDI_cube.shape[2],PCA_ASDI_cube.shape[3]])
                tmp = np.zeros([ntest_pcs,nz,PCA_ASDI_cube.shape[2],PCA_ASDI_cube.shape[3]])
                for pp, npc in enumerate(test_pcs_adi_full):
                    for zz in range(nz):
                        params_pca = PCA_Params(cube=PCA_ASDI_cube[zz], angle_list=derot_angles, cube_ref=ref_cube[zz],
                                               scale_list=None, ncomp=int(npc), svd_mode=svd_mode, scaling=scaling,
                                               mask_center_px=mask_IWA_px, delta_rot=delta_rot, fwhm=fwhm,
                                               collapse='median', check_memory=True,  full_output=False,
                                               verbose=verbose, nproc=nproc)
                        tmp[pp,zz] = pca(algo_params=params_pca)
                        if planet:
                            snr_tmp[pp,zz] = snr(tmp[pp,zz], (xx_comp,yy_comp), fwhm[zz], plot=False, exclude_negative_lobes=exclude_negative_lobes,
                                                          verbose=False)
                            if zz ==0:
                                label='npc adi = {:.0f}'.format(npc)
                            else:
                                label = None
                            plt.plot(zz+1, snr_tmp[pp,zz], colors[pp%len(colors)]+'o', label=label)
                    if planet:
                        plt.legend(loc='best')
                        plt.savefig(outpath_fig+'SNR_'+source+'_PCA-ADI-full_npc{:.0f}'.format(npc)+'.pdf', format='pdf')
                        write_fits(outpath+'PCA-ADI_full_SNR_npc{:.0f}'.format(npc)+label_test+'.fits', snr_tmp)
                    write_fits(outpath+'PCA-ADI_full_npc{:.0f}'.format(npc)+label_test+'.fits', tmp[pp])  # all channels
                    tmp_tmp[pp] = np.median(tmp[pp], axis=0)  # axis zero because we sliced the pcs axis
                
                if planet:
                    opt_pp = np.argmax(snr_tmp,axis=0)
                    tmp_tmp_tmp = np.zeros([nz,PCA_ASDI_cube.shape[2],PCA_ASDI_cube.shape[3]])
                    for zz in range(nz):
                        tmp_tmp_tmp[zz] = tmp[opt_pp[zz],zz]
                    write_fits(outpath+'final_PCA-ADI_full_opt_npc_planet{}.fits'.format(label_test), tmp_tmp_tmp)
                    tmp=None
                    tmp_tmp_tmp = None
                    write_fits(outpath+'PCA-ADI_full_OPT_NPC_planet{}.fits'.format(label_test), test_pcs_adi_full[opt_pp])
                    
                    snr_tmp = np.zeros(ntest_pcs)
                    plt.close() 
                    plt.figure()
                    plt.title('SNR for '+sourcename+' b'+details+ '(PCA-ADI full - spectral channels collapsed)')
                    plt.ylabel('SNR')
                    plt.xlabel('npc')
                    for pp, npc in enumerate(test_pcs_adi_full):
                        snr_tmp[pp] = snr(tmp_tmp[pp], source_xy=(xx_comp,yy_comp), fwhm_med=fwhm_med, plot=False, exclude_negative_lobes=exclude_negative_lobes,
                                                      verbose=False)
                        if snr_tmp[pp] > 5:
                            marker = 'go'
                        elif snr_tmp[pp] > 3:
                            marker = 'bo'
                        else:
                            marker = 'ro'
                        plt.plot(npc, snr_tmp[pp], marker)
                    opt_pp = np.argmax(snr_tmp)
                    npc_opt = test_pcs_adi_full[opt_pp]
                    plt.savefig(outpath_fig+'SNR_'+source+'_PCA-ADI-full_collapsed.pdf', format='pdf')
                    write_fits(outpath+'PCA-ADI_full_collapsed_SNR{}.fits'.format(label_test), snr_tmp)
                    write_fits(outpath+'final_PCA-ADI_full_collapsed_opt_npc{:.0f}_planet{}.fits'.format(npc_opt,label_test), tmp_tmp[opt_pp])  
                write_fits(outpath+'PCA-ADI_full_collapsed_{}{}.fits'.format(test_pcs_str,label_test), tmp_tmp)    
            else:
                snr_tmp = np.zeros([nspi,ntest_pcs,nfcp])
                tmp_tmp = np.zeros([ntest_pcs,PCA_ASDI_cube.shape[2],PCA_ASDI_cube.shape[3]])
                for ns in range(nspi):
                    theta0 = th0+ns*th_step
                    PCA_ASDI_cube = open_fits(outpath+'PCA_cube'+label_test+'_fcp_spi{:.0f}.fits'.format(ns))
                    tmp = np.zeros([nz,PCA_ASDI_cube.shape[2],PCA_ASDI_cube.shape[3]])
                    for pp, npc in enumerate(test_pcs_adi_full):
                        for zz in range(nz):
                            params_pca = PCA_Params(cube=PCA_ASDI_cube[zz], angle_list=derot_angles, cube_ref=None,
                                                   scale_list=None, ncomp=int(npc), svd_mode=svd_mode, scaling=None,
                                                   mask_center_px=mask_IWA_px,crop_ifs=crop_ifs, delta_rot=delta_rot,
                                                   fwhm=fwhm_med, collapse='median', check_memory=True,
                                                   adimsdi='double', full_output=False, verbose=verbose, nproc=nproc)
                            tmp[zz] = pca(algo_params=params_pca)
                        if debug:
                            write_fits(outpath+'TMP_PCA-ADI_full_npc{:.0f}{}_fcp_spi{:.0f}.fits'.format(npc,label_test,ns), tmp)
                        tmp_tmp[pp] = np.median(tmp, axis=0)
                        for ff in range(nfcp):
                            xx_fcp = cx + rad_arr[ff]*np.cos(np.deg2rad(theta0+ff*th_step))
                            yy_fcp = cy + rad_arr[ff]*np.sin(np.deg2rad(theta0+ff*th_step))
                            snr_tmp[ns,pp,ff] = snr(tmp_tmp[pp], (xx_fcp,yy_fcp), fwhm_med, plot=False, exclude_negative_lobes=exclude_negative_lobes,
                                                                  verbose=True)   
                    write_fits(outpath+'PCA-ADI_full_collapsed_'+test_pcs_str+label_test+'_fcp_spi{:.0f}.fits'.format(ns), tmp_tmp)                                             
                snr_fcp = np.median(snr_tmp, axis=0)
                plt.close() 
                plt.figure()
                plt.title('SNR for fcps '+details+ '(PCA-ADI full)')
                plt.ylabel('SNR')
                plt.xlabel('npc')
                for ff in range(nfcp):
                    marker = all_markers[ff]
                    for pp, npc in enumerate(test_pcs_adi_full):
                        plt.plot(npc, snr_fcp[pp,ff], marker)
                plt.savefig(outpath_fig+'SNR_fcps_PCA-ADI_full'+'.pdf', format='pdf')
                write_fits(outpath+'PCA-ADI_full_SNR_fcps_'+test_pcs_str+label_test+'.fits', snr_fcp)
                
                ## Find best npc for each radius
                for ff in range(nfcp):
                    idx_best_snr = np.argmax(snr_fcp[:,ff])
                    id_npc_adi_full_df[ff] = test_pcs_adi_full[idx_best_snr]
         
                ## 7.3. Final PCA-ADI full frames with optimal npcs        
                tmp_tmp = np.zeros([nfcp,PCA_ASDI_cube_ori.shape[1],PCA_ASDI_cube_ori.shape[2]])
                test_pcs_str_list = [str(int(x)) for x in id_npc_adi_full_df]                               
                test_pcs_str = "npc_opt"+"-".join(test_pcs_str_list)
                test_rad_str_list = ["{:.1f}".format(x) for x in rad_arr*plsc]                               
                test_rad_str = "rad"+"-".join(test_rad_str_list)
                for pp, npc in enumerate(id_npc_adi_full_df):
                    for zz in range(nz):
                        params_pca = PCA_Params(cube=PCA_ASDI_cube_ori[zz], angle_list=derot_angles, cube_ref=None,
                                               scale_list=None, ncomp=(int(npc),None), svd_mode=svd_mode, scaling=None,
                                               mask_center_px=mask_IWA_px, crop_ifs=crop_ifs, delta_rot=delta_rot,
                                               fwhm=fwhm_med, collapse='median', check_memory=True, adimsdi='double',
                                               full_output=False, verbose=verbose, nproc=nproc)
                        tmp[zz] = pca(algo_params=params_pca)
                    if debug:
                        write_fits(outpath+'TMP_PCA-ADI_full_opt_npc{:.0f}{}_fcp_spi{:.0f}.fits'.format(npc,label_test,ns), tmp)
                    tmp_tmp[pp] = np.median(tmp, axis=0)            
                write_fits(outpath+'final_PCA-ADI_full_image_{}_at_{}as'.format(test_pcs_str,test_rad_str)+label_test+'.fits', tmp_tmp)
                write_fits(outpath+'final_PCA-ADI_full_opt_npc_at_{}as'.format(test_rad_str)+label_test+'.fits', id_npc_adi_full_df)           
                
                ### SNR map  
                if (not isfile(outpath+'final_PCA-ADI_full_snrmap_{}_at_{}as'.format(test_pcs_str,test_rad_str)+label_test+'.fits') or overwrite_pp) and do_snr_map:
                    tmp = open_fits(outpath+'final_PCA-ADI_full_image_{}_at_{}as'.format(test_pcs_str,test_rad_str)+label_test+'.fits')
                    rad_in = mask_IWA # we comment it for a better visualization of the snr map (there are spurious values in the center)
                    #rad_in = 1.5
                    #tmp_tmp = np.ones_like(tmp)
                    for pp in range(tmp.shape[0]):
                        tmp[pp] = snrmap(tmp[pp], fwhm_med, plot=False, nproc=nproc)
                        tmp[pp] = mask_circle(tmp[pp],rad_in*fwhm_med)
                    write_fits(outpath+'final_PCA-ADI_full_snrmap_{}_at_{}as'.format(test_pcs_str,test_rad_str)+label_test+'.fits', tmp, verbose=False)   
                
                      
        ######################## 6. PCA-ADI annular #######################
        if do_adi_ann:
            test_pcs_str_list = [str(x) for x in test_pcs_adi_ann]
            test_pcs_str = "npc"+"-".join(test_pcs_str_list)     
            ntest_pcs = len(test_pcs_adi_ann)
            if not fake_planet:
                if planet:
                    opt_pp = np.zeros(ntest_pcs)             
                    snr_tmp = np.zeros([ntest_pcs,nz])
                    plt.close() 
                    plt.figure()
                    plt.title('SNR for '+sourcename+' b'+details+ '(PCA-ADI ann)')
                    plt.ylabel('SNR')
                    plt.xlabel('npc adi')
                tmp_tmp = np.zeros([ntest_pcs, PCA_ASDI_cube.shape[2],PCA_ASDI_cube.shape[3]])
                tmp = np.zeros([ntest_pcs,nz,PCA_ASDI_cube.shape[2],PCA_ASDI_cube.shape[3]])
                for pp, npc in enumerate(test_pcs_adi_ann):
                    if not isfile(outpath+'PCA-ADI_ann_npc{:.0f}'.format(npc)+label_test_ann+'.fits') or overwrite_pp:
                        for zz in range(start_nz,nz):
                            params_ann = PCA_ANNULAR_Params(cube=PCA_ASDI_cube[zz], angle_list=derot_angles,
                                                            radius_int=mask_IWA_px, fwhm=fwhm_med, asize=asize,
                                                            delta_rot=delta_rot_ann, ncomp=int(npc), svd_mode=svd_mode,
                                                            scale_list=None, min_frames_lib=max(npc,10),
                                                            max_frames_lib=max(max_fr,npc+1), collapse='median',
                                                            full_output=False, verbose=verbose, nproc=nproc)
                            tmp[pp, zz] = pca_annular(algo_params=params_ann)
                            if planet:
                                snr_tmp[pp, zz] = snr(tmp[pp,zz], (xx_comp,yy_comp), fwhm[zz], plot=False,
                                                     exclude_negative_lobes=exclude_negative_lobes, verbose=False)
                                if zz ==0:
                                    label='npc adi = {:.0f}'.format(npc)
                                else:
                                    label = None
                                plt.plot(zz+1, snr_tmp[pp,zz], colors[pp%len(colors)]+'o', label=label)
                        if planet:
                            plt.legend(loc='best')
                            plt.savefig(outpath_fig+'SNR_'+source+'_PCA-ADI-ann_npc{:.0f}'.format(npc)+'.pdf', format='pdf')
                            write_fits(outpath+'PCA-ADI_ann_SNR_npc{:.0f}'.format(npc)+label_test_ann+'.fits', snr_tmp)
                        write_fits(outpath+'PCA-ADI_ann_npc{:.0f}'.format(npc)+label_test_ann+'.fits', tmp[pp])  # all channels
                        tmp_tmp[pp] = np.median(tmp[pp, start_nz:], axis=0)  # axis zero because we sliced the pcs axis
                
                if planet:
                    opt_pp = np.argmax(snr_tmp,axis=0)
                    opt_npc = np.zeros(nz)
                    tmp_tmp_tmp = np.zeros([nz,PCA_ASDI_cube.shape[2],PCA_ASDI_cube.shape[3]])
                    for zz in range(start_nz,nz):
                        tmp_tmp_tmp[zz] = tmp[opt_pp[zz],zz]
                        opt_npc[zz] = test_pcs_adi_ann[opt_pp[zz]]
                    write_fits(outpath+'final_PCA-ADI_ann_opt_npc_planet{}.fits'.format(label_test_ann), tmp_tmp_tmp)
                    tmp=None
                    tmp_tmp_tmp = None
                    write_fits(outpath+'PCA-ADI_ann_OPT_NPC_planet{}.fits'.format(label_test_ann), opt_npc)
                    
                    snr_tmp = np.zeros(ntest_pcs)
                    plt.close() 
                    plt.figure()
                    plt.title('SNR for '+sourcename+' b'+details+ '(PCA-ADI ann - specctral channels collapsed)')
                    plt.ylabel('SNR')
                    plt.xlabel('npc')
                    for pp, npc in enumerate(test_pcs_adi_ann):
                        snr_tmp[pp] = snr(tmp_tmp[pp], (xx_comp,yy_comp),
                                                      fwhm_med, plot=False, exclude_negative_lobes=exclude_negative_lobes,
                                                      verbose=False)
                        if snr_tmp[pp] > 5:
                            marker = 'go'
                        elif snr_tmp[pp] > 3:
                            marker = 'bo'
                        else:
                            marker = 'ro'
                        plt.plot(npc, snr_tmp[pp], marker)
                    opt_pp = np.argmax(snr_tmp)
                    npc_opt = test_pcs_adi_ann[opt_pp]
                    plt.savefig(outpath_fig+'SNR_'+source+'_PCA-ADI-ann_collapsed.pdf', format='pdf')
                    write_fits(outpath+'PCA-ADI_ann_collapsed_SNR{}.fits'.format(label_test_ann), snr_tmp)
                    write_fits(outpath+'final_PCA-ADI_ann_collapsed_opt_npc{:.0f}_planet{}.fits'.format(npc_opt,label_test_ann), tmp_tmp[opt_pp])
                write_fits(outpath+'PCA-ADI_ann_collapsed_{}{}.fits'.format(test_pcs_str,label_test_ann), tmp_tmp)
            else:
                snr_tmp = np.zeros([nspi,ntest_pcs,nfcp])
                tmp_tmp = np.zeros([ntest_pcs,PCA_ASDI_cube.shape[2],PCA_ASDI_cube.shape[3]])
                for ns in range(nspi):
                    theta0 = th0+ns*th_step
                    PCA_ASDI_cube = open_fits(outpath+'PCA_cube'+label_test+'_fcp_spi{:.0f}.fits'.format(ns))
                    tmp = np.zeros([nz,PCA_ASDI_cube.shape[2],PCA_ASDI_cube.shape[3]])
                    for pp, npc in enumerate(test_pcs_adi_ann):
                        for zz in range(start_nz,nz):
                            params_ann = PCA_ANNULAR_Params(cube=PCA_ASDI_cube[zz], angle_list=derot_angles,
                                                            radius_int=mask_IWA_px, fwhm=fwhm_med, asize=asize,
                                                            delta_rot=delta_rot_ann, ncomp=int(npc), svd_mode=svd_mode,
                                                            scale_list=None, min_frames_lib=max(npc,10),
                                                            max_frames_lib=max(max_fr,npc+1), collapse='median',
                                                            full_output=False, verbose=verbose, nproc=nproc)
                            tmp[zz] = pca_annular(algo_params=params_ann)
                        if debug:
                            write_fits(outpath+'TMP_PCA-ADI_ann_npc{:.0f}{}_fcp_spi{:.0f}.fits'.format(npc,label_test_ann,ns), tmp)
                        tmp_tmp[pp] = np.median(tmp[start_nz:], axis=0)
                        for ff in range(nfcp):
                            xx_fcp = cx + rad_arr[ff]*np.cos(np.deg2rad(theta0+ff*th_step))
                            yy_fcp = cy + rad_arr[ff]*np.sin(np.deg2rad(theta0+ff*th_step))
                            snr_tmp[ns,pp,ff] = snr(tmp_tmp[pp], (xx_fcp,yy_fcp), fwhm_med, plot=False, exclude_negative_lobes=exclude_negative_lobes,
                                                                  verbose=True)   
                    write_fits(outpath+'PCA-ADI_ann_collapsed_'+test_pcs_str+label_test_ann+'_fcp_spi{:.0f}.fits'.format(ns), tmp_tmp)
                snr_fcp = np.median(snr_tmp, axis=0)
                plt.close() 
                plt.figure()
                plt.title('SNR for fcps '+details+ '(PCA-ADI ann)')
                plt.ylabel('SNR')
                plt.xlabel('npc')
                for ff in range(nfcp):
                    marker = all_markers[ff]
                    for pp, npc in enumerate(test_pcs_adi_ann):
                        plt.plot(npc, snr_fcp[pp,ff], marker)
                plt.savefig(outpath_fig+'SNR_fcps_PCA-ADI_ann'+'.pdf', format='pdf')
                write_fits(outpath+'PCA-ADI_ann_SNR_fcps_'+test_pcs_str+label_test_ann+'.fits', snr_fcp)
                
                ## Find best npc for each radius
                for ff in range(nfcp):
                    idx_best_snr = np.argmax(snr_fcp[:,ff])
                    id_npc_adi_ann_df[ff] = test_pcs_adi_ann[idx_best_snr]
         
                ## 7.3. Final PCA-ADI full frames with optimal npcs        
                tmp_tmp = np.zeros([nfcp,PCA_ASDI_cube_ori.shape[1],PCA_ASDI_cube_ori.shape[2]])
                test_pcs_str_list = [str(int(x)) for x in id_npc_adi_ann_df]                               
                test_pcs_str = "npc_opt"+"-".join(test_pcs_str_list)
                test_rad_str_list = ["{:.1f}".format(x) for x in rad_arr*plsc]                               
                test_rad_str = "rad"+"-".join(test_rad_str_list)
                for pp, npc in enumerate(id_npc_adi_ann_df):
                    for zz in range(start_nz,nz):
                        params_ann = PCA_ANNULAR_Params(cube=PCA_ASDI_cube_ori[zz], angle_list=derot_angles,
                                                        radius_int=mask_IWA_px, fwhm=fwhm_med, asize=asize,
                                                        delta_rot=delta_rot_ann, ncomp=int(npc), svd_mode=svd_mode,
                                                        scale_list=None, min_frames_lib=max(npc,10),
                                                        max_frames_lib=max(max_fr,npc+1), collapse='median',
                                                        full_output=False, verbose=verbose, nproc=nproc)
                        tmp[zz] = pca_annular(algo_params=params_ann)
                    if debug:
                        write_fits(outpath+'TMP_PCA-ADI_ann_opt_npc{:.0f}{}_fcp_spi{:.0f}.fits'.format(npc,label_test_ann,ns), tmp)
                    tmp_tmp[pp] = np.median(tmp[start_nz:], axis=0)            
                write_fits(outpath+'final_PCA-ADI_ann_image_{}_at_{}as'.format(test_pcs_str,test_rad_str)+label_test_ann+'.fits', tmp_tmp)
                write_fits(outpath+'final_PCA-ADI_ann_opt_npc_at_{}as'.format(test_rad_str)+label_test_ann+'.fits', id_npc_adi_ann_df)
                
                ### SNR map  
                if (not isfile(outpath+'final_PCA-ADI_ann_snrmap_{}_at_{}as'.format(test_pcs_str,test_rad_str)+label_test_ann+'.fits') or overwrite_pp) and do_snr_map:
                    tmp = open_fits(outpath+'final_PCA-ADI_ann_image_{}_at_{}as'.format(test_pcs_str,test_rad_str)+label_test_ann+'.fits')
                    rad_in = mask_IWA # we comment it for a better visualization of the snr map (there are spurious values in the center)
                    #rad_in = 1.5
                    #tmp_tmp = np.ones_like(tmp)
                    for pp in range(tmp.shape[0]):
                        tmp[pp] = snrmap(tmp[pp], fwhm_med, plot=False, nproc=nproc)
                        tmp[pp] = mask_circle(tmp[pp],rad_in*fwhm_med)
                    write_fits(outpath+'final_PCA-ADI_ann_snrmap_{}_at_{}as'.format(test_pcs_str,test_rad_str)+label_test_ann+'.fits', tmp, verbose=False)
        
        
        label_test_ori = label_test
        ###################### 7. PCA-SADI full (single step) #########################
        if do_sadi_full and sadi_steps==1:
            test_pcs_str_list = [str(x) for x in test_pcs_sadi_full]
            ntest_pcs = len(test_pcs_sadi_full)
            test_pcs_str = "npc"+"-".join(test_pcs_str_list)
            if mask_PCA is None:
                mask_rdi = None
            else:
                mask_tmp = np.ones_like(PCA_ASDI_cube[0,0])
                mask_rdi = mask_circle(mask_tmp, mask_PCA, fillwith=0, mode='in')
            for ss, scal in enumerate(scalings):
                if scal is None:
                    scal_lab = ''
                else:
                    scal_lab = scal
                for ii, ifs_collapse_range in enumerate(ifs_collapse_range_list):
                    label_test = label_test_ori+'_'+scal_lab+'_'+ifs_collapse_range_lab[ii]           
                    if not fake_planet:     
                        tmp = np.zeros([ntest_pcs,PCA_ASDI_cube.shape[2],PCA_ASDI_cube.shape[3]])
                        if planet:                
                            snr_tmp = np.zeros(ntest_pcs)
                        if do_stim_map:
                            stim_map = np.zeros([ntest_pcs,PCA_ASDI_cube.shape[2],PCA_ASDI_cube.shape[3]])
                            inv_stim_map = np.zeros([ntest_pcs,PCA_ASDI_cube.shape[2],PCA_ASDI_cube.shape[3]])
                            thr = np.zeros(ntest_pcs)
                        for pp, npc in enumerate(test_pcs_sadi_full):
                            params_pca = PCA_Params(cube=PCA_ASDI_cube, angle_list=derot_angles, cube_ref=None,
                                                   scale_list=scale_list, ncomp=int(npc), svd_mode=svd_mode,
                                                   scaling=scal, mask_center_px=mask_IWA_px, adimsdi='single',
                                                   delta_rot=delta_rot, fwhm=fwhm_med, collapse='median',
                                                   check_memory=True, ifs_collapse_range=ifs_collapse_range,
                                                   full_output=True, verbose=verbose, mask_rdi=mask_rdi,
                                                   nproc=nproc)
                            tmp[pp], _, tmp_tmp = pca(algo_params=params_pca)
                            if debug:
                                write_fits(outpath+'TMP_final_PCA-SADI1_full_npc{:.0f}_'.format(npc)+test_pcs_str+label_test+'.fits', tmp_tmp)
                            if do_stim_map:
                                tmp_tmp_der = cube_derotate(tmp_tmp, derot_angles, imlib='opencv', nproc=nproc)
                                stim_map[pp] = compute_stim_map(tmp_tmp_der)
                                inv_stim_map[pp] = compute_inverse_stim_map(tmp_tmp,derot_angles, nproc=nproc)
                                thr[pp] = np.amax(inv_stim_map[pp])
                        if planet:
                            snr_tmp[pp] = snr(tmp[pp], (xx_comp,yy_comp), fwhm_med, plot=False, exclude_negative_lobes=exclude_negative_lobes,
                                                          verbose=False)                  
                        if planet:
                            plt.close() 
                            plt.figure()
                            plt.title('SNR for '+sourcename+' b'+details+ '(PCA-SADI1 full-frame)')
                            plt.ylabel('SNR')
                            plt.xlabel('npc')  
                            for pp, npc in enumerate(test_pcs_sadi_full):
                                if snr_tmp[pp] > 5:
                                    marker = 'go'
                                elif snr_tmp[pp] > 3:
                                    marker = 'bo'
                                else:
                                    marker = 'ro'
                                plt.plot(npc, snr_tmp[pp], marker)   
                            try:
                                plt.savefig(outpath_fig+'SNR_'+source+'_PCA-SADI1-full'+'.pdf', format='pdf')
                            except:
                                pass
                            write_fits(outpath+'final_PCA-SADI1_full_SNR_'+test_pcs_str+label_test+'.fits', snr_tmp)
                        write_fits(outpath+'final_PCA-SADI1_full_'+test_pcs_str+label_test+'.fits', tmp)
                        if do_stim_map:
                            write_fits(outpath+'final_PCA-SADI1_full_'+test_pcs_str+label_test+'_stimmap.fits', stim_map)
                            write_fits(outpath+'final_PCA-SADI1_full_'+test_pcs_str+label_test+'_invstimmap.fits', inv_stim_map)
                            write_fits(outpath+'final_PCA-SADI1_full_'+test_pcs_str+label_test+'_stimthr.fits', thr)
                        ### SNR map  
                        if (not isfile(outpath+'final_PCA-SADI1_full_'+test_pcs_str+label_test+'_snrmap.fits') or overwrite_pp) and do_snr_map:
                            tmp = open_fits(outpath+'final_PCA-SADI1_full_'+test_pcs_str+label_test+'.fits')
                            rad_in = mask_IWA # we comment it for a better visualization of the snr map (there are spurious values in the center)
                            #rad_in = 1.5
                            tmp_tmp = np.ones_like(tmp)
                            tmp_tmp = mask_circle(tmp_tmp,rad_in*fwhm_med)
                            for pp in range(tmp.shape[0]):
                                tmp[pp] = snrmap(tmp[pp], fwhm_med, plot=False, nproc=nproc)
                            write_fits(outpath+'final_PCA-SADI1_full_'+test_pcs_str+label_test+'_snrmap.fits', tmp, verbose=False)  
                    else:
                        snr_tmp_tmp = np.zeros([nspi,ntest_pcs,nfcp])
                        tmp_tmp = np.zeros([ntest_pcs,PCA_ASDI_cube.shape[2],PCA_ASDI_cube.shape[3]])
                        for ns in range(nspi):
                            theta0 = th0+ns*th_step
                            PCA_ASDI_cube = open_fits(outpath+'PCA_cube'+label_test+'_fcp_spi{:.0f}.fits'.format(ns))
                            for pp, npc in enumerate(test_pcs_sadi_full):
                                if svd_mode == 'randsvd':
                                    tmp_tmp_tmp = np.zeros([3,PCA_ASDI_cube.shape[2],PCA_ASDI_cube.shape[3]])
                                    for nr in range(3):
                                        params_pca = PCA_Params(cube=PCA_ASDI_cube, angle_list=derot_angles,
                                                               cube_ref=None, scale_list=scale_list, ncomp=int(npc),
                                                               svd_mode=svd_mode, scaling=None,
                                                               mask_center_px=mask_IWA_px, crop_ifs=crop_ifs,
                                                               ifs_collapse_range=ifs_collapse_range,
                                                               delta_rot=1, fwhm=fwhm_med, collapse='median',
                                                               check_memory=True, adimsdi='single', full_output=False,
                                                               verbose=verbose, mask_rdi=mask_rdi,
                                                               nproc=nproc)
                                        tmp_tmp_tmp[nr] = pca(algo_params=params_pca)
                                    tmp_tmp[pp] = np.median(tmp_tmp_tmp, axis=0)
                                else:
                                    params_pca = PCA_Params(cube=PCA_ASDI_cube, angle_list=derot_angles, cube_ref=None,
                                                           scale_list=scale_list, ncomp=int(npc), svd_mode=svd_mode,
                                                           scaling=None, mask_center_px=mask_IWA_px, adimsdi='single',
                                                           delta_rot=1, fwhm=fwhm_med, collapse='median',
                                                           check_memory=True, ifs_collapse_range=ifs_collapse_range,
                                                           full_output=False, verbose=verbose, mask_rdi=mask_rdi,
                                                           nproc=nproc)
                                    tmp_tmp[pp] = pca(algo_params=params_pca)
                                for ff in range(nfcp):
                                    xx_fcp = cx + rad_arr[ff]*np.cos(np.deg2rad(theta0+ff*th_step))
                                    yy_fcp = cy + rad_arr[ff]*np.sin(np.deg2rad(theta0+ff*th_step))
                                    snr_tmp_tmp[ns,pp,ff] = snr(tmp_tmp[pp], (xx_fcp,yy_fcp), fwhm_med, plot=False, exclude_negative_lobes=exclude_negative_lobes,
                                                                          verbose=True)   
                            write_fits(outpath+'TMP_PCA-SADI1_full_'+test_pcs_str+label_test+'_fcp_spi{:.0f}.fits'.format(ns), tmp_tmp)                                             
                        snr_fcp = np.median(snr_tmp_tmp, axis=0)
                        plt.close() 
                        plt.figure()
                        plt.title('SNR for fcps '+details+ '(PCA-SADI1 full-frame)')
                        plt.ylabel('SNR')
                        plt.xlabel('npc')
                        for ff in range(nfcp):
                            marker = all_markers[ff]
                            for pp, npc in enumerate(test_pcs_sadi_full):
                                plt.plot(npc, snr_fcp[pp,ff], marker)
                        try:
                            plt.savefig(outpath_fig+'SNR_fcps_PCA-SADI1-full'+'.pdf', format='pdf')
                        except:
                            pass
                        write_fits(outpath+'final_PCA-SADI1_full_SNR_fcps_'+test_pcs_str+label_test+'.fits', snr_fcp)   
                        
                        ## Find best npc for each radius
                        for ff in range(nfcp):
                            idx_best_snr = np.argmax(snr_fcp[:,ff])
                            id_npc_full_df[ff] = test_pcs_sadi_full[idx_best_snr]
                 
                        
                        ## 7.3. Final PCA-ADI frames with optimal npcs        
                        tmp_tmp = np.zeros([nfcp,PCA_ASDI_cube_ori.shape[1],PCA_ASDI_cube_ori.shape[2]])
                        test_pcs_str_list = [str(int(x)) for x in id_npc_full_df]                               
                        test_pcs_str = "npc_opt"+"-".join(test_pcs_str_list)
                        test_rad_str_list = ["{:.1f}".format(x) for x in rad_arr*plsc]                               
                        test_rad_str = "rad"+"-".join(test_rad_str_list)
                        for pp, npc in enumerate(id_npc_full_df):
                            params_pca = PCA_Params(cube=PCA_ASDI_cube_ori, angle_list=derot_angles, cube_ref=None,
                                                   scale_list=scale_list, ncomp=int(npc), svd_mode=svd_mode,
                                                   scaling=None, mask_center_px=mask_IWA_px, crop_ifs=crop_ifs,
                                                   ifs_collapse_range=ifs_collapse_range, delta_rot=1, fwhm=fwhm_med,
                                                   collapse='median', check_memory=True, adimsdi='single',
                                                   full_output=False, verbose=verbose, mask_rdi=mask_rdi,
                                                   nproc=nproc)
                            tmp_tmp[pp] = pca(algo_params=params_pca)
                                              
                        write_fits(outpath+'final_PCA-SADI1_full_image_{}_at_{}as'.format(test_pcs_str,test_rad_str)+label_test+'.fits', tmp_tmp)
                        write_fits(outpath+'final_PCA-SADI1_full_opt_npc_at_{}as'.format(test_rad_str)+label_test+'.fits', id_npc_full_df)           
                        
                        ### SNR map  
                        if (not isfile(outpath+'final_PCA-SADI1_full_snrmap_{}_at_{}as'.format(test_pcs_str,test_rad_str)+label_test+'.fits') or overwrite_pp) and do_snr_map:
                            tmp = open_fits(outpath+'final_PCA-SADI1_full_image_{}_at_{}as'.format(test_pcs_str,test_rad_str)+label_test+'.fits')
                            rad_in = mask_IWA # we comment it for a better visualization of the snr map (there are spurious values in the center)
                            #rad_in = 1.5
                            tmp_tmp = np.ones_like(tmp)
                            tmp_tmp = mask_circle(tmp_tmp,rad_in*fwhm_med)
                            for pp in range(tmp.shape[0]):
                                tmp[pp] = snrmap(tmp[pp], fwhm_med, plot=False, nproc=nproc)
                            write_fits(outpath+'final_PCA-SADI1_full_snrmap_{}_at_{}as'.format(test_pcs_str,test_rad_str)+label_test+'.fits', tmp, verbose=False)
        
            label_test = label_test_ori
                    
                    
        ####################### 8. PCA-SADI full (2 steps) ###########################
        if do_sadi_full and sadi_steps==2:
            test_pcs_str_list_sdi = [str(x) for x in test_pcs_sadi_full_sdi]
            ntest_pcs_sdi = len(test_pcs_sadi_full_sdi)
            test_pcs_str_sdi = "npc_sdi"+"-".join(test_pcs_str_list_sdi)
            test_pcs_str_list_adi = [str(x) for x in test_pcs_sadi_full_adi]
            ntest_pcs_adi = len(test_pcs_sadi_full_adi)
            test_pcs_str_adi = "_npc_adi"+"-".join(test_pcs_str_list_adi)
        
            ntest_pcs = int(ntest_pcs_sdi*ntest_pcs_adi)
            test_pcs_str = test_pcs_str_sdi+test_pcs_str_adi
            
            if mask_PCA is None:
                mask_rdi = None
            else:
                mask_tmp = np.ones_like(PCA_ASDI_cube[0,0])
                mask_rdi = mask_circle(mask_tmp, mask_PCA, fillwith=0, mode='in')
                
            for ss, scal in enumerate(scalings):
                if scal is None:
                    scal_lab = ''
                else:
                    scal_lab = scal
                for ii, ifs_collapse_range in enumerate(ifs_collapse_range_list):
                    label_test = label_test_ori+'_'+scal_lab+'_'+ifs_collapse_range_lab[ii]
                    if not fake_planet:     
                        tmp = np.zeros([ntest_pcs,PCA_ASDI_cube.shape[2],PCA_ASDI_cube.shape[3]])
                        if planet:                
                            snr_tmp = np.zeros(ntest_pcs)
                        if do_stim_map:
                            stim_map = np.zeros([ntest_pcs,PCA_ASDI_cube.shape[2],PCA_ASDI_cube.shape[3]])
                            inv_stim_map = np.zeros([ntest_pcs,PCA_ASDI_cube.shape[2],PCA_ASDI_cube.shape[3]])
                            thr = np.zeros(ntest_pcs)
                        counter = 0
                        for pp1, npc1 in enumerate(test_pcs_sadi_full_sdi):
                            for pp2, npc2 in enumerate(test_pcs_sadi_full_adi):
                                params_pca = PCA_Params(cube=PCA_ASDI_cube, angle_list=derot_angles, cube_ref=None,
                                                       scale_list=scale_list, ncomp=(int(npc1),int(npc2)),
                                                       svd_mode=svd_mode, scaling=scal, mask_center_px=mask_IWA_px,
                                                       adimsdi='double', delta_rot=1, fwhm=fwhm_med, collapse='median',
                                                       check_memory=True, ifs_collapse_range=ifs_collapse_range,
                                                       full_output=True, verbose=verbose, mask_rdi=mask_rdi,
                                                       nproc=nproc)
                                tmp[counter], tmp_tmp, tmp_tmp_der = pca(algo_params=params_pca)
                                if do_stim_map:
                                    stim_map[counter] = compute_stim_map(tmp_tmp_der)
                                    inv_stim_map[counter] = compute_inverse_stim_map(tmp_tmp,derot_angles, nproc=nproc)
                                    thr[counter] = np.amax(inv_stim_map[counter])                       
                
                                if planet:  
                                    snr_tmp[counter] = snr(tmp_tmp[counter], (xx_comp,yy_comp), fwhm_med, plot=False,
                                                           exclude_negative_lobes=exclude_negative_lobes, verbose=False)
                                counter+=1                                          
                        if planet:
                            plt.close() 
                            plt.figure()
                            plt.title('SNR for '+sourcename+' b'+details+ '(PCA-SADI2 full)')
                            plt.ylabel('SNR')
                            plt.xlabel('npc adi')  
                            counter = 0
                            for pp1, npc1 in enumerate(test_pcs_sadi_full_sdi):
                                for pp2, npc2 in enumerate(test_pcs_sadi_full_adi):
                                    if snr_tmp[counter] > 5:
                                        marker = 'go'
                                    elif snr_tmp[counter] > 3:
                                        marker = 'bo'
                                    else:
                                        marker = 'ro'
                                    if pp2 ==0:
                                        label='npc sdi = {:.0f}'.format(npc1)
                                    else:
                                        label = None
                                    plt.plot(npc2, snr_tmp[counter], colors[pp1%len(colors)]+'o', label=label)
                                    counter+=1
                            plt.legend(loc='best')
                            plt.savefig(outpath_fig+'SNR_'+source+'_PCA-SADI2-full'+'.pdf', format='pdf')
                            write_fits(outpath+'final_PCA-SADI2_full_SNR_'+test_pcs_str+label_test+'.fits', snr_tmp)
                        write_fits(outpath+'final_PCA-SADI2_full_'+test_pcs_str+label_test+'.fits', tmp)
                        if do_stim_map:
                            write_fits(outpath+'final_PCA-SADI2_full_'+test_pcs_str+label_test+'_stimmap.fits', stim_map)
                            write_fits(outpath+'final_PCA-SADI2_full_'+test_pcs_str+label_test+'_invstimmap.fits', inv_stim_map)
                            write_fits(outpath+'final_PCA-SADI2_full_'+test_pcs_str+label_test+'_stimthr.fits', thr)
                    else:
                        snr_tmp_tmp = np.zeros([nspi,ntest_pcs,nfcp])
                        tmp_tmp = np.zeros([ntest_pcs,PCA_ASDI_cube.shape[1],PCA_ASDI_cube.shape[2]])
                        for ns in range(nspi):
                            theta0 = th0+ns*th_step
                            PCA_ASDI_cube = open_fits(outpath+'PCA_cube'+label_test+'_fcp_spi{:.0f}.fits'.format(ns))
                            counter = 0
                            for pp1, npc1 in enumerate(test_pcs_sadi_full_sdi):
                                for pp2, npc2 in enumerate(test_pcs_sadi_full_adi):
                                    params_pca = PCA_Params(cube=PCA_ASDI_cube, angle_list=derot_angles, cube_ref=None,
                                                           scale_list=scale_list, ncomp=(int(npc1),int(npc2)),
                                                           svd_mode=svd_mode, scaling=scal, mask_center_px=mask_IWA_px,
                                                           adimsdi='double', delta_rot=1, fwhm=fwhm_med,
                                                           collapse='median', check_memory=True,
                                                           ifs_collapse_range=ifs_collapse_range,  full_output=False,
                                                           verbose=verbose, mask_rdi=mask_rdi, nproc=nproc)
                                    tmp_tmp[counter] = pca(algo_params=params_pca)
                                    for ff in range(nfcp):
                                        xx_fcp = cx + rad_arr[ff]*np.cos(np.deg2rad(theta0+ff*th_step))
                                        yy_fcp = cy + rad_arr[ff]*np.sin(np.deg2rad(theta0+ff*th_step))
                                        snr_tmp_tmp[ns,pp,ff] = snr(tmp_tmp[counter], (xx_fcp,yy_fcp), fwhm_med, plot=False, exclude_negative_lobes=exclude_negative_lobes,
                                                                              verbose=True)   
                                    counter +=1
                            write_fits(outpath+'TMP_PCA-SADI2_full_'+test_pcs_str+label_test+'_fcp_spi{:.0f}.fits'.format(ns), tmp_tmp)                                             
                        snr_fcp = np.median(snr_tmp_tmp, axis=0)
                        plt.close() 
                        plt.figure()
                        plt.title('SNR for fcps '+details+ '(PCA-SADI2 full)')
                        plt.ylabel('SNR')
                        plt.xlabel('npc')
                        for ff in range(nfcp):
                            marker = all_markers_shape[ff]
                            counter = 0
                            for pp1, npc1 in enumerate(test_pcs_sadi_full_sdi):
                                for pp2, npc2 in enumerate(test_pcs_sadi_full_adi):
                                    if pp2 == 0:
                                        label = 'fcp {:.0f} - npc sdi = {:.0f}'.format(ff+1,npc1)
                                    else:
                                        label = None
                                    plt.plot(npc2, snr_fcp[counter,ff], colors[pp1%len(colors)]+marker,label = label)
                                    counter+=1
                        plt.savefig(outpath_fig+'SNR_fcps_PCA-SADI2-full'+'.pdf', format='pdf')
                
                        write_fits(outpath+'final_PCA-SADI2_full_SNR_fcps_'+test_pcs_str+label_test+'.fits', snr_fcp)   
                        
                        ## Find best npc for each radius
                        for ff in range(nfcp):
                            idx_best_snr = np.argmax(snr_fcp[:,ff])
                            id_npc_full_df[ff,0] = test_pcs_sadi_full_sdi[int(idx_best_snr/ntest_pcs_adi)]
                            id_npc_full_df[ff,1] = test_pcs_sadi_full_adi[int(idx_best_snr%ntest_pcs_adi)]
                        
                        ## 8.3. Final PCA-ADI frames with optimal npcs        
                        tmp_tmp = np.zeros([nfcp,PCA_ASDI_cube_ori.shape[1],PCA_ASDI_cube_ori.shape[2]])
                        test_pcs_str_list_sdi = [str(x) for x in test_pcs_sadi_full_sdi]
                        ntest_pcs_sdi = len(test_pcs_sadi_full_sdi)                
                        test_pcs_str_sdi = "npc_sdi"+"-".join(test_pcs_str_list_sdi)
                        test_pcs_str_list_adi = [str(x) for x in test_pcs_sadi_full_adi]
                        ntest_pcs_adi = len(test_pcs_sadi_full_adi)
                        test_pcs_str_adi = "_npc_adi"+"-".join(test_pcs_str_list_adi)
                        ntest_pcs = int(ntest_pcs_sdi*ntest_pcs_adi)
                        test_pcs_str = test_pcs_str_sdi+test_pcs_str_adi
                    
                        test_pcs_str_list_sdi = [str(int(x)) for x in id_npc_full_df[:,0]]                               
                        test_pcs_str_sdi = "npc_sdi_opt"+"-".join(test_pcs_str_list_sdi)
                        test_pcs_str_list_adi = [str(int(x)) for x in id_npc_full_df[:,1]]                               
                        test_pcs_str_adi = "npc_adi_opt"+"-".join(test_pcs_str_list_adi)
                        
                        test_rad_str_list = ["{:.1f}".format(x) for x in rad_arr*plsc]                               
                        test_rad_str = "rad"+"-".join(test_rad_str_list)
                        for pp, idx_npc_full in enumerate(id_npc_full_df):
                            params_pca = PCA_Params(cube=PCA_ASDI_cube_ori, angle_list=derot_angles, cube_ref=None,
                                                   scale_list=scale_list, ncomp=(int(npc1), int(npc2)),
                                                   svd_mode=svd_mode, scaling=scal, ifs_collapse_range=ifs_collapse_range,
                                                   mask_center_px=mask_IWA_px, adimsdi='double', delta_rot=1,
                                                   fwhm=fwhm_med, collapse='median', check_memory=True,
                                                   full_output=False, verbose=verbose, mask_rdi=mask_rdi, nproc=nproc)
                            tmp_tmp[pp] = pca(algo_params=params_pca)
                        write_fits(outpath+'final_PCA-SADI2_full_image_{}_at_{}as'.format(test_pcs_str,test_rad_str)+label_test+'.fits', tmp_tmp)
                        write_fits(outpath+'final_PCA-SADI2_full_opt_npc_at_{}as'.format(test_rad_str)+label_test+'.fits', id_npc_full_df)
                
                    
                        ### SNR map  
                        if (not isfile(outpath+'final_PCA-SADI2_full_snrmap_{}_at_{}as'.format(test_pcs_str,test_rad_str)+label_test+'.fits') or overwrite_pp) and do_snr_map:
                            tmp = open_fits(outpath+'final_PCA-SADI2_image_full_{}_at_{}as'.format(test_pcs_str,test_rad_str)+label_test+'.fits')
                            rad_in = mask_IWA # we comment it for a better visualization of the snr map (there are spurious values in the center)
                            #rad_in = 1.5
                            tmp_tmp = np.ones_like(tmp)
                            tmp_tmp = mask_circle(tmp_tmp,rad_in*fwhm_med)
                            for pp in range(tmp.shape[0]):
                                tmp[pp] = snrmap(tmp[pp], fwhm_med, plot=False, nproc=nproc)
                            write_fits(outpath+'final_PCA-SADI2_full_snrmap_{}_at_{}as'.format(test_pcs_str,test_rad_str)+label_test+'.fits', tmp, verbose=False)                     
            label_test = label_test_ori
        
                    
                    
        ###################### 9. PCA-SADI annular (2 steps) ##########################
        if do_sadi_ann and sadi_steps==2:
            test_pcs_str_list_sdi = [str(x) for x in test_pcs_sadi_ann_sdi]
            ntest_pcs_sdi = len(test_pcs_sadi_ann_sdi)
            test_pcs_str_sdi = "npc_sdi"+"-".join(test_pcs_str_list_sdi)
            test_pcs_str_list_adi = [str(x) for x in test_pcs_sadi_ann_adi]
            ntest_pcs_adi = len(test_pcs_sadi_ann_adi)
            test_pcs_str_adi = "_npc_adi"+"-".join(test_pcs_str_list_adi)
        
            ntest_pcs = int(ntest_pcs_sdi*ntest_pcs_adi)
            test_pcs_str = test_pcs_str_sdi+test_pcs_str_adi
            for ss, scal in enumerate(scalings):
                if scal is None:
                    scal_lab = ''
                else:
                    scal_lab = scal
                for ii, ifs_collapse_range in enumerate(ifs_collapse_range_list):
                    label_test_ann_ori = label_test_ann
                    label_test_ann = label_test_ann_ori+'_'+scal_lab+'_'+ifs_collapse_range_lab[ii]
                    
                    if not fake_planet:
                        tmp = np.zeros([ntest_pcs,PCA_ASDI_cube.shape[2],PCA_ASDI_cube.shape[3]])
                        if planet:                
                            snr_tmp = np.zeros(ntest_pcs)
                        if do_stim_map:
                            stim_map = np.zeros([ntest_pcs,PCA_ASDI_cube.shape[2],PCA_ASDI_cube.shape[3]])
                            inv_stim_map = np.zeros([ntest_pcs,PCA_ASDI_cube.shape[2],PCA_ASDI_cube.shape[3]])
                            thr = np.zeros(ntest_pcs)
                        counter = 0
                        for pp1, npc1 in enumerate(test_pcs_sadi_ann_sdi):
                            for pp2, npc2 in enumerate(test_pcs_sadi_ann_adi):
                                params_ann = PCA_ANNULAR_Params(cube=PCA_ASDI_cube, angle_list=derot_angles,
                                                                radius_int=mask_IWA_px, fwhm=fwhm_med, asize=asize,
                                                                delta_rot=delta_rot_ann, ncomp=(int(npc1),int(npc2)),
                                                                svd_mode=svd_mode, ifs_collapse_range=ifs_collapse_range,
                                                                scaling=scal, scale_list=scale_list,
                                                                min_frames_lib=max(npc1,npc2,10),
                                                                max_frames_lib=max(max_fr,npc1+1,npc2+1), collapse='median',
                                                                full_output=True, verbose=verbose, nproc=nproc)
                                tmp_tmp, tmp_tmp_der, tmp[counter] = pca_annular(algo_params=params_ann)
                                if do_stim_map:
                                    stim_map[counter] = compute_stim_map(tmp_tmp_der)
                                    inv_stim_map[counter] = compute_inverse_stim_map(tmp_tmp,derot_angles, nproc=nproc)
                                    thr[counter] = np.amax(inv_stim_map[counter])
                
                                if planet:  
                                    snr_tmp[counter] = snr(tmp_tmp[counter], (xx_comp,yy_comp), fwhm_med, plot=False, exclude_negative_lobes=exclude_negative_lobes,
                                                                       verbose=False)
                                counter+=1                                          
                        if planet:
                            plt.close() 
                            plt.figure()
                            plt.title('SNR for '+sourcename+' b'+details+ '(PCA-SADI2 ann)')
                            plt.ylabel('SNR')
                            plt.xlabel('npc adi')  
                            counter = 0
                            for pp1, npc1 in enumerate(test_pcs_sadi_ann_sdi):
                                for pp2, npc2 in enumerate(test_pcs_sadi_ann_adi):
                                    if snr_tmp[counter] > 5:
                                        marker = 'go'
                                    elif snr_tmp[counter] > 3:
                                        marker = 'bo'
                                    else:
                                        marker = 'ro'
                                    if pp2 ==0:
                                        label='npc sdi = {:.0f}'.format(npc1)
                                    else:
                                        label = None
                                    plt.plot(npc2, snr_tmp[counter], colors[pp1%len(colors)]+'o', label=label)
                                    counter+=1
                            plt.legend(loc='best')
                            plt.savefig(outpath_fig+'SNR_'+source+'_PCA-SADI2-ann'+'.pdf', format='pdf')
                            write_fits(outpath+'final_PCA-SADI2_ann_SNR_'+test_pcs_str+label_test_ann+'.fits', snr_tmp)
                        write_fits(outpath+'final_PCA-SADI2_ann_'+test_pcs_str+label_test_ann+'.fits', tmp)
                        if do_stim_map:
                            write_fits(outpath+'final_PCA-SADI2_ann_'+test_pcs_str+label_test_ann+'_stimmap.fits', stim_map)
                            write_fits(outpath+'final_PCA-SADI2_ann_'+test_pcs_str+label_test_ann+'_invstimmap.fits', inv_stim_map)
                            write_fits(outpath+'final_PCA-SADI2_ann_'+test_pcs_str+label_test_ann+'_stimthr.fits', thr)
                    else:
                        snr_tmp_tmp = np.zeros([nspi,ntest_pcs,nfcp])
                        tmp_tmp = np.zeros([ntest_pcs,PCA_ASDI_cube.shape[1],PCA_ASDI_cube.shape[2]])
                        for ns in range(nspi):
                            theta0 = th0+ns*th_step
                            PCA_ASDI_cube = open_fits(outpath+'PCA_cube'+label_test+'_fcp_spi{:.0f}.fits'.format(ns))
                            counter = 0
                            for pp1, npc1 in enumerate(test_pcs_sadi_ann_sdi):
                                for pp2, npc2 in enumerate(test_pcs_sadi_ann_adi):
                                    params_ann = PCA_ANNULAR_Params(cube=PCA_ASDI_cube, angle_list=derot_angles,
                                                                    radius_int=mask_IWA_px, fwhm=fwhm_med, asize=asize,
                                                                    delta_rot=delta_rot_ann, ncomp=(int(npc1),int(npc2)),
                                                                    svd_mode=svd_mode, scale_list=scale_list,
                                                                    ifs_collapse_range=ifs_collapse_range, scaling=scal,
                                                                    min_frames_lib=max(npc1,npc2,10),
                                                                    max_frames_lib=max(max_fr,npc1+1,npc2+1),
                                                                    collapse='median', full_output=False, verbose=verbose,
                                                                    nproc=nproc)
                                    tmp_tmp[counter] = pca_annular(algo_params=params_ann)
                                    for ff in range(nfcp):
                                        xx_fcp = cx + rad_arr[ff]*np.cos(np.deg2rad(theta0+ff*th_step))
                                        yy_fcp = cy + rad_arr[ff]*np.sin(np.deg2rad(theta0+ff*th_step))
                                        snr_tmp_tmp[ns,pp,ff] = snr(tmp_tmp[counter], (xx_fcp,yy_fcp), fwhm_med, plot=False, exclude_negative_lobes=exclude_negative_lobes,
                                                                              verbose=True)   
                                    counter +=1
                            write_fits(outpath+'TMP_PCA-SADI2_ann_'+test_pcs_str+label_test_ann+'_fcp_spi{:.0f}.fits'.format(ns), tmp_tmp)
                        snr_fcp = np.median(snr_tmp_tmp, axis=0)
                        plt.close() 
                        plt.figure()
                        plt.title('SNR for fcps '+details+ '(PCA-SADI ann)')
                        plt.ylabel('SNR')
                        plt.xlabel('npc')
                        for ff in range(nfcp):
                            marker = all_markers_shape[ff]
                            counter = 0
                            for pp1, npc1 in enumerate(test_pcs_sadi_ann_sdi):
                                for pp2, npc2 in enumerate(test_pcs_sadi_ann_adi):
                                    if pp2 == 0:
                                        label = 'fcp {:.0f} - npc sdi = {:.0f}'.format(ff+1,npc1)
                                    else:
                                        label = None
                                    plt.plot(npc2, snr_fcp[counter,ff], colors[pp1%len(colors)]+marker,label = label)
                                    counter+=1
                        plt.savefig(outpath_fig+'SNR_fcps_PCA-SADI2-ann'+'.pdf', format='pdf')
                
                        write_fits(outpath+'final_PCA-SADI2_ann_SNR_fcps_'+test_pcs_str+label_test_ann+'.fits', snr_fcp)
                        
                        ## Find best npc for each radius
                        for ff in range(nfcp):
                            idx_best_snr = np.argmax(snr_fcp[:,ff])
                            id_npc_ann_df[ff,0] = test_pcs_sadi_ann_sdi[int(idx_best_snr/ntest_pcs_adi)]
                            id_npc_ann_df[ff,1] = test_pcs_sadi_ann_adi[int(idx_best_snr%ntest_pcs_adi)]
                        
                        ## 8.3. Final PCA-ADI frames with optimal npcs        
                        tmp_tmp = np.zeros([nfcp,PCA_ASDI_cube_ori.shape[1],PCA_ASDI_cube_ori.shape[2]])
                        test_pcs_str_list_sdi = [str(x) for x in test_pcs_sadi_ann_sdi]
                        ntest_pcs_sdi = len(test_pcs_sadi_ann_sdi)                
                        test_pcs_str_sdi = "npc_sdi"+"-".join(test_pcs_str_list_sdi)
                        test_pcs_str_list_adi = [str(x) for x in test_pcs_sadi_ann_adi]
                        ntest_pcs_adi = len(test_pcs_sadi_ann_adi)
                        test_pcs_str_adi = "_npc_adi"+"-".join(test_pcs_str_list_adi)
                        ntest_pcs = int(ntest_pcs_sdi*ntest_pcs_adi)
                        test_pcs_str = test_pcs_str_sdi+test_pcs_str_adi
                    
                        test_pcs_str_list_sdi = [str(int(x)) for x in id_npc_ann_df[:,0]]                               
                        test_pcs_str_sdi = "npc_sdi_opt"+"-".join(test_pcs_str_list_sdi)
                        test_pcs_str_list_adi = [str(int(x)) for x in id_npc_ann_df[:,1]]                               
                        test_pcs_str_adi = "npc_adi_opt"+"-".join(test_pcs_str_list_adi)
                        
                        test_rad_str_list = ["{:.1f}".format(x) for x in rad_arr*plsc]                               
                        test_rad_str = "rad"+"-".join(test_rad_str_list)
                        for pp, idx_npc_ann in enumerate(id_npc_ann_df):
                            params_ann = PCA_ANNULAR_Params(cube=PCA_ASDI_cube_ori, angle_list=derot_angles,
                                                            radius_int=mask_IWA_px, fwhm=fwhm_med, asize=asize,
                                                            delta_rot=delta_rot_ann,
                                                            ncomp=(int(idx_npc_ann[0]),int(idx_npc_ann[1])), scaling=scal,
                                                            svd_mode=svd_mode, scale_list=scale_list,
                                                            ifs_collapse_range=ifs_collapse_range, collapse='median',
                                                            max_frames_lib=max(max_fr,max(idx_npc_ann)+1), full_output=False,
                                                            verbose=verbose, nproc=nproc)
                            tmp_tmp[pp] = pca_annular(algo_params=params_ann)
                        write_fits(outpath+'final_PCA-SADI2_ann_image_{}_at_{}as'.format(test_pcs_str,test_rad_str)+label_test_ann+'.fits', tmp_tmp)
                        write_fits(outpath+'final_PCA-SADI2_ann_opt_npc_at_{}as'.format(test_rad_str)+label_test_ann+'.fits', id_npc_ann_df)
                
                    
                        ### SNR map  
                        if (not isfile(outpath+'final_PCA-SADI2_ann_snrmap_{}_at_{}as'.format(test_pcs_str,test_rad_str)+label_test_ann+'.fits') or overwrite_pp) and do_snr_map:
                            tmp = open_fits(outpath+'final_PCA-SADI2_ann_image_{}_at_{}as'.format(test_pcs_str,test_rad_str)+label_test_ann+'.fits')
                            rad_in = mask_IWA # we comment it for a better visualization of the snr map (there are spurious values in the center)
                            #rad_in = 1.5
                            tmp_tmp = np.ones_like(tmp)
                            for pp in range(tmp.shape[0]):
                                tmp[pp] = snrmap(tmp[pp], fwhm_med, plot=False, nproc=nproc)
                            tmp = mask_circle(tmp,rad_in*fwhm_med)
                            write_fits(outpath+'final_PCA-SADI2_ann_snrmap_{}_at_{}as'.format(test_pcs_str,test_rad_str)+label_test_ann+'.fits', tmp, verbose=False)
            
            label_test_ann = label_test_ann_ori
        
        
        
        ######################### 10. Final contrast curves ###########################
        if planet and planet_parameter is not None:
            # SUBTRACT THE PLANET FROM THE CUBE
            PCA_ASDI_cube_ori = cube_planet_free(planet_parameter, PCA_ASDI_cube_ori, derot_angles, psfn)
            label_emp = '_empty'
        else:
            label_emp = ''
        for ss, scal in enumerate(scalings):
            if scal is None:
                scal_lab = ''
            else:
                scal_lab = scal
            for ii, ifs_collapse_range in enumerate(ifs_collapse_range_list):
                label_test = label_emp+'_'+scal_lab+'_'+ifs_collapse_range_lab[ii]
                    
                # 10.1 PCA-SDI
                if do_sdi and fake_planet:
                    df_list = []
                    rsvd_list = []
                    for rr, rad in enumerate(rad_arr):
                        if svd_mode == 'randsvd':
                            for nr in range(3):
                                pn_contr_curve_sdi_rr_tmp = contrast_curve(PCA_ASDI_cube_ori, derot_angles, psfn,
                                                                                       fwhm, plsc, starphot=starphot, scaling=scal,
                                                                                       algo=pca, sigma=5., nbranch=1,
                                                                                       theta=0, inner_rad=1, wedge=(0,360),fc_snr=fc_snr,
                                                                                       student=True, transmission=transmission, 
                                                                                       plot=True, dpi=100, adimsdi='double',
                                                                                       verbose=verbose, ncomp=(int(id_npc_full_df[rr]),None), 
                                                                                       svd_mode=svd_mode, nproc=nproc)
                                rsvd_list.append(pn_contr_curve_sdi_rr_tmp)
                            pn_contr_curve_sdi_rr = pn_contr_curve_sdi_rr_tmp.copy()
                            for jj in range(pn_contr_curve_full_rr.shape[0]): 
                                sensitivities = []
                                for nr in range(3):
                                    sensitivities.append(rsvd_list[nr]['sensitivity (Student)'][jj])
                                print("Sensitivities at {}: ".format(rsvd_list[rr]['distance'][jj]), sensitivities)
                                idx_min = np.argmin(sensitivities)
                                pn_contr_curve_sdi_rr['sensitivity (Student)'][jj] = rsvd_list[idx_min]['sensitivity (Student)'][jj]
                                pn_contr_curve_sdi_rr['sensitivity (Gauss)'][jj] = rsvd_list[idx_min]['sensitivity (Gauss)'][jj]
                                pn_contr_curve_sdi_rr['throughput'][jj] = rsvd_list[idx_min]['throughput'][jj]
                                pn_contr_curve_sdi_rr['noise'][jj] = rsvd_list[idx_min]['noise'][jj]
                                pn_contr_curve_sdi_rr['sigma corr'][jj] = rsvd_list[idx_min]['sigma corr'][jj]
                        else:
                            pn_contr_curve_sdi_rr = contrast_curve(PCA_ASDI_cube_ori, derot_angles, psfn,
                                                                               fwhm, plsc, starphot=starphot, scaling=scal,
                                                                               algo=pca, sigma=5., nbranch=1,
                                                                               theta=0, inner_rad=1, wedge=(0,360),fc_snr=fc_snr,
                                                                               student=True, transmission=transmission, 
                                                                               plot=True, dpi=100, adimsdi='double',
                                                                               verbose=verbose, ncomp=(int(id_npc_full_df[rr]),None), 
                                                                               svd_mode=svd_mode, nproc=nproc)
                        DF.to_csv(pn_contr_curve_sdi_rr, path_or_buf=outpath+'contrast_curve_PCA-SDI_optimal_at_{:.1f}as.csv'.format(rad*plsc), 
                                  sep=',', na_rep='', float_format=None)
                        df_list.append(pn_contr_curve_sdi_rr)
                    pn_contr_curve_sdi_opt = pn_contr_curve_sdi_rr.copy()
                
                    for jj in range(pn_contr_curve_sdi_opt.shape[0]):  
                        sensitivities = []
                        for rr, rad in enumerate(rad_arr):
                            sensitivities.append(df_list[rr]['sensitivity (Student)'][jj])
                        print("Sensitivities at {}: ".format(df_list[rr]['distance'][jj]), sensitivities)
                        idx_min = np.argmin(sensitivities)
                        pn_contr_curve_sdi_opt['sensitivity (Student)'][jj] = df_list[idx_min]['sensitivity (Student)'][jj]
                        pn_contr_curve_sdi_opt['sensitivity (Gauss)'][jj] = df_list[idx_min]['sensitivity (Gauss)'][jj]
                        pn_contr_curve_sdi_opt['throughput'][jj] = df_list[idx_min]['throughput'][jj]
                        pn_contr_curve_sdi_opt['noise'][jj] = df_list[idx_min]['noise'][jj]
                        pn_contr_curve_sdi_opt['sigma corr'][jj] = df_list[idx_min]['sigma corr'][jj]
                    DF.to_csv(pn_contr_curve_sdi_opt, path_or_buf=outpath+'final_optimal_contrast_curve_PCA-SDI'+label_test+'.csv', 
                              sep=',', na_rep='', float_format=None)
                    
                    
                    arr_dist = np.array(pn_contr_curve_sdi_opt['distance'])
                    arr_contrast = np.array(pn_contr_curve_sdi_opt['sensitivity (Student)'])
                    for ff in range(nfcp):
                        idx = find_nearest(arr_dist, rad_arr[ff])
                        sensitivity_5sig_sdi_df[ff] = arr_contrast[idx]
                    
                    
                # 10.2 PCA-SADI full - either 1 or 2 steps   
                if do_sadi_full and fake_planet:
        
                    df_list = []
            
                    if sadi_steps == 1:
                        adimsdi = 'single'
                        npc_opt = int(id_npc_full_df[rr])
                    else:
                        adimsdi = 'double'
                        npc_opt = (int(id_npc_full_df[rr,0]),int(id_npc_full_df[rr,1]))
                    for rr, rad in enumerate(rad_arr):
                        pn_contr_curve_full_rr = contrast_curve(PCA_ASDI_cube_ori, derot_angles, psfn,
                                                                fwhm, plsc, starphot=starphot,
                                                                algo=pca, sigma=5., nbranch=1,
                                                                theta=0, inner_rad=1, wedge=(0,360),fc_snr=fc_snr,
                                                                student=True, transmission=transmission, scaling=scal,
                                                                plot=True, dpi=100, adimsdi=adimsdi,
                                                                verbose=verbose, ncomp=npc_opt, svd_mode=svd_mode,
                                                                nproc=nproc)
                        DF.to_csv(pn_contr_curve_full_rr, 
                                  path_or_buf=outpath+'contrast_curve_PCA-SADI{:.0f}-full_optimal_at_{:.1f}as.csv'.format(sadi_steps,rad*plsc), 
                                  sep=',', na_rep='', float_format=None)
                        df_list.append(pn_contr_curve_full_rr)
                    pn_contr_curve_full_opt = pn_contr_curve_full_rr.copy()
                
                    for jj in range(pn_contr_curve_full_opt.shape[0]):
                        sensitivities = []
                        for rr, rad in enumerate(rad_arr):
                            sensitivities.append(df_list[rr]['sensitivity (Student)'][jj])
                        print("Sensitivities at {}: ".format(df_list[rr]['distance'][jj]), sensitivities)
                        idx_min = np.argmin(sensitivities)
                        pn_contr_curve_full_opt['sensitivity (Student)'][jj] = df_list[idx_min]['sensitivity (Student)'][jj]
                        pn_contr_curve_full_opt['sensitivity (Gauss)'][jj] = df_list[idx_min]['sensitivity (Gauss)'][jj]
                        pn_contr_curve_full_opt['throughput'][jj] = df_list[idx_min]['throughput'][jj]
                        pn_contr_curve_full_opt['noise'][jj] = df_list[idx_min]['noise'][jj]
                        pn_contr_curve_full_opt['sigma corr'][jj] = df_list[idx_min]['sigma corr'][jj]
                    DF.to_csv(pn_contr_curve_full_opt, 
                              path_or_buf=outpath+'final_optimal_contrast_curve_PCA-SADI{:.0f}-full{}.csv'.format(sadi_steps,label_test), 
                              sep=',', na_rep='', float_format=None)
                    arr_dist = np.array(pn_contr_curve_full_opt['distance'])
                    arr_contrast = np.array(pn_contr_curve_full_opt['sensitivity (Student)'])
                    for ff in range(nfcp):
                        idx = find_nearest(arr_dist, rad_arr[ff])
                        sensitivity_5sig_full_df[ff] = arr_contrast[idx]
            
                # 10.3 PCA-SADI annular               
                if do_sadi_ann and fake_planet:   
                                                         
                    df_list = []
                    for rr, rad in enumerate(rad_arr):
                        pn_contr_curve_ann_rr = contrast_curve(PCA_ASDI_cube_ori, derot_angles, psfn,
                                                      fwhm, plsc, starphot=starphot, 
                                                      algo=pca_annular, sigma=5., nbranch=1,
                                                      theta=0, inner_rad=1, wedge=(0,360),fc_snr=fc_snr,
                                                      student=True, transmission=transmission, 
                                                      plot=True, dpi=100, scaling=scal,
                                                      verbose=verbose, ncomp=(int(id_npc_ann_df[rr,0]),int(id_npc_ann_df[rr,1])), 
                                                      svd_mode=svd_mode, 
                                                      radius_int=mask_IWA_px, asize=asize, 
                                                      delta_rot=delta_rot, nproc=nproc)
                        DF.to_csv(pn_contr_curve_ann_rr, 
                                  path_or_buf=outpath+'contrast_curve_PCA-SADI2-ann_optimal_at_{:.1f}as.csv'.format(rad*plsc), 
                                  sep=',', na_rep='', float_format=None)
                        df_list.append(pn_contr_curve_ann_rr)
                    pn_contr_curve_ann_opt = pn_contr_curve_ann_rr.copy()
                
                    for jj in range(pn_contr_curve_ann_opt.shape[0]):  
                        sensitivities = []
                        for rr, rad in enumerate(rad_arr):
                            sensitivities.append(df_list[rr]['sensitivity (Student)'][jj])
                        print("Sensitivities at {}: ".format(df_list[rr]['distance'][jj]), sensitivities)
                        idx_min = np.argmin(sensitivities)
                        pn_contr_curve_ann_opt['sensitivity (Student)'][jj] = df_list[idx_min]['sensitivity (Student)'][jj]
                        pn_contr_curve_ann_opt['sensitivity (Gauss)'][jj] = df_list[idx_min]['sensitivity (Gauss)'][jj]
                        pn_contr_curve_ann_opt['throughput'][jj] = df_list[idx_min]['throughput'][jj]
                        pn_contr_curve_ann_opt['noise'][jj] = df_list[idx_min]['noise'][jj]
                        pn_contr_curve_ann_opt['sigma corr'][jj] = df_list[idx_min]['sigma corr'][jj]
                    DF.to_csv(pn_contr_curve_ann_opt, path_or_buf=outpath+'final_optimal_contrast_curve_PCA-SADI2-ann.csv', 
                              sep=',', na_rep='', float_format=None)
                    arr_dist = np.array(pn_contr_curve_ann_opt['distance'])
                    arr_contrast = np.array(pn_contr_curve_ann_opt['sensitivity (Student)'])
                    for ff in range(nfcp):
                        idx = find_nearest(arr_dist, rad_arr[ff])
                        sensitivity_5sig_ann_df[ff] = arr_contrast[idx]
        
                if fake_planet:
                    plt.close()              
                    plt.figure()
                    plt.title('5-sigma contrast curve for '+source+' (NACO+AGPM)')
                    plt.ylabel('Contrast')
                    plt.xlabel('Separation (arcsec)')
                    if do_sdi:
                        plt.semilogy(pn_contr_curve_sdi_opt['distance']*plsc, pn_contr_curve_sdi_opt['sensitivity (Student)'],'r', 
                                     linewidth=2, label='PCA-SDI (Student correction)')
                    if do_sadi_full and fake_planet:
                        plt.semilogy(pn_contr_curve_full_opt['distance']*plsc, pn_contr_curve_full_opt['sensitivity (Student)'],'b', 
                                     linewidth=2, label='PCA-SADI{:.0f} full frame (Student, lapack)'.format(sadi_steps))
                    if do_sadi_ann and fake_planet:
                        plt.semilogy(pn_contr_curve_ann_opt['distance']*plsc, pn_contr_curve_ann_opt['sensitivity (Student)'],'g', 
                                     linewidth=2, label='PCA-SADI annular (Student)')                                    
                    plt.legend()
                    plt.savefig(outpath_fig+'contr_curves'+'.pdf', format='pdf')
                    
                    plt.close()              
                    plt.figure()
                    plt.title('5-sigma contrast curve for '+source+' (NACO+AGPM)')
                    plt.ylabel('Contrast')
                    plt.gca().invert_yaxis()
                    plt.xlabel('Separation (arcsec)')
                    if do_sdi:
                        plt.plot(pn_contr_curve_sdi_opt['distance']*plsc, -2.5*np.log10(pn_contr_curve_sdi_opt['sensitivity (Student)']), 
                                 'r', linewidth=2, label='PCA-SDI (Student)')
                    if do_sadi_full and fake_planet:
                        plt.plot(pn_contr_curve_full_opt['distance']*plsc, -2.5*np.log10(pn_contr_curve_full_opt['sensitivity (Student)']),
                                 'b', linewidth=2, label='PCA-SADI{:.0f} full frame (Student, lapack)'.format(sadi_steps))
                    if do_sadi_ann and fake_planet:
                        plt.plot(pn_contr_curve_ann_opt['distance']*plsc, -2.5*np.log10(pn_contr_curve_ann_opt['sensitivity (Student)']),
                                 'g', linewidth=2, label='PCA-SADI2 annular (Student)')                                    
                    plt.legend()
                    plt.savefig(outpath_fig+'contr_curves_MAG'+'.pdf', format='pdf')
                  
                    # WRITE THE CSV FILE
                    datafr1 = DF(data=nfcp_df, columns=['Index of injected fcp'])
                    datafr2 = DF(data=rad_arr, columns=['Radius (px)'])  
                    datafr3 = DF(data=rad_arr*plsc, columns=['Radius (as)'])  
                    datafr = datafr1.join(datafr2).join(datafr3)
                    #datafr7 = DF(data=id_snr_adi_df, columns=['Ideal SNR (m-adi)'])
                    if do_sdi:
                        datafr8 = DF(data=sensitivity_5sig_sdi_df, columns=["5-sig Student sensitivity (PCA-SDI)"])
                        datafr = datafr.join(datafr8)
                    if do_sadi_full:
                        datafr10 = DF(data=id_npc_full_df, columns=["Ideal npc (PCA-SADI{:.0f}-full)".format(sadi_steps)])
                        datafr12 = DF(data=sensitivity_5sig_full_df, 
                                      columns=["5-sig Student sensitivity (PCA-SADI{:.0f}-full, lapack)".format(sadi_steps)])
                        datafr = datafr.join(datafr10).join(datafr12)
                    if do_sadi_ann and fake_planet:
                        datafr14 = DF(data=id_npc_ann_df, columns=["Ideal npc (PCA-SADI2 ann)"])
                        datafr16 = DF(data=sensitivity_5sig_ann_df, columns=["5-sig Student sensitivity (PCA-SADI2-ann)"])
                        datafr = datafr.join(datafr14).join(datafr16)
                    DF.to_csv(datafr, path_or_buf=outpath+'Final_contrast_curves_comparison'+label_test+'.csv', 
                              sep=',', na_rep='', float_format=None)
    return None
