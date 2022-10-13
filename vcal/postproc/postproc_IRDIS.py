#! /usr/bin/env python
# coding: utf-8
"""
Module with the postprocessing routine for SPHERE/IRDIS data.
"""

__author__ = 'V. Christiaens'
__all__ = ['postproc_IRDIS']

# *Version 1 (2019/12)* 
# Version 2 (2021/07) - this version

######################### Importations and definitions ########################

import json
import matplotlib as mpl
import matplotlib.pyplot as plt
from multiprocessing import cpu_count
import numpy as np
from os.path import isfile, isdir
import os
from pandas import DataFrame as DF
import pdb

from vip_hci.fits import open_fits, write_fits
from vip_hci.metrics import snrmap, contrast_curve, snr
from vip_hci.preproc import cube_shift, frame_shift, cube_crop_frames, cube_recenter_via_speckles #cube_subtract_sky_pca,
                             #cube_crop_frames, cube_derotate, cube_collapse)
from vip_hci.preproc.rescaling import _cube_resc_wave
from vip_hci.var import mask_circle, cube_filter_highpass, frame_center, frame_filter_lowpass
from ..utils import find_nearest

try:
    from vip_hci.itpca import pca_it, pca_annular_it, pca_1rho_it, feves
except:
    print("Note: iterative pca not available in your version of VIP")
try:
    from vip_hci.psfsub import median_sub, pca, pca_annular, nmf
    from vip_hci.fm import normalize_psf, cube_inject_companions, cube_planet_free
    from vip_hci.psfsub.utils_pca import pca_annulus
    from vip_hci.metrics import stim_map as compute_stim_map
    from vip_hci.metrics import inverse_stim_map as compute_inverse_stim_map
    from vip_hci.config import time_ini, timing
except:
    from vip_hci.medsub import median_sub
    from vip_hci.pca.utils_pca import pca_annulus
    from vip_hci.pca import pca, pca_annular
    from vip_hci.nmf import nmf
    from vip_hci.metrics import normalize_psf, compute_stim_map, compute_inverse_stim_map, cube_inject_companions
    from vip_hci.negfc import cube_planet_free
    from vip_hci.conf import time_ini, timing

from vcal import __path__ as vcal_path
mpl.use('Agg')

def postproc_IRDIS(params_postproc_name='VCAL_params_postproc_IRDIS.json',
                   params_preproc_name='VCAL_params_preproc_IRDIS.json', 
                   params_calib_name='VCAL_params_calib.json',
                   planet_parameter=None):
    """
    Postprocessing of SPHERE/IRDIS data using preproc parameters provided in 
    json file.

    *Suggestion: run this routine several times with the following parameters 
    set in the parameter file:
        #1. planet = False, fake_planet=False 
            => do_adi= True, do_pca_full=True, do_pca_ann=True
        #2. If a blob is found: set planet_pos_crop to the coordinates of the 
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
    None. All preprocessed products are written as fits files, and can then be 
    used for post-processing.
    
    """
    mpl.style.use('default')
    with open(params_postproc_name, 'r') as read_file_params_postproc:
        params_postproc = json.load(read_file_params_postproc)
    with open(params_preproc_name, 'r') as read_file_params_preproc:
        params_preproc = json.load(read_file_params_preproc)
    with open(params_calib_name, 'r') as read_file_params_calib:
        params_calib = json.load(read_file_params_calib)
    with open(vcal_path[0] + "/instr_param/sphere_filt_spec.json", 'r') as filt_spec_file:
        filt_spec = json.load(filt_spec_file)[params_calib['comb_iflt']]  # Get infos of current filters combination

    # from calib
    path = params_calib['path']
    filters = filt_spec['filters'] 
    path_irdis = path+"IRDIS_reduction/"
    
    # from preproc
    coro = params_preproc['coro']
    plsc_ori = params_preproc['plsc']
    bin_fac = params_preproc.get('bin_fac',1)
    distort_corr = params_preproc['distort_corr']
    if distort_corr:
        distort_corr_labs = ["_DistCorr"]
    else:
        distort_corr_labs = [""]
    final_crop_sz = params_preproc.get('final_crop_sz', 101)
    final_cubename = params_preproc.get('final_cubename', 'final_cube')
    final_anglename = params_preproc.get('final_anglename', 'final_derot_angles')
    final_psfname = params_preproc.get('final_psfname', 'final_psf_med')
    final_fluxname = params_preproc.get('final_fluxname','final_flux')
    final_fwhmname = params_preproc.get('final_fwhmname','final_fwhm')
    final_scalefacname = params_preproc.get('final_scalefacname',None)
    if final_scalefacname is None:
        final_scalefacname = params_postproc.get('final_scalefacname',None)
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
    label_test_pre = params_preproc.get('label_test','')
    outpath_2 = path_irdis+"2_preproc_vip{}/".format(label_test_pre)
    
    # from postproc param file
    source = params_postproc['source']            # should be without space
    sourcename = params_postproc['sourcename']    # can have spaces
    details = params_postproc['details']
    label_test = params_postproc.get('label_test','')
    
    ## Options
    verbose = params_postproc.get("verbose",0)                      # whether to print(more information during the reduction
    debug = params_postproc.get("debug",False)   
    #debug = True                        # whether to print(even more information, helpful for debugging
    #debug_ = True                       # whether to keep all the intermediate fits files of the reduction by the end of the notebook (useful for debugging)
    nproc = params_postproc.get('nproc',int(cpu_count()/2))                          # number of processors to use - can also be set to cpu_count()/2 for efficiency
    overwrite_ADI = params_postproc.get('overwrite_ADI',1)        # whether to overwrite median-ADI results 
    overwrite_pp = params_postproc.get('overwrite_pp',1)         # whether to overwrite PCA-ADI results
    overwrite_it = params_postproc.get('overwrite_it',1)         # whether to overwrite PCA-ADI results
        
    ## TO DO?
    do_adi=params_postproc.get('do_adi',1)
    do_pca_sann = params_postproc.get('do_pca_sann',1) # PCA on a single annulus (requires to provide a planet position)
    do_pca_full = params_postproc.get('do_pca_full',1)
    do_pca_ann = params_postproc.get('do_pca_ann',1)
    do_pca_1zone = params_postproc.get('do_pca_1zone',0)
    #do_pca_2zones = params_postproc.get('do_pca_2zones',0)
    do_nmf = params_postproc.get('do_nmf',0)
    do_feves = params_postproc.get('do_feves',0)
    
    ## Planet?
    planet = params_postproc.get('planet',0)                      # is there a companion?
    planet_pos_crop = params_postproc.get('planet_pos_crop',None) # If so, where is it (or where is it expected)?   (x, y) in cropped frames
    planet_pos_full = params_postproc.get('planet_pos_full',None) #(92,117) # CROPPED IN ANY CASE
    subtract_planet = params_postproc.get('subtract_planet',0)    # this should only be used as a second iteration, after negfc on the companion has enabled to determine its parameters
    
    ## Inject fake companions? If True => will compute contrast curves
    fake_planet = params_postproc.get('fake_planet',0)                 #  FIRST RUN IT AS FALSE TO CHECK FOR THE PRESENCE OF TRUE COMPANIONS
    fcp_pos_r_crop = np.array(params_postproc.get('fcp_pos_r_crop',[0.5]))   # list of r in arcsec where the fcps are injected in the cropped cube
    fcp_pos_r_full = np.array(params_postproc.get('fcp_pos_r_full',[0.5])) # same for the uncropped cube
    injection_fac = params_postproc.get('injection_fac',1.)  # scaling factor for the injection of fcps with respect to first 5-sigma contrast estimate (e.g. 3/5. to inject at 3 sigma instead of 5 sigma)
    fc_snr = params_postproc.get('fc_snr',10.) # snr of the injected fcp in contrast_curve to compute throughput
    nspi = params_postproc.get('nspi',9)        # number of spirals where fcps should be injected - also corresponds to number of PAs where the contrast curve is computed
    wedge = tuple(params_postproc.get('wedge',[0,360])) # in which range of PA should the contrast curve be computed
    
    ## Post-processing
    high_pass_filter_list = params_postproc.get('high_pass_filter_list',[0])  #True # whether to filter out small spatial frequencies - might be useful to remove large scale noise variations in the image, but risky in case of presence of extended authentic disk signal which can get subtracted.
    mask_IWA_px = params_postproc.get('mask_IWA',5)  # just show pca images beyond the provided mask radius (in pixels)
    do_conv = params_postproc.get('do_conv',0) # whether to smooth final images
    do_snr_map = params_postproc.get('do_snr_map',[0,0,0]) # to plot the snr_map (warning: computer intensive); useful only when point-like features are seen in the image
    if not isinstance(do_snr_map, list):
        do_snr_map = [do_snr_map]*3
    do_stim_map = params_postproc.get('do_stim_map',[0,0,0]) # to plot the snr_map (warning: computer intensive); useful only when point-like features are seen in the image
    if not isinstance(do_stim_map, list):
        do_stim_map = [do_stim_map]*3
    do_color_map = params_postproc.get('do_color_map',[0,0,0]) 
    if not isinstance(do_color_map, list):
        do_color_map = [do_color_map]*3
    flux_weights = False # whether to combine residual frames based on original measured fluxes of the star (proxy of AO quality) ## TRY BOTH!
    ###RDI
    ref_cube_name = params_postproc.get('ref_cube_name',None)
    prep_ref_cube = params_postproc.get('prep_ref_cube',[1,2])
    scaling = params_postproc.get('scaling',None) # for RDI
    mask_PCA = params_postproc.get('mask_PCA',None)
    ##DBI
    scale_list = params_postproc.get('scale_list',None)
    adimsdi = params_postproc.get('adimsdi',"double")
    ## it. PCA?
    n_it = params_postproc.get('n_it',0)
    thr_it = params_postproc.get('thr_it',1)
    n_neigh = params_postproc.get('n_neigh',0)
    throughput_corr = params_postproc.get('throughput_corr',0)
    add_res = params_postproc.get('add_res',0)
    strategy = params_postproc.get('strategy','ADI')
    buffer =  params_postproc.get('buffer',1)
    delta_rot_it=params_postproc.get('delta_rot_it',[0,0.5,1])
    ### PCA options
    delta_rot=params_postproc.get('delta_rot',(1,3)) # float or tuple expressed in FWHM # Threshold in azimuthal motion to keep frames in the PCA library created by PCA-annular. If a tuple, corresponds to the threshold for the innermost and outermost annuli, respectively.
    asize=params_postproc.get('asize',3) # width of the annnuli for either pca in concentric annuli or on a single annulus, provided in FWHM
    #### how is SVD done for PCA:
    svd_mode = params_postproc.get('svd_mode','lapack')
    #### number of principal components
    firstguess_pcs = params_postproc.get('firstguess_pcs',[1,21,1])  # explored for first contrast curve 
    test_pcs_sann = params_postproc.get('pcs_sann',[1,21,1]) 
    test_pcs_full = params_postproc.get('pcs_full',[1,21,1]) 
    test_pcs_ann = params_postproc.get('pcs_ann',[1,11,1])
    test_pcs_1zone = params_postproc.get('pcs_1zone',[1,21,1]) 
    test_pcs_2zones = params_postproc.get('pcs_2zones',[1,11,1]) 

    # contrast curves
    n_br =  params_postproc.get('n_br',6) 

    #### min/max number of frames to create PCA library
    min_fr = params_postproc.get('min_fr',test_pcs_ann[1]-1) 
    max_fr = params_postproc.get('max_fr',200)
        
    ################ LOADING FILES AND FORMATTING  - don't change #################
    
    ## Formatting paths
    outpath_4 = path_irdis+"3_postproc_bin{:.0f}"+label_test+"/"
    outpath_5 = outpath_4+"{}_{}/"
    
    ref_cube = None
    label_stg = strategy
    if ref_cube_name is not None:
        if scaling is not None:
            label_stg += "_"+scaling
        if mask_PCA is not None:
            label_stg += "_mask{:.1f}".format(mask_PCA)
            mask_PCA = int(mask_PCA/np.median(plsc_ori))
        
    if coro:
        transmission_name = vcal_path[0] + "/../Static/" + "SPHERE_IRDIS_ALC_transmission_px.fits"
        transmission = open_fits(transmission_name)
        transmission = (transmission[1],transmission[0])
    else:
        transmission = None
    
    if isinstance(delta_rot, list):
        delta_rot = tuple(delta_rot)
        delta_rot_tmp = delta_rot[0]
    else:
        delta_rot_tmp = delta_rot    
    label_test='_thr{:.0f}_mask{:.1f}_maxfr{:.0f}'.format(delta_rot_tmp,mask_IWA_px,max_fr)
    
    if isinstance(svd_mode,str):
        svd_mode_all = [svd_mode,svd_mode]
    elif isinstance(svd_mode, list):
        svd_mode_all = svd_mode
    n_randsvd = 3 # number of times we do PCA rand-svd, before taking the median of all results (there is a risk of significant self-subtraction when just doing it once)
    firstguess_pcs = list(range(firstguess_pcs[0],firstguess_pcs[1],firstguess_pcs[2]))
    test_pcs_sann = list(range(test_pcs_sann[0],test_pcs_sann[1],test_pcs_sann[2]))
    test_pcs_full = list(range(test_pcs_full[0],test_pcs_full[1],test_pcs_full[2]))
    test_pcs_ann = list(range(test_pcs_ann[0],test_pcs_ann[1],test_pcs_ann[2]))
    test_pcs_1zone = list(range(test_pcs_1zone[0],test_pcs_1zone[1],test_pcs_1zone[2]))
    test_pcs_2zones = list(range(test_pcs_2zones[0],test_pcs_2zones[1],test_pcs_2zones[2]))    
    
    #fr_sel_str = "-".join(frame_selection)
    ## Default is post-process twice: 1) crop, 2) no crop
    #crop_list = [final_crop_sz_px,0] # !!! ALREADY CROPPED VERSION OPENED BELOW ! IMPORTANT: always put the case with cropping first (in case you wish to crop)
    if isinstance(final_crop_sz,list):
        ncrop = len(final_crop_sz)
        for i in range(ncrop):
            if final_crop_sz[ncrop-1-i]%2:
                final_crop_sz = final_crop_sz[ncrop-1-i]
                break
    final_crop_as = final_crop_sz*np.median(plsc_ori)
    crop_lab_list = ["crop_{:.1f}as".format(final_crop_as),"no_crop"]
        
    # DEFINE NPC RANGES FOR DIFFERENT PCA algorithms
    npc_ann = None #[((1,1,1,1,1,1,1),(2,1,1,1,1,1,1),(3,1,1,1,1,1,1),(4,1,1,1,1,1,1),(5,1,1,1,1,1,1),(6,1,1,1,1,1,1),(7,1,1,1,1,1,1)),# otherwise None
    #[((1,1,1,1,1,1,1,1),(2,1,1,1,1,1,1,1),(3,1,1,1,1,1,1,1),(4,1,1,1,1,1,1,1),(5,1,1,1,1,1,1,1),(6,1,1,1,1,1,1,1),(7,1,1,1,1,1,1,1)),
               #((1,1,1,1,1,1,1),(2,1,1,1,1,1,1),(3,1,1,1,1,1,1),(4,1,1,1,1,1,1),(5,1,1,1,1,1,1),(6,1,1,1,1,1,1),(7,1,1,1,1,1,1))]# otherwise None
        
    # single ANN (without PA thresholding)
    test_pcs_sann_crop = test_pcs_sann   #list(range(1,21))
    test_pcs_sann_nocrop = test_pcs_sann #list(range(1,21))
    test_pcs_sann_all = [test_pcs_sann_crop,test_pcs_sann_nocrop]
    # FULL
    test_pcs_full_crop = test_pcs_full #[1] 
    #for ii,jj in enumerate(range(1,11,1)): 
    #    test_pcs_full_crop.append(test_pcs_full_crop[ii]+jj)
    print("test pcs full (crop): ", test_pcs_full_crop)
    test_pcs_full_nocrop = test_pcs_full #[5] # randsvd
    #for ii,jj in enumerate(range(1,11,1)):
    #    test_pcs_full_nocrop.append(test_pcs_full_nocrop[ii]+jj)
    print("test pcs full (no crop): ", test_pcs_full_nocrop)
    test_pcs_full_all = [test_pcs_full_crop,test_pcs_full_nocrop]
    # ANN
    test_pcs_ann_crop = test_pcs_ann #[2] # randsvd
    #for ii,jj in enumerate(range(1,11,1)):
    #    test_pcs_ann_crop.append(test_pcs_ann_crop[ii]+jj)
    print("test pcs ann (crop): ", test_pcs_ann_crop)    
    test_pcs_ann_nocrop = None # does not matter, we will never do pca-ann on non-cropped cubes !!!
    test_pcs_ann_all = [test_pcs_ann_crop,test_pcs_ann_nocrop]
    
    th0 = wedge[0]  # trigonometric angle for the first fcp to be injected
    all_markers= ['ro','yo','bo','go','ko','co','mo']*nspi # for plotting the snr of the fcps (should contain at least as many elements as fcps)
        
    ########################## START POST-PROCESSING ##############################
    
    # DEFINE FUTURE PANDAS DATAFRAMES
    # (3 figures of merit: contrast at 0.15'', snr of the companion (~0.27), contrast at 0.40'')
    n_tests = len(crop_lab_list)*len(filters)
    n_ch = len(filters)
    if final_scalefacname is not None:
        scale_list = open_fits(outpath_2+final_scalefacname)
    else:
        scale_list = None
    # LOOP ON ALL PARAMETERS
    counter = 0
       
    for distort_corr_lab in distort_corr_labs:
        #for bb, bin_fac in enumerate(bin_fac_list):
        bin_fac_list = [bin_fac] # dirty hack to avoid re-writing it all
        if not isdir(outpath_4.format(bin_fac)):
            os.system("mkdir "+outpath_4.format(bin_fac))
        for cc, crop_lab in enumerate(crop_lab_list):
            print("*** TESTING binning x{:.0f} - {} (test {}/{})***".format(bin_fac, crop_lab_list[cc],counter+1,n_tests))                  
            for high_pass_filter in high_pass_filter_list:
                #1. (R)DBI if requested
                if scale_list is not None and cc == 0 and high_pass_filter==high_pass_filter_list[0]:
                    if not isdir(outpath_5.format(bin_fac,'DBI',crop_lab_list[cc])):
                        os.system("mkdir "+outpath_5.format(bin_fac,'DBI',crop_lab_list[cc]))
                    if not isfile(outpath_2+"final_DBI_cube.fits") or not isfile(outpath_2+"final_max_fwhm.fits"):
                        fwhm = []
                        for ff, filt in enumerate(filters):
                            ADI_cube= open_fits(outpath_2+final_cubename+"{}.fits".format(filt))
                            fwhm.append(open_fits(outpath_2+final_fwhmname+"{}.fits".format(filt)))
                            if ff == 0:
                                ASDI_cube = np.zeros([len(filters),ADI_cube.shape[0],ADI_cube.shape[1],ADI_cube.shape[2]])
                            else:
                                if ASDI_cube.shape[1] != ADI_cube.shape[0]:
                                    raise TypeError("The ADI cubes of different filters must have same length")
                            ASDI_cube[ff] = ADI_cube
                        fwhm = np.amax(fwhm)
                        write_fits(outpath_2+"final_DBI_cube.fits", ASDI_cube)
                        write_fits(outpath_2+"final_max_fwhm.fits", np.array([fwhm]))
                    else:
                        ASDI_cube = open_fits(outpath_2+"final_DBI_cube.fits")
                        fwhm = open_fits(outpath_2+"final_max_fwhm.fits")[0]
                    derot_angles = open_fits(outpath_2+final_anglename+"{}.fits".format(filters[-1]))
                    
                    if do_pca_full:
                        #PCA_ADI_cube, derot_angles = vip.fits.open_adicube(outpath_5.format(bin_fac,filt,crop_lab_list[cc])+'7_final_crop_PCA_cube'+label_filt+'.fits')
                        # First let's readapt the number of pcs to be tested
                        test_pcs_full = test_pcs_full_all[cc]   
    
                        test_pcs_str_list = [str(x) for x in test_pcs_full]
                        ntest_pcs = len(test_pcs_full)
                        test_pcs_str = "npc"+"-".join(test_pcs_str_list)
                        #tmp_tmp = np.zeros([ntest_pcs,ASDI_cube.shape[2],ASDI_cube.shape[3]])
                        if planet:                
                            snr_tmp = np.zeros(ntest_pcs)
                        
                        #1a. DBI
                        scale_list = np.array(scale_list)
                        final_DBI = np.zeros([len(test_pcs_full),ASDI_cube.shape[-2],ASDI_cube.shape[-1]])
                        for pp, npc in enumerate(test_pcs_full):
                            if adimsdi == "double":
                                if npc == 1:
                                    ncomp = (int(n_ch-1),None)
                                else:
                                    ncomp = (int(n_ch-1),npc-1)
                            else:
                                ncomp = npc
                            DBI_res = pca(ASDI_cube, angle_list=derot_angles, cube_ref=None, scale_list=scale_list, 
                                          adimsdi=adimsdi, ncomp=ncomp, svd_mode=svd_mode_all[cc], scaling=scaling,
                                          mask_center_px=mask_IWA_px, delta_rot=delta_rot, fwhm=fwhm, 
                                          collapse='median', check_memory=True, full_output=True, 
                                          verbose=verbose, conv=do_conv)
                            DBI, residuals, residuals_der = DBI_res
                            final_DBI[pp] = DBI
                            if do_stim_map[1] and adimsdi == 'double' and pp == 0:
                                stim_map = compute_stim_map(residuals_der)
                                inv_stim_map = compute_inverse_stim_map(residuals, derot_angles)
                                norm_stim_map = stim_map/np.percentile(inv_stim_map,99.9)
                                write_fits(outpath_5.format(bin_fac,'DBI',crop_lab_list[cc])+'PCA-DBI_stim_inv_norm_{}.fits'.format(adimsdi), 
                                            np.array([stim_map, inv_stim_map, norm_stim_map]))
                            if pp == 0:
                                write_fits(outpath_5.format(bin_fac,'DBI',crop_lab_list[cc])+'PCA-DBI_{}_npc{:.0f}_res_der.fits'.format(adimsdi,npc), residuals_der, verbose=False)
                                write_fits(outpath_5.format(bin_fac,'DBI',crop_lab_list[cc])+'PCA-DBI_{}_npc{:.0f}_res.fits'.format(adimsdi,npc), residuals, verbose=False)

                            residuals_crop = cube_shift(residuals, 0.5, 0.5)
                            residuals_crop = residuals_crop[:,1:,1:]
                            residuals_crop = cube_crop_frames(residuals_crop,128)
                            write_fits(outpath_5.format(bin_fac,'DBI',crop_lab_list[cc])+'PCA-DBI_{}_npc{:.0f}_res_crop128.fits'.format(adimsdi,npc), residuals_crop, verbose=False)
                            if pp < 2:
                                # color?
                                residuals_col = np.zeros([3,residuals_der.shape[1],residuals_der.shape[2]])
                                residuals_col[0] = residuals_der[-1]-residuals_der[0]
                                residuals_col[1] = residuals_der[-1]/residuals_der[0]
                                norm_H2 = residuals_der[0]
                                norm_H2[np.where(norm_H2<0)] = 0
                                norm_H2 /= np.amax(norm_H2)
                                norm_H3 = residuals_der[-1]
                                norm_H3[np.where(norm_H3<0)] = 0
                                norm_H3 /= np.amax(norm_H3)
                                residuals_col[2] = np.zeros_like(norm_H2)
                                cond1 = norm_H2>0
                                cond2 = norm_H3>0
                                cond = cond1 & cond2
                                residuals_col[2][np.where(cond)] = norm_H3[np.where(cond)]/norm_H2[np.where(cond)]
                                write_fits(outpath_5.format(bin_fac,'DBI',crop_lab_list[cc])+'final_PCA-DBI_{}_npc{:.0f}_colors.fits'.format(adimsdi,npc), residuals_col, verbose=False)
                        write_fits(outpath_5.format(bin_fac,'DBI',crop_lab_list[cc])+'final_PCA-DBI_{}_npcNone-{:.0f}.fits'.format(adimsdi,test_pcs_full[-1]-1), final_DBI, verbose=False)
    
                    #1c. RDI + DBI (on RDI images), if RDI result exists
                    label_filt = label_test
                    if isfile(outpath_5.format(bin_fac,filters[-1],crop_lab_list[cc])+'final_PCA-{}_full_'.format(label_stg)+test_pcs_str+label_filt+'.fits'):
                        tmp = open_fits(outpath_5.format(bin_fac,filters[-1],crop_lab_list[cc])+'final_PCA-{}_full_'.format(label_stg)+test_pcs_str+label_filt+'.fits')
                        RDBI_res = np.zeros_like(tmp)
                        col_res = np.zeros_like(tmp)
                        cc_map = np.zeros_like(tmp)
                        for i in range(tmp.shape[0]):
                            tmp_tmp = np.zeros([len(filters),1,tmp.shape[1],tmp.shape[2]])
                            norm_tmp = np.zeros(len(filters)) 
                            for ff, filt in enumerate(filters):
                                tmp_tmp[ff,0] = open_fits(outpath_5.format(bin_fac,filt,crop_lab_list[cc])+'final_PCA-{}_full_'.format(label_stg)+test_pcs_str+label_filt+'.fits')[i]
                                norm_tmp[ff] = np.amax(tmp_tmp[ff,0])
                            derot_tmp = np.zeros(1)
                            # norm colors
                            cond1 = tmp_tmp[0,0]>0
                            cond2 = tmp_tmp[-1,0]>0
                            cond = cond1 & cond2
                            col_res[i][np.where(cond)] = (tmp_tmp[-1,0][np.where(cond)]/norm_tmp[-1])/(tmp_tmp[0,0][np.where(cond)]/norm_tmp[0])
                            cc_map[i][np.where(cond)] = (tmp_tmp[-1,0][np.where(cond)]/norm_tmp[-1])*(tmp_tmp[0,0][np.where(cond)]/norm_tmp[0])
                            # DBI
                            RDBI_res[i]= pca(tmp_tmp, angle_list=derot_tmp, cube_ref=None, scale_list=scale_list, adimsdi='double',
                                              ncomp=(int(n_ch-1),None), svd_mode=svd_mode_all[cc], scaling=scaling,
                                              mask_center_px=mask_IWA_px, delta_rot=delta_rot, fwhm=fwhm, 
                                              collapse='median', check_memory=True, full_output=False, 
                                              verbose=verbose, conv=do_conv)

                        write_fits(outpath_5.format(bin_fac,'DBI',crop_lab_list[cc])+'PCA-RDBI_{}{}.fits'.format(adimsdi,test_pcs_str), 
                                    RDBI_res, verbose=False)
                        write_fits(outpath_5.format(bin_fac,'DBI',crop_lab_list[cc])+'RDI_norm_colors{}.fits'.format(test_pcs_str), 
                                    col_res, verbose=False)
                        write_fits(outpath_5.format(bin_fac,'DBI',crop_lab_list[cc])+'RDI_cross_corr{}.fits'.format(test_pcs_str), 
                                    cc_map, verbose=False)
                        
                    # 1c. RDI + DBI (on RDI res cubes)
                    if isfile(outpath_5.format(bin_fac,filters[-1],crop_lab_list[cc])+'TMP_PCA-{}_full_'.format(label_stg)+'npc{:.0f}'.format(test_pcs_full[-1])+label_filt+'_res.fits'):
                        tmp = open_fits(outpath_5.format(bin_fac,filters[-1],crop_lab_list[cc])+'TMP_PCA-{}_full_'.format(label_stg)+'npc{:.0f}'.format(test_pcs_full[-1])+label_filt+'_res.fits')
                        RDBI_res = np.zeros([len(test_pcs_full),tmp.shape[1],tmp.shape[2]])
                        for i, npc in enumerate(test_pcs_full):
                            tmp_tmp = np.zeros([len(filters),tmp.shape[0],tmp.shape[1],tmp.shape[2]])
                            for ff, filt in enumerate(filters):
                                tmp_tmp[ff] = open_fits(outpath_5.format(bin_fac,filt,crop_lab_list[cc])+'TMP_PCA-{}_full_'.format(label_stg)+'npc{:.0f}'.format(test_pcs_full[i])+label_filt+'_res.fits')
                            DBI_res= pca(tmp_tmp, angle_list=derot_angles, cube_ref=None, scale_list=scale_list, adimsdi='double',
                                         ncomp=(int(n_ch-1),None), svd_mode=svd_mode_all[cc], scaling=scaling,
                                         mask_center_px=mask_IWA_px, delta_rot=delta_rot, fwhm=fwhm, 
                                         collapse='median', check_memory=True, full_output=True, 
                                         verbose=verbose, conv=do_conv)
                            RDBI_res[i], residuals, residuals_der = DBI_res
                            if npc < 5:
                                write_fits(outpath_5.format(bin_fac,'DBI',crop_lab_list[cc])+'TMP_PCA-RDBI_indiv_{}_npc{:.0f}_res.fits'.format(adimsdi,npc), 
                                       residuals, verbose=False)
                                write_fits(outpath_5.format(bin_fac,'DBI',crop_lab_list[cc])+'TMP_PCA-RDBI_indiv_{}_npc{:.0f}_res_der.fits'.format(adimsdi,npc), 
                                   residuals_der, verbose=False)
                                if do_stim_map[1]:
                                    stim_map = compute_stim_map(residuals_der)
                                    inv_stim_map = compute_inverse_stim_map(residuals, derot_angles)
                                    norm_stim_map = stim_map/np.percentile(inv_stim_map,99.9)
                                    write_fits(outpath_5.format(bin_fac,'DBI',crop_lab_list[cc])+'TMP_PCA-RDBI_STIM_inv_norm_{}_npc{:.0f}.fits'.format(adimsdi, npc), 
                                               np.array([stim_map, inv_stim_map, norm_stim_map]))
                        write_fits(outpath_5.format(bin_fac,'DBI',crop_lab_list[cc])+'final_PCA-RDBI_indiv_{}{}.fits'.format(adimsdi,test_pcs_str), 
                                   RDBI_res, verbose=False)
                        
                    # 1d. manual RDI + DBI (on RDI res cubes): to measure colors
                    if isfile(outpath_5.format(bin_fac,filters[-1],crop_lab_list[cc])+'TMP_PCA-{}_full_'.format(label_stg)+'npc{:.0f}'.format(test_pcs_full[-1])+label_filt+'_res.fits'):
                        tmp = open_fits(outpath_5.format(bin_fac,filters[-1],crop_lab_list[cc])+'TMP_PCA-{}_full_'.format(label_stg)+'npc{:.0f}'.format(test_pcs_full[-1])+label_filt+'_res.fits')
                        n_cubes = tmp.shape[0]
                        RDBI_res = np.zeros([len(test_pcs_full),len(filters),n_cubes,tmp.shape[1],tmp.shape[2]])
                        RDBI_fin = np.zeros([len(test_pcs_full),len(filters),tmp.shape[1],tmp.shape[2]])
                        for i, npc in enumerate(test_pcs_full):
                            tmp_tmp = np.zeros([len(filters),tmp.shape[0],tmp.shape[1],tmp.shape[2]])
                            for ff, filt in enumerate(filters):
                                tmp_tmp[ff] = open_fits(outpath_5.format(bin_fac,filt,crop_lab_list[cc])+'TMP_PCA-{}_full_'.format(label_stg)+'npc{:.0f}'.format(test_pcs_full[i])+label_filt+'_res.fits')
                            for ff, filt in enumerate(filters):
                                # scale the other one
                                cube_ref = _cube_resc_wave(tmp_tmp[ff-1], ref_xy=None, 
                                                           scaling_list=[scale_list[ff-1]/scale_list[ff]]*n_cubes)
                                #cube_ref_tmp = np.zeros([1,cube_ref.shape[1],cube_ref.shape[2]])
                                for nn in range(n_cubes):
                                    #cube_ref_tmp[0] = 
                                    RDBI_res[i,ff,nn] = pca(np.array([tmp_tmp[ff,nn]]), angle_list=np.array([derot_angles[nn]]), 
                                                        cube_ref=np.array([cube_ref[nn]]), ncomp=1,
                                                        #scale_list=scale_list, #adimsdi='double',
                                                        #ncomp=(int(n_ch-1),None), svd_mode=svd_mode_all[cc], scaling=scaling,
                                                        mask_center_px=mask_IWA_px, delta_rot=delta_rot, fwhm=fwhm, 
                                                        collapse='median', check_memory=True, full_output=False, 
                                                        verbose=verbose, conv=do_conv)
                                if i ==0:
                                    write_fits(outpath_5.format(bin_fac,'DBI',crop_lab_list[cc])+'TMP_PCA-RDBIman_indiv_{}_npc{:.0f}_res_der_{}.fits'.format(adimsdi,npc, filt), 
                                               RDBI_res[ff,i], verbose=False)
                                RDBI_fin[i,ff] = np.median(RDBI_res[i,ff], axis=0) 
                            write_fits(outpath_5.format(bin_fac,'DBI',crop_lab_list[cc])+'TMP_PCA-RDBIman_indiv_{}_npc{:.0f}_res_der.fits'.format(adimsdi,npc), 
                                       RDBI_fin[i], verbose=False)
                        for ff, filt in enumerate(filters):
                            write_fits(outpath_5.format(bin_fac,'DBI',crop_lab_list[cc])+'final_PCA-RDBIman_indiv_{}{}{}.fits'.format(adimsdi,test_pcs_str,filt), 
                                       RDBI_fin[:,ff], verbose=False)
                        
                # 2. Skip DBI on large crops
                elif scale_list is not None:
                    pass
                # 3. ADI or (A)RDI iterative on smallest crop
                elif cc ==0 and n_it>0:
                    for ff, filt in enumerate(filters):
                        plsc = float(plsc_ori[ff])
                        if not isdir(outpath_5.format(bin_fac,filt,crop_lab_list[cc])):
                            os.system("mkdir "+outpath_5.format(bin_fac,filt,crop_lab_list[cc]))
                        fwhm = float(open_fits(outpath_2+final_fwhmname+"{}.fits".format(filt))[0])
                        flux = float(open_fits(outpath_2+final_fluxname+"{}.fits".format(filt))[0])
                        if cc == 0 or not isfile(outpath_2+final_cubename+"_full{}.fits".format(filt)):
                            ADI_cube= open_fits(outpath_2+final_cubename+"{}.fits".format(filt))
                        else:
                            ADI_cube= open_fits(outpath_2+final_cubename+"_full{}.fits".format(filt))
                        if ref_cube_name is not None:
                            ref_cube = open_fits(ref_cube_name.format(filt))
                                
                        derot_angles = open_fits(outpath_2+final_anglename+"{}.fits".format(filt))
    #                    if derot_name == "rotnth":
    #                        derot_angles*=-1
                        psfn = open_fits(outpath_2+final_psfname+"{}.fits".format(filt)) # this has all the unsat psf frames
                        #3a. pca it in full (cropped) frames
                        if do_pca_full and not do_pca_1zone:
                            final_imgs = np.zeros([len(test_pcs_full),ADI_cube.shape[1],ADI_cube.shape[2]])
                            #wmean_imgs = np.zeros_like(final_imgs)
                            if mask_PCA is None:
                                mask_rdi = None
                            else:
                                mask_tmp = np.ones_like(ADI_cube[0])
                                mask_rdi = mask_circle(mask_tmp, mask_PCA, fillwith=0, mode='in')
                            for pp, npc in enumerate(test_pcs_full):
                                res = pca_it(ADI_cube, derot_angles, cube_ref=ref_cube, mask_center_px=mask_IWA_px, fwhm=fwhm,  
                                             strategy=strategy, thr=thr_it, n_it=n_it, n_neigh=n_neigh, ncomp=npc, scaling=scaling, 
                                             thru_corr=throughput_corr, psfn=psfn, n_br=n_br, mask_rdi=mask_rdi, full_output=True)
                                final_imgs[pp], it_cube, sig_cube, res, res_der, thru_2d_cube, stim_cube, it_cube_nd = res
                                write_fits(outpath_5.format(bin_fac,filt,crop_lab_list[cc])+"TMP_PCA-{}_it{:.0f}_thr{:.1f}_npc{:.0f}_{}_last_res.fits".format(label_stg,n_it,thr_it,npc,filt), res)
                                write_fits(outpath_5.format(bin_fac,filt,crop_lab_list[cc])+"TMP_PCA-{}_it{:.0f}_thr{:.1f}_npc{:.0f}_{}_last_res_der.fits".format(label_stg,n_it,thr_it,npc,filt), res_der)
                                write_fits(outpath_5.format(bin_fac,filt,crop_lab_list[cc])+"TMP_PCA-{}_it{:.0f}_thr{:.1f}_npc{:.0f}_{}_it_cube.fits".format(label_stg,n_it,thr_it,npc,filt), it_cube)
                                write_fits(outpath_5.format(bin_fac,filt,crop_lab_list[cc])+"TMP_PCA-{}_it{:.0f}_thr{:.1f}_npc{:.0f}_{}_it_cube_nd.fits".format(label_stg,n_it,thr_it,npc,filt), it_cube_nd)
                                write_fits(outpath_5.format(bin_fac,filt,crop_lab_list[cc])+"TMP_PCA-{}_it{:.0f}_thr{:.1f}_npc{:.0f}_{}_sig_cube.fits".format(label_stg,n_it,thr_it,npc,filt), sig_cube)
                                write_fits(outpath_5.format(bin_fac,filt,crop_lab_list[cc])+"TMP_PCA-{}_it{:.0f}_thr{:.1f}_npc{:.0f}_{}_thru_2d_cube.fits".format(label_stg,n_it,thr_it,npc,filt), thru_2d_cube)
                                write_fits(outpath_5.format(bin_fac,filt,crop_lab_list[cc])+"TMP_PCA-{}_it{:.0f}_thr{:.1f}_npc{:.0f}_{}_stim_cube.fits".format(label_stg,n_it,thr_it,npc,filt), stim_cube)
                                # stim = compute_stim_map(res_der)
                                # inv_stim = compute_inverse_stim_map(res, derot_angles)
                                # norm_stim = stim/np.percentile(inv_stim,99.7)
                                # write_fits(outpath_5.format(bin_fac,filt,crop_lab_list[cc])+"TMP_PCA-{}_it{:.0f}_thr{:.1f}_npc{:.0f}_{}_stim.fits".format(strategy,n_it,thr_it,npc,filt), 
                                #            np.array([stim, inv_stim, norm_stim]))
                                # FIRST: define mask following spirals: max stim map (2-5)!
                                # good_mask = np.zeros_like(stim)
                                # good_mask[np.where(norm_stim>1)]=1
                                # ccorr_coeff = cube_distance(res_der,final_imgs[pp],mode='mask',mask=good_mask)
                                # norm_cc = ccorr_coeff/np.sum(ccorr_coeff)
                                # wmean_imgs[pp] = cube_collapse(res_der,mode='wmean',w=norm_cc)
                            #write_fits(outpath_3.format(data_folder)+"final_PCA-RDI_it{:.0f}_thr{:.1f}_{:.0f}-{:.0f}_{}_wmean.fits".format(label_stg,n_it,thr,test_npcs[0], test_npcs[-1],filt), wmean_imgs)
                            write_fits(outpath_5.format(bin_fac,filt,crop_lab_list[cc])+"final_PCA-{}_it{:.0f}_thr{:.1f}_{:.0f}-{:.0f}_{}.fits".format(label_stg,n_it,thr_it,test_pcs_full[0],test_pcs_full[-1],filt), final_imgs)
                            # correction by AGPM transmission
                            # for pp, npc in enumerate(test_npcs):
                            #     wmean_imgs[pp]/=transmission_2d
                            #     final_imgs[pp]/=transmission_2d
                            # #write_fits(outpath_5.format(bin_fac,filt,crop_lab_list[cc])+"final_PCA-RDI_it{:.0f}_thr{:.1f}_{:.0f}-{:.0f}_ann{:.0f}_wmean_AGPMcorr.fits".format(n_it,thr_it,test_npcs[0], test_npcs[-1],ann_sz), wmean_imgs)
                            # write_fits(outpath_5.format(bin_fac,filt,crop_lab_list[cc])+"final_PCA-RDI_it{:.0f}_thr{:.1f}_ann_{:.0f}-{:.0f}_ann{:.0f}_AGPMcorr.fits".format(n_it,thr_it,test_npcs[0],test_npcs[-1],ann_sz), final_imgs)

                        if do_pca_ann and not do_pca_1zone:
                            final_imgs = np.zeros([len(test_pcs_ann),ADI_cube.shape[1],ADI_cube.shape[2]])
                            #wmean_imgs = np.zeros_like(final_imgs)
                            for pp, npc in enumerate(test_pcs_ann):
                                res = pca_annular_it(ADI_cube, derot_angles, cube_ref=ref_cube, radius_int=mask_IWA_px, fwhm=fwhm,  
                                                      thr=thr_it, asize=int(asize*fwhm), n_it=n_it, ncomp=npc, 
                                                      thru_corr=throughput_corr, psfn=psfn, n_br=n_br,
                                                      delta_rot=delta_rot, scaling=scaling, full_output=True)
                                final_imgs[pp], it_cube, sig_cube, res, res_der = res
                                write_fits(outpath_5.format(bin_fac,filt,crop_lab_list[cc])+"TMP_PCA-{}ann_it{:.0f}_thr{:.1f}_npc{:.0f}_{}_last_res.fits".format(label_stg,n_it,thr_it,npc,filt), res)
                                write_fits(outpath_5.format(bin_fac,filt,crop_lab_list[cc])+"TMP_PCA-{}ann_it{:.0f}_thr{:.1f}_npc{:.0f}_{}_last_res_der.fits".format(label_stg,n_it,thr_it,npc,filt), res_der)
                                write_fits(outpath_5.format(bin_fac,filt,crop_lab_list[cc])+"TMP_PCA-{}ann_it{:.0f}_thr{:.1f}_npc{:.0f}_{}_it_cube.fits".format(label_stg,n_it,thr_it,npc,filt), it_cube)
                                write_fits(outpath_5.format(bin_fac,filt,crop_lab_list[cc])+"TMP_PCA-{}ann_it{:.0f}_thr{:.1f}_npc{:.0f}_{}_sig_cube.fits".format(label_stg,n_it,thr_it,npc,filt), sig_cube)
                                stim = compute_stim_map(res_der)
                                inv_stim = compute_inverse_stim_map(res, derot_angles)
                                norm_stim = stim/np.percentile(inv_stim,99.7)
                                write_fits(outpath_5.format(bin_fac,filt,crop_lab_list[cc])+"TMP_PCA-{}ann_it{:.0f}_thr{:.1f}_npc{:.0f}_{}_stim.fits".format(label_stg,n_it,thr_it,npc,filt), 
                                           np.array([stim, inv_stim, norm_stim]))
                                # FIRST: define mask following spirals: max stim map (2-5)!
                                # good_mask = np.zeros_like(stim)
                                # good_mask[np.where(norm_stim>1)]=1
                                # ccorr_coeff = cube_distance(res_der,final_imgs[pp],mode='mask',mask=good_mask)
                                # norm_cc = ccorr_coeff/np.sum(ccorr_coeff)
                                # wmean_imgs[pp] = cube_collapse(res_der,mode='wmean',w=norm_cc)
                            #write_fits(outpath_3.format(data_folder)+"final_PCA-RDI_it{:.0f}_thr{:.1f}_{:.0f}-{:.0f}_{}_wmean.fits".format(label_stg,n_it,thr,test_npcs[0], test_npcs[-1],filt), wmean_imgs)
                            write_fits(outpath_5.format(bin_fac,filt,crop_lab_list[cc])+"final_PCA-{}ann_it{:.0f}_thr{:.1f}_ann_{:.0f}-{:.0f}_{}.fits".format(label_stg,n_it,thr_it,test_pcs_ann[0],test_pcs_ann[-1],filt), final_imgs)
                            # correction by AGPM transmission
                            # for pp, npc in enumerate(test_npcs):
                            #     wmean_imgs[pp]/=transmission_2d
                            #     final_imgs[pp]/=transmission_2d
                            # #write_fits(outpath_5.format(bin_fac,filt,crop_lab_list[cc])+"final_PCA-RDI_it{:.0f}_thr{:.1f}_{:.0f}-{:.0f}_ann{:.0f}_wmean_AGPMcorr.fits".format(n_it,thr_it,test_npcs[0], test_npcs[-1],ann_sz), wmean_imgs)
                            # write_fits(outpath_5.format(bin_fac,filt,crop_lab_list[cc])+"final_PCA-RDI_it{:.0f}_thr{:.1f}_ann_{:.0f}-{:.0f}_ann{:.0f}_AGPMcorr.fits".format(n_it,thr_it,test_npcs[0],test_npcs[-1],ann_sz), final_imgs)
                        
                        if do_pca_1zone:
                            final_imgs = np.zeros([len(test_pcs_full),ADI_cube.shape[1],ADI_cube.shape[2]])
                            #wmean_imgs = np.zeros_like(final_imgs)
                            if mask_PCA is None:
                                mask_rdi = None
                            else:
                                mask_tmp = np.ones_like(ADI_cube[0])
                                mask_rdi = mask_circle(mask_tmp, mask_PCA, fillwith=0, mode='in')
                            #res = pca_1zone_it(ADI_cube, derot_angles, cube_ref=ref_cube, 
                            res = pca_1rho_it(ADI_cube, derot_angles, cube_ref=ref_cube,                  
                                               fwhm=fwhm, buffer=buffer, strategy=strategy, 
                                               ncomp_range=test_pcs_1zone, n_it_max=n_it, 
                                               thr=thr_it, n_neigh=n_neigh, thru_corr=throughput_corr, 
                                               n_br=n_br, psfn=psfn, starphot=flux, 
                                               plsc=plsc, svd_mode=svd_mode, 
                                               scaling=scaling, delta_rot=delta_rot_it, 
                                               mask_center_px=mask_IWA_px, add_res=add_res, 
                                               collapse='median',  mask_rdi=mask_rdi, 
                                               full_output=True, verbose=verbose, 
                                               weights=None, debug=debug, 
                                               path=outpath_5.format(bin_fac,filt,crop_lab_list[cc]),
                                               overwrite=overwrite_it)
                            if isinstance(thr_it,(int,float)):
                                thr_it1 = thr_it
                                thr_it2 = thr_it
                            else:
                                thr_it1 = thr_it[0]
                                thr_it2 = thr_it[-1]
                            write_fits(outpath_5.format(bin_fac,filt,crop_lab_list[cc])+"final_1rhoPCA-{}_it{:.0f}_thr{:.1f}-{:.1f}_{:.0f}-{:.0f}_{}.fits".format(label_stg,n_it,thr_it1,thr_it2,test_pcs_1zone[0],test_pcs_1zone[-1],filt), res[0])
                            write_fits(outpath_5.format(bin_fac,filt,crop_lab_list[cc])+"TMP_1rhoPCA-{}_it{:.0f}_thr{:.1f}-{:.1f}_npc{:.0f}-{:.0f}_{}_it_cube.fits".format(label_stg,n_it,thr_it1,thr_it2,test_pcs_1zone[0],test_pcs_1zone[-1],filt), res[1])
                            write_fits(outpath_5.format(bin_fac,filt,crop_lab_list[cc])+"TMP_1rhoPCA-{}_it{:.0f}_thr{:.1f}-{:.1f}_npc{:.0f}-{:.0f}_{}_stim_cube.fits".format(label_stg,n_it,thr_it1,thr_it2,test_pcs_1zone[0],test_pcs_1zone[-1],filt), res[2])
                            write_fits(outpath_5.format(bin_fac,filt,crop_lab_list[cc])+"TMP_1rhoPCA-{}_it{:.0f}_thr{:.1f}-{:.1f}_npc{:.0f}-{:.0f}_{}_sig_cube.fits".format(label_stg,n_it,thr_it1,thr_it2,test_pcs_1zone[0],test_pcs_1zone[-1],filt), res[3])
                            write_fits(outpath_5.format(bin_fac,filt,crop_lab_list[cc])+"TMP_1rhoPCA-{}_it{:.0f}_thr{:.1f}-{:.1f}_npc{:.0f}-{:.0f}_{}_drot_opt_arr.fits".format(label_stg,n_it,thr_it1,thr_it2,test_pcs_1zone[0],test_pcs_1zone[-1],filt), res[4])
                            write_fits(outpath_5.format(bin_fac,filt,crop_lab_list[cc])+"TMP_1rhoPCA-{}_it{:.0f}_thr{:.1f}-{:.1f}_npc{:.0f}-{:.0f}_{}_thr_opt_arr.fits".format(label_stg,n_it,thr_it1,thr_it2,test_pcs_1zone[0],test_pcs_1zone[-1],filt), res[5])
                            write_fits(outpath_5.format(bin_fac,filt,crop_lab_list[cc])+"TMP_1rhoPCA-{}_it{:.0f}_thr{:.1f}-{:.1f}_npc{:.0f}-{:.0f}_{}_npc_opt_arr.fits".format(label_stg,n_it,thr_it1,thr_it2,test_pcs_1zone[0],test_pcs_1zone[-1],filt), res[6])
                            write_fits(outpath_5.format(bin_fac,filt,crop_lab_list[cc])+"TMP_1rhoPCA-{}_it{:.0f}_thr{:.1f}-{:.1f}_npc{:.0f}-{:.0f}_{}_nit_opt_arr.fits".format(label_stg,n_it,thr_it1,thr_it2,test_pcs_1zone[0],test_pcs_1zone[-1],filt), res[7])
                            write_fits(outpath_5.format(bin_fac,filt,crop_lab_list[cc])+"TMP_1rhoPCA-{}_it{:.0f}_thr{:.1f}-{:.1f}_npc{:.0f}-{:.0f}_{}_cc_rad_ws_ss_opt_arr.fits".format(label_stg,n_it,thr_it1,thr_it2,test_pcs_1zone[0],test_pcs_1zone[-1],filt), np.array([res[8], res[9], res[10]]))
                          


                
                # 4. regular ADI/RDI per channel for all crops, as long as no scale_list
                if scale_list is None:    
                    for ff, filt in enumerate(filters):
                        if not isdir(outpath_5.format(bin_fac,filt,crop_lab_list[cc])):
                            os.system("mkdir "+outpath_5.format(bin_fac,filt,crop_lab_list[cc]))
                        #fwhm = fwhm_ori[ff]
                        plsc = plsc_ori[ff]
    
                        if cc == 0 or not isfile(outpath_2+final_cubename+"_full{}.fits".format(filt)):
                            ADI_cube = open_fits(outpath_2+final_cubename+"{}.fits".format(filt))
                        else:
                            ADI_cube = open_fits(outpath_2+final_cubename+"_full{}.fits".format(filt))
                        if ref_cube_name is not None:
                            ref_cube = open_fits(ref_cube_name.format(filt))
                                
                        derot_angles = open_fits(outpath_2+final_anglename+"{}.fits".format(filt))
    #                    if derot_name == "rotnth":
    #                        derot_angles*=-1
                        psf = open_fits(outpath_2+final_psfname+"{}.fits".format(filt)) # this has all the unsat psf frames
                    
                        # crop ADI_cube if even
                        if not ADI_cube.shape[-1] %2:
                            ADI_cube = ADI_cube[:,1:,1:]
                            ADI_cube = cube_shift(ADI_cube,0.5,0.5)
                        # crop psf if even
                        #psf = np.median(psf_cube, axis=0)
                        #psf=psf_cube[0]
                        #write_fits(outpath_2+psf_name+'_'+filt,psf_cube)
                        if not psf.shape[-1] %2:
                            psf = psf[1:,1:]
                            psf = frame_shift(psf,0.5,0.5)
                            
                        # measure flux and fwhm
                        psfn, starphot, fwhm = normalize_psf(psf,fwhm='fit',size=19,full_output=True,force_odd=True,mask_core=6)
                        #mask_IWA_px = int(mask_IWA*fwhm)
                        if starphot < 0 or fwhm<3:
                            print("There is a problem with the unsat psf")
                            pdb.set_trace()
         
                        if high_pass_filter:
                            label_filt = '_hpf'
                            # MODIFY THE IF BELOW
                            if isfile(outpath_4.format(bin_fac)+final_cubename+"{}{}.fits".format(filt,label_filt)):
                                ADI_cube = open_fits(outpath_4.format(bin_fac)+final_cubename+"{}{}.fits".format(filt,label_filt))
                            else:
                                ADI_cube = cube_filter_highpass(ADI_cube, 'median-subt', median_size=int(4*fwhm))
                                write_fits(outpath_4.format(bin_fac)+final_cubename+"{}{}.fits".format(filt,label_filt), ADI_cube)
                                #vip.fits.append_extension(outpath_4.format(bin_fac)+"final_cube_{}{}.fits".format(filt,label_filt), derot_angles)
                            
                        else:
                            label_filt = label_test
                           
                        ## SUBTRACT COMPANION, IF ANY
                        if subtract_planet:
                            ADI_cube = cube_planet_free(planet_parameter, ADI_cube, derot_angles, psfn, plsc)
            
                        
                        ######################## 2. Crop the cube #########################
                        
                        # DEPENDING ON THE SITUATION, CROP
                        PCA_ADI_cube_ori = ADI_cube.copy()
                        cy, cx = frame_center(PCA_ADI_cube_ori[0])
                        if planet:
                            if cc==0: # use crop cube
                                xx_comp = planet_pos_crop[0]
                                yy_comp = planet_pos_crop[1]
                            else:
                                xx_comp = planet_pos_full[0]
                                yy_comp = planet_pos_full[1]
                            r_pl = np.sqrt((xx_comp-cx)**2+(yy_comp-cy)**2)
                        if fake_planet:
                            if cc==0: # use crop cube
                                rad_arr = fcp_pos_r_crop/plsc
                            else:
                                rad_arr = fcp_pos_r_full/plsc
                            while rad_arr[-1] >= PCA_ADI_cube_ori.shape[2]/2:
                                rad_arr = rad_arr[:-1]
                            nfcp = rad_arr.shape[0]                        
                        if not do_adi:
                            ADI_cube = None  
            
                        ################# 3. First quick contrast curve ###################
                        # This is to determine the level at which each fcp should be injected
                        if fake_planet and cc == 0:
                            if not isfile(outpath_5.format(bin_fac,filt,crop_lab_list[cc])+'TMP_first_guess_5sig_sensitivity_'+label_stg+label_filt+'.fits') or not isfile(outpath_5.format(bin_fac,filt,crop_lab_list[cc])+'TMP_first_guess_contrast_curve_PCA-{}-full.csv'.format(label_stg)) or overwrite_pp:
                                df_list = []
                            # CROP ADI / REF CUBE to min size for sizes to match
                            if ref_cube is not None:
                                if ref_cube.shape[-1] > PCA_ADI_cube_ori.shape[-1]:
                                    ref_cube = cube_crop_frames(ref_cube, PCA_ADI_cube_ori.shape[-1])
                                elif ref_cube.shape[-1] < PCA_ADI_cube_ori.shape[-1]:
                                    PCA_ADI_cube_ori = cube_crop_frames(PCA_ADI_cube_ori, ref_cube.shape[-1])
                            for nn, npc in enumerate(firstguess_pcs):
                                pn_contr_curve_full_rr = contrast_curve(PCA_ADI_cube_ori, derot_angles, psfn,
                                                                                    fwhm, plsc, starphot=starphot, 
                                                                                    algo=pca, sigma=5., nbranch=n_br,
                                                                                    theta=0, inner_rad=1, wedge=(0,360),
                                                                                    fc_snr=fc_snr, cube_ref=ref_cube,
                                                                                    scaling=scaling,
                                                                                    student=True, transmission=transmission, 
                                                                                    plot=True, dpi=100, 
                                                                                    verbose=verbose, ncomp=int(npc), 
                                                                                    svd_mode=svd_mode_all[0])
                                #DF.to_csv(pn_contr_curve_full_nn, path_or_buf=outpath_4.format(crop_lab_list[cc])+'contrast_curve_PCA-ADI-full_optimal_at_{:.1f}as.csv'.format(rad*plsc), sep=',', na_rep='', float_format=None)
                                df_list.append(pn_contr_curve_full_rr)
                            pn_contr_curve_full_rsvd_opt = pn_contr_curve_full_rr.copy()
            
                            for jj in range(pn_contr_curve_full_rsvd_opt.shape[0]):  
                                sensitivities = []
                                for nn, npc in enumerate(firstguess_pcs):
                                    sensitivities.append(df_list[nn]['sensitivity_student'][jj])
                                print("Sensitivities at {}: ".format(df_list[nn]['distance'][jj]), sensitivities)
                                idx_min = np.argmin(sensitivities)
                                pn_contr_curve_full_rsvd_opt['sensitivity_student'][jj] = df_list[idx_min]['sensitivity_student'][jj]
                                pn_contr_curve_full_rsvd_opt['sensitivity_gaussian'][jj] = df_list[idx_min]['sensitivity_gaussian'][jj]
                                pn_contr_curve_full_rsvd_opt['throughput'][jj] = df_list[idx_min]['throughput'][jj]
                                pn_contr_curve_full_rsvd_opt['noise'][jj] = df_list[idx_min]['noise'][jj]
                                pn_contr_curve_full_rsvd_opt['sigma corr'][jj] = df_list[idx_min]['sigma corr'][jj]
                            DF.to_csv(pn_contr_curve_full_rsvd_opt, path_or_buf=outpath_5.format(bin_fac,filt,crop_lab_list[cc])+'TMP_optimal_contrast_curve_PCA-{}-full_randsvd.csv'.format(label_stg), sep=',', na_rep='', float_format=None)
                            arr_dist = np.array(pn_contr_curve_full_rsvd_opt['distance'])
                            arr_contrast = np.array(pn_contr_curve_full_rsvd_opt['sensitivity_student'])
                            
                            sensitivity_5sig_full_rsvd_df = np.zeros(nfcp)
                            for ff in range(nfcp):
                                idx = find_nearest(arr_dist, rad_arr[ff])
                                sensitivity_5sig_full_rsvd_df[ff] = arr_contrast[idx]
                            write_fits(outpath_5.format(bin_fac,filt,crop_lab_list[cc])+'TMP_first_guess_5sig_sensitivity_{}'.format(label_stg)+label_filt+'.fits', sensitivity_5sig_full_rsvd_df)
                        elif fake_planet:
                            sensitivity_5sig_full_rsvd_df = open_fits(outpath_5.format(bin_fac,filt,crop_lab_list[cc])+'TMP_first_guess_5sig_sensitivity_{}'.format(label_stg)+label_filt+'.fits')               
                            
                             
                        ############### 4. INJECT FAKE PLANETS AT 5-sigma #################
                        PCA_ADI_cube = PCA_ADI_cube_ori.copy()
                        if fake_planet:
                            th_step = (wedge[1]-wedge[0])/nspi
                            for ns in range(nspi):
                                theta0 = th0+ns*th_step
                                
                                for ff in range(nfcp):
                                    if ff+1 > sensitivity_5sig_full_rsvd_df.shape[0]:
                                        flevel = np.median(starphot)*sensitivity_5sig_full_rsvd_df[-1]*injection_fac/np.sqrt(((rad_arr[ff]*plsc)/0.5))
                                    else:
                                        flevel = np.median(starphot)*sensitivity_5sig_full_rsvd_df[ff]*injection_fac # injected at ~3 sigma level instead of 5 sigma (rule is normalized at 0.5'', empirically it seems one has to be more conservative below 1'', hence division by radius)
                                    PCA_ADI_cube = cube_inject_companions(PCA_ADI_cube, psfn,
                                                                          derot_angles, flevel,
                                                                          plsc, rad_dists=rad_arr[ff:ff+1],
                                                                          n_branches=1, theta=(theta0+ff*th_step)%360, imlib='opencv', verbose=verbose)
                                write_fits(outpath_5.format(bin_fac,filt,crop_lab_list[cc])+'7_final_crop_PCA_cube'+label_filt+'_fcp_spi{:.0f}.fits'.format(ns), PCA_ADI_cube)
                                #vip.fits.append_extension(outpath_5.format(bin_fac,filt,crop_lab_list[cc])+'7_final_crop_PCA_cube'+label_filt+'_fcp_spi{:.0f}.fits'.format(ns), derot_angles)
            
                            nfcp_df = range(1,nfcp+1)
                            if do_adi:
                                sensitivity_5sig_adi_df = np.zeros(nfcp)
                            if do_pca_full:
                                id_npc_full_df = np.zeros(nfcp)
                                sensitivity_5sig_full_df = np.zeros(nfcp)
                            if do_pca_ann and cc == 0 and bin_fac == np.amax(bin_fac_list):
                                id_npc_ann_df = np.zeros(nfcp)
                                sensitivity_5sig_ann_df = np.zeros(nfcp)
                        
                        
                        
                        ######################### 5. Simple ADI ###########################
                        if do_adi:
                            if flux_weights:
                                if not '_fw' in label_filt:
                                    label_filt+='_fw'
                            if not isfile(outpath_5.format(bin_fac,filt,crop_lab_list[cc])+'final_ADI_simple'+label_filt+'.fits') or overwrite_ADI:
                                _, tmp, tmp_tmp = median_sub(ADI_cube, derot_angles, fwhm=fwhm,
                                                                        radius_int=0, asize=2, delta_rot=delta_rot, 
                                                                        full_output=True, verbose=True)    
                                if flux_weights:
                                     w = starphot/np.nansum(starphot)
                                     for zz in range(w.shape[0]):
                                         tmp[zz]*=w[zz]
                                     tmp_tmp = np.nansum(tmp, axis=0)
                                     
                                tmp_tmp = mask_circle(tmp_tmp,0.9*fwhm)  # we mask the IWA
                                write_fits(outpath_5.format(bin_fac,filt,crop_lab_list[cc])+'final_ADI_simple'+label_filt+'.fits', tmp_tmp)
                            else:
                                tmp_tmp = open_fits(outpath_5.format(bin_fac,filt,crop_lab_list[cc])+'final_ADI_simple'+label_filt+'.fits')
                            #id_snr_adi_df[counter] = vip.metrics.snr(tmp_tmp, (xx_comp,yy_comp), fwhm)
                            ## Convolution
                            if (not isfile(outpath_5.format(bin_fac,filt,crop_lab_list[cc])+'final_ADI_simple'+label_filt+'_conv.fits') or overwrite_ADI) and do_conv:
                                tmp = open_fits(outpath_5.format(bin_fac,filt,crop_lab_list[cc])+'final_ADI_simple'+label_filt+'.fits')
                                tmp = frame_filter_lowpass(tmp, mode='gauss', fwhm_size=fwhm/2, gauss_mode='conv')
                                write_fits(outpath_5.format(bin_fac,filt,crop_lab_list[cc])+'final_ADI_simple'+label_filt+'_conv.fits', tmp, verbose=False)
                            ## SNR map  
                            if (not isfile(outpath_5.format(bin_fac,filt,crop_lab_list[cc])+'final_ADI_simple'+label_filt+'_snrmap.fits') or overwrite_ADI) and do_snr_map[0]:
                                tmp = open_fits(outpath_5.format(bin_fac,filt,crop_lab_list[cc])+'final_ADI_simple'+label_filt+'.fits')
                                #rad_in = mask_IWA 
                                tmp_tmp = snrmap(tmp, fwhm, plot=False)
                                tmp_tmp = mask_circle(tmp_tmp,mask_IWA_px)#rad_in*fwhm)
                                write_fits(outpath_5.format(bin_fac,filt,crop_lab_list[cc])+'final_ADI_simple'+label_filt+'_snrmap.fits', tmp_tmp, verbose=False)
                            if flux_weights:
                                label_filt=label_filt[:-3] # remove _fw
                            ## Contrast curve ADI
                            if fake_planet:
                                #psfn = open_fits(outpath_2+'master_unsat_psf_norm'+'.fits')
                                #starphot = open_fits(outpath_4.format(bin_fac)+'7_norm_fact'+'.fits')
                                pn_contr_curve_adi = contrast_curve(ADI_cube, derot_angles, psfn,
                                                                                fwhm, plsc, starphot=starphot, 
                                                                                algo=median_sub, sigma=5., nbranch=n_br,
                                                                                theta=0, inner_rad=1, wedge=(0,360),fc_snr=fc_snr,
                                                                                student=True, transmission=None, smooth=True,
                                                                                plot=False, dpi=100, debug=False, 
                                                                                verbose=verbose)
                                arr_dist_adi = np.array(pn_contr_curve_adi['distance'])
                                arr_contrast_adi = np.array(pn_contr_curve_adi['sensitivity_student'])
                                for ff in range(nfcp):
                                    idx = find_nearest(arr_dist_adi, rad_arr[ff])
                                    sensitivity_5sig_adi_df[ff] = arr_contrast_adi[idx]
                            ADI_cube = None 
            
            
                        # CROP ADI / REF CUBE to min size for sizes to match
                        if ref_cube is not None:
                            if ref_cube.shape[-1] > PCA_ADI_cube.shape[-1]:
                                ref_cube = cube_crop_frames(ref_cube, PCA_ADI_cube.shape[-1])
                            elif ref_cube.shape[-1] < PCA_ADI_cube.shape[-1]:
                                PCA_ADI_cube = cube_crop_frames(PCA_ADI_cube, ref_cube.shape[-1])
                            if isinstance(prep_ref_cube,list):
                                #1. fine centering with speckles
                                if 1 in prep_ref_cube or 2 in prep_ref_cube:
                                    cube_sci_tmp = PCA_ADI_cube.copy()
                                    cube_ref_tmp = ref_cube.copy()
                                    cube_sci_tmp[np.where(cube_sci_tmp<=0)]=np.amin(np.abs(cube_sci_tmp[np.where(cube_sci_tmp>0)]))
                                    cube_ref_tmp[np.where(cube_ref_tmp<=0)]=np.amin(np.abs(cube_ref_tmp[np.where(cube_ref_tmp>0)]))
                                    cube_sci_tmp = np.array([np.median(cube_sci_tmp,axis=0)])
                                    dim = cube_sci_tmp.shape[1]
                                    if 1 in prep_ref_cube:
                                        #1. fine centering with speckles in full frame
                                        res = cube_recenter_via_speckles(cube_sci_tmp, cube_ref=cube_ref_tmp, alignment_iter=5,
                                                                   gammaval=1, min_spat_freq=0.5, max_spat_freq=3,
                                                                   fwhm=4, debug=False, recenter_median=False,
                                                                   fit_type='gaus', negative=False, crop=False,
                                                                   subframesize=dim-2, imlib='opencv', 
                                                                   interpolation='lanczos4', plot=True, 
                                                                   full_output=True)
                                        _, ref_cube, _, _, _, _, shifts_x_ref, shifts_y_ref = res
                                    else:
                                        #2. fine centering with speckles in 8 subframes (avoiding center -- biased by mask)
                                        sub_sz = int(cube_sci_tmp.shape[1]/3)
                                        if not sub_sz %2:
                                            sub_sz -= 1
                                        idx_ini_y = [0,sub_sz,2*sub_sz,0,2*sub_sz,0,sub_sz,2*sub_sz]
                                        idx_ini_x = [0,0,0,sub_sz,sub_sz,2*sub_sz,2*sub_sz,2*sub_sz]
                                        shifts_x_ref = np.zeros([8,cube_ref_tmp.shape[0]])
                                        shifts_y_ref = np.zeros([8,cube_ref_tmp.shape[0]])
                                        for tt in range(8):
                                            cube_crop_sci = cube_sci_tmp[:,idx_ini_y[tt]:idx_ini_y[tt]+sub_sz,idx_ini_x[tt]:idx_ini_x[tt]+sub_sz]
                                            cube_crop_ref = cube_ref_tmp[:,idx_ini_y[tt]:idx_ini_y[tt]+sub_sz,idx_ini_x[tt]:idx_ini_x[tt]+sub_sz]
                                            res = cube_recenter_via_speckles(cube_crop_sci, cube_ref=cube_crop_ref, alignment_iter=5,
                                                                   gammaval=1, min_spat_freq=0.5, max_spat_freq=3,
                                                                   fwhm=4, debug=False, recenter_median=False,
                                                                   fit_type='gaus', negative=False, crop=False,
                                                                   subframesize=dim-2, imlib='opencv', 
                                                                   interpolation='lanczos4', plot=True, 
                                                                   full_output=True)
                                            _, _, _, _, _, _, shifts_x_ref[tt], shifts_y_ref[tt] = res
                                        shifts_x_std = np.std(shifts_x_ref,axis=0)
                                        shifts_y_std = np.std(shifts_y_ref,axis=0)
                                        shifts_x_ref = np.median(shifts_x_ref,axis=0)
                                        shifts_y_ref = np.median(shifts_y_ref,axis=0)
                                        write_fits(outpath_5.format(bin_fac,filt,crop_lab_list[cc])+"TMP_xy_shifts_std_cen_ref_cube_cc.fits", 
                                                   np.array([shifts_x_std,shifts_y_std]))
                                    if debug:
                                        shifts_ref = np.array([shifts_x_ref, shifts_y_ref])
                                        write_fits(outpath_5.format(bin_fac,filt,crop_lab_list[cc])+"TMP_xy_shifts_cen_ref_cube_cc.fits", 
                                                   shifts_ref)
                                        write_fits(outpath_5.format(bin_fac,filt,crop_lab_list[cc])+"TMP_final_cen_ref_cube_cc.fits", 
                                                   ref_cube)
                                #3. convert neg values to 0. 
                                if 3 in prep_ref_cube:
                                    ref_cube[np.where(ref_cube<0)]=0
                            
                        ########################## 6. PCA-annulus #########################
                        if do_pca_sann and planet and cc ==0:
                            test_pcs_sann = test_pcs_sann_all[cc]
                            test_pcs_str_list = [str(int(x)) for x in test_pcs_sann]
                            ntest_pcs = len(test_pcs_sann)                                
                            test_pcs_str = "npc"+"-".join(test_pcs_str_list)
            
                            # FURTHER CROP THE ADI CUBE, IF POSSIBLE
                            crop_sz = int(2*(r_pl+asize*fwhm))+3
                            if crop_sz < PCA_ADI_cube.shape[1] and crop_sz != 0:
                                if not isfile(outpath_5.format(bin_fac,filt,crop_lab_list[cc])+'7_final_crop2_PCA_cube'+label_filt+'.fits') or overwrite_pp:
                                    PCA_ADI_cube_sann = cube_crop_frames(PCA_ADI_cube,crop_sz)
                                    write_fits(outpath_5.format(bin_fac,filt,crop_lab_list[cc])+'7_final_crop2_PCA_cube'+label_filt+'.fits', PCA_ADI_cube_sann)
                                    #vip.fits.append_extension(outpath_5.format(bin_fac,filt,crop_lab_list[cc])+'7_final_crop2_PCA_cube'+label_filt+'.fits', derot_angles)
                                else:
                                    PCA_ADI_cube_sann = open_fits(outpath_5.format(bin_fac,filt,crop_lab_list[cc])+'7_final_crop2_PCA_cube'+label_filt+'.fits')
                                if ref_cube is not None:
                                    ref_cube_tmp = cube_crop_frames(ref_cube, crop_sz)
                                xx_comp = xx_comp+int((crop_sz-PCA_ADI_cube.shape[1])/2.)
                                yy_comp = yy_comp+int((crop_sz-PCA_ADI_cube.shape[1])/2.)
                            else:
                                PCA_ADI_cube_sann = open_fits(outpath_5.format(bin_fac,filt,crop_lab_list[cc])+'7_final_crop_PCA_cube'+label_filt+'.fits')
                            if verbose:
                                print("(new) planet position: ({:.1f},{:.1f})".format(xx_comp,yy_comp))
                            #PCA_ADI_cube = None
                            
                            tmp_tmp = np.zeros([ntest_pcs,PCA_ADI_cube_sann.shape[1],PCA_ADI_cube_sann.shape[2]])
                            snr_tmp = np.zeros(ntest_pcs)
                            for pp, npc in enumerate(test_pcs_sann):
                                if verbose:
                                    t0 = time_ini()
                                tmp_tmp[pp] = pca_annulus(PCA_ADI_cube_sann, derot_angles, int(npc), asize*fwhm, r_pl, cube_ref=ref_cube_tmp,
                                                          scaling=scaling, svd_mode=svd_mode_all[1], collapse='median',
                                                          imlib='opencv', interpolation='lanczos4')
                                snr_tmp[pp] = snr(tmp_tmp[pp], (xx_comp,yy_comp), fwhm, plot=False, exclude_negative_lobes=True,
                                                                 verbose=False)
                                if verbose:                                     
                                    print("SNR of the candidate at ({},{}) for npc = {:.0f} : {:.1f}".format(xx_comp,yy_comp,npc,snr_tmp[pp]))
                                    timing(t0)
                            write_fits(outpath_5.format(bin_fac,filt,crop_lab_list[cc])+'final_PCA-{}_sann_'.format(label_stg)+test_pcs_str+label_filt+'.fits', tmp_tmp)
                            write_fits(outpath_5.format(bin_fac,filt,crop_lab_list[cc])+'final_PCA-{}_sann_SNR_'.format(label_stg)+test_pcs_str+label_filt+'.fits', snr_tmp)
            
                            ## Find best npc
                            idx_best_snr = np.argmax(snr_tmp)                
                            # Find second best npc
                            snr_tmp_tmp = snr_tmp.tolist()
                            del snr_tmp_tmp[idx_best_snr]
                            idx_best_snr2 = np.argmax(snr_tmp_tmp)
                            if idx_best_snr2 >= idx_best_snr:
                                idx_best_snr2+=1
                            opt_npc = test_pcs_sann[idx_best_snr]
        
                            plt.close()                
                            plt.figure()
                            plt.title('SNR of '+sourcename+' b '+details)
                            plt.ylabel('SNR')
                            plt.xlabel('npc')
                            for ii, npc in enumerate(test_pcs_sann):
                                if snr_tmp[ii] > 5:
                                    marker = 'go'
                                elif snr_tmp[ii] > 3:
                                    marker = 'bo'
                                else:
                                    marker = 'ro'
                                plt.plot(test_pcs_sann[ii], snr_tmp[ii],marker)                            
                            plt.legend(["PCA-{} single annulus (optimal npc ={:.0f})".format(label_stg,opt_npc)])
                            plt.savefig(outpath_5.format(bin_fac,filt,crop_lab_list[cc])+'SNR_vs_npc_PCA_{}_sann.pdf'.format(label_stg), format='pdf')
            
                            PCA_ADI_cube_sann = None
                        
            
                        ####################### 7. PCA-ADI full ###########################
                        if do_pca_full:
                            #PCA_ADI_cube, derot_angles = vip.fits.open_adicube(outpath_5.format(bin_fac,filt,crop_lab_list[cc])+'7_final_crop_PCA_cube'+label_filt+'.fits')
                            # First let's readapt the number of pcs to be tested
                            test_pcs_full = test_pcs_full_all[cc]   
        
                            test_pcs_str_list = [str(x) for x in test_pcs_full]
                            ntest_pcs = len(test_pcs_full)
                            test_pcs_str = "npc"+"-".join(test_pcs_str_list)
                            if flux_weights and not '_fw' in label_filt:
                                label_filt+='_fw'                     
                            if not fake_planet:
                                tmp_tmp = np.zeros([ntest_pcs,PCA_ADI_cube.shape[1],PCA_ADI_cube.shape[2]])
                                if do_stim_map[1]:
                                    stim_map = np.zeros_like(tmp_tmp)
                                    inv_stim_map = np.zeros_like(tmp_tmp)
                                    norm_stim_map = np.zeros_like(tmp_tmp)
                                if planet:                
                                    snr_tmp = np.zeros(ntest_pcs)
                                if do_nmf:
                                    tmp_nmf = np.zeros_like(tmp_tmp)
                                for pp, npc in enumerate(test_pcs_full):
                                    if do_nmf:
                                        tmp_nmf[pp] = nmf(PCA_ADI_cube, derot_angles, cube_ref=ref_cube, ncomp=npc, 
                                                          scaling=scaling, max_iter=100, random_state=None, 
                                                          mask_center_px=mask_IWA_px, imlib='opencv',
                                                          interpolation='lanczos4', collapse='median', full_output=False,
                                                          verbose=True)
                                    if svd_mode_all[cc] == 'randsvd':
                                        tmp_tmp_tmp = np.zeros([n_randsvd,PCA_ADI_cube.shape[1],PCA_ADI_cube.shape[2]])
                                        for nr in range(n_randsvd):
                                            tmp_tmp_tmp[nr] = pca(PCA_ADI_cube, angle_list=derot_angles, cube_ref=ref_cube, scale_list=None, ncomp=int(npc),
                                                                     scaling=scaling, svd_mode=svd_mode_all[cc], mask_center_px=mask_IWA_px,
                                                                     delta_rot=delta_rot, fwhm=fwhm, collapse='median', check_memory=True, 
                                                                     full_output=False, verbose=verbose)
                                        tmp_tmp[pp] = np.median(tmp_tmp_tmp, axis=0)
                                    else:
                                        if mask_PCA is None:
                                            mask_rdi = None
                                        else:
                                            mask_tmp = np.ones_like(PCA_ADI_cube[0])
                                            mask_rdi = mask_circle(mask_tmp, mask_PCA, fillwith=0, mode='in')
                                        tmp_tmp[pp], pcs, recon, tmp_res, tmp = pca(PCA_ADI_cube, angle_list=derot_angles,
                                                       cube_ref=ref_cube, scale_list=None, ncomp=int(npc), 
                                                      svd_mode=svd_mode_all[cc], scaling=scaling,  mask_center_px=mask_IWA_px,
                                                      delta_rot=delta_rot, fwhm=fwhm, collapse='median', check_memory=True, 
                                                      full_output=True, verbose=verbose, mask_rdi=mask_rdi)
                                        if debug:
                                            write_fits(outpath_5.format(bin_fac,filt,crop_lab_list[cc])+'TMP_PCA-{}_full_'.format(label_stg)+'npc{:.0f}'.format(npc)+label_filt+'_res.fits', tmp_res)
                                            if pp == 0:
                                                write_fits(outpath_5.format(bin_fac,filt,crop_lab_list[cc])+'TMP_PCA-{}_full_'.format(label_stg)+'npc{:.0f}'.format(npc)+label_filt+'_pcs.fits', pcs)
                                                write_fits(outpath_5.format(bin_fac,filt,crop_lab_list[cc])+'TMP_PCA-{}_full_'.format(label_stg)+'npc{:.0f}'.format(npc)+label_filt+'_recon.fits', recon)
                                        if do_stim_map[1]:
                                            stim_map[pp] = compute_stim_map(tmp)
                                            inv_stim_map[pp] = compute_inverse_stim_map(tmp_res, derot_angles)
                                            norm_stim_map[pp] = stim_map[pp]/np.percentile(inv_stim_map[pp],99.99)

                                        if flux_weights:
                                            w = starphot/np.nansum(starphot)
                                            for zz in range(starphot.shape[0]):
                                                tmp[zz] = tmp[zz]*w[zz]
                                            tmp_tmp[pp] = np.nansum(tmp,axis=0)
                                    if planet:
                                        snr_tmp[pp] = snr(tmp_tmp[pp], (xx_comp,yy_comp), fwhm, plot=False, exclude_negative_lobes=True,
                                                                              verbose=False)                  
                                if planet:
                                    plt.close() 
                                    plt.figure()
                                    plt.title('SNR for '+sourcename+' b'+details+ '(PCA-{} full-frame)'.format(label_stg))
                                    plt.ylabel('SNR')
                                    plt.xlabel('npc')  
                                    for pp, npc in enumerate(test_pcs_full):
                                        if snr_tmp[pp] > 5:
                                            marker = 'go'
                                        elif snr_tmp[pp] > 3:
                                            marker = 'bo'
                                        else:
                                            marker = 'ro'
                                        plt.plot(npc, snr_tmp[pp], marker)   
                                    try:
                                        plt.savefig(outpath_5.format(bin_fac,filt,crop_lab_list[cc])+'SNR_'+source+'_PCA-{}-full.pdf'.format(label_stg), format='pdf')
                                    except:
                                        pass
                                    write_fits(outpath_5.format(bin_fac,filt,crop_lab_list[cc])+'final_PCA-{}_full_SNR_'.format(label_stg)+test_pcs_str+label_filt+'.fits', snr_tmp)
                                write_fits(outpath_5.format(bin_fac,filt,crop_lab_list[cc])+'final_PCA-{}_full_'.format(label_stg)+test_pcs_str+label_filt+'.fits', tmp_tmp)
                                if do_nmf:
                                    write_fits(outpath_5.format(bin_fac,filt,crop_lab_list[cc])+'final_NMF-{}_full_'.format(label_stg)+test_pcs_str+label_filt+'.fits', tmp_nmf)
                                if do_stim_map[1]:
                                    write_fits(outpath_5.format(bin_fac,filt,crop_lab_list[cc])+'final_PCA-{}_full_'.format(label_stg)+test_pcs_str+label_filt+'_stim.fits', stim_map)
                                    write_fits(outpath_5.format(bin_fac,filt,crop_lab_list[cc])+'final_PCA-{}_full_'.format(label_stg)+test_pcs_str+label_filt+'_stim_inv.fits', inv_stim_map)
                                    write_fits(outpath_5.format(bin_fac,filt,crop_lab_list[cc])+'final_PCA-{}_full_'.format(label_stg)+test_pcs_str+label_filt+'_stim_norm.fits', norm_stim_map)
                                    
                                ### Convolution
                                if (not isfile(outpath_5.format(bin_fac,filt,crop_lab_list[cc])+'final_PCA-{}_full_'.format(label_stg)+test_pcs_str+label_filt+'_conv.fits') or overwrite_pp) and do_conv:
                                    tmp = open_fits(outpath_5.format(bin_fac,filt,crop_lab_list[cc])+'final_PCA-{}_full_'.format(label_stg)+test_pcs_str+label_filt+'.fits')
                                    for nn in range(tmp.shape[0]):
                                        tmp[nn] = frame_filter_lowpass(tmp[nn], mode='gauss',  fwhm_size=fwhm, gauss_mode='conv')
                                    write_fits(outpath_5.format(bin_fac,filt,crop_lab_list[cc])+'final_PCA-{}_full_'.format(label_stg)+test_pcs_str+label_filt+'_conv.fits', tmp, verbose=False)
                                ### SNR map  
                                if (not isfile(outpath_5.format(bin_fac,filt,crop_lab_list[cc])+'final_PCA-{}_full_'.format(label_stg)+test_pcs_str+label_filt+'_snrmap.fits') or overwrite_pp) and do_snr_map[1] and cc==0:
                                    tmp = open_fits(outpath_5.format(bin_fac,filt,crop_lab_list[cc])+'final_PCA-{}_full_'.format(label_stg)+test_pcs_str+label_filt+'.fits')
                                    #rad_in = mask_IWA # we comment it for a better visualization of the snr map (there are spurious values in the center)
                                    #rad_in = 1.5
                                    tmp_tmp = np.ones_like(tmp)
                                    for pp in range(tmp.shape[0]):
                                        tmp[pp] = snrmap(tmp[pp], fwhm, plot=False)
                                    tmp = mask_circle(tmp,mask_IWA_px)#rad_in*fwhm)
                                    write_fits(outpath_5.format(bin_fac,filt,crop_lab_list[cc])+'final_PCA-{}_full_'.format(label_stg)+test_pcs_str+label_filt+'_snrmap.fits', tmp, verbose=False)  
                                if flux_weights:
                                    label_filt = label_filt[:-3]
                            else:
                                snr_tmp_tmp = np.zeros([nspi,ntest_pcs,nfcp])
                                tmp_tmp = np.zeros([ntest_pcs,PCA_ADI_cube.shape[1],PCA_ADI_cube.shape[2]])
                                for ns in range(nspi):
                                    theta0 = th0+ns*th_step
                                    PCA_ADI_cube = open_fits(outpath_5.format(bin_fac,filt,crop_lab_list[cc])+'7_final_crop_PCA_cube'+label_filt+'_fcp_spi{:.0f}.fits'.format(ns))
                                    for pp, npc in enumerate(test_pcs_full):
                                        if svd_mode_all[cc] == 'randsvd':
                                            tmp_tmp_tmp = np.zeros([n_randsvd,PCA_ADI_cube.shape[1],PCA_ADI_cube.shape[2]])
                                            for nr in range(n_randsvd):
                                                tmp_tmp_tmp[nr] = pca(PCA_ADI_cube, angle_list=derot_angles, cube_ref=ref_cube, scale_list=None, ncomp=int(npc),
                                                                         svd_mode=svd_mode_all[cc], scaling=scaling,  mask_center_px=mask_IWA_px,
                                                                         delta_rot=1, fwhm=fwhm, collapse='median', check_memory=True, 
                                                                         full_output=False, verbose=verbose)
                                            tmp_tmp[pp] = np.median(tmp_tmp_tmp, axis=0)
                                        else:
                                            tmp_tmp[pp] = pca(PCA_ADI_cube, angle_list=derot_angles, cube_ref=ref_cube, scale_list=None, ncomp=int(npc),
                                                          svd_mode=svd_mode_all[cc], scaling=scaling,  mask_center_px=mask_IWA_px,
                                                          delta_rot=1, fwhm=fwhm, collapse='median', check_memory=True, 
                                                          full_output=False, verbose=verbose)
                                        for ff in range(nfcp):
                                            xx_fcp = cx + rad_arr[ff]*np.cos(np.deg2rad(theta0+ff*th_step))
                                            yy_fcp = cy + rad_arr[ff]*np.sin(np.deg2rad(theta0+ff*th_step))
                                            snr_tmp_tmp[ns,pp,ff] = snr(tmp_tmp[pp], (xx_fcp,yy_fcp), fwhm, plot=False, exclude_negative_lobes=True,
                                                                                  verbose=True)   
                                    write_fits(outpath_5.format(bin_fac,filt,crop_lab_list[cc])+'TMP_PCA-{}_full_'.format(label_stg)+test_pcs_str+label_filt+'_fcp_spi{:.0f}.fits'.format(ns), tmp_tmp)                                             
                                snr_fcp = np.median(snr_tmp_tmp, axis=0)
                                plt.close() 
                                plt.figure()
                                plt.title('SNR for fcps '+details+ '(PCA-{} full-frame)'.format(label_stg))
                                plt.ylabel('SNR')
                                plt.xlabel('npc')
                                for ff in range(nfcp):
                                    marker = all_markers[ff]
                                    for pp, npc in enumerate(test_pcs_full):
                                        plt.plot(npc, snr_fcp[pp,ff], marker)
                                try:
                                    plt.savefig(outpath_5.format(bin_fac,filt,crop_lab_list[cc])+'SNR_fcps_PCA-{}-full.pdf'.format(label_stg), format='pdf')
                                except:
                                    pass
                                write_fits(outpath_5.format(bin_fac,filt,crop_lab_list[cc])+'final_PCA-{}_full_SNR_fcps_'.format(label_stg)+test_pcs_str+label_filt+'.fits', snr_fcp)   
                                
                                ## Find best npc for each radius
                                for ff in range(nfcp):
                                    idx_best_snr = np.argmax(snr_fcp[:,ff])
                                    id_npc_full_df[ff] = test_pcs_full[idx_best_snr]
                         
                                
                                ## 7.3. Final PCA-ADI frames with optimal npcs        
                                tmp_tmp = np.zeros([nfcp,PCA_ADI_cube_ori.shape[1],PCA_ADI_cube_ori.shape[2]])
                                test_pcs_str_list = [str(int(x)) for x in id_npc_full_df]                               
                                test_pcs_str = "npc_opt"+"-".join(test_pcs_str_list)
                                test_rad_str_list = ["{:.1f}".format(x) for x in rad_arr*plsc]                               
                                test_rad_str = "rad"+"-".join(test_rad_str_list)
                                for pp, npc in enumerate(id_npc_full_df):
                                    if svd_mode_all[cc] == 'randsvd':
                                        tmp_tmp_tmp = np.zeros([n_randsvd,PCA_ADI_cube_ori.shape[1],PCA_ADI_cube_ori.shape[2]])
                                        for nr in range(n_randsvd):
                                            tmp_tmp_tmp[nr] = pca(PCA_ADI_cube_ori, angle_list=derot_angles, cube_ref=ref_cube, scale_list=None, ncomp=int(npc),
                                                                     svd_mode=svd_mode_all[cc], scaling=scaling, mask_center_px=mask_IWA_px,
                                                                     delta_rot=delta_rot, fwhm=fwhm, collapse='median', check_memory=True, 
                                                                     full_output=False, verbose=verbose)
                                        tmp_tmp[pp] = np.median(tmp_tmp_tmp, axis=0)
                                    else:
                                        tmp_tmp[pp] = pca(PCA_ADI_cube_ori, angle_list=derot_angles, cube_ref=ref_cube, scale_list=None, ncomp=int(npc),
                                                      svd_mode=svd_mode_all[cc], scaling=scaling,  mask_center_px=mask_IWA_px,
                                                      delta_rot=delta_rot, fwhm=fwhm, collapse='median', check_memory=True, 
                                                      full_output=False, verbose=verbose)
                                                      
                                write_fits(outpath_5.format(bin_fac,filt,crop_lab_list[cc])+'final_PCA-{}_full_{}_at_{}as'.format(label_stg,test_pcs_str,test_rad_str)+label_filt+'.fits', tmp_tmp)
                                write_fits(outpath_5.format(bin_fac,filt,crop_lab_list[cc])+'final_PCA-{}_full_npc_id_at_{}as'.format(label_stg,test_rad_str)+label_filt+'.fits', id_npc_full_df)           
                                
                                ### Convolution
                                if (not isfile(outpath_5.format(bin_fac,filt,crop_lab_list[cc])+'final_PCA-{}_full_{}_at_{}as'.format(label_stg,test_pcs_str,test_rad_str)+label_filt+'_conv.fits') or overwrite_pp) and do_conv:
                                    tmp = open_fits(outpath_5.format(bin_fac,filt,crop_lab_list[cc])+'final_PCA-{}_full_{}_at_{}as'.format(label_stg,test_pcs_str,test_rad_str)+label_filt+'.fits')
                                    for nn in range(tmp.shape[0]):
                                        tmp[nn] = frame_filter_lowpass(tmp[nn], mode='gauss',  fwhm_size=fwhm, gauss_mode='conv')
                                    write_fits(outpath_5.format(bin_fac,filt,crop_lab_list[cc])+'final_PCA-{}_full_{}_at_{}as'.format(label_stg,test_pcs_str,test_rad_str)+label_filt+'_conv.fits', tmp, verbose=False)
                                ### SNR map  
                                if (not isfile(outpath_5.format(bin_fac,filt,crop_lab_list[cc])+'final_PCA-{}_full_{}_at_{}as'.format(label_stg,test_pcs_str,test_rad_str)+label_filt+'_snrmap.fits') or overwrite_pp) and do_snr_map[1] and cc==0:
                                    tmp = open_fits(outpath_5.format(bin_fac,filt,crop_lab_list[cc])+'final_PCA-{}_full_{}_at_{}as'.format(label_stg,test_pcs_str,test_rad_str)+label_filt+'.fits')
                                    #rad_in = mask_IWA # we comment it for a better visualization of the snr map (there are spurious values in the center)
                                    #rad_in = 1.5
                                    tmp_tmp = np.ones_like(tmp)
                                    for pp in range(tmp.shape[0]):
                                        tmp[pp] = snrmap(tmp[pp], fwhm, plot=False)
                                    tmp = mask_circle(tmp,mask_IWA_px)#rad_in*fwhm)
                                    write_fits(outpath_5.format(bin_fac,filt,crop_lab_list[cc])+'final_PCA-{}_full_{}_at_{}as'.format(label_stg,test_pcs_str,test_rad_str)+label_filt+'_snrmap.fits', tmp, verbose=False)            
                        
                        ######################## 8. PCA-ADI annular #######################
                        if do_pca_ann and cc == 0 and bin_fac == np.amax(bin_fac_list):  # should only be done for cropped and bin cube
                            
                            #PCA_ADI_cube = vip.fits.open_fits(outpath_5.format(bin_fac,filt,crop_lab_list[cc])+'7_final_crop_PCA_cube'+label_filt+'.fits')
                            if npc_ann is None:                    
                                # First let's readapt the number of pcs to be tested
                                test_pcs_ann = test_pcs_ann_all[cc]   
                                test_pcs_str_list = [str(x) for x in test_pcs_ann]
                                ntest_pcs = len(test_pcs_ann)
                                test_pcs_str = "npc"+"-".join(test_pcs_str_list)
                            else:
                                test_pcs_ann = npc_ann[ff]
                                test_pcs_str_list = [str(x) for x in test_pcs_ann[-1]]
                                ntest_pcs = len(test_pcs_ann)
                                test_pcs_str = "npc_set"+"-".join(test_pcs_str_list)
                            if flux_weights:
                                if not '_fw' in label_filt:
                                    label_filt+='_fw'                      
                            if not fake_planet:
                                if planet:                
                                    snr_tmp = np.zeros(ntest_pcs)
                                tmp_tmp = np.zeros([ntest_pcs,PCA_ADI_cube.shape[1],PCA_ADI_cube.shape[2]])   
                                if npc_ann is not None:
                                    for pp, npc in enumerate(test_pcs_ann):
                                        _, tmp, tmp_tmp[pp] = pca_annular(PCA_ADI_cube, derot_angles, radius_int=mask_IWA_px, fwhm=fwhm, asize=asize*fwhm, 
                                                                  delta_rot=delta_rot, ncomp=npc, svd_mode=svd_mode_all[1], cube_ref=ref_cube,
                                                                  scaling=scaling, min_frames_lib=max(max(npc),min_fr), max_frames_lib=max(max_fr,max(npc)+1), 
                                                                  collapse='median', full_output=True, verbose=verbose, nproc=nproc)
                                        if flux_weights:
                                            w = starphot/np.nansum(starphot)
                                            for zz in range(starphot.shape[0]):
                                                tmp[zz] = tmp[zz]*w[zz]
                                            tmp_tmp[pp] = np.nansum(tmp,axis=0)
                                else:
                                    for pp, npc in enumerate(test_pcs_ann):
                                        _, tmp, tmp_tmp[pp] = pca_annular(PCA_ADI_cube, derot_angles, radius_int=mask_IWA_px, fwhm=fwhm, asize=asize*fwhm, 
                                                                          delta_rot=delta_rot, ncomp=int(npc), svd_mode=svd_mode_all[1], cube_ref=ref_cube,
                                                                          scaling=scaling, min_frames_lib=max(npc,min_fr), max_frames_lib=max(max_fr,npc+1),
                                                                          collapse='median', full_output=True, verbose=verbose, nproc=nproc)
                                        if flux_weights:
                                            w = starphot/np.nansum(starphot)
                                            for zz in range(starphot.shape[0]):
                                                tmp[zz] = tmp[zz]*w[zz]
                                            tmp_tmp[pp] = np.nansum(tmp,axis=0)
                                if planet:  
                                    snr_tmp[pp] = snr(tmp_tmp[pp], (xx_comp,yy_comp), fwhm, plot=False, exclude_negative_lobes=True,
                                                                          verbose=False)                  
                                if planet:
                                    plt.close() 
                                    plt.figure()
                                    plt.title('SNR for '+sourcename+' b'+details+ '(PCA-{} ann)'.format(label_stg))
                                    plt.ylabel('SNR')
                                    plt.xlabel('npc')  
                                    for pp, npc in enumerate(test_pcs_ann):
                                        if snr_tmp[pp] > 5:
                                            marker = 'go'
                                        elif snr_tmp[pp] > 3:
                                            marker = 'bo'
                                        else:
                                            marker = 'ro'
                                        plt.plot(npc, snr_tmp[pp], marker)   
                                    try:
                                        plt.savefig(outpath_5.format(bin_fac,filt,crop_lab_list[cc])+'SNR_'+source+'_PCA-{}-ann.pdf'.format(label_stg), format='pdf')
                                    except:
                                        pass
                                    write_fits(outpath_5.format(bin_fac,filt,crop_lab_list[cc])+'final_PCA-{}_ann_SNR_'.format(label_stg)+test_pcs_str+label_filt+'.fits', snr_tmp)
                                write_fits(outpath_5.format(bin_fac,filt,crop_lab_list[cc])+'final_PCA-{}_ann_'.format(label_stg)+test_pcs_str+label_filt+'.fits', tmp_tmp)
                                ### Convolution
                                if (not isfile(outpath_5.format(bin_fac,filt,crop_lab_list[cc])+'final_PCA-{}_ann_'.format(label_stg)+test_pcs_str+label_filt+'_conv.fits') or overwrite_pp) and do_conv:
                                    tmp = open_fits(outpath_5.format(bin_fac,filt,crop_lab_list[cc])+'final_PCA-{}_ann_'.format(label_stg)+test_pcs_str+label_filt+'.fits')
                                    for nn in range(tmp.shape[0]):
                                        tmp[nn] = frame_filter_lowpass(tmp[nn], mode='gauss',  fwhm_size=fwhm, gauss_mode='conv')
                                    write_fits(outpath_5.format(bin_fac,filt,crop_lab_list[cc])+'final_PCA-{}_ann_'.format(label_stg)+test_pcs_str+label_filt+'_conv.fits', tmp, verbose=False)
                                ### SNR map  
                                if (not isfile(outpath_5.format(bin_fac,filt,crop_lab_list[cc])+'final_PCA-{}_ann_'.format(label_stg)+test_pcs_str+label_filt+'_snrmap.fits') or overwrite_pp) and do_snr_map[2]:
                                    tmp = open_fits(outpath_5.format(bin_fac,filt,crop_lab_list[cc])+'final_PCA-{}_ann_'.format(label_stg)+test_pcs_str+label_filt+'.fits')
                                    #rad_in = mask_IWA # we comment it for a better visualization of the snr map (there are spurious values in the center)
                                    #rad_in = 1.5
                                    tmp_tmp = np.ones_like(tmp)
                                    for pp in range(tmp.shape[0]):
                                        tmp[pp] = snrmap(tmp[pp], fwhm, plot=False)
                                    tmp = mask_circle(tmp,mask_IWA_px)#rad_in*fwhm)
                                    write_fits(outpath_5.format(bin_fac,filt,crop_lab_list[cc])+'final_PCA-{}_ann_'.format(label_stg)+test_pcs_str+label_filt+'_snrmap.fits', tmp, verbose=False)  
                                if flux_weights:
                                    label_filt = label_filt[:-3]
                            else:
                                snr_tmp_tmp = np.zeros([nspi,ntest_pcs,nfcp])
                                tmp_tmp = np.zeros([ntest_pcs,PCA_ADI_cube.shape[1],PCA_ADI_cube.shape[2]])
                                for ns in range(nspi):
                                    theta0 = th0+ns*th_step
                                    PCA_ADI_cube = open_fits(outpath_5.format(bin_fac,filt,crop_lab_list[cc])+'7_final_crop_PCA_cube'+label_filt+'_fcp_spi{:.0f}.fits'.format(ns))
                                    for pp, npc in enumerate(test_pcs_ann):
                                        if svd_mode_all[1] == 'randsvd':
                                            tmp_tmp_tmp = np.zeros([n_randsvd,PCA_ADI_cube.shape[1],PCA_ADI_cube.shape[2]])
                                            for nr in range(n_randsvd):
                                                tmp_tmp_tmp[nr] = pca_annular(PCA_ADI_cube, derot_angles, radius_int=mask_IWA_px, fwhm=fwhm, asize=asize*fwhm, 
                                                              delta_rot=delta_rot, ncomp=int(npc), svd_mode=svd_mode_all[1], max_frames_lib=max(max_fr,npc+1), 
                                                              cube_ref=ref_cube, scaling=scaling, 
                                                              min_frames_lib=max(npc,min_fr), collapse='median', full_output=False, verbose=verbose, nproc=nproc) 
                                            tmp_tmp[pp] = np.median(tmp_tmp_tmp, axis=0)
                                        else:
                                            _, tmp, tmp_tmp[pp] = pca_annular(PCA_ADI_cube, derot_angles, radius_int=mask_IWA_px, fwhm=fwhm, asize=asize*fwhm, 
                                                              delta_rot=delta_rot, ncomp=int(npc), svd_mode=svd_mode_all[1], max_frames_lib=max(max_fr,npc+1),
                                                              cube_ref=ref_cube, scaling=scaling, min_frames_lib=max(npc,min_fr), collapse='median', full_output=True, verbose=verbose, nproc=nproc)
                                            if flux_weights:
                                                w = starphot/np.nansum(starphot)
                                                for zz in range(starphot.shape[0]):
                                                    tmp[zz] = tmp[zz]*w[zz]
                                                tmp_tmp[pp] = np.nansum(tmp,axis=0)                  
                                        for ff in range(nfcp):
                                            xx_fcp = cx + rad_arr[ff]*np.cos(np.deg2rad(theta0+ff*th_step))
                                            yy_fcp = cy + rad_arr[ff]*np.sin(np.deg2rad(theta0+ff*th_step))
                                            snr_tmp_tmp[ns,pp,ff] = snr(tmp_tmp[pp], (xx_fcp,yy_fcp), fwhm, plot=False, exclude_negative_lobes=True,
                                                                                  verbose=True)   
                                    write_fits(outpath_5.format(bin_fac,filt,crop_lab_list[cc])+'TMP_PCA-{}_ann_'.format(label_stg)+test_pcs_str+label_filt+'_fcp_spi{:.0f}.fits'.format(ns), tmp_tmp)                                             
                                snr_fcp = np.median(snr_tmp_tmp, axis=0)
                                plt.close() 
                                plt.figure()
                                plt.title('SNR for fcps '+details+ '(PCA-{} ann)'.format(label_stg))
                                plt.ylabel('SNR')
                                plt.xlabel('npc')
                                for ff in range(nfcp):
                                    marker = all_markers[ff]
                                    for pp, npc in enumerate(test_pcs_ann):
                                        plt.plot(npc, snr_fcp[pp,ff], marker)
                                try:
                                    plt.savefig(outpath_5.format(bin_fac,filt,crop_lab_list[cc])+'SNR_fcps_PCA-{}-ann.pdf'.format(label_stg), format='pdf')
                                except:
                                    pass
                                write_fits(outpath_5.format(bin_fac,filt,crop_lab_list[cc])+'final_PCA-{}_ann_SNR_fcps_'.format(label_stg)+test_pcs_str+label_filt+'.fits', snr_fcp)   
                                
                                ## Find best npc for each radius
                                for ff in range(nfcp):
                                    idx_best_snr = np.argmax(snr_fcp[:,ff])
                                    id_npc_ann_df[ff] = test_pcs_ann[idx_best_snr]
                                
                                ## 8.3. Final PCA-ADI frames with optimal npcs        
                                tmp_tmp = np.zeros([nfcp,PCA_ADI_cube_ori.shape[1],PCA_ADI_cube_ori.shape[2]])
                                test_pcs_str_list = [str(int(x)) for x in id_npc_ann_df]                               
                                test_pcs_str = "npc_opt"+"-".join(test_pcs_str_list)
                                test_rad_str_list = ["{:.1f}".format(x) for x in rad_arr*plsc]                               
                                test_rad_str = "rad"+"-".join(test_rad_str_list)
                                for pp, npc in enumerate(id_npc_ann_df):
                                    if svd_mode_all[1] == 'randsvd':
                                        tmp_tmp_tmp = np.zeros([n_randsvd,PCA_ADI_cube_ori.shape[1],PCA_ADI_cube_ori.shape[2]])
                                        for nr in range(n_randsvd):
                                            tmp_tmp_tmp[nr] = pca_annular(PCA_ADI_cube_ori, derot_angles, radius_int=mask_IWA_px, fwhm=fwhm, asize=asize*fwhm, 
                                                                  delta_rot=delta_rot, ncomp=int(npc), svd_mode=svd_mode_all[1], max_frames_lib=max(max_fr,npc+1),
                                                                  cube_ref=ref_cube, scaling=scaling, min_frames_lib=max(npc,min_fr), collapse='median', full_output=False, verbose=verbose, nproc=nproc)
                                        tmp_tmp[pp] = np.median(tmp_tmp_tmp, axis=0)
                                    else:
                                        _, tmp, tmp_tmp[pp] = pca_annular(PCA_ADI_cube_ori, derot_angles, radius_int=mask_IWA_px, fwhm=fwhm, asize=asize*fwhm, 
                                                                  delta_rot=delta_rot, ncomp=int(npc), svd_mode=svd_mode_all[1], max_frames_lib=max(max_fr,npc+1),
                                                                  cube_ref=ref_cube, scaling=scaling, min_frames_lib=max(npc,min_fr), collapse='median', full_output=True, verbose=verbose, nproc=nproc)
                                        if flux_weights:
                                            w = starphot/np.nansum(starphot)
                                            for zz in range(starphot.shape[0]):
                                                tmp[zz] = tmp[zz]*w[zz]
                                            tmp_tmp[pp] = np.nansum(tmp,axis=0)                          
                                write_fits(outpath_5.format(bin_fac,filt,crop_lab_list[cc])+'final_PCA-{}_ann_{}_at_{}as'.format(label_stg,test_pcs_str,test_rad_str)+label_filt+'.fits', tmp_tmp)
                                write_fits(outpath_5.format(bin_fac,filt,crop_lab_list[cc])+'final_PCA-{}_ann_npc_id_at_{}as'.format(label_stg,test_rad_str)+label_filt+'.fits', id_npc_ann_df)
            
                            
                                ### Convolution
                                if (not isfile(outpath_5.format(bin_fac,filt,crop_lab_list[cc])+'final_PCA-{}_ann_{}_at_{}as'.format(label_stg,test_pcs_str,test_rad_str)+label_filt+'_conv.fits') or overwrite_pp) and do_conv:
                                    tmp = open_fits(outpath_5.format(bin_fac,filt,crop_lab_list[cc])+'final_PCA-{}_ann_{}_at_{}as'.format(label_stg,test_pcs_str,test_rad_str)+label_filt+'.fits')
                                    for nn in range(tmp.shape[0]):
                                        tmp[nn] = frame_filter_lowpass(tmp[nn], mode='gauss', fwhm_size=fwhm, gauss_mode='conv')
                                    write_fits(outpath_5.format(bin_fac,filt,crop_lab_list[cc])+'final_PCA-{}_ann_{}_at_{}as'.format(label_stg,test_pcs_str,test_rad_str)+label_filt+'_conv.fits', tmp, verbose=False)
                                ### SNR map  
                                if (not isfile(outpath_5.format(bin_fac,filt,crop_lab_list[cc])+'final_PCA-{}_ann_{}_at_{}as'.format(label_stg,test_pcs_str,test_rad_str)+label_filt+'_snrmap.fits') or overwrite_pp) and do_snr_map[2]:
                                    tmp = open_fits(outpath_5.format(bin_fac,filt,crop_lab_list[cc])+'final_PCA-{}_ann_{}_at_{}as'.format(label_stg,test_pcs_str,test_rad_str)+label_filt+'.fits')
                                    #rad_in = mask_IWA # we comment it for a better visualization of the snr map (there are spurious values in the center)
                                    #rad_in = 1.5
                                    tmp_tmp = np.ones_like(tmp)
                                    for pp in range(tmp.shape[0]):
                                        tmp[pp] = snrmap(tmp[pp], fwhm, plot=False)
                                    tmp = mask_circle(tmp,mask_IWA_px)#rad_in*fwhm)
                                    write_fits(outpath_5.format(bin_fac,filt,crop_lab_list[cc])+'final_PCA-{}_ann_{}_at_{}as'.format(label_stg,test_pcs_str,test_rad_str)+label_filt+'_snrmap.fits', tmp, verbose=False)                     
                            if flux_weights:
                                label_filt = label_filt[:-3]
                            
                        ###################### 9. Final contrast curve #####################
                        if fake_planet:
                            if planet:
                                # SUBTRACT THE PLANET FROM THE CUBE
                                cube_emp = cube_planet_free(planet_parameter, PCA_ADI_cube_ori, derot_angles, psfn, plsc)
                                PCA_ADI_cube_ori = cube_emp
                                label_emp = '_empty'+label_filt
                            else:
                                label_emp = label_filt                # 9.1 Recompute the contrast curve for optimal npcs
                        if do_pca_full and fake_planet:
                            df_list = []
                            rsvd_list = []
                            for rr, rad in enumerate(rad_arr):
                                if svd_mode_all[cc] == 'randsvd':
                                    for nr in range(n_randsvd):
                                        pn_contr_curve_full_rr_tmp = contrast_curve(PCA_ADI_cube_ori, derot_angles, psfn,
                                                                                    fwhm, plsc, starphot=starphot,
                                                                                    algo=pca, sigma=5., nbranch=n_br,
                                                                                    theta=0, inner_rad=1, wedge=wedge, fc_snr=fc_snr,
                                                                                    student=True, transmission=transmission,
                                                                                    plot=True, dpi=100,cube_ref=ref_cube,scaling=scaling,
                                                                                    verbose=verbose, ncomp=int(id_npc_full_df[rr]), svd_mode=svd_mode_all[cc])
                                        rsvd_list.append(pn_contr_curve_full_rr_tmp)
                                    pn_contr_curve_full_rr = pn_contr_curve_full_rr_tmp.copy()
                                    for jj in range(pn_contr_curve_full_rr.shape[0]): 
                                        sensitivities = []
                                        for nr in range(n_randsvd):
                                            sensitivities.append(rsvd_list[nr]['sensitivity_student'][jj])
                                        print("Sensitivities at {}: ".format(rsvd_list[rr]['distance'][jj]), sensitivities)
                                        idx_min = np.argmin(sensitivities)
                                        pn_contr_curve_full_rr['sensitivity_student'][jj] = rsvd_list[idx_min]['sensitivity_student'][jj]
                                        pn_contr_curve_full_rr['sensitivity_gaussian'][jj] = rsvd_list[idx_min]['sensitivity_gaussian'][jj]
                                        pn_contr_curve_full_rr['throughput'][jj] = rsvd_list[idx_min]['throughput'][jj]
                                        pn_contr_curve_full_rr['noise'][jj] = rsvd_list[idx_min]['noise'][jj]
                                        pn_contr_curve_full_rr['sigma corr'][jj] = rsvd_list[idx_min]['sigma corr'][jj]
                                else:
                                    pn_contr_curve_full_rr = contrast_curve(PCA_ADI_cube_ori, derot_angles, psfn,
                                                                            fwhm, plsc, starphot=starphot,
                                                                            algo=pca, sigma=5., nbranch=n_br,
                                                                            theta=0, inner_rad=1, wedge=wedge,fc_snr=fc_snr,
                                                                            student=True, transmission=transmission,
                                                                            plot=True, dpi=100, cube_ref=ref_cube,scaling=scaling,
                                                                            verbose=verbose, ncomp=int(id_npc_full_df[rr]), svd_mode=svd_mode_all[cc])
                                DF.to_csv(pn_contr_curve_full_rr, path_or_buf=outpath_5.format(bin_fac,filt,crop_lab_list[cc])+'contrast_curve_PCA-{}-full_optimal_at_{:.1f}as{}.csv'.format(label_stg,rad*plsc,label_emp), sep=',', na_rep='', float_format=None)
                                df_list.append(pn_contr_curve_full_rr)
                            pn_contr_curve_full_opt = pn_contr_curve_full_rr.copy()
            
                            for jj in range(pn_contr_curve_full_opt.shape[0]):
                                sensitivities = []
                                for rr, rad in enumerate(rad_arr):
                                    sensitivities.append(df_list[rr]['sensitivity_student'][jj])
                                print("Sensitivities at {}: ".format(df_list[rr]['distance'][jj]), sensitivities)
                                idx_min = np.argmin(sensitivities)
                                pn_contr_curve_full_opt['sensitivity_student'][jj] = df_list[idx_min]['sensitivity_student'][jj]
                                pn_contr_curve_full_opt['sensitivity_gaussian'][jj] = df_list[idx_min]['sensitivity_gaussian'][jj]
                                pn_contr_curve_full_opt['throughput'][jj] = df_list[idx_min]['throughput'][jj]
                                pn_contr_curve_full_opt['noise'][jj] = df_list[idx_min]['noise'][jj]
                                pn_contr_curve_full_opt['sigma corr'][jj] = df_list[idx_min]['sigma corr'][jj]
                            DF.to_csv(pn_contr_curve_full_opt, path_or_buf=outpath_5.format(bin_fac,filt,crop_lab_list[cc])+'final_optimal_contrast_curve_PCA-{}-full'.format(label_stg)+label_emp+'.csv', sep=',', na_rep='', float_format=None)
                            arr_dist = np.array(pn_contr_curve_full_opt['distance'])
                            arr_contrast = np.array(pn_contr_curve_full_opt['sensitivity_student'])
                            for ff in range(nfcp):
                                idx = find_nearest(arr_dist, rad_arr[ff])
                                sensitivity_5sig_full_df[ff] = arr_contrast[idx]
                                           
                        if do_pca_ann and cc == 0 and bin_fac == np.amax(bin_fac_list) and fake_planet:                                              
                            df_list = []
                            rsvd_list = []
                            for rr, rad in enumerate(rad_arr):
                                if svd_mode_all[1] == 'randsvd':
                                    for nr in range(n_randsvd):
                                        pn_contr_curve_ann_rr_tmp = contrast_curve(PCA_ADI_cube_ori, derot_angles, psfn,
                                                                                   fwhm, plsc, starphot=starphot,
                                                                                   algo=pca_annular, sigma=5., nbranch=nspi,
                                                                                   theta=0, inner_rad=1, wedge=wedge, fc_snr=fc_snr,
                                                                                   student=True, transmission=transmission,
                                                                                   plot=True, dpi=100,
                                                                                   verbose=verbose, ncomp=int(id_npc_ann_df[rr]), svd_mode=svd_mode_all[1],
                                                                                   radius_int=mask_IWA_px, asize=asize*fwhm,
                                                                                   delta_rot=delta_rot, cube_ref=ref_cube,scaling=scaling,
                                                                                   min_frames_lib=max(id_npc_ann_df[rr],min_fr),
                                                                                   max_frames_lib=max(max_fr,id_npc_ann_df[rr]+1))
                                        rsvd_list.append(pn_contr_curve_ann_rr_tmp)
                                    pn_contr_curve_ann_rr = pn_contr_curve_ann_rr_tmp.copy()
                                    for jj in range(pn_contr_curve_ann_rr.shape[0]):
                                        sensitivities = []
                                        for nr in range(n_randsvd):
                                            sensitivities.append(rsvd_list[nr]['sensitivity_student'][jj])
                                        print("Sensitivities at {}: ".format(rsvd_list[rr]['distance'][jj]), sensitivities)
                                        idx_min = np.argmin(sensitivities)
                                        pn_contr_curve_ann_rr['sensitivity_student'][jj] = rsvd_list[idx_min]['sensitivity_student'][jj]
                                        pn_contr_curve_ann_rr['sensitivity_gaussian'][jj] = rsvd_list[idx_min]['sensitivity_gaussian'][jj]
                                        pn_contr_curve_ann_rr['throughput'][jj] = rsvd_list[idx_min]['throughput'][jj]
                                        pn_contr_curve_ann_rr['noise'][jj] = rsvd_list[idx_min]['noise'][jj]
                                        pn_contr_curve_ann_rr['sigma corr'][jj] = rsvd_list[idx_min]['sigma corr'][jj]
                                else:
                                    pn_contr_curve_ann_rr = contrast_curve(PCA_ADI_cube_ori, derot_angles, psfn,
                                                                           fwhm, plsc, starphot=starphot,
                                                                           algo=pca_annular, sigma=5., nbranch=n_br,
                                                                           theta=0, inner_rad=1, wedge=wedge,fc_snr=fc_snr,
                                                                           student=True, transmission=transmission,
                                                                           plot=True, dpi=100,
                                                                           verbose=verbose, ncomp=int(id_npc_ann_df[rr]), svd_mode=svd_mode_all[1],
                                                                           radius_int=mask_IWA_px, asize=asize*fwhm,
                                                                           delta_rot=delta_rot, cube_ref=ref_cube, scaling=scaling,
                                                                           min_frames_lib=max(id_npc_ann_df[rr],min_fr), max_frames_lib=max(max_fr,id_npc_ann_df[rr]+1))
                                DF.to_csv(pn_contr_curve_ann_rr, path_or_buf=outpath_5.format(bin_fac,filt,crop_lab_list[cc])+'contrast_curve_PCA-{}-ann_optimal_at_{:.1f}as{}.csv'.format(label_stg,rad*plsc,label_emp), sep=',', na_rep='', float_format=None)
                                df_list.append(pn_contr_curve_ann_rr)
                            pn_contr_curve_ann_opt = pn_contr_curve_ann_rr.copy()
            
                            for jj in range(pn_contr_curve_ann_opt.shape[0]):  
                                sensitivities = []
                                for rr, rad in enumerate(rad_arr):
                                    sensitivities.append(df_list[rr]['sensitivity_student'][jj])
                                print("Sensitivities at {}: ".format(df_list[rr]['distance'][jj]), sensitivities)
                                idx_min = np.argmin(sensitivities)
                                pn_contr_curve_ann_opt['sensitivity_student'][jj] = df_list[idx_min]['sensitivity_student'][jj]
                                pn_contr_curve_ann_opt['sensitivity_gaussian'][jj] = df_list[idx_min]['sensitivity_gaussian'][jj]
                                pn_contr_curve_ann_opt['throughput'][jj] = df_list[idx_min]['throughput'][jj]
                                pn_contr_curve_ann_opt['noise'][jj] = df_list[idx_min]['noise'][jj]
                                pn_contr_curve_ann_opt['sigma corr'][jj] = df_list[idx_min]['sigma corr'][jj]
                            DF.to_csv(pn_contr_curve_ann_opt, path_or_buf=outpath_5.format(bin_fac,filt,crop_lab_list[cc])+'final_optimal_contrast_curve_PCA-{}-ann{}.csv'.format(label_stg,label_emp), sep=',', na_rep='', float_format=None)
                            arr_dist = np.array(pn_contr_curve_ann_opt['distance'])
                            arr_contrast = np.array(pn_contr_curve_ann_opt['sensitivity_student'])
                            for ff in range(nfcp):
                                idx = find_nearest(arr_dist, rad_arr[ff])
                                sensitivity_5sig_ann_df[ff] = arr_contrast[idx]      
            
                        if fake_planet:
                            plt.close()              
                            plt.figure()
                            plt.title('5-sigma contrast curve for '+source+details)
                            plt.ylabel('Contrast')
                            plt.xlabel('Separation (arcsec)')
                            if do_adi:
                                plt.semilogy(pn_contr_curve_adi['distance']*plsc, pn_contr_curve_adi['sensitivity_student'],'r', linewidth=2, label='median-ADI (Student correction)')
                            if do_pca_full and fake_planet:
                                plt.semilogy(pn_contr_curve_full_opt['distance']*plsc, pn_contr_curve_full_opt['sensitivity_student'],'b', linewidth=2, label='PCA-{} full frame (Student, lapack)'.format(label_stg))
                            # if cc == 0 and fake_planet:
                            #     pn_contr_curve_full_rsvd_opt = pandas.read_csv(outpath_5.format(bin_fac,filt,crop_lab_list[cc])+'TMP_first_guess_contrast_curve_PCA-{}-full.csv'.format(label_stg))
                            #     plt.semilogy(pn_contr_curve_full_rsvd_opt['distance']*plsc, pn_contr_curve_full_rsvd_opt['sensitivity_student'],'c', linewidth=2, label='PCA-{} full frame (Student, randsvd)'.format(label_stg))
                            if do_pca_ann and cc == 0 and bin_fac == np.amax(bin_fac_list) and fake_planet:
                                plt.semilogy(pn_contr_curve_ann_opt['distance']*plsc, pn_contr_curve_ann_opt['sensitivity_student'],'g', linewidth=2, label='PCA-{} annular (Student)'.format(label_stg))                                    
                            plt.legend()
                            try:
                                plt.savefig(outpath_5.format(bin_fac,filt,crop_lab_list[cc])+'contr_curves'+label_emp+'.pdf', format='pdf')
                            except:
                                pass
                            
                            plt.close()              
                            plt.figure()
                            plt.title('5-sigma contrast curve for '+source+details)
                            plt.ylabel('Contrast')
                            plt.gca().invert_yaxis()
                            plt.xlabel('Separation (arcsec)')
                            if do_adi:
                                plt.plot(pn_contr_curve_adi['distance']*plsc, -2.5*np.log10(pn_contr_curve_adi['sensitivity_student']),'r', linewidth=2, label='median-ADI (Student)')
                            if do_pca_full and fake_planet:
                                plt.plot(pn_contr_curve_full_opt['distance']*plsc, -2.5*np.log10(pn_contr_curve_full_opt['sensitivity_student']),'b', linewidth=2, label='PCA-{} full frame (Student, lapack)'.format(label_stg))
                            # if cc == 0 and fake_planet:
                            #     plt.plot(pn_contr_curve_full_rsvd_opt['distance']*plsc, -2.5*np.log10(pn_contr_curve_full_rsvd_opt['sensitivity_student']),'c', linewidth=2, label='PCA-{} full frame (Student, {})'.format(label_stg, svd_mode_all[0]))
                            if do_pca_ann and cc == 0 and bin_fac == np.amax(bin_fac_list) and fake_planet:
                                plt.plot(pn_contr_curve_ann_opt['distance']*plsc, -2.5*np.log10(pn_contr_curve_ann_opt['sensitivity_student']),
                                         'g', linewidth=2, label='PCA-{} annular - npc={:.0f} (Student)'.format(label_stg,id_npc_ann_df[rr]))                                    
                            plt.legend()
                            try:
                                plt.savefig(outpath_5.format(bin_fac,filt,crop_lab_list[cc])+'contr_curves_MAG'+label_emp+'.pdf', format='pdf')
                            except:
                                pass
              
                            # WRITE THE CSV FILE
                            datafr1 = DF(data=nfcp_df, columns=['Index of injected fcp'])
                            datafr2 = DF(data=rad_arr, columns=['Radius (px)'])  
                            datafr3 = DF(data=rad_arr*plsc, columns=['Radius (as)'])  
                            datafr = datafr1.join(datafr2).join(datafr3)
                            #datafr7 = DF(data=id_snr_adi_df, columns=['Ideal SNR (m-adi)'])
                            if do_adi:
                                datafr8 = DF(data=sensitivity_5sig_adi_df, columns=["5-sig Student sensitivity (m-adi)"])
                                datafr = datafr.join(datafr8)
                            if do_pca_full:
                                datafr10 = DF(data=id_npc_full_df, columns=["Ideal npc (PCA-{} full)".format(label_stg)])
                                datafr12 = DF(data=sensitivity_5sig_full_df, columns=["5-sig Student sensitivity (PCA-{} full, lapack)".format(label_stg)])
                                datafr = datafr.join(datafr10).join(datafr12)
                            # if cc == 0 and do_pca_full:
                            #     datafr11 = DF(data=sensitivity_5sig_full_rsvd_df, columns=["5-sig Student sensitivity (PCA-{} full, {})".format(label_stg,svd_mode_all[0])])
                            #     datafr = datafr.join(datafr11)
                            if do_pca_ann and cc == 0 and bin_fac == np.amax(bin_fac_list) and fake_planet:
                                datafr14 = DF(data=id_npc_ann_df, columns=["Ideal npc (PCA-{} ann)".format(label_stg)])
                                datafr16 = DF(data=sensitivity_5sig_ann_df, columns=["5-sig Student sensitivity (PCA-{} ann)".format(label_stg)])
                                datafr = datafr.join(datafr14).join(datafr16)
                            #datafr = datafr1.join(datafr2).join(datafr3).join(datafr4).join(datafr5).join(datafr6).join(datafr7).join(datafr8).join(datafr9).join(datafr10).join(datafr11).join(datafr12).join(datafr13).join(datafr14).join(datafr15).join(datafr16).join(datafr17)
                            #datafr = datafr1.join(datafr2).join(datafr3).join(datafr4).join(datafr5).join(datafr6).join(datafr8).join(datafr9).join(datafr10).join(datafr11).join(datafr12).join(datafr13).join(datafr14).join(datafr15).join(datafr16).join(datafr17)
                            DF.to_csv(datafr, path_or_buf=outpath_5.format(bin_fac,filt,crop_lab_list[cc])+'_final_results'+label_emp+'.csv', sep=',', na_rep='', float_format=None)
                            
                        counter += 1
                        
                # 4. iterative ADI or (A)RDI on smallest crop, if requested
                if cc ==0 and n_it>0:
                    for ff, filt in enumerate(filters):
                        plsc = float(plsc_ori[ff])
                        if not isdir(outpath_5.format(bin_fac,filt,crop_lab_list[cc])):
                            os.system("mkdir "+outpath_5.format(bin_fac,filt,crop_lab_list[cc]))
                        fwhm = float(open_fits(outpath_2+final_fwhmname+"{}.fits".format(filt))[0])
                        flux = float(open_fits(outpath_2+final_fluxname+"{}.fits".format(filt))[0])
                        if cc == 0 or not isfile(outpath_2+final_cubename+"_full{}.fits".format(filt)):
                            ADI_cube= open_fits(outpath_2+final_cubename+"{}.fits".format(filt))
                        else:
                            ADI_cube= open_fits(outpath_2+final_cubename+"_full{}.fits".format(filt))
                        if ref_cube_name is not None:
                            ref_cube = open_fits(ref_cube_name.format(filt))
                                
                        derot_angles = open_fits(outpath_2+final_anglename+"{}.fits".format(filt))
    #                    if derot_name == "rotnth":
    #                        derot_angles*=-1
                        psfn = open_fits(outpath_2+final_psfname+"{}.fits".format(filt)) # this has all the unsat psf frames
                        #3a. pca it in full (cropped) frames
                        if do_pca_full and not do_pca_1zone:
                            final_imgs = np.zeros([len(test_pcs_full),ADI_cube.shape[1],ADI_cube.shape[2]])
                            #wmean_imgs = np.zeros_like(final_imgs)
                            if mask_PCA is None:
                                mask_rdi = None
                            else:
                                mask_tmp = np.ones_like(ADI_cube[0])
                                mask_rdi = mask_circle(mask_tmp, mask_PCA, fillwith=0, mode='in')
                            for pp, npc in enumerate(test_pcs_full):
                                res = pca_it(ADI_cube, derot_angles, cube_ref=ref_cube, mask_center_px=mask_IWA_px, fwhm=fwhm,  
                                             strategy=strategy, thr=thr_it, n_it=n_it, n_neigh=n_neigh, ncomp=npc, scaling=scaling, 
                                             thru_corr=throughput_corr, psfn=psfn, n_br=n_br, mask_rdi=mask_rdi, full_output=True)
                                final_imgs[pp], it_cube, sig_cube, res, res_der, thru_2d_cube, stim_cube, it_cube_nd = res
                                write_fits(outpath_5.format(bin_fac,filt,crop_lab_list[cc])+"TMP_PCA-{}_it{:.0f}_thr{:.1f}_npc{:.0f}_{}_last_res.fits".format(label_stg,n_it,thr_it,npc,filt), res)
                                write_fits(outpath_5.format(bin_fac,filt,crop_lab_list[cc])+"TMP_PCA-{}_it{:.0f}_thr{:.1f}_npc{:.0f}_{}_last_res_der.fits".format(label_stg,n_it,thr_it,npc,filt), res_der)
                                write_fits(outpath_5.format(bin_fac,filt,crop_lab_list[cc])+"TMP_PCA-{}_it{:.0f}_thr{:.1f}_npc{:.0f}_{}_it_cube.fits".format(label_stg,n_it,thr_it,npc,filt), it_cube)
                                write_fits(outpath_5.format(bin_fac,filt,crop_lab_list[cc])+"TMP_PCA-{}_it{:.0f}_thr{:.1f}_npc{:.0f}_{}_it_cube_nd.fits".format(label_stg,n_it,thr_it,npc,filt), it_cube_nd)
                                write_fits(outpath_5.format(bin_fac,filt,crop_lab_list[cc])+"TMP_PCA-{}_it{:.0f}_thr{:.1f}_npc{:.0f}_{}_sig_cube.fits".format(label_stg,n_it,thr_it,npc,filt), sig_cube)
                                write_fits(outpath_5.format(bin_fac,filt,crop_lab_list[cc])+"TMP_PCA-{}_it{:.0f}_thr{:.1f}_npc{:.0f}_{}_thru_2d_cube.fits".format(label_stg,n_it,thr_it,npc,filt), thru_2d_cube)
                                write_fits(outpath_5.format(bin_fac,filt,crop_lab_list[cc])+"TMP_PCA-{}_it{:.0f}_thr{:.1f}_npc{:.0f}_{}_stim_cube.fits".format(label_stg,n_it,thr_it,npc,filt), stim_cube)
                                # stim = compute_stim_map(res_der)
                                # inv_stim = compute_inverse_stim_map(res, derot_angles)
                                # norm_stim = stim/np.percentile(inv_stim,99.7)
                                # write_fits(outpath_5.format(bin_fac,filt,crop_lab_list[cc])+"TMP_PCA-{}_it{:.0f}_thr{:.1f}_npc{:.0f}_{}_stim.fits".format(strategy,n_it,thr_it,npc,filt), 
                                #            np.array([stim, inv_stim, norm_stim]))
                                # FIRST: define mask following spirals: max stim map (2-5)!
                                # good_mask = np.zeros_like(stim)
                                # good_mask[np.where(norm_stim>1)]=1
                                # ccorr_coeff = cube_distance(res_der,final_imgs[pp],mode='mask',mask=good_mask)
                                # norm_cc = ccorr_coeff/np.sum(ccorr_coeff)
                                # wmean_imgs[pp] = cube_collapse(res_der,mode='wmean',w=norm_cc)
                            #write_fits(outpath_3.format(data_folder)+"final_PCA-RDI_it{:.0f}_thr{:.1f}_{:.0f}-{:.0f}_{}_wmean.fits".format(label_stg,n_it,thr,test_npcs[0], test_npcs[-1],filt), wmean_imgs)
                            write_fits(outpath_5.format(bin_fac,filt,crop_lab_list[cc])+"final_PCA-{}_it{:.0f}_thr{:.1f}_{:.0f}-{:.0f}_{}.fits".format(label_stg,n_it,thr_it,test_pcs_full[0],test_pcs_full[-1],filt), final_imgs)
                            # correction by AGPM transmission
                            # for pp, npc in enumerate(test_npcs):
                            #     wmean_imgs[pp]/=transmission_2d
                            #     final_imgs[pp]/=transmission_2d
                            # #write_fits(outpath_5.format(bin_fac,filt,crop_lab_list[cc])+"final_PCA-RDI_it{:.0f}_thr{:.1f}_{:.0f}-{:.0f}_ann{:.0f}_wmean_AGPMcorr.fits".format(n_it,thr_it,test_npcs[0], test_npcs[-1],ann_sz), wmean_imgs)
                            # write_fits(outpath_5.format(bin_fac,filt,crop_lab_list[cc])+"final_PCA-RDI_it{:.0f}_thr{:.1f}_ann_{:.0f}-{:.0f}_ann{:.0f}_AGPMcorr.fits".format(n_it,thr_it,test_npcs[0],test_npcs[-1],ann_sz), final_imgs)

                        if do_pca_ann and not do_pca_1zone:
                            final_imgs = np.zeros([len(test_pcs_ann),ADI_cube.shape[1],ADI_cube.shape[2]])
                            #wmean_imgs = np.zeros_like(final_imgs)
                            for pp, npc in enumerate(test_pcs_ann):
                                res = pca_annular_it(ADI_cube, derot_angles, cube_ref=ref_cube, radius_int=mask_IWA_px, fwhm=fwhm,  
                                                      thr=thr_it, asize=int(asize*fwhm), n_it=n_it, ncomp=npc, 
                                                      thru_corr=throughput_corr, psfn=psfn, n_br=n_br,
                                                      delta_rot=delta_rot, scaling=scaling, full_output=True)
                                final_imgs[pp], it_cube, sig_cube, res, res_der = res
                                write_fits(outpath_5.format(bin_fac,filt,crop_lab_list[cc])+"TMP_PCA-{}ann_it{:.0f}_thr{:.1f}_npc{:.0f}_{}_last_res.fits".format(label_stg,n_it,thr_it,npc,filt), res)
                                write_fits(outpath_5.format(bin_fac,filt,crop_lab_list[cc])+"TMP_PCA-{}ann_it{:.0f}_thr{:.1f}_npc{:.0f}_{}_last_res_der.fits".format(label_stg,n_it,thr_it,npc,filt), res_der)
                                write_fits(outpath_5.format(bin_fac,filt,crop_lab_list[cc])+"TMP_PCA-{}ann_it{:.0f}_thr{:.1f}_npc{:.0f}_{}_it_cube.fits".format(label_stg,n_it,thr_it,npc,filt), it_cube)
                                write_fits(outpath_5.format(bin_fac,filt,crop_lab_list[cc])+"TMP_PCA-{}ann_it{:.0f}_thr{:.1f}_npc{:.0f}_{}_sig_cube.fits".format(label_stg,n_it,thr_it,npc,filt), sig_cube)
                                stim = compute_stim_map(res_der)
                                inv_stim = compute_inverse_stim_map(res, derot_angles)
                                norm_stim = stim/np.percentile(inv_stim,99.7)
                                write_fits(outpath_5.format(bin_fac,filt,crop_lab_list[cc])+"TMP_PCA-{}ann_it{:.0f}_thr{:.1f}_npc{:.0f}_{}_stim.fits".format(label_stg,n_it,thr_it,npc,filt), 
                                           np.array([stim, inv_stim, norm_stim]))
                                # FIRST: define mask following spirals: max stim map (2-5)!
                                # good_mask = np.zeros_like(stim)
                                # good_mask[np.where(norm_stim>1)]=1
                                # ccorr_coeff = cube_distance(res_der,final_imgs[pp],mode='mask',mask=good_mask)
                                # norm_cc = ccorr_coeff/np.sum(ccorr_coeff)
                                # wmean_imgs[pp] = cube_collapse(res_der,mode='wmean',w=norm_cc)
                            #write_fits(outpath_3.format(data_folder)+"final_PCA-RDI_it{:.0f}_thr{:.1f}_{:.0f}-{:.0f}_{}_wmean.fits".format(label_stg,n_it,thr,test_npcs[0], test_npcs[-1],filt), wmean_imgs)
                            write_fits(outpath_5.format(bin_fac,filt,crop_lab_list[cc])+"final_PCA-{}ann_it{:.0f}_thr{:.1f}_ann_{:.0f}-{:.0f}_{}.fits".format(label_stg,n_it,thr_it,test_pcs_ann[0],test_pcs_ann[-1],filt), final_imgs)
                            # correction by AGPM transmission
                            # for pp, npc in enumerate(test_npcs):
                            #     wmean_imgs[pp]/=transmission_2d
                            #     final_imgs[pp]/=transmission_2d
                            # #write_fits(outpath_5.format(bin_fac,filt,crop_lab_list[cc])+"final_PCA-RDI_it{:.0f}_thr{:.1f}_{:.0f}-{:.0f}_ann{:.0f}_wmean_AGPMcorr.fits".format(n_it,thr_it,test_npcs[0], test_npcs[-1],ann_sz), wmean_imgs)
                            # write_fits(outpath_5.format(bin_fac,filt,crop_lab_list[cc])+"final_PCA-RDI_it{:.0f}_thr{:.1f}_ann_{:.0f}-{:.0f}_ann{:.0f}_AGPMcorr.fits".format(n_it,thr_it,test_npcs[0],test_npcs[-1],ann_sz), final_imgs)
                        
                        if do_feves and not do_pca_1zone:
                            final_imgs = np.zeros([len(test_pcs_ann),ADI_cube.shape[1],ADI_cube.shape[2]])
                            #wmean_imgs = np.zeros_like(final_imgs)
                            for pp, npc in enumerate(test_pcs_ann):
                                res = feves(ADI_cube, derot_angles, cube_ref=ref_cube, 
                                            ncomp=npc, n_it=n_it, fwhm=fwhm, thr=thr_it, 
                                            thr_per_ann=False, asizes=[16,8,4,2,2,2], 
                                            n_segments=[1,1,1,1,3,6], thru_corr=False, 
                                            n_neigh=0, strategy='ADI', psfn=None, n_br=6, 
                                            radius_int=mask_IWA_px, delta_rot=delta_rot, 
                                            svd_mode=svd_mode, nproc=1, 
                                            min_frames_lib=2, max_frames_lib=200, 
                                            tol=1e-1, scaling=scaling, imlib='opencv', 
                                            interpolation='lanczos4', collapse='median', 
                                            full_output=True, verbose=True, weights=None, 
                                            add_res=False, interp_order=2, rtol=1e-2, atol=1)
                                final_imgs[pp], it_cube, sig_cube, res, res_der = res
                                write_fits(outpath_5.format(bin_fac,filt,crop_lab_list[cc])+"TMP_feves-{}_it{:.0f}_thr{:.1f}_npc{:.0f}_{}_last_res.fits".format(label_stg,n_it,thr_it,npc,filt), res)
                                write_fits(outpath_5.format(bin_fac,filt,crop_lab_list[cc])+"TMP-feves-{}_it{:.0f}_thr{:.1f}_npc{:.0f}_{}_last_res_der.fits".format(label_stg,n_it,thr_it,npc,filt), res_der)
                                write_fits(outpath_5.format(bin_fac,filt,crop_lab_list[cc])+"TMP_feves-{}_it{:.0f}_thr{:.1f}_npc{:.0f}_{}_it_cube.fits".format(label_stg,n_it,thr_it,npc,filt), it_cube)
                                write_fits(outpath_5.format(bin_fac,filt,crop_lab_list[cc])+"TMP_feves-{}_it{:.0f}_thr{:.1f}_npc{:.0f}_{}_sig_cube.fits".format(label_stg,n_it,thr_it,npc,filt), sig_cube)
                                stim = compute_stim_map(res_der)
                                inv_stim = compute_inverse_stim_map(res, derot_angles)
                                norm_stim = stim/np.percentile(inv_stim,99.7)
                                write_fits(outpath_5.format(bin_fac,filt,crop_lab_list[cc])+"TMP_feves-{}ann_it{:.0f}_thr{:.1f}_npc{:.0f}_{}_stim.fits".format(label_stg,n_it,thr_it,npc,filt), 
                                           np.array([stim, inv_stim, norm_stim]))
                                # FIRST: define mask following spirals: max stim map (2-5)!
                                # good_mask = np.zeros_like(stim)
                                # good_mask[np.where(norm_stim>1)]=1
                                # ccorr_coeff = cube_distance(res_der,final_imgs[pp],mode='mask',mask=good_mask)
                                # norm_cc = ccorr_coeff/np.sum(ccorr_coeff)
                                # wmean_imgs[pp] = cube_collapse(res_der,mode='wmean',w=norm_cc)
                            #write_fits(outpath_3.format(data_folder)+"final_PCA-RDI_it{:.0f}_thr{:.1f}_{:.0f}-{:.0f}_{}_wmean.fits".format(label_stg,n_it,thr,test_npcs[0], test_npcs[-1],filt), wmean_imgs)
                            write_fits(outpath_5.format(bin_fac,filt,crop_lab_list[cc])+"final_feves-{}ann_it{:.0f}_thr{:.1f}_ann_{:.0f}-{:.0f}_{}.fits".format(label_stg,n_it,thr_it,test_pcs_ann[0],test_pcs_ann[-1],filt), final_imgs)
                            # correction by AGPM transmission
                            # for pp, npc in enumerate(test_npcs):
                            #     wmean_imgs[pp]/=transmission_2d
                            #     final_imgs[pp]/=transmission_2d
                            # #write_fits(outpath_5.format(bin_fac,filt,crop_lab_list[cc])+"final_PCA-RDI_it{:.0f}_thr{:.1f}_{:.0f}-{:.0f}_ann{:.0f}_wmean_AGPMcorr.fits".format(n_it,thr_it,test_npcs[0], test_npcs[-1],ann_sz), wmean_imgs)
                            # write_fits(outpath_5.format(bin_fac,filt,crop_lab_list[cc])+"final_PCA-RDI_it{:.0f}_thr{:.1f}_ann_{:.0f}-{:.0f}_ann{:.0f}_AGPMcorr.fits".format(n_it,thr_it,test_npcs[0],test_npcs[-1],ann_sz), final_imgs)
                        
                        if do_pca_1zone:
                            final_imgs = np.zeros([len(test_pcs_full),ADI_cube.shape[1],ADI_cube.shape[2]])
                            #wmean_imgs = np.zeros_like(final_imgs)
                            if mask_PCA is None:
                                mask_rdi = None
                            else:
                                mask_tmp = np.ones_like(ADI_cube[0])
                                mask_rdi = mask_circle(mask_tmp, mask_PCA, fillwith=0, mode='in')
                            #res = pca_1zone_it(ADI_cube, derot_angles, cube_ref=ref_cube, 
                            res = pca_1rho_it(ADI_cube, derot_angles, cube_ref=ref_cube,                  
                                              fwhm=fwhm, buffer=buffer, strategy=strategy, 
                                              ncomp_range=test_pcs_1zone, n_it_max=n_it, 
                                              thr=thr_it, n_neigh=n_neigh, thru_corr=throughput_corr, 
                                              n_br=n_br, psfn=psfn, starphot=flux, 
                                              plsc=plsc, svd_mode=svd_mode, 
                                              scaling=scaling, delta_rot=delta_rot_it, 
                                              mask_center_px=mask_IWA_px, add_res=add_res, 
                                              collapse='median',  mask_rdi=mask_rdi, 
                                              full_output=True, verbose=verbose, 
                                              weights=None, debug=debug, 
                                              path=outpath_5.format(bin_fac,filt,crop_lab_list[cc]),
                                              overwrite=overwrite_it)
                            if isinstance(thr_it,(int,float)):
                                thr_it1 = thr_it
                                thr_it2 = thr_it
                            else:
                                thr_it1 = thr_it[0]
                                thr_it2 = thr_it[-1]
                            write_fits(outpath_5.format(bin_fac,filt,crop_lab_list[cc])+"final_1rhoPCA-{}_it{:.0f}_thr{:.1f}-{:.1f}_{:.0f}-{:.0f}_{}.fits".format(label_stg,n_it,thr_it1,thr_it2,test_pcs_1zone[0],test_pcs_1zone[-1],filt), res[0])
                            write_fits(outpath_5.format(bin_fac,filt,crop_lab_list[cc])+"TMP_1rhoPCA-{}_it{:.0f}_thr{:.1f}-{:.1f}_npc{:.0f}-{:.0f}_{}_it_cube.fits".format(label_stg,n_it,thr_it1,thr_it2,test_pcs_1zone[0],test_pcs_1zone[-1],filt), res[1])
                            write_fits(outpath_5.format(bin_fac,filt,crop_lab_list[cc])+"TMP_1rhoPCA-{}_it{:.0f}_thr{:.1f}-{:.1f}_npc{:.0f}-{:.0f}_{}_stim_cube.fits".format(label_stg,n_it,thr_it1,thr_it2,test_pcs_1zone[0],test_pcs_1zone[-1],filt), res[2])
                            write_fits(outpath_5.format(bin_fac,filt,crop_lab_list[cc])+"TMP_1rhoPCA-{}_it{:.0f}_thr{:.1f}-{:.1f}_npc{:.0f}-{:.0f}_{}_sig_cube.fits".format(label_stg,n_it,thr_it1,thr_it2,test_pcs_1zone[0],test_pcs_1zone[-1],filt), res[3])
                            write_fits(outpath_5.format(bin_fac,filt,crop_lab_list[cc])+"TMP_1rhoPCA-{}_it{:.0f}_thr{:.1f}-{:.1f}_npc{:.0f}-{:.0f}_{}_drot_opt_arr.fits".format(label_stg,n_it,thr_it1,thr_it2,test_pcs_1zone[0],test_pcs_1zone[-1],filt), res[4])
                            write_fits(outpath_5.format(bin_fac,filt,crop_lab_list[cc])+"TMP_1rhoPCA-{}_it{:.0f}_thr{:.1f}-{:.1f}_npc{:.0f}-{:.0f}_{}_thr_opt_arr.fits".format(label_stg,n_it,thr_it1,thr_it2,test_pcs_1zone[0],test_pcs_1zone[-1],filt), res[5])
                            write_fits(outpath_5.format(bin_fac,filt,crop_lab_list[cc])+"TMP_1rhoPCA-{}_it{:.0f}_thr{:.1f}-{:.1f}_npc{:.0f}-{:.0f}_{}_npc_opt_arr.fits".format(label_stg,n_it,thr_it1,thr_it2,test_pcs_1zone[0],test_pcs_1zone[-1],filt), res[6])
                            write_fits(outpath_5.format(bin_fac,filt,crop_lab_list[cc])+"TMP_1rhoPCA-{}_it{:.0f}_thr{:.1f}-{:.1f}_npc{:.0f}-{:.0f}_{}_nit_opt_arr.fits".format(label_stg,n_it,thr_it1,thr_it2,test_pcs_1zone[0],test_pcs_1zone[-1],filt), res[7])
                            write_fits(outpath_5.format(bin_fac,filt,crop_lab_list[cc])+"TMP_1rhoPCA-{}_it{:.0f}_thr{:.1f}-{:.1f}_npc{:.0f}-{:.0f}_{}_cc_rad_ws_ss_opt_arr.fits".format(label_stg,n_it,thr_it1,thr_it2,test_pcs_1zone[0],test_pcs_1zone[-1],filt), np.array([res[8], res[9], res[10]]))
                            # stim = compute_stim_map(res_der)
                            # inv_stim = compute_inverse_stim_map(res, derot_angles)
                            # norm_stim = stim/np.percentile(inv_stim,99.7)
                            # write_fits(outpath_5.format(bin_fac,filt,crop_lab_list[cc])+"TMP_PCA-{}_it{:.0f}_thr{:.1f}_npc{:.0f}_{}_stim.fits".format(strategy,n_it,thr_it,npc,filt), 
                            #            np.array([stim, inv_stim, norm_stim]))
                            # FIRST: define mask following spirals: max stim map (2-5)!
                            # good_mask = np.zeros_like(stim)
                            # good_mask[np.where(norm_stim>1)]=1
                            # ccorr_coeff = cube_distance(res_der,final_imgs[pp],mode='mask',mask=good_mask)
                            # norm_cc = ccorr_coeff/np.sum(ccorr_coeff)
                            # wmean_imgs[pp] = cube_collapse(res_der,mode='wmean',w=norm_cc)
                            #write_fits(outpath_3.format(data_folder)+"final_PCA-RDI_it{:.0f}_thr{:.1f}_{:.0f}-{:.0f}_{}_wmean.fits".format(label_stg,n_it,thr,test_npcs[0], test_npcs[-1],filt), wmean_imgs)
                            


  