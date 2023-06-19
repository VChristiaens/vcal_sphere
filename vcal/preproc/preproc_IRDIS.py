#! /usr/bin/env python
# coding: utf-8

"""
Module with the preprocessing routine for SPHERE/IRDIS data.
"""

__author__ = 'V. Christiaens, J. Baird'
__all__ = ['preproc_IRDIS']

import ast
from astropy.stats import sigma_clipped_stats
from astropy.io import fits
import csv
import json
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from multiprocessing import cpu_count
import pdb
from os.path import isfile, isdir#, join, dirname, abspath
from vip_hci.fits import open_fits, write_fits
from vip_hci.metrics import peak_coordinates
try:
    from vip_hci.psfsub import median_sub
    from vip_hci.psfsub import MedsubParams
    from vip_hci.fm import normalize_psf
    from vip_hci.metrics import stim_map as compute_stim_map
    from vip_hci.metrics import inverse_stim_map as compute_inverse_stim_map
except:
    from vip_hci.medsub import median_sub
    from vip_hci.metrics import normalize_psf, compute_stim_map, compute_inverse_stim_map
from vip_hci.preproc import (cube_fix_badpix_clump, cube_recenter_2dfit, cube_recenter_dft_upsampling, 
                             frame_shift, frame_rotate, cube_detect_badfr_pxstats, 
                             cube_detect_badfr_ellipticity, cube_detect_badfr_correlation, 
                             frame_crop, cube_recenter_satspots, cube_recenter_radon, 
                             cube_recenter_via_speckles, cube_crop_frames, check_pa_vector,
                             cube_shift, find_scal_vector, cube_derotate)
from vip_hci.preproc.rescaling import _cube_resc_wave
from vip_hci.var import (frame_center, fit_2dmoffat, get_annulus_segments,
                         mask_circle, frame_filter_lowpass)
from ..utils import (cube_recenter_bkg, fit2d_bkg_pos, interpolate_bkg_pos, 
                     find_rot_cen, circ_interp, find_nearest)

from vcal import __path__ as vcal_path
matplotlib.use('Agg')

#**************************** PARAMS TO BE ADAPTED ****************************  

def preproc_IRDIS(params_preproc_name='VCAL_params_preproc_IRDIS.json', 
                  params_calib_name='VCAL_params_calib.json'):
    """
    Preprocessing of SPHERE/IRDIS data using preproc parameters provided in 
    json file.
    
    Parameters:
    ***********
    params_preproc_name: str, opt
        Full path + name of the json file containing preproc parameters.
    params_calib_name: str, opt
        Full path + name of the json file containing calibration parameters.
        
    Returns:
    ********
    None. All preprocessed products are written as fits files, and can then be 
    used for post-processing.
    
    """
    plt.style.use('default')
    with open(params_preproc_name, 'r') as read_file_params_preproc:
        params_preproc = json.load(read_file_params_preproc)
    with open(params_calib_name, 'r') as read_file_params_calib:
        params_calib = json.load(read_file_params_calib)
        
    with open(vcal_path[0] + "/instr_param/sphere_filt_spec.json", 'r') as filt_spec_file:
        filt_spec = json.load(filt_spec_file)[params_calib['comb_iflt']]  # Get infos of current filters combinaison
    with open(vcal_path[0] + "/instr_param/sphere.json", 'r') as instr_param_file:
        instr_cst = json.load(instr_param_file)
    
    path = params_calib['path'] #"/Volumes/Val_stuff/VLT_SPHERE/J1900_3645/" # parent path
    path_irdis = path+"IRDIS_reduction/"
    inpath = path_irdis+"1_calib_esorex/fits/"
    nd_filename = vcal_path[0] + "/../Static/SPHERE_CPI_ND.dat" # FILE WITH TRANSMISSION OF NEUTRAL DENSITY FILTER

    # OBS
    coro = params_preproc['coro']  # whether the observations were coronagraphic or not
    coro_sz = params_preproc.get('coro_sz',8)  # if coronagraphic, provide radius of the coronagraph in pixels (check header for size in mas and convert)
    #dit_irdis = 64.      # read later from header
    #dit_psf_irdis = 4.   # read later from header
    
    # Parts of pipeline to be run/overwritten
    to_do = params_preproc['to_do'] # parts of pre-processing to be run. 
    #1. crop odd + bad pix corr
    #   a. with provided mask (static) => iterative because clumps
    #   b. sigma filtering (cosmic rays) => non-iterative
    #2. Recentering
    #3. Make first master cube + derot angles calculation
    #4. anamorphism correction + final ADI cube writing
    #5. fine recentering based on bkg star (if relevant)
    #6. bad frame rejection
    #   a. Plots before rejection
    #   b. Rejection
    #   c. Plots after rejection
    #7. final PSF cube + FWHM + unsat flux (PSF)
    #   a. Gaussian
    #   b. Airy
    #8. Subtract satspots if 'CENTER' used as object (e.g. if OBJ is a ref cube)
    #9. final ADI cube (bin+crop)
    #10. Calculate scale factors (if DBI)

    overwrite = params_preproc.get('overwrite',[1]*10)   # list of bools corresponding to parts of pre-processing to be run again even if files already exist. Same order as to_do
    debug = params_preproc['debug']           # whether to print more info - useful for debugging
    nproc = params_preproc.get('nproc', int(cpu_count() / 2))  # number of processors to use, set to cpu_count()/2 for efficiency
    save_space = params_preproc['save_space'] # whether to progressively delete intermediate products as new products are calculated (can save space but will make you start all over from the beginning in case of bug)
    plot = params_preproc['plot']             # whether to plot additional info (evo of: Strehl, cross-corr level, airmass, flux)
    verbose = params_preproc['verbose']
    plot_obs_cond = params_preproc.get('plot_obs_cond',False)
    
    # Preprocessing options
    rec_met = params_preproc['rec_met']    # recentering method. choice among {"gauss_2dfit", "moffat_2dfit", "dft_nn", "satspots", "radon", "speckle"} # either a single string or a list of string to be tested. If not provided will try both gauss_2dfit and dft. Note: "nn" stand for upsampling factor, it should be an integer (recommended: 100)
    rec_met_psf = params_preproc['rec_met_psf']

    # if recentering by satspots provide here a tuple of 4 tuples:  top-left, top-right, bottom-left and bottom-right spots
    xy_spots = params_preproc.get('xy_spots',[])    
    if "xy_spots" in filt_spec.keys() : xy_spots = filt_spec["xy_spots"]    
    sigfactor = params_preproc.get('sigfactor',3)
    
    badfr_crit_names = params_preproc['badfr_crit_names']
    badfr_crit_names_psf = params_preproc['badfr_crit_names_psf']
    badfr_crit = params_preproc['badfr_crit']
    badfr_crit_psf = params_preproc['badfr_crit_psf']
    bad_fr_idx = params_preproc.get('bad_fr_idx',[[],[],[]])   # list of bad indices of science cube images (to force discarding those frames)
    
    if len(badfr_crit_names) != len(badfr_crit):
        raise TypeError("Length of bad fr. criteria is different")
    if len(badfr_crit_names_psf) != len(badfr_crit_psf):
        raise TypeError("Length of psf bad fr. criteria is different")
         
    #******************** PARAMS LIKELY GOOD AS DEFAULT ***************************  
    instr = params_calib['instr'] # instrument name in file name
    # First run  dfits *.fits |fitsort DET.SEQ1.DIT INS1.FILT.NAME INS1.OPTI2.NAME DPR.TYPE INS4.COMB.ROT
    # then adapt below
    filters = filt_spec['filters'] #DBI filters
    filters_lab = ['_left','_right'] # should be hard-coded because not an option in calib
    if len(filters) == 1:
        filters = [filters[0]+'_l', filters[0]+'_r'] # should have length of 2
    lbdas = np.array(filt_spec['lbda']) # get lamba(s) assosiated with filter(s)
    if len(lbdas) == 1:
        lbdas = np.array([lbdas[0], lbdas[0]]) # should have length of 2
    #n_z = lbdas.shape[0]
    diam = instr_cst['diam']
    plsc = np.array(instr_cst['plsc']) #0.01227 #arcsec/pixel

    # Systematic errors (cfr. Maire et al. 2016)
    pup_off = instr_cst.get('pup_off',135.99) #=-0.11de d dg  ## !!! MANUAL IS WRONG ACCORDING TO ALICE: should be +135.99 (in particular if TN = -1.75)
    TN = instr_cst.get('TN',-1.75) # pm0.08 deg
    ifs_off = instr_cst.get('ifs_off',0)            # for ifs data: -100.48 pm 0.13 deg # for IRDIS: 0
    scal_x_distort = instr_cst.get('scal_x_distort',1.0)      # for IFS: 1.0059
    scal_y_distort = instr_cst.get('scal_y_distort',1.0062) # for IFS: 1.0011
    mask_scal = params_preproc.get('mask_scal',[0.15,0])
    if isinstance(mask_scal,str):
        pass
    elif not isinstance(mask_scal, (tuple,list)):
        raise TypeError("mask_scal can only be str, tuple or list")
    elif len(mask_scal)!=2:
        raise TypeError("if a tuple/list, mask_scal should have 2 elements")

    # preprocessing options
    idx_test_cube = params_preproc.get('idx_test_cube', [0,0,0])                                            # if rec_met is list: provide index of test cube (should be the most tricky one) where all methods will be tested => best method should correspond min(stddev of shifts)
    #idx_test_cube_psf = params_preproc['']                                    # id as above for psf
    cen_box_sz = params_preproc.get('cen_box_sz',[31,71,31])                   # size of the subimage for 2d fit, for OBJ, PSF and CEN
    true_ncen = params_preproc.get('true_ncen', None)                          # number of points in time to use for interpolation of center location in OBJ cubes based on location inferred in CEN cubes. Min value: 2 (set to 2 even if only 1 CEN cube available). Important: this is not necessarily equal to the number of CEN cubes (e.g. if there are 3 CEN cubes, 2 before the OBJ sequence and 1 after, true_ncen should be set to 2, not 3)
    #norm_per_s = params_preproc['']                                           # if True, divide all frames by respective dit => have all fluxes in adu/s
    template_strehl = params_preproc['template_strehl']
    distort_corr = params_preproc.get('distort_corr',1)                        # whether to correct manually for distortion (anamorphism) or not
    bp_crop_sz = params_preproc.get('bp_crop_sz',801)                          #823 # 361 => 2.25'' radius; but better to keep it as large as possible and only crop before post-processing. Here we just cut the useless edges (100px on each side)
    bp_crop_sz_psf = params_preproc.get('bp_crop_sz_psf',801)   
    final_crop_sz = params_preproc['final_crop_sz']                            #823 # 361 => 2.25'' radius; but better to keep it as large as possible and only crop before post-processing. Here we just cut the useless edges (100px on each side)
    final_crop_sz_psf = params_preproc['final_crop_sz_psf']                    # 51 => 0.25'' radius (~3.5 FWHM)
    psf_model = params_preproc.get('psf_model','moff')                         #'airy' #model to be used to measure FWHM and flux. Choice between {'gauss', 'moff', 'airy'}
    separate_trim = params_preproc.get('separate_trim', True)                  # whether to separately trim K1 and K2. If False, will only trim based on the K1 frames
    bin_fac = params_preproc.get('bin_fac',1)                                  # binning factors for final cube. If the cube is not too large, do not bin.
    approx_xy_bkg = params_preproc.get('approx_xy_bkg',0)                      # approx bkg star position in full ADI frame obtained after rough centering 
    sub_med4bkg = bool(params_preproc.get('sub_med4bkg',1))                    # Median subtraction before fiting gaussian to find bkg star position
    snr_thr_bkg = params_preproc.get('snr_thr_bkg',5)                          # SNR threshold for the bkg star: only frames where the SNR is above that threshold are used to find bkg star position 
    good_cen_idx = params_preproc.get('good_cen_idx', None)                    # good indices of center cubes (to be used for fine centering)
    bin_fit = params_preproc.get('bin_fit',1)
    convolve_bkg = params_preproc.get('convolve_bkg',1)
    
    if isinstance(separate_trim,str):
        trim_ch=filters.index(separate_trim)
    else:
        trim_ch=0
    
    # output names
    label_test = params_preproc.get('label_test', '')
    outpath = path_irdis+"2_preproc_vip{}/".format(label_test)
    use_cen_only = params_preproc.get('use_cen_only', 0)
    final_cubename = params_preproc.get('final_cubename', 'final_cube')
    final_anglename = params_preproc.get('final_anglename', 'final_derot_angles')
    final_psfname = params_preproc.get('final_psfname', 'final_psf_med')
    final_fluxname = params_preproc.get('final_fluxname','final_flux')
    final_fwhmname = params_preproc.get('final_fwhmname','final_fwhm')
    final_scalefac_name = params_preproc.get('final_scalefacname', 'final_scale_fac')
    # norm output names
    if final_cubename.endswith(".fits"):
        final_cubename = final_cubename[:-5]
    if final_anglename.endswith(".fits"):
        final_anglename = final_anglename[:-5]
    if final_psfname.endswith(".fits"):
        final_psfname = final_psfname[:-5]
    if final_fluxname.endswith(".fits"):
        final_fluxname = final_fluxname[:-5]
    if final_scalefac_name.endswith(".fits"):
        final_scalefac_name = final_scalefac_name[:-5]
    final_cubename_norm = final_cubename+"_norm"
    final_psfname_norm = final_psfname+"_norm"
    
    
    ###### DO NOT CHANGE BELOW THIS LINE UNLESS YOU KNOW WHAT YOU'RE DOING ######

    # STRUCTURE:
    #1. crop odd + bad pix corr
    #   a. with provided mask (static) => iterative because clumps
    #   b. sigma filtering (cosmic rays) => non-iterative
    #2. Recenter
    #3. final crop
    #4. combine all cropped cubes + compute derot_angles [bin if requested]
    #5. fine recentering based on bkg star (if relevant)
    #6. bad frame rejection
    #   a. Plots before rejection
    #   b. Rejection
    #   c. Plots after rejection
    #7. final PSF cube + FWHM + unsat flux (PSF)
    #   a. Gaussian
    #   b. Airy
    #8. temporal binning of ADI cube
    #9. anamorphism correction + final ADI cube writing
    
    
    # List of OBJ and PSF files
    dico_lists = {}
    reader = csv.reader(open(path+'dico_files.csv', 'r'))
    for row in reader:
         dico_lists[row[0]] = ast.literal_eval(row[1])
        
    # SEPARATE LISTS FOR OBJ AND PSF
    OBJ_IRDIS_list = dico_lists['sci_list_irdis']
    PSF_IRDIS_list = dico_lists['psf_list_irdis']
    CEN_IRDIS_list = dico_lists['cen_list_irdis']   
    OBJ_IRDIS_list = [name[:-5] for name in OBJ_IRDIS_list]
    PSF_IRDIS_list = [name[:-5] for name in PSF_IRDIS_list]
    CEN_IRDIS_list = [name[:-5] for name in CEN_IRDIS_list]
    
    nobj = len(OBJ_IRDIS_list)
    npsf = len(PSF_IRDIS_list)
    ncen = len(CEN_IRDIS_list)
    
    obj_psf_list = [OBJ_IRDIS_list]
    labels = ['']
    labels2 = ['obj']
    final_crop_szs = [final_crop_sz]
      
    if npsf>0:
        obj_psf_list.append(PSF_IRDIS_list)
        labels.append('_psf')
        labels2.append('psf')
        final_crop_szs.append(final_crop_sz_psf)
    if ncen>0:
        obj_psf_list.append(CEN_IRDIS_list)
        labels.append('_cen')
        labels2.append('cen')
        final_crop_szs.append(final_crop_sz)
    
    #if len(prefix) == 3:
    #    CEN_IRDIS_list = [x[:-5] for x in os.listdir(inpath) if x.startswith(prefix[2])]  # don't include ".fits"
    #    CEN_IRDIS_list.sort()
    #    ncen = len(CEN_IRDIS_list)
    #    obj_psf_list.append(CEN_IRDIS_list)
    #    labels.append('_cen')
    #    labels2.append('cen')
    #    final_crop_szs.append(final_crop_sz) # add the same crop for CEN as for OBJ
    
    if isinstance(plsc, float):
        plsc_med = plsc
    else:
        plsc_med = np.median(plsc)
    resel = lbdas*0.206265/(plsc*diam)
    print("Resel:")
    for i in range(len(resel)):
         print("{:.2f} px ({})".format(resel[i],filters[i]))
    max_resel = np.amax(resel)   
    
    if bool(to_do):
    
        if not isdir(outpath):
            os.system("mkdir "+outpath)
                
        # Extract info from example files
        if not use_cen_only:
            tmp, header = open_fits(inpath+OBJ_IRDIS_list[0]+'_left.fits', header=True)
        else:
            tmp, header = open_fits(inpath+CEN_IRDIS_list[0]+'_left.fits', header=True)
        ori_sz = tmp.shape[-1]
        ori_cy, ori_cx = frame_center(tmp[0])
        dit_irdis = float(header['EXPTIME'])
        #ndit_irdis = float(header['HIERARCH ESO DET NDIT'])
        #filt1 = header['HIERARCH ESO INS1 FILT NAME']
        #filt2 = header['HIERARCH ESO INS1 OPTI2 NAME']
        dits = [dit_irdis]
        #ndits = [ndit_irdis]
        
        if npsf > 0:
            _, header = open_fits(inpath+PSF_IRDIS_list[0]+'_left.fits', header=True)
            dit_psf_irdis = float(header['EXPTIME'])
            #ndit_psf_irdis = float(header['HIERARCH ESO DET NDIT'])
            dits.append(dit_psf_irdis)
            #ndits.append(ndit_psf_irdis)       
        if ncen > 0:
            _, header = open_fits(inpath+CEN_IRDIS_list[0]+'_left.fits', header=True)
            dit_cen_irdis = float(header['EXPTIME'])
            #ndit_cen_irdis = float(header['HIERARCH ESO DET NDIT'])
            dits.append(dit_cen_irdis)
            #ndits.append(ndit_cen_irdis)
    
        # COMBINED LISTS OF K1 and K2 (i.e. including OBJ AND PSF AND CEN, if any)
        all_files = os.listdir(inpath)
        file_list_K1 = [name[:-5] for name in all_files if (name.startswith(instr) and name.endswith("left.fits"))]
        file_list_K2 = [name[:-5] for name in all_files if (name.startswith(instr) and name.endswith("right.fits"))]    
        file_list_K1.sort()
        file_list_K2.sort()
        ncubes = len(file_list_K1)
        print("File list ({} cubes total): ".format(ncubes), file_list_K1)    
        file_lists = [file_list_K1,file_list_K2]
            
        
        # TRANSMISSION in case of a neutral density filter is used
        nd_filter_SCI = header['HIERARCH ESO INS4 FILT2 NAME'].strip()
        
        nd_file = pd.read_csv(nd_filename, sep = "   ", comment='#', engine="python",
                              header=None, names=['wavelength', 'ND_0.0', 'ND_1.0','ND_2.0', 'ND_3.5'])
        nd_wavelen = nd_file['wavelength']
        try:
            nd_transmission_SCI = nd_file[nd_filter_SCI]
        except:
            nd_transmission_SCI = [1]*len(nd_wavelen)
    
        if PSF_IRDIS_list :
            _, header = open_fits(inpath+PSF_IRDIS_list[0]+'_left.fits', header=True)
            #dit_psf_ifs = float(header['HIERARCH ESO DET SEQ1 DIT'])
            #ndit_psf_ifs = float(header['HIERARCH ESO DET NDIT'])
            nd_filter_PSF = header['HIERARCH ESO INS4 FILT2 NAME'].strip()
            try:
                nd_transmission_PSF = nd_file[nd_filter_PSF]
            except:
                nd_transmission_PSF = [1]*len(nd_wavelen)
                  
            nd_trans = [nd_transmission_SCI, nd_transmission_PSF, nd_transmission_SCI]
        else : nd_trans = [nd_transmission_SCI, None, nd_transmission_SCI]
        
        # Format xy_spots if provided
        if len(xy_spots) == len(lbdas):
            pass
        elif len(xy_spots) == 4: # i.e. just one set of 4 sat spot coordinates
            # assume it was provided for first wavelength
            cx_tmp = np.mean([xy_spots[i][0] for i in range(4)])
            cy_tmp = np.mean([xy_spots[i][1] for i in range(4)])
            dx = np.mean(np.array([xy_spots[0][0]-xy_spots[-1][0], xy_spots[2][0]-xy_spots[1][0]]))/2
            dy = np.mean(np.array([xy_spots[0][1]-xy_spots[-1][1], xy_spots[2][1]-xy_spots[1][1]]))/2
            r = np.sqrt(np.power(dx,2)+np.power(dy,2))
            thetas = []
            for i in range(4):
                thetas.append(np.arctan2(xy_spots[i][1]-cy_tmp,xy_spots[i][0]-cx_tmp))
                
            xy_spots_fin = [xy_spots]
            for l in range(1,len(lbdas)):
                new_xy_pos = []
                for i in range(4):
                    new_x = cx_tmp+r*(lbdas[l]/lbdas[0])*np.cos(thetas[i])
                    new_y = cy_tmp+r*(lbdas[l]/lbdas[0])*np.sin(thetas[i])             
                    new_xy_pos.append((new_x,new_y))
                xy_spots_fin.append(tuple(new_xy_pos))
            xy_spots = tuple(xy_spots_fin)
        
        #********************************* BPIX CORR ******************************       
        if 1 in to_do:
            ## OBJECT + PSF
            for fi, file_list in enumerate(file_lists):
                if fi==0:
                    str_idx=5
                else:
                    str_idx=6
                for ff, filename in enumerate(file_list):
                    if ff == 0:
                        full_output = True
                    else:
                        full_output = False
                    if not isfile(outpath+"{}_1bpcorr.fits".format(filename)) or overwrite[0]:
                        bp_crop_sz_tmp = bp_crop_sz
                        cube, header = open_fits(inpath+filename, header=True)
                        if filename[:-str_idx] in obj_psf_list[0] and use_cen_only:
                            continue
                        if npsf>0 :
                            if filename[:-str_idx] in obj_psf_list[1]:
                                bp_crop_sz_tmp = bp_crop_sz_psf 
                        if cube.shape[1]%2==0 and cube.shape[2]%2==0:
                            cube = cube[:,1:,1:]
                            header["NAXIS1"] = cube.shape[1]
                            header["NAXIS2"] = cube.shape[2]
                        if bp_crop_sz_tmp>0 and bp_crop_sz_tmp<cube.shape[1]:
                            cube = cube_crop_frames(cube,bp_crop_sz_tmp)
                        
                        cube = cube_fix_badpix_clump(cube, bpm_mask=None, cy=None, cx=None, fwhm=1.2*resel[fi], 
                                                     sig=6., protect_mask=False, verbose=full_output,
                                                     half_res_y=False, max_nit=10, full_output=full_output,
                                                     nproc=nproc)
                        if full_output:
                            write_fits(outpath+filename+"_1bpcorr_bpmap.fits", cube[1], header=header) 
                            cube = cube[0]
                        write_fits(outpath+filename+"_1bpcorr.fits", cube, header=header)
            if save_space:
                os.system("rm {}*total.fits".format(inpath))
                    
    
        #******************************* RECENTERING ******************************
        # just set OBJ list as CEN list if only CEN matter
        if use_cen_only:
            obj_psf_list[0] = obj_psf_list[-1]
            OBJ_IRDIS_list = CEN_IRDIS_list
        if 2 in to_do:
            for fi, file_list in enumerate(obj_psf_list): ## OBJECT, then possibly PSF (but not CEN)
                if fi == 0:
                    negative=coro
                    rec_met_tmp = rec_met
                elif fi == 1:
                    negative=False
                    rec_met_tmp = rec_met_psf
                else:
                    break  # CEN # Note: they are centered at the same time as OBJ (when dealing with the first OBJ cube more specifically)
                if not file_list:
                    break  # if file_list is empty, which append when there is no psf/cen then we break.
                
                for ff, filt in enumerate(filters_lab):
                    if not isfile(outpath+"{}_2cen.fits".format(file_list[-1])) or overwrite[1]:
                        if isinstance(rec_met, list):
                            # PROCEED ONLY ON TEST CUBE
                            cube, header = open_fits(outpath+file_list[idx_test_cube[fi]]+filt+"_1bpcorr.fits", header=True)
                            n_fr=cube.shape[0]
                            std_shift = []
                            for ii in range(len(rec_met_tmp)):
                                if "2dfit" in rec_met_tmp[ii]:
                                    tmp = frame_filter_lowpass(np.median(cube,axis=0))
                                    y_max, x_max = np.unravel_index(np.argmax(tmp),tmp.shape)
                                    cy, cx = frame_center(tmp)
                                    cube, y_shifts, x_shifts = cube_recenter_2dfit(cube, xy=(int(x_max), int(y_max)), fwhm=1.2*resel[ff], subi_size=cen_box_sz[fi], model=rec_met_tmp[:-6],
                                                                               nproc=nproc, interpolation='lanczos4',
                                                                               offset=None, negative=negative, threshold=False,
                                                                               save_shifts=False, full_output=True, verbose=verbose,
                                                                               debug=False, plot=plot)
 
                                    std_shift.append(np.sqrt(np.std(y_shifts)**2+np.std(x_shifts)**2))
                                    if debug:
                                        write_fits(outpath+"TMP_test_cube_cen{}{}_{}.fits".format(labels[fi],filt,rec_met_tmp[ii]), cube)                                         
                                elif "dft" in rec_met_tmp[ii]:
                                    #1 rough centering with peak
                                    _, peak_y, peak_x = peak_coordinates(cube, fwhm=1.2*resel[ff], 
                                                                         approx_peak=None, 
                                                                         search_box=None,
                                                                         channels_peak=False)
                                    _, peak_yx_ch = peak_coordinates(cube, fwhm=1.2*resel[ff], 
                                                                     approx_peak=(peak_y, peak_x), 
                                                                     search_box=31,
                                                                     channels_peak=True)
                                    cy, cx = frame_center(cube[0])
                                    for zz in range(cube.shape[0]):
                                        cube[zz] = frame_shift(cube[zz], cy-peak_yx_ch[zz,0], cx-peak_yx_ch[zz,1])
                                    #2. alignment with upsampling
                                    cube, y_shifts, x_shifts = cube_recenter_dft_upsampling(cube, center_fr1=None, negative=negative,
                                                                                            fwhm=1.2*resel[ff], subi_size=cen_box_sz[fi], upsample_factor=int(rec_met_tmp[ii][4:]),
                                                                                            interpolation='lanczos4',
                                                                                            full_output=True, verbose=verbose, nproc=nproc,
                                                                                            save_shifts=False, debug=False, plot=plot)
                                    std_shift.append(np.sqrt(np.std(y_shifts)**2+np.std(x_shifts)**2))                                
                                    #3 final centering based on 2d fit
                                    cube_tmp = np.zeros([1,cube.shape[1],cube.shape[2]])
                                    cube_tmp[0] = np.median(cube,axis=0)
                                    _, y_shifts_tmp, x_shifts_tmp = cube_recenter_2dfit(cube_tmp, xy=None, fwhm=1.2*resel[ff], subi_size=cen_box_sz[fi], model='moff',
                                                                                nproc=nproc, interpolation='lanczos4',
                                                                                offset=None, negative=negative, threshold=False,
                                                                                save_shifts=False, full_output=True, verbose=True,
                                                                                debug=False, plot=plot)
                                    for zz in range(cube.shape[0]):
                                        cube[zz] = frame_shift(cube[zz], y_shifts_tmp[0], x_shifts_tmp[0])
                                    y_shifts = y_shifts+y_shifts_tmp[0]
                                    x_shifts = x_shifts+x_shifts_tmp[0]                                              
                                    if debug:
                                        write_fits(outpath+"TMP_test_cube_cen{}{}_{}.fits".format(labels[fi],filt,rec_met_tmp[ii]), cube)  
                                elif "satspots" in rec_met_tmp[ii]:
                                    if ncen == 0:
                                        raise ValueError("No CENTER file found. Cannot recenter based on satellite spots.")
                                    # INFER SHIFTS FROM CEN CUBES
                                    cen_cube_names = obj_psf_list[-1]
                                    mjd_cen = np.zeros(ncen)
                                    for cc in range(ncen):
                                        _, head_cc = open_fits(inpath+cen_cube_names[cc]+filters_lab[ff], header = True)
                                        cube_cen = open_fits(outpath+cen_cube_names[cc]+filters_lab[ff]+"_1bpcorr.fits")
                                        if cc==0:
                                            #n_frc=cube_cen.shape[0]
                                            y_shifts_cen_tmp = [] #np.zeros([ncen,n_frc])
                                            x_shifts_cen_tmp = [] #np.zeros([ncen,n_frc])
                                            y_shifts_cen_med = [] #np.zeros([ncen,n_frc])
                                            x_shifts_cen_med = [] #np.zeros([ncen,n_frc])
                                            y_shifts_cen_std = [] #np.zeros([ncen,n_frc])
                                            x_shifts_cen_std = [] #np.zeros([ncen,n_frc])
                                        mjd_cen[cc] = float(head_cc['MJD-OBS'])
                                        # SUBTRACT TEST OBJ CUBE (to easily find sat spots)
                                        if not use_cen_only:
                                            cube_cen -= np.median(cube,axis=0)
                                        diff = int((ori_sz-bp_crop_sz)/2)
                                        xy_spots_tmp = tuple([(xy_spots[ff][i][0]-diff,xy_spots[ff][i][1]-diff) for i in range(len(xy_spots[ff]))])
                                        _, y_tmp, x_tmp, _, _ = cube_recenter_satspots(cube_cen, xy_spots_tmp, 
                                                                                       subi_size=cen_box_sz[2], 
                                                                                       sigfactor=sigfactor, plot=plot,
                                                                                       fit_type='moff', lbda=None, 
                                                                                       debug=False, verbose=True, 
                                                                                       full_output=True)
                                        y_shifts_cen_tmp.append(y_tmp)
                                        x_shifts_cen_tmp.append(x_tmp)
                                        y_shifts_cen_med.append(np.median(y_tmp))
                                        x_shifts_cen_med.append(np.median(x_tmp))
                                        y_shifts_cen_std.append(np.std(y_tmp))
                                        x_shifts_cen_std.append(np.std(x_tmp))
                                    if not use_cen_only:
                                        # median combine results for all MJD CEN bef and all after SCI obs
                                        mjd = float(header['MJD-OBS']) # mjd of first obs 
                                        mjd_fin = mjd
                                        if true_ncen is None:
                                            unique_mjd_cen = mjd_cen.copy()
                                            y_shifts_cen = y_shifts_cen_med
                                            x_shifts_cen = x_shifts_cen_med
                                            y_shifts_cen_err = y_shifts_cen_std
                                            x_shifts_cen_err = x_shifts_cen_std
                                            true_ncen = ncen
                                        elif true_ncen > 4:
                                            unique_mjd_cen = mjd_cen.copy()
                                            y_shifts_cen = y_shifts_cen_med
                                            x_shifts_cen = x_shifts_cen_med
                                            y_shifts_cen_err = y_shifts_cen_std
                                            x_shifts_cen_err = x_shifts_cen_std
                                        else:
                                            if true_ncen > ncen:
                                                raise ValueError("Code not compatible with true_ncen > ncen")
                                            if true_ncen>2:
                                                _, header_fin = open_fits(inpath+OBJ_IRDIS_list[-1]+'_left.fits', header=True)
                                                mjd_fin = float(header_fin['MJD-OBS'])
                                            elif true_ncen>3:
                                                _, header_mid = open_fits(inpath+OBJ_IRDIS_list[int(nobj/2)]+'_left.fits', header=True)
                                                mjd_mid = float(header_mid['MJD-OBS'])
                                                                    
                                            unique_mjd_cen = np.zeros(true_ncen)  
                                            y_shifts_cen = np.zeros([true_ncen])
                                            x_shifts_cen = np.zeros([true_ncen])
                                            y_shifts_cen_err = np.zeros([true_ncen])
                                            x_shifts_cen_err = np.zeros([true_ncen])
                                            for cc in range(true_ncen):
                                                if cc == 0:
                                                    cond = mjd_cen < mjd
                                                elif cc == true_ncen-1:
                                                    cond = mjd_cen > mjd_fin
                                                elif cc == 1 and true_ncen == 3:
                                                    cond = (mjd_cen > mjd & mjd_cen < mjd_fin)
                                                elif cc == 1 and true_ncen == 4:
                                                    cond = (mjd_cen > mjd & mjd_cen < mjd_mid)
                                                else:
                                                    cond = (mjd_cen < mjd_fin & mjd_cen > mjd_mid)
                                                    
                                                unique_mjd_cen[cc] = np.median(mjd_cen[np.where(cond)])
                                                y_shifts_cen[cc] = np.median([y_shifts_cen_med[i] for i in range(len(y_shifts_cen_med)) if cond[i]])
                                                y_shifts_cen_err[cc] = np.std([y_shifts_cen_std[i]for i in range(len(y_shifts_cen_std)) if cond[i]])
                                                x_shifts_cen[cc] = np.median([x_shifts_cen_med[i] for i in range(len(x_shifts_cen_med)) if cond[i]])
                                                x_shifts_cen_err[cc] = np.std([x_shifts_cen_std[i]for i in range(len(x_shifts_cen_std)) if cond[i]])
        
                                                
                                        # APPLY THEM TO OBJ CUBES             
                                        ## interpolate based on cen shifts                                                     
                                        y_shifts = float(np.interp([mjd],unique_mjd_cen,y_shifts_cen))  
                                        x_shifts = float(np.interp([mjd],unique_mjd_cen,x_shifts_cen))
                                        cube = cube_shift(cube, y_shifts, x_shifts, nproc=nproc)
                                        std_shift.append(np.sqrt((y_shifts_cen_err)**2+(x_shifts_cen_err)**2))                                                     
#                                        if debug:
#                                            plt.show()
#                                            plt.plot(range(n_frc),y_shifts,'ro', label = 'shifts y')
#                                            plt.plot(range(n_frc),x_shifts,'bo', label = 'shifts x')
#                                            plt.show()
#                                            write_fits(outpath+"TMP_test_cube_cen{}_{}.fits".format(labels[fi],rec_met_tmp[ii]), cube)  
                                    
                                elif "radon" in rec_met_tmp[ii]:
                                    cube, y_shifts, x_shifts = cube_recenter_radon(cube, full_output=True, verbose=True, 
                                                                                   interpolation='lanczos4')
                                    std_shift.append(np.sqrt(np.std(y_shifts)**2+np.std(x_shifts)**2))                                                    
                                    if debug:
                                        write_fits(outpath+"TMP_test_cube_cen{}_{}.fits".format(labels[fi],rec_met_tmp[ii]), cube)  
                                elif "speckle" in rec_met_tmp[ii]:
                                    cube, x_shifts, y_shifts = cube_recenter_via_speckles(cube, cube_ref=None, alignment_iter=5,
                                                                                          gammaval=1, min_spat_freq=0.5, max_spat_freq=3,
                                                                                          fwhm=1.2*max_resel, debug=False, negative=negative,
                                                                                          recenter_median=False, subframesize=20,
                                                                                          interpolation='bilinear',
                                                                                          save_shifts=False, plot=plot,
                                                                                          nproc=nproc)
                                    std_shift.append(np.sqrt(np.std(y_shifts)**2+np.std(x_shifts)**2))                                                           
                                    if debug:
                                        write_fits(outpath+"TMP_test_cube_cen{}_{}.fits".format(labels[fi],rec_met_tmp[ii]), cube)  
                                else:
                                    raise ValueError("Centering method not recognized")           
                                    
                            #infer best method from min(stddev of shifts)
                            std_shift = np.array(std_shift)
                            idx_min_shift = np.nanargmin(std_shift)
                            rec_met_tmp = rec_met_tmp[idx_min_shift]
                            if fi == 1:
                                rec_met_psf = rec_met_tmp[idx_min_shift] 
                            else:
                                rec_met = rec_met_tmp[idx_min_shift] 
                                
                            print("Best centering method for {}{}: {}".format(labels[fi],filt,rec_met_tmp))
                            print("Press c if satisfied. q otherwise")                   
                          #  pdb.set_trace()
                
                    if isinstance(rec_met_tmp, str):
                        final_y_shifts = []
                        final_x_shifts = []
                        final_y_shifts_std = []
                        final_x_shifts_std = []
                        mjd_all = []
                        mjd_mean = []
                        pa_sci_ini = []
                        pa_sci_fin = []
                        for fn_tmp, filename_tmp in enumerate(file_list):
                            cube_tmp, head_tmp = open_fits(inpath + OBJ_IRDIS_list[fn_tmp] + filters_lab[ff],
                                                           header=True)
                            mjd_tmp = float(head_tmp['MJD-OBS'])
                            mjd_tmp_list = [mjd_tmp + i * dit_irdis/86400 for i in range(cube_tmp.shape[0])]  # DIT in seconds to MJD
                            mjd_all.extend(mjd_tmp_list)
                            mjd_mean.append(np.mean(mjd_tmp_list))
                            pa_sci_ini.append(float(head_tmp["HIERARCH ESO TEL PARANG START"]))
                            pa_sci_fin.append(float(head_tmp["HIERARCH ESO TEL PARANG END"]))
                        mjd_all = np.array(mjd_all)
                        for fn, filename in enumerate(file_list):
                            if ((fn>0 and fi==0) or fn>npsf-1) and use_cen_only:
                                continue
                            cube, header = open_fits(outpath+filename+filt+"_1bpcorr.fits", header=True)
                            n_fr=cube.shape[0]
                            if "2dfit" in rec_met_tmp:
                                tmp = frame_filter_lowpass(np.median(cube,axis=0))
                                y_max, x_max = np.unravel_index(np.argmax(tmp),tmp.shape)
                                cube, y_shifts, x_shifts = cube_recenter_2dfit(cube, xy=(int(x_max), int(y_max)),
                                                                               fwhm=1.2*resel[ff], subi_size=cen_box_sz[fi], model=rec_met_tmp[:-6],
                                                                               nproc=nproc, interpolation='lanczos4',
                                                                               offset=None, negative=negative, threshold=False,
                                                                               save_shifts=False, full_output=True, verbose=verbose,
                                                                               debug=False, plot=False)
                                                     
                            elif "dft" in rec_met_tmp:
                                #1 rough centering with peak
                                _, peak_y, peak_x = peak_coordinates(cube, fwhm=1.2*resel[ff], 
                                                                     approx_peak=None, 
                                                                     search_box=None,
                                                                     channels_peak=False)
                                _, peak_yx_ch = peak_coordinates(cube, fwhm=1.2*resel[ff], 
                                                                 approx_peak=(peak_y, peak_x), 
                                                                 search_box=31,
                                                                 channels_peak=True)
                                cy, cx = frame_center(cube[0])
                                for zz in range(cube.shape[0]):
                                    cube[zz] = frame_shift(cube[zz], cy-peak_yx_ch[zz,0], cx-peak_yx_ch[zz,1])
                                #2. alignment with upsampling
                                cube, y_shifts, x_shifts = cube_recenter_dft_upsampling(cube, center_fr1=None, negative=False,
                                                                                        fwhm=4, subi_size=cen_box_sz[fi],
                                                                                        upsample_factor=int(rec_met_tmp[4:]),
                                                                                        interpolation='lanczos4',
                                                                                        full_output=True, verbose=verbose, nproc=nproc,
                                                                                        save_shifts=False, debug=False, plot=plot)                              
                                #3 final centering based on 2d fit
                                cube_tmp = np.zeros([1,cube.shape[1],cube.shape[2]])
                                cube_tmp[0] = np.median(cube,axis=0)
                                _, y_shifts_tmp, x_shifts_tmp = cube_recenter_2dfit(cube_tmp, xy=None, fwhm=1.2*resel[ff], subi_size=cen_box_sz[fi], model='moff',
                                                                            nproc=nproc, interpolation='lanczos4',
                                                                            offset=None, negative=False, threshold=False,
                                                                            save_shifts=False, full_output=True, verbose=verbose,
                                                                            debug=False, plot=plot)
                                for zz in range(cube.shape[0]):
                                    cube[zz] = frame_shift(cube[zz], y_shifts_tmp[0], x_shifts_tmp[0])
                                y_shifts = y_shifts+y_shifts_tmp[0]
                                x_shifts = x_shifts+x_shifts_tmp[0]
                                if debug:
                                    print('dft{} + 2dfit centering: xshift: {} px, yshift: {} px for cube {}_1bpcorr.fits'
                                          .format(int(rec_met_tmp[4:]), x_shifts[0], y_shifts[0], filename), flush=True)
                            elif "satspots" in rec_met_tmp:
                                if fn==0:
                                    if ncen == 0:
                                        raise ValueError("No CENTER file found. Cannot recenter based on satellite spots.")
                                    # INFER SHIFTS FROM CEN CUBES
                                    cen_cube_names = obj_psf_list[-1]
                                    mjd_cen = np.zeros(ncen)
                                    pa_cen = []
                                    for cc in range(ncen):
                                        ### first get the MJD time of each cube     
                                        _, head_cc = open_fits(inpath+cen_cube_names[cc]+filters_lab[ff], header = True)
                                        pa_cen.append(float(head_cc["HIERARCH ESO TEL PARANG START"]))
                                        cube_cen = open_fits(outpath+cen_cube_names[cc]+filters_lab[ff]+"_1bpcorr.fits")
                                        nfr_tmp = cube_cen.shape[0]
                                        if cc==0:
                                            #n_frc=cube_cen.shape[0]
                                            y_shifts_cen_tmp = [] #np.zeros([ncen,n_frc])
                                            x_shifts_cen_tmp = [] #np.zeros([ncen,n_frc])
                                            y_shifts_cen_med = np.zeros([ncen])
                                            x_shifts_cen_med = np.zeros([ncen])
                                            y_shifts_cen_std = np.zeros([ncen])
                                            x_shifts_cen_std = np.zeros([ncen])
                                        mjd_cen[cc] = float(head_cc['MJD-OBS'])+(nfr_tmp*dits[-1]/2.)/(3600*24) # MJD-OBS corresponds to start of exposure
                                        # SUBTRACT NEAREST OBJ CUBE (to easily find sat spots)
                                        cube_cen_sub = cube_cen.copy()
                                        if not use_cen_only:
                                            m_idx = find_nearest(mjd_mean,mjd_cen[cc])
                                            cube_near = open_fits(outpath+file_list[m_idx]+filt+"_1bpcorr.fits")
                                            cube_cen_sub -= np.median(cube_near,axis=0)
                                        diff = int((ori_sz-bp_crop_sz)/2)
                                        xy_spots_tmp = tuple([(xy_spots[ff][i][0]-diff,xy_spots[ff][i][1]-diff) for i in range(len(xy_spots[ff]))])
                                        cube_cen_sub, y_tmp, x_tmp, _, _ = cube_recenter_satspots(cube_cen_sub, xy_spots_tmp, subi_size=cen_box_sz[2], 
                                                                                                  sigfactor=sigfactor, plot=plot,
                                                                                                  fit_type='moff', lbda=None, 
                                                                                                  debug=debug, verbose=verbose, 
                                                                                                  full_output=True)
                                        y_shifts_cen_tmp.append(y_tmp)
                                        x_shifts_cen_tmp.append(x_tmp)
                                        y_shifts_cen_med[cc] = np.median(y_tmp)
                                        x_shifts_cen_med[cc] = np.median(x_tmp)
                                        y_shifts_cen_std[cc] = np.std(y_tmp)
                                        x_shifts_cen_std[cc] = np.std(x_tmp)
                                        write_fits(outpath+cen_cube_names[cc]+filters_lab[ff]+"_2cen_sub.fits", cube_cen_sub, header=head_cc)
                                        cube_cen = cube_shift(cube_cen, y_tmp, x_tmp, nproc=nproc)
                                        write_fits(outpath+cen_cube_names[cc]+filters_lab[ff]+"_2cen.fits", cube_cen, header=head_cc)
                                    
                                       # pdb.set_trace()
                                    #if not use_cen_only:
                                        # median combine results for all MJD CEN bef and all after SCI obs
                                    #cube, header_ini = open_fits(inpath+OBJ_IRDIS_list[fn]+'_left.fits', header=True)
                                    nfr_tmp = cube.shape[0]
                                    mjd = float(header['MJD-OBS'])+(nfr_tmp*dits[fi]/2.)/(3600*24) # mjd of first obs 
                                    mjd_fin = mjd
                                    if true_ncen is None:
                                        unique_mjd_cen = mjd_cen.copy()
                                        y_shifts_cen = y_shifts_cen_med
                                        x_shifts_cen = x_shifts_cen_med
                                        y_shifts_cen_err = y_shifts_cen_std
                                        x_shifts_cen_err = x_shifts_cen_std
                                        true_ncen = ncen
                                    elif true_ncen > 4:
                                        unique_mjd_cen = mjd_cen.copy()
                                        y_shifts_cen = y_shifts_cen_med
                                        x_shifts_cen = x_shifts_cen_med
                                        y_shifts_cen_err = y_shifts_cen_std
                                        x_shifts_cen_err = x_shifts_cen_std
                                    else:
                                        if true_ncen > ncen:
                                            raise ValueError("Code not compatible with true_ncen > ncen")
                                        if true_ncen > 2:
                                            _, header_fin = open_fits(inpath+OBJ_IRDIS_list[-1]+'_left.fits', header=True)
                                            mjd_fin = float(header_fin['MJD-OBS'])
                                        if true_ncen > 3:
                                            _, header_mid = open_fits(inpath+OBJ_IRDIS_list[int(nobj/2)]+'_left.fits', header=True)
                                            mjd_mid = float(header_mid['MJD-OBS'])
                                                                
                                        unique_mjd_cen = np.zeros(true_ncen)  
                                        unique_pa_cen = np.zeros(true_ncen)  
                                        y_shifts_cen = np.zeros(true_ncen)
                                        x_shifts_cen = np.zeros(true_ncen)
                                        y_shifts_cen_err = np.zeros(true_ncen)
                                        x_shifts_cen_err = np.zeros(true_ncen)
                                        for cc in range(true_ncen):
                                            if cc == 0:
                                                cond = mjd_cen < mjd
                                            elif cc == true_ncen-1:
                                                cond = mjd_cen > mjd_fin  # if a science cube is taken after the last center file, this will give False for cond
                                            elif cc == 1 and true_ncen == 3:
                                                cond = ((mjd_cen > mjd) & (mjd_cen < mjd_fin))
                                            elif cc == 1 and true_ncen == 4:
                                                cond = ((mjd_cen > mjd) & (mjd_cen < mjd_mid))
                                            else:
                                                cond = ((mjd_cen < mjd_fin) & (mjd_cen > mjd_mid))
                                            unique_mjd_cen[cc] = np.median(mjd_cen[np.where(cond)])
                                            unique_pa_cen[cc]= np.median(np.array(pa_cen)[np.where(cond)])
                                            y_shifts_cen[cc] = np.median(y_shifts_cen_med[np.where(cond)])
                                            x_shifts_cen[cc] = np.median(x_shifts_cen_med[np.where(cond)])
                                            y_shifts_cen_err[cc] = np.std(y_shifts_cen_std[np.where(cond)])
                                            x_shifts_cen_err[cc] = np.std(x_shifts_cen_std[np.where(cond)])  # SAVE UNCERTAINTY ON CENTERING
                                        unc_cen = np.sqrt(np.power(np.amax(y_shifts_cen_std),2)+np.power(np.amax(x_shifts_cen_std),2))
                                        write_fits(outpath+"Uncertainty_on_centering_sat_spots_px.fits", np.array([unc_cen]))
                                    if np.amax(x_shifts_cen_err)>3 or np.amax(y_shifts_cen_err)>3:
                                        msg = "Warning: large std found for calculated shifts (std_x: {:.1f}, std_y: {:.1f}) px." 
                                        msg+= "Make sure CEN cubes and sat spots fits look good."
                                        print(msg)
                                        pdb.set_trace()
                                        
                                if not use_cen_only:
                                    # APPLY THEM TO OBJ CUBES
                                    
                                    ## OLD: linear interpolation based on cen shifts  
                                    #y_shifts = np.zeros(n_fr)
                                    #x_shifts = np.zeros(n_fr)
                                    #mjd_ori = float(header['MJD-OBS'])           
                                    
                                    #for zz in range(n_fr):     
                                        #y_shifts[zz] = np.interp([mjd_ori+(dits[fi]*zz/n_fr)/(3600*24)],unique_mjd_cen,y_shifts_cen)    
                                        #x_shifts[zz] = np.interp([mjd_ori+(dits[fi]*zz/n_fr)/(3600*24)],unique_mjd_cen,x_shifts_cen)                                                       
                                        
                                    ## NEW: "circular" interpolation based on cen shifts
                                    cy, cx = frame_center(cube)
                                    cen_xy = (cx, cy)
                                    rot_x, rot_y, r, th0 = find_rot_cen(cen_xy, 
                                                                        y_shifts_cen, 
                                                                        x_shifts_cen, 
                                                                        unique_pa_cen,
                                                                        verbose=verbose)
                                    rot_xy = (rot_x, rot_y)
                                    pos_xy = circ_interp(n_fr, rot_xy, r, th0,
                                                         unique_pa_cen, 
                                                         pa_sci_ini[fn],
                                                         pa_sci_fin[fn])
                                    if verbose:
                                        print(pos_xy)
                                    x_shifts = cx - pos_xy[0]
                                    y_shifts = cy - pos_xy[1]
                                    
                                    for zz in range(n_fr):  
                                        cube[zz] = frame_shift(cube[zz], y_shifts[zz], x_shifts[zz])                                    
                                    if plot and fn == 0:
                                        plt.show() # show whichever previous plot is in memory
                                        colors = ['k','r','b','y','c','m','g']
                                        # y
                                        plt.plot(range(n_fr),y_shifts,colors[0]+'-', label = 'shifts y (first cube)')
                                        print("True number of CENTER cubes:", true_ncen)
                                        plt.errorbar(range(true_ncen), y_shifts_cen, 
                                                     yerr=y_shifts_cen_err, fmt=colors[cc+1]+'o',
                                                     label='y cen shifts')
                                        plt.legend()
                                        plt.show()
                                        # x
                                        plt.plot(range(n_fr),x_shifts,colors[0]+'-', label = 'shifts x (first cube)')
                                        plt.errorbar(range(true_ncen),x_shifts_cen, 
                                                     yerr=x_shifts_cen_err, fmt=colors[cc+1]+'o',label='x cen shifts')
                                        plt.legend()
                                        plt.show()
                                        write_fits(outpath+"TMP_test_cube_cen{}_{}.fits".format(labels[fi],rec_met_tmp), cube)                                                     

                            elif "radon" in rec_met_tmp:
                                cube, y_shifts, x_shifts = cube_recenter_radon(cube, full_output=True, verbose=True, 
                                                                               interpolation='lanczos4')                                         
                            elif "speckle" in rec_met_tmp:
                                cube, x_shifts, y_shifts = cube_recenter_via_speckles(cube, cube_ref=None, alignment_iter=5,
                                                                                      gammaval=1, min_spat_freq=0.5, 
                                                                                      max_spat_freq=3,
                                                                                      fwhm=1.2*max_resel, debug=False, 
                                                                                      negative=negative,
                                                                                      recenter_median=False, subframesize=20,
                                                                                      interpolation='bilinear',
                                                                                      save_shifts=False, plot=False,
                                                                                      nproc=nproc)
                            else:
                                raise ValueError("Centering method not recognized")
                            if fi>0 or not use_cen_only:                                                                     
                                write_fits(outpath+filename+filt+"_2cen.fits", cube, header=header)
                                final_y_shifts.extend(y_shifts.tolist())
                                final_x_shifts.extend(x_shifts.tolist())
                                final_y_shifts_std.extend([np.std(y_shifts)]*len(y_shifts))
                                final_x_shifts_std.extend([np.std(x_shifts)]*len(x_shifts))         
                            # write_fits(outpath+"TMP_final_shifts{}_{}.fits".format(labels[fi],rec_met_tmp[ii]), np.array([final_y_shifts,final_x_shifts]))
                        if "satspots" in rec_met_tmp:
                            write_fits(outpath+"TMP_shifts_cen_y{}_{}_{}.fits".format(labels[fi],filters[ff],rec_met_tmp), y_shifts_cen)   
                            write_fits(outpath+"TMP_shifts_cen_x{}_{}_{}.fits".format(labels[fi],filters[ff],rec_met_tmp), x_shifts_cen)                                
                        if fi>0 or not use_cen_only: 
                            write_fits(outpath+"TMP_shifts_y{}_{}_{}.fits".format(labels[fi],filters[ff],rec_met_tmp), np.array(final_y_shifts))
                            write_fits(outpath+"TMP_shifts_x{}_{}_{}.fits".format(labels[fi],filters[ff],rec_met_tmp), np.array(final_x_shifts))
                        if fi != 1 and plot and not use_cen_only:
                            f, (ax1) = plt.subplots(1,1, figsize=(15,10))
                            t0 = np.amin(mjd_all)  # first file
                            ax1.errorbar(#np.arange(1,len(file_list)+1,1./cube.shape[0]),
                                         (mjd_all-t0)*60*24,  # convert to minutes since first file
                                         final_y_shifts, final_y_shifts_std,
                                         fmt='bo', label='y')
                            ax1.errorbar(#np.arange(1,len(file_list)+1,1./cube.shape[0]),
                                         (mjd_all-t0)*60*24,
                                         final_x_shifts, final_x_shifts_std,
                                         fmt='ro',label='x')
                            if "satspots" in rec_met_tmp:
                                t0 = np.amin(unique_mjd_cen)
                                ax1.errorbar((unique_mjd_cen-t0)*60*24,y_shifts_cen,y_shifts_cen_err,
                                             fmt='co',label='y cen')
                                ax1.errorbar((unique_mjd_cen-t0)*60*24,x_shifts_cen,x_shifts_cen_err,
                                             fmt='mo',label='x cen')
                            ax1.set_xlabel("Time from start of obs. (min)")
                            ax1.set_ylabel("Shift (px)")
                            ax1.minorticks_on()
                            plt.legend(loc='best')
                            plt.savefig(outpath+"Shifts_xy{}_{}.pdf".format(labels[fi],rec_met_tmp),bbox_inches='tight', format='pdf')
                            plt.clf()
                
    
        #******************************* MASTER CUBES ******************************
        if 3 in to_do:
            for ff, filt in enumerate(filters_lab):
                for fi,file_list in enumerate(obj_psf_list):
                    if fi == 0 and use_cen_only:
                        continue
                    elif fi == 2 and not "satspots" in rec_met:
                        msg = "Are you sure you do not want to use the satellite spots for centering?"
                        # msg += "(If so press 'c' to continue, else 'q' to abort then re-run step 2 after changing the value of 'rec_met' in parameter file)"
                        print(msg)
                        # pdb.set_trace()
                        print("Will proceed with {}".format(rec_met))
                        break
                    elif not file_list : break # If file_list is empty, which append when there is no psf/cen then we break.
                   
                    if not isfile(outpath+"1_master_cube{}_{}.fits".format(labels[fi],filters[ff])) or not isfile(outpath+"1_master_derot_angles.fits") or overwrite[2]:
                        if fi!=1 and ff==0: # only SCI and CEN
                            parang_st = []
                            parang_nd = []
                        interp_trans = np.interp(lbdas[ff]*1000, np.array(nd_wavelen), nd_trans[fi]) # file lbdas are in nm
                        for nn, filename in enumerate(file_list):
                            cube, header  = open_fits(outpath+filename+filt+"_2cen", header=True)
                            if nn == 0:
                                master_cube = [] #np.zeros([int(len(file_list)*ndits[fi]),cube.shape[1],cube.shape[2]])
                            try:
                                for jj in range(cube.shape[0]):
                                    master_cube.append(cube[jj])
                                #master_cube[int(nn*ndits[fi]):int((nn+1)*ndits[fi])] = cube
                            except:
                               pdb.set_trace()
                            if fi!=1 and ff==0:
                                parang_st.append(float(header["HIERARCH ESO TEL PARANG START"]))
                                parang_nd_tmp=float(header["HIERARCH ESO TEL PARANG END"])
                                if nn> 0:
                                    if abs(parang_st[-1]-parang_nd_tmp)>180:
                                        sign_tmp=np.sign(parang_st[-1]-parang_nd_tmp)
                                        parang_nd_tmp=parang_nd_tmp+sign_tmp*360
                                parang_nd.append(parang_nd_tmp)
                        master_cube = np.array(master_cube)
                        master_cube = master_cube/interp_trans  
                        if debug:
                            print("transmission correction: ",interp_trans)
                        
                    
                        # IMPORTANT WE DO NOT NORMALIZE BY DIT (any more!)
                        write_fits(outpath+"1_master{}_cube_{}.fits".format(labels[fi],filters[ff]), master_cube) #/dits[fi])
                       
                        if fi!=1 and ff==0:
                            final_derot_angles = [] #np.zeros(int(len(file_list)*ndits[fi]))
                            final_par_angles = [] #np.zeros(int(len(file_list)*ndits[fi]))
                            counter = 0
                            for nn, filename in enumerate(file_list):
                                cube, header  = open_fits(outpath+filename+filt+"_2cen", header=True)
                                nfr_tmp = cube.shape[0]
                                x = parang_st[nn]
                                y = parang_nd[nn]
                                parang = x +(y-x)*(0.5+np.arange(nfr_tmp))/nfr_tmp
#                                if nn> 0:
#                                    if abs(parang[0]-final_par_angles[-1])>180:
#                                        sign_tmp=np.sign(parang[0]-final_par_angles[-1]/360)
#                                        parang=parang+sign_tmp*360
                                final_derot_angles.extend(list(parang + TN + pup_off + ifs_off)) #+ posang[nn] 
                                final_par_angles.extend(list(parang))
                                counter+=nfr_tmp
                            write_fits(outpath+"1_master_derot_angles{}.fits".format(labels[fi]), np.array(final_derot_angles))
                            write_fits(outpath+"1_master_par_angles{}.fits".format(labels[fi]), np.array(final_par_angles))
        
                        if fi!=1:
                            # median-ADI
                            master_cube = open_fits(outpath+"1_master{}_cube_{}.fits".format(labels[fi],filters[ff]))
                            final_derot_angles = open_fits(outpath+"1_master_derot_angles{}.fits".format(labels[fi]))
                            params = MedsubParams(cube=master_cube, angle_list=final_derot_angles, radius_int=10,
                                                  nproc=nproc)
                            ADI_frame = median_sub(algo_params=params)
                            write_fits(outpath+"median_ADI1_{}{}.fits".format(labels[fi],filters[ff]), ADI_frame)
                            #master_cube_full = None
                            #cube_full = None
    
    
        #********************** DISTORTION (ANAMORPHISM) ***********************
        if distort_corr:
            dist_lab = "_DistCorr"
        else:
            dist_lab = ""
        if 4 in to_do:
            for fi,file_list in enumerate(obj_psf_list):
                if fi == 0 and use_cen_only:
                    continue
                if fi == 1:
                    dist_lab_tmp = "" # no need for PSF
                elif fi == 2 and not "satspots" in rec_met: # no 2cen files
                    break
                else:
                    dist_lab_tmp = dist_lab
                if not file_list : break # If file_list is empty, which append when there is no psf/cen then we break.

                for ff, filt in enumerate(filters):
                    if not isfile(outpath+"2_master{}_cube_{}{}.fits".format(labels[fi],filters[ff],dist_lab_tmp)) or overwrite[3]:
                        cube, header = open_fits(outpath+"1_master{}_cube_{}.fits".format(labels[fi],filters[ff]), 
                                                 header=True)
                        if distort_corr:
                            cube = _cube_resc_wave(cube, scaling_list=None, ref_xy=None, 
                                                   imlib="opencv", #Note: FFT unusable because scaling_y!=scaling_x
                                                   interpolation='lanczos4', 
                                                   scaling_y=scal_y_distort, 
                                                   scaling_x=scal_x_distort)
                        write_fits(outpath+"2_master{}_cube_{}{}.fits".format(labels[fi],filters[ff],dist_lab_tmp), 
                                   cube, header=header)
                    if fi!=1:
                        # median-ADI
                        master_cube = open_fits(outpath+"2_master{}_cube_{}{}.fits".format(labels[fi],filters[ff],dist_lab_tmp))
                        final_derot_angles = open_fits(outpath+"1_master_derot_angles{}.fits".format(labels[fi]))
                        params = MedsubParams(cube=master_cube, angle_list=final_derot_angles, radius_int=10,
                                              nproc=nproc)
                        ADI_frame = median_sub(algo_params=params)
                        write_fits(outpath+"median_ADI2_{}{}{}.fits".format(labels[fi],filters[ff],dist_lab_tmp), ADI_frame)
                        #cube_full = None
                            
#                    for fi, file_list in enumerate(obj_psf_list):
#                        if fi == 2 or (fi==0 and coro):
#                            break
#                        med_psf = open_fits(outpath+"3_final_{}_med_{}.fits".format(labels2[fi],filt))
#                        if distort_corr:
#                            norm_psf = np.zeros([final_crop_sz_psf,final_crop_sz_psf])
#                            fwhm=np.zeros(n_z)
#                            med_flux = np.zeros(n_z)
#                            med_psf_tmp = np.array([med_psf])
#                            med_psf = _cube_resc_wave(med_psf_tmp, scaling_list=None, ref_xy=None, 
#                                              interpolation='lanczos4', 
#                                              scaling_y=scal_y_distort, 
#                                              scaling_x=scal_x_distort)[0]
##                            med_psf = frame_px_resampling(med_psf, scale=(scal_x_distort, scal_y_distort), 
##                                                      interpolation='lanczos4', verbose=True)
##                            if med_psf.shape[0] > ori_sz or med_psf.shape[1] > ori_sz:                              
##                                med_psf = frame_crop(med_psf,ori_sz,verbose=debug)
#                            norm_psf, med_flux, fwhm = normalize_psf(med_psf, fwhm='fit', size=final_crop_sz_psf, threshold=None, mask_core=None,
#                                                                     model=psf_model, interpolation='lanczos4',
#                                                                     force_odd=True, full_output=True, verbose=debug, debug=False)
#                            fwhm=np.array([fwhm])                                         
#                        else:
#                            if fi == 0:
#                                psf_model_tmp=''
#                            else:
#                                psf_model_tmp='_'+psf_model
#                            norm_psf = open_fits(outpath+"3_final_{}_norm_med_{}{}.fits".format(labels2[fi],filt,psf_model_tmp))
#                            med_flux = open_fits(outpath+"3_final_{}_flux_med_{}{}.fits".format(labels2[fi],filt,psf_model_tmp))
#                            fwhm = open_fits(outpath+"3_final_{}_fwhm_{}{}.fits".format(labels2[fi],filt,psf_model_tmp))
#                                                     
#                        write_fits(outpath+final_medname.format(labels2[fi],dist_lab,filt), med_psf)
#                        write_fits(outpath+"4_final_{}_norm_med{}_{}.fits".format(labels2[fi],dist_lab,filt), norm_psf)
#                        write_fits(outpath+"4_final_{}_flux_med{}_{}.fits".format(labels2[fi],dist_lab,filt), med_flux)
#                        write_fits(outpath+"4_final_{}_fwhm{}_{}.fits".format(labels2[fi],dist_lab,filt), fwhm)
                            

        # ******* FINE RECENTERING BASED ON BKG STAR (IF RELEVANT) ************
        if approx_xy_bkg == 0 or use_cen_only:
            label_cen = ''
        else:
            label_cen = 'bkg_cen'
        if 5 in to_do and not use_cen_only:
            for ff, filt in enumerate(filters_lab):
                if not isfile(outpath+"2{}_master{}_cube_{}{}.fits".format(label_cen,labels[0],filters[ff],dist_lab)) or overwrite[4]:
                    if approx_xy_bkg != 0:
                        master_cube = open_fits(outpath+"2_master{}_cube_{}{}.fits".format(labels[0],filters[ff],dist_lab))
                        derot_angles = open_fits(outpath+"1_master_derot_angles.fits")
                        derot_angles = check_pa_vector(derot_angles)
                        write_fits(outpath+"1_master_derot_angles.fits",derot_angles)
                        # RECENTER BASED ON BKG STAR
                        ## first give good frame
                        if len(obj_psf_list)==3: # ie when CEN cubes are available, use that
                            cen_cube = open_fits(outpath+"2_master{}_cube_{}{}.fits".format(labels[-1],filters[ff],dist_lab))
                            cen_derot = open_fits(outpath+"1_master_derot_angles{}.fits".format(labels[-1]))                  
                        else:
                            cen_cube = open_fits(outpath+"2_master{}_cube_{}{}.fits".format(labels[0],filters[ff],dist_lab))
                            cen_derot = open_fits(outpath+"1_master_derot_angles{}.fits".format(labels[0]))
                            
                        if good_cen_idx is None:
                            good_frame1 = frame_rotate(cen_cube[-1]-cen_cube[0],-cen_derot[-1])
                            good_frame2 = frame_rotate(cen_cube[0]-cen_cube[-1],-cen_derot[0])
                            good_frame = np.median([good_frame1,good_frame2],axis=0)
                        elif isinstance(good_cen_idx,int):
                            derot_angles_cen = open_fits(outpath+"1_master_derot_angles{}.fits".format(labels[-1]))
                            idx_max = np.argmax(np.abs(derot_angles_cen[good_cen_idx]-derot_angles_cen[:]))
                            good_frame = frame_rotate(cen_cube[good_cen_idx]-cen_cube[idx_max], 
                                                      -derot_angles_cen[good_cen_idx])
                        elif isinstance(good_cen_idx,list):
                            good_frame = []
                            derot_angles_cen = open_fits(outpath+"1_master_derot_angles{}.fits".format(labels[-1]))
                            for gg, good_idx in enumerate(good_cen_idx):
                                idx_max = np.argmax(np.abs(derot_angles_cen[good_idx]-derot_angles_cen[:]))
                                good_frame.append(frame_rotate(cen_cube[good_idx]-cen_cube[idx_max], 
                                                               -derot_angles_cen[good_idx]))
                            good_frame = np.median(good_frame,axis=0)
                        else:
                            raise TypeError("good_cen_idx can only be int, list or None")
                        write_fits(outpath+"TMP_good_frame_for_fine_centering.fits", good_frame)
                        if ff ==0:
                            debug = False
                        else:
                            debug=True
                        master_cube, shifts, sunc = cube_recenter_bkg(master_cube, 
                                                        derot_angles, 
                                                        fwhm=1.2*resel[ff], 
                                                        approx_xy_bkg=approx_xy_bkg, #inital_pos, final_pos, 
                                                        fit_type=psf_model,
                                                        snr_thr=snr_thr_bkg,
                                                        verbose=verbose,
                                                        crop_sz=21,
                                                        sub_med=sub_med4bkg,
                                                        good_frame=good_frame,
                                                        sigfactor=sigfactor,
                                                        bin_fit=bin_fit, 
                                                        convolve=convolve_bkg, 
                                                        path_debug=outpath,
                                                        full_output=True,
                                                        debug=debug,
                                                        nproc=nproc)
                        n_fr = master_cube.shape[0]
                        final_shifts = np.zeros([4,n_fr])
                        final_shifts[1] = shifts[0]
                        final_shifts[2] = shifts[1]
                        for i in range(n_fr):
                            final_shifts[0,i] = np.sqrt(np.power(shifts[0,i],2)+np.power(shifts[1,i],2))
                        final_shifts[3] = sunc
                        if plot or debug:
                            #idx_shifts = badfr_crit_names.index("shifts")
                            # default params
                            thr = 0.4 # arbitrarily set to 0.4 px here (SPHERE specs), i.e. significant flux dilution on neighbouring pixels.
                            err = np.median(sunc)/np.sqrt(2) # from anamorphism uncertainty (0.016px/1''); dominates over plsc unc (~0.001 px). Maire et al.
                            if verbose:
#                                msg = "Conservatively setting uncertainty on star position based on BKG to 0.1px"
#                                msg+= ". This is because the anamorphism uncertainty is ~0.016px/1'' and IRDIS BKG stars must be < 6'' radius."
#                                msg+=" and because PSF centroid uncertainty < 0.05px."
                                cen_y, cen_x = frame_center(master_cube[0])
                                approx_r_bkg = np.sqrt((approx_xy_bkg[0]-cen_x)**2+(approx_xy_bkg[1]-cen_y)**2)
                                msg = "Shift uncertainties calculated from 1) distortion uncertainty "
                                msg+= "(here ~{:.2f} px given separation of BKG); ".format(approx_r_bkg*2e-4)
                                msg+= "2) uncertainty on centroid of BKG star"
                                print(msg)
                            fig, (ax1,ax2,ax3) = plt.subplots(1,3,figsize=(13,3))
                            for i in range(n_fr):
                                if abs(shifts[0,i]) < thr:
                                    col = 'b'
                                else:
                                    col = 'r'
                                ax1.errorbar(i+1, shifts[0,i], err, fmt=col+'o')
                                if abs(shifts[1,i]) < thr:
                                    col = 'b'
                                else:
                                    col = 'r'
                                ax2.errorbar(i+1, shifts[1,i], err, fmt=col+'o')
                                if final_shifts[0,i] < thr:
                                    col = 'b'
                                else:
                                    col = 'r'
                                ax3.errorbar(i+1, final_shifts[0,i], sunc[i], fmt=col+'o')
                            ax1.plot([0,n_fr+1],[thr,thr],'k--')
                            ax1.plot([0,n_fr+1],[-thr,-thr],'k--')
                            ax1.set_xlabel("Index of frame in cube")
                            ax1.set_ylabel("Residual shift along x (px)")
                            ax2.plot([0,n_fr+1],[thr,thr],'k--')
                            ax2.plot([0,n_fr+1],[-thr,-thr],'k--')
                            ax2.set_xlabel("Index of frame in cube")
                            ax2.set_ylabel("Residual shift along y (px)")
                            ax3.plot([0,n_fr+1],[thr,thr],'k--')
                            ax3.set_xlabel("Index of frame in cube")
                            ax3.set_ylabel("Residual shift amplitude (px)")
                            plt.savefig(outpath+"Residual_shifts_bkg_VS_satspots_{}.pdf".format(filters[ff]), bbox_inches='tight', format='pdf')
                        write_fits(outpath+"TMP_shifts_fine_recentering_bkg_{}.fits".format(filters[ff]), final_shifts)
                        # REDO median-ADI
                        ADI_frame = median_sub(master_cube,derot_angles,radius_int=10, nproc=nproc)
                        write_fits(outpath+"median_ADI3_{}{}_{}.fits".format(filters[ff],dist_lab,label_cen), ADI_frame)
       
                        # CROP
    #                    crop_sz_tmp = final_crop_szs[0]
    #                    master_cube = cube_crop_frames(master_cube,crop_sz_tmp)
                        
                        write_fits(outpath+"2{}_master{}_cube_{}{}.fits".format(label_cen,labels[0],filters[ff],dist_lab), master_cube)
                        #write_fits(outpath+"2{}_master{}_cube_full_{}.fits".format(label_cen,labels[0],filters[ff]), master_cube_full)
                        #master_cube_full = None                  
                
        #********************* PLOTS + TRIM BAD FRAMES OUT ********************
        if 6 in to_do:
            if plot_obs_cond: # PLOT BEFORE  
                # observing conditions
                alphas = [0.3,1] # before and after trimming
                markers_1 = ['o','.','o']
                markers_2 = ['D','d','D']
                markers_3 = ['X','x','X']
                cols =  ['blue','red','blue']
                fig, ((ax1,ax2),(ax3,ax4),(ax5,ax6)) = plt.subplots(3,2,figsize=(40,20))
                ax1b = ax1.twinx()
                ax2b = ax2.twinx()
                ax3b = ax3.twinx()
                ax4b = ax4.twinx()
                
            for fi,file_list in enumerate(obj_psf_list):
                label_cen_tmp = ''
                if fi == 1:
                    dist_lab_tmp = "" # no need for PSF
                else:
                    label_cen_tmp = label_cen
                    dist_lab_tmp = dist_lab
                if fi == 0 and use_cen_only:
                    continue
                elif fi == 2 and not use_cen_only: # no need for CEN, except if no OBJ
                    break
                if not file_list : break # If file_list is empty, which append when there is no psf/cen then we break.
                
                for ff, filt in enumerate(filters):            
                    cube = open_fits(outpath+"2{}_master{}_cube_{}{}.fits".format(label_cen_tmp,labels[fi],filt,dist_lab_tmp))
                    ntot = cube.shape[0]
    
                    # OBSERVATION PARAMS
                    if plot_obs_cond:
                        strehl = np.zeros(ntot)
                        seeing = np.zeros(ntot)
                        tau0 = np.zeros(ntot)
                        UTC = np.zeros(ntot)
                        counter = 0
                        for nn, filename in enumerate(file_list):
                            cube, header  = open_fits(outpath+filename+filt+"_2cen", header=True)
                            nfr_tmp =cube.shape[0]
                            ### Strehl (different header)
                            _, header = open_fits(inpath+template_strehl[fi].format(filt,nn), header=True)
                            strehl[counter:counter+nfr_tmp] = 100*float(header['HIERARCH ESO QC STREHL {}'.format(filters_lab[ff][1:].upper())])
                            _, header = open_fits(outpath+filename+filters_lab[ff]+"_2cen.fits", header=True)
                            seeing[counter:counter+nfr_tmp] = float(header['HIERARCH ESO TEL IA FWHM'])       
                            tau0[counter:counter+nfr_tmp] = 1000*float(header['HIERARCH ESO TEL AMBI TAU0'])
                            UTC[counter:counter+nfr_tmp] = np.linspace(float(header['UTC'])+0.5*dits[fi],float(header['UTC'])+((nfr_tmp+0.5)*dits[fi]),nfr_tmp,endpoint=False)
                            counter+=nfr_tmp
                        if fi != 1 and ff == 0:
                            UTC_0 = UTC[0]
                    if not coro or fi == 1:
                        # first fit on median
                        crop_sz = int(6*1.2*resel[ff])
                        if not crop_sz%2:
                            crop_sz+=1
                        _, _, fwhm = normalize_psf(np.median(cube,axis=0), fwhm='fit', size=crop_sz, threshold=None, mask_core=None,
                                                   model=psf_model, interpolation='lanczos4',
                                                   force_odd=True, full_output=True, verbose=debug, debug=False)
                        if not isfile(outpath+"TMP_fluxes{}_{}.fits".format(labels[fi],filt)) or overwrite[5]:
                            fluxes = np.zeros(ntot)
                            for nn in range(ntot):                            
                                _, fluxes[nn], _ = normalize_psf(cube[nn], fwhm=fwhm, size=crop_sz, threshold=None, mask_core=None,
                                                       model=psf_model, interpolation='lanczos4',
                                                       force_odd=True, full_output=True, verbose=debug, debug=False)
                            write_fits(outpath+"TMP_fluxes{}_{}.fits".format(labels[fi],filt), fluxes)                           
                        else:
                            fluxes = open_fits(outpath+"TMP_fluxes{}_{}.fits".format(labels[fi],filt))
                    else:
                        fwhm = 1.2*resel[ff]
                    if plot_obs_cond:               
                        color = 'tab:{}'.format(cols[ff])
                        if fi != 1:
                            label='{} - Sr'.format(filt)
                            if ff ==0:
                                ax1.set_xlabel('Time from start (s)')
                                ax1.set_ylabel('Strehl ratio (%)')
                        else:
                            label=None
                        ax1.plot(UTC-UTC_0,strehl,cols[ff][0]+markers_1[fi], alpha=alphas[0], label=label)
                        if not coro:
                            color = 'tab:{}'.format(cols[ff])
                            if fi != 1:
                                label = '{} - flux'.format(filt)
                                if ff == 0:
                                
                                    ax1b = ax1.twinx()
                                    ax1b.set_ylabel('Stellar flux (ADUs/s)')
                            else:
                                label=None
                            ax1b.plot(UTC-UTC_0,fluxes,cols[ff][0]+markers_2[fi], alpha=alphas[1], label=label)
                        if ff == 0:    
                            color = 'tab:blue'
                            if fi != 1 :
                                ax3.set_xlabel('Time from start (s)')
                                ax3.set_ylabel('Seeing (arcsec)', color=color)
                            ax3.plot(UTC-UTC_0,seeing,'b'+markers_2[fi], alpha=alphas[0])
                            color = 'tab:red'
                            if fi != 1 :
                                ax3b = ax3.twinx()
                                ax3b.set_ylabel('Coherence time (ms)', color=color)   
                            ax3b.plot(UTC-UTC_0,tau0,'r'+markers_2[fi], alpha=alphas[1])
    
    
    #        if plot:
    #            fig, (ax1) = plt.subplots(1,1,figsize=(15,10))
    #
    #        for ff, filt in enumerate(filters):               
    #            for fi,file_list in enumerate(obj_psf_list):
                    perc = 0 #perc_min[fi]
                    if fi != 1:
                        badfr_critn_tmp = badfr_crit_names
                        badfr_crit_tmp = badfr_crit
                    else:
                        badfr_critn_tmp = badfr_crit_names_psf
                        badfr_crit_tmp = badfr_crit_psf
                    bad_str = "-".join(badfr_critn_tmp)
                    if not isfile(outpath+"3_master{}_cube_clean_{}{}{}.fits".format(labels[fi],filt,dist_lab_tmp,bad_str)) or overwrite[5]:
                        # OBJECT                 
                        if fi != 1:
                            derot_angles = open_fits(outpath+"1_master_derot_angles{}.fits".format(labels[fi]))
                            
                        # Rejection based on pixel statistics
                        if ff == trim_ch or separate_trim==1:
                            
                            final_good_index_list = list(range(cube.shape[0]))
                            if len(bad_fr_idx[fi])>0:
                                final_good_index_list = [final_good_index_list[i] for i in range(cube.shape[0]) if final_good_index_list[i] not in bad_fr_idx[fi]]
                            counter = 0
                            
                            if "stat" in badfr_critn_tmp:
                                idx_stat = badfr_critn_tmp.index("stat")
                                # Default parameters
                                mode = "circle"
                                rad = int(2*fwhm)
                                width = 0
                                if coro and fi != 1:
                                    mode = "annulus"
                                    rad = int(coro_sz+1)
                                    width = int(fwhm*2)
                                top_sigma = 1.0
                                low_sigma=1.0
                                # Update if provided
                                if "mode" in badfr_crit_tmp[idx_stat].keys():
                                    mode = badfr_crit_tmp[idx_stat]["mode"]
                                if "rad" in badfr_crit_tmp[idx_stat].keys():
                                    rad = int(badfr_crit_tmp[idx_stat]["rad"]*fwhm)
                                if "width" in badfr_crit_tmp[idx_stat].keys():
                                    width = int(badfr_crit_tmp[idx_stat]["width"]*fwhm)                             
                                if "thr_top" in badfr_crit_tmp[idx_stat].keys():
                                    top_sigma = badfr_crit_tmp[idx_stat]["thr_top"]
                                if "thr_low" in badfr_crit_tmp[idx_stat].keys():
                                    low_sigma = badfr_crit_tmp[idx_stat]["thr_low"]
                                good_index_list, bad_index_list = cube_detect_badfr_pxstats(cube, mode=mode, in_radius=rad, width=width, 
                                                                                            top_sigma=top_sigma, low_sigma=low_sigma, window=None, 
                                                                                            plot=plot, verbose=debug)
                                final_good_index_list = [idx for idx in list(good_index_list) if idx in final_good_index_list]
                                if 100*len(bad_index_list)/cube.shape[0] > perc:
                                    perc = 100*len(bad_index_list)/cube.shape[0]
                                    print("Percentile updated to {:.1f} based on stat".format(perc))
                                if plot_obs_cond:
                                    val = counter + (2*(ff%2)-1)*0.2
                                    if fi != 1 :
                                        label=filt+'- stat'
                                    else:
                                        label = None
                                    ax5.plot(UTC[good_index_list]-UTC_0, [val]*len(good_index_list), cols[ff][0]+markers_1[fi], label=label)
                                counter+=1
                                
                            if "ell" in badfr_critn_tmp:
                                idx_ell = badfr_critn_tmp.index("ell")
                                # default params
                                roundhi = 0.2
                                roundlo = -0.2
                                crop_sz = 11
                                # Update if provided
                                if "roundhi" in badfr_crit_tmp[idx_ell].keys():
                                    roundhi = badfr_crit_tmp[idx_ell]["roundhi"]
                                if "roundlo" in badfr_crit_tmp[idx_ell].keys():
                                    roundlo = badfr_crit_tmp[idx_ell]["roundlo"]
                                if "crop_sz" in badfr_crit_tmp[idx_ell].keys():
                                    crop_sz = badfr_crit_tmp[idx_ell]["crop_sz"]
                                crop_size = int(crop_sz*fwhm)
                                if not crop_sz%2:
                                    crop_size+=1
                                good_index_list, bad_index_list = cube_detect_badfr_ellipticity(cube, fwhm=fwhm, 
                                                                                                crop_size=crop_size, 
                                                                                                roundlo=roundlo,
                                                                                                roundhi=roundhi, 
                                                                                                plot=plot, verbose=debug)
                                if 100*len(bad_index_list)/cube.shape[0] > perc:
                                    perc = 100*len(bad_index_list)/cube.shape[0]
                                    print("Percentile updated to {:.1f} based on ell".format(perc))                                                                                        
                                final_good_index_list = [idx for idx in list(good_index_list) if idx in final_good_index_list]
                                if plot_obs_cond:
                                    val = counter + (2*(ff%2)-1)*0.2
                                    if fi != 1:
                                        label=filt+'- ell'
                                    else:
                                        label = None
                                    ax5.plot(UTC[good_index_list]-UTC_0, [val]*len(good_index_list), cols[ff][0]+markers_2[fi], label=label)                        
                                counter+=1
                                
                            if "bkg" in badfr_critn_tmp:
                                idx_bkg = badfr_critn_tmp.index("bkg")
                                crop_sz = 21
                                # Update if provided
                                if "crop_sz" in badfr_crit_tmp[idx_bkg].keys():
                                    crop_sz = badfr_crit_tmp[idx_bkg]["crop_sz"]
                                crop_size = int(crop_sz*fwhm)
                                if not crop_sz%2:
                                    crop_size+=1
                                # default params
                                sigma = 3
                                # Update if provided
                                if "thr" in badfr_crit_tmp[idx_bkg].keys():
                                    sigma = badfr_crit_tmp[idx_bkg]["thr"]
                                # infer rough bkg location in each frame
                                if isfile(outpath+"median_ADI2_{}{}{}.fits".format(labels[-1],filters[ff],dist_lab_tmp)):
                                    cen_adi_img = open_fits(outpath+"median_ADI2_{}{}{}.fits".format(labels[-1],filters[ff],
                                                                                                     dist_lab_tmp))
                                else:
                                    cen_adi_img = open_fits(outpath+"median_ADI2_{}{}{}.fits".format(labels[0],filters[ff],
                                                                                                     dist_lab_tmp))
                                med_x, med_y = fit2d_bkg_pos(np.array([cen_adi_img]), 
                                                             np.array([approx_xy_bkg[0]]), 
                                                             np.array([approx_xy_bkg[1]]), 
                                                             fwhm, fit_type=psf_model,
                                                             crop_sz=crop_sz,
                                                             sigfactor=sigfactor)
                                xy_bkg_derot = (med_x,med_y)
                                cy, cx = frame_center(cube[0])
                                center_bkg = (cx, cy)
                                x_bkg, y_bkg = interpolate_bkg_pos(xy_bkg_derot, 
                                                                   center_bkg, 
                                                                   derot_angles)
                                final_x_bkg, final_y_bkg = fit2d_bkg_pos(cube, 
                                                                         x_bkg, 
                                                                         y_bkg, 
                                                                         fwhm, 
                                                                         fit_type=psf_model,
                                                                         crop_sz=crop_sz,
                                                                         sigfactor=sigfactor)
                                # measure bkg star fluxes
                                n_fr, ny, nx = cube.shape
                                flux_bkg = np.zeros(n_fr)
                                crop_sz = int(6*fwhm)
                                if not crop_sz%2:
                                    crop_sz+=1
                                for ii in range(n_fr):
                                    cond1 = np.isnan(final_x_bkg[ii])
                                    cond3 = False
                                    if not cond1:
                                        cond1 = int(final_x_bkg[ii])<crop_sz
                                        cond3 = int(final_x_bkg[ii])>nx-crop_sz
                                    cond2 = np.isnan(final_y_bkg[ii])
                                    cond4 = False
                                    if not cond2:
                                        cond2 = int(final_y_bkg[ii])<crop_sz
                                        cond4 = int(final_y_bkg[ii])>ny-crop_sz
                                    
                                    if cond1 or cond2 or cond3 or cond4:
                                        flux_bkg[ii] = np.nan
                                    else:
                                        subframe = frame_crop(cube[ii], crop_sz,
                                                              cenxy=(int(final_x_bkg[ii]), 
                                                                      int(final_y_bkg[ii])),
                                                                      force=True, verbose=verbose)
                                        subpx_shifts = (final_x_bkg[ii]-int(final_x_bkg[ii]),
                                                        final_y_bkg[ii]-int(final_y_bkg[ii]))
                                        subframe = frame_shift(subframe, subpx_shifts[1],
                                                               subpx_shifts[0])
                                        _, flux_bkg[ii], _ = normalize_psf(subframe, fwhm=fwhm, 
                                                                           full_output=True, 
                                                                           verbose=verbose, debug=debug)
                                # infer outliers
                                med_fbkg = np.nanmedian(flux_bkg)
                                std_fbkg = np.nanstd(flux_bkg)
                                nonan_index_list = [i for i in range(n_fr) if not np.isnan(flux_bkg[i])]
                                good_index_list = [i for i in nonan_index_list if flux_bkg[i] > med_fbkg-sigma*std_fbkg]
                                bad_index_list = [i for i in range(n_fr) if i not in good_index_list]
                                final_good_index_list = [idx for idx in list(good_index_list) if idx in final_good_index_list]
                                if 100*len(bad_index_list)/cube.shape[0] > perc:
                                    perc = 100*len(bad_index_list)/cube.shape[0]
                                    print("Percentile updated to {:.1f} based on bkg".format(perc))
                                if plot_obs_cond:
                                    val = counter + (2*(ff%2)-1)*0.2
                                    if fi!= 1:
                                        label=filt+'- bkg'
                                    else:
                                        label = None
                                    ax5.plot(UTC[good_index_list]-UTC_0, [val]*len(good_index_list), cols[ff][0]+markers_1[fi], label=label)
                                counter+=1        
                                
                            if "shifts" in badfr_critn_tmp:
                                # Note: this only makes sense for bkg-based centering, 
                                # since satspots centering will linearly interpolate CEN cube stellar position 
                                # hence should not identify any outlier.
                                if not isfile(outpath+"TMP_shifts_fine_recentering_bkg_{}.fits".format(filt)):
                                    msg = "File with fine recentering {} does not exist. "
                                    msg+= "Is there a BKG star and have you run step 5?"
                                    fn = outpath+"TMP_shifts_fine_recentering_bkg_{}.fits".format(filt)
                                    raise NameError(msg.format(fn))
                                if rec_met != 'satspots':
                                    raise TypeError("For this bad frame removal criterion to work, only 'CENTER' i.e. satellite spot images must be used")
                                idx_shifts = badfr_critn_tmp.index("shifts")
                                # default params
                                thr = 0.4
                                err = 0.4
                                # Update if provided
                                if "thr" in badfr_crit_tmp[idx_shifts].keys():
                                    thr = badfr_crit_tmp[idx_shifts]["thr"]
                                if "err" in badfr_crit_tmp[idx_shifts].keys():
                                    err = badfr_crit_tmp[idx_shifts]["err"]
                                
                                # LOAD CEN SHIFTS (i.e. expected shift for OBJ CUBES if star perfectly)
                                good_cen_shift_x = open_fits(outpath+"TMP_shifts_cen_x{}_{}_{}.fits".format(labels[0],filters[ff],rec_met))
                                good_cen_shift_y = open_fits(outpath+"TMP_shifts_cen_y{}_{}_{}.fits".format(labels[0],filters[ff],rec_met))
                                if good_cen_idx is None:
                                    good_cen_shift_x = np.median(good_cen_shift_x)
                                    good_cen_shift_y = np.median(good_cen_shift_y)
                                elif isinstance(good_cen_idx,int):
                                    good_cen_shift_x = good_cen_shift_x[good_cen_idx]
                                    good_cen_shift_y = good_cen_shift_y[good_cen_idx]
                                elif isinstance(good_cen_idx,list):
                                    good_cen_shift_x = np.median(good_cen_shift_x[good_cen_idx])
                                    good_cen_shift_y = np.median(good_cen_shift_y[good_cen_idx])                                    
                                else:
                                    raise TypeError("good_cen_idx can only be int or None")
                                # INFER TOTAL CUBE SHIFTS
                                ## Load rough shifts
                                shifts_y = open_fits(outpath+"TMP_shifts_y{}_{}_{}.fits".format(labels[0],filters[ff],rec_met))
                                shifts_x = open_fits(outpath+"TMP_shifts_x{}_{}_{}.fits".format(labels[0],filters[ff],rec_met))
                                err = np.array([err]*len(shifts_y))
                                ## Load fine shifts
                                #if isfile(outpath+"TMP_shifts_fine_recentering_bkg_{}.fits".format(filters[ff])):
                                fine_shifts = open_fits(outpath+"TMP_shifts_fine_recentering_bkg_{}.fits".format(filters[ff]))
                                err = np.sqrt(np.power(err,2)+np.power(fine_shifts[-1],2))
                                # update shifts
                                shifts_x += fine_shifts[1]
                                shifts_y += fine_shifts[2]
                                final_dshifts = np.sqrt(np.power(shifts_x[:]-good_cen_shift_x,2)+np.power(shifts_y[:]-good_cen_shift_y,2))
                                n_fr = final_dshifts.shape[0]
                                write_fits(outpath+"TMP_final_dshifts.fits",final_dshifts)

                                if plot or debug:
                                    fig, ax1 = plt.subplots(1,1,figsize=(4,4))
                                    for i in range(n_fr):
                                        if final_dshifts[i] < thr:
                                            col = 'b'
                                        else:
                                            col = 'r'
                                        ax1.errorbar(i+1, final_dshifts[i], err[i], fmt=col+'o')
                                    ax1.plot([0,n_fr+1],[thr,thr],'k--')
                                    ax1.set_xlabel("Index of frame in cube")
                                    ax1.set_ylabel("Differential shift with respect to mask center (px)")
                                    ax1.set_ylim(-0.1,3)
                                    plt.savefig(outpath+"Residual_shifts_bkg_VS_satspots_{}_badfrrm.pdf".format(filters[ff]), bbox_inches='tight', format='pdf')
                                good_index_list = [i for i in range(n_fr) if (final_dshifts[i] < thr and err[i] < thr)]
                                bad_index_list = [i for i in range(n_fr) if i not in good_index_list]
                                final_good_index_list = [idx for idx in good_index_list if idx in final_good_index_list]
                                if 100*len(bad_index_list)/cube.shape[0] > perc:
                                    perc = 100*len(bad_index_list)/cube.shape[0]
                                    print("Percentile updated to {:.1f} based on shifts > {:.1f} px".format(perc,thr))
                                if plot_obs_cond:
                                    val = counter + (2*(ff%2)-1)*0.2
                                    if fi != 1:
                                        label=filt+'- shifts'
                                    else:
                                        label = None
                                    ax5.plot(UTC[good_index_list]-UTC_0, [val]*len(good_index_list), cols[ff][0]+markers_1[fi], label=label)
                                counter+=1  
                                
                            if "corr" in badfr_critn_tmp:
                                idx_corr = badfr_critn_tmp.index("corr")
                                # default params
                                thr = 0.8
                                perc = 0
                                ref = "median"
                                dist = 'pearson'
                                mode = 'annulus'
                                inradius = 10
                                width = 20
                                crop_sz = int(2*int(inradius+width)+3)
                                # update if provided
                                if "perc" in badfr_crit_tmp[idx_corr].keys():
                                    perc = max(perc, badfr_crit_tmp[idx_corr]["perc"])
                                if "thr" in badfr_crit_tmp[idx_corr].keys():
                                    thr = badfr_crit_tmp[idx_corr]["thr"]
                                else:
                                    thr = None
                                if "ref" in badfr_crit_tmp[idx_corr].keys():
                                    ref = badfr_crit_tmp[idx_corr]["ref"]
                                if ref== "median":
                                    good_frame = np.median(cube[final_good_index_list],axis=0)
                                else:
                                    good_frame = cube[badfr_crit_tmp[idx_corr]["ref"]]
                                if "crop_sz" in badfr_crit_tmp[idx_corr].keys():
                                    crop_sz = badfr_crit_tmp[idx_corr]["crop_sz"]
                                crop_size = int(crop_sz*fwhm)
                                if not crop_size%2:
                                    crop_size+=1
                                if  crop_size > cube.shape[-1] or crop_size > good_frame.shape[-1] : 
                                    crop_size = max([good_frame.shape[-1],cube.shape[-1]]) - 2
                                if "dist" in badfr_crit_tmp[idx_corr].keys(): 
                                    dist = badfr_crit_tmp[idx_corr]["dist"]
                                if "mode" in badfr_crit_tmp[idx_corr].keys():
                                    mode = badfr_crit_tmp[idx_corr]["mode"]
                                if "inradius" in badfr_crit_tmp[idx_corr].keys():
                                    mode = badfr_crit_tmp[idx_corr]["inradius"]
                                if "width" in badfr_crit_tmp[idx_corr].keys():
                                    mode = badfr_crit_tmp[idx_corr]["width"]
                                
                                good_index_list, bad_index_list = cube_detect_badfr_correlation(cube, good_frame, 
                                                                                                crop_size=crop_size, 
                                                                                                threshold=thr,
                                                                                                dist=dist, 
                                                                                                mode=mode,
                                                                                                inradius=inradius,
                                                                                                width=width,
                                                                                                percentile=perc, 
                                                                                                plot=plot, verbose=debug)
                                if plot:
                                    plt.savefig(outpath+"badfr_corr_plot{}{}.pdf".format(labels[fi],filt),bbox_inches='tight')                                       
                                final_good_index_list = [idx for idx in list(good_index_list) if idx in final_good_index_list]
                                if plot_obs_cond:
                                    val = counter + (2*(ff%2)-1)*0.2
                                    if fi != 1:
                                        label=filt+'- corr'
                                    else:
                                        label = None
                                    ax5.plot(UTC[good_index_list]-UTC_0, [val]*len(good_index_list), cols[ff][0]+markers_3[fi], label=label)                                                                 
                                counter+=1

                                
                        if plot_obs_cond:
                            ax5.set_xlabel('Time from start (s)')
                            ax6.set_xlabel('Time from start (s)')
                            ax6.set_ylabel('Wavelength')
                            color = 'tab:{}'.format(cols[ff])
                            if fi != 1:
                                label='{} - Sr'.format(filt)
                                if ff ==0:
                                    ax2.set_xlabel('Time from start (s)')
                                    ax2.set_ylabel('Strehl ratio (%)')
                            else:
                                label=None
                            ax2.plot(UTC[final_good_index_list]-UTC_0,strehl[final_good_index_list],cols[ff][0]+markers_1[fi], alpha=alphas[0], label=label)
                            if not coro:
                                color = 'tab:{}'.format(cols[ff])
    #                            if ff == 0 and fi ==0:
    #                                ax2b = ax2.twinx()
    #                                ax2b.set_ylabel('Stellar flux (ADUs/s)')
                                if fi != 1:
                                    label = '{} - flux'.format(filt)
                                    if ff == 0:
                                        ax2b = ax2.twinx()
                                        ax2b.set_ylabel('Stellar flux (ADUs/s)')
                                else:
                                    label=None
                                ax2b.plot(UTC[final_good_index_list]-UTC_0,fluxes[final_good_index_list],cols[ff][0]+markers_2[fi], alpha=alphas[1], label=label)
                            if ff == 0:    
                                color = 'tab:blue'
                                if fi != 1:
                                    ax4.set_xlabel('Time from start (s)')
                                    ax4.set_ylabel('Seeing (arcsec)', color=color)
                                ax4.plot(UTC[final_good_index_list]-UTC_0,seeing[final_good_index_list],'b'+markers_2[fi], alpha=alphas[0])
                                color = 'tab:red'
                                if fi != 1:
                                    ax4b = ax4.twinx()
                                    ax4b.set_ylabel('Coherence time (ms)', color=color)
    #                            ax4b = ax4.twinx()
    #                            ax4b.set_ylabel('Coherence time (ms)', color=color)                          
                                ax4b.plot(UTC[final_good_index_list]-UTC_0,tau0[final_good_index_list],'r'+markers_1[fi], alpha=alphas[1])
                                
                            val_fin = lbdas[ff]
                            ax6.plot(UTC[final_good_index_list]-UTC_0, [val_fin]*len(final_good_index_list), cols[ff][0]+markers_1[fi], label='{} - frames kept'.format(filt))
                              
                        cube = cube[final_good_index_list]
                        write_fits(outpath+"3_master{}_cube_clean_{}{}{}.fits".format(labels[fi],filt,dist_lab_tmp,bad_str), cube)
                        if fi != 1:
                            derot_angles = derot_angles[final_good_index_list]
                            write_fits(outpath+"3_master{}_derot_angles_clean_{}{}.fits".format(labels[fi],filt,bad_str), derot_angles)
                    
                                                      
                #for ff, filt in enumerate(filters):
                    #for fi,file_list in enumerate(obj_psf_list):
#                        if fi == 0:
#                            badfr_crit_tmp = badfr_crit_names
#                        elif fi == 1:
#                            badfr_crit_tmp = badfr_crit_names_psf
#                        else:
#                            break
                    #bad_str = "-".join(badfr_crit_tmp)
                    cube_ori = open_fits(outpath+"2{}_master{}_cube_{}{}.fits".format(label_cen_tmp,labels[fi],filt,dist_lab_tmp))
                    cube = open_fits(outpath+"3_master{}_cube_clean_{}{}{}.fits".format(labels[fi],filt,dist_lab_tmp,bad_str))
                    frac_good = cube.shape[0]/cube_ori.shape[0]
                    print("In total we keep {:.1f}% of all frames for {} {} \n".format(100*frac_good,filt,labels[fi]))

            if plot_obs_cond:
                ax1.legend(loc='upper left')
                ax2.legend(loc='upper left')
                ax3.legend(loc='upper left')
                ax4.legend(loc='upper left')
                ax1b.legend(loc='upper right')
                ax2b.legend(loc='upper right')
                ax3b.legend(loc='upper right')
                ax4b.legend(loc='upper right')            
                ax5.legend(loc='best')
                ax6.legend(loc='best')
                plt.savefig(outpath+"Observing_conditions_bef_VS_aft_trim_{}.pdf".format("-".join(badfr_crit_names)),bbox_inches='tight', format='pdf')
                plt.clf()
                                      
            if save_space:
                os.system("rm {}*1bpcorr.fits".format(outpath))                                    

        #************** 7. FINAL PSF + FLUX + FWHM (incl. CROP) ***************              
        if 7 in to_do:
            if len(obj_psf_list)>1:
                idx_psf=1
            else:
                idx_psf=0
            if isinstance(final_crop_szs[idx_psf], (float,int)):
                crop_sz_list = [int(final_crop_szs[idx_psf])]
            elif isinstance(final_crop_szs[idx_psf], list):
                crop_sz_list = final_crop_szs[idx_psf]
            else:
                raise TypeError("final_crop_sz_psf should be either int or list of int")
            for crop_sz in crop_sz_list:
                # PSF ONLY
                for ff, filt in enumerate(filters):
                    if not isfile(outpath+final_psfname+".fits") or overwrite[6]:
                        cube = open_fits(outpath+"3_master{}_cube_clean_{}{}.fits".format(labels[idx_psf],filt,"-".join(badfr_crit_names_psf)))
                        # crop
                        if cube.shape[1] > crop_sz or cube.shape[2] > crop_sz:
                            if crop_sz%2 != cube.shape[1]%2:
                                cube = cube_shift(cube,0.5,0.5, nproc=nproc)
                                cube = cube[:,1:,1:]
                            cube = cube_crop_frames(cube, crop_sz, verbose=verbose)
                        med_psf = np.median(cube,axis=0)
                        norm_psf, med_flux, fwhm = normalize_psf(med_psf, fwhm='fit', size=None, threshold=None, mask_core=None,
                                                                 model=psf_model, interpolation='lanczos4',
                                                                 force_odd=False, full_output=True, verbose=debug, debug=False)
                        if crop_sz%2: # only save final with VIP conventions, for use in postproc.  
                            write_fits(outpath+final_psfname+"{}.fits".format(filt), med_psf)
                            write_fits(outpath+final_psfname_norm+"{}.fits".format(filt), norm_psf)
                            header = fits.Header()
                            header['Flux 0'] = 'Flux scaled to coronagraphic DIT'
                            header['Flux 1'] = 'Flux measured in PSF image'
                            write_fits(outpath+final_fluxname+"{}.fits".format(filt),
                                       np.array([med_flux*dit_irdis/dit_psf_irdis, med_flux]),
                                       header=header)
                            write_fits(outpath+final_fwhmname+"{}.fits".format(filt), np.array([fwhm]))
                        write_fits(outpath+"4_final_psf_med{}_{}{:.0f}.fits".format(filt,psf_model,crop_sz), med_psf)
                        write_fits(outpath+"4_final_psf_med{}_{}_norm{:.0f}.fits".format(filt,psf_model,crop_sz), norm_psf)
                        write_fits(outpath+"4_final_psf_flux_med_{}_{}{:.0f}.fits".format(filt,psf_model,crop_sz), np.array([med_flux]))
                        write_fits(outpath+"4_final_psf_fwhm_{}_{}.fits".format(filt,psf_model), np.array([fwhm]))
                        
                        ntot = cube.shape[0]
                        fluxes = np.zeros(ntot)
                        for nn in range(ntot):
                            _, fluxes[nn], _ = normalize_psf(cube[nn], fwhm=fwhm, size=None, threshold=None, mask_core=None,
                                                             model=psf_model, interpolation='lanczos4',
                                                             force_odd=False, full_output=True, verbose=debug, debug=False)
                        write_fits(outpath+"4_final_psf_fluxes_{}_{}.fits".format(filt,psf_model), fluxes)            

            if save_space:
                os.system("rm {}*2cen.fits".format(outpath))
                                      
                
                
        #********** 8. SUBTRACT SAT SPOTS IF CEN cubes USED as OBJ cubes *********
        if 8 in to_do and use_cen_only:
            diff = int((ori_sz-bp_crop_sz)/2)
            for ff, filt in enumerate(filters):
                fwhm = open_fits(outpath+final_fwhmname+"{}.fits".format(filt))[0]
                psf_norm = open_fits(outpath+final_psfname_norm+"{}.fits".format(filt))
                crop_sz = int(6*fwhm)
                if not crop_sz %2:
                    crop_sz+=1
                y_shifts_cen = np.median(open_fits(outpath+"TMP_shifts_cen_y{}_{}_{}.fits".format(labels[0],filters[ff],rec_met)))
                x_shifts_cen = np.median(open_fits(outpath+"TMP_shifts_cen_x{}_{}_{}.fits".format(labels[0],filters[ff],rec_met)))
                xy_spots_tmp = tuple([(xy_spots[ff][i][0]-diff+x_shifts_cen,
                                       xy_spots[ff][i][1]-diff+y_shifts_cen) for i in range(len(xy_spots[ff]))])
                for tt in range(2): # do it twice for non-trimmed and trimmed
                    if tt == 0:
                        cube = open_fits(outpath+"2{}_master{}_cube_{}{}.fits".format(label_cen,labels[-1],filters[ff],dist_lab))
                    else:
                        cube = open_fits(outpath+"3_master{}_cube_clean_{}{}{}.fits".format(labels[-1],filt,dist_lab,"-".join(badfr_crit_names)))                  
                    cy, cx = frame_center(cube[0])
                    # Fit sat spots as psfs (measure flux)
                    fluxes = np.zeros([cube.shape[0],4])
                    for cc in range(cube.shape[0]):
                        xy_pos = []
                        for ss in range(4):
                            ## find exact position
                            y_tmp, x_tmp = fit_2dmoffat(cube[cc], crop=True, 
                                                        cent=xy_spots_tmp[ss],
                                                        cropsize=crop_sz,
                                                        fwhm=int(fwhm),
                                                        threshold=True,
                                                        sigfactor=6,
                                                        full_output=False)
                            xy_pos.append((x_tmp,y_tmp))
                            sub_array = frame_crop(cube[cc], crop_sz, cenxy=xy_pos[ss])
                            r = np.sqrt((y_tmp-cy)**2+(x_tmp-cx)**2)
                            ## measure flux
                            _, flux_tmp, _ = normalize_psf(sub_array, fwhm=fwhm, size=None, threshold=None, mask_core=None,
                                                                model=psf_model, interpolation='lanczos4',
                                                                force_odd=False, full_output=True, verbose=debug, debug=False)
                            if ss == 0:
                                ann_vals = get_annulus_segments(cube[cc], r-fwhm, 2*fwhm, mode='val')
                                _, ann_med, _ = sigma_clipped_stats(ann_vals, sigma=5)
                            fluxes[cc,ss] = flux_tmp - (ann_med*np.pi*(fwhm/2)**2)
                            cy_tmp, cx_tmp = frame_center(psf_norm)
                            # Subtract scaled norm psf at each position
                            if np.any(psf_norm.shape<cube[cc].shape):
                                psf_tmp = np.zeros_like(cube[cc])
                                psf_tmp[:psf_norm.shape[0],:psf_norm.shape[1]] = psf_norm*fluxes[cc,ss]
                            else:
                                psf_tmp = psf_norm*fluxes[cc,ss]
                            cube[cc] -= frame_shift(psf_tmp, y_tmp-cy_tmp, x_tmp-cx_tmp, border_mode='constant')
                                
                    if debug:
                        write_fits(outpath+"TMP_last_cube_cen-sat_spots.fits",cube[cc])
                        write_fits(outpath+"TMP_fluxes_sat_spots.fits",fluxes)
                        
                    if tt == 0:
                        write_fits(outpath+"2{}_master_cube_{}{}.fits".format(label_cen,filters[ff],dist_lab), cube)
                    else:
                        write_fits(outpath+"3_master_cube_clean_{}{}{}.fits".format(filt,dist_lab,"-".join(badfr_crit_names)), cube)
                                
                        
        #******************** 9. FINAL OBJ CUBE (BIN & CROP IF NECESSARY) ******************
        if 9 in to_do:
            if isinstance(final_crop_szs[0], (float,int)):
                crop_sz_list = [int(final_crop_szs[0])]
            elif isinstance(final_crop_szs[0], list):
                crop_sz_list = final_crop_szs[0]
            else:
                raise TypeError("final_crop_sz_psf should be either int or list of int")
            for cc, crop_sz in enumerate(crop_sz_list):
                # OBJ ONLY
                #for bb, bin_fac in enumerate(bin_fac_list):
                for ff, filt in enumerate(filters):
                    if not isfile(outpath+final_cubename+".fits") or overwrite[7]:
                        cube_notrim = open_fits(outpath+"2{}_master{}_cube_{}{}.fits".format(label_cen,labels[0],filters[ff],dist_lab))
                        cube = open_fits(outpath+"3_master_cube_clean_{}{}{}.fits".format(filt,dist_lab,"-".join(badfr_crit_names)))
                        if use_cen_only:
                            fi_tmp = -1
                        else:
                            fi_tmp=0
                        derot_angles = open_fits(outpath+"3_master{}_derot_angles_clean_{}{}.fits".format(labels[fi_tmp],filt,"-".join(badfr_crit_names)))
                        derot_angles_notrim = open_fits(outpath+"1_master_derot_angles{}.fits".format(labels[fi_tmp]))
                        ntot = cube.shape[0]
                        ntot_notrim = cube_notrim.shape[0]
                        if bin_fac != 1:
                            bin_fac = int(bin_fac)
                            ntot_bin = int(np.ceil(ntot/bin_fac))
                            ntot_bin_notrim = int(np.ceil(ntot_notrim/bin_fac))
                            cube_bin = np.zeros([ntot_bin,cube.shape[1],cube.shape[2]])
                            cube_bin_notrim = np.zeros([ntot_bin_notrim,cube_notrim.shape[1],cube_notrim.shape[2]])
                            derot_angles_bin = np.zeros(ntot_bin)
                            derot_angles_bin_notrim = np.zeros(ntot_bin_notrim)
                            for nn in range(ntot_bin):
                                cube_bin[nn] = np.median(cube[nn*bin_fac:(nn+1)*bin_fac],axis=0)
                                derot_angles_bin[nn] = np.median(derot_angles[nn*bin_fac:(nn+1)*bin_fac])
                            for nn in range(ntot_bin_notrim):
                                cube_bin_notrim [nn] = np.median(cube_notrim[nn*bin_fac:(nn+1)*bin_fac],axis=0)
                                derot_angles_bin_notrim[nn] = np.median(derot_angles_notrim[nn*bin_fac:(nn+1)*bin_fac])
                            cube = cube_bin
                            cube_notrim = cube_bin_notrim
                            derot_angles = derot_angles_bin
                            derot_angles_notrim = derot_angles_bin_notrim
                        if not cc:
                            write_fits(outpath+final_cubename+"_full{}.fits".format(filt), cube)
                        # crop
                        if cube.shape[1] > crop_sz or cube.shape[2] > crop_sz:
                            if crop_sz%2 != cube.shape[1]%2:
                                cube = cube_shift(cube,0.5,0.5, nproc=nproc)
                                cube = cube[:,1:,1:]
                                cube_notrim = cube_shift(cube_notrim,0.5,0.5, nproc=nproc)
                                cube_notrim = cube_notrim[:,1:,1:]
                            cube = cube_crop_frames(cube,crop_sz,verbose=verbose)
                            cube_notrim = cube_crop_frames(cube_notrim,crop_sz,verbose=verbose)
                        flux = open_fits(outpath+final_fluxname+"{}.fits".format(filt))
                        if crop_sz%2: # only save final with VIP conventions, for use in postproc.
                            write_fits(outpath+final_cubename+"{}.fits".format(filt), cube)
                            write_fits(outpath+final_cubename_norm+"{}.fits".format(filt), cube/flux[0])
                            write_fits(outpath+final_anglename+"{}.fits".format(filt), derot_angles)
                        write_fits(outpath+"4_final_cube_all_bin{:.0f}{}_{}_{:.0f}.fits".format(bin_fac,dist_lab,filt, crop_sz), cube_notrim)
                        write_fits(outpath+"4_final_cube_all_bin{:.0f}{}_{}_{:.0f}_norm.fits".format(bin_fac,dist_lab,filt, crop_sz), cube_notrim/flux[0])
                        write_fits(outpath+"4_final_derot_angles_all_bin{:.0f}_{}.fits".format(bin_fac,filt), derot_angles_notrim)
                        write_fits(outpath+"4_final_cube_bin{:.0f}{}_{}_{:.0f}.fits".format(bin_fac,dist_lab,filt, crop_sz), cube)
                        write_fits(outpath+"4_final_cube_bin{:.0f}{}_{}_{:.0f}_norm.fits".format(bin_fac,dist_lab,filt, crop_sz), cube/flux[0])
                        write_fits(outpath+"4_final_derot_angles_bin{:.0f}_{}.fits".format(bin_fac,filt), derot_angles)
                        
                        med_psf = np.median(cube,axis=0)
                                        
                    if not coro and not isfile(outpath+"4_final_obj_fluxes_bin{:.0f}{}_{}.fits".format(bin_fac,dist_lab,filt)):
                        
                        if isinstance(final_crop_sz_psf, int) : final_crop_sz_psf = [final_crop_sz_psf]
                        for crop_siz_i in final_crop_sz_psf : 
                            norm_psf, med_flux, fwhm = normalize_psf(med_psf, fwhm='fit', size=crop_siz_i, threshold=None, 
                                                                     mask_core=None, model=psf_model, 
                                                                     interpolation='lanczos4', force_odd=False, full_output=True, 
                                                                     verbose=debug, debug=False)
                                                 
                            write_fits(outpath+"4_final_obj_med{}_{}_{}.fits".format(dist_lab,filt,crop_siz_i), med_psf)
                            write_fits(outpath+"4_final_obj_norm_med{}_{}_{}.fits".format(dist_lab,filt,crop_siz_i), norm_psf)
                            write_fits(outpath+"4_final_obj_flux_med{}_{}_{}.fits".format(dist_lab,filt,crop_siz_i), np.array([med_flux]))
                            write_fits(outpath+"4_final_obj_fwhm{}_{}_{}.fits".format(dist_lab,filt,crop_siz_i), np.array([fwhm]))
                        
                            if crop_siz_i == min(final_crop_sz_psf):
                                ntot = cube.shape[0]
                                fluxes = np.zeros(ntot)
                                for nn in range(ntot):
                                    _, fluxes[nn], _ = normalize_psf(cube[nn], fwhm=fwhm, size=None, threshold=None, mask_core=None,
                                                                     model=psf_model, interpolation='lanczos4',
                                                                     force_odd=False, full_output=True, verbose=debug, debug=False)
                                write_fits(outpath+"4_final_obj_fluxes_bin{:.0f}{}_{}.fits".format(bin_fac,dist_lab,filt), fluxes)

            if save_space:
                os.system("rm {}3_*.fits".format(outpath))
                
        #******************* 10. SCALE FACTOR CALCULATION *********************

                
        if 10 in to_do and len(filters)>1 and not separate_trim:
            nfp = 2 # number of free parameters for simplex search
            n_ch = len(filters)
            fluxes = np.zeros(n_ch)
            lbdas_tmp = np.zeros_like(fluxes)
            print("************* 10. FINDING SCALING FACTORS ***************")
            for ff, filt in enumerate(filters):
                fluxes[ff] = open_fits(outpath+final_fluxname+"{}.fits".format(filt))
                lbdas_tmp[ff] = open_fits(outpath+final_fwhmname+"{}.fits".format(filt))
                derot_angles = open_fits(outpath+final_anglename+"{}.fits".format(filt))
                
            n_cubes = len(derot_angles)
            scal_vector = np.zeros([n_cubes,n_ch])
            flux_fac_vec = np.zeros([n_cubes,n_ch])
            resc_cube_res_all = []
            for i in range(n_cubes):
                for ff, filt in enumerate(filters):
                    frame = open_fits(outpath+final_cubename+"{}.fits".format(filt))[i]
                    if ff == 0:
                        master_cube = np.zeros([n_ch,frame.shape[-2],frame.shape[-1]])
                        if i == 0:
                            if isinstance(mask_scal,str):
                                mask_scal = open_fits(mask_scal)
                                ny_m, nx_m = mask_scal.shape
                                ny, nx = frame.shape
                                if ny_m>ny:
                                    if ny_m%2 != ny%2:
                                        mask_scal = frame_shift(mask_scal, 0.5, 0.5)
                                        mask_scal = mask_scal[1:,1:]
                                    mask_scal=frame_crop(mask_scal,ny)
                                elif ny>ny_m:
                                    mask_scal_fin = np.zeros_like(frame)
                                    mask_scal_fin[:ny_m,:nx_m]=mask_scal
                                    mask_scal = frame_shift(mask_scal_fin, 
                                                            (ny-ny_m)/2, 
                                                            (nx-nx_m)/2)
                            else:
                                mask = np.ones_like(frame)
                                if mask_scal[0]:
                                    if mask_scal[1]:
                                        mask_scal = get_annulus_segments(mask, mask_scal[0]/plsc_med, 
                                                                         mask_scal[1]/plsc_med, nsegm=1, theta_init=0,
                                                                         mode="mask")
                                    else:
                                        mask_scal = mask_circle(mask, mask_scal[0]/plsc_med)
                                if debug:
                                    write_fits(outpath+"TMP_mask_scal.fits", mask_scal)
                    master_cube[ff] = frame
                res = find_scal_vector(master_cube, lbdas_tmp, fluxes, 
                                       mask=mask_scal, nfp=nfp, debug=debug)
                scal_vector[i], flux_fac_vec[i] = res

                resc_cube = master_cube.copy()
                for z in range(resc_cube.shape[0]):
                    resc_cube[z]*=flux_fac_vec[i,z]
                resc_cube = _cube_resc_wave(resc_cube, scal_vector[i])
                resc_cube_res = np.zeros([master_cube.shape[0]+1,master_cube.shape[1],master_cube.shape[2]])
                resc_cube_res[:-1] = resc_cube
                resc_cube_res[-1] = resc_cube[-1]-resc_cube[0]
                write_fits(outpath+"TMP_resc_cube_res.fits", resc_cube_res)
                resc_cube_res_all.append(resc_cube_res[-1])
            resc_cube_res_all = np.array(resc_cube_res_all)
            write_fits(outpath+"TMP_resc_cube_res_all.fits", resc_cube_res_all)
            # perform simple SDI
            derot_cube = cube_derotate(resc_cube_res_all, derot_angles, nproc=nproc)
            sdi_frame = np.median(derot_cube,axis=0)
            write_fits(outpath+"median_SDI.fits", 
                       mask_circle(sdi_frame,coro_sz))
            stim_map = compute_stim_map(derot_cube)
            inv_stim_map = compute_inverse_stim_map(resc_cube_res_all, derot_angles)
            thr = np.percentile(mask_circle(inv_stim_map,coro_sz), 99.9)
            norm_stim_map = stim_map/thr
            stim_maps = np.array([mask_circle(stim_map,coro_sz),
                                  mask_circle(inv_stim_map,coro_sz),
                                  mask_circle(norm_stim_map,coro_sz)])
            write_fits(outpath+"median_SDI_stim.fits", stim_maps)

            final_scal_vector = np.median(scal_vector,axis=0)
            final_flux_fac = np.median(flux_fac_vec,axis=0)
            std_scal_vector = np.std(scal_vector,axis=0)
            std_flux_fac_vector = np.std(flux_fac_vec,axis=0)
            print("original scal guess: ",lbdas_tmp[-1]/lbdas_tmp[:])
            print("original flux fac guess: ",fluxes[-1]/fluxes[:])
            print("final scal result: ",final_scal_vector)
            print("final flux fac result ({:.0f}): ".format(nfp),final_flux_fac)
            print("std scal (from cube to cube): ",std_scal_vector)
            print("std flux fac (from cube to cube): ",std_flux_fac_vector)
            write_fits(outpath+final_scalefac_name, final_scal_vector)
            write_fits(outpath+"final_flux_fac.fits", final_flux_fac)
            write_fits(outpath+"final_scale_fac_std.fits", std_scal_vector)
            write_fits(outpath+"final_flux_fac.fits", std_flux_fac_vector)
  
    return None