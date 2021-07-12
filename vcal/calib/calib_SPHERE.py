#! /usr/bin/env python

"""
Module with the calibration routine for SPHERE data.
"""

__author__ = 'V. Christiaens'
__all__ = ['calib']

from astropy.io import fits
from astropy.stats import sigma_clipped_stats
import ast
import csv
import json
import numpy as np
import os
import pdb
from os.path import isfile, isdir#, join, dirname, abspath
import pathlib
import photutils
import vip_hci
from ..utils import make_lists, sph_ifs_correct_spectral_xtalk


def calib(params_calib_name='VCAL_params_calib.json'):
    """
    Basic calibration of SPHERE data using calibration parameters provided in 
    json file.
    
    Input:
    ******
    params_calib_name: str, opt
        Full path + name of the json file containing calibration parameters.
        
    Output:
    *******
    None. All calibrated products are written as fits files, and can then be 
    used for preprocessing.
    
    """
    with open(params_calib_name, 'r') as read_file_params_calib:
        params_calib = json.load(read_file_params_calib)
        
    path = params_calib['path'] #"/Volumes/Val_stuff/VLT_SPHERE/J1900_3645/" # parent path
    inpath = path+"raw/"
    inpath_filt_table = params_calib.get('inpath_filt_table','~/') #"/Applications/ESO/spher-calib-0.38.0/cal/"
    
    # subdirectories
    path_ifs = path+"IFS_reduction/"
    path_irdis = path+"IRDIS_reduction/"
    outpath_filenames = path+"filenames/"
    outpath_ifs_calib = path_ifs+"1_calib_esorex/calib/"
    outpath_ifs_sof = path_ifs+"1_calib_esorex/sofs/"
    outpath_ifs_fits = path_ifs+"1_calib_esorex/fits/"
    outpath_irdis_sof = path_irdis+"1_calib_esorex/sofs/"
    outpath_irdis_fits = path_irdis+"1_calib_esorex/fits/"
    if not isdir(path_ifs):
        os.makedirs(path_ifs)
    if not isdir(path_irdis):
        os.makedirs(path_irdis)
    if not isdir(outpath_filenames):
        os.makedirs(outpath_filenames)
    if not isdir(outpath_ifs_calib):
        os.makedirs(outpath_ifs_calib)
    if not isdir(outpath_ifs_sof):
        os.makedirs(outpath_ifs_sof)
    if not isdir(outpath_irdis_sof):
        os.makedirs(outpath_irdis_sof)
    if not isdir(outpath_ifs_fits):
        os.makedirs(outpath_ifs_fits)
    if not isdir(outpath_irdis_fits):
        os.makedirs(outpath_irdis_fits)
        
    # First run  dfits *.fits |fitsort DET.ID DET.SEQ1.DIT INS1.FILT.NAME INS1.OPTI2.NAME DPR.TYPE INS4.COMB.ROT
    # then adapt below
    # PARAMS    
    dit_ifs = params_calib.get('dit_ifs',None)
    dit_irdis = params_calib.get('dit_irdis',None)
    dit_psf_ifs = params_calib.get('dit_psf_ifs',None)
    dit_psf_irdis = params_calib.get('dit_psf_irdis',None)
    dit_cen_ifs = params_calib.get('dit_cen_ifs',None) # or None if there is no CEN cube
    dit_cen_irdis = params_calib.get('dit_cen_irdis',None) # or None if there is no CEN cube
    
    filt1 = params_calib.get('filt1',None) #'INS1.FILT.NAME'
    filt2 = params_calib.get('filt2',None) #'INS1.OPTI2.NAME'
    filters = params_calib.get('filters',None) #DBI filters (list of 2 elements) or CI filter (single string)
      
    # Reduction params
    ## Check the sky cubes by eye to select appropriate value for params below:
    good_sky_irdis = params_calib.get('good_sky_irdis',["all"]) #'all' or list of indices. If empty, will either use_ins_bg or just subtract the dark.
    good_sky_list = params_calib.get('good_sky_list',["all"]) #which skies to use for IFS? If [-1]: just uses the first frame of the first sky, if a list of good skies: uses all frames from all those cubes, if ['all']: uses them all.
    good_psf_sky_list = params_calib.get('good_psf_sky_list',["all"])
    good_psf_sky_irdis = params_calib.get('good_psf_sky_irdis',["all"])
    manual_sky = params_calib.get('manual_sky',1) # the sky evolves quickly with time, in case of poor sky sampling during sequence the background level after sky subtraction can be anywhere between -15 and +15 instead of 0
    mask_pca_sky_sub = params_calib.get('mask_pca_sky_sub',[250,420,0,0,0])
    # => First try with manual_sky set to False (no need to change params below then). If average background level different than 0 => re-run by setting it to True (possibly adaot values below -- in partciular for the psf): this will subtract manually the average pixel values measured at the provided coords (ideally corners far from star)
    corner_coords = params_calib.get('corner_coords',[[39,215],[72,42],[247,78],[216,250]]) # x,y coords where the sky will be manually estimated # !!! CAREFUL IF STAR NOT CENTERED => adapt
    msky_ap = params_calib.get('msky_ap',15) #aperture for manual sky level estimation
    corner_coords_psf = params_calib.get('corner_coords_psf',[[98,88]]) # x,y coords where the sky will be manually estimated # !!! CAREFUL IF STAR NOT CENTERED => adapt
    msky_ap_psf = params_calib.get('msky_ap_psf',20)
    
    
    ############### PARAMS THAT WILL LIKELY NOT NEED TO BE CHANGED ################
    
    # TO DO LIST:
    do_irdis = params_calib['do_irdis']
    do_ifs = params_calib['do_ifs']
    to_do = params_calib['to_do']
    # double-check or overwrite other params depending on do_ifs and do_irdis:
    if do_ifs:
        if dit_ifs is None:
            raise ValueError("dit_ifs should be provided if do_ifs is True")
    else:
        dit_ifs=None
    if do_irdis:
        if dit_irdis is None:
            raise ValueError("dit_irdis should be provided if do_irdis is True")
    else:
        dit_irdis=None
        
    
    #0. file list for both instruments
    ## 1-6. IRDIS
    #1. master dark
    #2. calculate gains
    #3. good SKY BKG or INS BG (in order of priority)
    #4. make master SKY BKG or INS BG (or pca sky subtr)
    #5. master flat
    #6. actual reduction
    ## 10-19 IFS
    
    instr = params_calib['instr']# instrument name in file name
    science_mode = params_calib['science_mode'] # current choice between {'DBI','CI'}
    mode = params_calib.get('mode','YJH') # only matters for IFS data calibration
    
    overwrite_sof = params_calib['overwrite_sof']
    overwrite_fits = params_calib['overwrite_fits']
    
    #crop_sz_irdis = 800 # size of the centered window extracted from the 1024x1024 frames. Recommended: 400 to make process much faster (corresponding to keeping ~2.4'' radius). # Except: coronagraphic obs => check for bkg stars
    #crop_sz_psf_irdis = 700
    #crop_sz_ifs = 0
    #crop_sz_psf_ifs = 0
    
    illum_pattern_corr = params_calib.get('illum_pattern_corr',1)
    flat_fit = params_calib.get('flat_fit',1) # whether to fit flat with polynomial
    large_scale_flat = params_calib.get('large_scale_flat',"some") # choice between 'all' (v1.38 manual), 'some' (v1.40? cf. dr recipe), False (not used, v1.40? cf. flat recipe)
    flat_smooth_method_idx = params_calib.get('flat_smooth_method_idx',1)# default 1 in esorex recipe (0: CPL filter, 1: FFT filter)
    flat_smooth_length = params_calib.get('flat_smooth_length',5)
    specpos_distort_corr = params_calib.get('specpos_distort_corr',1) # default is True
    specpos_nonlin_corr = params_calib.get('specpos_nonlin_corr',1) # default is True (cfr. D. Mesa 2015)
    xtalk_corr = params_calib.get('xtalk_corr',0) # whether cross talk should be corrected
    pca_subtr = params_calib.get('pca_subtr',1)  # whether to subtract dark and sky using pca
    npc = params_calib.get('npc',1)
    npc_psf = params_calib.get('npc_psf',npc)
    dark_ifs = params_calib.get('dark_ifs',[None]) # list containing either False or any combination of 'OBJ', 'PSF', 'CEN' and 'FLAT'. Tells whether to subtract the MASTER dark, and if so for which type of files. Recommended: either [False] or ['FLAT'] (in most cases a SKY is available for OBJ, CEN or PSF which already includes a DARK). If ['FLAT'] just provide a DARK file with the min DIT among FLATs in the raw folder (and remove the DARK of the OBJ!).
    indiv_fdark = params_calib.get('indiv_fdark',1)  # whether subtract individual dark to each flat
    poly_order_wc = params_calib.get('poly_order_wc',1) # used to find wavelength model
    wc_win_sz = params_calib.get('wc_win_sz',2)# default: 4
    sky=params_calib.get('sky',1) # for IFS only, will subtract the sky before the science_dr recipe (corrects also for dark, incl. bias, and vast majority of bad pixels!!)
    
    ### Formatting
    skysub_lab_IRD = "skysub/"
    skysub_lab_IFS = "skycorr_IFS/"
    bpcorr_lab_IFS = "bpcorr_IFS/"
    xtalkcorr_lab_IFS = "xtalkcorr_IFS/"
    
    ## 0. Create list of dictionaries
    if 0 in to_do or not isfile(path+"dico_files.csv"):
        dico_lists = make_lists(inpath, outpath_filenames, dit_ifs=dit_ifs, 
                                dit_irdis=dit_irdis, dit_psf_ifs=dit_psf_ifs, 
                                dit_psf_irdis=dit_psf_irdis, dit_cen_ifs=dit_cen_ifs, 
                                dit_cen_irdis=dit_cen_irdis,filt1=filt1, 
                                filt2=filt2)
        w = csv.writer(open(path+"dico_files.csv", "w"))
        for key, val in dico_lists.items():
            w.writerow([key, val])
    else:
        dico_lists = {}
        reader = csv.reader(open(path+'dico_files.csv', 'r'))
        for row in reader:
             dico_lists[row[0]] = ast.literal_eval(row[1])
        
    ## 1-5 IRDIS
    if do_irdis:
        # SKY labels
        label_fd = 'fake_' # label for fake master dark
        label_ss = ''
        if pca_subtr:
            label_ss = skysub_lab_IRD # label for sky subtracted cubes
            if not isdir(inpath+label_ss):
                os.makedirs(inpath+label_ss)
        manual_sky_irdis = manual_sky
        manual_sky_irdis_psf = manual_sky
        pca_subtr_psf = pca_subtr
        pca_subtr_cen = pca_subtr
        
        # DARKS
        if 1 in to_do:
            ## OBJECT
            if not isfile(outpath_irdis_sof+"master_dark.sof") or overwrite_sof:
                dark_list_irdis = dico_lists['dark_list_irdis']
                with open(outpath_irdis_sof+"master_dark.sof", 'w+') as f:
                    for ii in range(len(dark_list_irdis)):
                        f.write(inpath+dark_list_irdis[ii]+'\t'+'IRD_DARK_RAW\n')
            if not isfile(outpath_irdis_fits+"master_dark.fits") or overwrite_sof or overwrite_fits:
                command = "esorex sph_ird_master_dark"
                command+= " --ird.master_dark.sigma_clip=10.0"
                command+= " --ird.master_dark.save_addprod=TRUE"
                command+= " --ird.master_dark.outfilename={}master_dark.fits".format(outpath_irdis_fits)
                command+= " --ird.master_dark.badpixfilename={}master_badpixelmap.fits".format(outpath_irdis_fits)
                command+= " {}master_dark.sof".format(outpath_irdis_sof)
                os.system(command)
        
        # GAINS
        if 2 in to_do:
            ## OBJECT
            if not isfile(outpath_irdis_sof+"master_gain.sof") or overwrite_sof:
                gain_list_irdis = dico_lists['gain_list_irdis']
                with open(outpath_irdis_sof+"master_gain.sof", 'w+') as f:
                    for ii in range(len(gain_list_irdis)):
                        f.write(inpath+gain_list_irdis[ii]+'\t'+'IRD_GAIN_RAW\n')
                    f.write("{}master_badpixelmap.fits".format(outpath_irdis_fits)+'\t'+'IRD_STATIC_BADPIXELMAP\n')
            if not isfile(outpath_irdis_fits+"master_gain.fits") or overwrite_sof or overwrite_fits:
                command = "esorex sph_ird_gain"
                command+= " --ird.gain.save_addprod=TRUE"
                command+= " --ird.gain.outfilename={}master_gain_map.fits".format(outpath_irdis_fits)
                command+= " --ird.gain.nonlin_filename={}nonlin_map.fits".format(outpath_irdis_fits)
                command+= " --ird.gain.nonlin_bpixname={}nonlin_badpixelmap.fits".format(outpath_irdis_fits)
                command+= " --ird.gain.vacca=TRUE"
                command+= " {}master_gain.sof".format(outpath_irdis_sof)
                os.system(command)
                
        # Identify good SKY BKG / INS BKG (in order of priority)
        if 3 in to_do or 4 in to_do:
            # OBJ
            ## sky or ins bg list?     
            sky_list_irdis = dico_lists['sky_list_irdis']
            ins_bg_list_irdis = dico_lists['ins_bg_list_irdis']
            if len(sky_list_irdis) > 0 and good_sky_irdis is not None:
                if -1 in good_sky_irdis:
                    tmp = vip_hci.fits.open_fits(inpath+sky_list_irdis[0])
                    master_sky = np.zeros([1,tmp.shape[1],tmp.shape[2]])
                    master_sky[0] = tmp[0]
                    vip_hci.fits.write_fits("{}master_sky_cube.fits".format(outpath_irdis_fits), master_sky) # just take the first (closest difference in time to that of consecutive SCIENCE cubes - reproduce best the remanence effect)
                    sky_list_irdis = [sky_list_irdis[0]]
                elif 'all' in good_sky_irdis:
                    counter=0
                    nsky = len(sky_list_irdis)
                    for gg in range(nsky):
                        tmp = vip_hci.fits.open_fits(inpath+sky_list_irdis[gg])
                        if counter == 0:
                            master_sky = np.zeros([nsky*tmp.shape[0],tmp.shape[1],tmp.shape[2]])
                        master_sky[counter:counter+tmp.shape[0]] = tmp
                        counter+=tmp.shape[0]
                    #master_sky = np.median(master_sky,axis=0)
                    vip_hci.fits.write_fits("{}master_sky_cube.fits".format(outpath_irdis_fits), master_sky[:counter]) # just take the first (closest difference in time to that of consecutive SCIENCE cubes - reproduce best the remanence effect)
                elif len(good_sky_irdis)>0:
                    counter=0
                    nsky = len(good_sky_irdis)
                    sky_list_irdis_tmp = []
                    for gg in good_sky_irdis:
                        tmp = vip_hci.fits.open_fits(inpath+sky_list_irdis[gg])
                        if counter == 0:
                            master_sky = np.zeros([nsky*tmp.shape[0],tmp.shape[1],tmp.shape[2]])
                        master_sky[counter:counter+tmp.shape[0]] = tmp
                        counter+=tmp.shape[0]
                        sky_list_irdis_tmp.append(sky_list_irdis[ii])
                    #master_sky = np.median(master_sky,axis=0)
                    vip_hci.fits.write_fits("{}master_sky_cube.fits".format(outpath_irdis_fits), master_sky[:counter]) # just take the first (closest difference in time to that of consecutive SCIENCE cubes - reproduce best the remanence effect)
                    sky_list_irdis = sky_list_irdis_tmp
                else:
                    raise TypeError("good_sky_irdis format not recognised")
            elif len(ins_bg_list_irdis)>0:
                nsky = len(ins_bg_list_irdis)
                counter = 0
                for ii in range(len(ins_bg_list_irdis)):
                    tmp = vip_hci.fits.open_fits(inpath+ins_bg_list_irdis[ii])
                    if counter == 0:
                        master_sky = np.zeros([nsky*tmp.shape[0],tmp.shape[1],tmp.shape[2]])
                    master_sky[counter:counter+tmp.shape[0]] = tmp
                    counter+=tmp.shape[0]
                #master_sky = np.median(master_sky,axis=0)
                vip_hci.fits.write_fits("{}master_sky_cube.fits".format(outpath_irdis_fits), master_sky[:counter]) # just take the first (closest difference in time to that of consecutive SCIENCE cubes - reproduce best the remanence effect)
            else:
                print("WARNING: no SKY cube available.")
                print("Sky subtraction proxy based on median background value")
            
            # PSF
            ## sky or ins bg list?     
            psf_sky_list_irdis = dico_lists['psf_sky_list_irdis']
            psf_ins_bg_list_irdis = dico_lists['psf_ins_bg_list_irdis']
            if len(psf_sky_list_irdis) > 0:
                if -1 in good_psf_sky_irdis:
                    tmp = vip_hci.fits.open_fits(inpath+psf_sky_list_irdis[0])
                    master_psf_sky = np.zeros([1,tmp.shape[1],tmp.shape[2]])
                    master_psf_sky[0] = tmp[0]
                    vip_hci.fits.write_fits("{}master_sky_psf_cube.fits".format(outpath_irdis_fits), master_psf_sky) # just take the first (closest difference in time to that of consecutive SCIENCE cubes - reproduce best the remanence effect)
                elif 'all' in good_psf_sky_irdis:
                    counter=0
                    nsky = len(psf_sky_list_irdis)
                    for gg in range(nsky):
                        tmp = vip_hci.fits.open_fits(inpath+psf_sky_list_irdis[gg])
                        if counter == 0:
                            master_psf_sky = np.zeros([nsky*tmp.shape[0],tmp.shape[1],tmp.shape[2]])
                        master_psf_sky[counter:counter+tmp.shape[0]] = tmp
                        counter+=tmp.shape[0]
                    #master_sky = np.median(master_sky,axis=0)
                    vip_hci.fits.write_fits("{}master_sky_psf_cube.fits".format(outpath_irdis_fits), master_psf_sky[:counter]) 
                elif len(good_psf_sky_irdis)>0:
                    counter=0
                    nsky = len(good_psf_sky_irdis)
                    for gg in good_psf_sky_irdis:
                        tmp = vip_hci.fits.open_fits(inpath+psf_sky_list_irdis[gg])
                        if counter == 0:
                            master_psf_sky = np.zeros([nsky*tmp.shape[0],tmp.shape[1],tmp.shape[2]])
                        master_psf_sky[counter:counter+tmp.shape[0]] = tmp
                        counter+=tmp.shape[0]
                    #master_sky = np.median(master_sky,axis=0)
                    vip_hci.fits.write_fits("{}master_sky_psf_cube.fits".format(outpath_irdis_fits), master_psf_sky[:counter]) 
                elif good_psf_sky_irdis is None:
                    pass
                else:
                    raise TypeError("good_psf_sky_irdis format not recognised")
            elif len(psf_ins_bg_list_irdis)>0:
                nsky = len(psf_ins_bg_list_irdis)
                counter = 0
                for ii in range(len(psf_ins_bg_list_irdis)):
                    tmp = vip_hci.fits.open_fits(inpath+psf_ins_bg_list_irdis[ii])
                    if counter == 0:
                        master_psf_sky = np.zeros([nsky*tmp.shape[0],tmp.shape[1],tmp.shape[2]])
                    master_psf_sky[counter:counter+tmp.shape[0]] = tmp
                    counter+=tmp.shape[0]
                #master_sky = np.median(master_sky,axis=0)
                vip_hci.fits.write_fits("{}master_sky_psf_cube.fits".format(outpath_irdis_fits), master_psf_sky[:counter]) 

        if not isfile("{}master_sky_cube.fits".format(outpath_irdis_fits)):
            pca_subtr = False
            manual_sky_irdis = True
        if not isfile("{}master_sky_psf_cube.fits".format(outpath_irdis_fits)):
            pca_subtr_psf = False
            manual_sky_irdis_psf = True

        # FLAT + final bp map
        if 4 in to_do:
            if not isfile(outpath_irdis_sof+"master_flat.sof") or overwrite_sof:
                flat_dark_list_irdis = dico_lists['flat_dark_list_irdis']
                flat_list_irdis = dico_lists['flat_list_irdis']
                with open(outpath_irdis_sof+"master_flat.sof", 'w+') as f:
                    for ii in range(len(flat_list_irdis)):
                        f.write(inpath+flat_list_irdis[ii]+'\t'+'IRD_FLAT_FIELD_RAW\n')
                    if len(flat_dark_list_irdis)>0:
                        if len(flat_list_irdis)%len(flat_dark_list_irdis) == 0 and len(flat_list_irdis)/len(flat_dark_list_irdis) <= 3:
                            for ii in range(len(flat_dark_list_irdis)):
                                f.write(inpath+flat_dark_list_irdis[ii]+'\t'+'IRD_DARK_RAW\n')
                        else:
                            f.write("{}master_dark.fits".format(outpath_irdis_fits)+'\t'+'IRD_MASTER_DARK\n')
                    else:
                        f.write("{}master_dark.fits".format(outpath_irdis_fits)+'\t'+'IRD_MASTER_DARK\n')
                    f.write("{}master_badpixelmap.fits".format(outpath_irdis_fits)+'\t'+'IRD_STATIC_BADPIXELMAP')
         
            if not isfile(outpath_irdis_fits+"master_flat.fits") or overwrite_sof or overwrite_fits:
                command = "esorex sph_ird_instrument_flat"
                command+= " --ird.instrument_flat.badpix_lowtolerance=0.5"
                command+= " --ird.instrument_flat.badpix_uptolerance=1.5"
                command+= " --ird.instrument_flat.save_addprod=TRUE"
                command+= " --ird.instrument_flat.outfilename={}master_flat.fits".format(outpath_irdis_fits)
                command+= " --ird.instrument_flat.badpixfilename={}master_flat_bpmap.fits".format(outpath_irdis_fits)
                command+= " {}master_flat.sof".format(outpath_irdis_sof)
                os.system(command)
                
            ## manually merge DARK bp map and FLAT bp map. => DON'T! THE REDUCE RECIPE CONSIDERS BOTH
            dark_bpmap, header = vip_hci.fits.open_fits("{}master_badpixelmap.fits".format(outpath_irdis_fits), header=True)
            flat_bpmap = vip_hci.fits.open_fits("{}master_flat_bpmap.fits".format(outpath_irdis_fits))
            dark_bpmap[np.where(flat_bpmap)] = 1
            vip_hci.fits.write_fits("{}FINAL_badpixelmap.fits".format(outpath_irdis_fits), dark_bpmap, header=header)


        # SKY CUBE (optionally PCA SUBTRACTION)
        if 5 in to_do:                
            # OBJECT  
            sci_list_irdis = dico_lists['sci_list_irdis']
            n_sci =  len(sci_list_irdis)
            if n_sci>0:
                # bad pixel maps
                bp_map = vip_hci.fits.open_fits("{}FINAL_badpixelmap.fits".format(outpath_irdis_fits))
                # SKY PCA-SUBTR
                if isfile("{}master_sky_cube.fits".format(outpath_irdis_fits)) and pca_subtr:
                    
                    master_sky = vip_hci.fits.open_fits("{}master_sky_cube.fits".format(outpath_irdis_fits))
                    if npc > master_sky.shape[0]:
                        msg = "WARNING: input npc ({:.0f}) larger than number of sky frames, automatically changed to {:.0f}."
                        print(msg.format(npc,master_sky.shape[0]))
                        npc = master_sky.shape[0]
                    
                    if science_mode == 'CI':
                        n_s = 1
                    elif science_mode == 'DBI':
                        n_s = 2
                    star_coords_xy = np.zeros([n_sci,2,n_s])
                    for ii in range(n_sci):
                        hdulist_sci = fits.open(inpath+sci_list_irdis[ii], 
                                                ignore_missing_end=False,
                                                memmap=True)
                        sci_cube = hdulist_sci[0].data
                        xcuts = [0,1024,2048]
                        for s in range(n_s):
                            sci_cube_tmp = sci_cube[:,:,xcuts[s]:xcuts[s+1]]
                            bp_map_tmp = bp_map[:,xcuts[s]:xcuts[s+1]]
                            master_sci_sky_tmp = master_sky[:,:,xcuts[s]:xcuts[s+1]]
                            
                            #tmp = sci_cube_tmp.copy()
                            # look for smooth max location in med-sky subtraction
                            #if tmp.ndim == 3:
                            tmp = np.median(sci_cube_tmp,axis=0)
                            if ii == 0 and s == 0:
                                n_y_tmp, n_x_tmp = tmp.shape
                                cy_tmp, cx_tmp = vip_hci.var.frame_center(tmp)
                                mask_arr = np.zeros([n_sci,2,n_y_tmp, n_x_tmp])
                            tmp = tmp - np.median(master_sci_sky_tmp,axis=0) - np.median(tmp-np.median(master_sci_sky_tmp,axis=0))
                            
                            if mask_pca_sky_sub[2] > 0:
                                mask_tmp = vip_hci.var.create_ringed_spider_mask(tmp.shape, 
                                                     mask_pca_sky_sub[1], 
                                                     ann_in=mask_pca_sky_sub[0], 
                                                     sp_width=mask_pca_sky_sub[3],
                                                     sp_angle=mask_pca_sky_sub[4], 
                                                     nlegs=mask_pca_sky_sub[2])
                                coords_tmp = vip_hci.metrics.peak_coordinates(tmp, fwhm=4)
                                #print("peak coords: ", coords_tmp)
                                star_coords_xy[ii,1,s] = coords_tmp[0]
                                star_coords_xy[ii,0,s] = coords_tmp[1]
                                mask_arr[ii][s] = vip_hci.preproc.frame_shift(mask_tmp,
                                                                             int(star_coords_xy[ii,1,s]-cy_tmp),
                                                                             int(star_coords_xy[ii,0,s]-cx_tmp),
                                                                             border_mode='constant'
                                                                             )
                            else:
                                ## create template annular mask
                                mask_tmp = np.ones_like(tmp)
                                mask_circ1 = vip_hci.var.mask_circle(mask_tmp, mask_pca_sky_sub[0], fillwith=0, mode='in')
                                mask_circ2 = vip_hci.var.mask_circle(mask_tmp, mask_pca_sky_sub[1], fillwith=0, mode='out')
                            
                                coords_tmp = vip_hci.metrics.peak_coordinates(tmp, fwhm=4)
                                #print("peak coords: ", coords_tmp)
                                star_coords_xy[ii,1,s] = coords_tmp[0]
                                star_coords_xy[ii,0,s] = coords_tmp[1]
                                mask_tmp_shift1 = vip_hci.preproc.frame_shift(mask_circ1,
                                                                             int(star_coords_xy[ii,1,s]-cy_tmp),
                                                                             int(star_coords_xy[ii,0,s]-cx_tmp),
                                                                             border_mode='constant'
                                                                             )
                                mask_tmp_shift2 = vip_hci.preproc.frame_shift(mask_circ2,
                                                                             int(star_coords_xy[ii,1,s]-cy_tmp),
                                                                             int(star_coords_xy[ii,0,s]-cx_tmp),
                                                                             border_mode='constant'
                                                                             )
                                mask_arr[ii][s]= mask_tmp_shift1*mask_tmp_shift2
                            # remove bad pixels from mask
                            mask_arr[ii][s][np.where(bp_map_tmp)]=0
                            # make masks
        #                    mask_tmp = vip_hci.metrics.mask_source_centers(tmp, fwhm=4, 
        #                                                                   y=star_coords_xy[ii,1,:], 
        #                                                                   x=star_coords_xy[ii,0,:])
                            #vip_hci.metrics.mask_sources(mask_tmp, ap_rad=40)
    #                            med_sky_lvl = []
    #                            for i in range(master_sci_sky_tmp.shape[0]):
    #                                med_sky_lvl.append(master_psf_sky_tmp[i][np.where(mask_arr[ii][s])])
    #                            med_sky_lvl = np.median(med_sky_lvl)
    #                            med_psf_lvl = []
    #                            for i in range(psf_cube_tmp.shape[0]):
    #                                med_psf_lvl.append(psf_cube_tmp[i][np.where(mask_arr[ii][s])])
    #                            med_psf_lvl = np.median(med_psf_lvl)
    #                            # PCA-sky subtraction
    #                            master_psf_sky_tmp = master_psf_sky_tmp-med_sky_lvl+med_psf_lvl
                            sci_cube_tmp = vip_hci.preproc.cube_subtract_sky_pca(sci_cube_tmp, 
                                                                                 sky_cube=master_sci_sky_tmp, 
                                                                                 mask=mask_arr[ii][s], 
                                                                                 ref_cube=None, 
                                                                                 ncomp=npc)
                            sci_cube[:,:,xcuts[s]:xcuts[s+1]] = sci_cube_tmp
                        hdulist_sci[0].data = sci_cube
                        hdulist_sci.writeto(inpath+label_ss+sci_list_irdis[ii], output_verify='ignore', overwrite=True)                    
    #                    hdulist_sc = fits.open(inpath+sci_list_irdis[ii], 
    #                                           ignore_missing_end=False,
    #                                           memmap=True)
    #                    sci_cube = hdulist_sc[0].data
    #                    tmp = sci_cube.copy()
    #                    # look for smooth max location in med-sky subtraction
    #                    if tmp.ndim == 3:
    #                        tmp = np.median(tmp,axis=0)
    #                    if ii == 0:
    #                        n_y_tmp, n_x_tmp = tmp.shape
    #                        cy_tmp, cx_tmp = vip_hci.var.frame_center(tmp)
    #                        mask_arr = np.zeros([n_sci,n_y_tmp, n_x_tmp])
    #                    tmp = tmp - np.median(master_sky,axis=0)
    #                    xcuts = [0]
    #                    ## create template ringed spider mask
    #                    mask_tmp = vip_hci.var.create_ringed_spider_mask(tmp.shape, 
    #                                                                     mask_pca_sky_sub[1], 
    #                                                                     ann_in=mask_pca_sky_sub[0], 
    #                                                                     sp_width=mask_pca_sky_sub[3],
    #                                                                     sp_angle=mask_pca_sky_sub[4], 
    #                                                                     nlegs=mask_pca_sky_sub[2])
    #                    for s in range(n_s):
    #                        xcuts.append(int(((s+1)/n_s)*n_x_tmp))
    #                        coords_tmp = vip_hci.metrics.peak_coordinates(tmp[:,xcuts[s]:xcuts[s+1]], fwhm=4)
    #                        star_coords_xy[ii,1,s] = coords_tmp[0]
    #                        star_coords_xy[ii,0,s] = coords_tmp[1] + xcuts[s]
    #                        mask_tmp_shift = vip_hci.preproc.frame_shift(mask_tmp,
    #                                                                     int(star_coords_xy[ii,1,s]-cy_tmp),
    #                                                                     int(star_coords_xy[ii,0,s]-cx_tmp),
    #                                                                     border_mode='constant'
    #                                                                     )
    #                        mask_arr[ii][np.where(mask_tmp_shift)]=1
    #                    # remove bad pixels from mask
    #                    mask_arr[ii][np.where(bp_map)]=0
    #                        
    #                        
    #                    # make masks
    ##                    mask_tmp = vip_hci.metrics.mask_source_centers(tmp, fwhm=4, 
    ##                                                                   y=star_coords_xy[ii,1,:], 
    ##                                                                   x=star_coords_xy[ii,0,:])
    #                    #vip_hci.metrics.mask_sources(mask_tmp, ap_rad=40)
    #                    # PCA-sky subtraction
    #                    sci_cube = vip_hci.preproc.cube_subtract_sky_pca(sci_cube, 
    #                                                                     sky_cube=master_sky, 
    #                                                                     mask=mask_arr[ii], 
    #                                                                     ref_cube=None, 
    #                                                                     ncomp=npc)
    #                    hdulist_sc[0].data = sci_cube
    #                    hdulist_sc.writeto(inpath+label_ss+sci_list_irdis[ii], output_verify='ignore', overwrite=True)#, output_verify)
                    vip_hci.fits.write_fits("{}master_masks_for_PCA_sky.fits".format(outpath_irdis_fits), mask_arr)
                    # write fake dark
                    hdulist = fits.open("{}master_dark.fits".format(outpath_irdis_fits), 
                                        ignore_missing_end=False,
                                        memmap=True)
                    dark = hdulist[0].data
                    hdulist[0].data = np.zeros_like(dark)
                    hdulist.writeto("{}{}master_dark.fits".format(outpath_irdis_fits,label_fd), 
                                    output_verify='ignore', overwrite=True)
                        
                # ELSE: we compute the master sky with sph sky recipe, which will then be passed to the reduction recipe
                else:
                    if not isfile(outpath_irdis_sof+"master_bg.sof") or overwrite_sof:
                        ins_bg_list_irdis = dico_lists['ins_bg_list_irdis']
                        with open(outpath_irdis_sof+"master_bg.sof", 'w+') as f:
                            if len(sky_list_irdis)>0:
                                for ii in range(len(sky_list_irdis)):
                                    f.write(inpath+sky_list_irdis[ii]+'\t'+'IRD_SKY_BG_RAW\n')
                            ## IF THERE is no SKY AT ALL => take INS BG (automatic calibration, but can have up to 20% flux difference level - e.g. J1900-3645 K band dataset)   
                            elif len(ins_bg_list_irdis)>0:
                                for ii in range(len(ins_bg_list_irdis)):
                                    f.write(inpath+ins_bg_list_irdis[ii]+'\t'+'IRD_INS_BG_RAW\n')
                            else:
                                print("WARNING: NO SKY NOR INS BG - PROXY SKY SUBTR AFTER REDUCTION WILL BE PERFORMED")
                                #raise ValueError("There is no appropriate sky nor ins bg")
                            f.write("{}master_badpixelmap.fits".format(outpath_irdis_fits)+'\t'+'IRD_STATIC_BADPIXELMAP')
                    if len(sky_list_irdis)>0:
                        if not isfile(outpath_irdis_fits+"sky_bg_fit.fits") or overwrite_sof or overwrite_fits:
                            command = "esorex sph_ird_sky_bg"
                            command+= " --ird.sky_bg.save_addprod=TRUE"
                            command+= " --ird.sky_bg.outfilename={}sky_bg.fits".format(outpath_irdis_fits)                  
                    else:
                        if not isfile(outpath_irdis_fits+"ins_bg_fit.fits") or overwrite_sof or overwrite_fits:
                            command = "esorex sph_ird_ins_bg"
                            command+= " --ird.ins_bg.save_addprod=TRUE"
                            command+= " --ird.ins_bg.outfilename={}ins_bg.fits".format(outpath_irdis_fits)
                    command+= " {}master_bg.sof".format(outpath_irdis_sof)
                    os.system(command)
                
                
            # CENTER
            cen_list_irdis = dico_lists['cen_list_irdis']
            n_cen = len(cen_list_irdis)
            if n_cen>0:
                # SKY PCA-SUBTR
                if isfile("{}master_sky_cube.fits".format(outpath_irdis_fits)) and pca_subtr_cen:
                    bp_map = vip_hci.fits.open_fits("{}FINAL_badpixelmap.fits".format(outpath_irdis_fits))
                    master_sky = vip_hci.fits.open_fits("{}master_sky_cube.fits".format(outpath_irdis_fits))
                    
                    
                    if science_mode == 'CI':
                        n_s = 1
                    elif science_mode == 'DBI':
                        n_s = 2
                    star_coords_xy = np.zeros([n_cen,2,n_s])
                    for ii in range(n_cen):
                        hdulist_cen = fits.open(inpath+cen_list_irdis[ii], 
                                                ignore_missing_end=False,
                                                memmap=True)
                        cen_cube = hdulist_cen[0].data
                        xcuts = [0,1024,2048]
                        for s in range(n_s):
                            cen_cube_tmp = cen_cube[:,:,xcuts[s]:xcuts[s+1]]
                            bp_map_tmp = bp_map[:,xcuts[s]:xcuts[s+1]]
                            master_cen_sky_tmp = master_sky[:,:,xcuts[s]:xcuts[s+1]]
                            
                            #tmp = cen_cube_tmp.copy()
                            # look for smooth max location in med-sky subtraction
                            #if tmp.ndim == 3:
                            tmp = np.median(cen_cube_tmp,axis=0)
                            if ii == 0 and s == 0:
                                n_y_tmp, n_x_tmp = tmp.shape
                                cy_tmp, cx_tmp = vip_hci.var.frame_center(tmp)
                                mask_arr = np.zeros([n_cen,2,n_y_tmp, n_x_tmp])
                            tmp = tmp - np.median(master_cen_sky_tmp,axis=0) - np.median(tmp-np.median(master_cen_sky_tmp,axis=0))
    
                            if mask_pca_sky_sub[2] > 0:
                                mask_tmp = vip_hci.var.create_ringed_spider_mask(tmp.shape, 
                                                     mask_pca_sky_sub[1], 
                                                     ann_in=mask_pca_sky_sub[0], 
                                                     sp_width=mask_pca_sky_sub[3],
                                                     sp_angle=mask_pca_sky_sub[4], 
                                                     nlegs=mask_pca_sky_sub[2])
                                coords_tmp = vip_hci.metrics.peak_coordinates(tmp, fwhm=4)
                                #print("peak coords: ", coords_tmp)
                                star_coords_xy[ii,1,s] = coords_tmp[0]
                                star_coords_xy[ii,0,s] = coords_tmp[1]
                                mask_arr[ii][s] = vip_hci.preproc.frame_shift(mask_tmp,
                                                                             int(star_coords_xy[ii,1,s]-cy_tmp),
                                                                             int(star_coords_xy[ii,0,s]-cx_tmp),
                                                                             border_mode='constant'
                                                                             )
                            else:                        
                                ## create template annular mask
                                mask_tmp = np.ones_like(tmp)
                                mask_circ1 = vip_hci.var.mask_circle(mask_tmp, mask_pca_sky_sub[0], fillwith=0, mode='in')
                                mask_circ2 = vip_hci.var.mask_circle(mask_tmp, mask_pca_sky_sub[1], fillwith=0, mode='out')
                            
                                coords_tmp = vip_hci.metrics.peak_coordinates(tmp, fwhm=4)
                                #print("peak coords: ", coords_tmp)
                                star_coords_xy[ii,1,s] = coords_tmp[0]
                                star_coords_xy[ii,0,s] = coords_tmp[1]
                                mask_tmp_shift1 = vip_hci.preproc.frame_shift(mask_circ1,
                                                                             int(star_coords_xy[ii,1,s]-cy_tmp),
                                                                             int(star_coords_xy[ii,0,s]-cx_tmp),
                                                                             border_mode='constant'
                                                                             )
                                mask_tmp_shift2 = vip_hci.preproc.frame_shift(mask_circ2,
                                                                             int(star_coords_xy[ii,1,s]-cy_tmp),
                                                                             int(star_coords_xy[ii,0,s]-cx_tmp),
                                                                             border_mode='constant'
                                                                             )
                                mask_arr[ii][s]= mask_tmp_shift1*mask_tmp_shift2
                            # remove bad pixels from mask
                            mask_arr[ii][s][np.where(bp_map_tmp)]=0
                            # make masks
        #                    mask_tmp = vip_hci.metrics.mask_source_centers(tmp, fwhm=4, 
        #                                                                   y=star_coords_xy[ii,1,:], 
        #                                                                   x=star_coords_xy[ii,0,:])
                            #vip_hci.metrics.mask_sources(mask_tmp, ap_rad=40)
    #                            med_sky_lvl = []
    #                            for i in range(master_cen_sky_tmp.shape[0]):
    #                                med_sky_lvl.append(master_psf_sky_tmp[i][np.where(mask_arr[ii][s])])
    #                            med_sky_lvl = np.median(med_sky_lvl)
    #                            med_psf_lvl = []
    #                            for i in range(psf_cube_tmp.shape[0]):
    #                                med_psf_lvl.append(psf_cube_tmp[i][np.where(mask_arr[ii][s])])
    #                            med_psf_lvl = np.median(med_psf_lvl)
    #                            # PCA-sky subtraction
    #                            master_psf_sky_tmp = master_psf_sky_tmp-med_sky_lvl+med_psf_lvl
                            cen_cube_tmp = vip_hci.preproc.cube_subtract_sky_pca(cen_cube_tmp, 
                                                                                 sky_cube=master_cen_sky_tmp, 
                                                                                 mask=mask_arr[ii][s], 
                                                                                 ref_cube=None, 
                                                                                 ncomp=npc)
                            cen_cube[:,:,xcuts[s]:xcuts[s+1]] = cen_cube_tmp
                        hdulist_cen[0].data = cen_cube
                        hdulist_cen.writeto(inpath+label_ss+cen_list_irdis[ii], output_verify='ignore', overwrite=True)
                    vip_hci.fits.write_fits("{}master_masks_for_PCA_sky.fits".format(outpath_irdis_fits), mask_arr)
                    # write fake dark
                    hdulist = fits.open("{}master_dark.fits".format(outpath_irdis_fits), 
                                        ignore_missing_end=False,
                                        memmap=True)
                    dark = hdulist[0].data
                    hdulist[0].data = np.zeros_like(dark)
                    hdulist.writeto("{}{}master_dark.fits".format(outpath_irdis_fits,label_fd), 
                                    output_verify='ignore', overwrite=True)   
                    
            # PSF
            psf_list_irdis = dico_lists['psf_list_irdis']
            n_psf = len(psf_list_irdis)
            if n_psf>0:
                bp_map = vip_hci.fits.open_fits("{}FINAL_badpixelmap.fits".format(outpath_irdis_fits))
                # SKY PCA-SUBTR
                if isfile("{}master_sky_psf_cube.fits".format(outpath_irdis_fits)) and pca_subtr_psf:
                    
                    master_psf_sky = vip_hci.fits.open_fits("{}master_sky_psf_cube.fits".format(outpath_irdis_fits))
                    
                    
                    if science_mode == 'CI':
                        n_s = 1
                    elif science_mode == 'DBI':
                        n_s = 2
                    star_coords_xy = np.zeros([n_psf,2,n_s])
    #                for ii in range(n_psf):
    #                    hdulist_psf = fits.open(inpath+psf_list_irdis[ii], 
    #                                           ignore_missing_end=False,
    #                                           memmap=True)
    #                    psf_cube = hdulist_psf[0].data
    #                    tmp = psf_cube.copy()
    #                    # look for smooth max location in med-sky subtraction
    #                    if tmp.ndim == 3:
    #                        tmp = np.median(tmp,axis=0)
    #                    if ii == 0:
    #                        n_y_tmp, n_x_tmp = tmp.shape
    #                        mask_arr = np.ones([n_psf,n_y_tmp, n_x_tmp])
    #                    tmp = tmp - np.median(master_psf_sky,axis=0)
    #                    x_cut_ori = 0
    #                    for s in range(n_s):
    #                        x_cut = int(((s+1)/n_s)*n_x_tmp)
    #                        coords_tmp = vip_hci.metrics.peak_coordinates(tmp[:,x_cut_ori:x_cut], fwhm=4)
    #                        star_coords_xy[ii,1,s] = coords_tmp[0]
    #                        star_coords_xy[ii,0,s] = coords_tmp[1] + x_cut_ori
    #                        x_cut_ori = x_cut
    #                    # make masks
    #                    mask_tmp = vip_hci.metrics.mask_source_centers(tmp, fwhm=4, 
    #                                                                   y=star_coords_xy[ii,1,:], 
    #                                                                   x=star_coords_xy[ii,0,:])
    #                    mask_arr[ii] = vip_hci.metrics.mask_sources(mask_tmp, ap_rad=40)
    #                    # PCA-sky subtraction
    #                    psf_cube = vip_hci.preproc.cube_subtract_sky_pca(psf_cube, 
    #                                                                     sky_cube=master_psf_sky, 
    #                                                                     mask=mask_arr[ii], 
    #                                                                     ref_cube=None, 
    #                                                                     ncomp=1)
    #                    hdulist_psf[0].data = psf_cube
    #                    hdulist_psf.writeto(inpath+label_ss+psf_list_irdis[ii], output_verify='ignore', overwrite=True)#, output_verify)
    #                vip_hci.fits.write_fits("{}master_masks_for_PCA_sky_psf.fits".format(outpath_irdis_fits), mask_arr)
    
                    for ii in range(n_psf):
                        hdulist_psf = fits.open(inpath+psf_list_irdis[ii], 
                                                ignore_missing_end=False,
                                                memmap=True)
                        psf_cube = hdulist_psf[0].data
                        xcuts = [0,1024,2048]
                        for s in range(n_s):
                            psf_cube_tmp = psf_cube[:,:,xcuts[s]:xcuts[s+1]]
                            bp_map_tmp = bp_map[:,xcuts[s]:xcuts[s+1]]
                            master_psf_sky_tmp = master_psf_sky[:,:,xcuts[s]:xcuts[s+1]]
                            
                            #tmp = psf_cube_tmp.copy()
                            # look for smooth max location in med-sky subtraction
                            #if tmp.ndim == 3:
                            tmp = np.median(psf_cube_tmp,axis=0)
                            if ii == 0 and s == 0:
                                n_y_tmp, n_x_tmp = tmp.shape
                                cy_tmp, cx_tmp = vip_hci.var.frame_center(tmp)
                                mask_arr = np.zeros([n_psf,2,n_y_tmp, n_x_tmp])
                            tmp = tmp - np.median(master_psf_sky_tmp,axis=0) - np.median(tmp-np.median(master_psf_sky_tmp,axis=0))
                            
                            ## create template annular mask
                            mask_tmp = np.ones_like(tmp)
                            mask_circ1 = vip_hci.var.mask_circle(mask_tmp, mask_pca_sky_sub[0], fillwith=0, mode='in')
                            mask_circ2 = vip_hci.var.mask_circle(mask_tmp, mask_pca_sky_sub[1], fillwith=0, mode='out')
                        
                            coords_tmp = vip_hci.metrics.peak_coordinates(tmp, fwhm=4)
                            #print("peak coords: ", coords_tmp)
                            star_coords_xy[ii,1,s] = coords_tmp[0]
                            star_coords_xy[ii,0,s] = coords_tmp[1]
                            mask_tmp_shift1 = vip_hci.preproc.frame_shift(mask_circ1,
                                                                         int(star_coords_xy[ii,1,s]-cy_tmp),
                                                                         int(star_coords_xy[ii,0,s]-cx_tmp),
                                                                         border_mode='constant'
                                                                         )
                            mask_tmp_shift2 = vip_hci.preproc.frame_shift(mask_circ2,
                                                                         int(star_coords_xy[ii,1,s]-cy_tmp),
                                                                         int(star_coords_xy[ii,0,s]-cx_tmp),
                                                                         border_mode='constant'
                                                                         )
                            mask_arr[ii][s]= mask_tmp_shift1*mask_tmp_shift2
                            # remove bad pixels from mask
                            mask_arr[ii][s][np.where(bp_map_tmp)]=0
                            # make masks
        #                    mask_tmp = vip_hci.metrics.mask_source_centers(tmp, fwhm=4, 
        #                                                                   y=star_coords_xy[ii,1,:], 
        #                                                                   x=star_coords_xy[ii,0,:])
                            #vip_hci.metrics.mask_sources(mask_tmp, ap_rad=40)
                            med_sky_lvl = []
                            for i in range(master_psf_sky_tmp.shape[0]):
                                med_sky_lvl.append(master_psf_sky_tmp[i][np.where(mask_arr[ii][s])])
                            med_sky_lvl = np.median(med_sky_lvl)
                            med_psf_lvl = []
                            for i in range(psf_cube_tmp.shape[0]):
                                med_psf_lvl.append(psf_cube_tmp[i][np.where(mask_arr[ii][s])])
                            med_psf_lvl = np.median(med_psf_lvl)
                            # PCA-sky subtraction
                            master_psf_sky_tmp = master_psf_sky_tmp-med_sky_lvl+med_psf_lvl
                            psf_cube_tmp = vip_hci.preproc.cube_subtract_sky_pca(psf_cube_tmp, 
                                                                                 sky_cube=master_psf_sky_tmp, 
                                                                                 mask=mask_arr[ii][s], 
                                                                                 ref_cube=None, 
                                                                                 ncomp=npc_psf)
                            psf_cube[:,:,xcuts[s]:xcuts[s+1]] = psf_cube_tmp
                        hdulist_psf[0].data = psf_cube
                        hdulist_psf.writeto(inpath+label_ss+psf_list_irdis[ii], output_verify='ignore', overwrite=True)#, output_verify)
                    vip_hci.fits.write_fits("{}master_masks_for_PCA_sky_psf.fits".format(outpath_irdis_fits), mask_arr)
                    #pdb.set_trace()
    
                        
                # ELSE: we compute the master sky with sph sky recipe, which will then be passed to the reduction recipe
                else:
                    if not isfile(outpath_irdis_sof+"master_bg_psf.sof") or overwrite_sof:
                        psf_sky_list_irdis = dico_lists['psf_sky_list_irdis']
                        psf_ins_bg_list_irdis = dico_lists['psf_ins_bg_list_irdis']
                        with open(outpath_irdis_sof+"master_bg_psf.sof", 'w+') as f:
                            if len(psf_sky_list_irdis)>0:
                                for ii in range(len(sky_list_irdis)):
                                    f.write(inpath+sky_list_irdis[ii]+'\t'+'IRD_SKY_BG_RAW\n')
                            ## IF THERE is no SKY AT ALL => take INS BG (automatic calibration, but can have up to 20% flux difference level - e.g. J1900-3645 K band dataset)   
                            elif len(ins_bg_list_irdis)>0:
                                for ii in range(len(ins_bg_list_irdis)):
                                    f.write(inpath+ins_bg_list_irdis[ii]+'\t'+'IRD_INS_BG_RAW\n')
                            else:
                                print("WARNING: NO SKY NOR INS BG - PROXY SKY SUBTR AFTER REDUCTION WILL BE PERFORMED")
                                #raise ValueError("There is no appropriate sky nor ins bg")
                            f.write("{}master_badpixelmap.fits".format(outpath_irdis_fits)+'\t'+'IRD_STATIC_BADPIXELMAP')
                    pdb.set_trace()
                    if len(sky_list_irdis)>0:
                        if not isfile(outpath_irdis_fits+"psf_sky_bg.fits") or overwrite_sof or overwrite_fits:
                            command = "esorex sph_ird_sky_bg"
                            command+= " --ird.sky_bg.save_addprod=TRUE"
                            command+= " --ird.sky_bg.outfilename={}psf_sky_bg.fits".format(outpath_irdis_fits)                 
                    else:
                        if not isfile(outpath_irdis_fits+"psf_sky_bg.fits") or overwrite_sof or overwrite_fits:
                            command = "esorex sph_ird_ins_bg"
                            command+= " --ird.ins_bg.save_addprod=TRUE"
                            command+= " --ird.ins_bg.outfilename={}psf_sky_bg.fits".format(outpath_irdis_fits)
                    command+= " {}master_bg.sof".format(outpath_irdis_sof)
                    os.system(command)
        
            
        # REDUCE
        if 6 in to_do:
            if science_mode == 'DBI':
                lab_SCI = 'IRD_SCIENCE_DBI_RAW\n'
                lab_rec = 'science_dbi'
                lab_lr=["_"+filters[0],"_"+filters[1]]
            elif science_mode == 'CI':
                lab_SCI = 'IRD_SCIENCE_IMAGING_RAW\n'
                lab_rec = 'science_imaging'
                lab_lr = ["_CI_l_"+filters,"_CI_r_"+filters]
            else:
                raise ValueError("science_mode not recognized: should be DBI or CI.")
                
            # OBJECT  
            sci_list_irdis = dico_lists['sci_list_irdis']
            n_sci = len(sci_list_irdis)
            if n_sci>0:
    
                for ii in range(len(sci_list_irdis)):
                    if not isfile(outpath_irdis_sof+"OBJECT{:.0f}.sof".format(ii)) or overwrite_sof:
                        with open(outpath_irdis_sof+"OBJECT{:.0f}.sof".format(ii), 'w') as f:
                            f.write(inpath+label_ss+sci_list_irdis[ii]+'\t'+lab_SCI)
                            if pca_subtr:
                                f.write("{}{}master_dark.fits".format(outpath_irdis_fits,label_fd)+' \t'+'IRD_MASTER_DARK\n')
                            else:
                                if isfile("{}sky_bg.fits".format(outpath_irdis_fits)):
                                    f.write("{}sky_bg.fits".format(outpath_irdis_fits)+'\t'+'IRD_SKY_BG\n')
                                elif isfile("{}ins_bg.fits".format(outpath_irdis_fits)):
                                    f.write("{}ins_bg.fits".format(outpath_irdis_fits)+'\t'+'IRD_INS_BG\n')
                                else:
                                    f.write("{}master_dark.fits".format(outpath_irdis_fits)+' \t'+'IRD_MASTER_DARK\n')
                            f.write("{}master_flat.fits".format(outpath_irdis_fits)+'\t'+'IRD_FLAT_FIELD\n')
                            if isfile(inpath_filt_table+"sph_ird_filt_table.fits"):
                                f.write(inpath_filt_table+"sph_ird_filt_table.fits"+'\t'+'IRD_FILTER_TABLE\n')
                            f.write("{}FINAL_badpixelmap.fits".format(outpath_irdis_fits)+'\t'+'IRD_STATIC_BADPIXELMAP')
                            
                    if not isfile(outpath_irdis_fits+"science_dbi{:.0f}.fits".format(ii)) or overwrite_sof or overwrite_fits:
                        command = "esorex sph_ird_{}".format(lab_rec)
                        command+= " --ird.{}.outfilename={}science_dbi{:.0f}.fits".format(lab_rec,outpath_irdis_fits,ii)
                        command+= " --ird.{}.outfilename_left={}science_dbi{}{:.0f}.fits".format(lab_rec,outpath_irdis_fits,lab_lr[0],ii)
                        command+= " --ird.{}.outfilename_right={}science_dbi{}{:.0f}.fits".format(lab_rec,outpath_irdis_fits,lab_lr[1],ii)
    #                    if crop_sz_irdis>0 and crop_sz_irdis<1024:
    #                        command+= " --ird.{}.window_size={:.0f}".format(lab_rec,crop_sz_irdis)
                        command+= " {}OBJECT{:.0f}.sof".format(outpath_irdis_sof,ii)
                        os.system(command)
                    
            # CEN
            cen_list_irdis = dico_lists['cen_list_irdis']
            if len(cen_list_irdis)>0:
                for ii in range(len(cen_list_irdis)):
                    if not isfile(outpath_irdis_sof+"CEN{:.0f}.sof".format(ii)) or overwrite_sof:
                        with open(outpath_irdis_sof+"CEN{:.0f}.sof".format(ii), 'w') as f:
                            f.write(inpath+label_ss+cen_list_irdis[ii]+'\t'+lab_SCI)
                            if pca_subtr:
                                f.write("{}{}master_dark.fits".format(outpath_irdis_fits,label_fd)+' \t'+'IRD_MASTER_DARK\n')
                            else:
                                if isfile("{}sky_bg.fits".format(outpath_irdis_fits)):
                                    f.write("{}sky_bg.fits".format(outpath_irdis_fits)+'\t'+'IRD_SKY_BG\n')
                                elif isfile("{}ins_bg.fits".format(outpath_irdis_fits)):
                                    f.write("{}ins_bg.fits".format(outpath_irdis_fits)+'\t'+'IRD_INS_BG\n')
                                else:
                                    f.write("{}master_dark.fits".format(outpath_irdis_fits)+' \t'+'IRD_MASTER_DARK\n')
                            f.write("{}master_flat.fits".format(outpath_irdis_fits)+'\t'+'IRD_FLAT_FIELD\n')
                            if isfile(inpath_filt_table+"sph_ird_filt_table.fits"):
                                f.write(inpath_filt_table+"sph_ird_filt_table.fits"+'\t'+'IRD_FILTER_TABLE\n')
                            f.write("{}FINAL_badpixelmap.fits".format(outpath_irdis_fits)+'\t'+'IRD_STATIC_BADPIXELMAP')
                            
                    if not isfile(outpath_irdis_fits+"cen_dbi{:.0f}.fits".format(ii)) or overwrite_sof or overwrite_fits:
                        command = "esorex sph_ird_{}".format(lab_rec)
                        command+= " --ird.{}.outfilename={}cen_dbi{:.0f}.fits".format(lab_rec,outpath_irdis_fits,ii)
                        command+= " --ird.{}.outfilename_left={}cen_dbi{}{:.0f}.fits".format(lab_rec,outpath_irdis_fits,lab_lr[0],ii)
                        command+= " --ird.{}.outfilename_right={}cen_dbi{}{:.0f}.fits".format(lab_rec,outpath_irdis_fits,lab_lr[1],ii)
    #                    if crop_sz_irdis>0 and crop_sz_irdis<1024:
    #                        command+= " --ird.{}.window_size={:.0f}".format(lab_rec,crop_sz_irdis)
                        command+= " {}CEN{:.0f}.sof".format(outpath_irdis_sof,ii)
                        os.system(command)
                    
            # PSF
            psf_list_irdis = dico_lists['psf_list_irdis']
            if len(psf_list_irdis)>0:
                for ii in range(len(psf_list_irdis)):
                    if not isfile(outpath_irdis_sof+"PSF{:.0f}.sof".format(ii)) or overwrite_sof:  
                        with open(outpath_irdis_sof+"PSF{:.0f}.sof".format(ii), 'w') as f:
                            f.write(inpath+label_ss+psf_list_irdis[ii]+'\t'+lab_SCI)
                            if pca_subtr_psf:
                                f.write("{}{}master_dark.fits".format(outpath_irdis_fits,label_fd)+' \t'+'IRD_MASTER_DARK\n')
                            else:
                                if isfile("{}psf_sky_bg.fits".format(outpath_irdis_fits)):
                                    f.write("{}psf_sky_bg.fits".format(outpath_irdis_fits)+'\t'+'IRD_SKY_BG\n')
                                elif isfile("{}psf_ins_bg.fits".format(outpath_irdis_fits)):
                                    f.write("{}psf_ins_bg.fits".format(outpath_irdis_fits)+'\t'+'IRD_INS_BG\n')
                                else:
                                    f.write("{}master_dark.fits".format(outpath_irdis_fits)+'\t'+'IRD_MASTER_DARK\n')
                            f.write("{}master_flat.fits".format(outpath_irdis_fits)+'\t'+'IRD_FLAT_FIELD\n')
                            if isfile(inpath_filt_table+"sph_ird_filt_table.fits"):
                                f.write(inpath_filt_table+"sph_ird_filt_table.fits"+'\t'+'IRD_FILTER_TABLE\n')                            
                            f.write("{}FINAL_badpixelmap.fits".format(outpath_irdis_fits)+'\t'+'IRD_STATIC_BADPIXELMAP')
                            
                    if not isfile(outpath_irdis_fits+"psf_dbi{:.0f}.fits".format(ii)) or overwrite_sof or overwrite_fits:
                        command = "esorex sph_ird_{}".format(lab_rec)
                        command+= " --ird.{}.outfilename={}psf_dbi{:.0f}.fits".format(lab_rec,outpath_irdis_fits,ii)
                        command+= " --ird.{}.outfilename_left={}psf_dbi{}{:.0f}.fits".format(lab_rec,outpath_irdis_fits,lab_lr[0],ii)
                        command+= " --ird.{}.outfilename_right={}psf_dbi{}{:.0f}.fits".format(lab_rec,outpath_irdis_fits,lab_lr[1],ii)
    #                    if crop_sz_psf_irdis>0 and crop_sz_psf_irdis<1024:
    #                        command+= " --ird.{}.window_size={:.0f}".format(lab_rec,crop_sz_psf_irdis)
                        command+= " {}PSF{:.0f}.sof".format(outpath_irdis_sof,ii)
                        os.system(command)
                    
            curr_path = str(pathlib.Path().absolute())+'/'          
            # subtract residual sky level if required (FOR ALL: OBJECT, CENTER, PSF)   
            if not pca_subtr or not pca_subtr_psf:
                lab_ird = ["_left","_right"] # always for IRDIS
                file_list = os.listdir(curr_path)
                for rr in range(len(lab_lr)):
                    products_list = [x for x in file_list if (x.startswith(instr) and x.endswith(lab_ird[rr]+".fits"))]
                    for pp, prod in enumerate(products_list):
                        hdul = fits.open(curr_path+prod)
                        tmp = hdul[0].data
                        head_tmp = hdul[0].header
                        if head_tmp['HIERARCH ESO DET SEQ1 DIT']==dit_irdis and (not manual_sky_irdis and pca_subtr):
                            continue
                        elif head_tmp['HIERARCH ESO DET SEQ1 DIT']==dit_psf_irdis and (not manual_sky_irdis_psf and pca_subtr_psf):
                            continue
                        tmp_med = np.median(tmp,axis=0)
                        # estimate star coords in median frame
                        peak_y, peak_x = vip_hci.metrics.peak_coordinates(tmp_med, fwhm=4, 
                                                                          approx_peak=None, 
                                                                          search_box=None,
                                                                          channels_peak=False)                       
                        ny, nx = tmp_med.shape
                        rad = int(min(ny/2, nx/2))
                        edge = min(peak_y, peak_x, ny-peak_y, nx-peak_x)
                        cy, cx = vip_hci.var.frame_center(tmp_med)                        
                        for zz in range(tmp.shape[0]):
                            # if star too close from edge, estimate bkg in opp. quadrant
                            if edge < rad/4:
                                #quadrant?
                                qua = np.arctan2(peak_y-cy, peak_x-cx)
                                if qua < -np.pi/2:
                                    ap = (ny-int(rad/4),nx-int(rad/4))
                                elif qua < 0:
                                    ap = (ny-int(rad/4),int(rad/4))
                                elif qua < np.pi/2:
                                    ap = (int(rad/4),int(rad/4))
                                else:
                                    ap = (int(rad/4),nx-int(rad/4))                                    
                                aper = photutils.CircularAperture(ap, r=int(rad/8)) # small r because risk of aberrant values near edges
                                aper_mask = aper.to_mask(method='center')
                                aper_data = aper_mask.multiply(tmp[zz])
                                masked_data = aper_data[aper_mask.data > 0]
                                _, avg, _ = sigma_clipped_stats(masked_data)
                            
                            # else (ie. if star near center, estimate bkg flux in an annulus)
                            else:
                                r_in = int(min(0.5*rad,edge-25)) # 25px ~ 5-6*lambda/D in most cases
                                r_out = int(min(0.75*rad,edge-5)) # 5px ~ 1-1.2*lambda/D in most cases
                                ann_aper = photutils.CircularAnnulus((peak_x,peak_y), 
                                                                     r_in=r_in, 
                                                                     r_out=r_out)
                                ann_mask = ann_aper.to_mask(method='center')
                                ann_data = ann_mask.multiply(tmp[zz])
                                masked_data = ann_data[ann_mask.data > 0]
                                _, avg, _ = sigma_clipped_stats(masked_data)     
                                
                            tmp[zz] = tmp[zz] - avg
                        hdul[0].data = tmp                                            
                        hdul.writeto(prod, output_verify='ignore', overwrite=True)
                        #pdb.set_trace()

            # MOVE ALL FILES TO OUTPATH                        
            os.system("mv {}/*.fits {}.".format(curr_path,outpath_irdis_fits))
    
    # 10-19 IFS
    if do_ifs:
        # DARK labels
        label_fd = '' # label for fake master dark
        label_ds = '' # label for dark subtracted flats
        if indiv_fdark and len(dico_lists['dit_ifs_flat'])>1:
            label_ds = 'darksub_flats/'
            if not isdir(inpath+label_ds):
                os.makedirs(inpath+label_ds)
        if indiv_fdark:
            label_fd = 'fake_'
                
        if 10 in to_do or 15 in to_do or 17 in to_do:
            dark_list_ifs = dico_lists['dark_list_ifs']
            if len(dark_list_ifs)<1:
                raise ValueError("There should be at least one dark! Double-check archive?")
            else:
                ### always read DIT in header
                dark_cube, dark_head = vip_hci.fits.open_fits(inpath+dark_list_ifs[0], header=True)
            ## OBJECT
            #if not isfile(outpath_ifs_sof+"master_dark.sof") or overwrite_sof:
            dark_list_ifs = dico_lists['dark_list_ifs']
            nd = len(dark_list_ifs)
            nd_fr = dark_cube.shape[0]
            counter = 0
            nmd_fr = int(nd_fr*nd)
            master_dark_cube = np.zeros([nmd_fr,dark_cube.shape[1],dark_cube.shape[2]])
            with open(outpath_ifs_sof+"master_dark.sof", 'w+') as f:
                for ii in range(len(dark_list_ifs)):
                    dark_cube, dark_head = vip_hci.fits.open_fits(inpath+dark_list_ifs[ii], header=True)
                    f.write(inpath+dark_list_ifs[ii]+'\t'+'IFS_DARK_RAW\n')
                    master_dark_cube[counter:counter+dark_cube.shape[0]]
                    counter+=dark_cube.shape[0]
            vip_hci.fits.write_fits(outpath_ifs_fits+"master_dark_cube.fits", master_dark_cube[:counter])                    
            if not isfile(outpath_ifs_fits+"master_dark.fits") or overwrite_sof or overwrite_fits:
                command = "esorex sph_ifs_master_dark"
                command+= " --ifs.master_dark.sigma_clip=10.0"
                #command+= " --ifs.master_dark.save_addprod=TRUE"
                command+= " --ifs.master_dark.outfilename={}master_dark.fits".format(outpath_ifs_fits)
                command+= " --ifs.master_dark.badpixfilename={}master_badpixelmap.fits".format(outpath_ifs_fits)
                command+= " {}master_dark.sof".format(outpath_ifs_sof)
                os.system(command)
                
            ## FLAT DARKS
            ### subtract manually the MASTER DARK CREATED FOR EACH FLAT DIT!
            ### Save with label "_ds" in raw folder - add it to FLAT part
            if indiv_fdark and len(dico_lists['dit_ifs_flat'])>1:
                dit_ifs_flat_list = dico_lists['dit_ifs_flat']
                nfdits = len(dit_ifs_flat_list)
                fdark_list_ifs = dico_lists['flat_dark_list_ifs']
                if len(fdark_list_ifs)<1:
                    raise ValueError("There should be at least one flat dark! Double-check archive?")
                flat_list_ifs = dico_lists['flat_list_ifs']
                flat_list_ifs_det = dico_lists['flat_list_ifs_det']
                flat_list_ifs_det_BB = dico_lists['flat_list_ifs_det_BB']
                all_flat_lists_ifs = [flat_list_ifs,flat_list_ifs_det,flat_list_ifs_det_BB]
                nfd = len(fdark_list_ifs)
                fdit_list_nn = []
                hdulist_bp = fits.open("{}master_badpixelmap.fits".format(outpath_ifs_fits), 
                                       ignore_missing_end=False,
                                       memmap=True)
                bpmap = hdulist_bp[0].data #vip_hci.fits.open_fits("{}master_badpixelmap.fits".format(outpath_ifs_fits))
                for nn, fdit in enumerate(dit_ifs_flat_list):
                    ## first open an example dark
                    dark_cube = vip_hci.fits.open_fits(inpath+fdark_list_ifs[0], header=False)
                    nd_fr = dark_cube.shape[0]
                    counter = 0
                    nmd_fr = int(nd_fr*nfd)
                    master_dark_cube = np.zeros([nmd_fr,dark_cube.shape[1],dark_cube.shape[2]])
                    fdit_list_nn.append((nn,fdit))
                    # CREATE master DARK for each DIT of raw FLAT
                    #if not isfile(outpath_ifs_sof+"master_fdark{:.0f}.sof".format(nn)) or overwrite_sof:
                    with open(outpath_ifs_sof+"master_fdark{:.0f}.sof".format(nn), 'w+') as f:
                        for dd in range(nfd):
                            dark_cube, fdark_head = vip_hci.fits.open_fits(inpath+fdark_list_ifs[dd], header=True)
                            dit_fdark = fdark_head['HIERARCH ESO DET SEQ1 DIT']
                            if dit_fdark == fdit:
                                f.write(inpath+fdark_list_ifs[dd]+'\t'+'IFS_DARK_RAW\n')
                                master_dark_cube[counter:counter+dark_cube.shape[0]]=dark_cube
                                counter+=dark_cube.shape[0]
                    vip_hci.fits.write_fits(outpath_ifs_fits+"master_dark_cube{:.0f}.fits".format(nn),master_dark_cube[:counter])
                    if not isfile(outpath_ifs_fits+"master_dark{:.0f}.fits".format(nn)) or overwrite_sof or overwrite_fits:
                        command = "esorex sph_ifs_master_dark"
                        command+= " --ifs.master_dark.sigma_clip=10.0"
                        #command+= " --ifs.master_dark.save_addprod=TRUE"
                        command+= " --ifs.master_dark.outfilename={}master_dark{:.0f}.fits".format(outpath_ifs_fits,nn)
                        command+= " --ifs.master_dark.badpixfilename={}master_badpixelmap{:.0f}.fits".format(outpath_ifs_fits,nn)
                        command+= " {}master_fdark{:.0f}.sof".format(outpath_ifs_sof,nn)
                        os.system(command)                
                        # find raw flats with same dit and subtract master dark
                    
                    # SUBTRACT DARK
                    ## just subtract median
                    master_dark_tmp = vip_hci.fits.open_fits("{}master_dark{:.0f}.fits".format(outpath_ifs_fits,nn))
                    for nf in range(len(all_flat_lists_ifs)):
                        for ff, ff_name in enumerate(all_flat_lists_ifs[nf]):
                            hdulist = fits.open(inpath+ff_name, 
                                            ignore_missing_end=False,
                                            memmap=True)
                            flat = hdulist[0].data
                            ff_head = hdulist[0].header
                            #flat, ff_head = vip_hci.fits.open_fits(inpath+ff_name, header=True)
                            dit_flat = ff_head['HIERARCH ESO DET SEQ1 DIT']
                            if dit_flat == fdit:
                                hdulist[0].data = flat-master_dark_tmp
                                hdulist.writeto(inpath+label_ds+ff_name, output_verify='ignore', overwrite=True)#, output_verify)
                                #vip_hci.fits.write_fits(inpath+label_ds+ff_name,flat-master_dark_tmp)
                                
            # CREATE FAKE MASTER DARK BUT WITH TRUE BAD PIXEL MAP! 
            ## NOTE: BAD PIXEL MAP should always be the one obtained with the DARK with the longest DIT
            if indiv_fdark:
                lab='{:.0f}'.format(len(dit_ifs_flat_list)-1)
                os.system("rsync -va {}master_badpixelmap{:.0f}.fits {}master_badpixelmap.fits".format(outpath_ifs_fits,len(dit_ifs_flat_list)-1,outpath_ifs_fits))
                hdulist = fits.open("{}master_dark{}.fits".format(outpath_ifs_fits,'{:.0f}'.format(len(dit_ifs_flat_list)-1)), 
                                    ignore_missing_end=False,
                                    memmap=True)
                dark = hdulist[0].data
                hdulist[0].data = np.zeros_like(dark)
                hdulist.writeto("{}{}master_dark.fits".format(outpath_ifs_fits,label_fd), 
                                output_verify='ignore', overwrite=True)
            
        # GAINS
        if 11 in to_do:
            ## OBJECT
            if not isfile(outpath_ifs_sof+"master_gain.sof") or overwrite_sof:
                gain_list_ifs = dico_lists['gain_list_ifs']
                with open(outpath_ifs_sof+"master_gain.sof", 'w+') as f:
                    for ii in range(len(gain_list_ifs)):
                        f.write(inpath+gain_list_ifs[ii]+'\t'+'IFS_GAIN_RAW\n')
            if not isfile(outpath_ifs_fits+"master_gain.fits") or overwrite_sof or overwrite_fits:
                command = "esorex sph_ifs_gain"
                command+= " --ifs.gain.outfilename={}master_gain_map.fits".format(outpath_ifs_fits)
                command+= " --ifs.gain.nonlin_filename={}nonlin_map.fits".format(outpath_ifs_fits)
                command+= " --ifs.gain.nonlin_bpixname={}nonlin_badpixelmap.fits".format(outpath_ifs_fits)
                command+= " --ifs.gain.vacca=TRUE"
                command+= " {}master_gain.sof".format(outpath_ifs_sof)
                os.system(command)
                
        # MASTER DETECTOR FLAT (4 steps - see MANUAL)
        if 12 in to_do: # to check whether it should be set to True
            flat_list_ifs_det_BB = dico_lists['flat_list_ifs_det_BB']
            flat_list_ifs_det = dico_lists['flat_list_ifs_det']         
            lab_flat = ''     
               
            #0. xtalk corr them all if needed
            if xtalk_corr:
                for ii in range(len(flat_list_ifs_det_BB)):
                    hdul = fits.open(inpath+label_ds+flat_list_ifs_det_BB[ii])
                    cube = hdul[0].data
                    ## CROSS-TALK CORR
                    for j in range(cube.shape[0]):
                        cube[j] = sph_ifs_correct_spectral_xtalk(cube[j], boundary='fill', 
                                                                 fill_value=0)
                    hdul[0].data = cube
                    lab_flat = xtalkcorr_lab_IFS
                    hdul.writeto(inpath+xtalkcorr_lab_IFS+label_ds+flat_list_ifs_det_BB[ii], output_verify='ignore', overwrite=True)      
                    #pdb.set_trace()
                for ii in range(len(flat_list_ifs_det)):
                    hdul = fits.open(inpath+label_ds+flat_list_ifs_det[ii])
                    cube = hdul[0].data
                    ## CROSS-TALK CORR
                    for j in range(cube.shape[0]):
                        cube[j] = sph_ifs_correct_spectral_xtalk(cube[j], boundary='fill', 
                                                                 fill_value=0)
                    hdul[0].data = cube
                    lab_flat = xtalkcorr_lab_IFS
                    hdul.writeto(inpath+xtalkcorr_lab_IFS+label_ds+flat_list_ifs_det[ii], output_verify='ignore', overwrite=True)      
                    #pdb.set_trace()
            
            # 1. White preamp flat field (for stripe correction)
            with open(outpath_ifs_sof+"preamp.sof", 'w+') as f:
                #max_dit_flat = 0
                for ii in range(len(flat_list_ifs_det_BB)):
                    tmp, header = vip_hci.fits.open_fits(inpath+flat_list_ifs_det_BB[ii], header=True)
                    f.write(inpath+lab_flat+label_ds+flat_list_ifs_det_BB[ii]+'\t'+'IFS_DETECTOR_FLAT_FIELD_RAW\n')
#                        if float(header['EXPTIME'])>max_dit_flat:
#                            max_dit_flat = float(header['EXPTIME'])
#                    tmp, header = vip_hci.fits.open_fits("{}master_dark.fits".format(outpath_ifs_fits), header=True)
                if ('FLAT' in dark_ifs and not indiv_fdark) or indiv_fdark:
                    f.write("{}master_dark.fits".format(outpath_ifs_fits+label_fd)+'\t'+'IFS_MASTER_DARK\n')
                f.write("{}master_badpixelmap.fits".format(outpath_ifs_fits)+'\t'+'IFS_STATIC_BADPIXELMAP')
       
            if not isfile(outpath_ifs_fits+"preamp_l5.fits") or overwrite_sof or overwrite_fits:
                command = "esorex sph_ifs_master_detector_flat"
                command+= " --ifs.master_detector_flat.badpix_lowtolerance=0.2"
                command+= " --ifs.master_detector_flat.badpix_uptolerance=5."
                command+= " --ifs.master_detector_flat.save_addprod=TRUE"
                if flat_fit:# and len(flat_list_ifs_det_BB)>4: 
                    command+= " --ifs.master_detector_flat.robust_fit=TRUE"
                command+= " --ifs.master_detector_flat.outfilename={}tmp1.fits".format(outpath_ifs_fits)
                command+= " --ifs.master_detector_flat.lss_outfilename={}tmp2.fits".format(outpath_ifs_fits)
                command+= " --ifs.master_detector_flat.preamp_outfilename={}preamp.fits".format(outpath_ifs_fits)            
                command+= " --ifs.master_detector_flat.badpixfilename={}tmp3.fits".format(outpath_ifs_fits)
                command+= " {}preamp.sof".format(outpath_ifs_sof)
                os.system(command)
            
            #2 large scale coloured flat field (NB 1 to 4)
            if large_scale_flat:
                flat_list_ifs_det = dico_lists['flat_list_ifs_det']
                for kk in range(1,5):
                    with open(outpath_ifs_sof+"large_scale_flat_l{:.0f}.sof".format(kk), 'w+') as f:
                        run_rec = False
                        counter = 0
                        for ii in range(len(flat_list_ifs_det)):
                            tmp, header = vip_hci.fits.open_fits(inpath+flat_list_ifs_det[ii], header=True)
                            if 'HL{:.0f}'.format(kk) in header['HIERARCH ESO INS2 CAL']:
                                f.write(inpath+lab_flat+label_ds+flat_list_ifs_det[ii]+'\t'+'IFS_DETECTOR_FLAT_FIELD_RAW\n')
                                counter+=1
                                run_rec=True # needs to enter at least once to run recipe
                        if ('FLAT' in dark_ifs and not indiv_fdark) or indiv_fdark:
                            f.write("{}master_dark.fits".format(outpath_ifs_fits+label_fd)+'\t'+'IFS_MASTER_DARK\n')
                        f.write("{}master_badpixelmap.fits".format(outpath_ifs_fits)+'\t'+'IFS_STATIC_BADPIXELMAP')
                        f.write("{}preamp_l5.fits".format(outpath_ifs_fits)+'\t'+'IFS_PREAMP_FLAT\n')
             
                    if run_rec and (not isfile(outpath_ifs_fits+"large_scale_flat_l{:.0f}.fits".format(kk)) or overwrite_sof or overwrite_fits):
                        command = "esorex sph_ifs_master_detector_flat"
                        command+= " --ifs.master_detector_flat.badpix_lowtolerance=0.2"
                        command+= " --ifs.master_detector_flat.badpix_uptolerance=5."
                        command+= " --ifs.master_detector_flat.save_addprod=TRUE"
                        if flat_fit:# and counter>4: 
                            command+= " --ifs.master_detector_flat.robust_fit=TRUE"
                        command+= " --ifs.master_detector_flat.outfilename={}tmp1.fits".format(outpath_ifs_fits)
                        command+= " --ifs.master_detector_flat.lss_outfilename={}large_scale_flat.fits".format(outpath_ifs_fits)
                        command+= " --ifs.master_detector_flat.preamp_outfilename={}tmp2.fits".format(outpath_ifs_fits)            
                        command+= " --ifs.master_detector_flat.badpixfilename={}tmp3.fits".format(outpath_ifs_fits)
                        #command+= " --ifs.master_detector_flat.lambda={:.0f}".format(kk)
                        command+= " --ifs.master_detector_flat.smoothing_length={:.0f}".format(flat_smooth_length)
                        command+= " --ifs.master_detector_flat.smoothing_method={}".format(flat_smooth_method_idx)
                        command+= " {}large_scale_flat_l{:.0f}.sof".format(outpath_ifs_sof,kk)
                        os.system(command)
    
                #3. Large scale white flat field (BB) - NOTE SHOULDN'T BE USED LATER ON. EDIT: ACTUALLY YES THEY SHOULD in dr!
                if not isfile(outpath_ifs_sof+"large_scale_flat_BB.sof") or overwrite_sof:
                    flat_list_ifs_det_BB = dico_lists['flat_list_ifs_det_BB']
                    with open(outpath_ifs_sof+"large_scale_flat_BB.sof", 'w+') as f:
                        for ii in range(len(flat_list_ifs_det_BB)):
                            f.write(inpath+lab_flat+label_ds+flat_list_ifs_det_BB[ii]+'\t'+'IFS_DETECTOR_FLAT_FIELD_RAW\n')
                        if ('FLAT' in dark_ifs and not indiv_fdark) or indiv_fdark:
                            f.write("{}master_dark.fits".format(outpath_ifs_fits+label_fd)+'\t'+'IFS_MASTER_DARK\n')
                        f.write("{}master_badpixelmap.fits".format(outpath_ifs_fits)+'\t'+'IFS_STATIC_BADPIXELMAP')
                        f.write("{}preamp_l5.fits".format(outpath_ifs_fits)+'\t'+'IFS_PREAMP_FLAT\n')
             
                if not isfile(outpath_ifs_fits+"large_scale_flat_l5.fits") or overwrite_sof or overwrite_fits:
                    command = "esorex sph_ifs_master_detector_flat"
                    command+= " --ifs.master_detector_flat.badpix_lowtolerance=0.2"
                    command+= " --ifs.master_detector_flat.badpix_uptolerance=5."
                    command+= " --ifs.master_detector_flat.save_addprod=TRUE"
                    if flat_fit:# and len(flat_list_ifs_det_BB)>4: 
                        command+= " --ifs.master_detector_flat.robust_fit=TRUE"
                    command+= " --ifs.master_detector_flat.outfilename={}tmp1.fits".format(outpath_ifs_fits)                 
                    command+= " --ifs.master_detector_flat.lss_outfilename={}large_scale_flat.fits".format(outpath_ifs_fits)
                    command+= " --ifs.master_detector_flat.preamp_outfilename={}tmp2.fits".format(outpath_ifs_fits)            
                    command+= " --ifs.master_detector_flat.badpixfilename={}tmp3.fits".format(outpath_ifs_fits)
                    command+= " --ifs.master_detector_flat.smoothing_length=5."
                    command+= " {}large_scale_flat_BB.sof".format(outpath_ifs_sof)
                    os.system(command)
                
            if large_scale_flat != 'all':
                flat_list_ifs_det = dico_lists['flat_list_ifs_det']
                for kk in range(1,5):
                    with open(outpath_ifs_sof+"master_flat_det_l{:.0f}.sof".format(kk), 'w+') as f:
                        run_rec = False
                        counter = 0
                        for ii in range(len(flat_list_ifs_det)):
                            tmp, header = vip_hci.fits.open_fits(inpath+flat_list_ifs_det[ii], header=True)
                            if 'HL{:.0f}'.format(kk) in header['HIERARCH ESO INS2 CAL']:
                                f.write(inpath+lab_flat+label_ds+flat_list_ifs_det[ii]+'\t'+'IFS_DETECTOR_FLAT_FIELD_RAW\n')
                                counter+=1
                                run_rec=True
                        if ('FLAT' in dark_ifs and not indiv_fdark) or indiv_fdark:
                            f.write("{}master_dark.fits".format(outpath_ifs_fits+label_fd)+'\t'+'IFS_MASTER_DARK\n')
                        f.write("{}master_badpixelmap.fits".format(outpath_ifs_fits)+'\t'+'IFS_STATIC_BADPIXELMAP')
                        f.write("{}preamp_l5.fits".format(outpath_ifs_fits)+'\t'+'IFS_PREAMP_FLAT\n')
                        if large_scale_flat:
                            f.write("{}large_scale_flat_l{:.0f}.fits".format(outpath_ifs_fits,kk)+'\t'+'IFS_LARGE_SCALE_FLAT\n')
             
                    if run_rec and (not isfile(outpath_ifs_fits+"master_flat_det_l{:.0f}.fits".format(kk)) or overwrite_sof or overwrite_fits):
                        command = "esorex sph_ifs_master_detector_flat"
                        command+= " --ifs.master_detector_flat.badpix_lowtolerance=0.2"
                        command+= " --ifs.master_detector_flat.badpix_uptolerance=5."
                        #command+= " --ifs.master_detector_flat.save_addprod=TRUE"
                        #command+= " --ifs.master_detector_flat.outfilename={}tmp1.fits".format(outpath_ifs_fits)
                        command+= " --ifs.master_detector_flat.outfilename={}master_flat_det.fits".format(outpath_ifs_fits)
                        if flat_fit:# and counter>4: 
                            command+= " --ifs.master_detector_flat.robust_fit=TRUE"  
                        #command+= " --ifs.master_detector_flat.preamp_outfilename={}tmp2.fits".format(outpath_ifs_fits)            
                        #command+= " --ifs.master_detector_flat.badpixfilename={}tmp3.fits".format(outpath_ifs_fits)
                        #command+= " --ifs.master_detector_flat.lambda={:.0f}".format(kk)
                        command+= " {}master_flat_det_l{:.0f}.sof".format(outpath_ifs_sof,kk)
                        os.system(command)

            
                #5. Final detector flat field (BB) 
                if not isfile(outpath_ifs_sof+"master_flat_det_BB.sof") or overwrite_sof:
                    flat_list_ifs_det_BB = dico_lists['flat_list_ifs_det_BB']
                    with open(outpath_ifs_sof+"master_flat_det_BB.sof", 'w+') as f:
                        for ii in range(len(flat_list_ifs_det_BB)):
                            f.write(inpath+lab_flat+label_ds+flat_list_ifs_det_BB[ii]+'\t'+'IFS_DETECTOR_FLAT_FIELD_RAW\n')
                        if ('FLAT' in dark_ifs and not indiv_fdark) or indiv_fdark:
                            f.write("{}master_dark.fits".format(outpath_ifs_fits+label_fd)+'\t'+'IFS_MASTER_DARK\n')
                        f.write("{}master_badpixelmap.fits".format(outpath_ifs_fits)+'\t'+'IFS_STATIC_BADPIXELMAP')
                        f.write("{}preamp_l5.fits".format(outpath_ifs_fits)+'\t'+'IFS_PREAMP_FLAT\n')
                        if large_scale_flat:
                            f.write("{}large_scale_flat_l5.fits".format(outpath_ifs_fits)+'\t'+'IFS_LARGE_SCALE_FLAT\n')
             
                if not isfile(outpath_ifs_fits+"master_flat_det_l5.fits") or overwrite_sof or overwrite_fits:
                    command = "esorex sph_ifs_master_detector_flat"
                    #command+= " --ifs.master_detector_flat.badpix_lowtolerance=0.2"
                    #command+= " --ifs.master_detector_flat.badpix_uptolerance=5."
                    #command+= " --ifs.master_detector_flat.save_addprod=TRUE"
                    #command+= " --ifs.master_detector_flat.make_badpix=TRUE"
                    command+= " --ifs.master_detector_flat.outfilename={}master_flat_det.fits".format(outpath_ifs_fits)
                    if flat_fit:# and len(flat_list_ifs_det_BB)>4: 
                        command+= " --ifs.master_detector_flat.robust_fit=TRUE"
                    #command+= " --ifs.master_detector_flat.lss_outfilename={}tmp1.fits".format(outpath_ifs_fits)
                    #command+= " --ifs.master_detector_flat.preamp_outfilename={}tmp2.fits".format(outpath_ifs_fits)            
                    command+= " --ifs.master_detector_flat.badpixfilename={}tmp3.fits".format(outpath_ifs_fits)
                    command+= " {}master_flat_det_BB.sof".format(outpath_ifs_sof)
                    os.system(command)
        
            os.system("rm {}tmp*.fits".format(outpath_ifs_fits)) 
    
        # SPECTRA POSITIONS
        if 13 in to_do:
            dit_ifs_flat_list = dico_lists['dit_ifs_flat']
            nfdits = len(dit_ifs_flat_list)
            if not isfile(outpath_ifs_sof+"spectra_pos.sof") or overwrite_sof:
                specpos_IFS = dico_lists['specpos_IFS']
                with open(outpath_ifs_sof+"spectra_pos.sof", 'w+') as f:
                    for ii in range(len(specpos_IFS)):
                        f.write(inpath+specpos_IFS[ii]+'\t'+'IFS_SPECPOS_RAW\n')
                    if large_scale_flat == 'all':
                        f.write("{}large_scale_flat_l5.fits".format(outpath_ifs_fits)+'\t'+'IFS_INSTRUMENT_FLAT_FIELD\n')
                    else:
                        f.write("{}master_flat_det_l5.fits".format(outpath_ifs_fits)+'\t'+'IFS_INSTRUMENT_FLAT_FIELD\n')
                    if indiv_fdark:
                        tmp, header = vip_hci.fits.open_fits(inpath+specpos_IFS[ii], header=True)
                        for ff in range(nfdits):
                            if fdit_list_nn[ff][1] == header['EXPTIME']:
                                nn = fdit_list_nn[ff][0]
                                break
                            elif ff == nfdits-1:
                                print("no master dark with appropriate DIT was found for spec_pos")
                                pdb.set_trace()
                        f.write("{}master_dark{:.0f}.fits".format(outpath_ifs_fits,nn)+'\t'+'IFS_MASTER_DARK\n')
                    elif 'SPEC_POS' in dark_ifs:
                        f.write("{}master_dark.fits".format(outpath_ifs_fits)+'\t'+'IFS_MASTER_DARK\n')
                        
                    
            if not isfile(outpath_ifs_fits+"spectra_pos.fits") or overwrite_sof or overwrite_fits:
                command = "esorex sph_ifs_spectra_positions"
                if mode == "YJH":
                    command += " --ifs.spectra_positions.hmode=TRUE"
                else:
                    command += " --ifs.spectra_positions.hmode=FALSE"
                if not specpos_distort_corr:
                    command += " --ifs.spectra_positions.distortion=FALSE"
                if not specpos_nonlin_corr:
                    command += " --ifs.spectra_positions.correct_nonlin=FALSE"
                command+= " --ifs.spectra_positions.outfilename={}spectra_pos.fits".format(outpath_ifs_fits)
                command+= " {}spectra_pos.sof".format(outpath_ifs_sof)
                os.system(command)
                
        # TOTAL INSTRUMENT FLAT
        if 14 in to_do:
            if not isfile(outpath_ifs_fits+"master_flat_tot.fits") or overwrite_sof or overwrite_fits:
                if not isfile(outpath_ifs_sof+"master_flat_tot.sof") or overwrite_sof:
                    flat_list_ifs = dico_lists['flat_list_ifs']
                    with open(outpath_ifs_sof+"master_flat_tot.sof", 'w+') as f:
                        for ii in range(len(flat_list_ifs)):
                            f.write(inpath+label_ds+flat_list_ifs[ii]+'\t'+'IFS_FLAT_FIELD_RAW\n')
                        f.write("{}spectra_pos.fits".format(outpath_ifs_fits)+'\t'+'IFS_SPECPOS\n')
                        for jj in range(1,6):
                            if not large_scale_flat or (jj == 5 and large_scale_flat == 'some'):
                                lab_sof = "{}master_flat_det_l{:.0f}.fits".format(outpath_ifs_fits,jj) #v5c
                            else:
                                lab_sof = "{}large_scale_flat_l{:.0f}.fits".format(outpath_ifs_fits,jj) #v5c
                            if not isfile(lab_sof) and jj==4: # sometimes l4 is not taken (e.g. for IRDIFS YJH+H23 mode) 
                                continue
                            elif not isfile(lab_sof):
                                pdb.set_trace() # check what's happening
                            if jj == 5:
                                lab = "BB"
                            else:
                                lab = "{:.0f}".format(jj)                                             
                            f.write(lab_sof+'\t'+'IFS_MASTER_DFF_LONG{}\n'.format(lab))   
                        if ('FLAT' in dark_ifs and not indiv_fdark) or indiv_fdark:
                            f.write("{}master_dark.fits".format(outpath_ifs_fits+label_fd)+'\t'+'IFS_MASTER_DARK\n')
                        f.write("{}preamp_l5.fits".format(outpath_ifs_fits)+'\t'+'IFS_PREAMP_FLAT\n')
    
                if not isfile(outpath_ifs_fits+"master_flat_tot.fits") or overwrite_sof or overwrite_fits:
                    command = "esorex sph_ifs_instrument_flat"
                    if len(flat_list_ifs) > 1 and flat_fit:
                        command+= " --ifs.instrument_flat.make_badpix=TRUE"
                        command+= " --ifs.instrument_flat.badpixfilename={}master_badpixelmap_tot.fits".format(outpath_ifs_fits)
                        command+= " --ifs.instrument_flat.badpix_lowtolerance=0.2"
                        command+= " --ifs.instrument_flat.badpix_uptolerance=5.0"
                    command+= " --ifs.instrument_flat.iff_filename={}master_flat_tot.fits".format(outpath_ifs_fits)
                    if flat_fit: #and len(flat_list_ifs) > 4: 
                        command+= " --ifs.instrument_flat.robust_fit=TRUE"
                    else:
                        command+= " --ifs.instrument_flat.nofit=TRUE"                        
                    if illum_pattern_corr:
                        command+= " --ifs.instrument_flat.use_illumination=TRUE"
                    command+= " {}master_flat_tot.sof".format(outpath_ifs_sof)
                    os.system(command)

        # update below with final master flat tot filename    
        master_flatname = "master_flat_tot"
                    
        # WAVE CALIBRATION 
        if 15 in to_do:
            dit_ifs_flat_list = dico_lists['dit_ifs_flat']
            nfdits = len(dit_ifs_flat_list)

            if not isfile(outpath_ifs_sof+"wave_calib.sof") or overwrite_sof:
                wave_calib_list_ifs = dico_lists['wave_IFS'] # v2 
                hdulist_bp = fits.open("{}master_badpixelmap.fits".format(outpath_ifs_fits), 
                                       ignore_missing_end=False,
                                       memmap=True)
                
                bpmap = hdulist_bp[0].data #vip_hci.fits.open_fits("{}master_badpixelmap.fits".format(outpath_ifs_fits))

                for ii in range(len(wave_calib_list_ifs)):
                    hdul = fits.open(inpath+wave_calib_list_ifs[ii])
                    wc_head = hdul[0].header
                    cube = hdul[0].data
                    cube = np.array(cube, dtype=np.float)
                    ## MANUAL DARK SUBTRACTION
                    dit_wc = wc_head['HIERARCH ESO DET SEQ1 DIT']
                    for nn, fdit in enumerate(dit_ifs_flat_list):
                        ## subtract with PCA
                        master_dark_cube = vip_hci.fits.open_fits(outpath_ifs_fits+"master_dark_cube{:.0f}.fits".format(nn))                                
                        if fdit == dit_wc:
                            dark_tmp = np.median(master_dark_cube,axis=0)
                            for j in range(cube.shape[0]):
                                cube[j]-=dark_tmp
                            break
                    hdul[0].data = cube
                    lab_wc = skysub_lab_IFS
                    if not isdir(inpath+lab_wc):
                        os.makedirs(inpath+lab_wc)
                    hdul.writeto(inpath+skysub_lab_IFS+wave_calib_list_ifs[ii], output_verify='ignore', overwrite=True)
                    
                    
                        

                    cube = vip_hci.preproc.cube_fix_badpix_clump(cube, bpm_mask=bpmap, cy=None, cx=None, fwhm=3, 
                                                                 sig=6., protect_psf=False, verbose=False,
                                                                 half_res_y=False, max_nit=10, full_output=False)
                    lab_wc = bpcorr_lab_IFS
                    hdul[0].data = cube
                    if not isdir(inpath+lab_wc):
                        os.makedirs(inpath+lab_wc)
                    hdul.writeto(inpath+bpcorr_lab_IFS+'1_'+wave_calib_list_ifs[ii], output_verify='ignore', overwrite=True)
                    cube = vip_hci.preproc.cube_fix_badpix_clump(cube, bpm_mask=None, cy=None, cx=None, fwhm=3, 
                                                                 sig=10., protect_psf=False, verbose=False,
                                                                 half_res_y=False, max_nit=1, full_output=False)
                    lab_wc = bpcorr_lab_IFS
                    hdul[0].data = cube
                    if not isdir(inpath+lab_wc):
                        os.makedirs(inpath+lab_wc)
                    hdul.writeto(inpath+bpcorr_lab_IFS+wave_calib_list_ifs[ii], output_verify='ignore', overwrite=True)

                
                    if xtalk_corr: # too agressive
                        ## CROSS-TALK CORR
                        lab_wc = xtalkcorr_lab_IFS
                        if not isdir(inpath+lab_wc):
                            os.makedirs(inpath+lab_wc)
                        for j in range(cube.shape[0]):
                            cube[j] = sph_ifs_correct_spectral_xtalk(cube[j], boundary='fill', 
                                                                     fill_value=0)
                        hdul[0].data = cube
                        hdul.writeto(inpath+xtalkcorr_lab_IFS+wave_calib_list_ifs[ii], output_verify='ignore', overwrite=True)      
                    #pdb.set_trace()
                    
                with open(outpath_ifs_sof+"wave_calib.sof", 'w+') as f:
                    for ii in range(len(wave_calib_list_ifs)):
                        f.write(inpath+lab_wc+wave_calib_list_ifs[ii]+'\t'+'IFS_WAVECALIB_RAW\n')
                    f.write("{}spectra_pos.fits".format(outpath_ifs_fits)+'\t'+'IFS_SPECPOS\n')
                    f.write("{}{}.fits".format(outpath_ifs_fits,master_flatname)+'\t'+'IFS_INSTRUMENT_FLAT_FIELD\n')

         
            if not isfile(outpath_ifs_fits+"wave_calib.fits") or overwrite_sof or overwrite_fits:
                command = "esorex sph_ifs_wave_calib"
                command+= " --ifs.wave_calib.number_lines=0"
                if wc_win_sz != 4:
                    command+= " --ifs.wave_calib.fit_window_size={:.0f}".format(wc_win_sz)
                command+= "ifs.wave_calib.polyfit_order={:.0f}".format(poly_order_wc)
                command+= " --ifs.wave_calib.outfilename={}wave_calib.fits".format(outpath_ifs_fits)
                command+= " {}wave_calib.sof".format(outpath_ifs_sof)
                os.system(command)
                
    
        # IFU FLAT
        if 16 in to_do:
            if not isfile(outpath_ifs_fits+"master_flat_ifu.fits") or overwrite_sof or overwrite_fits:
                if not isfile(outpath_ifs_sof+"master_flat_ifu.sof") or overwrite_sof:
                    flat_list_ifs = dico_lists['flat_list_ifs_det'] # v1
                    flat_list_ifs = dico_lists['flat_list_ifs'] # v2 (as manual?)
                    #flat_list_ifs_BB = dico_lists['flat_list_ifs_det_BB']
                    with open(outpath_ifs_sof+"master_flat_ifu.sof", 'w+') as f:
                        for ii in range(len(flat_list_ifs)):
                            f.write(inpath+label_ds+flat_list_ifs[ii]+'\t'+'IFS_FLAT_FIELD_RAW\n')
                        # TEST: UNCOMMENTED 2 LINES BELOW
#                        for ii in range(len(flat_list_ifs_BB)):
#                            f.write(inpath+flat_list_ifs_BB[ii]+'\t'+'IFS_FLAT_FIELD_RAW\n')    
                        f.write("{}wave_calib.fits".format(outpath_ifs_fits)+'\t'+'IFS_WAVECALIB\n')
                        #f.write("{}spectra_pos.fits".format(outpath_ifs_fits)+'\t'+'IFS_SPECPOS\n')
                        f.write("{}preamp_l5.fits".format(outpath_ifs_fits)+'\t'+'IFS_PREAMP_FLAT\n')
                        if ('FLAT' in dark_ifs and not indiv_fdark) or indiv_fdark:
                            f.write("{}master_dark.fits".format(outpath_ifs_fits+label_fd)+'\t'+'IFS_MASTER_DARK\n')
                        for jj in range(1,6):
                            if not large_scale_flat or (jj == 5 and large_scale_flat == 'some'):
                                lab_sof = "{}master_flat_det_l{:.0f}.fits".format(outpath_ifs_fits,jj) #v5c
                            else:
                                lab_sof = "{}large_scale_flat_l{:.0f}.fits".format(outpath_ifs_fits,jj) #v5c
                            if not isfile(lab_sof) and jj==4: # sometimes l4 is not taken (e.g. for IRDIFS YJH+H23 mode) 
                                continue
                            elif not isfile(lab_sof):
                                pdb.set_trace() # check what's happening
                            if jj == 5:
                                lab = "BB"
                            else:
                                lab = "{:.0f}".format(jj)                          
#                            if large_scale_flat == 'all':
#                                lab_sof = "{}large_scale_flat_l{:.0f}.fits".format(outpath_ifs_fits,jj)
#                            else:
#                                lab_sof = "{}master_flat_det_l{:.0f}.fits".format(outpath_ifs_fits,jj)
#                            if jj == 5:
#                                lab = "BB"
#                            else:
#                                lab = "{:.0f}".format(jj)
                            f.write(lab_sof+'\t'+'IFS_MASTER_DFF_LONG{}\n'.format(lab))
         
                if not isfile(outpath_ifs_fits+"master_flat_ifu.fits") or overwrite_sof or overwrite_fits:
                    command = "esorex sph_ifs_instrument_flat"
                    if len(flat_list_ifs) > 1 and flat_fit:
                        command+= " --ifs.instrument_flat.make_badpix=TRUE"
                        command+= " --ifs.instrument_flat.badpixfilename={}master_badpixelmap_tot.fits".format(outpath_ifs_fits)
                        command+= " --ifs.instrument_flat.badpix_lowtolerance=0.2"
                        command+= " --ifs.instrument_flat.badpix_uptolerance=5.0"
                    command+= " --ifs.instrument_flat.ifu_filename={}master_flat_ifu.fits".format(outpath_ifs_fits)
                    if flat_fit: #and len(flat_list_ifs) > 4:      
                        command+= " --ifs.instrument_flat.robust_fit=TRUE"
                    else:
                        command+= " --ifs.instrument_flat.nofit=TRUE"
                    if illum_pattern_corr:
                        command+= " --ifs.instrument_flat.use_illumination=TRUE"
                    command+= " {}master_flat_ifu.sof".format(outpath_ifs_sof)
                    os.system(command)
                    
            
        # PRODUCE SKY CUBES
        if 17 in to_do and sky:
            ## OBJ
            sky_list_ifs = dico_lists['sky_list_ifs']
            if -1 in good_sky_list:
                tmp = vip_hci.fits.open_fits(inpath+sky_list_ifs[0])
                vip_hci.fits.write_fits("{}master_sky.fits".format(outpath_ifs_fits), tmp[0]) # just take the first (closest difference in time to that of consecutive SCIENCE cubes - reproduce best the remanence effect)
            elif 'all' in good_sky_list:
                counter=0
                nsky = len(sky_list_ifs)
                for gg in range(nsky):
                    tmp = vip_hci.fits.open_fits(inpath+sky_list_ifs[gg])
                    if counter == 0:
                        master_sky = np.zeros([nsky*tmp.shape[0],tmp.shape[1],tmp.shape[2]])
                    master_sky[counter:counter+tmp.shape[0]] = tmp
                    counter+=tmp.shape[0]
                master_sky = np.median(master_sky,axis=0)
                vip_hci.fits.write_fits("{}master_sky.fits".format(outpath_ifs_fits), master_sky) 
            else:
                counter=0
                nsky = len(good_sky_list)
                for gg in good_sky_list:
                    tmp = vip_hci.fits.open_fits(inpath+sky_list_ifs[gg])
                    if counter == 0:
                        master_sky = np.zeros([nsky*tmp.shape[0],tmp.shape[1],tmp.shape[2]])
                    master_sky[counter:counter+tmp.shape[0]] = tmp
                    counter+=tmp.shape[0]
                master_sky = np.median(master_sky,axis=0)
                vip_hci.fits.write_fits("{}master_sky.fits".format(outpath_ifs_fits), master_sky) 
    
            ## PSF
            psf_sky_list_ifs = dico_lists['psf_sky_list_ifs']
            if len(psf_sky_list_ifs) < 1:
                psf_sky_list_ifs = dico_lists['psf_ins_bg_list_ifs']
            if len(psf_sky_list_ifs) < 1:
                master_psf_sky = master_sky.copy() #assume that bkg and DARK current are negligible compared to bias
                vip_hci.fits.write_fits("{}dit_psf_sky.fits".format(outpath_ifs_fits), np.array([dit_ifs]))
            else:
                vip_hci.fits.write_fits("{}dit_psf_sky.fits".format(outpath_ifs_fits), np.array([dit_psf_ifs]))
                if -1 in good_psf_sky_list:
                    tmp = vip_hci.fits.open_fits(inpath+sky_list_ifs[0])
                    vip_hci.fits.write_fits("{}master_psf_sky.fits".format(outpath_ifs_fits), tmp[0]) # just take the first (closest difference in time to that of consecutive SCIENCE cubes - reproduce best the remanence effect)
                elif 'all' in good_psf_sky_list:
                    counter=0
                    nsky = len(psf_sky_list_ifs)
                    for gg in range(nsky):
                        tmp = vip_hci.fits.open_fits(inpath+psf_sky_list_ifs[gg])
                        if counter == 0:
                            master_psf_sky = np.zeros([nsky*tmp.shape[0],tmp.shape[1],tmp.shape[2]])
                        master_psf_sky[counter:counter+tmp.shape[0]] = tmp
                        counter+=tmp.shape[0]
                    master_psf_sky = np.median(master_psf_sky,axis=0)
                    vip_hci.fits.write_fits("{}master_psf_sky.fits".format(outpath_ifs_fits), master_psf_sky) # just take the first (closest difference in time to that of consecutive SCIENCE cubes - reproduce best the remanence effect)
                else:
                    counter=0
                    nsky = len(good_psf_sky_list)
                    for gg in good_psf_sky_list:
                        tmp = vip_hci.fits.open_fits(inpath+psf_sky_list_ifs[gg])
                        if counter == 0:
                            master_psf_sky = np.zeros([nsky*tmp.shape[0],tmp.shape[1],tmp.shape[2]])
                        master_psf_sky[counter:counter+tmp.shape[0]] = tmp
                        counter+=tmp.shape[0]
                    master_psf_sky = np.median(master_psf_sky,axis=0)
            vip_hci.fits.write_fits("{}master_psf_sky.fits".format(outpath_ifs_fits), master_psf_sky) # just take the first (closest difference in time to that of consecutive SCIENCE cubes - reproduce best the remanence effect)
    
        
        # REDUCE OBJECT
        lab_distort="" # no distortion correction possible for ifs in pipeline
        if 18 in to_do:
            # MANUAL SKY SUBTRACTION BEF REDUCTION
            sci_list_ifs = dico_lists['sci_list_ifs']
            hdulist_bp = fits.open("{}master_badpixelmap.fits".format(outpath_ifs_fits), 
                   ignore_missing_end=False,
                   memmap=True)
        
            bpmap = hdulist_bp[0].data #vip_hci.fits.open_fits("{}master_badpixelmap.fits".format(outpath_ifs_fits))
            for ii in range(len(sci_list_ifs)):
                hdul = fits.open(inpath+sci_list_ifs[ii])
                cube = hdul[0].data
                cube = np.array(cube, dtype=np.float)
                lab_bp=''
                
                if sky:
                    #tmp, head = vip_hci.fits.open_fits(inpath+sci_list_ifs[ii], header=True)
                    tmp_tmp = vip_hci.fits.open_fits("{}master_sky.fits".format(outpath_ifs_fits))
                    for zz in range(cube.shape[0]):
                        cube[zz] = cube[zz]-tmp_tmp
                    lab_bp = 'skycorr_'
                    lab_sci = skysub_lab_IFS
                    hdul[0].data = cube
                    hdul.writeto(inpath+skysub_lab_IFS+lab_bp+sci_list_ifs[ii], output_verify='ignore', overwrite=True)

                cube = vip_hci.preproc.cube_fix_badpix_clump(cube, bpm_mask=bpmap, cy=None, cx=None, fwhm=3, 
                                                             sig=6., protect_psf=False, verbose=False,
                                                             half_res_y=False, max_nit=10, full_output=False)
                cube = vip_hci.preproc.cube_fix_badpix_clump(cube, bpm_mask=None, cy=None, cx=None, fwhm=3, 
                                                             sig=10., protect_psf=False, verbose=False,
                                                             half_res_y=False, max_nit=1, full_output=False)
                hdul[0].data = cube
                lab_sci = bpcorr_lab_IFS
                if not isdir(inpath+lab_sci):
                    os.makedirs(inpath+lab_sci)
                hdul.writeto(inpath+bpcorr_lab_IFS+lab_bp+sci_list_ifs[ii], output_verify='ignore', overwrite=True) 
                #pdb.set_trace()
                
                
                if xtalk_corr: # too agressive
                    ## CROSS-TALK CORR
                    for j in range(cube.shape[0]):
                        cube[j] = sph_ifs_correct_spectral_xtalk(cube[j], boundary='fill', 
                                                                 fill_value=0)
                    hdul[0].data = cube
                    lab_sci = xtalkcorr_lab_IFS
                    hdul.writeto(inpath+xtalkcorr_lab_IFS+lab_bp+sci_list_ifs[ii], output_verify='ignore', overwrite=True)      
                    #pdb.set_trace()
                    
                if not isfile(outpath_ifs_sof+"OBJECT{}{:.0f}.sof".format(lab_distort,ii)) or overwrite_sof:
                    with open(outpath_ifs_sof+"OBJECT{}{:.0f}.sof".format(lab_distort,ii), 'w') as f:
                        f.write(inpath+lab_sci+lab_bp+sci_list_ifs[ii]+'\t'+'IFS_SCIENCE_DR_RAW\n')
                        for jj in range(1,6):
                            if not large_scale_flat or (jj == 5 and large_scale_flat == 'some'):
                                lab_sof = "{}master_flat_det_l{:.0f}.fits".format(outpath_ifs_fits,jj) #v5c      
                            else:
                                lab_sof = "{}large_scale_flat_l{:.0f}.fits".format(outpath_ifs_fits,jj) #v5c  
                            if not isfile(lab_sof) and jj==4: # sometimes l4 is not taken (e.g. for IRDIFS YJH+H23 mode) 
                                continue
                            elif not isfile(lab_sof):
                                pdb.set_trace() # check what's happening                                
                            if jj == 5:
                                lab = "BB"
                            else:
                                lab = "{:.0f}".format(jj)
                            f.write(lab_sof+'\t'+'IFS_MASTER_DFF_LONG{}\n'.format(lab))            
                        f.write("{}preamp_l5.fits".format(outpath_ifs_fits)+'\t'+'IFS_PREAMP_FLAT\n')
                        f.write("{}master_flat_ifu.fits".format(outpath_ifs_fits)+'\t'+'IFS_IFU_FLAT_FIELD\n')
#                        if cal_bkg:
#                            if usefit:
#                                if isfile("{}master_sky_bg.fits".format(outpath_ifs_fits)):
#                                    f.write("{}master_sky_bg_fit{:.0f}.fits".format(outpath_ifs_fits,fitorder_ifs)+'\t'+'IFS_CAL_BACKGROUND\n')
#                                elif isfile("{}master_cal_bg_fit{:.0f}.fits".format(outpath_ifs_fits,fitorder_ifs)):
#                                    f.write("{}master_cal_bg_fit{:.0f}.fits".format(outpath_ifs_fits,fitorder_ifs)+'\t'+'IFS_CAL_BACKGROUND\n')
#                                elif ('OBJ' in dark_ifs and not sky):
#                                    f.write("{}master_dark.fits".format(outpath_ifs_fits)+' \t'+'IFS_MASTER_DARK\n')
#                            else:
#                                if isfile("{}master_sky_bg.fits".format(outpath_ifs_fits)):
#                                    f.write("{}master_sky_bg.fits".format(outpath_ifs_fits)+'\t'+'IFS_CAL_BACKGROUND\n')
#                                elif isfile("{}master_cal_bg.fits".format(outpath_ifs_fits)):
#                                    f.write("{}master_cal_bg.fits".format(outpath_ifs_fits)+'\t'+'IFS_CAL_BACKGROUND\n')
#                                elif ('OBJ' in dark_ifs and not sky):
#                                    f.write("{}master_dark.fits".format(outpath_ifs_fits)+' \t'+'IFS_MASTER_DARK\n')
#                        f.write("{}master_badpixelmap.fits".format(outpath_ifs_fits)+'\t'+'IFS_STATIC_BADPIXELMAP\n')
                        f.write("{}master_dark.fits".format(outpath_ifs_fits+label_fd)+' \t'+'IFS_MASTER_DARK\n')
                        f.write("{}wave_calib.fits".format(outpath_ifs_fits)+'\t'+'IFS_WAVECALIB\n')
                
                if not isfile(outpath_ifs_fits+"ifs{}_{:.0f}.fits".format(lab_distort,ii)) or overwrite_sof or overwrite_fits:
                    command = "esorex sph_ifs_science_dr"
                    command+= " --ifs.science_dr.outfilename={}tmp{}_{:.0f}.fits".format(outpath_ifs_fits,lab_distort,ii)                
                    command+= " --ifs.science_dr.use_adi=0"
                    command+= " --ifs.science_dr.spec_deconv=FALSE" # should not do SDI because not centered !!!
                    command+= " --ifs.science_dr.reflambda=1.65" # should not matter since SDI is not done - but just in case
                    if xtalk_corr:
                        command+= " --ifs.science_dr.xtalkco.apply=TRUE"                 
                    if illum_pattern_corr:
                        command+= " --ifs.science_dr.use_illumination=TRUE"
                    command+= " {}OBJECT{}{:.0f}.sof".format(outpath_ifs_sof,lab_distort,ii)
                    os.system(command)
                    
                if manual_sky: # subtract residual sky level
                    curr_path = pathlib.Path().absolute()
                    file_list = os.listdir(curr_path)
                    products_list = [x for x in file_list if (x.startswith(lab_bp+instr) and x.endswith(".fits"))]
                    for pp, prod in enumerate(products_list):
                        hdul = fits.open(prod)
                        tmp = hdul[0].data
                        for zz in range(tmp.shape[0]):
                            fluxes = np.zeros(len(corner_coords))
                            for aa, ap in enumerate(corner_coords):
                                aper = photutils.CircularAperture(ap, r=msky_ap)
                                flux_tmp = photutils.aperture_photometry(tmp[zz], aper, method='exact')
                                fluxes[aa] = np.array(flux_tmp['aperture_sum'])
                            avg = np.median(fluxes)/(np.pi*msky_ap**2)
                            tmp[zz] = tmp[zz] - avg
                        hdul[0].data = tmp                                            
                        hdul.writeto(prod, output_verify='ignore', overwrite=True)
                        
                    
                os.system("mv {} {}.".format(lab_bp+instr+'*.fits',outpath_ifs_calib))
                os.system("rm {}{}".format(outpath_ifs_fits,"tmp*.fits"))

        # REDUCE CEN
        if 19 in to_do:
            cen_list_ifs = dico_lists['cen_list_ifs']
            hdulist_bp = fits.open("{}master_badpixelmap.fits".format(outpath_ifs_fits), 
                   ignore_missing_end=False,
                   memmap=True)
        
            bpmap = hdulist_bp[0].data #vip_hci.fits.open_fits("{}master_badpixelmap.fits".format(outpath_ifs_fits))
            for ii in range(len(cen_list_ifs)):
                hdul = fits.open(inpath+cen_list_ifs[ii])
                cube = hdul[0].data
                cube = np.array(cube, dtype=np.float)
                lab_bp=''
                
                if sky:
                    #tmp, head = vip_hci.fits.open_fits(inpath+sci_list_ifs[ii], header=True)
                    tmp_tmp = vip_hci.fits.open_fits("{}master_sky.fits".format(outpath_ifs_fits))
                    for zz in range(cube.shape[0]):
                        cube[zz] = cube[zz]-tmp_tmp
                    lab_bp = 'skycorr_cen_'
                    lab_sci = skysub_lab_IFS
                    hdul[0].data = cube
                    hdul.writeto(inpath+skysub_lab_IFS+lab_bp+cen_list_ifs[ii], output_verify='ignore', overwrite=True)    
# ADAPT BELOW IF NECESSARY
#                if bpix:
#                    tmp = vip_hci.preproc.cube_fix_badpix_clump(tmp, bpm_mask=None, cy=None, cx=None, fwhm=4., 
#                          sig=4., protect_psf=True, verbose=True, 
#                          half_res_y=False, min_thr=None, max_nit=15, 
#                          full_output=False)
#                    lab_bp = 'bpcorr_'
#                    fits.writeto(inpath+'skysub_lab_IFS'+lab_bp+sci_list_ifs[ii], tmp, head, output_verify='ignore', overwrite=True)

                
                cube = vip_hci.preproc.cube_fix_badpix_clump(cube, bpm_mask=bpmap, cy=None, cx=None, fwhm=3, 
                                                             sig=6., protect_psf=False, verbose=False,
                                                             half_res_y=False, max_nit=10, full_output=False)
                cube = vip_hci.preproc.cube_fix_badpix_clump(cube, bpm_mask=None, cy=None, cx=None, fwhm=3, 
                                                             sig=10., protect_psf=False, verbose=False,
                                                             half_res_y=False, max_nit=1, full_output=False)
                hdul[0].data = cube
                lab_sci = bpcorr_lab_IFS
                if not isdir(inpath+lab_sci):
                    os.makedirs(inpath+lab_sci)
                hdul.writeto(inpath+bpcorr_lab_IFS+lab_bp+cen_list_ifs[ii], output_verify='ignore', overwrite=True) 
                #pdb.set_trace()
                
                
                if xtalk_corr: # too agressive
                    ## CROSS-TALK CORR
                    for j in range(cube.shape[0]):
                        cube[j] = sph_ifs_correct_spectral_xtalk(cube[j], boundary='fill', 
                                                                 fill_value=0)
                    hdul[0].data = cube
                    lab_sci = xtalkcorr_lab_IFS
                    hdul.writeto(inpath+xtalkcorr_lab_IFS+lab_bp+cen_list_ifs[ii], output_verify='ignore', overwrite=True)      
                    #pdb.set_trace()
                        
                
                    
                if not isfile(outpath_ifs_sof+"CEN{}{:.0f}.sof".format(lab_distort,ii)) or overwrite_sof:
                    with open(outpath_ifs_sof+"CEN{}{:.0f}.sof".format(lab_distort,ii), 'w') as f:
                        f.write(inpath+lab_sci+lab_bp+cen_list_ifs[ii]+'\t'+'IFS_SCIENCE_DR_RAW\n')
                        for jj in range(1,6):
                            if not large_scale_flat or (jj == 5 and large_scale_flat == 'some'):
                                lab_sof = "{}master_flat_det_l{:.0f}.fits".format(outpath_ifs_fits,jj) #v5c
                            else:
                                lab_sof = "{}large_scale_flat_l{:.0f}.fits".format(outpath_ifs_fits,jj) #v5c
                            if not isfile(lab_sof) and jj==4: # sometimes l4 is not taken (e.g. for IRDIFS YJH+H23 mode) 
                                continue
                            elif not isfile(lab_sof):
                                pdb.set_trace() # check what's happening        
                            if jj == 5:
                                lab = "BB"
                            else:
                                lab = "{:.0f}".format(jj)
                            f.write(lab_sof+'\t'+'IFS_MASTER_DFF_LONG{}\n'.format(lab))            
                        f.write("{}preamp_l5.fits".format(outpath_ifs_fits)+'\t'+'IFS_PREAMP_FLAT\n')
                        f.write("{}master_flat_ifu.fits".format(outpath_ifs_fits)+'\t'+'IFS_IFU_FLAT_FIELD\n')  
#                        if cal_bkg:
#                            if usefit:
#                                if isfile("{}master_sky_bg_fit{:.0f}.fits".format(outpath_ifs_fits,fitorder_ifs)):
#                                    f.write("{}master_sky_bg_fit{:.0f}.fits".format(outpath_ifs_fits,fitorder_ifs)+'\t'+'IFS_CAL_BACKGROUND\n')
#                                elif isfile("{}master_cal_bg_fit{:.0f}.fits".format(outpath_ifs_fits,fitorder_ifs)):
#                                    f.write("{}master_cal_bg_fit{:.0f}.fits".format(outpath_ifs_fits,fitorder_ifs)+'\t'+'IFS_CAL_BACKGROUND\n')
#                                elif ('CEN' in dark_ifs and not sky):
#                                    f.write("{}master_dark.fits".format(outpath_ifs_fits)+' \t'+'IFS_MASTER_DARK\n')
#                            else:
#                                if isfile("{}master_sky_bg.fits".format(outpath_ifs_fits)):
#                                    f.write("{}master_sky_bg.fits".format(outpath_ifs_fits)+'\t'+'IFS_CAL_BACKGROUND\n')
#                                elif isfile("{}master_cal_bg.fits".format(outpath_ifs_fits)):
#                                    f.write("{}master_cal_bg.fits".format(outpath_ifs_fits)+'\t'+'IFS_CAL_BACKGROUND\n')
#                                elif ('CEN' in dark_ifs and not sky):
#                                    f.write("{}master_dark.fits".format(outpath_ifs_fits)+' \t'+'IFS_MASTER_DARK\n')
                        f.write("{}master_dark.fits".format(outpath_ifs_fits+label_fd)+' \t'+'IFS_MASTER_DARK\n')
                        f.write("{}wave_calib.fits".format(outpath_ifs_fits)+'\t'+'IFS_WAVECALIB\n')
                
                if not isfile(outpath_ifs_fits+"ifs_cen{}_{:.0f}.fits".format(lab_distort,ii)) or overwrite_sof or overwrite_fits:
                    command = "esorex sph_ifs_science_dr"
                    command+= " --ifs.science_dr.outfilename={}tmp_cen{}_{:.0f}.fits".format(outpath_ifs_fits,lab_distort,ii)                
                    command+= " --ifs.science_dr.use_adi=0"
                    command+= " --ifs.science_dr.spec_deconv=FALSE" # should not do SDI because not centered !!!
                    command+= " --ifs.science_dr.reflambda=1.65" # should not matter since SDI is not done - but just in case
                    if xtalk_corr:
                        command+= " --ifs.science_dr.xtalkco.apply=TRUE"
                    if illum_pattern_corr:
                        command+= " --ifs.science_dr.use_illumination=TRUE"
                    command+= " {}CEN{}{:.0f}.sof".format(outpath_ifs_sof,lab_distort,ii)
                    os.system(command)
    
                if manual_sky: # subtract residual sky level
                    curr_path = pathlib.Path().absolute()
                    file_list = os.listdir(curr_path)
                    products_list = [x for x in file_list if (x.startswith(lab_bp+instr) and x.endswith(".fits"))]
                    for pp, prod in enumerate(products_list):
                        hdul = fits.open(prod)
                        tmp = hdul[0].data
                        for zz in range(tmp.shape[0]):
                            fluxes = np.zeros(len(corner_coords))
                            for aa, ap in enumerate(corner_coords):
                                aper = photutils.CircularAperture(ap, r=msky_ap)
                                flux_tmp = photutils.aperture_photometry(tmp[zz], aper, method='exact')
                                fluxes[aa] = np.array(flux_tmp['aperture_sum'])
                            avg = np.median(fluxes)/(np.pi*msky_ap**2)
                            tmp[zz] = tmp[zz] - avg
                        hdul[0].data = tmp                                            
                        hdul.writeto(prod, output_verify='ignore', overwrite=True)
                    
                os.system("mv {} {}.".format(lab_bp+instr+'*.fits',outpath_ifs_calib))
                os.system("rm {}{}".format(outpath_ifs_fits,"tmp*.fits"))
           
           
        # REDUCE PSF
        if 20 in to_do:
            psf_list_ifs = dico_lists['psf_list_ifs']
            hdulist_bp = fits.open("{}master_badpixelmap.fits".format(outpath_ifs_fits), 
                   ignore_missing_end=False,
                   memmap=True)
        
            bpmap = hdulist_bp[0].data #vip_hci.fits.open_fits("{}master_badpixelmap.fits".format(outpath_ifs_fits))
            for ii in range(len(psf_list_ifs)):
                hdul = fits.open(inpath+psf_list_ifs[ii])
                cube = hdul[0].data
                cube = np.array(cube, dtype=np.float)
                lab_bp=''
                
                if sky:
                    #tmp, head = vip_hci.fits.open_fits(inpath+sci_list_ifs[ii], header=True)
                    tmp_tmp = vip_hci.fits.open_fits("{}master_psf_sky.fits".format(outpath_ifs_fits))
                    for zz in range(cube.shape[0]):
                        cube[zz] = cube[zz]-tmp_tmp
                    lab_bp = 'skycorr_psf_'
                    lab_sci = skysub_lab_IFS
                    hdul[0].data = cube
                    hdul.writeto(inpath+skysub_lab_IFS+lab_bp+psf_list_ifs[ii], output_verify='ignore', overwrite=True)    
# ADAPT BELOW IF NECESSARY
#                if bpix:
#                    tmp = vip_hci.preproc.cube_fix_badpix_clump(tmp, bpm_mask=None, cy=None, cx=None, fwhm=4., 
#                          sig=4., protect_psf=True, verbose=True, 
#                          half_res_y=False, min_thr=None, max_nit=15, 
#                          full_output=False)
#                    lab_bp = 'bpcorr_'
#                    fits.writeto(inpath+'skysub_lab_IFS'+lab_bp+sci_list_ifs[ii], tmp, head, output_verify='ignore', overwrite=True)
                
                
                cube = vip_hci.preproc.cube_fix_badpix_clump(cube, bpm_mask=bpmap, cy=None, cx=None, fwhm=3, 
                                                             sig=6., protect_psf=False, verbose=False,
                                                             half_res_y=False, max_nit=10, full_output=False)
                cube = vip_hci.preproc.cube_fix_badpix_clump(cube, bpm_mask=None, cy=None, cx=None, fwhm=3, 
                                                             sig=10., protect_psf=False, verbose=False,
                                                             half_res_y=False, max_nit=1, full_output=False)
                hdul[0].data = cube
                lab_sci = bpcorr_lab_IFS
                if not isdir(inpath+lab_sci):
                    os.makedirs(inpath+lab_sci)
                hdul.writeto(inpath+bpcorr_lab_IFS+lab_bp+psf_list_ifs[ii], output_verify='ignore', overwrite=True) 
                #pdb.set_trace()
                
                
                if xtalk_corr: # too agressive
                    ## CROSS-TALK CORR
                    for j in range(cube.shape[0]):
                        cube[j] = sph_ifs_correct_spectral_xtalk(cube[j], boundary='fill', 
                                                                 fill_value=0)
                    hdul[0].data = cube
                    lab_sci = xtalkcorr_lab_IFS
                    hdul.writeto(inpath+xtalkcorr_lab_IFS+lab_bp+psf_list_ifs[ii], output_verify='ignore', overwrite=True)      
                    #pdb.set_trace()
                        
                    
                if not isfile(outpath_ifs_sof+"PSF{}{:.0f}.sof".format(lab_distort,ii)) or overwrite_sof:
                    with open(outpath_ifs_sof+"PSF{}{:.0f}.sof".format(lab_distort,ii), 'w') as f:
                        f.write(inpath+skysub_lab_IFS+lab_bp+psf_list_ifs[ii]+'\t'+'IFS_SCIENCE_DR_RAW\n')
                        for jj in range(1,6):
                            if not large_scale_flat or (jj == 5 and large_scale_flat == 'some'):
                                lab_sof = "{}master_flat_det_l{:.0f}.fits".format(outpath_ifs_fits,jj) #v5c
                            else:
                                lab_sof = "{}large_scale_flat_l{:.0f}.fits".format(outpath_ifs_fits,jj) #v5c
                            if not isfile(lab_sof) and jj==4: # sometimes l4 is not taken (e.g. for IRDIFS YJH+H23 mode) 
                                continue
                            elif not isfile(lab_sof):
                                pdb.set_trace() # check what's happening        
                            if jj == 5:
                                lab = "BB"
                            else:
                                lab = "{:.0f}".format(jj)
                            f.write(lab_sof+'\t'+'IFS_MASTER_DFF_LONG{}\n'.format(lab))            
                        f.write("{}preamp_l5.fits".format(outpath_ifs_fits)+'\t'+'IFS_PREAMP_FLAT\n')
                        f.write("{}master_flat_ifu.fits".format(outpath_ifs_fits)+'\t'+'IFS_IFU_FLAT_FIELD\n')
                        if 'PSF' in dark_ifs:
                            f.write("{}master_dark.fits".format(outpath_ifs_fits)+' \t'+'IFS_MASTER_DARK\n')
                        else:
                            f.write("{}master_dark.fits".format(outpath_ifs_fits+label_fd)+' \t'+'IFS_MASTER_DARK\n')
                        f.write("{}wave_calib.fits".format(outpath_ifs_fits)+'\t'+'IFS_WAVECALIB\n')
                
                if not isfile(outpath_ifs_fits+"ifs_psf{}_{:.0f}.fits".format(lab_distort,ii)) or overwrite_sof or overwrite_fits:
                    command = "esorex sph_ifs_science_dr"
                    command+= " --ifs.science_dr.outfilename={}tmp_psf{}_{:.0f}.fits".format(outpath_ifs_fits,lab_distort,ii)                
                    command+= " --ifs.science_dr.use_adi=0"
                    command+= " --ifs.science_dr.spec_deconv=FALSE" # should not do SDI because not centered !!!
                    command+= " --ifs.science_dr.reflambda=1.65" # should not matter since SDI is not done - but just in case
                    if xtalk_corr:
                        command+= " --ifs.science_dr.xtalkco.apply=TRUE"
                    if illum_pattern_corr:
                        command+= " --ifs.science_dr.use_illumination=TRUE"
                    command+= " {}PSF{}{:.0f}.sof".format(outpath_ifs_sof,lab_distort,ii)
                    os.system(command)
    
                if manual_sky: # subtract residual sky level
                    curr_path = pathlib.Path().absolute()
                    file_list = os.listdir(curr_path)
                    products_list = [x for x in file_list if (x.startswith(lab_bp+instr) and x.endswith(".fits"))]
                    for pp, prod in enumerate(products_list):
                        hdul = fits.open(prod)
                        tmp = hdul[0].data
                        for zz in range(tmp.shape[0]):
                            fluxes = np.zeros(len(corner_coords_psf))
                            for aa, ap in enumerate(corner_coords_psf):
                                aper = photutils.CircularAperture(ap, r=msky_ap_psf)
                                flux_tmp = photutils.aperture_photometry(tmp[zz], aper, method='exact')
                                fluxes[aa] = np.array(flux_tmp['aperture_sum'])
                            avg = np.median(fluxes)/(np.pi*msky_ap_psf**2)
                            tmp[zz] = tmp[zz] - avg
                        hdul[0].data = tmp                                            
                        hdul.writeto(prod, output_verify='ignore', overwrite=True)   
                    
                os.system("mv {} {}.".format(lab_bp+instr+'*.fits',outpath_ifs_calib))
                os.system("rm {}{}".format(outpath_ifs_fits,"tmp*.fits"))
    return None