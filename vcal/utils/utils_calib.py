#! /usr/bin/env python

"""
Utility routines for calibration and data sorting.
"""

__author__ = 'V. Christiaens, Iain Hammond'
__all__ = ['make_lists',
           'sph_ifs_correct_spectral_xtalk']

from os import listdir
from os.path import isfile, join

import numpy as np
from astropy.convolution.convolve import convolve

from vip_hci.fits import open_header

instr = 'SPHER' # instrument name in file name


# create sof files for esorex, for both IFS and IRDIS data
def make_lists(inpath, outpath_filenames, dit_ifs=None, dit_irdis=None,
               dit_psf_ifs=None, dit_psf_irdis=None, dit_cen_ifs=None,
               dit_cen_irdis=None, filt1=None, filt2=None, readonly=False):
    """
    Create lists of each type of data files for a given dataset.
       
    Parameters:
    ***********
    outpath_filenames: str
        Path where the text files are written (to double check each list 
    contains desired files)
    
    dit_ifs, dit_irdis, dit_psf_irdis, dit_psf_irdis: floats, opt
        Depending on the dataset, detector integration time of the IFS or IRDIS
    data for the OBJECT and the PSF frames.
    
    readonly: bool, opt
        If the text files were already written, and the user is satsfied with 
    their content, can be set to True to build the dictionary directly from
    the content of the files, instead of parsing the headers of all files 
    (slow).
    
    Returns:
    ********
    dico_lists: python dictionary
        Dictionary with lists of each kind of observation & calibration files.
       """
    def _check_mode(ifs_mode_list):
        if len(ifs_mode_list)==0:
            ifs_mode_list.append(header["HIERARCH ESO INS2 COMB IFS"][-3:])
        elif header["HIERARCH ESO INS2 COMB IFS"][-3:] != ifs_mode_list[0]:
            msg = "There should only be one IFS observation mode for all non-FLAT and non-DARK fits files in the folder: both {} and {} detected"
            raise ValueError(msg.format(header["HIERARCH ESO INS2 COMB IFS"][-3:],ifs_mode_list[0]))
        return ifs_mode_list

    if dit_cen_irdis is not None:
        if isinstance(dit_cen_irdis, (float, int)):
            dit_cen_irdis = [dit_cen_irdis]
        elif not isinstance(dit_cen_irdis,list):
            raise TypeError("dit_cen_irdis can only be float, int or list")
    if dit_cen_ifs is not None:
        if isinstance(dit_cen_ifs, (float, int)):
            dit_cen_ifs = [dit_cen_ifs]
        elif not isinstance(dit_cen_ifs,list):
            raise TypeError("dit_cen_irdis can only be float, int or list")

    dico_files={}
    dit_ifs_flat = []
    dit_ifs_flat_dark = [] # used to only pick closest darks in time to the flats
    dit_irdis_flat = []
    dit_irdis_flat_dark = [] # used to only pick closest darks in time to the flats
    dit_ifs_distort = []
    dit_irdis_distort = []
    filt1_ifs_distort = []
    filt1_irdis_distort = []
    filt2_ifs_distort = []
    filt2_irdis_distort = []

    if dit_ifs is not None and dit_irdis is not None:
        mode = 'IRDIFS'
    elif dit_irdis is not None:
        mode = 'IRDIS'
    elif dit_ifs is not None:
        mode = 'IFS'

    if not readonly:

        file_list = [f for f in listdir(inpath) if isfile(join(inpath, f))] # all files in inpaths
        fits_list = []
        calib_list = []
        ifs_mode = []
    #if dit_ifs is not None:
        sci_list_ifs = []
        sky_list_ifs = []
        center_list_ifs = []
        sci_list_mjd_ifs = []
        sky_list_mjd_ifs = []
        center_list_mjd_ifs = []
        flat_list_ifs = []
        flat_list_ifs_det = []
        flat_list_ifs_det_BB = []
        flat_dark_list_ifs = []
        mjdobs_fdark_list_ifs = []
        dark_list_ifs = []
        ins_bg_list_ifs = []
        spec_pos_list_ifs = []
        wave_list_ifs = []
    #if dit_psf_ifs is not None:
        psf_list_ifs = []
        psf_sky_list_ifs = []
        psf_ins_bg_list_ifs = []
        psf_list_mjd_ifs = []
    #if dit_cen_ifs is not None:
        cen_list_ifs = []
        cen_sky_list_ifs = []
        cen_ins_bg_list_ifs = []
        cen_list_mjd_ifs = []

    #if dit_irdis is not None:
        sci_list_irdis = []
        sky_list_irdis = []
        psf_list_irdis = []
        center_list_irdis = []
        sci_list_mjd_irdis = []
        sky_list_mjd_irdis = []
        psf_list_mjd_irdis = []
        center_list_mjd_irdis = []
        flat_list_irdis = []
        flat_dark_list_irdis = []
        mjdobs_fdark_list_irdis = []
        dark_list_irdis = []
        ins_bg_list_irdis = []
    #if dit_psf_irdis is not None:
        psf_list_irdis = []
        psf_sky_list_irdis = []
        psf_ins_bg_list_irdis = []
        psf_list_mjd_irdis = []
    #if dit_cen_irdis is not None:
        cen_list_irdis = []
        cen_sky_list_irdis = []
        cen_ins_bg_list_irdis = []
        cen_list_mjd_irdis = []

        ao_files = []
        calib_IFS = []
        calib_IRDIS = []
        distort_IFS = []
        distort_ins_bg_IFS = []
        distort_IRDIS = []
        distort_ins_bg_IRDIS = []

        # GAINS
        gain_list_IRDIS = []
        gain_list_IFS = []

        for fname in file_list: # all fits files
            if fname.startswith(instr) and fname.endswith('.fits'):
                # Note: conditions can be refined if deemed appropriate.
                header = open_header(inpath+fname)
                if not "ESO DPR TYPE" in header.keys() and not "HIERARCH ESO DPR TYPE" in header.keys():
                    continue
                fits_list.append(fname)
                if header['HIERARCH ESO DPR TYPE'] == 'OBJECT,AO':
                    ao_files.append(fname)
                    continue
                if dit_ifs is not None:
                    if header['HIERARCH ESO DET SEQ1 DIT'] == dit_ifs and header['HIERARCH ESO DET NAME'] == 'IFS' and header['HIERARCH ESO DPR TYPE'] == 'OBJECT' and header['HIERARCH ESO INS1 FILT NAME'] == filt1 and header['HIERARCH ESO INS1 OPTI2 NAME'] == filt2:
                        sci_list_ifs.append(fname)
                        sci_list_mjd_ifs.append(header['MJD-OBS'])
                        ifs_mode = _check_mode(ifs_mode)
                    elif header['HIERARCH ESO DET SEQ1 DIT'] == dit_ifs and header['HIERARCH ESO DET NAME'] == 'IFS' and header['HIERARCH ESO DPR TYPE'] == 'SKY':
                        sky_list_ifs.append(fname)
                        sky_list_mjd_ifs.append(header['MJD-OBS'])
                        ifs_mode = _check_mode(ifs_mode)
                    elif header['HIERARCH ESO DET NAME'] == 'IFS' and header['HIERARCH ESO DPR TYPE'] == 'OBJECT,CENTRE':
                        center_list_ifs.append(fname)
                        center_list_mjd_ifs.append(header['MJD-OBS'])
                        ifs_mode = _check_mode(ifs_mode)
                    elif header['HIERARCH ESO DET NAME'] == 'IFS' and header['HIERARCH ESO DPR TYPE'] == 'DARK,BACKGROUND' and header['HIERARCH ESO INS1 FILT NAME'] == filt1 and header['HIERARCH ESO INS1 OPTI2 NAME'] == filt2:
                        ins_bg_list_ifs.append(fname)
                    elif header['HIERARCH ESO DET NAME'] == 'IFS' and header['HIERARCH ESO DPR TYPE'] == 'DARK':
                        dark_list_ifs.append(fname)
                    elif header['HIERARCH ESO DET NAME'] == 'IFS' and header['HIERARCH ESO DPR TYPE'] == 'FLAT,LAMP':
                        # six cases to deal with: BB (CAL_BB_2_[mode]), 1.02µm (CAL_NB1_1_[mode]),
                        # 1.23µm (CAL_NB2_1_[mode]), 1.3µm (CAL_NB3_1_[mode]), OBS_[mode]
                        # and 1.55µm (CAL_NB4_1_[mode]) if YJH
                        if "CAL_BB_2_" in header["HIERARCH ESO INS2 COMB IFS"]:  # white, two files
                            flat_list_ifs_det_BB.append(fname)
                        elif "CAL_NB" in header["HIERARCH ESO INS2 COMB IFS"]:  # narrow bands, two files for each laser
                            flat_list_ifs_det.append(fname)
                            # add list for each laser? then check after ...
                        elif 'IFSFLAT' in header['HIERARCH ESO OCS DET1 IMGNAME']:  # ifu flat, one file
                            flat_list_ifs.append(fname)
                        if float(header['HIERARCH ESO DET SEQ1 DIT']) not in dit_ifs_flat:
                            dit_ifs_flat.append(float(header['HIERARCH ESO DET SEQ1 DIT']))

                    elif header['HIERARCH ESO DET NAME'] == 'IFS' and header['HIERARCH ESO DPR TYPE'] == 'LAMP,DISTORT' and len(distort_IFS)==0:
                        distort_IFS.append(fname)
                        dit_ifs_distort.append(float(header['HIERARCH ESO DET SEQ1 DIT']))
                        filt1_ifs_distort.append(float(header['HIERARCH ESO DET SEQ1 DIT']))
                        filt2_ifs_distort.append(float(header['HIERARCH ESO DET SEQ1 DIT']))
                    elif header['HIERARCH ESO DET NAME'] == 'IFS' and header['HIERARCH ESO DPR TYPE'] == 'SPECPOS,LAMP':
                        spec_pos_list_ifs.append(fname)
                        ifs_mode = _check_mode(ifs_mode)
                    elif header['HIERARCH ESO DET NAME'] == 'IFS' and header['HIERARCH ESO DPR TYPE'] == 'WAVE,LAMP':
                        wave_list_ifs.append(fname)
                        ifs_mode = _check_mode(ifs_mode)
                    elif header['HIERARCH ESO DET NAME'] == 'IFS' and header['HIERARCH ESO DPR TYPE'] == 'FLAT,LAMP,RONGAIN':
                        gain_list_IFS.append(fname)
                        # ifs_mode = _check_mode(ifs_mode)

                if dit_irdis is not None:
                    if header['HIERARCH ESO DET SEQ1 DIT'] == dit_irdis and header['HIERARCH ESO DET NAME']== 'IRDIS' and header['HIERARCH ESO DPR TYPE'] == 'OBJECT' and header['HIERARCH ESO INS1 FILT NAME'] == filt1 and header['HIERARCH ESO INS1 OPTI2 NAME'] == filt2:
                        sci_list_irdis.append(fname)
                        sci_list_mjd_irdis.append(header['MJD-OBS'])
                    elif header['HIERARCH ESO DET SEQ1 DIT'] == dit_irdis and header['HIERARCH ESO DET NAME']== 'IRDIS' and header['HIERARCH ESO DPR TYPE'] == 'SKY' and header['HIERARCH ESO INS1 FILT NAME'] == filt1 and header['HIERARCH ESO INS1 OPTI2 NAME'] == filt2:
                        sky_list_irdis.append(fname)
                        sky_list_mjd_irdis.append(header['MJD-OBS'])
                    elif header['HIERARCH ESO DET NAME'] == 'IRDIS' and header['HIERARCH ESO DPR TYPE'] == 'OBJECT,CENTRE':
                        center_list_irdis.append(fname)
                        center_list_mjd_irdis.append(header['MJD-OBS'])
                    elif header['HIERARCH ESO DET SEQ1 DIT'] == dit_irdis and header['HIERARCH ESO DET NAME']== 'IRDIS' and 'DARK,BACKGROUND' in header['HIERARCH ESO DPR TYPE'] and header['HIERARCH ESO INS1 FILT NAME'] == filt1 and header['HIERARCH ESO INS1 OPTI2 NAME'] == filt2:
                        ins_bg_list_irdis.append(fname)
                    elif header['HIERARCH ESO DET NAME'] == 'IRDIS' and header['HIERARCH ESO DPR TYPE'] == 'DARK':
                        dark_list_irdis.append(fname)
                    elif header['HIERARCH ESO DET NAME']== 'IRDIS' and header['HIERARCH ESO DPR TYPE'] == 'FLAT,LAMP' and header['HIERARCH ESO INS1 FILT NAME'] == filt1 and header['HIERARCH ESO INS1 OPTI2 NAME'] == filt2:
                        if mode in header['HIERARCH ESO OCS DET1 IMGNAME']:
                            flat_list_irdis.append(fname)
                            if float(header['HIERARCH ESO DET SEQ1 DIT']) not in dit_irdis_flat:
                                dit_irdis_flat.append(float(header['HIERARCH ESO DET SEQ1 DIT']))
                            #mjd_obs_flat_irdis = abs(float(header['MJD-OBS']))
                    elif header['HIERARCH ESO DET NAME'] == 'IRDIS' and header['HIERARCH ESO DPR TYPE'] == 'LAMP,DISTORT' and len(distort_IRDIS)==0:
                        distort_IRDIS.append(fname)
                        dit_irdis_distort.append(float(header['HIERARCH ESO DET SEQ1 DIT']))
                        filt1_irdis_distort.append(float(header['HIERARCH ESO DET SEQ1 DIT']))
                        filt2_irdis_distort.append(float(header['HIERARCH ESO DET SEQ1 DIT']))
                    elif header['HIERARCH ESO DET NAME'] == 'IRDIS' and header['HIERARCH ESO DPR TYPE'] == 'FLAT,LAMP,RONGAIN':
                        gain_list_IRDIS.append(fname)

                if dit_psf_ifs is not None:
                    if header['HIERARCH ESO DET SEQ1 DIT'] == dit_psf_ifs and header['HIERARCH ESO DET NAME'] == 'IFS' and header['HIERARCH ESO DPR TYPE'] == 'OBJECT,FLUX' and header['HIERARCH ESO INS1 OPTI2 NAME'] == filt2:
                        psf_list_ifs.append(fname)
                        psf_list_mjd_ifs.append(header['MJD-OBS'])
                    elif header['HIERARCH ESO DET SEQ1 DIT'] == dit_psf_ifs and header['HIERARCH ESO DET NAME'] == 'IFS' and header['HIERARCH ESO DPR TYPE'] == 'SKY' and header['HIERARCH ESO INS1 OPTI2 NAME'] == filt2:
                        psf_sky_list_ifs.append(fname)
                    elif header['HIERARCH ESO DET SEQ1 DIT'] == dit_psf_ifs and header['HIERARCH ESO DET NAME'] == 'IFS' and header['HIERARCH ESO DPR TYPE'] == 'DARK,BACKGROUND' and header['HIERARCH ESO INS1 OPTI2 NAME'] == filt2:
                        psf_ins_bg_list_ifs.append(fname)

                if dit_psf_irdis is not None:
                    if header['HIERARCH ESO DET SEQ1 DIT'] == dit_psf_irdis and header['HIERARCH ESO DET NAME'] == 'IRDIS' and header['HIERARCH ESO DPR TYPE'] == 'OBJECT,FLUX' and header['HIERARCH ESO INS1 FILT NAME'] == filt1 and header['HIERARCH ESO INS1 OPTI2 NAME'] == filt2:
                        psf_list_irdis.append(fname)
                        psf_list_mjd_irdis.append(header['MJD-OBS'])
                    elif header['HIERARCH ESO DET SEQ1 DIT'] == dit_psf_irdis and header['HIERARCH ESO DET NAME'] == 'IRDIS' and header['HIERARCH ESO DPR TYPE'] == 'SKY' and header['HIERARCH ESO INS1 FILT NAME'] == filt1 and header['HIERARCH ESO INS1 OPTI2 NAME'] == filt2:
                        psf_sky_list_irdis.append(fname)
                    elif header['HIERARCH ESO DET SEQ1 DIT'] == dit_psf_irdis and header['HIERARCH ESO DET NAME'] == 'IRDIS' and header['HIERARCH ESO DPR TYPE'] == 'DARK,BACKGROUND' and header['HIERARCH ESO INS1 FILT NAME'] == filt1 and header['HIERARCH ESO INS1 OPTI2 NAME'] == filt2:
                        psf_ins_bg_list_irdis.append(fname)

                if dit_cen_ifs is not None:
                    if header['HIERARCH ESO DET SEQ1 DIT'] in dit_cen_ifs and header['HIERARCH ESO DET NAME'] == 'IFS' and header['HIERARCH ESO DPR TYPE'] == 'OBJECT,CENTER' and header['HIERARCH ESO INS1 FILT NAME'] == filt1 and header['HIERARCH ESO INS1 OPTI2 NAME'] == filt2:
                        cen_list_ifs.append(fname)
                        cen_list_mjd_ifs.append(header['MJD-OBS'])
                    elif header['HIERARCH ESO DET SEQ1 DIT'] in dit_cen_ifs and header['HIERARCH ESO DET NAME'] == 'IFS' and header['HIERARCH ESO DPR TYPE'] == 'SKY' and header['HIERARCH ESO INS1 FILT NAME'] == filt1 and header['HIERARCH ESO INS1 OPTI2 NAME'] == filt2:
                        cen_sky_list_ifs.append(fname)
                    elif header['HIERARCH ESO DET SEQ1 DIT'] in dit_cen_ifs and header['HIERARCH ESO DET NAME'] == 'IFS' and header['HIERARCH ESO DPR TYPE'] == 'DARK,BACKGROUND' and header['HIERARCH ESO INS1 FILT NAME'] == filt1 and header['HIERARCH ESO INS1 OPTI2 NAME'] == filt2:
                        cen_ins_bg_list_ifs.append(fname)

                if dit_cen_irdis is not None:
                    if header['HIERARCH ESO DET SEQ1 DIT'] in dit_cen_irdis and header['HIERARCH ESO DET NAME'] == 'IRDIS' and header['HIERARCH ESO DPR TYPE'] == 'OBJECT,CENTER' and header['HIERARCH ESO INS1 FILT NAME'] == filt1 and header['HIERARCH ESO INS1 OPTI2 NAME'] == filt2:
                        cen_list_irdis.append(fname)
                        cen_list_mjd_irdis.append(header['MJD-OBS'])
                    elif header['HIERARCH ESO DET SEQ1 DIT'] in dit_cen_irdis and header['HIERARCH ESO DET NAME'] == 'IRDIS' and header['HIERARCH ESO DPR TYPE'] == 'SKY' and header['HIERARCH ESO INS1 FILT NAME'] == filt1 and header['HIERARCH ESO INS1 OPTI2 NAME'] == filt2:
                        cen_sky_list_irdis.append(fname)
                    elif header['HIERARCH ESO DET SEQ1 DIT'] in dit_cen_irdis and header['HIERARCH ESO DET NAME'] == 'IRDIS' and header['HIERARCH ESO DPR TYPE'] == 'DARK,BACKGROUND' and header['HIERARCH ESO INS1 FILT NAME'] == filt1 and header['HIERARCH ESO INS1 OPTI2 NAME'] == filt2:
                        cen_ins_bg_list_irdis.append(fname)

            elif fname.startswith('M.'+instr) and fname.endswith('.fits'):
                calib_list.append(fname)
                header = open_header(inpath+fname)
                if header['HIERARCH ESO PRO CATG'] == 'IFS_POINT_PATTERN':
                    calib_IFS.append(fname)
                elif header['HIERARCH ESO PRO CATG'] == 'IRD_POINT_PATTERN':
                    calib_IRDIS.append(fname)

        for fname in fits_list: # all fits files
            header = open_header(inpath+fname)
            if header['HIERARCH ESO DPR TYPE'] == 'OBJECT,AO':
                continue
            if header['HIERARCH ESO DET NAME'] == 'IFS' and header['HIERARCH ESO DPR TYPE'] == 'DARK' and float(header['HIERARCH ESO DET SEQ1 DIT']) in dit_ifs_flat: #and header['HIERARCH ESO INS1 FILT NAME'] == filt1 and header['HIERARCH ESO INS1 OPTI2 NAME'] == filt2: # by elimination must be a flat lamp dark
#                if float(header['HIERARCH ESO DET SEQ1 DIT']) not in dit_ifs_flat_dark:
                dit_ifs_flat_dark.append(float(header['HIERARCH ESO DET SEQ1 DIT']))
                flat_dark_list_ifs.append(fname)
                mjdobs_fdark_list_ifs.append(float(header['MJD-OBS']))
#                else:
#                    idx_ff = dit_ifs_flat_dark.index(float(header['HIERARCH ESO DET SEQ1 DIT']))
#                    if abs(float(header['MJD-OBS'])-mjd_obs_flat_ifs) < abs(mjdobs_fdark_list_ifs[idx_ff]-mjd_obs_flat_ifs):
#                        mjdobs_fdark_list_ifs[idx_ff] = float(header['MJD-OBS'])
#                        flat_dark_list_ifs[idx_ff] = fname
            elif header['HIERARCH ESO DET NAME']== 'IRDIS' and header['HIERARCH ESO DPR TYPE'] == 'DARK' and float(header['HIERARCH ESO DET SEQ1 DIT']) in dit_irdis_flat: #and header['HIERARCH ESO INS1 FILT NAME'] == filt1 and header['HIERARCH ESO INS1 OPTI2 NAME'] == filt2: # by elimination must be a flat lamp dark
#                if float(header['HIERARCH ESO DET SEQ1 DIT']) not in dit_irdis_flat_dark:
                dit_irdis_flat_dark.append(float(header['HIERARCH ESO DET SEQ1 DIT']))
                flat_dark_list_irdis.append(fname)
                mjdobs_fdark_list_irdis.append(float(header['MJD-OBS']))
#                else:
#                    idx_ff = dit_irdis_flat_dark.index(float(header['HIERARCH ESO DET SEQ1 DIT']))
#                    if abs(float(header['MJD-OBS'])-mjd_obs_flat_irdis) < abs(mjdobs_fdark_list_irdis[idx_ff]-mjd_obs_flat_irdis):
#                        mjdobs_fdark_list_irdis[idx_ff] = float(header['MJD-OBS'])
#                        flat_dark_list_irdis[idx_ff] = fname
            if header['HIERARCH ESO DET NAME'] == 'IFS' and header['HIERARCH ESO DPR TYPE'] == 'DARK,BACKGROUND' and float(header['HIERARCH ESO DET SEQ1 DIT']) in dit_ifs_distort:
                distort_ins_bg_IFS.append(fname)
            elif header['HIERARCH ESO DET NAME']== 'IRDIS' and header['HIERARCH ESO DPR TYPE'] == 'DARK,BACKGROUND' and float(header['HIERARCH ESO DET SEQ1 DIT']) in dit_irdis_distort:
                distort_ins_bg_IRDIS.append(fname)


        # SORT ALL LISTS IN ALPHABETICAL ORDER (will be chronological)
        file_list.sort()
        fits_list.sort()
        calib_list.sort()
        if dit_ifs is not None:
            sci_list_ifs.sort()
            sky_list_ifs.sort()
            center_list_ifs.sort()
            sci_list_mjd_ifs.sort()
            sky_list_mjd_ifs.sort()
            center_list_mjd_ifs.sort()
            flat_list_ifs.sort()
            flat_list_ifs_det.sort()
            flat_list_ifs_det_BB.sort()
            dit_ifs_flat.sort()
            flat_dark_list_ifs.sort()
            dark_list_ifs.sort()
            ins_bg_list_ifs.sort()
            spec_pos_list_ifs.sort()
            wave_list_ifs.sort()
        if dit_psf_ifs is not None:
            psf_list_ifs.sort()
            psf_sky_list_ifs.sort()
            psf_ins_bg_list_ifs.sort()
            psf_list_mjd_ifs.sort()
        if dit_cen_ifs is not None:
            cen_list_ifs.sort()
            cen_sky_list_ifs.sort()
            cen_ins_bg_list_ifs.sort()
            cen_list_mjd_ifs.sort()
        if dit_irdis is not None:
            sci_list_irdis.sort()
            sky_list_irdis.sort()
            psf_list_irdis.sort()
            center_list_irdis.sort()
            sci_list_mjd_irdis.sort()
            sky_list_mjd_irdis.sort()
            psf_list_mjd_irdis.sort()
            center_list_mjd_irdis.sort()
            flat_list_irdis.sort()
            dit_irdis_flat.sort()
            flat_dark_list_irdis.sort()
            dark_list_irdis.sort()
            ins_bg_list_irdis.sort()
        if dit_psf_irdis is not None:
            psf_list_irdis.sort()
            psf_sky_list_irdis.sort()
            psf_ins_bg_list_irdis.sort()
            psf_list_mjd_irdis.sort()
        if dit_cen_irdis is not None:
            cen_list_irdis.sort()
            cen_sky_list_irdis.sort()
            cen_ins_bg_list_irdis.sort()
            cen_list_mjd_irdis.sort()

        # Write all lists in text files
        with open(outpath_filenames+"_all.txt", 'w') as f:
            for ii in range(len(file_list)):
                f.write(file_list[ii]+'\n')
        with open(outpath_filenames+"_fits.txt", 'w') as f:
            for ii in range(len(fits_list)):
                f.write(fits_list[ii]+'\n')
        with open(outpath_filenames+"_calib.txt", 'w') as f:
            for ii in range(len(calib_list)):
                    f.write(calib_list[ii]+'\n')

        if dit_ifs is not None:
            with open(outpath_filenames+"_IFS_mode.txt", 'w') as f:
                f.write(ifs_mode[0]+'\n')
            with open(outpath_filenames+"IFS_OBJ.txt", 'w') as f:
                for ii in range(len(sci_list_ifs)):
                    f.write(sci_list_ifs[ii]+'\n')
            with open(outpath_filenames+"IFS_SKY.txt", 'w') as f:
                for ii in range(len(sky_list_ifs)):
                    f.write(sky_list_ifs[ii]+'\n')
            with open(outpath_filenames+"IFS_FLAT.txt", 'w') as f:
                for ii in range(len(flat_list_ifs)):
                    f.write(flat_list_ifs[ii]+'\n')
            with open(outpath_filenames+"IFS_FLAT_DET.txt", 'w') as f:
                for ii in range(len(flat_list_ifs_det)):
                    f.write(flat_list_ifs_det[ii]+'\n')
            with open(outpath_filenames+"IFS_flat_det_BB.txt", 'w') as f:
                for ii in range(len(flat_list_ifs_det_BB)):
                    f.write(flat_list_ifs_det_BB[ii]+'\n')
            with open(outpath_filenames+"IFS_INS_BG.txt", 'w') as f:
                for ii in range(len(ins_bg_list_ifs)):
                    f.write(ins_bg_list_ifs[ii]+'\n')
            with open(outpath_filenames+"IFS_DARK.txt", 'w') as f:
                for ii in range(len(dark_list_ifs)):
                    f.write(dark_list_ifs[ii]+'\n')
            with open(outpath_filenames+"DIT_IFS_FLAT.txt", 'w') as f:
                for ii in range(len(dit_ifs_flat)):
                    f.write(str(dit_ifs_flat[ii])+'\n')
            with open(outpath_filenames+"IFS_FLAT_DARK.txt", 'w') as f:
                for ii in range(len(flat_dark_list_ifs)):
                    f.write(flat_dark_list_ifs[ii]+'\n')
            with open(outpath_filenames+"IFS_OBJ_MJD.txt", 'w') as f:
                for ii in range(len(sci_list_mjd_ifs)):
                    f.write('{:.5f} \n'.format(sci_list_mjd_ifs[ii]))
            with open(outpath_filenames+"IFS_SKY_MJD.txt", 'w') as f:
                for ii in range(len(sky_list_mjd_ifs)):
                    f.write('{:.5f} \n'.format(sky_list_mjd_ifs[ii]))
            with open(outpath_filenames+"IFS_CALIB.txt", 'w') as f:
                for ii in range(len(calib_IFS)):
                    f.write(calib_IFS[ii]+'\n')
            with open(outpath_filenames+"IFS_DISTORT.txt", 'w') as f:
                for ii in range(len(distort_IFS)):
                    f.write(distort_IFS[ii]+'\n')
            with open(outpath_filenames+"IFS_DISTORT_INS_BG.txt", 'w') as f:
                for ii in range(len(distort_ins_bg_IFS)):
                    f.write(distort_ins_bg_IFS[ii]+'\n')
            with open(outpath_filenames+"IFS_SPECPOS.txt", 'w') as f:
                for ii in range(len(spec_pos_list_ifs)):
                    f.write(spec_pos_list_ifs[ii]+'\n')
            with open(outpath_filenames+"IFS_WAVE.txt", 'w') as f:
                for ii in range(len(wave_list_ifs)):
                    f.write(wave_list_ifs[ii]+'\n')
            with open(outpath_filenames+"IFS_GAIN.txt", 'w') as f:
                for ii in range(len(gain_list_IFS)):
                    f.write(gain_list_IFS[ii]+'\n')

        if dit_irdis is not None:
            with open(outpath_filenames+"IRDIS_OBJ.txt", 'w') as f:
                for ii in range(len(sci_list_irdis)):
                    f.write(sci_list_irdis[ii]+'\n')
            with open(outpath_filenames+"IRDIS_SKY.txt", 'w') as f:
                for ii in range(len(sky_list_irdis)):
                    f.write(sky_list_irdis[ii]+'\n')
            with open(outpath_filenames+"IRDIS_FLAT.txt", 'w') as f:
                for ii in range(len(flat_list_irdis)):
                    f.write(flat_list_irdis[ii]+'\n')
            with open(outpath_filenames+"IRDIS_INS_BG.txt", 'w') as f:
                for ii in range(len(ins_bg_list_irdis)):
                    f.write(ins_bg_list_irdis[ii]+'\n')
            with open(outpath_filenames+"IRDIS_DARK.txt", 'w') as f:
                for ii in range(len(dark_list_irdis)):
                    f.write(dark_list_irdis[ii]+'\n')
            with open(outpath_filenames+"DIT_IRDIS_FLAT.txt", 'w') as f:
                for ii in range(len(dit_irdis_flat)):
                    f.write(str(dit_irdis_flat[ii])+'\n')
            with open(outpath_filenames+"IRDIS_FLAT_DARK.txt", 'w') as f:
                for ii in range(len(flat_dark_list_irdis)):
                    f.write(flat_dark_list_irdis[ii]+'\n')
            with open(outpath_filenames+"IRDIS_OBJ_MJD.txt", 'w') as f:
                for ii in range(len(sci_list_mjd_irdis)):
                    f.write('{:.5f} \n'.format(sci_list_mjd_irdis[ii]))
            with open(outpath_filenames+"IRDIS_SKY_MJD.txt", 'w') as f:
                for ii in range(len(sky_list_mjd_irdis)):
                    f.write('{:.5f} \n'.format(sky_list_mjd_irdis[ii]))
            with open(outpath_filenames+"IRDIS_CALIB.txt", 'w') as f:
                for ii in range(len(calib_IRDIS)):
                    f.write(calib_IRDIS[ii]+'\n')
            with open(outpath_filenames+"IRDIS_DISTORT.txt", 'w') as f:
                for ii in range(len(distort_IRDIS)):
                    f.write(distort_IRDIS[ii]+'\n')
            with open(outpath_filenames+"IRDIS_DISTORT_INS_BG.txt", 'w') as f:
                for ii in range(len(distort_ins_bg_IRDIS)):
                    f.write(distort_ins_bg_IRDIS[ii]+'\n')
            with open(outpath_filenames+"IRDIS_GAIN.txt", 'w') as f:
                for ii in range(len(gain_list_IRDIS)):
                    f.write(gain_list_IRDIS[ii]+'\n')

        if dit_psf_ifs is not None:
            with open(outpath_filenames+"IFS_PSF.txt", 'w') as f:
                for ii in range(len(psf_list_ifs)):
                    f.write(psf_list_ifs[ii]+'\n')
            with open(outpath_filenames+"IFS_PSF_SKY.txt", 'w') as f:
                for ii in range(len(psf_sky_list_ifs)):
                    f.write(psf_sky_list_ifs[ii]+'\n')
            with open(outpath_filenames+"IFS_PSF_INS_BG.txt", 'w') as f:
                for ii in range(len(psf_ins_bg_list_ifs)):
                    f.write(psf_ins_bg_list_ifs[ii]+'\n')
            with open(outpath_filenames+"IFS_PSF_MJD.txt", 'w') as f:
                for ii in range(len(psf_list_mjd_ifs)):
                    f.write('{:.5f} \n'.format(psf_list_mjd_ifs[ii]))

        if dit_psf_irdis is not None:
            with open(outpath_filenames+"IRDIS_PSF.txt", 'w') as f:
                for ii in range(len(psf_list_irdis)):
                    f.write(psf_list_irdis[ii]+'\n')
            with open(outpath_filenames+"IRDIS_PSF_SKY.txt", 'w') as f:
                for ii in range(len(psf_sky_list_irdis)):
                    f.write(psf_sky_list_irdis[ii]+'\n')
            with open(outpath_filenames+"IRDIS_PSF_INS_BG.txt", 'w') as f:
                for ii in range(len(psf_ins_bg_list_irdis)):
                    f.write(psf_ins_bg_list_irdis[ii]+'\n')
            with open(outpath_filenames+"IRDIS_PSF_MJD.txt", 'w') as f:
                for ii in range(len(psf_list_mjd_irdis)):
                    f.write('{:.5f} \n'.format(psf_list_mjd_irdis[ii]))

        if dit_cen_ifs is not None:
            with open(outpath_filenames+"IFS_CEN.txt", 'w') as f:
                for ii in range(len(cen_list_ifs)):
                    f.write(cen_list_ifs[ii]+'\n')
            with open(outpath_filenames+"IFS_CEN_SKY.txt", 'w') as f:
                for ii in range(len(cen_sky_list_ifs)):
                    f.write(cen_sky_list_ifs[ii]+'\n')
            with open(outpath_filenames+"IFS_CEN_INS_BG.txt", 'w') as f:
                for ii in range(len(cen_ins_bg_list_ifs)):
                    f.write(cen_ins_bg_list_ifs[ii]+'\n')
            with open(outpath_filenames+"IFS_CEN_MJD.txt", 'w') as f:
                for ii in range(len(cen_list_mjd_ifs)):
                    f.write('{:.5f} \n'.format(cen_list_mjd_ifs[ii]))

        if dit_cen_irdis is not None:
            with open(outpath_filenames+"IRDIS_CEN.txt", 'w') as f:
                for ii in range(len(cen_list_irdis)):
                    f.write(cen_list_irdis[ii]+'\n')
            with open(outpath_filenames+"IRDIS_CEN_SKY.txt", 'w') as f:
                for ii in range(len(cen_sky_list_irdis)):
                    f.write(cen_sky_list_irdis[ii]+'\n')
            with open(outpath_filenames+"IRDIS_CEN_INS_BG.txt", 'w') as f:
                for ii in range(len(cen_ins_bg_list_irdis)):
                    f.write(cen_ins_bg_list_irdis[ii]+'\n')
            with open(outpath_filenames+"IRDIS_CEN_MJD.txt", 'w') as f:
                for ii in range(len(cen_list_mjd_irdis)):
                    f.write('{:.5f} \n'.format(cen_list_mjd_irdis[ii]))

        with open(outpath_filenames+"AO_FILES.txt", 'w') as f:
            for ii in range(len(ao_files)):
                f.write(ao_files[ii]+'\n')
            dico_files['ao_files'] = ao_files

    else:
        with open(outpath_filenames+"_all.txt", 'r') as f:
            file_list = f.readlines()
        with open(outpath_filenames+"_fits.txt", 'r') as f:
            fits_list = f.readlines()
        with open(outpath_filenames+"_calib.txt", 'r') as f:
            calib_list = f.readlines()

        if dit_ifs is not None:
            with open(outpath_filenames+"IFS_mode.txt", 'r') as f:
                ifs_mode = f.readlines()
            with open(outpath_filenames+"IFS_OBJ.txt", 'r') as f:
                sci_list_ifs = f.readlines()
            with open(outpath_filenames+"IFS_SKY.txt", 'r') as f:
                sky_list_ifs = f.readlines()
            with open(outpath_filenames+"IFS_FLAT.txt", 'r') as f:
                flat_list_ifs = f.readlines()
            with open(outpath_filenames+"IFS_FLAT_DET.txt", 'r') as f:
                flat_list_ifs_det = f.readlines()
            with open(outpath_filenames+"IFS_FLAT_DET_BB.txt", 'r') as f:
                flat_list_ifs_det_BB = f.readlines()
            with open(outpath_filenames+"IFS_DARK.txt", 'r') as f:
                dark_list_ifs = f.readlines()
            with open(outpath_filenames+"IFS_INS_BG.txt", 'r') as f:
                ins_bg_list_ifs = f.readlines()
            with open(outpath_filenames+"DIT_IFS_FLAT.txt", 'r') as f:
                dit_ifs_flat = f.readlines()
            with open(outpath_filenames+"IFS_FLAT_DARK.txt", 'r') as f:
                flat_dark_list_ifs = f.readlines()
            with open(outpath_filenames+"IFS_OBJ_MJD.txt", 'r') as f:
                sci_list_mjd_ifs = f.readlines()
            with open(outpath_filenames+"IFS_SKY_MJD.txt", 'r') as f:
                sky_list_mjd_ifs = f.readlines()
            with open(outpath_filenames+"IFS_CALIB.txt", 'r') as f:
                calib_IFS = f.readlines()
            with open(outpath_filenames+"IFS_DISTORT.txt", 'r') as f:
                distort_IFS = f.readlines()
            with open(outpath_filenames+"IFS_DISTORT_INS_BG.txt", 'r') as f:
                distort_ins_bg_IFS = f.readlines()
            with open(outpath_filenames+"IFS_SPECPOS.txt", 'r') as f:
                spec_pos_list_ifs = f.readlines()
            with open(outpath_filenames+"IFS_WAVE.txt", 'r') as f:
                wave_list_ifs = f.readlines()
            with open(outpath_filenames+"IFS_GAIN.txt", 'r') as f:
                gain_list_IFS = f.readlines()

        if dit_irdis is not None:
            with open(outpath_filenames+"IRDIS_OBJ.txt", 'r') as f:
                sci_list_irdis = f.readlines()
            with open(outpath_filenames+"IRDIS_SKY.txt", 'r') as f:
                sky_list_irdis = f.readlines()
            with open(outpath_filenames+"IRDIS_FLAT.txt", 'r') as f:
                flat_list_irdis = f.readlines()
            with open(outpath_filenames+"IRDIS_DARK.txt", 'r') as f:
                dark_list_irdis = f.readlines()
            with open(outpath_filenames+"IRDIS_INS_BG.txt", 'r') as f:
                ins_bg_list_irdis = f.readlines()
            with open(outpath_filenames+"DIT_IRDIS_FLAT.txt", 'r') as f:
                dit_irdis_flat = f.readlines()
            with open(outpath_filenames+"IRDIS_FLAT_DARK.txt", 'r') as f:
                flat_dark_list_irdis = f.readlines()
            with open(outpath_filenames+"IRDIS_OBJ_MJD.txt", 'r') as f:
                sci_list_mjd_irdis = f.readlines()
            with open(outpath_filenames+"IRDIS_SKY_MJD.txt", 'r') as f:
                sky_list_mjd_irdis = f.readlines()
            with open(outpath_filenames+"IRDIS_CALIB.txt", 'r') as f:
                calib_IRDIS = f.readlines()
            with open(outpath_filenames+"IRDIS_DISTORT.txt", 'r') as f:
                distort_IRDIS = f.readlines()
            with open(outpath_filenames+"IRDIS_DISTORT_INS_BG.txt", 'r') as f:
                distort_ins_bg_IRDIS = f.readlines()
            with open(outpath_filenames+"IRDIS_GAIN.txt", 'r') as f:
                gain_list_IRDIS = f.readlines()

        if dit_psf_ifs is not None:
            with open(outpath_filenames+"IFS_PSF.txt", 'r') as f:
                psf_list_ifs = f.readlines()
            with open(outpath_filenames+"IFS_PSF_SKY.txt", 'r') as f:
                psf_sky_list_ifs = f.readlines()
            with open(outpath_filenames+"IFS_PSF_INS_BG.txt", 'r') as f:
                psf_ins_bg_list_ifs = f.readlines()
            with open(outpath_filenames+"IFS_PSF_MJD.txt", 'r') as f:
                psf_list_mjd_ifs = f.readlines()

        if dit_psf_irdis is not None:
            with open(outpath_filenames+"IRDIS_PSF.txt", 'r') as f:
                psf_list_irdis = f.readlines()
            with open(outpath_filenames+"IRDIS_PSF_SKY.txt", 'r') as f:
                psf_sky_list_irdis = f.readlines()
            with open(outpath_filenames+"IRDIS_PSF_INS_BG.txt", 'r') as f:
                psf_ins_bg_list_irdis = f.readlines()
            with open(outpath_filenames+"IRDIS_PSF_MJD.txt", 'r') as f:
                psf_list_mjd_irdis = f.readlines()

        if dit_cen_ifs is not None:
            with open(outpath_filenames+"IFS_CEN.txt", 'r') as f:
                cen_list_ifs = f.readlines()
            with open(outpath_filenames+"IFS_CEN_SKY.txt", 'r') as f:
                cen_sky_list_ifs = f.readlines()
            with open(outpath_filenames+"IFS_CEN_INS_BG.txt", 'r') as f:
                cen_ins_bg_list_ifs = f.readlines()
            with open(outpath_filenames+"IFS_CEN_MJD.txt", 'r') as f:
                cen_list_mjd_ifs = f.readlines()

        if dit_cen_irdis is not None:
            with open(outpath_filenames+"IRDIS_CEN.txt", 'r') as f:
                cen_list_irdis = f.readlines()
            with open(outpath_filenames+"IRDIS_CEN_SKY.txt", 'r') as f:
                cen_sky_list_irdis = f.readlines()
            with open(outpath_filenames+"IRDIS_CEN_INS_BG.txt", 'r') as f:
                cen_ins_bg_list_irdis = f.readlines()
            with open(outpath_filenames+"IRDIS_CEN_MJD.txt", 'r') as f:
                cen_list_mjd_irdis = f.readlines()

        with open(outpath_filenames+"AO_FILES.txt", 'r') as f:
            ao_files = f.readlines()

        file_list = [x.strip() for x in file_list]
        fits_list = [x.strip() for x in fits_list]
        calib_list = [x.strip() for x in calib_list]
        ifs_mode = [x.strip() for x in ifs_mode]
        sci_list_ifs = [x.strip() for x in sci_list_ifs]
        sky_list_ifs = [x.strip() for x in sky_list_ifs]
        flat_list_ifs = [x.strip() for x in flat_list_ifs]
        flat_list_ifs_det = [x.strip() for x in flat_list_ifs_det]
        flat_list_ifs_det_BB = [x.strip() for x in flat_list_ifs_det_BB]
        dark_list_ifs = [x.strip() for x in dark_list_ifs]
        ins_bg_list_ifs = [x.strip() for x in ins_bg_list_ifs]
        dit_ifs_flat = [float(x) for x in dit_ifs_flat]
        flat_dark_list_ifs = [x.strip() for x in flat_dark_list_ifs]
        sci_list_mjd_ifs = [x.strip() for x in sci_list_mjd_ifs]
        sky_list_mjd_ifs = [x.strip() for x in sky_list_mjd_ifs]
        calib_IFS = [x.strip() for x in calib_IFS]
        distort_IFS = [x.strip() for x in distort_IFS]
        distort_ins_bg_IFS = [x.strip() for x in distort_ins_bg_IFS]
        spec_pos_list_ifs = [x.strip() for x in spec_pos_list_ifs]
        wave_list_ifs = [x.strip() for x in wave_list_ifs]
        sci_list_irdis = [x.strip() for x in sci_list_irdis]
        sky_list_irdis = [x.strip() for x in sky_list_irdis]
        flat_list_irdis = [x.strip() for x in flat_list_irdis]
        dark_list_irdis = [x.strip() for x in dark_list_irdis]
        ins_bg_list_irdis = [x.strip() for x in ins_bg_list_irdis]
        dit_irdis_flat = [float(x) for x in dit_irdis_flat]
        flat_dark_list_irdis = [x.strip() for x in flat_dark_list_irdis]
        sci_list_mjd_irdis = [x.strip() for x in sci_list_mjd_irdis]
        sky_list_mjd_irdis = [x.strip() for x in sky_list_mjd_irdis]
        calib_IRDIS = [x.strip() for x in calib_IRDIS]
        distort_IRDIS = [x.strip() for x in distort_IRDIS]
        distort_ins_bg_IRDIS = [x.strip() for x in distort_ins_bg_IRDIS]
        psf_list_ifs = [x.strip() for x in psf_list_ifs]
        psf_sky_list_ifs = [x.strip() for x in psf_sky_list_ifs]
        psf_ins_bg_list_ifs = [x.strip() for x in psf_ins_bg_list_ifs]
        psf_list_mjd_ifs = [x.strip() for x in psf_list_mjd_ifs]
        psf_list_irdis = [x.strip() for x in psf_list_irdis]
        psf_sky_list_irdis = [x.strip() for x in psf_sky_list_irdis]
        psf_ins_bg_list_irdis = [x.strip() for x in psf_ins_bg_list_irdis]
        psf_list_mjd_irdis = [x.strip() for x in psf_list_mjd_irdis]
        cen_list_ifs = [x.strip() for x in cen_list_ifs]
        cen_sky_list_ifs = [x.strip() for x in cen_sky_list_ifs]
        cen_ins_bg_list_ifs = [x.strip() for x in cen_ins_bg_list_ifs]
        cen_list_mjd_ifs = [x.strip() for x in cen_list_mjd_ifs]
        cen_list_irdis = [x.strip() for x in cen_list_irdis]
        cen_sky_list_irdis = [x.strip() for x in cen_sky_list_irdis]
        cen_ins_bg_list_irdis = [x.strip() for x in cen_ins_bg_list_irdis]
        cen_list_mjd_irdis = [x.strip() for x in cen_list_mjd_irdis]

    # perform a check of how many calibration files were picked up for IFS
    if dit_ifs is not None:
        first_sci_mjd = sci_list_mjd_ifs[0]
        if len(flat_list_ifs_det_BB) > 2:
            print("WARNING: More than two broadband flats detected for IFS. Keeping the two closest to the science observations.", flush=True)
            flat_list_ifs_det_BB_mjd = [float(open_header(inpath+fname)['MJD-OBS']) for fname in flat_list_ifs_det_BB]
            idx = closest_to_obj(first_sci_mjd, flat_list_ifs_det_BB_mjd, n=2)
            flat_list_ifs_det_BB = [flat_list_ifs_det_BB[i] for i in idx]  # only use the closest

        if len(flat_list_ifs) > 1:
            print("WARNING: More than one IFU flat detected for IFS. Keeping the closest to the science observations.", flush=True)
            flat_list_ifs_mjd = [float(open_header(inpath+fname)['MJD-OBS']) for fname in flat_list_ifs]
            idx = closest_to_obj(first_sci_mjd, flat_list_ifs_mjd, n=1)
            flat_list_ifs = [flat_list_ifs[i] for i in idx]

        if len(wave_list_ifs) > 1:
            print("WARNING: More than one wavelength calibration detected for IFS. Keeping the closest to the science observations.", flush=True)
            wave_list_ifs_mjd = [float(open_header(inpath+fname)['MJD-OBS']) for fname in wave_list_ifs]
            idx = closest_to_obj(first_sci_mjd, wave_list_ifs_mjd, n=1)
            wave_list_ifs = [wave_list_ifs[i] for i in idx]

        if len(spec_pos_list_ifs) > 1:
            print("WARNING: More than one spectra position file detected for IFS. Keeping the closest to the science observations.", flush=True)
            spec_pos_list_ifs_mjd = [float(open_header(inpath+fname)['MJD-OBS']) for fname in spec_pos_list_ifs]
            idx = closest_to_obj(first_sci_mjd, spec_pos_list_ifs_mjd, n=1)
            spec_pos_list_ifs = [spec_pos_list_ifs[i] for i in idx]

    dico_files['ifs_mode'] = ifs_mode
    dico_files['sci_list_ifs'] = sci_list_ifs
    dico_files['sky_list_ifs'] = sky_list_ifs
    dico_files['flat_list_ifs'] = flat_list_ifs
    dico_files['flat_list_ifs_det'] = flat_list_ifs_det
    dico_files['flat_list_ifs_det_BB'] = flat_list_ifs_det_BB
    dico_files['dark_list_ifs'] = dark_list_ifs
    dico_files['ins_bg_list_ifs'] = ins_bg_list_ifs
    dico_files['dit_ifs_flat'] = dit_ifs_flat
    dico_files['flat_dark_list_ifs'] = flat_dark_list_ifs
    dico_files['sci_list_mjd_ifs'] = sci_list_mjd_ifs
    dico_files['sky_list_mjd_ifs'] = sky_list_mjd_ifs
    dico_files['calib_IFS'] = calib_IFS
    dico_files['distort_IFS'] = distort_IFS
    dico_files['distort_ins_bg_IFS'] = distort_ins_bg_IFS
    dico_files['gain_list_ifs'] = gain_list_IFS
    dico_files['specpos_IFS'] = spec_pos_list_ifs
    dico_files['wave_IFS'] = wave_list_ifs
    dico_files['sci_list_irdis'] = sci_list_irdis
    dico_files['sky_list_irdis'] = sky_list_irdis
    dico_files['flat_list_irdis'] = flat_list_irdis
    dico_files['dark_list_irdis'] = dark_list_irdis
    dico_files['ins_bg_list_irdis'] = ins_bg_list_irdis
    dico_files['dit_irdis_flat'] = dit_irdis_flat
    dico_files['flat_dark_list_irdis'] = flat_dark_list_irdis
    dico_files['sci_list_mjd_irdis'] = sci_list_mjd_irdis
    dico_files['sky_list_mjd_irdis'] = sky_list_mjd_irdis
    dico_files['calib_IRDIS'] = calib_IRDIS
    dico_files['distort_IRDIS'] = distort_IRDIS
    dico_files['distort_ins_bg_IRDIS'] = distort_ins_bg_IRDIS
    dico_files['gain_list_irdis'] = gain_list_IRDIS
    dico_files['psf_list_ifs'] = psf_list_ifs
    dico_files['psf_sky_list_ifs'] = psf_sky_list_ifs
    dico_files['psf_ins_bg_list_ifs'] = psf_ins_bg_list_ifs
    dico_files['psf_list_mjd_ifs'] = psf_list_mjd_ifs
    dico_files['psf_list_irdis'] = psf_list_irdis
    dico_files['psf_sky_list_irdis'] = psf_sky_list_irdis
    dico_files['psf_ins_bg_list_irdis'] = psf_ins_bg_list_irdis
    dico_files['psf_list_mjd_irdis'] = psf_list_mjd_irdis
    dico_files['cen_list_ifs'] = cen_list_ifs
    dico_files['cen_sky_list_ifs'] = cen_sky_list_ifs
    dico_files['cen_ins_bg_list_ifs'] = cen_ins_bg_list_ifs
    dico_files['cen_list_mjd_ifs'] = cen_list_mjd_ifs
    dico_files['cen_list_irdis'] = cen_list_irdis
    dico_files['cen_sky_list_irdis'] = cen_sky_list_irdis
    dico_files['cen_ins_bg_list_irdis'] = cen_ins_bg_list_irdis
    dico_files['cen_list_mjd_irdis'] = cen_list_mjd_irdis
    dico_files['file_list'] = file_list
    dico_files['fits_list'] = fits_list
    dico_files['calib_list'] = calib_list
    dico_files['ao_files'] = ao_files

    return dico_files


def sph_ifs_correct_spectral_xtalk(img, bpmap=None, boundary='fill',
                                   fill_value=np.nan):
    """
    CREDIT: A. Vigan (https://github.com/avigan/SPHERE/blob/master/sphere/IFS.py),
    based on obscure IDL routines from D. Mesa.

    Corrects a IFS frame from the spectral crosstalk
    This routines corrects for the SPHERE/IFS spectral crosstalk at
    small scales and (optionally) at large scales. This correction is
    necessary to correct the signal that is "leaking" between
    lenslets. See Antichi et al. (2009ApJ...695.1042A) for a
    theoretical description of the IFS crosstalk. Some informations
    regarding its correction are provided in Vigan et al. (2015), but
    this procedure still lacks a rigorous description and performance
    analysis.
    Since the correction of the crosstalk involves a convolution by a
    kernel of size 41x41, the values at the edges of the frame depend
    on how you choose to apply the convolution. Current implementation
    is EDGE_TRUNCATE. In other parts of the image (i.e. far from the
    edges), the result is identical to original routine by Dino
    Mesa. Note that in the original routine, the convolution that was
    coded did not treat the edges in a clean way defined
    mathematically. The scipy.ndimage.convolve() function offers
    different possibilities for the edges that are all documented.
    Parameters
    ----------
    img : array_like
        Input IFS science frame
    Returns
    -------
    img_corr : array_like
        Science frame corrected from the spectral crosstalk
    """

    # definition of the dimension of the matrix
    sepmax = 20
    dim    = sepmax*2+1
    bfac   = 0.727986/1.8

    # defines a matrix to be used around each pixel
    # (the value of the matrix is lower for greater
    # distances form the center.
    x, y = np.meshgrid(np.arange(dim)-sepmax, np.arange(dim)-sepmax)
    rdist  = np.sqrt(x**2 + y**2)
    kernel = 1 / (1+rdist**3 / bfac**3)
    kernel[(np.abs(x) <= 1) & (np.abs(y) <= 1)] = 0

    img_tmp = img.copy()
    if bpmap is not None:
        if bpmap.shape != img.shape:
            raise TypeError("If provided bpmap must have same shape as img")
        else:
            img_tmp[np.where(bpmap)] = np.nan

    # convolution and subtraction
    conv = convolve(img_tmp, kernel, boundary=boundary, fill_value=fill_value)
    img_corr = img - conv

    return img_corr


def closest_to_obj(obj_mjd: float, compare_mjd: list, n=1) -> list:
    """
    Find the closest file(s) in time to the OBJ.

    obj_mjd: float
        MJD of OBJ file.
    compare_mjd : list
        MJD to compare to OBJ.
    n : int
        Quantity of closest files to keep.
    """
    # convert strings to float
    if obj_mjd is str:
        obj_mjd = float(obj_mjd)
    compare_mjd = [float(x) for x in compare_mjd]

    compare_mjd = np.asarray(compare_mjd)
    delta_t = np.absolute(compare_mjd - obj_mjd)
    idx = delta_t.argsort()[:n]
    idx_list = idx.tolist()  # indices of the smallest time difference
    return idx_list
