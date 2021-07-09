#! /usr/bin/env python

"""
Utility routines for recentering based on background star for SPHERE/IRDIS data.
"""

__author__ = 'V. Christiaens, J. Baird'
__all__ = ['cube_recenter_bkg',
           'rough_centering',
           'fit2d_bkg_pos',
           'interpolate_bkg_pos',
           'leastsq_circle',
           'plot_data_circle'
           ]

import pdb
#import vip_hci 
from vip_hci.fits import open_fits, write_fits
from vip_hci.medsub import median_sub
from vip_hci.metrics import snr
from vip_hci.preproc import (cube_derotate, frame_rotate, frame_shift, 
                             approx_stellar_position)
from vip_hci.var import (get_square, fit_2dgaussian, fit_2dmoffat, dist, 
                         fit_2dairydisk, frame_center)


#from hciplot import plot_frames
#import pandas as pd 
import numpy as np

#from circle_fit import *
from scipy import optimize
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt#, cm, colors

pi = np.pi


def cube_recenter_bkg(array, derot_angles, fwhm, approx_xy_bkg, good_frame=None,
                      sub_med=True, fit_type='moff', snr_thr=3, rough_cen=False, 
                      nmin=10, crop_sz=None, full_output=False, verbose=False, 
                      debug=False, path_debug='./', rel_dist_unc=2e-4):
    """ Recenters a cube with a background star seen in the individual 
    images. The algorithm is based on the fact that the trajectory of the bkg 
    star should lie on a perfectly circular arc if the centering is done 
    correctly, with angular spacing between images given by derotation angles.
    
    Note: it is recommended to perform a first rough centering of the cube 
    based on either satellite spots or a 2d fit of the bright coronagraphic 
    mask signal. Otherwise, you can try setting rough_cen to True if the input 
    array is not roughly centered to a couple px accuracy. For the latter to 
    work, the rough center should be near the max intensity of convolved images. 

    Parameters
    ----------
    array : numpy ndarray
        Input 3d array or cube.
    derot_angles: numpy 1D ndarray
        Derotation angles associated to each frame of the input cube.
    approx_xy_bkg : tuple of 2 elements
        Approximate (x,y) coordinates of bkg star in derotated final frame (e.g. 
        in a median ADI image of a roughly centered cube).
    good_frame: None or numpy 2d ndarray, opt
        Derotated frame with the most accurate centering; it will be used for 
        estimating radius of circular arc (e.g. assuming the star is exactly at 
        center coords in that frame). E.g.: if available provide a satellite-spot 
        centered frame. If None, the median frame of the input sequence will be 
        used to estimate R (usually suboptimal when residual jitter is present).
    sub_med: bool, opt
        Whether to subtract median of sequence before fitting the bkg star. 
        Recommended if enough rotation available.
    fit_type : str, opt {"moffat", "gaussian", "airy"}
        Type of fit to infer the centroid of the BKG star in individual images.
    snr_thr: float, opt
        Minimum threshold in SNR of the BKG star for images of the sequence to 
        be considered for 2D fit. Recentering shifts in images where threshold 
        is not met will be interpolated.
    rough_cen : {True, False}, bool optional
        Whether to attempt a rough centering of the cube before centering based
        on circular arc fit.
    nmin: int, optional
        Minimum number of frames where the SNR condition must be met. Otherwise 
        an Error is raised.
    crop_sz: int (odd), optional
        Crop size of subimage used for 2d fitting. Default if None: 6*fwhm. 
    full_output : {True, False}, bool optional
        Whether to also output the calculated shifts in addition to the
        recentered cube.
    verbose : {True, False}, bool optional
        Whether to print phases of reduction.
    debug : {True, False}, bool optional
        Whether to print/plot/save more info. Helpful for debugging.
    path_debug : str, opt
        If debug set to True, this is the path where intermediate products will
        be saved.
    rel_dist_unc: float, opt
        Relative uncertainty on distortion. Recommended: 2e-4
        (Maire et al. 2016). Used for uncertainty estimate, in case 
        full_output is set to True.
        
    Returns
    -------
    array_rec : 3d ndarray
        Recentered cube.
    x, y : 2d array of floats
        [full_output] Shifts in x and y (dimensions: 2 x n_fr).
    x_unc, y_unc : 2d array of floats
        [full_output] Uncertainties on shifts in x and y (dimensions: 2 x n_fr).
    """

    cube=array.copy()
    if crop_sz is None:
        crop_sz = int(6*fwhm)
    if not crop_sz%2:
        crop_sz+=1
        
    #step 1 - ROUGH CENTERING based on max in convolved images
    if rough_cen:
        if verbose:
            print('First rough centering...')    
        fwhm_odd = int(fwhm)
        if not fwhm%2:
            fwhm_odd+=1
        cube, shifts = rough_centering(cube, fwhm_odd=fwhm_odd)
        mean_shift = np.mean(shifts, axis=0)
        approx_xy_bkg = (approx_xy_bkg[0]+mean_shift[1], 
                         approx_xy_bkg[1]+mean_shift[0])
        if debug:
            write_fits(path_debug+"roughly_centered_cube.fits", cube)

    #step 2 - GET MEDIAN CUBES AND EXACT MED POS 
    ## adapt: just calculate first and last expected positions based on input guess
    cy,cx = frame_center(cube[0])
    center_bkg = (cx, cy)
#    ini_pos_xy, fin_pos_xy = get_ini_fin_pos(approx_xy_bkg, center_bkg, derot_angles)
    
#    master_cube_rc_derotated, med_master_cube_rc_derotated = Get_Median(cube,derot_angles)
#    x_cen_med=approx_xy_bkg[0]
#    y_cen_med=approx_xy_bkg[1]
#    
#    med_pos_PS, std_err_med_pos_PS=Get_Median_Position_PS(med_master_cube_rc_derotated,x_cen_med,y_cen_med)

    #step 3  - INTERPOLATE bkg POS FROM approx derot position 
    x_pos_arr, y_pos_arr = interpolate_bkg_pos(approx_xy_bkg, center_bkg, 
                                               derot_angles)
    if debug:
        print("interpolated approx (x,y) positions:", x_pos_arr,y_pos_arr)
    #x_pos_arr, y_pos_arr,  radius, indexs, derot_angs_compn, cens_frame_x, cens_frame_y = res
    #plot_data_circle(x_pos_arr,y_pos_arr,cens_frame_x,cens_frame_y,radius)
#    xc, yc, R, residu = leastsq_circle(x_pos_arr,y_pos_arr)
#    plot_data_circle(x_pos_arr,y_pos_arr, xc, yc, R)
#    print(xc, yc, R, residu)
#    plt.show()
#
#    x_j, y_j = get_pos_in_circ(xc,yc,R,x_pos_arr,y_pos_arr,derot_angs_compn)
#
#
#    #step 4 - FIT GAUSSIAN/MOFFAT TO GET EXACT POS OF PS
#    Moffat=None
#    Gaussian=None
#    if fit_type=='Gaussian':
#        Gaussian=True
#        print('setting fit type to Gaussian')
#    if fit_type=='Moffat':
#        Moffat=True
#        print('setting fit type to Moffat')
#    if fit_type != 'Gaussian' and fit_type !='Moffat':
#        Moffat=True
#        print('Please Specfiy fit type in argument either Gaussian or Moffat. Setting to Moffat by default and contiuing.')
#    print(Gaussian, Moffat)
    if verbose:
        print('Fitting 2d {} to find BKG star centroid...'.format(fit_type))
    if sub_med:
        cube = cube-np.median(cube,axis=0)
        write_fits(path_debug+"sub_cube.fits",cube)
    bkg_x, bkg_y = fit2d_bkg_pos(cube, x_pos_arr, y_pos_arr, fwhm, 
                                 fit_type=fit_type, crop_sz=crop_sz)

    #step 5 - TAKE CUBE WITH THRESHOLD SNR
    if debug:
        print("bkg star x positions:", bkg_x)
        print("bkg star y positions:", bkg_y)
    if verbose:
        print('Identifying cubes with high SNR for BKG...')
    above_thr_idx = snr_thresholding(snr_thr, cube, fwhm, bkg_x, bkg_y,
                                     verbose=verbose, debug=debug)
    #below_thr_idx = [i for i in range(cube.shape[0]) if i not in above_thr_idx]
    ngood = len(above_thr_idx)
    if ngood < nmin:
        msg = "SNR threshold met in only {:.0f} frames (<{:.0f})."
        raise ValueError(msg.format(ngood,nmin))
    
    if debug:
        xc, yc, R, residu = leastsq_circle(cx, cy, bkg_x[above_thr_idx],
                                           bkg_y[above_thr_idx])
        print(xc, yc, R, residu)

        plot_data_circle(bkg_x[above_thr_idx], bkg_y[above_thr_idx], xc, yc, R)
        plt.savefig(path_debug+"TMP_circ_fit_bef_cen.pdf", bbox_inches='tight')
        plt.show()
        plot_data_circle(bkg_x[above_thr_idx], bkg_y[above_thr_idx], xc, 
                         yc, R, zoom=True)
        plt.savefig(path_debug+"TMP_circ_fit_bef_cen_zoom.pdf", 
                    bbox_inches='tight')
        plt.show()

    #step 6 - median ADI position
    if good_frame is None:
        good_frame = median_sub(cube[above_thr_idx], derot_angles[above_thr_idx])
    med_x, med_y = fit2d_bkg_pos(np.array([good_frame]), 
                                 np.array([approx_xy_bkg[0]]), 
                                 np.array([approx_xy_bkg[1]]), fwhm, 
                                 fit_type=fit_type, crop_sz=crop_sz)
    med_x = med_x[0]
    med_y = med_y[0]
    cen_y, cen_x = frame_center(good_frame)
    med_r = np.sqrt(np.power(med_x-cen_x,2)+np.power(med_y-cen_y,2))
    unc_r = med_r*rel_dist_unc    

    if verbose:
        print("Position (x,y) in median frame: ", med_x, med_y)
        print("Radial uncertainty on position from distortion unc: {:.2f} px".format(unc_r))
    #if debug:
    #    write_fits(path_debug+"TMP_med_ADI_fine_recentering.fits", good_frame)
    
    #step 7 - FIND SHIFTS IN CUBES WITH high SNR FOR BKG star w.r.t ADI image
    ## adapt to return shifts
    if verbose:
        print('Calculating shifts in cubes with high SNR for BKG...')
    shifts_x, shifts_y, unc_shifts = shifts_from_med_circ(cube[above_thr_idx], 
                                              derot_angles[above_thr_idx], 
                                              med_x, med_y, fwhm=fwhm, 
                                              crop_sz=crop_sz,
                                              fit_type=fit_type, debug=debug,
                                              path_debug=path_debug,
                                              full_output=True)
    
    unc_shift_r = np.zeros(ngood)
    for i in range(ngood):
        unc_shift_r[i] = np.sqrt(np.sum(np.power(unc_shifts[i,:],2)))
    med_unc_shifts = np.median(unc_shift_r)
    final_med_unc = np.sqrt(unc_r**2+med_unc_shifts**2)
    final_unc = np.sqrt(unc_r**2+np.power(unc_shift_r,2))
    
    if verbose:
        print("Median uncertainty on BKG star position: {:.2f}, px".format(med_unc_shifts))
        print("FINAL MEDIAN UNCERTAINTY ON star position based on BKG: {:.2f}".format(final_med_unc))
    if debug:
        write_fits(path_debug+"shifts_above_thr.fits",np.array([shifts_x, 
                                                                shifts_y]))
    #step 8 - INTERPOLATE SHIFTS FOR CUBES where SNR was too low
    if len(above_thr_idx) < cube.shape[0]:
        if verbose:
            print('Interpolating shifts in cubes with low SNR for BKG...')
        final_shifts_fx = interp1d(derot_angles[above_thr_idx], shifts_x)
        final_shifts_x = final_shifts_fx(derot_angles)
        final_shifts_fy = interp1d(derot_angles[above_thr_idx], shifts_y)
        final_shifts_y = final_shifts_fy(derot_angles)
    else:
        final_shifts_x = shifts_x
        final_shifts_y = shifts_y
        
    #step 9 - final recentering with all shifts
    if verbose:
        print('Final recentering of all frames.')
    for i in range(cube.shape[0]):
        cube[i] = frame_shift(array[i], final_shifts_y[i], final_shifts_x[i])
    final_shifts = np.array([final_shifts_x,final_shifts_y])

    #step 10 - double check it worked by 
    if debug:
        # plotting new BKG pos and best-fit circular arc
        fin_bkg_x, fin_bkg_y = fit2d_bkg_pos(cube, x_pos_arr, y_pos_arr, fwhm, 
                                             fit_type=fit_type, crop_sz=crop_sz)
        xc, yc, R, residu = leastsq_circle(cx, cy, fin_bkg_x[above_thr_idx],
                                           fin_bkg_y[above_thr_idx])
        print(xc, yc, R, residu)

        plot_data_circle(fin_bkg_x[above_thr_idx], fin_bkg_y[above_thr_idx], xc, 
                         yc, R)
        plt.savefig(path_debug+"TMP_double_check_cen.pdf", bbox_inches='tight')
        plt.show()
        plot_data_circle(fin_bkg_x[above_thr_idx], fin_bkg_y[above_thr_idx], xc, 
                         yc, R, zoom=True)
        plt.savefig(path_debug+"TMP_double_check_cen_zoom.pdf", 
                    bbox_inches='tight')
        plt.show()
        # plotting derotated positions
        n_fr = cube.shape[0]
        derot_cube=cube_derotate(cube, derot_angles, imlib='opencv', 
                                 interpolation='lanczos4',
                                 cxy=None, border_mode='constant')
        write_fits(path_debug+"TMP_doublecheck_derot_cube.fits", derot_cube)
        med_x_all = [med_x]*n_fr
        med_y_all = [med_y]*n_fr
        fin_der_x, fin_der_y = fit2d_bkg_pos(derot_cube, np.array(med_x_all), 
                                             np.array(med_y_all), fwhm, 
                                             fit_type=fit_type, crop_sz=crop_sz)        
        plot_data_derot(med_x, med_y, fin_der_x[above_thr_idx], 
                        fin_der_y[above_thr_idx], final_unc)
        plt.savefig(path_debug+"TMP_double_check_derot_cen.pdf", 
                    bbox_inches='tight')
        plt.show()
        
    if full_output:
        return cube, final_shifts, final_unc
    else:
        return cube


def rough_centering(array, fwhm_odd=5):
    
    nframes = array.shape[0]
    max_coords = approx_stellar_position(array, fwhm_odd, return_test=False, 
                                         verbose=False)
    cen = frame_center(array[0])
    # obj_cubes_all_frames - master cube created in step 1 
    
#    Rough=[]
#    maxlist=[]
#    shifts_rc_final=[]
#    censlist=[]
#
#    for nn in range(0, len(array)):
#        tmp=array[nn]
#        array=vip_hci.var.frame_filter_lowpass(tmp, mode=mode, 
#                                               median_size=median_size, 
#                                               fwhm_size=fwhm_size, 
#                                               gauss_mode=gauss_mode)
#        cens=vip_hci.var.frame_center(tmp)
#        censlist.append(cens)
#
#        max_ind=np.argmax(array,axis=None)
#        max_ind_coords=np.unravel_index(max_ind,array.shape)
#        max_ind_coords=np.array(max_ind_coords)
#        maxlist.append(max_ind_coords)
#
#
#    max_ind_coords=np.array(max_ind_coords)
#    max_ind_coords=max_ind_coords

    shifts_rc= np.array([cen]*nframes)-max_coords#np.array(censlist)-np.array(maxlist)
    
    cube=[]

    for i in range(0,len(array)):
        cen_frame=frame_shift(array[i], shifts_rc[i,0], shifts_rc[i,1], 
                              imlib='opencv', interpolation='lanczos4', 
                              border_mode='reflect')
        cube.append(cen_frame)

    cube=np.array(cube)
    
    return cube, shifts_rc



def interpolate_bkg_pos(approx_xy_bkg, center_bkg, derot_angles):
    """Infer all positions of BKG star in the non-derotated cube. """
    
    cx, cy = center_bkg
    
    # polar coords of bkg
    theta = np.arctan2(approx_xy_bkg[1]-cy,approx_xy_bkg[0]-cx)
    r = dist(cy, cx, approx_xy_bkg[1], approx_xy_bkg[0])
    
    # theta ini and fin
    thetas = np.rad2deg(theta)-derot_angles
    # convert back to cartesian
    x = cx + r*np.cos(np.deg2rad(thetas))
    y = cy + r*np.sin(np.deg2rad(thetas))
    
    return x, y


def fit2d_bkg_pos(cube, x_j, y_j, fwhm, fit_type='moffat', crop_sz=None):

    n_frames=cube.shape[0]

    bkg_x = np.zeros(n_frames)
    bkg_y = np.zeros(n_frames)
    #y0=np.zeros([n_frames])
    #x0=np.zeros([n_frames])
    #counter = 0
    if crop_sz is None:
        crop_sz = int(6*fwhm)
    if not crop_sz%2:
        crop_sz+=1
    
    for sc in range(0,cube.shape[0]):
        cent_coords=(int(x_j[sc]),int(y_j[sc]))
        #tmp_crop, y0[sc], x0[sc] =get_square(tmpn,crop_sz,int(y_j[sc]),int(x_j[sc]),position=True)
        if fit_type=='gauss':
            bkg_y[sc], bkg_x[sc] = fit_2dgaussian(cube[sc], cent=cent_coords, 
                                                  crop=True, cropsize=crop_sz,
                                                  fwhmx=fwhm, fwhmy=fwhm, 
                                                  theta=0, threshold=True, 
                                                  sigfactor=5, 
                                                  full_output=False, 
                                                  debug=False)
        elif fit_type == 'moff':
            bkg_y[sc], bkg_x[sc] = fit_2dmoffat(cube[sc], crop=True, 
                                                cropsize=crop_sz,
                                                cent=cent_coords, fwhm=fwhm,
                                                threshold=True, sigfactor=5, 
                                                full_output=False, debug=False)            
        elif fit_type == 'airy':
            bkg_y[sc], bkg_x[sc] = fit_2dairydisk(cube[sc], crop=False, 
                                                  cent=cent_coords, 
                                                  cropsize=crop_sz, fwhm=fwhm,
                                                  threshold=True, sigfactor=5, 
                                                  full_output=False,
                                                  debug=False)
        else:
            msg = "Fit type not recognised. Should be moffat, gaussian or airy"
            raise TypeError(msg)
    
    return bkg_x, bkg_y


def snr_thresholding(snr_thr, array, fwhm, fin_cen_x, fin_cen_y, verbose=False, 
                     debug=False):
    snr_comp = []#np.zeros(ntot_fr)
    counter = 0

    fin_cen_x=np.array(fin_cen_x)
    fin_cen_y=np.array(fin_cen_y)
    source_xy=list(zip(fin_cen_x, fin_cen_y))
    #print(source_xy_g)

    #tmp_crop_master=[]
    ##change this to be cropped + center co ords of cropped - not total image!!!!

    # identify outliers
    med_x = np.median(fin_cen_x)
    med_y = np.median(fin_cen_y)
    std_x = np.std(fin_cen_x)
    std_y = np.std(fin_cen_y)


        #tmp_crop =get_square(tmp,crop_sz,int(y_j[sc]),int(x_j[sc]),position=True)
        #tmp_crop_master=tmp_crop_master.append(tmp_crop)
    for nn in range(array.shape[0]):
        cond1 = source_xy[nn][0]>med_x+3*std_x
        cond2 = source_xy[nn][0]<med_x-3*std_x
        cond3 = source_xy[nn][1]>med_y+3*std_y
        cond4 = source_xy[nn][1]<med_y-3*std_y
        if cond1 or cond2 or cond3 or cond4:
            snr_comp.append(0) # outlier position => set snr to 0
        else:
            #tmp = master_cube_rc
            tmp_tmp = snr(array[nn], source_xy[nn], fwhm, plot=False, verbose=False, 
                          full_output=False)
            if verbose:
                print("SNR for frame {:.0f}: {:.1f}".format(nn,tmp_tmp))
            #pdb.set_trace()
            #tmp_tmp = vip_hci.metrics.snr(tmp_crop, source_xy_g_crop[nn], fwhmx, plot=False,verbose=False, full_output=False)
            snr_comp.append(tmp_tmp)
            if np.isnan(snr_comp[counter]):
                print("Measured SNR is nan for frame #{:.0f}".format(nn))
                print('Nan replaced by 0')
                snr_comp[counter] = 0
                #pdb.set_trace() # There is a bug if nan!
        counter+=1
        ntot_fr = counter

    snr_comp = np.array(snr_comp)
    #write_fits(path+'5_snr_comp.fits',snr_comp)
#    print("ntot_fr = {:.0f}".format(ntot_fr)) 

    #print(snr_comp)
#    snr_comp_txt=open('snr_comp.txt','a')
#    for x in range(len(snr_comp)): 
#        snr_comp_txt.write("\n"+str(snr_comp[x]))
#        #print(snr_comp[x])

    counter_le=array.shape[0]
#    counter_le2 = 0
#    counter_le3 = 0        
#    counter_le4 = 0
#    counter_le5 = 0        
#    counter_le6 = 0
#    counter_le7 = 0
#    counter_le8 = 0
#    counter_le9 = 0
#    counter_le10 = 0

    index_le=[]
#    index_le2=[]
#    index_le3=[]
#    index_le4=[]
#    index_le5=[]
#    index_le6=[]
#    index_le7=[]
#    index_le8=[]
#    index_le9=[]
#    index_le10=[]
    if debug:
        #fig = 
        plt.figure( facecolor='white')  #figsize=(7, 5.4), dpi=72,
        #plt.axis('equal')
    while counter_le == array.shape[0]:
        for ii in range(ntot_fr):
            if snr_comp[ii] > snr_thr:
                color = 'bo'
                counter_le -= 1
                index_le.append(ii)
        #        elif snr_comp[ii] < 2.:
        #            color = 'ro' 
        #            counter_le2 += 1
        #            index_le2.append(ii)
        #        elif snr_comp[ii] < 3.:
        #            color = 'ro' 
        #            counter_le3 += 1
        #            index_le3.append(ii)
        #        elif snr_comp[ii] < 4.:
        #            color = 'yo' 
        #            counter_le4 += 1
        #            index_le4.append(ii)
        #        elif snr_comp[ii] < 5:
        #            color = 'yo'
        #            counter_le5 += 1
        #            index_le5.append(ii)
        #        elif snr_comp[ii] < 6.:
        #            color = 'yo' 
        #            counter_le6 += 1
        #            index_le6.append(ii)
        #        elif snr_comp[ii] < 7:
        #            color = 'bo'
        #            counter_le7 += 1
        #            index_le7.append(ii)
        #        elif snr_comp[ii] < 8:
        #            color = 'go'
        #            counter_le8 += 1
        #            index_le8.append(ii)
        #        elif snr_comp[ii] < 9:
        #            color = 'go'
        #            counter_le9 += 1
        #            index_le9.append(ii)
        #        elif snr_comp[ii] < 10.:
        #            color = 'go' 
        #            counter_le10 += 1 
        #            index_le10.append(ii)
            else:
                color = 'ro'
                if debug:
                    plt.plot(ii, snr_comp[ii], color , lw=2)
        if counter_le == len(snr_comp):
            msg="No match for SNR threshold. It will now be reduced to {:.1f}"
            print(msg.format(snr_thr/2))
            snr_thr/=2
            pdb.set_trace()
    if debug:
        plt.grid()
        plt.show()

    # 6c. Define a threshold in SNR
        # count how many good frames are left
    print( "Total number of frames: {:.0f}".format(ntot_fr))
    print ("Total number of frames where SNR > {:.1f}: {:.0f}".format(snr_thr, ntot_fr-counter_le))
#    print ("Total number of frames where SNR > 3: ", ntot_fr-counter_le3-counter_le2-counter_le1)
#    print ("Total number of frames where SNR > 4: ", ntot_fr-counter_le4-counter_le3-counter_le2-counter_le1)
#    print ("Total number of frames where SNR > 5: ", ntot_fr-counter_le5-counter_le3-counter_le4-counter_le2-counter_le1)
#    print( "Total number of frames where SNR > 6: ", ntot_fr-counter_le6-counter_le5-counter_le3-counter_le4-counter_le2-counter_le1)
#    print ("Total number of frames where SNR > 7: ", ntot_fr-counter_le7-counter_le6-counter_le5-counter_le3-counter_le4-counter_le2-counter_le1)
#    print ("Total number of frames where SNR > 8: ", ntot_fr-counter_le8-counter_le7-counter_le6-counter_le5-counter_le4-counter_le3-counter_le2-counter_le1)
#    print ("Total number of frames where SNR > 9: ", ntot_fr-counter_le9-counter_le8-counter_le7-counter_le6-counter_le5-counter_le4-counter_le3-counter_le2-counter_le1)
#    print ("Total number of frames where SNR > 10: ", ntot_fr-counter_le10-counter_le9-counter_le8-counter_le7-counter_le6-counter_le5-counter_le4-counter_le3-counter_le2-counter_le1)

#    print("indexs of snr")
#    print("indexs of snr less than {:.1f}: {}".format(snr_thr, index_le,'\n', len(index_le))
#    print("indexs of snr 1-2",index_le2,'\n', len(index_le2))
#    print("indexs of snr 2-3",index_le3,'\n', len(index_le3))
#    print("indexs of snr 3-4",index_le4,'\n', len(index_le4))
#    print("indexs of snr 4-5",index_le5,'\n', len(index_le5))
#    print("indexs of snr 5-6",index_le6,'\n', len(index_le6))
#    print("indexs of snr 7-8",index_le7,'\n', len(index_le7))
#    print("indexs of snr 8-9",index_le8,'\n', len(index_le8))
#    print("indexs of snr 9-10",index_le9,'\n', len(index_le9))
#    print("indexs of snr more than 10",index_le10,'\n', len(index_le10))

    ### UPDATE THIS CELL BASED ON RESULT ABOVE

    final_thr_idx=[]

    #ngood_fr= ntot_fr-counter_le
    final_thr_idx.extend(index_le)
#        final_threshold_indexs.extend(index_le3)
#        final_threshold_indexs.extend(index_le4)
#        final_threshold_indexs.extend(index_le5)
#        final_threshold_indexs.extend(index_le6)
#        final_threshold_indexs.extend(index_le7)
#        final_threshold_indexs.extend(index_le8)
#        final_threshold_indexs.extend(index_le9)
#        final_threshold_indexs.extend(index_le10)
#    elif snr_thr==2:
#        ngood_fr=ntot_fr-counter_le1-counter_le2
#        final_threshold_indexs.extend(index_le3)
#        final_threshold_indexs.extend(index_le4)
#        final_threshold_indexs.extend(index_le5)
#        final_threshold_indexs.extend(index_le6)
#        final_threshold_indexs.extend(index_le7)
#        final_threshold_indexs.extend(index_le8)
#        final_threshold_indexs.extend(index_le9)
#        final_threshold_indexs.extend(index_le10)
#    elif snr_thr==3:
#        ngood_fr=ntot_fr-counter_le3-counter_le2-counter_le1
#        final_threshold_indexs.extend(index_le4)
#        final_threshold_indexs.extend(index_le5)
#        final_threshold_indexs.extend(index_le6)
#        final_threshold_indexs.extend(index_le7)
#        final_threshold_indexs.extend(index_le8)
#        final_threshold_indexs.extend(index_le9)
#        final_threshold_indexs.extend(index_le10)
#    elif snr_thr==4:
#        ngood_fr=ntot_fr-counter_le4-counter_le3-counter_le2-counter_le1
#        final_threshold_indexs.extend(index_le5)
#        final_threshold_indexs.extend(index_le6)
#        final_threshold_indexs.extend(index_le7)
#        final_threshold_indexs.extend(index_le8)
#        final_threshold_indexs.extend(index_le9)
#        final_threshold_indexs.extend(index_le10)
#    elif snr_thr==5:
#        ngood_fr=ntot_fr-counter_le5-counter_le3-counter_le4-counter_le2-counter_le1
#        final_threshold_indexs.extend(index_le6)
#        final_threshold_indexs.extend(index_le7)
#        final_threshold_indexs.extend(index_le8)
#        final_threshold_indexs.extend(index_le9)
#        final_threshold_indexs.extend(index_le10)
#    elif snr_thr==6:
#        ngood_fr=ntot_fr-counter_le6-counter_le5-counter_le3-counter_le4-counter_le2-counter_le1
#        final_threshold_indexs.extend(index_le7)
#        final_threshold_indexs.extend(index_le8)
#        final_threshold_indexs.extend(index_le9)
#        final_threshold_indexs.extend(index_le10)
#    elif snr_thr==7:
#        ngood_fr=ntot_fr-counter_le7-counter_le6-counter_le5-counter_le3-counter_le4-counter_le2-counter_le1
#        final_threshold_indexs.extend(index_le8)
#        final_threshold_indexs.extend(index_le9)
#        final_threshold_indexs.extend(index_le10)
#    elif snr_thr==8:
#        ngood_fr=ntot_fr-counter_le8-counter_le7-counter_le6-counter_le5-counter_le4-counter_le3-counter_le2-counter_le1
#        final_threshold_indexs.extend(index_le9)
#        final_threshold_indexs.extend(index_le10)
#    elif snr_thr==9:
#        ngood_fr=ntot_fr-counter_le9-counter_le8-counter_le7-counter_le6-counter_le5-counter_le4-counter_le3-counter_le2-counter_le1
#        final_threshold_indexs.extend(index_le10)
#    elif snr_thr==10:
#        ngood_fr=ntot_fr-counter_le10-counter_le9-counter_le8-counter_le7-counter_le6-counter_le5-counter_le4-counter_le3-counter_le2-counter_le1

    final_thr_idx.sort()
    
#    list.sort(final_threshold_indexs)
#    print(final_threshold_indexs)
#
#    tmp = snr_comp[np.where(snr_comp>=snr_thr)]
#    print(len(final_threshold_indexs))
#    print(tmp.shape)
#
#    master_cube_snrthr=cube[final_threshold_indexs]
#    fin_cen_x_snrthr=fin_cen_x[final_threshold_indexs]
#    fin_cen_y_snrthr=fin_cen_y[final_threshold_indexs]
#    print(master_cube_snrthr.shape)
    
    return final_thr_idx


def leastsq_circle(cx, cy, x, y, w=None):
    # coordinates of the barycenter
    #x_m = np.mean(x)
    #y_m = np.mean(y)
    #center_estimate = cx, cy
    ini_est = np.array([cx, cy])
    center, ier = optimize.leastsq(f_circ, ini_est, args=(x,y,w))
    xc, yc = center
    Ri       = dist(center[1],center[0],y,x)
    R        = Ri.mean()
    residu   = np.sum((Ri - R)**2)
    return xc, yc, R, residu


def f_circ(coords, x, y, w):
    """ calculate the algebraic distance between the data points and the mean 
    circle centered at c=(xc, yc) """
    cx, cy = coords
    Ri = dist(cy,cx,y,x)
    return Ri - np.average(Ri, axis=None, weights=w)


def plot_data_circle(x,y, xc, yc, R, zoom=False):
    #fig = 
    plt.figure(facecolor='white')  #figsize=(7, 5.4), dpi=72,
    plt.axis('equal')

    if zoom:
        theta_ini = np.amin(np.arctan2(y-yc,x-xc))-0.1*pi
        theta_fin = np.amax(np.arctan2(y-yc,x-xc))+0.1*pi
        theta_fit = np.linspace(theta_ini, theta_fin, 180)
    else:
        theta_fit = np.linspace(-pi, pi, 180)
    
    x_fit = xc + R*np.cos(theta_fit)
    y_fit = yc + R*np.sin(theta_fit)
    plt.plot(x_fit, y_fit, 'b-' , label="fitted circle", lw=2)
    plt.plot([xc], [yc], 'bD', mec='y', mew=1)
    plt.xlabel('x')
    plt.ylabel('y')   
    # plot data
    plt.plot(x, y, 'ro', label='data', mew=1, alpha=0.6)

    plt.legend(loc='best',labelspacing=0.1 )
    plt.grid()
    plt.title('Least Squares Circle')
    
    
def plot_data_derot(med_x, med_y, derot_x, derot_y, err, zoom=False):
    #fig = 
    plt.figure(facecolor='white')  #figsize=(7, 5.4), dpi=72,
    plt.axis('equal')

    label = None
    for i in range(len(derot_x)):
        if i== len(derot_x)-1:
            label = "BKG in OBJ after corr."
        alpha = 0.1+0.8*i/(len(derot_x)-1)
        plt.errorbar(derot_x[i], derot_y[i], err[i], err[i], fmt='ro', label=label, 
                     lw=2, alpha=alpha)
    plt.plot([med_x], [med_y], 'bD', mec='y', mew=1, label="BKG in derot CEN")
    plt.xlabel('x')
    plt.ylabel('y')

    plt.legend(loc='best',labelspacing=0.1 )
    plt.grid()
    plt.title('Least Squares Circle')
    
    
def shifts_from_med_circ(array, derot_angles, med_x, med_y, fwhm=5,
                         crop_sz=None, fit_type='moffat', debug=False,
                         path_debug='./', full_output=False):
    
    """
    Note: translation and rotation are not commutative! 
    Trick: find the shift in polar coordinates and apply it at original coords.
    """
    
    if crop_sz is None:
        crop_sz = int(8*fwhm)
    if not crop_sz%2:
        crop_sz+=1
    
    cen_y, cen_x = frame_center(array[0])
    derot_arr=np.zeros_like(array)
    
    derot_arr=cube_derotate(array, derot_angles, imlib='opencv', 
                            interpolation='lanczos4',
                            cxy=None, border_mode='constant')
    if debug:
        write_fits(path_debug+"TMP_derot_cube.fits",derot_arr)

    derot_small_cubes=[]
    corner_coords_small_cube=[]
    for i in range(array.shape[0]):

        #crop - centered around same co-ords as median crop
        derot_small_arr, y0, x0 = get_square(derot_arr[i], crop_sz, med_y,
                                             med_x, position=True)
        #print(derot_small_cube,y0_derot,x0_derot)
        derot_small_cubes.append(derot_small_arr)
        corner_coords_small_cube.append([x0,y0])
        #plot_frames(derot_small_cube)

    corner_coords_small_cube=np.array(corner_coords_small_cube)
    derot_small_cubes=np.array(derot_small_cubes)

    if debug:
        write_fits(path_debug+"TMP_derot_cube_crop.fits", derot_small_cubes)
    
    list_cens=[]
    #list_cens_err=[]
    if full_output:
        cen_unc = []
    for i in range(array.shape[0]):
        if 'moff' in fit_type:
            df_fit=fit_2dmoffat(derot_small_cubes[i], 
                                cent=[med_x,med_y], fwhm=fwhm,
                                threshold=True, sigfactor=5, 
                                full_output=full_output, debug=False)
        elif 'gauss' in fit_type:
            df_fit=fit_2dgaussian(derot_small_cubes[i], 
                                            cent=[med_x,med_y], 
                                            crop=False, fwhmx=fwhm, fwhmy=fwhm, 
                                            theta=0, threshold=True, 
                                            sigfactor=5, full_output=full_output, 
                                            debug=False)
        elif fit_type == 'airy':
            df_fit=fit_2dairydisk(derot_small_cubes[i], crop=False, 
                                            cent=[med_x,med_y], 
                                            fwhm=fwhm, threshold=True, 
                                            sigfactor=5, full_output=full_output, 
                                            debug=False)
        else:
            msg = "Fit type not recognised. Should be moff, gauss or airy"
            raise TypeError(msg)
        if full_output:
            cen_unc.append([float(df_fit['centroid_x_err']),float(df_fit['centroid_y_err'])])
            df_fit=[float(df_fit['centroid_y']),float(df_fit['centroid_x'])]
        list_cens.append([df_fit[1],df_fit[0]])
    
    list_cens=np.array(list_cens)
    coords_derot=list_cens+corner_coords_small_cube
    if debug:
        print("Position (x,y) inferred in individual frames: ", coords_derot)
        write_fits(path_debug+"TMP_xy_derot_pos.fits", coords_derot)
        print("Shape: ", coords_derot.shape)
        
    # Convert derot med pos in polar coords
    med_r = np.sqrt(np.power(med_x-cen_x,2)+np.power(med_y-cen_y,2))
    med_theta = np.rad2deg(np.arctan2(med_y-cen_y,med_x-cen_x))

    # Convert derot cart coords to polar coords
    coords_r = np.sqrt(np.power(coords_derot[:,0]-cen_x,2)+np.power(coords_derot[:,1]-cen_y,2))
    coords_theta = np.rad2deg(np.arctan2(coords_derot[:,1]-cen_y,coords_derot[:,0]-cen_x))
    
    # Compute original theta
    coords_theta_ori = coords_theta-derot_angles
    
    # Infer new non-derot theta
    new_coords_theta_ori = med_theta-derot_angles
    
    # Infer original coords from measurements
    x_ori = cen_x + coords_r*np.cos(np.deg2rad(coords_theta_ori))
    y_ori = cen_y + coords_r*np.sin(np.deg2rad(coords_theta_ori))
    
    # Infer new original coords that would lead to med derot position
    new_x_ori = cen_x + med_r*np.cos(np.deg2rad(new_coords_theta_ori))
    new_y_ori = cen_y + med_r*np.sin(np.deg2rad(new_coords_theta_ori))
    
    # Infer cartesian shifts at ori location
    shifts_x = new_x_ori-x_ori
    shifts_y = new_y_ori-y_ori

    if debug:
        print("Ori shifts (x) inferred in individual frames (MET1): ", shifts_x)
        print("Ori shifts (y) inferred in individual frames (MET1): ", shifts_y)

    # METHOD 2:
    if debug:
        print("Ori shifts (x) inferred in individual frames (MET2): ", shifts_x)
        print("Ori shifts (y) inferred in individual frames (MET2): ", shifts_y)
        
#    xc_v, yc_v, R_v, residu_v = leastsq_circle(coords_derot_med[:,0],
#                                               coords_derot_med[:,1])
#    if debug:
#        plot_data_circle(coords_derot_med[:,0], coords_derot_med[:,1], xc_v, 
#                         yc_v, R_v)
#        plt.show()
    
#    final_master=[]
#
#    for i in range(array.shape[0]):
#        tmp = frame_shift(array[i], shifts_y[i], shifts_x[i], imlib='opencv', 
#                          interpolation='lanczos4', border_mode='reflect')
#        final_master.append(tmp)
#
#    final_master=np.array(final_master)
    #write_fits('final_master.fits',final_master)
    if full_output:
        return shifts_x, shifts_y, np.array(cen_unc)
    else:
        return shifts_x, shifts_y