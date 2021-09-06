#! /usr/bin/env python

"""
General utility routines used in vcal.
"""

__author__='V. Christiaens'
__all__=['set_backend',
         'find_nearest',
         'nonzero_median',
         'cube_crop_quadrant']

# coding: utf-8
import matplotlib
import numpy as np
import os
import vip_hci as vip

def set_backend():
    gui_env = [i for i in matplotlib.rcsetup.interactive_bk]
    non_gui_backends = matplotlib.rcsetup.non_interactive_bk
    print("Non Gui backends are:", non_gui_backends)
    print("Gui backends I will test for", gui_env)
    for gui in gui_env:
        print("testing", gui)
        try:
            matplotlib.use(gui)
            from matplotlib import pyplot as plt
            print("    ",gui, "Is Available")
            #plt.plot([1.5,2.0,2.5])
            #fig = plt.gcf()
            #fig.suptitle(gui)
            #plt.show()
            print("Using ..... ",matplotlib.get_backend())
            return None
        except:
            print("    ",gui, "Not found")
    # if code reaches here, it couldn't find any GUI backend, so install one
    os.system("conda install -c anaconda pyqt")
    matplotlib.use('Qt5Agg')
    from matplotlib import pyplot as plt
    plt.plot([1.5,2.0,2.5])
    fig = plt.gcf()
    fig.suptitle(gui)
    plt.show()
    print("Using ..... ",matplotlib.get_backend())
    
    return None

def find_nearest(array, value, output='index', constraint=None, n=1):
    """
    Function to find the indices, and optionally the values, of an array's n closest elements to a certain value.
    Possible outputs: 'index','value','both' 
    Possible constraints: 'ceil', 'floor', None ("ceil" will return the closest element with a value greater than 'value', "floor" the opposite)
    """
    if type(array) is np.ndarray:
        pass
    elif type(array) is list:
        array = np.array(array)
    else:
        raise ValueError("Input type for array should be np.ndarray or list.")
    
    if constraint is None:
        fm = np.absolute(array-value)

    elif constraint == 'ceil':
        fm = array-value
        fm = fm[np.where(fm>0)]    
    elif constraint == 'floor':    
         fm = -(array-value)
         fm = fm[np.where(fm>0)]    
    else:
        raise ValueError("Constraint not recognised")
        
    idx = fm.argsort()[:n]
    if n == 1:
        idx = idx[0]

    if output=='index': return idx
    elif output=='value': return array[idx]
    else: return array[idx], idx
    

    
    
def nonzero_median(cube,axis=None,median=True,thr=1e-10):
    """ Returns the mean (or median) of the non-zero values of an array along given axis.
    """
    if cube.ndim != 2 and cube.ndim != 3 and cube.ndim != 4:
        raise TypeError('Input array is not a 3d or 4d array.')


    if axis is None:
        array = cube.copy()
    elif axis == 0:
        array = cube.copy()
        _ = cube.shape[0]
        n_1 = cube.shape[1]
        if cube.ndim > 2:
            n_2 = cube.shape[2]
            if cube.ndim == 4:
                n_3 = cube.shape[3]
    elif axis == 1:
        array = np.swapaxes(cube, 0, 1)
        _ = cube.shape[1]
        n_1 = cube.shape[0]
        if cube.ndim > 2:
            n_2 = cube.shape[2]
            if cube.ndim == 4: 
                n_3 = cube.shape[3]
    elif axis == 2 and cube.ndim > 2:
        array = np.swapaxes(cube, 0, 2)
        _ = cube.shape[2]
        n_1 = cube.shape[1]
        n_2 = cube.shape[0]
        if cube.ndim == 4: 
            n_3 = cube.shape[3]
    elif axis == 3 and cube.ndim == 4:
        array = np.swapaxes(cube, 0, 3)
        _ = cube.shape[3]
        n_1 = cube.shape[1]
        n_2 = cube.shape[2]
        n_3 = cube.shape[0]
    else:
        raise ValueError('The provided axis should be either 0, 1 or 2 (or 3 if ndim == 4)')
        
    if median: 
        collapse=np.median
    else: 
        collapse=np.mean


    if axis is None:
        abs_cut = np.abs(array)
        nonzero_idx = np.nonzero(abs_cut > thr)
        median_array = collapse(array[nonzero_idx])

    elif cube.ndim == 2:
        median_array = np.zeros(n_1)
        for yy in range(n_1):
            abs_cut = np.abs(array[:,yy])
            nonzero_idx = np.nonzero(abs_cut > thr)
            if array[nonzero_idx,yy].shape[0] > 0:
                nonzero_vec = array[nonzero_idx,yy]
                median_array[yy]=collapse(nonzero_vec)
            else: median_array[yy]= 0.

    elif cube.ndim == 3:
        median_array = np.zeros([n_1,n_2])
        for yy in range(n_1):
            for xx in range(n_2):
                abs_cut = np.abs(array[:,yy,xx])
                nonzero_idx = np.nonzero(abs_cut > thr)
                if array[nonzero_idx,yy,xx].shape[1] > 0:
                    nonzero_vec = array[nonzero_idx,yy,xx]
                    median_array[yy,xx]=collapse(nonzero_vec)
                else: median_array[yy,xx]= 0.

    else:
        median_array = np.zeros([n_1,n_2,n_3])
        for nn in range(n_1):
            for yy in range(n_2):
                for xx in range(n_3):
                    abs_cut = np.abs(array[:,nn,yy,xx])
                    nonzero_idx = np.nonzero(abs_cut > thr)
                    if array[nonzero_idx,nn,yy,xx].shape[1] > 0:
                        nonzero_vec = array[nonzero_idx,nn,yy,xx]
                        median_array[nn,yy,xx]=collapse(nonzero_vec)
                    else: median_array[nn,yy,xx]= 0.

    return median_array


def cube_crop_quadrant(array,quadrant):
    """
    Routine to crop a quadrant from either a cube or an array.
    array: 3d or 2D numpy array
    quadrant: integer (1, 2, 3 or 4) tracing the trigonometric quadrant to be cropped
    """
    if array.ndim == 3:
        cy, cx = vip.var.frame_center(array[0])
        if quadrant == 1:
            subarray = array[:,int(np.floor(cy))+1:,int(np.floor(cx))+1:]
        elif quadrant == 2:
            subarray = array[:,int(np.floor(cy))+1:,:int(np.ceil(cx))]
        elif quadrant == 3:
            subarray = array[:,:int(np.ceil(cy)),:int(np.ceil(cx))]
        else:
            subarray = array[:,:int(np.ceil(cy)),int(np.floor(cx))+1:,]            
    else:
        cy, cx = vip.var.frame_center(array)
        if quadrant == 1:
            subarray = array[int(np.floor(cy))+1:,int(np.floor(cx))+1:]
        elif quadrant == 2:
            subarray = array[int(np.floor(cy))+1:,:int(np.ceil(cx))]
        elif quadrant == 3:
            subarray = array[:int(np.ceil(cy)),:int(np.ceil(cx))]
        else:
            subarray = array[:int(np.ceil(cy)),int(np.floor(cx))+1:,]             
    return subarray