#! /usr/bin/env python
# coding: utf-8
"""
General utility routines used in vcal.
"""

__author__ = 'V. Christiaens'
__all__ = ['set_backend',
         'nonzero_median',
         'cube_crop_quadrant',
         'most_common']

from os import system
from statistics import mode

import numpy as np
from matplotlib import get_backend
from matplotlib import use as mpl_backend
from matplotlib.rcsetup import interactive_bk, non_interactive_bk

from vip_hci.var.coords import frame_center


def set_backend():
    gui_env = [i for i in interactive_bk]
    non_gui_backends = non_interactive_bk
    print("Non Gui backends are:", non_gui_backends)
    print("Gui backends I will test for", gui_env)
    for gui in gui_env:
        print("testing", gui)
        try:
            mpl_backend(gui)
            from matplotlib import pyplot as plt
            print("    ",gui, "Is Available")
            #plt.plot([1.5,2.0,2.5])
            #fig = plt.gcf()
            #fig.suptitle(gui)
            #plt.show()
            print("Using ..... ", get_backend())
            return None
        except:
            print("    ",gui, "Not found")
    # if code reaches here, it couldn't find any GUI backend, so install one
    system("conda install -c anaconda pyqt")
    mpl_backend('Qt5Agg')
    from matplotlib import pyplot as plt
    plt.plot([1.5,2.0,2.5])
    fig = plt.gcf()
    fig.suptitle(gui)
    plt.show()
    print("Using ..... ", get_backend())

    return None


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
        cy, cx = frame_center(array[0])
        if quadrant == 1:
            subarray = array[:,int(np.floor(cy))+1:,int(np.floor(cx))+1:]
        elif quadrant == 2:
            subarray = array[:,int(np.floor(cy))+1:,:int(np.ceil(cx))]
        elif quadrant == 3:
            subarray = array[:,:int(np.ceil(cy)),:int(np.ceil(cx))]
        else:
            subarray = array[:,:int(np.ceil(cy)),int(np.floor(cx))+1:,]
    else:
        cy, cx = frame_center(array)
        if quadrant == 1:
            subarray = array[int(np.floor(cy))+1:,int(np.floor(cx))+1:]
        elif quadrant == 2:
            subarray = array[int(np.floor(cy))+1:,:int(np.ceil(cx))]
        elif quadrant == 3:
            subarray = array[:int(np.ceil(cy)),:int(np.ceil(cx))]
        else:
            subarray = array[:int(np.ceil(cy)),int(np.floor(cx))+1:,]
    return subarray


def most_common(List):
	return(mode(List))