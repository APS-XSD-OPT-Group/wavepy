#!/usr/bin/env python
# -*- coding: utf-8 -*-
# #########################################################################
# Copyright (c) 2015, UChicago Argonne, LLC. All rights reserved.         #
#                                                                         #
# Copyright 2015. UChicago Argonne, LLC. This software was produced       #
# under U.S. Government contract DE-AC02-06CH11357 for Argonne National   #
# Laboratory (ANL), which is operated by UChicago Argonne, LLC for the    #
# U.S. Department of Energy. The U.S. Government has rights to use,       #
# reproduce, and distribute this software.  NEITHER THE GOVERNMENT NOR    #
# UChicago Argonne, LLC MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR        #
# ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE.  If software is     #
# modified to produce derivative works, such modified software should     #
# be clearly marked, so as not to confuse it with the version available   #
# from ANL.                                                               #
#                                                                         #
# Additionally, redistribution and use in source and binary forms, with   #
# or without modification, are permitted provided that the following      #
# conditions are met:                                                     #
#                                                                         #
#     * Redistributions of source code must retain the above copyright    #
#       notice, this list of conditions and the following disclaimer.     #
#                                                                         #
#     * Redistributions in binary form must reproduce the above copyright #
#       notice, this list of conditions and the following disclaimer in   #
#       the documentation and/or other materials provided with the        #
#       distribution.                                                     #
#                                                                         #
#     * Neither the name of UChicago Argonne, LLC, Argonne National       #
#       Laboratory, ANL, the U.S. Government, nor the names of its        #
#       contributors may be used to endorse or promote products derived   #
#       from this software without specific prior written permission.     #
#                                                                         #
# THIS SOFTWARE IS PROVIDED BY UChicago Argonne, LLC AND CONTRIBUTORS     #
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT       #
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS       #
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL UChicago     #
# Argonne, LLC OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,        #
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,    #
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;        #
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER        #
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT      #
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN       #
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE         #
# POSSIBILITY OF SUCH DAMAGE.                                             #
# #########################################################################


"""


Grating interferometry
----------------------


This library contain the function to analyse data from grating
interferometry experiments.

There are several different layouts for a grating interferometry experiments,
where one could use: one dimensional, two-dimensional (checked board) or
circular gratings; phase or absorption gratings; and, in experimetns with more
than one grating, we can have combination of different gratings.

For this reason, it is very difficult to write a function that covers all the
possibilities and (at least initally) we need a function for each particular
case.




"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import wavepy.utils as wpu
import wavepy.surface_from_grad as wps
from skimage.restoration import unwrap_phase


try:
    from  pyfftw.interfaces.numpy_fft import fft2, ifft2
except ImportError:
    from  numpy.fft import fft2, ifft2


__authors__ = "Walan Grizolli"
__copyright__ = "Copyright (c) 2016-2017, Argonne National Laboratory"
__version__ = "0.1.0"
__docformat__ = "restructuredtext en"
__all__ = ['exp_harm_period', 'extract_harmonic',
           'plot_harmonic_grid', 'plot_harmonic_peak',
           'single_grating_harmonic_images', 'single_2Dgrating_analyses',
           'visib_1st_harmonics']


def _idxPeak_ij(harV, harH, nRows, nColumns, periodVert, periodHor):
    """
    Calculates the theoretical indexes of the harmonic peak
    [`harV`, `harH`] in the main FFT image
    """
    return [nRows // 2 + harV * periodVert, nColumns // 2 + harH * periodHor]


def _idxPeak_ij_exp(imgFFT, harV, harH, periodVert, periodHor, searchRegion):
    """
    Returns the index of the maximum intensity in a harmonic sub image.
    """

    intensity = (np.abs(imgFFT))

    (nRows, nColumns) = imgFFT.shape

    idxPeak_ij = _idxPeak_ij(harV, harH, nRows, nColumns,
                             periodVert, periodHor)

    maskSearchRegion = np.zeros((nRows, nColumns))

    maskSearchRegion[idxPeak_ij[0] - searchRegion:
                     idxPeak_ij[0] + searchRegion,
                     idxPeak_ij[1] - searchRegion:
                     idxPeak_ij[1] + searchRegion] = 1.0

    idxPeak_ij_exp = np.where(intensity * maskSearchRegion ==
                              np.max(intensity * maskSearchRegion))

    return [idxPeak_ij_exp[0][0], idxPeak_ij_exp[1][0]]


def _check_harmonic_inside_image(harV, harH, nRows, nColumns,
                                 periodVert, periodHor):
    """
    Check if full harmonic image is within the main image
    """

    errFlag = False

    if (harV + .5)*periodVert > nRows / 2:
        wpu.print_red("ATTENTION: Harmonic Peak " +
                      "{:d}{:d}".format(harV, harH) +
                      " is out of image vertical range.")
        errFlag = True

    if (harH + .5)*periodHor > nColumns / 2:
        wpu.print_red("ATTENTION: Harmonic Peak " +
                      "{:d}{:d} is ".format(harV, harH) +
                      "is out of image horizontal range.")
        errFlag = True

    if errFlag:
        raise ValueError("ERROR: Harmonic Peak " +
                         "{:d}{:d} is ".format(harV, harH) +
                         "out of image frequency range.")


def _error_harmonic_peak(imgFFT, harV, harH,
                         periodVert, periodHor, searchRegion=10):
    """
    Error in pixels (in the reciprocal space) between the harmonic peak and
    the provided theoretical value
    """

    #  Estimate harmonic positions

    idxPeak_ij = _idxPeak_ij(harV, harH, imgFFT.shape[0], imgFFT.shape[1],
                             periodVert, periodHor)

    idxPeak_ij_exp = _idxPeak_ij_exp(imgFFT, harV, harH,
                                     periodVert, periodHor, searchRegion)

    del_i = idxPeak_ij_exp[0] - idxPeak_ij[0]
    del_j = idxPeak_ij_exp[1] - idxPeak_ij[1]

    return del_i, del_j


def exp_harm_period(img, harmonicPeriod,
                    harmonic_ij='00', searchRegion=10,
                    isFFT=False, verbose=True):
    """
    Function to obtain the position (in pixels) in the reciprocal space
    of the first harmonic ().
    """

    (nRows, nColumns) = img.shape

    harV = int(harmonic_ij[0])
    harH = int(harmonic_ij[1])

    periodVert = harmonicPeriod[0]
    periodHor = harmonicPeriod[1]

    # adjusts for 1D grating
    if periodVert <= 0 or periodVert is None:
        periodVert = nRows
        if verbose:
            wpu.print_blue("MESSAGE: Assuming Horizontal 1D Grating")

    if periodHor <= 0 or periodHor is None:
        periodHor = nColumns
        if verbose:
            wpu.print_blue("MESSAGE: Assuming Vertical 1D Grating")

    #    _check_harmonic_inside_image(harV, harH, nRows, nColumns,
    #                                 periodVert, periodHor)

    if isFFT:
        imgFFT = img
    else:
        imgFFT = np.fft.fftshift(fft2(img, norm='ortho'))

    del_i, del_j = _error_harmonic_peak(imgFFT, harV, harH,
                                        periodVert, periodHor,
                                        searchRegion)

    if verbose:
        wpu.print_blue("MESSAGE: error experimental harmonics " +
                       "vertical: {:d}".format(del_i))
        wpu.print_blue("MESSAGE: error experimental harmonics " +
                       "horizontal: {:d}".format(del_j))

    return periodVert + del_i, periodHor + del_j


def extract_harmonic(img, harmonicPeriod,
                     harmonic_ij='00', searchRegion=10, isFFT=False,
                     plotFlag=False, verbose=True):

    """
    Function to extract one harmonic image of the FFT of single grating
    Talbot imaging.


    The function use the provided value of period to search for the harmonics
    peak. The search is done in a rectangle of size
    ``periodVert*periodHor/searchRegion**2``. The final result is a rectagle of
    size ``periodVert x periodHor`` centered at
    ``(harmonic_Vertical*periodVert x harmonic_Horizontal*periodHor)``


    Parameters
    ----------

    img : 	ndarray – Data (data_exchange format)
        Experimental image, whith proper blank image, crop and rotation already
        applied.

    harmonicPeriod : list of integers in the format [periodVert, periodHor]
        ``periodVert`` and ``periodVert`` are the period of the harmonics in
        the reciprocal space in pixels. For the checked board grating,
        periodVert = sqrt(2) * pixel Size / grating Period * number of
        rows in the image. For 1D grating, set one of the values to negative or
        zero (it will set the period to number of rows or colunms).

    harmonic_ij : string or list of string
        string with the harmonic to extract, for instance '00', '01', '10'
        or '11'. In this notation negative harmonics are not allowed.

        Alternativelly, it accepts a list of string
        ``harmonic_ij=[harmonic_Vertical, harmonic_Horizontal]``, for instance
        ``harmonic_ij=['0', '-1']``

        Note that since the original image contain only real numbers (not
        complex), then negative and positive harmonics are symetric
        related to zero.
    isFFT : Boolean
        Flag that tells if the input image ``img`` is in the reciprocal
        (``isFFT=True``) or in the real space (``isFFT=False``)

    searchRegion: int
        search for the peak will be in a region of harmonicPeriod/searchRegion
        around the theoretical peak position

    plotFlag: Boolean
        Flag to plot the image in the reciprocal space and to show the position
        of the found peaked and the limits of the harmonic image

    verbose: Boolean
        verbose flag.


    Returns
    -------
    2D ndarray
        Copped Images of the harmonics ij


    This functions crops a rectagle of size ``periodVert x periodHor`` centered
    at ``(harmonic_Vertical*periodVert x harmonic_Horizontal*periodHor)`` from
    the provided FFT image.


    Note
    ----
        * Note that it is the FFT of the image that is required.
        * The search for the peak is only used to print warning messages.

    **Q: Why not the real image??**

    **A:** Because FFT can be time consuming. If we use the real image, it will
    be necessary to run FFT for each harmonic. It is encourage to wrap this
    function within a function that do the FFT, extract the harmonics, and
    return the real space harmonic image.


    See Also
    --------
    :py:func:`wavepy.grating_interferometry.plot_harmonic_grid`

    """

    (nRows, nColumns) = img.shape

    harV = int(harmonic_ij[0])
    harH = int(harmonic_ij[1])

    periodVert = harmonicPeriod[0]
    periodHor = harmonicPeriod[1]

    if verbose:
            wpu.print_blue("MESSAGE: Extracting harmonic " +
                           harmonic_ij[0] + harmonic_ij[1])
            wpu.print_blue("MESSAGE: Harmonic period " +
                           "Horizontal: {:d} pixels".format(periodHor))
            wpu.print_blue("MESSAGE: Harmonic period " +
                           "Vertical: {:d} pixels".format(periodVert))

    # adjusts for 1D grating
    if periodVert <= 0 or periodVert is None:
        periodVert = nRows
        if verbose:
            wpu.print_blue("MESSAGE: Assuming Horizontal 1D Grating")

    if periodHor <= 0 or periodHor is None:
        periodHor = nColumns
        if verbose:
            wpu.print_blue("MESSAGE: Assuming Vertical 1D Grating")

    try:
        _check_harmonic_inside_image(harV, harH, nRows, nColumns,
                                     periodVert, periodHor)
    except ValueError:
        raise SystemExit

    if isFFT:
        imgFFT = img
    else:
        imgFFT = np.fft.fftshift(fft2(img, norm='ortho'))

    intensity = (np.abs(imgFFT))

    #  Estimate harmonic positions
    idxPeak_ij = _idxPeak_ij(harV, harH, nRows, nColumns,
                             periodVert, periodHor)

    del_i, del_j = _error_harmonic_peak(imgFFT, harV, harH,
                                        periodVert, periodHor,
                                        searchRegion)

    if verbose:
        print("MESSAGE: extract_harmonic:" +
              " harmonic peak " + harmonic_ij[0] + harmonic_ij[1] +
              " is misplaced by:")
        print("MESSAGE: {:d} pixels in vertical, {:d} pixels in hor".format(
               del_i, del_j))

        print("MESSAGE: Theoretical peak index: {:d},{:d} [VxH]".format(
              idxPeak_ij[0], idxPeak_ij[1]))

    if ((np.abs(del_i) > searchRegion // 2) or
       (np.abs(del_j) > searchRegion // 2)):

        wpu.print_red("ATTENTION: Harmonic Peak " + harmonic_ij[0] +
                      harmonic_ij[1] + " is too far from theoretical value.")
        wpu.print_red("ATTENTION: {:d} pixels in vertical,".format(del_i) +
                      "{:d} pixels in hor".format(del_j))

    if plotFlag:

        from matplotlib.patches import Rectangle
        plt.figure(figsize=(8, 7))
        plt.imshow(np.log10(intensity), cmap='inferno', extent=wpu.extent_func(intensity))

        plt.xlabel('Pixels')
        plt.ylabel('Pixels')

        xo = idxPeak_ij[1] - nColumns//2 - periodHor//2
        yo = nRows//2 - idxPeak_ij[0] - periodVert//2
        # xo yo are the lower left position of the reangle

        plt.gca().add_patch(Rectangle((xo, yo),
                                      periodHor, periodVert,
                                      lw=2, ls='--', color='red',
                                      fill=None, alpha=1))

        plt.title('Selected Region ' + harmonic_ij[0] + harmonic_ij[1],
                  fontsize=18, weight='bold')
        plt.show(block=False)

    return imgFFT[idxPeak_ij[0] - periodVert//2:
                  idxPeak_ij[0] + periodVert//2,
                  idxPeak_ij[1] - periodHor//2:
                  idxPeak_ij[1] + periodHor//2]


def plot_harmonic_grid(img, harmonicPeriod=None, isFFT=False):
    """
    Takes the FFT of single 2D grating Talbot imaging and plot the grid from
    where we extract the harmonic in a image of the



    Parameters
    ----------
    img : 	ndarray – Data (data_exchange format)
        Experimental image, whith proper blank image, crop and rotation already
        applied.

    harmonicPeriod : integer or list of integers
        If list, it must be in the format ``[periodVert, periodHor]``. If
        integer, then [periodVert = periodHor``.
        ``periodVert`` and ``periodVert`` are the period of the harmonics in
        the reciprocal space in pixels. For the checked board grating,
        ``periodVert = sqrt(2) * pixel Size / grating Period * number of
        rows in the image``

    isFFT : Boolean
        Flag that tells if the input image ``img`` is in the reciprocal
        (``isFFT=True``) or in the real space (``isFFT=False``)

    """

    if not isFFT:
        imgFFT = np.fft.fftshift(fft2(np.fft.fftshift(img),
                                             norm='ortho'))
    else:
        imgFFT = img

    (nRows, nColumns) = imgFFT.shape

    periodVert = harmonicPeriod[0]
    periodHor = harmonicPeriod[1]

    # adjusts for 1D grating
    if periodVert <= 0 or periodVert is None:
        periodVert = nRows

    if periodHor <= 0 or periodHor is None:
        periodHor = nColumns

    plt.figure(figsize=(8, 7))
    plt.imshow(np.log10(np.abs(imgFFT)), cmap='inferno',
               extent=wpu.extent_func(imgFFT))

    plt.xlabel('Pixels')
    plt.ylabel('Pixels')

    harV_min = -(nRows + 1) // 2 // periodVert
    harV_max = (nRows + 1) // 2 // periodVert

    harH_min = -(nColumns + 1) // 2 // periodHor
    harH_max = (nColumns + 1) // 2 // periodHor

    for harV in range(harV_min + 1, harV_max + 2):

        idxPeak_ij = _idxPeak_ij(harV, 0, nRows, nColumns,
                                 periodVert, periodHor)

        plt.axhline(idxPeak_ij[0] - periodVert//2 - nRows//2,
                    lw=2, color='r')

    for harH in range(harH_min + 1, harH_max + 2):

        idxPeak_ij = _idxPeak_ij(0, harH, nRows, nColumns,
                                 periodVert, periodHor)
        plt.axvline(idxPeak_ij[1] - periodHor // 2 - nColumns//2,
                    lw=2, color='r')

    for harV in range(harV_min, harV_max + 1):
        for harH in range(harH_min, harH_max + 1):

            idxPeak_ij = _idxPeak_ij(harV, harH,
                                     nRows, nColumns,
                                     periodVert, periodHor)

            plt.plot(idxPeak_ij[1] - nColumns//2,
                     idxPeak_ij[0] - nRows//2,
                    'ko', mew=2, mfc="None", ms=15)

            plt.annotate('{:d}{:d}'.format(-harV, harH),
                         (idxPeak_ij[1] - nColumns//2,
                          idxPeak_ij[0] - nRows//2,),
                         color='red', fontsize=20)

    plt.xlim(-nColumns//2, nColumns - nColumns//2)
    plt.ylim(-nRows//2, nRows - nRows//2)
    plt.title('log scale FFT magnitude, Hamonics Subsets and Indexes',
              fontsize=16, weight='bold')


def plot_harmonic_peak(img, harmonicPeriod=None, isFFT=False, fname=None):
    """
    Funtion to plot the profile of the harmonic peaks ``10`` and ``01``.
    It is ploted 11 profiles of the 11 nearest vertical (horizontal)
    lines to the peak ``01`` (``10``)

    img : 	ndarray – Data (data_exchange format)
        Experimental image, whith proper blank image, crop and rotation already
        applied.

    harmonicPeriod : integer or list of integers
        If list, it must be in the format ``[periodVert, periodHor]``. If
        integer, then [periodVert = periodHor``.
        ``periodVert`` and ``periodVert`` are the period of the harmonics in
        the reciprocal space in pixels. For the checked board grating,
        ``periodVert = sqrt(2) * pixel Size / grating Period * number of
        rows in the image``

    isFFT : Boolean
        Flag that tells if the input image ``img`` is in the reciprocal
        (``isFFT=True``) or in the real space (``isFFT=False``)
    """

    if not isFFT:
        imgFFT = np.fft.fftshift(fft2(np.fft.fftshift(img),
                                             norm='ortho'))
    else:
        imgFFT = img

    (nRows, nColumns) = img.shape

    periodVert = harmonicPeriod[0]
    periodHor = harmonicPeriod[1]

    # adjusts for 1D grating
    if periodVert <= 0 or periodVert is None:
        periodVert = nRows

    if periodHor <= 0 or periodHor is None:
        periodHor = nColumns

    fig = plt.figure(figsize=(8, 7))

    ax1 = fig.add_subplot(121)

    ax2 = fig.add_subplot(122)

    idxPeak_ij = _idxPeak_ij(0, 1,
                             nRows, nColumns,
                             periodVert, periodHor)

    for i in range(-5, 5):

        ax1.plot(np.abs(imgFFT[idxPeak_ij[0] - 100:idxPeak_ij[0] + 100,
                               idxPeak_ij[1]-i]),
                 lw=2, label='01 Vert ' + str(i))

    ax1.grid()

    idxPeak_ij = _idxPeak_ij(1, 0,
                             nRows, nColumns,
                             periodVert, periodHor)

    for i in range(-5, 5):

        ax2.plot(np.abs(imgFFT[idxPeak_ij[0]-i,
                               idxPeak_ij[1] - 100:idxPeak_ij[1] + 100]),
                 lw=2, label='10 Horz ' + str(i))

    ax2.grid()

    ax1.set_xlabel('Pixels')
    ax1.set_ylabel(r'$| FFT |$ ')
    ax1.legend(loc=1, fontsize='xx-small')
    ax1.title.set_text('Horz')


    ax2.set_xlabel('Pixels')
    ax2.set_ylabel(r'$| FFT |$ ')
    ax2.legend(loc=1, fontsize='xx-small')
    ax2.title.set_text('Vert')
    plt.show(block=False)

    if fname is not None:
        plt.savefig(fname, transparent=True)


def single_grating_harmonic_images(img, harmonicPeriod,
                                   searchRegion=10,
                                   plotFlag=False, verbose=False):
    """
    Auxiliary function to process the data of single 2D grating Talbot imaging.
    It obtain the (real space) harmonic images  00, 01 and 10.

    Parameters
    ----------
    img : 	ndarray – Data (data_exchange format)
        Experimental image, whith proper blank image, crop and rotation already
        applied.

    harmonicPeriod : list of integers in the format [periodVert, periodHor]
        ``periodVert`` and ``periodVert`` are the period of the harmonics in
        the reciprocal space in pixels. For the checked board grating,
        periodVert = sqrt(2) * pixel Size / grating Period * number of
        rows in the image. For 1D grating, set one of the values to negative or
        zero (it will set the period to number of rows or colunms).

    searchRegion: int
        search for the peak will be in a region of harmonicPeriod/searchRegion
        around the theoretical peak position. See also
        `:py:func:`wavepy.grating_interferometry.plot_harmonic_grid`

    plotFlag: boolean
        Flag to plot the image in the reciprocal space and to show the position
        of the found peaked and the limits of the harmonic image

    verbose: Boolean
        verbose flag.

    Returns
    -------
    three 2D ndarray data
        Images obtained from the harmonics 00, 01 and 10.

    """

    imgFFT = np.fft.fftshift(fft2(img, norm='ortho'))

    if plotFlag:
        plot_harmonic_grid(imgFFT, harmonicPeriod=harmonicPeriod, isFFT=True)
        plt.show(block=False)

    imgFFT00 = extract_harmonic(imgFFT,
                                harmonicPeriod=harmonicPeriod,
                                harmonic_ij='00',
                                searchRegion=searchRegion,
                                isFFT=True,
                                plotFlag=plotFlag,
                                verbose=verbose)

    imgFFT01 = extract_harmonic(imgFFT,
                                harmonicPeriod=harmonicPeriod,
                                harmonic_ij=['0', '1'],
                                searchRegion=searchRegion,
                                isFFT=True,
                                plotFlag=plotFlag,
                                verbose=verbose)

    imgFFT10 = extract_harmonic(imgFFT,
                                harmonicPeriod=harmonicPeriod,
                                harmonic_ij=['1', '0'],
                                searchRegion=searchRegion,
                                isFFT=True,
                                plotFlag=plotFlag,
                                verbose=verbose)

    #  Plot Fourier image (intensity)
    if plotFlag:

        # Intensity is Fourier Space
        intFFT00 = np.log10(np.abs(imgFFT00))
        intFFT01 = np.log10(np.abs(imgFFT01))
        intFFT10 = np.log10(np.abs(imgFFT10))

        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(14, 5))

        for dat, ax, textTitle in zip([intFFT00, intFFT01, intFFT10],
                                      axes.flat,
                                      ['FFT 00', 'FFT 01', 'FFT 10']):

            # The vmin and vmax arguments specify the color limits
            im = ax.imshow(dat, cmap='inferno', vmin=np.min(intFFT00),
                           vmax=np.max(intFFT00),
                           extent=wpu.extent_func(dat))

            ax.set_title(textTitle)
            if textTitle == 'FFT 00':
                ax.set_ylabel('Pixels')

            ax.set_xlabel('Pixels')

        # Make an axis for the colorbar on the right side
        cax = fig.add_axes([0.92, 0.1, 0.03, 0.8])
        fig.colorbar(im, cax=cax)
        plt.suptitle('FFT subsets - Intensity', fontsize=18, weight='bold')
        plt.show(block=False)

    img00 = ifft2(np.fft.ifftshift(imgFFT00), norm='ortho')

    # non existing harmonics will return NAN, so here we check NAN
    if np.all(np.isfinite(imgFFT01)):
        img01 = ifft2(np.fft.ifftshift(imgFFT01), norm='ortho')
    else:
        img01 = imgFFT01

    if np.all(np.isfinite(imgFFT10)):
        img10 = ifft2(np.fft.ifftshift(imgFFT10), norm='ortho')
    else:
        img10 = imgFFT10

    return (img00, img01, img10)


def single_2Dgrating_analyses(img, img_ref=None, harmonicPeriod=None,
                              unwrapFlag=True, plotFlag=True, verbose=False):
    """
    Function to process the data of single 2D grating Talbot imaging. It
    wraps other functions in order to make all the process transparent

    """

    # Obtain Harmonic images
    h_img = single_grating_harmonic_images(img, harmonicPeriod,
                                           plotFlag=plotFlag,
                                           verbose=verbose)

    if img_ref is not None:  # relative wavefront

        h_img_ref = single_grating_harmonic_images(img_ref, harmonicPeriod,
                                                   plotFlag=plotFlag,
                                                   verbose=verbose)

        int00 = np.abs(h_img[0])/np.abs(h_img_ref[0])
        int01 = np.abs(h_img[1])/np.abs(h_img_ref[1])
        int10 = np.abs(h_img[2])/np.abs(h_img_ref[2])

        if unwrapFlag is True:

            arg01 = (unwrap_phase(np.angle(h_img[1]), seed=72673) -
                     unwrap_phase(np.angle(h_img_ref[1]), seed=72673))

            arg10 = (unwrap_phase(np.angle(h_img[2]), seed=72673) -
                     unwrap_phase(np.angle(h_img_ref[2]), seed=72673))

        else:
            arg01 = np.angle(h_img[1]) - np.angle(h_img_ref[1])
            arg10 = np.angle(h_img[2]) - np.angle(h_img_ref[2])

    else:  # absolute wavefront

        int00 = np.abs(h_img[0])
        int01 = np.abs(h_img[1])
        int10 = np.abs(h_img[2])

        if unwrapFlag is True:

            arg01 = unwrap_phase(np.angle(h_img[1]), seed=72673)
            arg10 = unwrap_phase(np.angle(h_img[2]), seed=72673)
        else:
            arg01 = np.angle(h_img[1])
            arg10 = np.angle(h_img[2])

    if unwrapFlag is True:  # remove pi jump
        arg01 -= int(np.round(np.mean(arg01/np.pi)))*np.pi
        arg10 -= int(np.round(np.mean(arg10/np.pi)))*np.pi

    darkField01 = int01/int00
    darkField10 = int10/int00

    return [int00, int01, int10,
            darkField01, darkField10,
            arg01, arg10]


def visib_1st_harmonics(img, harmonicPeriod, searchRegion=20,
                        unFilterSize=1, verbose=False):
    """
    This function obtain the visibility in a grating imaging experiment by the
    ratio of the amplitudes of the first and zero harmonics. See
    https://doi.org/10.1364/OE.22.014041 .

    Note
    ----
    Note that the absolute visibility also depends on the higher harmonics, and
    for a absolute value of visibility all of them must be considered.


    Parameters
    ----------
    img : 	ndarray – Data (data_exchange format)
        Experimental image, whith proper blank image, crop and rotation already
        applied.

    harmonicPeriod : list of integers in the format [periodVert, periodHor]
        ``periodVert`` and ``periodVert`` are the period of the harmonics in
        the reciprocal space in pixels. For the checked board grating,
        periodVert = sqrt(2) * pixel Size / grating Period * number of
        rows in the image. For 1D grating, set one of the values to negative or
        zero (it will set the period to number of rows or colunms).

    searchRegion: int
        search for the peak will be in a region of harmonicPeriod/searchRegion
        around the theoretical peak position. See also
        `:py:func:`wavepy.grating_interferometry.plot_harmonic_grid`

    verbose: Boolean
        verbose flag.


    Returns
    -------
    (float, float)
        horizontal and vertical visibilities respectivelly from
        harmonics 01 and 10


    """

    imgFFT = np.fft.fftshift(fft2(img, norm='ortho'))

    _idxPeak_ij_exp00 = _idxPeak_ij_exp(imgFFT, 0, 0,
                                        harmonicPeriod[0], harmonicPeriod[1],
                                        searchRegion)

    _idxPeak_ij_exp10 = _idxPeak_ij_exp(imgFFT, 1, 0,
                                        harmonicPeriod[0], harmonicPeriod[1],
                                        searchRegion)

    _idxPeak_ij_exp01 = _idxPeak_ij_exp(imgFFT, 0, 1,
                                        harmonicPeriod[0], harmonicPeriod[1],
                                        searchRegion)


    from scipy.ndimage.filters import uniform_filter

    arg_imgFFT = np.abs(imgFFT)

    if unFilterSize > 1:
        arg_imgFFT = uniform_filter(arg_imgFFT, unFilterSize)

    peak00 = arg_imgFFT[_idxPeak_ij_exp00[0], _idxPeak_ij_exp00[1]]
    peak10 = arg_imgFFT[_idxPeak_ij_exp10[0], _idxPeak_ij_exp10[1]]
    peak01 = arg_imgFFT[_idxPeak_ij_exp01[0], _idxPeak_ij_exp01[1]]

    return (2*peak10/peak00, 2*peak01/peak00, _idxPeak_ij_exp00, _idxPeak_ij_exp10, _idxPeak_ij_exp01)


def plot_intensities_harms(int00, int01, int10,
                           pixelsize, titleStr,
                           saveFigFlag=False, saveFileSuf='graph'):
    # Plot Real image (intensity)

    if titleStr is not '':
        titleStr = ', ' + titleStr

    factor, unit_xy = wpu.choose_unit(np.sqrt(int00.size)*pixelsize[0])

    plt.figure(figsize=(14, 6))

    plt.subplot(131)
    plt.imshow(int00, cmap='viridis',
               vmax=wpu.mean_plus_n_sigma(int00, 4),
               extent=wpu.extent_func(int00, pixelsize)*factor)
    plt.xlabel(r'$[{0} m]$'.format(unit_xy))
    plt.ylabel(r'$[{0} m]$'.format(unit_xy))
    plt.colorbar(shrink=0.5)
    plt.title('00', fontsize=18, weight='bold')

    plt.subplot(132)
    plt.imshow(int01, cmap='viridis',
               vmax=wpu.mean_plus_n_sigma(int01, 4),
               extent=wpu.extent_func(int01, pixelsize)*factor)
    plt.xlabel(r'$[{0} m]$'.format(unit_xy))
    plt.ylabel(r'$[{0} m]$'.format(unit_xy))
    plt.colorbar(shrink=0.5)
    plt.title('01', fontsize=18, weight='bold')

    plt.subplot(133)
    plt.imshow(int10, cmap='viridis',
               vmax=wpu.mean_plus_n_sigma(int10, 4),
               extent=wpu.extent_func(int10, pixelsize)*factor)
    plt.xlabel(r'$[{0} m]$'.format(unit_xy))
    plt.ylabel(r'$[{0} m]$'.format(unit_xy))
    plt.colorbar(shrink=0.5)
    plt.title('10', fontsize=18, weight='bold')

    plt.suptitle('Absorption obtained from the Harmonics' + titleStr,
                 fontsize=18, weight='bold')

    plt.tight_layout(rect=[0, 0, 1, 1])
    if saveFigFlag:
        wpu.save_figs_with_idx(saveFileSuf)
    plt.show(block=False)


def plot_dark_field(darkField01, darkField10,
                    pixelsize, titleStr='',
                    saveFigFlag=False, saveFileSuf='graph'):
    '''
    TODO: Write Docstring

    Plot Dark field

    '''

    if titleStr is not '':
        titleStr = ', ' + titleStr

    factor, unit_xy = wpu.choose_unit(np.sqrt(darkField01.size)*pixelsize[0])

    plt.figure(figsize=(14, 6))

    plt.subplot(121)
    plt.imshow(darkField01, cmap='viridis',
               vmax=wpu.mean_plus_n_sigma(darkField01, 4),
               extent=wpu.extent_func(darkField01, pixelsize)*factor)
    plt.xlabel(r'$[{0} m]$'.format(unit_xy))
    plt.ylabel(r'$[{0} m]$'.format(unit_xy))
    plt.colorbar(shrink=0.5)
    plt.title('Horizontal', fontsize=18, weight='bold')  # 01

    plt.subplot(122)
    plt.imshow(darkField10, cmap='viridis',
               vmax=wpu.mean_plus_n_sigma(darkField01, 4),
               extent=wpu.extent_func(darkField10, pixelsize)*factor)
    plt.xlabel(r'$[{0} m]$'.format(unit_xy))
    plt.ylabel(r'$[{0} m]$'.format(unit_xy))
    plt.colorbar(shrink=0.5)
    plt.title('Vertical', fontsize=18, weight='bold')  # 10

    plt.suptitle('Dark Field', fontsize=18, weight='bold')

    plt.tight_layout(rect=[0, 0, 1, 1])
    if saveFigFlag:
        wpu.save_figs_with_idx(saveFileSuf)
    plt.show(block=False)


def plot_DPC(dpc01, dpc10,
             pixelsize, titleStr='',
             saveFigFlag=False, saveFileSuf='graph'):
    '''
    TODO: Write Docstring
    Plot differencial phase signal
    '''
    if titleStr is not '':
        titleStr = ', ' + titleStr

    factor, unit_xy = wpu.choose_unit(np.sqrt(dpc01.size)*pixelsize[0])

    dpc01_plot = dpc01*pixelsize[1]/np.pi
    dpc10_plot = dpc10*pixelsize[0]/np.pi

    vlim01 = np.max((np.abs(wpu.mean_plus_n_sigma(dpc01_plot, -5)),
                     np.abs(wpu.mean_plus_n_sigma(dpc01_plot, 5))))
    vlim10 = np.max((np.abs(wpu.mean_plus_n_sigma(dpc10_plot, -5)),
                     np.abs(wpu.mean_plus_n_sigma(dpc10_plot, 5))))

    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.imshow(dpc01_plot, cmap='RdGy_r',
               vmin=-vlim01, vmax=vlim01,
               extent=wpu.extent_func(dpc01_plot, pixelsize)*factor)

    plt.xlabel(r'$[{0} m]$'.format(unit_xy))
    plt.ylabel(r'$[{0} m]$'.format(unit_xy))
    plt.colorbar(shrink=0.5)
    plt.title('DPC - Horizontal', fontsize=18, weight='bold')  # 01

    plt.subplot(122)
    plt.imshow(dpc10_plot, cmap='RdGy_r',
               vmin=-vlim10, vmax=vlim10,
               extent=wpu.extent_func(dpc10_plot, pixelsize)*factor)
    plt.xlabel(r'$[{0} m]$'.format(unit_xy))
    plt.ylabel(r'$[{0} m]$'.format(unit_xy))
    plt.colorbar(shrink=0.5)
    plt.title('DPC - Vertical', fontsize=18,
              weight='bold')

    plt.suptitle('Differential Phase ' + r'[$\pi$ rad]' + titleStr,
                 fontsize=18, weight='bold')

    plt.tight_layout(rect=[0, 0, 1, 1])
    if saveFigFlag:
        wpu.save_figs_with_idx(saveFileSuf)
    plt.show(block=False)


def dpc_integration(dpc01, dpc10, pixelsize, idx4crop=[0, -1, 0, -1],
                    plotErrorIntegration=False,
                    saveFileSuf=None,
                    shifthalfpixel=False, method='FC'):
    '''
    TODO: Write Docstring

    Integration of DPC to obtain phase. Currently only supports
    Frankot Chellappa
    '''

    if idx4crop == '':

        vmin = wpu.mean_plus_n_sigma(dpc01**2+dpc10**2, -3)
        vmax = wpu.mean_plus_n_sigma(dpc01**2+dpc10**2, 3)

        _, idx = wpu.crop_graphic_image(dpc01**2+dpc10**2,
                                        kargs4graph={'cmap': 'viridis',
                                                     'vmin': vmin,
                                                     'vmax': vmax})
    else:
        idx = idx4crop

    dpc01 = wpu.crop_matrix_at_indexes(dpc01, idx)
    dpc10 = wpu.crop_matrix_at_indexes(dpc10, idx)

    if method == 'FC':

        phase = wps.frankotchellappa(dpc01*pixelsize[1],
                                     dpc10*pixelsize[0],
                                     reflec_pad=True)
        phase = np.real(phase)

    else:
        wpu.print_red('ERROR: Unknown integration method: ' + method)

    if plotErrorIntegration:
        wps.error_integration(dpc01*pixelsize[1],
                              dpc10*pixelsize[0],
                              phase, pixelsize, errors=False,
                              shifthalfpixel=shifthalfpixel, plot_flag=True)

        if saveFileSuf is not None:
            wpu.save_figs_with_idx(saveFileSuf)

    return phase

# %%
def plot_integration(integrated, pixelsize,
                     titleStr='Title', ctitle=' ',
                     max3d_grid_points=101,
                     plotProfile=True,
                     plot3dFlag=True,
                     saveFigFlag=False,
                     saveFileSuf='graph',
                     **kwarg4surf):
    '''
    TODO: Write Docstring
    '''

    xxGrid, yyGrid = wpu.grid_coord(integrated, pixelsize)

    factor_x, unit_x = wpu.choose_unit(xxGrid)
    factor_y, unit_y = wpu.choose_unit(yyGrid)

    # Plot Integration 2

    if plot3dFlag:

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        rstride = integrated.shape[0] // max3d_grid_points + 1
        cstride = integrated.shape[1] // max3d_grid_points + 1

        surf = ax.plot_surface(xxGrid*factor_x, yyGrid*factor_y,
                               integrated[::-1, :],
                               rstride=rstride,
                               cstride=cstride,
                               cmap='viridis', linewidth=0.1, **kwarg4surf)

        ax_lim = np.max([np.abs(xxGrid*factor_x), np.abs(yyGrid*factor_y)])
        ax.set_xlim3d(-ax_lim, ax_lim)
        ax.set_ylim3d(-ax_lim, ax_lim)

        if 'vmin' in kwarg4surf:
            ax.set_zlim3d(bottom=kwarg4surf['vmin'])
        if 'vmax' in kwarg4surf:
            ax.set_zlim3d(top=kwarg4surf['vmax'])

        plt.xlabel(r'$x [' + unit_x + ' m]$', fontsize=24)
        plt.ylabel(r'$y [' + unit_y + ' m]$', fontsize=24)

        plt.title(titleStr, fontsize=24, weight='bold')
        cbar = plt.colorbar(surf, shrink=.8, aspect=20)
        cbar.ax.set_title(ctitle, y=1.01)

        plt.tight_layout(rect=[0, 0, 1, 1])

        ax.text2D(0.05, 0.9, 'strides = {}, {}'.format(rstride, cstride),
                    transform=ax.transAxes)

        if saveFigFlag:
            ax.view_init(elev=30, azim=60)
            wpu.save_figs_with_idx(saveFileSuf)
            ax.view_init(elev=30, azim=-120)
            wpu.save_figs_with_idx(saveFileSuf)
            plt.pause(.5)

        plt.show(block=False)

    if plotProfile:
        wpu.plot_profile(xxGrid*factor_x, yyGrid*factor_y,
                         integrated[::-1, :],
                         xlabel=r'$x [' + unit_x + ' m]$',
                         ylabel=r'$y [' + unit_y + ' m]$',
                         title=titleStr,
                         xunit='\mu m', yunit='\mu m',
                         arg4main={'cmap': 'viridis', 'lw': 3})

    if saveFigFlag:
        plt.ioff()
        plt.figure(figsize=(10, 8))

        plt.imshow(integrated[::-1, :], cmap='viridis',
                   extent=wpu.extent_func(integrated, pixelsize)*factor_x,
                   **kwarg4surf)

        plt.xlabel(r'$x [' + unit_x + ' m]$', fontsize=24)
        plt.ylabel(r'$y [' + unit_x + ' m]$', fontsize=24)

        plt.title(titleStr, fontsize=18, weight='bold')
        cbar = plt.colorbar()
        cbar.ax.set_title(ctitle, y=1.01)
        wpu.save_figs_with_idx(saveFileSuf)
        plt.close()
        plt.ion()



