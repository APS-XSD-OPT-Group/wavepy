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

'''
Author: Walan Grizolli

'''

import numpy as np
import matplotlib.pyplot as plt
from skimage.restoration import unwrap_phase
import dxchange

import wavepy.utils as wpu
import wavepy.grating_interferometry as wgi

from wavepy.utils import easyqt as easyqt

from scipy.interpolate import splrep, splev, sproot
from scipy import constants


rad2deg = np.rad2deg(1)
deg2rad = np.deg2rad(1)
NAN = float('Nan')  # not a number alias
hc = constants.value('inverse meter-electron volt relationship')  # hc

#wpu._mpl_settings_4_nice_graphs()


def _extent_func(img, pixelsize):

    if isinstance(pixelsize, float):
        pixelsize = [pixelsize, pixelsize]

    return np.array((-img.shape[1]*pixelsize[1] / 2,
                     img.shape[1]*pixelsize[1] / 2,
                    -img.shape[0]*pixelsize[0] / 2,
                    img.shape[0]*pixelsize[0] / 2))


# %%
def plot_chi2(xvec, c_matrix_data, a_matrix, chi2):

    axiscount = 0
    xvecForGraph = np.linspace(np.min(xvec),
                               np.max(xvec), 101)

    plt.figure()
    hist_y, hist_x, _ = plt.hist(chi2[np.where(chi2 < 20*np.std(chi2))],
                                 100, log=False, label=r'$\chi^2$')

    peak_chi2 = hist_x[np.argmax(hist_y)]

    fwhm_chi2_1 = np.min(hist_x[np.where(hist_y > 0.5*np.max(hist_y))])
    fwhm_chi2_2 = np.max(hist_x[np.where(hist_y > 0.5*np.max(hist_y))])


    list_arg_chi2 = []

    for i in range(4):

        list_arg_chi2.append(np.argmin((hist_x - np.min(hist_x) -
                             (fwhm_chi2_1 - np.min(hist_x))/4*i)**2))

        list_arg_chi2.append(np.argmin((hist_x - fwhm_chi2_1 -
                             (peak_chi2 - fwhm_chi2_1)/4*i)**2))
        list_arg_chi2.append(np.argmin((hist_x - peak_chi2 -
                             (fwhm_chi2_2-peak_chi2)/4*i)**2))
        list_arg_chi2.append(np.argmin((hist_x - fwhm_chi2_2 -
                             (np.max(hist_x)-fwhm_chi2_2)/4*i)**2) - 1)

    list_arg_chi2.sort()

    plt.plot(hist_x[list_arg_chi2], hist_y[list_arg_chi2], 'or',
             label=r'values for fit plot')
    plt.grid()
    plt.legend()
    plt.show()

    ii = np.mgrid[0:chi2.shape[0]-1:16j].astype(int)

    f, axarr = plt.subplots(4, 4, figsize=(10, 8))

    for i in chi2.argsort()[ii]:

        ax = axarr.flatten()[axiscount]

        ax.plot(xvec, c_matrix_data[:, i], '-ko')

        ax.plot(xvecForGraph,
                a_matrix[0, i] +
                a_matrix[1, i]*np.sin(2*np.pi*xvecForGraph) +
                a_matrix[2, i]*np.cos(2*np.pi*xvecForGraph), '--r')

        ax.annotate(r'$\chi$ = {:.3g}'.format(chi2[i]),
                    xy=(.80, .80), xycoords='axes fraction',
                    xytext=(-20, 20), textcoords='offset pixels', fontsize=10,
                    bbox=dict(boxstyle="round", fc="0.9"))

        ax.grid()

        if axiscount >= 12:
            ax.set_xlabel('Grating Steps [gr period units]')

        if axiscount % 4 == 0:
            ax.set_ylabel('Counts')

        axiscount += 1

    plt.suptitle('Intensity in a single pixel with fit',
                 fontsize=16, weight='bold')
    plt.show(block=True)


# %%
def fit_stepping_grating(img_stack, gratingPeriod, stepSize, plotFits=True):

    nsteps, nlines, ncolums = img_stack.shape

    xg = np.linspace(0.0, (nsteps-1)*stepSize, nsteps)

    c_matrix_data = img_stack.reshape((nsteps, nlines*ncolums))

    bigB_matrix = np.zeros((nsteps, 3))
    bigB_matrix[:, 0] = 1.0
    bigB_matrix[:, 1] = np.sin(2*np.pi*xg/gratingPeriod)
    bigB_matrix[:, 2] = np.cos(2*np.pi*xg/gratingPeriod)

    bigG_matrix = np.dot(np.linalg.inv(np.dot(np.transpose(bigB_matrix),
                                              bigB_matrix)),
                         np.transpose(bigB_matrix))

    a_matrix = np.dot(bigG_matrix, c_matrix_data)
    c_matrix_model = np.dot(bigB_matrix, a_matrix)

    chi2 = 1 / (nsteps - 3 - 1) * np.sum((c_matrix_data - c_matrix_model)**2 /
                                         np.abs(c_matrix_data), axis=0)

    if plotFits:
        plot_chi2(xg/gratingPeriod, c_matrix_data, a_matrix, chi2)

    return (a_matrix.reshape((3, nlines, ncolums)),
            chi2.reshape((nlines, ncolums)))


# %%

def intial_setup():

    [list_sample_files,
     list_ref_files,
     list_dark_files] = wpu.gui_list_data_phase_stepping()

    for fname in list_sample_files + list_ref_files + list_dark_files:
        wpu.print_blue('MESSAGE: Loading ' + fname.rsplit('/')[-1])

    pixelSize = easyqt.get_float("Enter Pixel Size [um]",
                                 title='Experimental Values',
                                 default_value=.65)*1e-6

    stepSize = easyqt.get_float("Enter scan step size [um]",
                                title='Experimental Values',
                                default_value=.2)*1e-6

    return (list_sample_files, list_ref_files, list_dark_files,
            pixelSize, stepSize)


def files_to_array(list_sample_files, list_ref_files, list_dark_files,
                   idx4crop=[0, -1, 0, -1]):

    img = wpu.crop_matrix_at_indexes(dxchange.read_tiff(list_sample_files[0]),
                                     idx4crop)

    (nlines, ncolums) = img.shape

    img_stack = np.zeros((len(list_sample_files), nlines, ncolums))
    ref_stack = img_stack*0.0

    dark_im = img_stack[0, :, :]*0.0

    for i in range(len(list_dark_files)):

        dark_im += wpu.crop_matrix_at_indexes(dxchange.read_tiff(list_dark_files[i]),
                                              idx4crop)

    for i in range(len(list_sample_files)):

        img_stack[i, :, :] = wpu.crop_matrix_at_indexes(dxchange.read_tiff(list_sample_files[i]),
                                                  idx4crop) - dark_im


        ref_stack[i, :, :] = wpu.crop_matrix_at_indexes(dxchange.read_tiff(list_ref_files[i]),
                                                        idx4crop) - dark_im

    return img_stack, ref_stack


def period_estimation_spline(signal_one_pixel, stepSize):

    signal_one_pixel -= np.mean(signal_one_pixel)

    nsteps = np.size(signal_one_pixel)

    xg = np.mgrid[0:(nsteps-1)*stepSize:nsteps*1j]
    xg2 = np.mgrid[0:(nsteps-1)*stepSize:nsteps*10j]

    tck = splrep(xg, signal_one_pixel)
    y2 = splev(xg2, tck)

    estimated_period = np.mean(np.diff(sproot(tck)))*2

    plt.figure()
    plt.plot(xg*1e6, signal_one_pixel, '-o', xg2*1e6, y2, '--.')

    plt.annotate(r'period = {:.3} $\mu m$'.format(estimated_period*1e6),
                 xy=(.80, .90), xycoords='axes fraction',
                 xytext=(-20, 20), textcoords='offset pixels', fontsize=16,
                 bbox=dict(boxstyle="round", fc="0.9"))

    plt.legend(['data', 'spline'], loc=4)
    plt.xlabel(r'$\mu m$')
    plt.ylabel('Counts')
    plt.grid()
    plt.show(block=False)

    return estimated_period


def main_stepping_grating(img_stack, ref_stack, period_oscilation, stepSize):

    # fit sample stack
    a_matrix_sample, chi2_sample = fit_stepping_grating(img_stack[:, :, :],
                                                        period_oscilation,
                                                        stepSize,
                                                        plotFits=True)
    # fit ref stack
    a_matrix_ref, chi2_ref = fit_stepping_grating(ref_stack[:, :, :],
                                                  period_oscilation,
                                                  stepSize, plotFits=False)
    # Obtain physical proprerties and plot
    # Intensity
    intensity = a_matrix_sample[0]/a_matrix_ref[0]

    # Dark Field
    dk_field_sample = np.sqrt(a_matrix_sample[2, :]**2 +
                              a_matrix_sample[1, :]**2)/a_matrix_sample[0, :]

    dk_field_ref = np.sqrt(a_matrix_ref[2, :]**2 +
                           a_matrix_ref[1, :]**2)/a_matrix_ref[0, :]

    dk_field = dk_field_sample/dk_field_ref

    # DPC
    dpc_1d = np.arctan2(a_matrix_sample[2, :], a_matrix_sample[1, :]) - \
        np.arctan2(a_matrix_ref[2, :], a_matrix_ref[1, :])

    return intensity, dk_field, dpc_1d, chi2_ref

# %%
#from importlib import reload
#
#reload

# %%

if __name__ == '__main__':

    # ==========================================================================
    # Experimental parameters
    # ==========================================================================

    (list_sample_files, list_ref_files, list_dark_files,
     pixelSize, stepSize) = intial_setup()

    # ==========================================================================
    # % % Load one image and crop
    # ==========================================================================

    img = dxchange.read_tiff(list_sample_files[0])

    [colorlimit,
     cmap] = wpu.plot_slide_colorbar(img, title='Raw Image',
                                         xlabel=r'x [$\mu m$ ]',
                                         ylabel=r'y [$\mu m$ ]',
                                   extent=wpu.extent_func(img, pixelSize)*1e6)

    img_croped, idx4crop = wpu.crop_graphic(zmatrix=img, verbose=True,
                                            kargs4graph={'cmap': cmap,
                                                         'vmin': colorlimit[0],
                                                         'vmax': colorlimit[1]})


    # ==========================================================================
    # %% Load tiff files to numpy array
    # ==========================================================================

    img_stack, ref_stack = files_to_array(list_sample_files,
                                          list_ref_files,
                                          list_dark_files,
                                          idx4crop=idx4crop)

    nimages, nlines, ncolumns = ref_stack.shape

    # ==========================================================================
    # %% use data to determine grating period
    # ==========================================================================

    period_estimated = period_estimation_spline(ref_stack[:, nlines//4,
                                                          ncolumns//4],
                                                stepSize)

    period_estimated += period_estimation_spline(ref_stack[:, nlines//4,
                                                           3*ncolumns//4],
                                                 stepSize)

    period_estimated += period_estimation_spline(ref_stack[:, 3*nlines//4,
                                                           ncolumns//4],
                                                 stepSize)

    period_estimated += period_estimation_spline(ref_stack[:, 3*nlines//4,
                                                           3*ncolumns//4],
                                                 stepSize)

    period_estimated /= 4.0

    wpu.print_red('MESSAGE: Pattern Period from the ' +
                  'data: {:.4f}'.format(period_estimated*1e6))

    # ==========================================================================
    # %% do your thing
    # ==========================================================================

    (intensity,
     dk_field,
     dpc_1d,
     chi2) = main_stepping_grating(img_stack, ref_stack,
                                   period_estimated, stepSize)

    # %% Intensity

    wpu.plot_slide_colorbar(intensity,
                            title='Intensity',
                            xlabel=r'x [$\mu m$]',
                            ylabel=r'y [$\mu m$]',
                            extent=wpu.extent_func(dpc_1d, pixelSize)*1e6)

    # %% Dark Field

    wpu.plot_slide_colorbar(dk_field, title='Dark Field',
                            xlabel=r'x [$\mu m$]',
                            ylabel=r'y [$\mu m$]',
                            extent=wpu.extent_func(dpc_1d, pixelSize)*1e6)

    # %% DPC

    dpc_1d = unwrap_phase(dpc_1d)
    wpu.plot_slide_colorbar(dpc_1d/np.pi/2.0,
                            title=r'DPC [$\pi rad$]',
                            xlabel=r'x [$\mu m$]',
                            ylabel=r'y [$\mu m$]',
                            extent=wpu.extent_func(dpc_1d, pixelSize)*1e6)

    # %%
    xx, yy = wpu.realcoordmatrix(dpc_1d.shape[1], pixelSize,
                                 dpc_1d.shape[0], pixelSize)
    wpu.plot_profile(xx*1e3, yy*1e3, dpc_1d/np.pi/2.0,
                     xlabel='[mm]', ylabel='[mm]')

    # %% chi2

    plt.figure()
    hist = plt.hist(chi2[np.where(chi2 < 10*np.std(chi2))], 100, log=False)
    plt.title(r'$\chi^2$', fontsize=14, weight='bold')
    plt.show(block=False)

    chi2_copy = np.copy(chi2)

    wpu.plot_slide_colorbar(chi2_copy, title=r'$\chi^2$ sample',
                            xlabel=r'x [$\mu m$ ]',
                            ylabel=r'y [$\mu m$ ]',
                            extent=wpu.extent_func(chi2, pixelSize)*1e6)

    # %% mask by chi2

    dpc_1d[np.where(np.abs(dpc_1d) < 1*np.std(dpc_1d))] = 0.0

    masked_plot = dpc_1d*1.0

    masked_plot[np.where(chi2 > 50)] = 0.0

    wpu.plot_slide_colorbar(masked_plot, title='DPC masked',
                            xlabel=r'x [$\mu m$ ]',
                            ylabel=r'y [$\mu m$ ]',
                            extent=wpu.extent_func(masked_plot, pixelSize)*1e6)
