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

# %%% imports cell
import numpy as np
import matplotlib.pyplot as plt

import dxchange


import wavepy.utils as wpu
import wavepy.grating_interferometry as wgi
import wavepy.surface_from_grad as wps


# %% Experimental values


pixelsize = [0.65e-6, 0.65e-6]  # vertical and horizontal pixel sizes in meters
distDet2sample = 0.18600  # in meters
sourceDistance = 100.0  # in meters, for divergence correction. to ignore it, use a big number >100

phenergy = 8e3  # in eV
wavelength = wpu.hc/phenergy  # wpu has an alias for hc
kwave = 2*np.pi/wavelength


# Phase grating paremeters
gratingPeriod = 4.8e-6  # in meters
# uncomment proper pattern period:
patternPeriod = gratingPeriod/np.sqrt(2.0)  # if half Pi grating
#patternPeriod = gratingPeriod/2.0  # if Pi grating

img = dxchange.read_tiff('data_example_for_single_grating/cb4p8um_halfPi_8KeV_10s_img.tif')
imgRef = dxchange.read_tiff('data_example_for_single_grating/cb4p8um_halfPi_8KeV_10s_ref.tif')
darkImg = dxchange.read_tiff('data_example_for_single_grating/10s_dark.tif')

img = img - darkImg
imgRef = imgRef - darkImg

# %% crop

img, idx4crop = wpu.crop_graphic_image(img)
imgRef = wpu.crop_matrix_at_indexes(imgRef, idx4crop)

# %% Find harmonic in the Fourier images

# calculate the theoretical position of the hamonics
period_harm_Vert_o = np.int(pixelsize[0]/patternPeriod*img.shape[0] /
                            (sourceDistance + distDet2sample)*sourceDistance)
period_harm_Hor_o = np.int(pixelsize[1]/patternPeriod*img.shape[1] /
                           (sourceDistance + distDet2sample)*sourceDistance)


# Obtain harmonic periods from images

wpu.print_blue('MESSAGE: Obtain harmonic 01 exprimentally')

(_,
 period_harm_Hor) = wgi.exp_harm_period(imgRef, [period_harm_Vert_o,
                                        period_harm_Hor_o],
                                        harmonic_ij=['0', '1'])

wpu.print_blue('MESSAGE: Obtain harmonic 10 exprimentally')

(period_harm_Vert,
 _) = wgi.exp_harm_period(imgRef, [period_harm_Vert_o,
                          period_harm_Hor_o],
                          harmonic_ij=['1', '0'])

harmPeriod = [period_harm_Vert, period_harm_Hor]

# %% Obtain DPC and Dark Field image

[int00, int01, int10,
 darkField01, darkField10,
 alpha_x,
 alpha_y] = wgi.single_2Dgrating_analyses(img, img_ref=imgRef,
                                          harmonicPeriod=harmPeriod,
                                          plotFlag=True,
                                          unwrapFlag=True,
                                          verbose=True)

# the spatial resolution is the grating period, what we call the virtual pixel size
virtual_pixelsize = [0, 0]
virtual_pixelsize[0] = pixelsize[0]*img.shape[0]/int00.shape[0]
virtual_pixelsize[1] = pixelsize[1]*img.shape[1]/int00.shape[1]

# covert phaseFFT to physical quantities, ie, differential phase
diffPhase01 = -alpha_x*virtual_pixelsize[1]/distDet2sample/wavelength
diffPhase10 = -alpha_y*virtual_pixelsize[0]/distDet2sample/wavelength


# %% post processing and plot

saveFigFlag = True
saveFileSuf = 'results'

wgi.plot_intensities_harms(int00, int01, int10,
                           virtual_pixelsize, saveFigFlag=saveFigFlag,
                           titleStr='Intensity',
                           saveFileSuf=saveFileSuf)

wgi.plot_dark_field(darkField01, darkField10,
                    virtual_pixelsize, saveFigFlag=saveFigFlag,
                    saveFileSuf=saveFileSuf)

wgi.plot_DPC(diffPhase01, diffPhase10,
             virtual_pixelsize, saveFigFlag=saveFigFlag,
             saveFileSuf=saveFileSuf)

plt.show(block=True)

# %% crop again before integration if you wish

_, idx4crop = wpu.crop_graphic_image(np.sqrt(diffPhase01**2 + diffPhase10**2))
diffPhase01 = wpu.crop_matrix_at_indexes(diffPhase01, idx4crop)
diffPhase10 = wpu.crop_matrix_at_indexes(diffPhase10, idx4crop)

# %%

phase = wgi.dpc_integration(diffPhase01, diffPhase10, virtual_pixelsize)

#phase = np.real(phase)

delta, _ = wpu.get_delta(phenergy, choice_idx=1, gui_mode=False)

wgi.plot_integration(-(phase - np.min(phase))/kwave/delta*1e6,
                     virtual_pixelsize,
                     titleStr=r'Thickness Beryllium $[\mu m]$,' +'\n Frankot Chellappa integration',
                     plot3dFlag=True,
                     saveFigFlag=True,
                     saveFileSuf=saveFileSuf)

wps.error_integration(diffPhase01*virtual_pixelsize[1],
                      diffPhase10*virtual_pixelsize[0],
                      phase, virtual_pixelsize, errors=False,
                      plot_flag=True)

wpu.save_figs_with_idx(saveFileSuf)

# %%

def _pad_2_make_square(array, mode='edge'):

    diff_shape = array.shape[0] - array.shape[1]

    if diff_shape > 1:
        return np.pad(array, ((0, 0), (0, diff_shape)), mode=mode)

    elif diff_shape < 1:
        return np.pad(array, ((0, -diff_shape), (0, 0)), mode=mode)


