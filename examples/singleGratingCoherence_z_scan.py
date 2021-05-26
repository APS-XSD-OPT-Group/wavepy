#!/usr/bin/env python
# -*- coding: utf-8 -*-  #
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

# %%
'''
Author: Walan Grizolli


This Script use the technique described by Xianbo Shi in
https://doi.org/10.1364/OE.22.014041

'''

import dxchange
import numpy as np

plotFourierImages = False

if plotFourierImages:
    import matplotlib
    matplotlib.use('Agg')

import matplotlib.pyplot as plt
import glob
import wavepy.utils as wpu
import wavepy.grating_interferometry as wgi

#from multiprocessing import Pool, cpu_count
#import time

import os
from wavepy.utils import easyqt

# =============================================================================
# %% Function for multiprocessing
# =============================================================================

def _func(i):

    wpu.print_blue("MESSAGE: loop " + str(i) + ": " +
                   listOfDataFiles[i])

    img = dxchange.read_tiff(listOfDataFiles[i])

    darkMeanValue = np.mean(wpu.crop_matrix_at_indexes(img, idx4cropDark))

    #TODO xshi, need to add option of input one value


    img = img - darkMeanValue  # calculate and remove dark
    img = wpu.crop_matrix_at_indexes(img, idx4crop)

    pv = int(period_harm_Vert /
             (sourceDistanceV + zvec[i])*(sourceDistanceV+np.min(zvec)))
    ph = int(period_harm_Horz /
             (sourceDistanceH + zvec[i])*(sourceDistanceH+np.min(zvec)))

    if plotFourierImages:

        wgi.plot_harmonic_grid(img,
                               [pv, ph],
                               isFFT=False)

        plt.savefig('FFT_{:.0f}mm.png'.format(zvec[i]*1e3))
        plt.show(block=False)
        plt.close()

        wgi.plot_harmonic_peak(img,
                               [pv, ph],
                               isFFT=False)

        plt.savefig('FFT_peaks_{:.0f}mm.png'.format(zvec[i]*1e3))
        plt.show(block=False)
        plt.close()

    return wgi.visib_1st_harmonics(img, [pv, ph],
                                   searchRegion=searchRegion,
                                   unFilterSize=unFilterSize)

# =============================================================================
# %% Script starts
# =============================================================================

wpu._mpl_settings_4_nice_graphs()

# =============================================================================
# %% Load Image
# =============================================================================

originalDir = os.getcwd()

samplefileName = easyqt.get_file_names("Choose one of the scan files")[0]

data_dir = samplefileName.rsplit('/', 1)[0]
os.chdir(data_dir)

try:
    os.mkdir(data_dir + '/output/')
except:
    pass

fname2save = data_dir + '/output/' + samplefileName.rsplit('_', 1)[0].rsplit('/', 1)[1]

wpu.print_blue('MESSAGE: Loading files ' +
               samplefileName.rsplit('_', 1)[0] + '*.tif')

listOfDataFiles = glob.glob(samplefileName.rsplit('_', 2)[0] + '*.tif')
listOfDataFiles.sort()
nfiles = len(listOfDataFiles)

zvec_from = easyqt.get_choice(message='z distances is calculated or from table?',
                              title='Title',
                              choices=['Calculated', 'Tabled'])

# %%

if zvec_from == 'Calculated':

    startDist = easyqt.get_float('Starting distance scan [mm]',
                                 title='Title',
                                 default_value=10)*1e-3

    step_z_scan = easyqt.get_float('Step size scan [mm]',
                                   title='Title',
                                   default_value=2)*1e-3

    image_per_point = easyqt.get_int('Number of images by step',
                                     title='Title',
                                     default_value=1)

    zvec = np.linspace(startDist,
                       startDist + step_z_scan*(nfiles/image_per_point-1),
                       int(nfiles/image_per_point))
    zvec = zvec.repeat(image_per_point)

    strideFile = easyqt.get_int('Stride (Use only every XX files)',
                                title='Title',
                                default_value=1)
    listOfDataFiles = listOfDataFiles[0::strideFile]
    zvec = zvec[0::strideFile]

elif zvec_from == 'Tabled':

    zvec = np.loadtxt(easyqt.get_file_names("Table with the z distance values in mm")[0])*1e-3
    step_z_scan = np.mean(np.diff(zvec))

if step_z_scan > 0:
    pass
else:
    listOfDataFiles = listOfDataFiles[::-1]
    zvec = zvec[::-1]

img = dxchange.read_tiff(listOfDataFiles[0])

# =============================================================================
# %% Experimental parameters
# =============================================================================

pixelSize = easyqt.get_float("Enter Pixel Size [um]",
                             title='Experimental Values',
                             default_value=.6500, decimals=5)*1e-6

gratingPeriod = easyqt.get_float("Enter CB Grating Period [um]",
                                 title='Experimental Values',
                                 default_value=4.8)*1e-6


pattern = easyqt.get_choice(message='Select CB Grating Pattern',
                            title='Title',
 #                           choices=['Diagonal', 'Edge'])
                            choices=['Edge', 'Diagonal'])

sourceDistanceV = easyqt.get_float("Enter Distance to Source\n in the VERTICAL [m]",
                                   title='Experimental Values',
                                   default_value=34.0)

sourceDistanceH = easyqt.get_float("Enter Distance to Source\n in the Horizontal [m]",
                                   title='Experimental Values',
                                   default_value=34.0)

unFilterSize = easyqt.get_int("Enter Size for Uniform Filter [Pixels]\n" +
                              "    (Enter 1 to NOT use the filter)",
                              title='Experimental Values',
                              default_value=1)

searchRegion = easyqt.get_int("Enter Size of Region for Searching\n the Peak [in Pixels]",
                              title='Experimental Values',
                              default_value=20)

os.chdir(originalDir)


# =============================================================================
# %% Crop
# =============================================================================

idx4crop = [0, -1, 0, -1]

[colorlimit,
 cmap] = wpu.plot_slide_colorbar(img,
                                 title='SELECT COLOR SCALE,\n' +
                                 'Raw Image, No Crop',
                                 xlabel=r'x [$\mu m$ ]',
                                 ylabel=r'y [$\mu m$ ]',
                                 extent=wpu.extent_func(img,
                                                        pixelSize)*1e6)

idx4crop = wpu.graphical_roi_idx(img, verbose=True,
                                 kargs4graph={'cmap': cmap,
                                              'vmin': colorlimit[0],
                                              'vmax': colorlimit[1]})

wpu.print_blue("MESSAGE: idx for cropping")
wpu.print_blue(idx4crop)

img = wpu.crop_matrix_at_indexes(img, idx4crop)

# =============================================================================
# %% Dark indexes
# =============================================================================


darkRegionSelctionFlag = easyqt.get_yes_or_no('Do you want to select ' +
                                              'region for dark calculation?\n' +
                                              'Press ESC to use [0, 100, 0, 100]')

if darkRegionSelctionFlag:

    idx4cropDark = wpu.graphical_roi_idx(img, verbose=True,
                                         kargs4graph={'cmap': cmap,
                                                      'vmin': colorlimit[0],
                                                      'vmax': colorlimit[1]})
else:
    idx4cropDark = [0, 100, 0, 100]

# dark_im = dxchange.read_tiff(listOfDataFiles[0])*0.0 + avgDark

# ==============================================================================
# %% Harmonic Periods
# ==============================================================================

if pattern == 'Diagonal':
    period_harm_Vert = np.int(np.sqrt(2)*pixelSize/gratingPeriod*img.shape[0])
    period_harm_Horz = np.int(np.sqrt(2)*pixelSize/gratingPeriod*img.shape[1])
elif pattern == 'Edge':
    period_harm_Vert = np.int(2*pixelSize/gratingPeriod*img.shape[0])
    period_harm_Horz = np.int(2*pixelSize/gratingPeriod*img.shape[1])

# Obtain harmonic periods from images

(period_harm_Vert,
 _) = wgi.exp_harm_period(img, [period_harm_Vert, period_harm_Horz],
                          harmonic_ij=['1', '0'],
                          searchRegion=40,
                          isFFT=False, verbose=True)

(_,
 period_harm_Horz) = wgi.exp_harm_period(img, [period_harm_Vert,
                                         period_harm_Horz],
                                         harmonic_ij=['0', '1'],
                                         searchRegion=40,
                                         isFFT=False, verbose=True)

wpu.log_this('Input files: ' + samplefileName.rsplit('_', 1)[0] + '*.tif',
             preffname=fname2save)
wpu.log_this('\nNumber of files : ' + str(nfiles))
wpu.log_this('Stride : ' + str(strideFile))
wpu.log_this('Z distances is ' + zvec_from)

if zvec_from == 'Calculated':
    wpu.log_this('Step zscan [mm] : {:.4g}'.format(step_z_scan*1e3))
    wpu.log_this('Start point zscan [mm] : {:.4g}'.format(startDist*1e3))

wpu.log_this('Pixel Size [um] : {:.4g}'.format(pixelSize*1e6))
wpu.log_this('Grating Period [um] : {:.4g}'.format(gratingPeriod*1e6))
wpu.log_this('Grating Pattern : ' + pattern)
wpu.log_this('Crop idxs : ' + str(idx4crop))
wpu.log_this('Dark idxs : ' + str(idx4cropDark))

wpu.log_this('Vertical Source Distance: ' + str(sourceDistanceV))
wpu.log_this('Horizontal Source Distance: ' + str(sourceDistanceH))

wpu.log_this('Uniform Filter Size : {:d}'.format(unFilterSize))

wpu.log_this('Search Region : {:d}'.format(searchRegion))


# =============================================================================
# %% Calculate everything
# =============================================================================


# =============================================================================
# %% multiprocessing
# =============================================================================
'''
ncpus = cpu_count()

wpu.print_blue("MESSAGE: %d cpu's available" % ncpus)

tzero = time.time()

p = Pool(ncpus-1)
res = p.map(_func, range(len(listOfDataFiles)))
p.close()

wpu.print_blue('MESSAGE: Time spent: {0:.3f} s'.format(time.time() - tzero))
'''

res = []
for i in range(len(listOfDataFiles)):
    res.append(_func(i))

# =============================================================================
# %% Sorting the data
# =============================================================================


contrastV = np.asarray([x[0] for x in res])
contrastH = np.asarray([x[1] for x in res])

p0 = np.asarray([x[2] for x in res])
pv = np.asarray([x[3] for x in res])
ph = np.asarray([x[4] for x in res])

pattern_period_Vert_z = pixelSize/(pv[:, 0] - p0[:, 0])*img.shape[0]
pattern_period_Horz_z = pixelSize/(ph[:, 1] - p0[:, 1])*img.shape[1]

# =============================================================================
# %% Save csv file
# =============================================================================

outputfname = wpu.get_unique_filename(fname2save, 'csv')

wpu.save_csv_file(np.c_[zvec.T,
                        contrastV.T,
                        contrastH.T,
                        pattern_period_Vert_z.T,
                        pattern_period_Horz_z.T],
                  outputfname,
                  headerList=['z [m]',
                              'Vert Contrast',
                              'Horz Contrast',
                              'Vert Period [m]',
                              'Horz Period [m]'])


wpu.log_this('\nOutput file: ' + outputfname)

# =============================================================================
# %% Plot
# =============================================================================

# contrast vs z
fig = plt.figure(figsize=(10, 7))
plt.plot(zvec*1e3, contrastV*100, '-ko', label='Vert')
plt.plot(zvec*1e3, contrastH*100, '-ro', label='Hor')
plt.xlabel(r'Distance $z$  [mm]', fontsize=14)


plt.ylabel(r'Visibility $\times$ 100 [%]', fontsize=14)
plt.title('Visibility vs detector distance', fontsize=14, weight='bold')

plt.legend(fontsize=14, loc=0)


wpu.save_figs_with_idx(fname2save)
plt.show(block=False)

# =============================================================================
# %% Plot Harmonic position and calculate source distance
# =============================================================================
from fit_singleGratingCoherence_z_scan import fit_period_vs_z

(sourceDistance_from_fit_V,
 patternPeriodFromData_V) = fit_period_vs_z(zvec, pattern_period_Vert_z,
                                            contrastV,
                                            direction='Vertical',
                                            threshold=.005,
                                            fname4graphs=fname2save)

(sourceDistance_from_fit_H,
 patternPeriodFromData_H) = fit_period_vs_z(zvec, pattern_period_Horz_z,
                                            contrastH,
                                            direction='Horizontal',
                                            threshold=0.005,
                                            fname4graphs=fname2save)
