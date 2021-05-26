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
Example of speckle tracking data analyses
"""

import numpy as np
from scipy import constants
import dxchange
import h5py as h5

import wavepy.utils as wpu
import wavepy.speckletracking as wps


__authors__ = "Walan Grizolli"
__copyright__ = "Copyright (c) 2016, Affiliation"
__version__ = "0.1.0"


# =============================================================================
# %% preamble. Load parameters from ini file
# =============================================================================


inifname = '.speckleAnalyses.ini'

config, ini_pars, ini_file_list = wpu.load_ini_file_terminal_dialog(inifname)

fname = ini_file_list.get('image_filename')
image = dxchange.read_tiff(fname)
image_ref = dxchange.read_tiff(ini_file_list.get('ref_filename'))

idx = list(map(int, ini_pars.get('crop').split(',')))
pixelsize = float(ini_pars.get('pixel size'))
phenergy = float(ini_pars.get('photon energy'))
distDet2sample = float(ini_pars.get('distance detector to sample'))
halfsubwidth = int(ini_pars.get('halfsubwidth'))
halfTemplateSize = int(ini_pars.get('halfTemplateSize'))
subpixelResolution = int(ini_pars.get('subpixelResolution'))
npointsmax = int(ini_pars.get('npointsmax'))
ncores = float(ini_pars.get('ncores')) / float(ini_pars.get('ncores of machine'))
saveH5 = ini_pars.get('save hdf5 files')

if subpixelResolution < 1: subpixelResolution = None
if halfTemplateSize < 1: halfTemplateSize = None



# %%
#
#dummy = wpu.dummy_images('Shapes',  shape=(110, 110), noise = 1)
#fname = 'dummy.tif'
#image = dummy[5:-5,5:-5]
#image_ref = dummy[7:-3,4:-6]

# =============================================================================
# %% parameters
# =============================================================================

rad2deg = np.rad2deg(1)
deg2rad = np.deg2rad(1)
NAN = float('Nan')  # not a number alias

hc = constants.value('inverse meter-electron volt relationship')  # hc

wavelength = hc/phenergy
kwave = 2*np.pi/wavelength


# =============================================================================
# %% Crop
# =============================================================================


#image = np.rot90(image)  # rotate images, good for sanity checks
#image_ref = np.rot90(image_ref)


kb_input = input('\nGraphic Crop? [N/y] : ')
if kb_input.lower() == 'y':
    # Graphical Crop

    idx = wpu.graphical_roi_idx(image, verbose=True)

    print('New idx:')
    print(idx)

    ini_pars['crop'] = str('{0}, {1}, {2}, {3}'.format(idx[0], idx[1], idx[2], idx[3]))
    with open(inifname, 'w') as configfile:  # update values in the ini file
        config.write(configfile)




image = wpu.crop_matrix_at_indexes(image, idx)
image_ref = wpu.crop_matrix_at_indexes(image_ref, idx)


# %%
# =============================================================================
# Displacement
# =============================================================================

sx, sy, \
error, step = wps.speckleDisplacement(image, image_ref,
                                      halfsubwidth=halfsubwidth,
                                      halfTemplateSize=halfTemplateSize,
                                      subpixelResolution=subpixelResolution,
                                      npointsmax=npointsmax,
                                      ncores=ncores, taskPerCore=15,
                                      verbose=True)




totalS = np.sqrt(sx**2 + sy**2)

xVec2 = wpu.realcoordvec(sx.shape[1], pixelsize*step)
yVec2 = wpu.realcoordvec(sx.shape[0], pixelsize*step)


# %%
# =============================================================================
# Save data in hdf5 format
# =============================================================================

fname_output = fname[:-4] + '_' + wpu.datetime_now_str() + ".h5"
f = h5.File(fname_output, "w")


h5rawdata = f.create_group('raw')
f.create_dataset("raw/image_sample", data=image)
f.create_dataset("raw/image_ref", data=image_ref)
h5rawdata.attrs['Pixel Size Detector [m]'] = pixelsize
h5rawdata.attrs['Distance Detector to Sample [m]'] = distDet2sample
h5rawdata.attrs['Photon Energy [eV]'] = phenergy

h5displacement = f.create_group('displacement')
f.create_dataset("displacement/displacement_x", data=sx)
f.create_dataset("displacement/displacement_y", data=sy)
f.create_dataset("displacement/error", data=error)
f.create_dataset("displacement/xvec", data=xVec2)
f.create_dataset("displacement/yvec", data=yVec2)



h5displacement.attrs['Comments'] = 'Created by Walan Grizolli at ' + wpu.datetime_now_str()
h5displacement.attrs['Pixel Size Processed images [m]'] = pixelsize*step
h5displacement.attrs['Distance Detector to Sample [m]'] = distDet2sample
h5displacement.attrs['Photon Energy [eV]'] = phenergy
h5displacement.attrs['ini file'] = '\n' + open(inifname, 'r').read()


f.flush()
f.close()

with open(fname_output[:-3] + '.log', 'w') as logfile:  # save ini files as log
    config.write(logfile)

wpu.print_blue("File saved at:\n{0}".format(fname_output))
