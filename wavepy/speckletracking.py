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
Functions for speckle tracking analises
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import itertools
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from skimage.feature import register_translation

from multiprocessing import Pool, cpu_count

import wavepy.utils as wpu


__authors__ = "Walan Grizolli"
__copyright__ = "Copyright (c) 2016-2017, Argonne National Laboratory"
__version__ = "0.1.0"
__docformat__ = "restructuredtext en"
__all__ = ['speckleDisplacement']


def _speckleDisplacementSingleCore_method1(image, image_ref, halfsubwidth,
                                           subpixelResolution, stride, verbose):
    '''
    see http://scikit-image.org/docs/dev/auto_examples/transform/plot_register_translation.html
    '''

    irange = np.arange(halfsubwidth,
                       image.shape[0] - halfsubwidth + 1,
                       stride)
    jrange = np.arange(halfsubwidth,
                       image.shape[1] - halfsubwidth + 1,
                       stride)

    pbar = tqdm(total=np.size(irange))  # progress bar

    sx = np.ones(image.shape) * NAN
    sy = np.ones(image.shape) * NAN
    error = np.ones(image.shape) * NAN

    for (i, j) in itertools.product(irange, jrange):

        interrogation_window = image_ref[i - halfsubwidth:i + halfsubwidth + 1,
                               j - halfsubwidth:j + halfsubwidth + 1]

        sub_image = image[i - halfsubwidth:i + halfsubwidth + 1,
                    j - halfsubwidth:j + halfsubwidth + 1]

        shift, error_ij, _ = register_translation(sub_image,
                                                  interrogation_window,
                                                  subpixelResolution)

        sx[i, j] = shift[1]
        sy[i, j] = shift[0]
        error[i, j] = error_ij

        if j == jrange[-1]: pbar.update()  # update progress bar

    print(" ")

    return (sx[halfsubwidth:-halfsubwidth:stride,
            halfsubwidth:-halfsubwidth:stride],
            sy[halfsubwidth:-halfsubwidth:stride,
            halfsubwidth:-halfsubwidth:stride],
            error[halfsubwidth:-halfsubwidth:stride,
            halfsubwidth:-halfsubwidth:stride],
            stride)

def _speckleDisplacementSingleCore_method2(image, image_ref, halfsubwidth,
                                           halfTemplateSize, stride, verbose):
    '''
    see http://scikit-image.org/docs/dev/auto_examples/plot_template.html
    '''

    from skimage.feature import match_template

    irange = np.arange(halfsubwidth,
                       image.shape[0] - halfsubwidth + 1,
                       stride)
    jrange = np.arange(halfsubwidth,
                       image.shape[1] - halfsubwidth + 1,
                       stride)

    pbar = tqdm(total=np.size(irange))  # progress bar

    sx = np.ones(image.shape) * NAN
    sy = np.ones(image.shape) * NAN
    error = np.ones(image.shape) * NAN

    for (i, j) in itertools.product(irange, jrange):

        interrogation_window = image_ref[i - halfTemplateSize: \
                                         i + halfTemplateSize+ 1,
                                         j - halfTemplateSize: \
                                         j + halfTemplateSize + 1]

        sub_image = image[i - halfsubwidth:i + halfsubwidth + 1,
                    j - halfsubwidth:j + halfsubwidth + 1]

        result = match_template(sub_image, interrogation_window)


        shift_y, shift_x = np.unravel_index(np.argmax(result), result.shape)

        shift_x -= halfsubwidth - halfTemplateSize
        shift_y -= halfsubwidth - halfTemplateSize
        error_ij = 1.0 - np.max(result)

        sx[i, j] = shift_x
        sy[i, j] = shift_y
        error[i, j] = error_ij

        if j == jrange[-1]: pbar.update()  # update progress bar

    print(" ")

    return (sx[halfsubwidth:-halfsubwidth:stride,
            halfsubwidth:-halfsubwidth:stride],
            sy[halfsubwidth:-halfsubwidth:stride,
            halfsubwidth:-halfsubwidth:stride],
            error[halfsubwidth:-halfsubwidth:stride,
            halfsubwidth:-halfsubwidth:stride],
            stride)

def _speckleDisplacementSingleCore(image, image_ref,
                                   stride,
                                   halfsubwidth,
                                   halfTemplateSize,
                                   subpixelResolution,
                                   verbose):

    print('MESSAGE: _speckleDisplacementSingleCore:')
    print("MESSAGE: Usind 1 core")

    if subpixelResolution is not None:
        if verbose: print('MESSAGE: register_translation method.')
        return _speckleDisplacementSingleCore_method2(image, image_ref,
                                                      halfsubwidth,
                                                      subpixelResolution,
                                                      stride, verbose)

    elif halfTemplateSize is not None:
        if verbose: print('MESSAGE: match_template method.')
        return _speckleDisplacementSingleCore_method2(image, image_ref,
                                                      halfsubwidth,
                                                      halfTemplateSize,
                                                      stride, verbose)


# ==============================================================================
# % Data Analysis Multicore
# ==============================================================================



def _func_4_starmap_async_method1(args, parList):
    '''
    see http://scikit-image.org/docs/dev/auto_examples/transform/plot_register_translation.html
    '''
    i = args[0]
    j = args[1]
    image = parList[0]
    image_ref = parList[1]
    halfsubwidth = parList[2]
    subpixelResolution = parList[3]


    interrogation_window = image_ref[i - halfsubwidth:i + halfsubwidth + 1,
                           j - halfsubwidth:j + halfsubwidth + 1]

    sub_image = image[i - halfsubwidth:i + halfsubwidth + 1,
                j - halfsubwidth:j + halfsubwidth + 1]

    shift, error_ij, _ = register_translation(sub_image,
                                              interrogation_window,
                                              subpixelResolution)

    return shift[1], shift[0], error_ij

def _func_4_starmap_async_method2(args, parList):
    '''
    see http://scikit-image.org/docs/dev/auto_examples/plot_template.html
    '''

    from skimage.feature import match_template


    i = args[0]
    j = args[1]
    image = parList[0]
    image_ref = parList[1]
    halfsubwidth = parList[2]
    halfTempSize = parList[3]


    #    halfTempSize = halfsubwidth // 4


    interrogation_window = image_ref[i - halfTempSize:i + halfTempSize + 1,
                           j - halfTempSize:j + halfTempSize + 1]

    sub_image = image[i - halfsubwidth:i + halfsubwidth + 1,
                j - halfsubwidth:j + halfsubwidth + 1]

    result = match_template(sub_image, interrogation_window)


    shift_y, shift_x = np.unravel_index(np.argmax(result), result.shape)

    shift_x -= halfsubwidth - halfTempSize
    shift_y -= halfsubwidth - halfTempSize


#    sub_image_at_interrogation = image_ref[i - halfTempSize:i + halfTempSize + 1,
#                                           j - halfTempSize:j + halfTempSize + 1]
     #    error_ij = 1 -  np.max(result)**2 /np.sum(interrogation_window**2)/np.sum(sub_image_at_interrogation**2)

    error_ij = 1.0 - np.max(result)

    return shift_x, shift_y, error_ij

def _speckleDisplacementMulticore(image, image_ref, stride,
                                  halfsubwidth, halfTemplateSize,
                                  subpixelResolution,
                                  ncores, taskPerCore, verbose):

    print('MESSAGE: _speckleDisplacementMulticore:')
    print("MESSAGE: %d cpu's available" % cpu_count())
    nprocesses = int(cpu_count() * ncores)
    p = Pool(processes=nprocesses)
    print("MESSAGE: Using %d cpu's" % p._processes)

    irange = np.arange(halfsubwidth, image.shape[0] - halfsubwidth + 1, stride)
    jrange = np.arange(halfsubwidth,image.shape[1] - halfsubwidth + 1, stride)



    ntasks = np.size(irange) * np.size(jrange)

    chunksize = ntasks // p._processes // taskPerCore + 1


    if subpixelResolution is not None:
        if verbose: print('MESSAGE: register_translation method.')
        parList = [image, image_ref, halfsubwidth, subpixelResolution]
        func_4_starmap_async = _func_4_starmap_async_method1

    elif halfTemplateSize is not None:
        if verbose: print('MESSAGE: match_template method.')
        parList = [image, image_ref, halfsubwidth, halfTemplateSize]
        func_4_starmap_async = _func_4_starmap_async_method2

    res = p.starmap_async(func_4_starmap_async,
                          zip(itertools.product(irange, jrange),
                              itertools.repeat(parList)),
                          chunksize=chunksize)

    p.close()  # No more work

    wpu.progress_bar4pmap(res)  # Holds the program in a loop waiting
                                 # starmap_async to finish

    sx = np.array(res.get())[:, 0].reshape(len(irange), len(jrange))
    sy = np.array(res.get())[:, 1].reshape(len(irange), len(jrange))
    error = np.array(res.get())[:, 2].reshape(len(irange), len(jrange))

    return (sx, sy, error, stride)


def speckleDisplacement(image, image_ref,
                        stride=1, npointsmax=None,
                        halfsubwidth=10, halfTemplateSize=None,
                        subpixelResolution=None,
                        ncores=1/2, taskPerCore=100,
                        verbose=False):
    '''
    This function track the movements of speckle in an image (with sample)
    related to a reference image (whith no sample). The function relies in two
    other functions that you are advised to check:  `register_translation <http://scikit-image.org/docs/dev/auto_examples/transform/plot_register_translation.html>`_ and
    see `match_template <http://scikit-image.org/docs/dev/auto_examples/plot_template.html>`_

    References
    ----------

    see http://scikit-image.org/docs/dev/auto_examples/transform/plot_register_translation.html

    see http://scikit-image.org/docs/dev/auto_examples/features_detection/plot_template.html
    '''


    if halfTemplateSize is None and subpixelResolution is None:
        raise SyntaxError('Either value of halfTemplateSize or' + \
                           ' subpixelResolution must be provided.')

    if halfTemplateSize is not None and subpixelResolution is not None:
        raise SyntaxError('wavepy: Either halfTemplateSize or' + \
                           ' subpixelResolution must be provided, but not both.')

    if npointsmax is not None:
        npoints = int((image.shape[0] - 2 * halfsubwidth) / stride)
        # DEBUG_print_var("npoints", npoints)
        if npoints > npointsmax:
            stride = int((image.shape[0] - 2 * halfsubwidth) / npointsmax)
            # DEBUG_print_var("stride", stride)
        if stride <= 0: stride = 1  # note that this is not very precise

    if verbose:
        print('MESSAGE: speckleDisplacement:')
        print("MESSAGE: stride =  %d" % stride)
        print("MESSAGE: npoints =  %d" %
                int((image.shape[0] - 2 * halfsubwidth) / stride))

    if ncores < 0 or ncores > 1: ncores = 1

    if int(cpu_count() * ncores) <= 1:

        res = _speckleDisplacementSingleCore(image, image_ref,
                                             stride=stride,
                                             halfsubwidth=halfsubwidth,
                                             halfTemplateSize=halfTemplateSize,
                                             subpixelResolution=subpixelResolution,
                                             verbose=verbose)




    else:

        res = _speckleDisplacementMulticore(image, image_ref,
                                            stride=stride,
                                            halfsubwidth=halfsubwidth,
                                            halfTemplateSize=halfTemplateSize,
                                            subpixelResolution=subpixelResolution,
                                            ncores=ncores, taskPerCore=taskPerCore,
                                            verbose=verbose)

    return res
