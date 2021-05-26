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
# %%
import sys
import os

if len(sys.argv) != 1: # if command line, dont show the plots
    import matplotlib
    matplotlib.use('Agg')

import numpy as np

import matplotlib.pyplot as plt

plt.rcParams.update({'figure.max_open_warning': 0})
from mpl_toolkits.mplot3d import Axes3D
import dxchange

import wavepy.utils as wpu
import wavepy.grating_interferometry as wgi

#import easygui_qt as easyqt
from wavepy.utils import easyqt as easyqt
# sys.path.append('/home/grizolli/workspace/pythonWorkspace/wgTools')

rad2deg = np.rad2deg(1)
deg2rad = np.deg2rad(1)
deg2rad = np.deg2rad(1)
hc = wpu.hc  # hc

wpu._mpl_settings_4_nice_graphs()


# %%
def main_single_gr_Talbot(img, imgRef,
                          phenergy, pixelsize, distDet2sample,
                          period_harm,
                          saveFileSuf,
                          unwrapFlag=True,
                          plotFlag=True,
                          saveFigFlag=False):

    img_size_o = np.shape(img)

    img, idx4crop = crop_dialog(img, saveFigFlag=saveFigFlag)
    if imgRef is not None:
        imgRef = wpu.crop_matrix_at_indexes(imgRef, idx4crop)

    # I dont know why in the past I had these lines: (WG 20180406)
    #    if imgRef is None:
    #        img, idx4crop = crop_dialog(img, saveFigFlag=saveFigFlag)
    #    else:
    #        imgRef, idx4crop = crop_dialog(imgRef, saveFigFlag=saveFigFlag)
    #        img = wpu.crop_matrix_at_indexes(img, idx4crop)
    # calculate harmonic position after crop

    period_harm_Vert_o = int(period_harm[0]*img.shape[0]/img_size_o[0]) + 1
    period_harm_Hor_o = int(period_harm[1]*img.shape[1]/img_size_o[1]) + 1

    # Obtain harmonic periods from images

    wpu.print_blue('MESSAGE: Obtain harmonic 01 exprimentally')

    if imgRef is None:

        harmPeriod = [period_harm_Vert_o, period_harm_Hor_o]

    else:

        (_,
         period_harm_Hor) = wgi.exp_harm_period(imgRef, [period_harm_Vert_o,
                                                period_harm_Hor_o],
                                                harmonic_ij=['0', '1'],
                                                searchRegion=60,
                                                isFFT=False, verbose=True)

        wpu.print_blue('MESSAGE: Obtain harmonic 10 exprimentally')

        (period_harm_Vert,
         _) = wgi.exp_harm_period(imgRef, [period_harm_Vert_o,
                                  period_harm_Hor_o],
                                  harmonic_ij=['1', '0'],
                                  searchRegion=60,
                                  isFFT=False, verbose=True)

        harmPeriod = [period_harm_Vert, period_harm_Hor]

    # Calculate everything

    [int00, int01, int10,
     darkField01, darkField10,
     phaseFFT_01,
     phaseFFT_10] = wgi.single_2Dgrating_analyses(img, img_ref=imgRef,
                                                  harmonicPeriod=harmPeriod,
                                                  plotFlag=plotFlag,
                                                  unwrapFlag=unwrapFlag,
                                                  verbose=True)

    virtual_pixelsize = [0, 0]
    virtual_pixelsize[0] = pixelsize[0]*img.shape[0]/int00.shape[0]
    virtual_pixelsize[1] = pixelsize[1]*img.shape[1]/int00.shape[1]

    diffPhase01 = -phaseFFT_01*virtual_pixelsize[1]/distDet2sample/hc*phenergy
    diffPhase10 = -phaseFFT_10*virtual_pixelsize[0]/distDet2sample/hc*phenergy
    # Note: the signals above were defined base in experimental data

    return [int00, int01, int10,
            darkField01, darkField10,
            diffPhase01, diffPhase10,
            virtual_pixelsize]


# %%
def crop_dialog(img, saveFigFlag=False):

    global inifname  # name of .ini file
    global gui_mode
    # take index from ini file
    idx4crop = list(map(int, (wpu.get_from_ini_file(inifname, 'Parameters',
                                                    'Crop').split(','))))

    wpu.print_red(idx4crop)

    # Plot Real Image wiht default crop

    tmpImage = wpu.crop_matrix_at_indexes(img, idx4crop)

    plt.figure()
    plt.imshow(tmpImage,
               cmap='viridis',
               extent=wpu.extent_func(tmpImage, pixelsize)*1e6)
    plt.xlabel(r'$[\mu m]$')
    plt.ylabel(r'$[\mu m]$')
    plt.colorbar()

    plt.title('Raw Image with initial Crop', fontsize=18, weight='bold')

    plt.show(block=False)
    plt.pause(.5)
    # ask if the crop need to be changed
    newCrop = gui_mode and easyqt.get_yes_or_no('New Crop?')

    if saveFigFlag and not newCrop:
        wpu.save_figs_with_idx(saveFileSuf)

    plt.close(plt.gcf())

    if newCrop:

        [colorlimit,
         cmap] = wpu.plot_slide_colorbar(img,
                                         title='SELECT COLOR SCALE,\n' +
                                         'Raw Image, No Crop',
                                         xlabel=r'x [$\mu m$ ]',
                                         ylabel=r'y [$\mu m$ ]',
                                         extent=wpu.extent_func(img,
                                                                pixelsize)*1e6)

        cmap2crop = plt.cm.get_cmap(cmap)
        cmap2crop.set_over('#FF0000')
        cmap2crop.set_under('#8B008B')
        idx4crop = wpu.graphical_roi_idx(img, verbose=True,
                                         kargs4graph={'cmap': cmap,
                                                      'vmin': colorlimit[0],
                                                      'vmax': colorlimit[1]})

        cmap2crop.set_over(cmap2crop(1))  # Reset Colorbar
        cmap2crop.set_under(cmap2crop(cmap2crop.N-1))

        wpu.set_at_ini_file(inifname, 'Parameters', 'Crop',
                            '{}, {}, {}, {}'.format(idx4crop[0], idx4crop[1],
                                                    idx4crop[2], idx4crop[3]))

        img = wpu.crop_matrix_at_indexes(img, idx4crop)

        # Plot Real Image AFTER crop

        plt.imshow(img, cmap='viridis',
                   extent=wpu.extent_func(img, pixelsize)*1e6)
        plt.xlabel(r'$[\mu m]$')
        plt.ylabel(r'$[\mu m]$')
        plt.colorbar()
        plt.title('Raw Image with New Crop', fontsize=18, weight='bold')

        if saveFigFlag:
            wpu.save_figs_with_idx(saveFileSuf)
        plt.show(block=True)

    else:
        img = tmpImage

    return img, idx4crop


# %%
def _intial_gui_setup(argvzero):

    global inifname  # name of .ini file
    global gui_mode

    defaults = wpu.load_ini_file(inifname)

    if defaults is None:
        p1, p2, p3, p4, p5, p6 = [0, 0, 0, 0, 0, 0]

    else:

        p0 = defaults['Parameters'].get('Mode')
        p1 = float(defaults['Parameters'].get('Pixel Size'))
        p2 = float(defaults['Parameters'].get('Chekerboard Grating Period'))
        p3 = defaults['Parameters'].get('Pattern')
        p4 = float(defaults['Parameters'].get('Distance Detector to Gr'))
        p5 = float(defaults['Parameters'].get('Photon Energy'))
        p6 = float(defaults['Parameters'].get('Source Distance'))

    if p0 == 'Absolute':
        mode_options = ['Absolute', 'Relative']
    else:
        mode_options = ['Relative', 'Absolute']

    analysis_mode = easyqt.get_choice(message='Absolute or Relative analysis?',
                                      title='Select Option',
                                      choices=mode_options)

    if analysis_mode == 'Absolute':

        fname2 = 'None'
        defaults['Files']['Reference'] = fname2
        title = 'Select WF meassurement and Dark Images. '
        title += 'Press ESC to repeat last run.'
        fname1, fname3 = wpu.gui_load_data_dark_filenames(title=title)

    elif analysis_mode == 'Relative':
        title = 'Select Sample, Reference and Dark Images. '
        title += 'Press ESC to repeat last run.'
        (fname1, fname2,
         fname3) = wpu.gui_load_data_ref_dark_filenames(title=title)

    if fname1 is None:
        fname1 = defaults['Files'].get('Sample')
        fname3 = defaults['Files'].get('Blank')

        if analysis_mode == 'Relative':
            fname2 = defaults['Files'].get('Reference')

    if fname3 is None:
        fname3 = 'None'

    wpu.print_blue('MESSAGE: File names:')
    wpu.print_blue('MESSAGE: Dark:   ' + fname3)
    wpu.print_blue('MESSAGE: Ref:    ' + fname2)
    wpu.print_blue('MESSAGE: Sample: ' + fname1)

    pixelsize = easyqt.get_float("Enter Pixel Size [um]",
                                 title='Experimental Values',
                                 default_value=p1*1e6, decimals=5)*1e-6

    gratingPeriod = easyqt.get_float("Enter CB Grating Period [um]",
                                     title='Experimental Values',
                                     default_value=p2*1e6)*1e-6

    if p3 == 'Diagonal half pi':
        choices = ['Diagonal half pi', 'Edge pi']
    else:
        choices = ['Edge pi', 'Diagonal half pi']

    pattern = easyqt.get_choice(message='Select CB Grating Pattern',
                                title='Select Option',
                                choices=choices)

    distDet2sample = easyqt.get_float("Enter Distance Sample - Detector [mm]",
                                      title='Experimental Values',
                                      default_value=p4*1e3)*1e-3

    phenergy = easyqt.get_float("Enter Photon Energy [KeV]",
                                title='Experimental Values',
                                default_value=p5*1e-3)*1e3

    sourceDistance = easyqt.get_float("Enter Distance to Source [m]",
                                      title='Experimental Values',
                                      default_value=p6)

    wpu.set_at_ini_file(inifname, 'Parameters', 'Mode', analysis_mode)
    wpu.set_at_ini_file(inifname, 'Parameters', 'Pixel Size', str(pixelsize))
    wpu.set_at_ini_file(inifname, 'Parameters',
                        'Chekerboard Grating Period', str(gratingPeriod))
    wpu.set_at_ini_file(inifname, 'Parameters', 'Pattern', pattern)
    wpu.set_at_ini_file(inifname, 'Parameters',
                        'Distance Detector to Gr', str(distDet2sample))
    wpu.set_at_ini_file(inifname, 'Parameters', 'Photon Energy', str(phenergy))
    wpu.set_at_ini_file(inifname, 'Parameters',
                        'Source Distance', str(sourceDistance))

    return (fname1, fname2, fname3,
            pixelsize, gratingPeriod, pattern, distDet2sample,
            phenergy, sourceDistance)


# %%
def _load_experimental_pars(argv):

    global gui_mode
    global inifname  # name of .ini file
    pwd, inifname = argv[0].rsplit('/', 1)
    inifname = pwd + '/.' + inifname.replace('.py', '.ini')

    if len(argv) == 17:

        fname_img, fname_imgRef, fname_blank = argv[1:4]

        pixelsize = float(argv[4])*1e-6
        gratingPeriod = float(argv[5])*1e-6
        pattern = argv[6]
        distDet2sample = float(argv[7])*1e-3
        phenergy = float(argv[8])*1e3
        sourceDistance = float(argv[9])

        correct_pi_jump = bool(int(argv[10]))
        remove_mean = bool(int(argv[11]))
        remove_linear = bool(int(argv[12]))
        do_integration = bool(int(argv[13]))
        calc_thickness = bool(int(argv[14]))
        remove_2nd_order = bool(int(argv[15]))
        material_idx = int(argv[16])

        wpu.print_blue('MESSAGE: File names:')
        wpu.print_blue('MESSAGE: Dark:   ' + fname_blank)
        wpu.print_blue('MESSAGE: Ref:    ' + fname_imgRef)
        wpu.print_blue('MESSAGE: Sample: ' + fname_img)

        if fname_imgRef == 'None':
            analysis_mode = 'Absolute'
        else:
            analysis_mode = 'Relative'

        wpu.set_at_ini_file(inifname, 'Parameters', 'Mode', analysis_mode)
        wpu.set_at_ini_file(inifname, 'Parameters',
                            'Pixel Size', str(pixelsize))
        wpu.set_at_ini_file(inifname, 'Parameters',
                            'Chekerboard Grating Period', str(gratingPeriod))

        if 'diag' in pattern.lower():  # 'Diagonal half pi':
            wpu.set_at_ini_file(inifname, 'Parameters',
                                'Pattern', 'Diagonal half pi')
        else:
            wpu.set_at_ini_file(inifname, 'Parameters',
                                'Pattern', 'Edge pi')

        wpu.set_at_ini_file(inifname, 'Parameters',
                            'Distance Detector to Gr', str(distDet2sample))
        wpu.set_at_ini_file(inifname, 'Parameters',
                            'Photon Energy', str(phenergy))
        wpu.set_at_ini_file(inifname, 'Parameters',
                            'Source Distance', str(sourceDistance))

        gui_mode = False

    elif len(argv) == 1:

        (fname_img, fname_imgRef, fname_blank,
         pixelsize, gratingPeriod, pattern, distDet2sample,
         phenergy, sourceDistance) = _intial_gui_setup(argv[0])

        correct_pi_jump = False
        remove_mean = False
        remove_linear = False
        do_integration = False
        calc_thickness = False
        remove_2nd_order = False
        material_idx = 0

        gui_mode = True

    else:

        argExplanations = ['arg0: ',
                           'arg1: file name main image',
                           'arg2: file name reference image',
                           'arg3: file name dark image',
                           'arg4: pixel size [um]',
                           'arg5: Check Board grating period [um]',
                           "arg6: pattern, 'Edge pi' or 'Diagonal half pi' ",
                           'arg7: distance detector to CB Grating [mm]',
                           'arg8: Photon Energy [KeV]',
                           'arg9: Distance to the source [m]',
                           'arg10: Flag correct pi jump in DPC signal',
                           'arg11: Flag remove mean DPC',
                           'arg12: Flag remove 2D linear fit from DPC',
                           'arg13: Flag Calculate Frankot-Chellappa integration',
                           'arg14: Flag Convert phase to thickness',
                           'arg15: Flag remove 2nd order polynomial from integrated Phase',
                           'arg16: Index for material: 0-Diamond, 1-Be']

        print('ERROR: wrong number of inputs: {} \n'.format(len(argv)-1) +
              'Usage: \n'
              '\n'
              'singleGratingTalbotImaging.py : (no inputs) load dialogs \n'
              '\n'
              'singleGratingTalbotImaging.py [args] \n'
              '\n')

        for i, arg in enumerate(argv):
            if i < len(argExplanations):
                print(argExplanations[i] + ':\t' + argv[i])
            else:
                print('arg {}: '.format(i) + argv[i])

        for j in range(i, 17):
            print(argExplanations[j])

        exit(-1)

    menu_options = [correct_pi_jump, remove_mean, remove_linear,
                    do_integration, calc_thickness,
                    remove_2nd_order, material_idx]

    wpu.set_at_ini_file(inifname, 'Files', 'sample', fname_img)
    wpu.set_at_ini_file(inifname, 'Files', 'reference', fname_imgRef)
    wpu.set_at_ini_file(inifname, 'Files', 'blank', fname_blank)

    img = dxchange.read_tiff(fname_img)

    if 'None' in fname_blank and gui_mode is True:
        defaultBlankV = np.int(np.mean(img[0:100, 0:100]))
        defaultBlankV = easyqt.get_int("No Dark File. Value of Dark [counts]\n"
                                       "(Default is the mean value of\n"
                                       "the 100x100 pixels top-left corner)",
                                       title='Experimental Values',
                                       default_value=defaultBlankV, max_=1e5)

        blank = img*0.0 + defaultBlankV

    else:
        blank = dxchange.read_tiff(fname_blank)

    img = img - blank

    if '/' in fname_img:
        saveFileSuf = fname_img.rsplit('/', 1)[0] +\
                      '/' + fname_img.rsplit('/', 1)[1].split('.')[0] + '_output/'
    else:
        saveFileSuf = fname_img.rsplit('/', 1)[1].split('.')[0] + '_output/'

    if os.path.isdir(saveFileSuf):
        saveFileSuf = wpu.get_unique_filename(saveFileSuf, isFolder=True)

    os.makedirs(saveFileSuf, exist_ok=True)

    if fname_imgRef == 'None':
        imgRef = None
        saveFileSuf += 'WF_'
    else:
        imgRef = dxchange.read_tiff(fname_imgRef)
        imgRef = imgRef - blank
        saveFileSuf += 'TalbotImaging_'

    pixelsize = [pixelsize, pixelsize]
    # change here if you need rectangular pixel

    if 'diag' in pattern.lower():  # 'Diagonal half pi':
        gratingPeriod *= 1.0/np.sqrt(2.0)
        phaseShift = 'halfPi'

    elif 'edge' in pattern.lower():  # 'Edge pi':
        gratingPeriod *= 1.0/2.0
        phaseShift = 'Pi'

    saveFileSuf += 'cb{:.2f}um_'.format(gratingPeriod*1e6)
    saveFileSuf += phaseShift
    saveFileSuf += '_d{:.0f}mm_'.format(distDet2sample*1e3)
    saveFileSuf += '{:.1f}KeV'.format(phenergy*1e-3)
    saveFileSuf = saveFileSuf.replace('.', 'p')

    return (img, imgRef, saveFileSuf,
            pixelsize, gratingPeriod, pattern,
            distDet2sample,
            phenergy, sourceDistance, menu_options)

# %%
# based on  http://stackoverflow.com/a/32297563


def _fit_lin_surface(zz, pixelsize):

    from numpy.polynomial import polynomial

    xx, yy = wpu.grid_coord(zz, pixelsize)

    f = zz.flatten()
    deg = np.array([1, 1])
    vander = polynomial.polyvander2d(xx.flatten(), yy.flatten(), deg)
    vander = vander.reshape((-1, vander.shape[-1]))
    f = f.reshape((vander.shape[0],))
    c = np.linalg.lstsq(vander, f)[0]

    wpu.print_blue('MESSAGE: c from fit lin surface')
    wpu.print_blue('MESSAGE: terms: cte, y, x, xy')
    wpu.print_blue(c)

    return polynomial.polyval2d(xx, yy, c.reshape(deg+1)), c


# %%
def _fit_lin_surfaceH(zz, pixelsize):

    xx, yy = wpu.grid_coord(zz, pixelsize)

    argNotNAN = np.isfinite(zz)
    f = zz[argNotNAN].flatten()
    x = xx[argNotNAN].flatten()

    X_matrix = np.vstack([x, x*0.0 + 1]).T

    beta_matrix = np.linalg.lstsq(X_matrix, f)[0]

    fit = (beta_matrix[0]*xx + beta_matrix[1])

    mask = zz*0.0 + 1.0
    mask[~argNotNAN] = np.nan

    return fit*mask, beta_matrix


#
def _fit_lin_surfaceV(zz, pixelsize):

    xx, yy = wpu.grid_coord(zz, pixelsize)

    argNotNAN = np.isfinite(zz)
    f = zz[argNotNAN].flatten()
    y = yy[argNotNAN].flatten()

    X_matrix = np.vstack([y, y*0.0 + 1]).T

    beta_matrix = np.linalg.lstsq(X_matrix, f)[0]

    fit = (beta_matrix[0]*yy + beta_matrix[1])

    mask = zz*0.0 + 1.0
    mask[~argNotNAN] = np.nan

    return fit*mask, beta_matrix


# %%
def _lsq_fit_parabola(zz, pixelsize):

    xx, yy = wpu.grid_coord(zz, pixelsize)
    f = zz.flatten()
    x = xx.flatten()
    y = yy.flatten()

    X_matrix = np.vstack([x**2, y**2, x, y, x*0.0 + 1]).T

    beta_matrix = np.linalg.lstsq(X_matrix, f)[0]

    fit = (beta_matrix[0]*(xx**2) +
           beta_matrix[1]*(yy**2) +
           beta_matrix[2]*xx +
           beta_matrix[3]*yy +
           beta_matrix[4])

    R_x = 1/2/beta_matrix[0]
    R_y = 1/2/beta_matrix[1]
    x_o = -beta_matrix[2]/beta_matrix[0]/2
    y_o = -beta_matrix[3]/beta_matrix[1]/2
    offset = beta_matrix[3]

    popt = [R_x, R_y, x_o, y_o, offset]

    return fit, popt


# %%
def correct_zero_from_unwrap(angleArray):

    pi_jump = np.round(angleArray/np.pi)

    j_o, i_o = wpu.graphical_select_point_idx(pi_jump)

    if j_o is not None:
        angleArray -= pi_jump[i_o, j_o]*np.pi

        return angleArray, int(pi_jump[i_o, j_o])
    else:
        return angleArray, None


# %%
def _default_plot_for_pickle(data, pixelsize, patternforpickle='graph',
                             title='', xlabel=r'$x$', ylabel=r'$y$', ctitle='',
                             removeSpark=True, cmap='viridis'):

    if removeSpark:
        vmin = wpu.mean_plus_n_sigma(data, -6)
        vmax = wpu.mean_plus_n_sigma(data, 6)
    else:
        vmin = np.min(data)
        vmax = np.max(data)

    #    vmax = np.max((np.abs(vmin), np.abs(vmax)))
    #    vmin = -vmax

    fig = plt.figure(figsize=(12, 9.5))

    plt.imshow(data,
               extent=wpu.extent_func(data, pixelsize)*1e6,
               cmap=cmap, vmin=vmin, vmax=vmax)

    plt.xlabel(xlabel, fontsize=24)
    plt.ylabel(ylabel, fontsize=24)

    cbar = plt.colorbar(shrink=0.9)
    cbar.ax.set_title(ctitle, y=1.01)

    plt.title(title, fontsize=24, weight='bold', y=1.01)

    wpu.save_figs_with_idx_pickle(fig, patternforpickle)

    plt.show(block=True)


# %%
def correct_zero_DPC(dpc01, dpc10,
                     pixelsize, distDet2sample, phenergy, saveFileSuf,
                     correct_pi_jump=False, remove_mean=False,
                     saveFigFlag=True):

    title = ['Angle displacement of fringes 01',
             'Angle displacement of fringes 10']

    factor = distDet2sample*hc/phenergy

    angle = [dpc01/pixelsize[1]*factor, dpc10/pixelsize[0]*factor]
    dpc = [dpc01, dpc10]

    wpu.log_this('Initial Hrz Mean angle/pi ' +
                 ': {:} pi'.format(np.mean(angle[0]/np.pi)))

    wpu.log_this('Initial Vt Mean angle/pi ' +
                 ': {:} pi'.format(np.mean(angle[1]/np.pi)))

    while True:

        pi_jump = [0, 0]

        pi_jump[0] = int(np.round(np.mean(angle[0]/np.pi)))
        pi_jump[1] = int(np.round(np.mean(angle[1]/np.pi)))

        plt.figure()
        h1 = plt.hist(angle[0].flatten()/np.pi, 201,
                      histtype='step', linewidth=2)
        h2 = plt.hist(angle[1].flatten()/np.pi, 201,
                      histtype='step', linewidth=2)

        plt.xlabel(r'Angle [$\pi$rad]')
        if pi_jump == [0, 0]:
            lim = np.ceil(np.abs((h1[1][0], h1[1][-1],
                                  h2[1][0], h2[1][-1])).max())
            plt.xlim([-lim, lim])

        plt.title('Correct DPC\n' +
                  r'Angle displacement of fringes $[\pi$ rad]' +
                  '\n' + r'Calculated jumps $x$ and $y$ : ' +
                  '{:d}, {:d} $\pi$'.format(pi_jump[0], pi_jump[1]))

        plt.legend(('DPC x', 'DPC y'))
        plt.tight_layout()
        if saveFigFlag:
                    wpu.save_figs_with_idx(saveFileSuf)
        plt.show(block=False)
        plt.pause(.5)

        if pi_jump == [0, 0]:
            break

        if (gui_mode and easyqt.get_yes_or_no('Subtract pi jump of DPC?') or
           correct_pi_jump):

            #plt.close('all')

            angle[0] -= pi_jump[0]*np.pi
            angle[1] -= pi_jump[1]*np.pi

            dpc01 = angle[0]*pixelsize[0]/factor
            dpc10 = angle[1]*pixelsize[1]/factor
            dpc = [dpc01, dpc10]

            wgi.plot_DPC(dpc01, dpc10,
                         virtual_pixelsize, saveFigFlag=saveFigFlag,
                         saveFileSuf=saveFileSuf)
            plt.show(block=False)

        else:
            break

    wpu.print_blue('MESSAGE: mean angle/pi ' +
                   '0: {:} pi'.format(np.mean(angle[0]/np.pi)))
    wpu.print_blue('MESSAGE: mean angle/pi ' +
                   '1: {:} pi'.format(np.mean(angle[1]/np.pi)))

    wpu.log_this('Horz Mean angle/pi ' +
                 ': {:} pi'.format(np.mean(angle[0]/np.pi)))

    wpu.log_this('Vert Mean angle/pi ' +
                 ': {:} pi'.format(np.mean(angle[1]/np.pi)))

    #    if easyqt.get_yes_or_no('Subtract mean of DPC?'):
    if (gui_mode and easyqt.get_yes_or_no('Subtract mean of DPC?') or
       remove_mean):

        wpu.log_this('%%% COMMENT: Subtrated mean value of DPC',
                     saveFileSuf)
        #plt.close('all')

        angle[0] -= np.mean(angle[0])
        angle[1] -= np.mean(angle[1])

        dpc01 = angle[0]*pixelsize[0]/factor
        dpc10 = angle[1]*pixelsize[1]/factor
        dpc = [dpc01, dpc10]

        plt.figure()
        plt.hist(angle[0].flatten()/np.pi, 201,
                 histtype='step', linewidth=2)
        plt.hist(angle[1].flatten()/np.pi, 201,
                 histtype='step', linewidth=2)

        plt.xlabel(r'Angle [$\pi$rad]')

        plt.title('Correct DPC\n' +
                  r'Angle displacement of fringes $[\pi$ rad]')

        plt.legend(('DPC x', 'DPC y'))
        plt.tight_layout()
        if saveFigFlag:
                    wpu.save_figs_with_idx(saveFileSuf)
        plt.show(block=False)
        plt.pause(.5)

        wgi.plot_DPC(dpc01, dpc10,
                     virtual_pixelsize, saveFigFlag=saveFigFlag,
                     saveFileSuf=saveFileSuf)
        plt.show(block=True)
        plt.pause(.1)
    else:
        pass

    #    if easyqt.get_yes_or_no('Correct DPC center?'):
    if gui_mode and easyqt.get_yes_or_no('Correct DPC center?'):

        wpu.log_this('%%% COMMENT: DCP center is corrected',
                     saveFileSuf)

        #plt.close('all')

        for i in [0, 1]:

            iamhappy = False
            while not iamhappy:

                angle[i], pi_jump[i] = correct_zero_from_unwrap(angle[i])

                wpu.print_blue('MESSAGE: pi jump ' +
                               '{:}: {:} pi'.format(i, pi_jump[i]))
                wpu.print_blue('MESSAGE: mean angle/pi ' +
                               '{:}: {:} pi'.format(i, np.mean(angle[i]/np.pi)))
                plt.figure()
                plt.hist(angle[i].flatten() / np.pi, 101,
                         histtype='step', linewidth=2)
                plt.title(r'Angle displacement of fringes $[\pi$ rad]')

                if saveFigFlag:
                    wpu.save_figs_with_idx(saveFileSuf)
                plt.show()

                plt.figure()

                vlim = np.max((np.abs(wpu.mean_plus_n_sigma(angle[i]/np.pi,
                                                            -5)),
                               np.abs(wpu.mean_plus_n_sigma(angle[i]/np.pi,
                                                            5))))

                plt.imshow(angle[i] / np.pi,
                           cmap='RdGy',
                           vmin=-vlim, vmax=vlim)

                plt.colorbar()
                plt.title(title[i] + r' [$\pi$ rad],')
                plt.xlabel('Pixels')
                plt.ylabel('Pixels')

                plt.pause(.1)

                iamhappy = easyqt.get_yes_or_no('Happy?')
                #plt.close('all')

            dpc[i] = angle[i]*pixelsize[i]/factor

    #plt.close('all')

    return dpc


# Integration
def doIntegration(diffPhase01, diffPhase10,
                  virtual_pixelsize, newCrop=True):

    global gui_mode

    if (gui_mode and newCrop and easyqt.get_yes_or_no('New Crop for Integration?')):
        idx4cropIntegration = ''
    else:
        idx4cropIntegration = [0, -1, 0, -1]

    phase = wgi.dpc_integration(diffPhase01, diffPhase10,
                                virtual_pixelsize,
                                idx4crop=idx4cropIntegration,
                                saveFileSuf=saveFileSuf,
                                plotErrorIntegration=True)

    phase -= np.min(phase)

    #    wpu.set_at_ini_file(inifname, 'Parameters', 'crop integration', #TODO:
    #                        '{}, {}, {}, {}'.format(idx4crop[0], idx4crop[1],
    #                                                idx4crop[2], idx4crop[3]))

    return phase


# %%
def fit_radius_dpc(dpx, dpy, pixelsize, kwave,
                   saveFigFlag=False, str4title=''):

    xVec = wpu.realcoordvec(dpx.shape[1], pixelsize[1])
    yVec = wpu.realcoordvec(dpx.shape[0], pixelsize[0])

    xmatrix, ymatrix = np.meshgrid(xVec, yVec)

    fig = plt.figure(figsize=(14, 5))
    fig.suptitle(str4title + 'Phase [rad]', fontsize=14)

    ax1 = plt.subplot(121)
    ax2 = plt.subplot(122, sharex=ax1, sharey=ax1)

    ax1.plot(xVec*1e6, dpx[dpx.shape[0]//4, :],
             '-ob', label='1/4')
    ax1.plot(xVec*1e6, dpx[dpx.shape[0]//4*3, :],
             '-og', label='3/4')
    ax1.plot(xVec*1e6, dpx[dpx.shape[0]//2, :],
             '-or', label='1/2')

    lin_fitx = np.polyfit(xVec,
                          dpx[dpx.shape[0]//2, :], 1)
    lin_funcx = np.poly1d(lin_fitx)
    ax1.plot(xVec*1e6, lin_funcx(xVec),
             '--c', lw=2,
             label='Fit 1/2')
    curvrad_x = kwave/(lin_fitx[0])

    wpu.print_blue('lin_fitx[0] x: {:.3g} m'.format(lin_fitx[0]))
    wpu.print_blue('lin_fitx[1] x: {:.3g} m'.format(lin_fitx[1]))

    wpu.print_blue('Curvature Radius of WF x: {:.3g} m'.format(curvrad_x))

    ax1.ticklabel_format(style='sci', axis='y', scilimits=(0, 1))
    ax1.set_xlabel(r'[$\mu m$]')
    ax1.set_ylabel('dpx [radians]')
    ax1.legend(loc=0, fontsize='small')
    ax1.set_title('Curvature Radius of WF {:.3g} m'.format(curvrad_x),
                  fontsize=16)
    ax1.set_adjustable('box-forced')

    ax2.plot(yVec*1e6, dpy[:, dpy.shape[1]//4],
             '-ob', label='1/4')
    ax2.plot(yVec*1e6, dpy[:, dpy.shape[1]//4*3],
             '-og', label='3/4')
    ax2.plot(yVec*1e6, dpy[:, dpy.shape[1]//2],
             '-or', label='1/2')

    lin_fity = np.polyfit(yVec,
                          dpy[:, dpy.shape[1]//2], 1)
    lin_funcy = np.poly1d(lin_fity)
    ax2.plot(yVec*1e6, lin_funcy(yVec),
             '--c', lw=2,
             label='Fit 1/2')
    curvrad_y = kwave/(lin_fity[0])
    wpu.print_blue('Curvature Radius of WF y: {:.3g} m'.format(curvrad_y))

    ax2.ticklabel_format(style='sci', axis='y', scilimits=(0, 1))
    ax2.set_xlabel(r'[$\mu m$]')
    ax2.set_ylabel('dpy [radians]')
    ax2.legend(loc=0, fontsize='small')
    ax2.set_title('Curvature Radius of WF {:.3g} m'.format(curvrad_y),
                  fontsize=16)
    ax2.set_adjustable('box-forced')

    if saveFigFlag:
        wpu.save_figs_with_idx(saveFileSuf, extension='png')
    plt.show(block=True)


# %%
# =============================================================================
# %% Main
# =============================================================================
if __name__ == '__main__':

    # ==========================================================================
    # %% Experimental parameters
    # ==========================================================================

    (img, imgRef, saveFileSuf,
     pixelsize, gratingPeriod, pattern,
     distDet2sample,
     phenergy, sourceDistance,
     menu_options) = _load_experimental_pars(sys.argv)

    (correct_pi_jump, remove_mean, remove_linear,
     do_integration, calc_thickness,
     remove_2nd_order, material_idx) = menu_options

    wavelength = hc/phenergy
    kwave = 2*np.pi/wavelength

    # calculate the theoretical position of the hamonics
    period_harm_Vert = np.int(pixelsize[0]/gratingPeriod*img.shape[0] /
                              (sourceDistance + distDet2sample)*sourceDistance)
    period_harm_Hor = np.int(pixelsize[1]/gratingPeriod*img.shape[1] /
                             (sourceDistance + distDet2sample)*sourceDistance)

    saveFigFlag = True

    # ==========================================================================
    # %% do the magic
    # ==========================================================================

    # for relative mode we need to have imgRef=None,
    result = main_single_gr_Talbot(img, imgRef,
                                    phenergy, pixelsize, distDet2sample,
                                    period_harm=[period_harm_Vert,
                                                 period_harm_Hor],
                                    saveFileSuf=saveFileSuf,
                                    unwrapFlag=True,
                                    plotFlag=False,
                                    saveFigFlag=saveFigFlag)

    if saveFigFlag:

        figs = [plt.figure(n) for n in plt.get_fignums()]
        for fig in figs:
            fig.savefig(wpu.get_unique_filename(saveFileSuf, 'png'))
        plt.show(block=True)

    # %%

    [int00, int01, int10,
     darkField01, darkField10,
     diffPhase01, diffPhase10,
     virtual_pixelsize] = result

    # due to beam divergence, the image will expand when propagating.
    # The script uses the increase of the pattern period compared to
    # the theoretical period, and apply the same factor to the pixel size.
    # Note that, since the hor and vert divergences can be different, the
    # virtual pixel size can also be different for hor and vert directions
    wpu.print_blue('VALUES: virtual pixelsize i, j: ' +
                   '{:.4f}um, {:.4f}um'.format(virtual_pixelsize[0]*1e6,
                                               virtual_pixelsize[1]*1e6))

    # %%
    #    exit()

    # %% Log some information

    wpu.log_this('\nvirtual_pixelsize = ' + str(virtual_pixelsize),
                 saveFileSuf)
    wpu.log_this('wavelength [m] = ' + str('{:.5g}'.format(wavelength)))

    lengthSensitivy100 = virtual_pixelsize[0]**2/distDet2sample/100
    # the 100 means that I arbitrarylly assumed the angular error in
    #  fringe displacement to be 2pi/100 = 3.6 deg
    wpu.log_this('WF Length Sensitivy 100 [m] = ' +
                 str('{:.5g}'.format(lengthSensitivy100)))
    wpu.log_this('WF Length Sensitivy 100 [1/lambda] = ' +
                 str('{:.5g}'.format(lengthSensitivy100/wavelength)) + '\n')

    # %%
    # ==========================================================================
    # % Plot
    # ==========================================================================

    # crop again

    idx2ndCrop = wpu.graphical_roi_idx(np.sqrt((diffPhase01 - diffPhase01.mean())**2 +
                                               (diffPhase10 - diffPhase10.mean())**2),
                                       verbose=True,
                                       kargs4graph={'cmap': 'viridis'})

    # %%
    if idx2ndCrop != [0, -1, 0, -1]:

        int00 = wpu.crop_matrix_at_indexes(int00, idx2ndCrop)
        int01 = wpu.crop_matrix_at_indexes(int01, idx2ndCrop)
        int10 = wpu.crop_matrix_at_indexes(int10, idx2ndCrop)
        darkField01 = wpu.crop_matrix_at_indexes(darkField01, idx2ndCrop)
        darkField10 = wpu.crop_matrix_at_indexes(darkField10, idx2ndCrop)
        diffPhase01 = wpu.crop_matrix_at_indexes(diffPhase01, idx2ndCrop)
        diffPhase10 = wpu.crop_matrix_at_indexes(diffPhase10, idx2ndCrop)

        factor_i = virtual_pixelsize[0]/pixelsize[0]
        factor_j = virtual_pixelsize[1]/pixelsize[1]

        idx1stCrop = list(map(int, (wpu.get_from_ini_file(inifname, 'Parameters',
                                                          'Crop').split(','))))

        idx4crop = [0, -1, 0, -1]
        idx4crop[0] = int(np.rint(idx1stCrop[0] + idx2ndCrop[0]*factor_i))
        idx4crop[1] = int(np.rint(idx1stCrop[0] + idx2ndCrop[1]*factor_i))
        idx4crop[2] = int(np.rint(idx1stCrop[2] + idx2ndCrop[2]*factor_j))
        idx4crop[3] = int(np.rint(idx1stCrop[2] + idx2ndCrop[3]*factor_j))

        print('New Crop: {}, {}, {}, {}'.format(idx4crop[0], idx4crop[1],
                                                idx4crop[2], idx4crop[3]))


        wpu.set_at_ini_file(inifname, 'Parameters', 'Crop',
                            '{}, {}, {}, {}'.format(idx4crop[0], idx4crop[1],
                                                    idx4crop[2], idx4crop[3]))

        plt.imshow(img[idx4crop[0]:idx4crop[1],
                       idx4crop[2]:idx4crop[3]], cmap='viridis',
                   extent=wpu.extent_func(img[idx4crop[0]:idx4crop[1],
                                              idx4crop[2]:idx4crop[3]], pixelsize)*1e6)
        plt.xlabel(r'$[\mu m]$')
        plt.ylabel(r'$[\mu m]$')
        plt.colorbar()
        plt.title('Raw Image with 2nd Crop', fontsize=18, weight='bold')

        if saveFigFlag:
            wpu.save_figs_with_idx(saveFileSuf)
        plt.show(block=True)

    # %% plot Intensities and dark field

    if imgRef is not None:

        wgi.plot_intensities_harms(int00, int01, int10,
                                   virtual_pixelsize, saveFigFlag=saveFigFlag,
                                   titleStr='Intensity',
                                   saveFileSuf=saveFileSuf)

        wgi.plot_dark_field(darkField01, darkField10,
                            virtual_pixelsize, saveFigFlag=saveFigFlag,
                            saveFileSuf=saveFileSuf)

        wpu.save_sdf_file(int00, virtual_pixelsize,
                          wpu.get_unique_filename(saveFileSuf + '_intensity', 'sdf'),
                          {'Title': 'Intensity', 'Zunit': 'au'})

        if gui_mode:
            plt.show(block=True)

    # %% plot DPCdiffPhase01

    wgi.plot_DPC(diffPhase01, diffPhase10,
                 virtual_pixelsize, saveFigFlag=saveFigFlag,
                 saveFileSuf=saveFileSuf)

    [diffPhase01,
     diffPhase10] = correct_zero_DPC(diffPhase01, diffPhase10,
                                     virtual_pixelsize,
                                     distDet2sample, phenergy, saveFileSuf,
                                     correct_pi_jump, remove_mean,
                                     saveFigFlag=saveFigFlag)

    # %% remove 2nd order polynomium of phase by
    # removing 1st order surface of DPC

    if gui_mode and easyqt.get_yes_or_no('Remove Linear Fit?') or remove_linear:

        wpu.log_this('%%% COMMENT: Removed Linear Component from DPC',
                     saveFileSuf)

        #        if gui_mode and easyqt.get_yes_or_no('New Linear Fit?'):
        if True:
            linfitDPC01, cH = _fit_lin_surfaceH(diffPhase01, virtual_pixelsize)
            linfitDPC10, cV = _fit_lin_surfaceV(diffPhase10, virtual_pixelsize)

            wpu.set_at_ini_file(inifname, 'Parameters',
                                'lin fitting coef cH',
                                '{}, {}'.format(cH[0], cH[1]))
            wpu.set_at_ini_file(inifname, 'Parameters',
                                'lin fitting coef cV',
                                '{}, {}'.format(cV[0], cV[1]))

        else:
            xx, yy = wpu.grid_coord(diffPhase01, virtual_pixelsize)

            cH = wpu.get_from_ini_file(inifname, 'Parameters',
                                       'lin fitting coef cH')
            cV = wpu.get_from_ini_file(inifname, 'Parameters',
                                       'lin fitting coef cV')

            cH = list(map(float, cH.split(',')))
            cV = list(map(float, cV.split(',')))

            linfitDPC01 = cH[0]*xx + cH[1]
            linfitDPC10 = cV[0]*yy + cV[1]

        wgi.plot_DPC(linfitDPC01, linfitDPC10,
                     virtual_pixelsize,
                     titleStr='\n(linear DPC component)',
                              saveFigFlag=saveFigFlag, saveFileSuf=saveFileSuf)

        wgi.plot_DPC(diffPhase01-linfitDPC01, diffPhase10-linfitDPC10,
                     virtual_pixelsize,
                     titleStr='\n(removed linear DPC component)',
                              saveFigFlag=saveFigFlag, saveFileSuf=saveFileSuf)

        plt.show(block=True)
    else:
        linfitDPC01 = None
        linfitDPC10 = None

    # %% DPC profiles

    if True:

        if linfitDPC01 is None:
            diffPhase01_2save = diffPhase01
            diffPhase10_2save = diffPhase10
        else:
            diffPhase01_2save = diffPhase01 - linfitDPC01
            diffPhase10_2save = diffPhase10 - linfitDPC10

        fnameH = wpu.get_unique_filename(saveFileSuf + '_dpc_X', 'sdf')
        fnameV = wpu.get_unique_filename(saveFileSuf + '_dpc_Y', 'sdf')

        wpu.save_sdf_file(diffPhase01_2save, virtual_pixelsize,
                          fnameH, {'Title': 'DPC 01', 'Zunit': 'rad'})

        wpu.save_sdf_file(diffPhase10_2save, virtual_pixelsize,
                          fnameV, {'Title': 'DPC 10', 'Zunit': 'rad'})

        from dpc_profile_analysis import dpc_profile_analysis  #TODO:

        projectionFromDiv = 1.0
        wpu.log_this('projectionFromDiv : ' + str('{:.4f}'.format(projectionFromDiv)))

        # remove2ndOrder = False #easyqt.get_yes_or_no('Remove 2nd Order for Profile?')

        dpc_profile_analysis(None, fnameV,
                             phenergy, grazing_angle=0,
                             projectionFromDiv=projectionFromDiv,
                             remove1stOrderDPC=False,
                             remove2ndOrder=False,
                             nprofiles=5, filter_width=50)

#        dpc_profile_analysis(fnameH, None,
#                             phenergy, grazing_angle=0,
#                             projectionFromDiv=projectionFromDiv,
#                             remove1stOrderDPC=True,
#                             remove2ndOrder=False,
#                             nprofiles=1, filter_width=50)
#
#        dpc_profile_analysis(fnameH, None,
#                             phenergy, grazing_angle=0,
#                             projectionFromDiv=projectionFromDiv,
#                             remove1stOrderDPC=False,
#                             remove2ndOrder=True,
#                             nprofiles=1, filter_width=50)

#        from dpc_profile_analysis_remove_best_fit import dpc_profile_analysis_remove_best_fit
#        dpc_profile_analysis_remove_best_fit(fnameH, None,
#                                             phenergy, grazing_angle=0,
#                                             projectionFromDiv=projectionFromDiv,
#                                             nprofiles=1, filter_width=50)

    # %% Fit DPC

    if True:
        fit_radius_dpc(diffPhase01, diffPhase10, virtual_pixelsize, kwave,
                       saveFigFlag=saveFigFlag, str4title='')

    # TODO:
    #        fit_radius_dpc(diffPhase01/virtual_pixelsize[1]*gratingPeriod,
    #                       diffPhase10/virtual_pixelsize[0]*gratingPeriod,
    #                       [gratingPeriod, gratingPeriod], kwave,
    #                       saveFigFlag=saveFigFlag, str4title='')
    # ==========================================================================
    # %% Integration
    # ==========================================================================

    #    if easyqt.get_yes_or_no('Perform Frankot-Chellapa Integration?'):

    if (gui_mode and
       easyqt.get_yes_or_no('Perform Frankot-Chellapa Integration?') or
       do_integration):

        do_integration = True

        wpu.print_blue('MESSAGE: Performing Frankot-Chellapa Integration')

        phase = doIntegration(diffPhase01, diffPhase10,
                              virtual_pixelsize)
        wpu.print_blue('DONE')

        #        wpu.print_blue('MESSAGE: Plotting Phase in radians')
        #        wgi.plot_integration(-1/2/np.pi*phase, virtual_pixelsize,
        #                             titleStr=r'-WF $[\lambda$ units $]$',
        #                             plotProfile=gui_mode,
        #                             plot3dFlag=True,
        #                             saveFigFlag=saveFigFlag,
        #                             saveFileSuf=saveFileSuf)
        #
        #        plt.show(block=True)
        #        wpu.print_blue('DONE')

        wpu.print_blue('MESSAGE: Plotting Phase in meters')
        wgi.plot_integration(-1/2/np.pi*phase*wavelength*1e9,
                             virtual_pixelsize,
                             titleStr=r'-WF $[nm]$',
                             plotProfile=gui_mode,
                             plot3dFlag=True,
                             saveFigFlag=saveFigFlag,
                             saveFileSuf=saveFileSuf)

        plt.show(block=True)
        wpu.print_blue('DONE')
        plt.close('all')

        if saveFigFlag:
            wpu.save_sdf_file(-1/2/np.pi*phase*wavelength, virtual_pixelsize,
                              wpu.get_unique_filename(saveFileSuf + '_phase', 'sdf'),
                              {'Title': 'WF Phase', 'Zunit': 'meters'})


        if (gui_mode and easyqt.get_yes_or_no('Convert to thickness?') or

           calc_thickness):

            wpu.print_blue('MESSAGE: Ploting Thickness')

            delta, material = wpu.get_delta(phenergy, choice_idx=material_idx, gui_mode=gui_mode)

            thickness = -(phase - np.min(phase))/kwave/delta

            titleStr = r'Material: ' + material + ', Thickness $[\mu m]$'
            ax = wgi.plot_integration(thickness*1e6,
                                      virtual_pixelsize,
                                      titleStr=titleStr,
                                      ctitle=r'$[\mu m]$',
                                      plotProfile=gui_mode,
                                      plot3dFlag=True,
                                      saveFigFlag=saveFigFlag,
                                      saveFileSuf=saveFileSuf)

            #            if gui_mode and easyqt.get_yes_or_no('Make animation of 3D' +
            #                                                 ' surface?\n' +
            #                                                 '(TAKES A LOT OF TIME)'):
            #
            #                wpu.rocking_3d_figure(ax, saveFileSuf + '.gif',
            #                                      elevOffset=45, azimOffset=60,
            #                                      elevAmp=30, azimAmpl=-60, dpi=80,
            #                                      npoints=200,
            #                                      del_tmp_imgs=True)
            #
            #                plt.show(block=True)

            # Log thickness properties
            wpu.log_this('Material = ' + material, saveFileSuf)
            wpu.log_this('delta = ' + str('{:.5g}'.format(delta)), saveFileSuf)
            thickSensitivy100 = virtual_pixelsize[0]**2/distDet2sample/delta/100
            # the 100 means that I arbitrarylly assumed the angular error in
            #  fringe displacement to be 2pi/100 = 3.6 deg
            wpu.log_this('Thickness Sensitivy 100 [m] = ' +
                         str('{:.5g}'.format(thickSensitivy100)), saveFileSuf)

            if saveFigFlag:
                wpu.save_sdf_file(thickness, virtual_pixelsize,
                                  wpu.get_unique_filename(saveFileSuf + '_thickness', 'sdf'),
                                  {'Title': 'Thickness', 'Zunit': 'meters'})

            #            d2T_dx2 = np.diff(thickness, 2, 1)/virtual_pixelsize[1]**2
            #            d2T_dy2 = np.diff(thickness, 2, 0)/virtual_pixelsize[0]**2
            #
            #            Rx = 1/d2T_dx2
            #            Ry = 1/d2T_dx2
            #
            #            print('Rx: {:.4g}m, sdv: {:.4g}'.format(np.nanmean(Rx), np.nanstd(Rx)))
            #            print('Ry: {:.4g}m, sdv: {:.4g}'.format(np.nanmean(Ry), np.nanstd(Ry)))

        # % 2nd order component of phase

        if linfitDPC01 is not None:

            phase_2nd_order = doIntegration(linfitDPC01,
                                            linfitDPC10,
                                            virtual_pixelsize, newCrop=False)

            wgi.plot_integration(1/2/np.pi*phase_2nd_order, virtual_pixelsize,
                                 titleStr=r'WF, 2nd order component' +
                                          r'$[\lambda$ units $]$',
                                 plotProfile=gui_mode,
                                 saveFigFlag=saveFigFlag,
                                 saveFileSuf=saveFileSuf)
            plt.show(block=True)

            phase_2nd_order = doIntegration(diffPhase01 - linfitDPC01,
                                            diffPhase10 - linfitDPC10,
                                            virtual_pixelsize, newCrop=False)

            wgi.plot_integration(1/2/np.pi*phase_2nd_order, virtual_pixelsize,
                                 titleStr=r'WF, difference to 2nd order component' +
                                          r'$[\lambda$ units $]$',
                                 plotProfile=gui_mode,
                                 saveFigFlag=saveFigFlag,
                                 saveFileSuf=saveFileSuf)

            if saveFigFlag:
                wpu.save_sdf_file(-1/2/np.pi*phase_2nd_order*wavelength, virtual_pixelsize,
                                  wpu.get_unique_filename(saveFileSuf + '_phase', 'sdf'),
                                  {'Title': 'WF Phase 2nd order removed', 'Zunit': 'meters'})

            plt.show(block=True)

    # Log ini file information
    wpu.log_this(preffname=saveFileSuf, inifname=inifname)

    # =============================================================================
    # %% sandbox to play
    # =============================================================================

    #    xxGrid, yyGrid = wpu.grid_coord(thickness, pixelsize)
    #
    #    wpu.plot_profile(xxGrid*1e6, yyGrid*1e6, thickness[::-1, :]*1e6,
    #                     xlabel=r'$x [\mu m]$', ylabel=r'$y [\mu m]$',
    #                     title=r'Thickness $[\mu m]$',
    #                     xunit='\mu m', yunit='\mu m',
    #                     xo=0.0, yo=0.0)
    #
    #    plt.show(block=True)


    # %%



    if (do_integration and gui_mode and
       easyqt.get_yes_or_no('Remove 2nd order polynomial from integrated Phase?') or
       remove_2nd_order):

        if 'thickness' in locals():

            thickness_2nd_order_lsq, popt = _lsq_fit_parabola(thickness, virtual_pixelsize)

            _, popt = _lsq_fit_parabola(thickness, virtual_pixelsize)

            wpu.print_blue('Thickness Radius of WF x: {:.3g} m'.format(popt[0]))
            wpu.print_blue('Thickness Radius of WF y: {:.3g} m'.format(popt[1]))

            err = -(thickness - thickness_2nd_order_lsq)  # [rad]
            err -= np.min(err)

            wgi.plot_integration(err*1e6, virtual_pixelsize,
                                 titleStr=r'Thickness $[\mu m ]$' + '\n' +
                                          r'Rx = {:.3f} $\mu m$, '.format(popt[0]*1e6) +
                                          r'Ry = {:.3f} $\mu m$'.format(popt[1]*1e6),
                                 plotProfile=gui_mode,
                                 plot3dFlag=True,
                                 saveFigFlag=saveFigFlag, saveFileSuf=saveFileSuf)

            plt.show(block=False)

            if saveFigFlag:
                wpu.save_sdf_file(err, virtual_pixelsize,
                                  wpu.get_unique_filename(saveFileSuf + '_thickness_residual', 'sdf'),
                                  {'Title': 'WF Phase', 'Zunit': 'meters'})


        phase_2nd_order_lsq, popt = _lsq_fit_parabola(phase, virtual_pixelsize)

        _, popt = _lsq_fit_parabola(1/2/np.pi*phase*wavelength, virtual_pixelsize)

        wpu.print_blue('Curvature Radius of WF x: {:.3g} m'.format(popt[0]))
        wpu.print_blue('Curvature Radius of WF y: {:.3g} m'.format(popt[1]))

        err = -(phase - phase_2nd_order_lsq)  # [rad]
        err -= np.min(err)

        wgi.plot_integration(err/2/np.pi*wavelength*1e9, virtual_pixelsize,
                             titleStr=r'WF $[nm ]$' +
                                      '\nRx = {:.3f} m, Ry = {:.3f} m'.format(popt[0], popt[1]),
                             plotProfile=gui_mode,
                             plot3dFlag=True,
                             saveFigFlag=saveFigFlag, saveFileSuf=saveFileSuf)

        plt.show(block=False)

        if saveFigFlag:
            wpu.save_sdf_file(err/2/np.pi*wavelength, virtual_pixelsize,
                              wpu.get_unique_filename(saveFileSuf + '_phase_residual', 'sdf'),
                              {'Title': 'WF Phase', 'Zunit': 'meters'})


        wpu.print_blue('DONE')

# %%


#    from fastplot_with_pyqtgraph import plot_surf_fast


#    plot_surf_fast(phase, [virtual_pixelsize[1], virtual_pixelsize[0]])

# %%

#    plot_surf_fast(err, [virtual_pixelsize[1], virtual_pixelsize[0]])


#
#
# %% Mirror
#    grazing_angle = 4e-3  # rad
#
#    wgi.plot_integration(err/np.sin(grazing_angle)*wavelength/4/np.pi*1e9,
#                         [virtual_pixelsize[1]/np.sin(grazing_angle), virtual_pixelsize[0]],
#                         titleStr=r'Mirror Surface [nm]',
#                         plotProfile=False,
#                         saveFigFlag=saveFigFlag,
#                         saveFileSuf=saveFileSuf)
#
#    plt.ylim([-100, 100])
#    plt.show(block=True)

# %% Propagate

if False:
    import sys
    sys.path.append('/home/grizolli/workspace/pythonWorkspace/wgTools')
    from myOpticsLib import *
    from myFourierLib import *
    import time
    from unwrap import unwrap

    # % Propagation 1

    zz = -6.49
    zz = -19.3

    int_pad = np.pad(int00, ((200*2, 200*2),
                             (200*2, 200*2)), 'constant', constant_values=0)
    phase_pad = np.pad(phase, ((200*2, 200*2),
                               (200*2, 200*2)), 'constant', constant_values=0)

    diff_size_ij = int_pad.shape[0] - int_pad.shape[1]
    if diff_size_ij > 0:
        int_pad = np.pad(int_pad, ((0, 0), (0, diff_size_ij)),
                         'constant', constant_values=0)
        phase_pad = np.pad(phase_pad, ((0, 0), (0, diff_size_ij)),
                           'constant', constant_values=0)
    else:

        int_pad = np.pad(int_pad, ((0, -diff_size_ij), (0, 0)),
                         'constant', constant_values=0)
        phase_pad = np.pad(phase_pad, ((0, -diff_size_ij), (0, 0)),
                           'constant', constant_values=0)

    Lx = virtual_pixelsize[1]*phase_pad.shape[1]
    Ly = virtual_pixelsize[0]*phase_pad.shape[0]
    fresnelNumber(Lx, zz, wavelength, verbose=True)

    emf = np.sqrt(int_pad)*np.exp(-1j*phase_pad*1)

    start_time = time.time()
    u2_xy = propTForIR(emf, Lx, Ly, wavelength, zz)
    wpu.print_blue("--- Running time: %.4f seconds ---" % (time.time() - start_time))

    #    start_time = time.time()
    #    Lx2 = 400e-6
    #    u2_xy = prop2step(emf, Lx, Lx2, wavelength, zz)
    #    wpu.print_blue("--- Running time: %.4f seconds ---" % (time.time() - start_time))

    #        prop2step(u1,L1,L2,wavelength,z)
    # %
    #    start_time = time.time()
    #    u2_xy = propIR_RayleighSommerfeld(emf,Lx,Ly,wavelength,zz)
    #    titleStr = str(r'propIR_RayleighSommerfeld, zz=%.3fmm, Intensity [a.u.]'
    #                   % (zz*1e3))

    # %
    #    start_time = time.time()
    #    u2_xy = propTF_RayleighSommerfeld(emf,Lx,Ly,wavelength,zz)
    #    wpu.print_blue("--- Running time: %.4f seconds ---" % (time.time() - start_time))

    print('Propagation Done!')

    intensityDet = np.abs(u2_xy)
    wfdet = unwrap(np.angle(u2_xy))

    xmatrix, ymatrix = wpu.realcoordmatrix(intensityDet.shape[1],
                                           virtual_pixelsize[1],
                                           intensityDet.shape[0],
                                           virtual_pixelsize[0])

    #    xmatrix *= Lx2/Lx
    #    ymatrix *= Lx2/Lx

    # %%
    mask = intensityDet*0.0 + np.nan

    mask[intensityDet > .0000] = 1.0
    #    mask[420:-420, 200:-200] = 1.0
    stride = 4

    wpu.plot_profile(xmatrix[::stride, ::stride]*1e6,
                     ymatrix[::stride, ::stride]*1e6,
                     intensityDet[::stride, ::stride]*mask[::stride, ::stride],
                     xunit='\mu m', yunit='\mu m')

    # %%
    wpu.plot_profile(xmatrix[::stride, ::stride]*1e6,
                     ymatrix[::stride, ::stride]*1e6,
                     wfdet[::stride, ::stride]*mask[::stride, ::stride],
                     xunit='\mu m', yunit='\mu m')

# %%

