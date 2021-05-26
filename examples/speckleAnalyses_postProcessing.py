# -*- coding: utf-8 -*-
"""
Created on Sat Aug 13 16:00:19 2016

@author: wcgrizolli
"""

#==============================================================================
# %%
#==============================================================================
import numpy as np

from numpy.fft import fft2, ifft2, fftfreq
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


import h5py as h5


import wavepy.utils as wpu
import wavepy.surface_from_grad as wpsg

import itertools



#==============================================================================
# %% preamble
#==============================================================================


# Flags
saveFigFlag = True

# useful constants
rad2deg = np.rad2deg(1)
deg2rad = np.deg2rad(1)
NAN = float('Nan')  # not a number alias

from scipy import constants
hc = constants.value('inverse meter-electron volt relationship') # hc

figCount = itertools.count()  # itera
next(figCount)


def mpl_settings_4_nice_graphs():
    '''
    Settings for latex fonts in the graphs
    ATTENTION: This will make the program slow because it will compile all
    latex text. This means that you also need latex and any latex package
    you want to use. I suggest to only use this when you want produce final
    graphs to publish or to make public. The latex dependecies are not taken
    care  by the installation scripts. You are by your own to solve these
    dependencies.
    '''

    plt.style.use('default')
    #Direct input
    plt.rcParams['text.latex.preamble']=[r"\usepackage[utopia]{mathdesign}"]
    ##Options
    params = {'text.usetex' : True,
              'font.size' : 16,
              'font.family' : 'utopia',
              'text.latex.unicode': True,
              'figure.facecolor' : 'white'
              }
    plt.rcParams.update(params)

# mpl_settings_4_nice_graphs()


#==============================================================================
# %% Some definitions
#==============================================================================


def mySimplePlot(array, title=''):

    plt.figure()
    plt.imshow(array, cmap='spectral', interpolation='none')
    plt.title(title)
    plt.colorbar()
    if saveFigFlag: wpu.save_figs_with_idx(foutname)
    plt.show(block=True)

#==============================================================================
# %% Load files
#==============================================================================

fname = wpu.select_file('**/*.h5')

foutname = fname.rsplit('.')[0]

f = h5.File(fname,'r')

#print(wpu.h5ListOfGroups(f))

#==============================================================================
# %% parameters
#==============================================================================

delta = 5.3265E-06
# real part refractive index Be at 8KeV from http://henke.lbl.gov/


pixelsizeDetector = f['raw'].attrs['Pixel Size Detector [m]']

pixelsizeImg = f['displacement'].attrs['Pixel Size Processed images [m]']
distDet2sample = f['displacement'].attrs['Distance Detector to Sample [m]']
phenergy = f['displacement'].attrs['Photon Energy [eV]']

wavelength = hc/phenergy
kwave = 2*np.pi/wavelength

print('MESSAGE: Comments from hdf5 files')
print('MESSAGE: '+ f['displacement'].attrs['Comments'])


# %%

sx_raw = np.array(f['displacement/displacement_x'])
sy_raw = np.array(f['displacement/displacement_y'])
error_raw = np.array(f['displacement/error'])

xVec_raw =  np.array(f['displacement/xvec'])
yVec_raw =  np.array(f['displacement/yvec'])

#==============================================================================
# %% Crop
#==============================================================================

idx4crop = wpu.graphical_roi_idx(np.sqrt(sx_raw**2 + sy_raw**2), verbose=True)



sx = wpu.crop_matrix_at_indexes(sx_raw, idx4crop)
sy = wpu.crop_matrix_at_indexes(sy_raw, idx4crop)
error = wpu.crop_matrix_at_indexes(error_raw, idx4crop)


xVec = wpu.realcoordvec(sx.shape[1], pixelsizeImg)
yVec = wpu.realcoordvec(sx.shape[0], pixelsizeImg)

xmatrix, ymatrix = np.meshgrid(xVec, yVec)




#==============================================================================
# %% Calculations of physical quantities
#==============================================================================


totalS = np.sqrt(sx**2 + sy**2)


# Differenctial Phase
dpx = kwave*np.arctan2(sx*pixelsizeDetector, distDet2sample)
dpy = kwave*np.arctan2(sy*pixelsizeDetector, distDet2sample)


# Differenctial Thickness
dTx = 1.0/delta*np.arctan2(sx*pixelsizeDetector, distDet2sample)
dTy = 1.0/delta*np.arctan2(sy*pixelsizeDetector, distDet2sample)


#==============================================================================
# %% quiver
#==============================================================================

#
#
#
#from skimage.io import imread, imsave
#
#tempImage = imread('/home/grizolli/workspace/pythonWorkspace/data/Be1x50um/set1/very_small_sample.jpg')
#
#
#fig = plt.figure(figsize=(10, 7.5))
#stride = 1
#plt.contourf(xmatrix*1e6,
#             ymatrix*1e6,
#             wpu.crop_matrix_at_indexes(tempImage, idx4crop),
#             1001, cmap='Greys_r')
#
#plt.xlabel('[um]')
#plt.ylabel('[um]')
#
#plt.show()
#wpu.save_figs_with_idx(foutname)
#
#
#
##
#
#fig = plt.figure(figsize=(10, 7.5))
#stride = 1
#plt.contourf(xmatrix*1e6,
#             ymatrix*1e6,
#             wpu.crop_matrix_at_indexes(tempImage, idx4crop),
#             1001, cmap='Greys_r')
#
#stride = 12
#plt.quiver(xmatrix[::stride, ::stride]*1e6,
#           ymatrix[::stride, ::stride]*1e6,
#           sx[::stride, ::stride], sy[::stride, ::stride],
#           angles='xy', minlength=2.2, color='c', lw=.5, width=0.005)
#
#plt.xlabel('[um]')
#plt.ylabel('[um]')
#
#plt.show()
#wpu.save_figs_with_idx(foutname)
#



#==============================================================================
# %% integration frankotchellappa
#==============================================================================


#integration_res = frankotchellappa(dTx,dTy)
integration_res = wpsg.frankotchellappa(dTx*pixelsizeImg, dTy*pixelsizeImg)

thickness = np.real(integration_res)

thickness = thickness - np.min(thickness)

# %%

wpsg.error_integration(dTx*pixelsizeImg, dTy*pixelsizeImg, thickness,
                       [pixelsizeImg, pixelsizeImg], shifthalfpixel=True, plot_flag=True)

#==============================================================================
# %% Thickness
#==============================================================================


stride = 1

wpu.plot_profile(xmatrix[::stride, ::stride]*1e6,
                 ymatrix[::stride, ::stride]*1e6,
                 thickness[::stride, ::stride]*1e6,
                 title='Thickness', xlabel='[um]', ylabel='[um]',
                 arg4main={'cmap':'spectral'}) #, xo=0.0, yo=0.0)_1Dparabol_4_fit
plt.show(block=True)

#==============================================================================
# %% Plot
#==============================================================================





def plotsidebyside(array1, array2, title1='', title2='', maintitle=''):

    fig = plt.figure(figsize=(14, 5))
    fig.suptitle(maintitle, fontsize=14)

    vmax = np.max([array1, array2])
    vmin = np.min([array1, array2])

    ax1 = plt.subplot(121)
    ax2 = plt.subplot(122, sharex=ax1, sharey=ax1)

    im1 = ax1.imshow(array1, cmap='RdGy',
                     interpolation='none',
                     vmin=vmin, vmax=vmax)
    ax1.set_title(title1, fontsize=22)
    ax1.set_adjustable('box-forced')
    fig.colorbar(im1, ax=ax1, shrink=.8, aspect=20)

    im2 = ax2.imshow(array2, cmap='RdGy',
                     interpolation='none',
                     vmin=vmin, vmax=vmax)
    ax2.set_title(title2, fontsize=22)
    ax2.set_adjustable('box-forced')
    fig.colorbar(im2, ax=ax2, shrink=.8, aspect=20)


    if saveFigFlag: wpu.save_figs_with_idx(foutname)
    plt.show(block=True)

#==============================================================================
# %% Plot dpx and dpy and fit Curvature Radius of WF
#==============================================================================


fig = plt.figure(figsize=(14, 5))
fig.suptitle('Phase [rad]', fontsize=14)


ax1 = plt.subplot(121)
ax2 = plt.subplot(122, sharex=ax1, sharey=ax1)



ax1.plot(xVec*1e6, dpx[dpx.shape[1]//4,:],'-ob')
ax1.plot(xVec*1e6, dpx[dpx.shape[1]//2,:],'-or')
ax1.plot(xVec*1e6, dpx[dpx.shape[1]//4*3,:],'-og')
ax1.ticklabel_format(style='sci', axis='y', scilimits=(0, 1))
ax1.set_xlabel('[um]')
ax1.set_ylabel('dpx [radians]')

lin_fitx = np.polyfit(xVec, dpx[dpx.shape[1]//2,:], 1)
lin_funcx = np.poly1d(lin_fitx)
ax1.plot(xVec*1e6, lin_funcx(xVec),'--c',lw=2)
curvrad_x = kwave/(lin_fitx[0])

ax1.set_title('Curvature Radius of WF {:.3g} m'.format(curvrad_x), fontsize=18)
ax1.set_adjustable('box-forced')


ax2.plot(yVec*1e6, dpy[:,dpy.shape[0]//4],'-ob')
ax2.plot(yVec*1e6, dpy[:,dpy.shape[0]//2],'-or')
ax2.plot(yVec*1e6, dpy[:,dpy.shape[0]//4*3],'-og')
ax2.ticklabel_format(style='sci', axis='y', scilimits=(0, 1))
ax2.set_xlabel('[um]')
ax2.set_ylabel('dpy [radians]')

lin_fity = np.polyfit(yVec, dpy[:,dpy.shape[0]//2], 1)
lin_funcy = np.poly1d(lin_fity)
ax2.plot(yVec*1e6, lin_funcy(yVec),'--c',lw=2)
curvrad_y = kwave/(lin_fity[0])

ax2.set_title('Curvature Radius of WF {:.3g} m'.format(curvrad_y), fontsize=18)
ax2.set_adjustable('box-forced')



if saveFigFlag: wpu.save_figs_with_idx(foutname)
plt.show(block=True)



# %%

plotsidebyside(sx, sy, r'Displacement $S_x$ [pixels]',
                         r'Displacement $S_y$ [pixels]')

# %%
mySimplePlot(totalS, title=r'Displacement Module $|\vec{S}|$ [pixels]')

# %%


fig = plt.figure(figsize=(14, 5))

ax1 = plt.subplot(121)
ax2 = plt.subplot(122, sharex=ax1, sharey=ax1)

ax1.plot(sx.flatten(),error.flatten(),'.')
ax1.set_xlabel('Sy [pixel]')
ax1.set_title('Error vs Sx', fontsize=22)
ax1.set_adjustable('box-forced')


ax2.plot(sy.flatten(),error.flatten(),'.')
ax2.set_xlabel('Sy [pixel]')
ax2.set_title('Error vs Sy', fontsize=22)
ax2.set_adjustable('box-forced')


if saveFigFlag: wpu.save_figs_with_idx(foutname)
plt.show(block=True)


#==============================================================================
# %% Histograms to evaluate data quality
#==============================================================================


fig = plt.figure(figsize=(14, 5))
fig.suptitle('Histograms to evaluate data quality', fontsize=16)

ax1 = plt.subplot(121)
ax1 = plt.hist(sx.flatten(), 51)
ax1 = plt.title(r'$S_x$ [pixels]', fontsize=16)

ax1 = plt.subplot(122)
ax2 = plt.hist(sy.flatten(), 51)
ax2 = plt.title(r'$S_y$ [pixels]', fontsize=16)


if saveFigFlag: wpu.save_figs_with_idx(foutname)
plt.show(block=True)

##==============================================================================
## %% Total displacement
##==============================================================================
#
#plt.figure()
#plt.hist(totalS.flatten(), 51)[0]
#plt.title(r'Total displacement $|\vec{S}|$ [pixels]', fontsize=16)
#if saveFigFlag: wpu.save_figs_with_idx(foutname)
#plt.show(block=True)


#==============================================================================
# %% Integration Real and Imgainary part
#==============================================================================


fig = plt.figure(figsize=(14, 5))
fig.suptitle('Histograms to evaluate data quality', fontsize=16)

ax1 = plt.subplot(121)
ax1 = plt.hist(np.real(integration_res).flatten()*1e6, 51)
ax1 = plt.title(r'Integration Real part', fontsize=16)

ax1 = plt.subplot(122)
ax2 = plt.hist(np.imag(integration_res).flatten()*1e6, 51)
ax2 = plt.title(r'Integration Imag part', fontsize=16)

if saveFigFlag: wpu.save_figs_with_idx(foutname)
plt.show(block=True)

# %% Crop Result and plot surface



(xVec_croped1, yVec_croped1,
 thickness_croped, _) = wpu.crop_graphic(xVec, yVec,
                                         thickness*1e6, verbose=True)

thickness_croped *= 1e-6
thickness_croped -= np.max(thickness_croped)

xmatrix_croped1, ymatrix_croped1 = wpu.realcoordmatrix_fromvec(xVec_croped1,
                                                               yVec_croped1)


# %% center fig

def center_max_2darray(array):
    '''
    crop the array in order to have the max at the center of the array
    '''
    center_i, center_j = np.unravel_index(array.argmax(), array.shape)

    if 2*center_i  > array.shape[0]:
        array = array[2*center_i-array.shape[0]:-1,:]
    else:
        array = array[0:2*center_i,:]

    if 2*center_j  > array.shape[1]:
        array = array[:, 2*center_j-array.shape[1]:-1]
    else:
        array = array[:,0:2*center_j]

    return array


# %%

thickness_croped = center_max_2darray(thickness_croped)


xVec_croped1 = wpu.realcoordvec(thickness_croped.shape[1], pixelsizeImg)
yVec_croped1 = wpu.realcoordvec(thickness_croped.shape[0], pixelsizeImg)

xmatrix_croped1, ymatrix_croped1 = np.meshgrid(xVec_croped1, yVec_croped1)



# %%

lim = 1

wpu.plot_profile(xmatrix_croped1[lim:-lim,lim:-lim]*1e6,
                 ymatrix_croped1[lim:-lim,lim:-lim]*1e6,
                 thickness_croped[lim:-lim,lim:-lim]*1e6,
                 title='Thickness centered [um]', xlabel='[um]', ylabel='[um]',
                 arg4main={'cmap':'spectral'}) #, xo=0.0, yo=0.0)




plt.show(block=True)

# %%

#



fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

stride = thickness_croped.shape[0] // 100
if stride == 0: stride = 1


surf = ax.plot_surface(xmatrix_croped1*1e6,
                       ymatrix_croped1*1e6,
                       thickness_croped*1e6,
                        rstride=stride, cstride=stride,
                        #vmin=-120, vmax=0,
                       cmap='spectral', linewidth=0.1)

plt.xlabel('[um]')
plt.ylabel('[um]')

plt.title('Thickness [um]', fontsize=18, weight='bold')
plt.colorbar(surf, shrink=.8, aspect=20)

plt.tight_layout()
if saveFigFlag: wpu.save_figs_with_idx(foutname)
plt.show(block=False)


