'''
Modified on Feb, 2020 based on the work of Yu Sun

author: Jiaming Liu
'''
from __future__ import print_function, division, absolute_import, unicode_literals
from scipy.optimize import fminbound
import numpy as np
import scipy.io as sio
import scipy.misc as smisc
import imageio

evaluatepsnr = lambda xtrue, x: 10*np.log10(1/np.mean((xtrue.flatten('F')-x.flatten('F'))**2))

def to_rgb(img):
    """
    Converts the given array into a RGB image. If the number of channels is not
    3 the array is tiled such that it has 3 channels. Finally, the values are
    rescaled to [0,255) 
    
    :param img: the array to convert [nx, ny, channels]
    
    :returns img: the rgb image [nx, ny, 3]
    """
    img = np.atleast_3d(img)
    channels = img.shape[2]
    if channels < 3:
        img = np.tile(img, 3)
    
    img[np.isnan(img)] = 0
    img -= np.amin(img)
    img /= np.amax(img)
    img *= 255
    return img

def to_double_each(img_norm):
    """
    Normalize img to 0~1.
    Calculate min and max per phase.
    :param img_norm: img to normalize
    """    
    img = img_norm.copy()
    if len(img.shape) == 3: # img.shape = nx*ny*np
        img[np.isnan(img)] = 0
        img_amin = np.tile(np.amin(img,axis=(0,1),keepdims=True),[img.shape[0],img.shape[1],1])
        img -= img_amin
        img_amax = np.tile(np.amax(img,axis=(0,1),keepdims=True),[img.shape[0],img.shape[1],1])
        img /= img_amax
    else:
        print('Incorrect img.shape')
        exit()
    return img,img_amin,img_amax

def to_double_all(img, clip=False):
    """
    Normalize img to 0~1.
    Calculate min and max over 10 phases.
    :param img_norm: img to normalize
    :param clip: clip to 0 ~ INF
    """     
    img_norm = img.copy()
    img_norm = np.clip(img_norm,0,np.inf) if clip else img_norm
    if len(img.shape) == 3: # img.shape = nx*ny*nz
        img_norm[np.isnan(img_norm)] = 0
        img_norm_amin = np.amin(img_norm,keepdims=True)
        img_norm -= img_norm_amin
        img_norm_amax = np.amax(img_norm, keepdims=True)
        img_norm /= img_norm_amax
    else:
        img_norm[np.isnan(img_norm)] = 0
        img_norm_amin = np.amin(img_norm,keepdims=True)
        img_norm -= img_norm_amin
        img_norm_amax = np.amax(img_norm, keepdims=True)
        img_norm /= img_norm_amax
    return img_norm, img_norm_amin, img_norm_amax

def save_mat(img, path):
    """
    Writes the image to disk
    
    :param img: the rgb image to save
    :param path: the target path
    """
    
    sio.savemat(path, {'img':img})


def save_img(img, path):
    """
    Writes the image to disk
    
    :param img: the rgb image to save
    :param path: the target path
    """
    img = to_rgb(img)
    imageio.imwrite(path, img.round().astype(np.uint8))

def addwagon(x,inputSnr):

    """
    Add AWGN to make the measurements to corresponding SNR.
    For simulation only.
    
    :param x: measurements
    :param path: inputSnr value
    """

    noiseNorm = np.linalg.norm(x.flatten('F')) * 10^(-inputSnr/20)
    xBool = np.isreal(x)
    real = True
    for e in np.nditer(xBool):
        if e == False:
            real = False
    if (real == True):
        noise = np.random.randn(np.shape(x)[0],np.shape(x)[1])
    else:
        noise = np.random.randn(np.shape(x)[0],np.shape(x)[1]) + 1j * np.random.randn(np.shape(x)[0],np.shape(x)[1])
    
    noise = noise/np.linalg.norm(noise.flatten('F')) * noiseNorm
    y = x + noise
    return y, noise

def optimizeTau(x, algoHandle, taurange, maxfun=20):

    """
    Optimize Tau
    
    :param x: reference "ground truth"
    :param maxfun: ~ number of iterations for optimization
    """

    
    evaluatepsnr = lambda xtrue, x: 10*np.log10(1/np.mean((xtrue.flatten('F')-x.flatten('F'))**2))
    fun = lambda tau: -cal_rPSNR(x,algoHandle(tau)[0])
    tau = fminbound(fun, taurange[0],taurange[1], xtol = 1e-6, maxfun = maxfun, disp = 3)
    return tau

def cal_rPSNR(xref, x, phase=6):

    """
    Calculate the relative PSNR to the references
    
    :param xref: reference "ground truth"
    :param x: input images
    :param phase: phase to be evaluated
    """

    if len(x.shape) == len(xref.shape):
        x_norm = np.abs(x.copy())
        x_norm,_,_ = to_double_all(x_norm)
        x_norm = np.abs(x_norm[160:480,160:480]) # crop from 640 to 320.
        x_norm = np.flip(x_norm,axis=1)
        my_psnr = evaluatepsnr(xref[:,:,phase],x_norm[:,:,phase])
    else:
        print(x.shape,xref.shape)
        exit()
    return my_psnr