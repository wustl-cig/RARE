from DataFidelities.MRIClass_torch import MRIClass
from Regularizers.robjects import *
from iterAlgs import *
from util import *

import os
import numpy as np
import tensorflow as tf
import scipy.io as sio
import matplotlib.pyplot as plt
'''
Python 3.6
pytorch 1.10
tensorflow 1.10~1.13
Windows 10 or Linux
Jiaming Liu (jiamingliu.jeremy@gmail.com)
github: https://github.com/wustl-cig/RARE
This is the GPU implementation of MCNUFFT using pytorch. 
If you have any questions, please feel free to contact with me.
By Jiaming Liu (16/Feb/2021)
'''
####################################################
####              HYPER-PARAMETERS               ###
####################################################
# Choose Gpu
gpu_ind = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ind # 0,1,2,3

def main():

    ####################################################
    ####              Slice Optimization             ###
    ####################################################
    tau = 0.34
    numIter = 10
    backtracking = False
    num_spokes = '400'
    reg_mode = 'RED'
    num_slices = [25]
    test_name = 'patient54'
    cnnmodel_name = 'network_A2A'
    phase2show = 2
    stor_root = os.path.join('Results', test_name, reg_mode)
    data_root = os.path.join('data', test_name, num_spokes+'spokes')
    model_root = os.path.join('models', cnnmodel_name+'.h5')
    
    for slices in num_slices:
        
        save_path = os.path.join(stor_root, 's' + str(slices))
        backtracking = True if numIter>9 else False
        ## Load forwardmodel
        mriObj = MRIClass(data_root, slices)
        ## MCNUFFT reconstruction
        recon_mcnufft = mriObj.recon_mcnufft
        ## Load 3D-DnCNN as Prior
        rObj = DnCNN3DClass(mriObj.sigSize, model_root)
        ## A2A reconstruction
        recon_a2a = rObj.deartifacts_A2A(recon_mcnufft,useNoise=True)
        ## A2A as initial
        xinit = recon_a2a
        ## start RARE
        recon_rare  = RARE(mriObj, rObj, tau=tau, numIter=numIter, step=2/(2*tau+1), 
                        backtracking=backtracking, accelerate=True, mode=reg_mode, useNoise=False, is_save=True, 
                        save_path=save_path, xref=None, xinit=xinit, clip=False, if_complex='complex', save_iter=1)
        print("Finish processing slice: ", slices,"\n")
        
        ## Display the output images, view window [0.03,0.65]
        plot= lambda x: plt.imshow(x,cmap=plt.cm.gray,vmin=0.03,vmax=0.65)

        recon_mcnufft_norm, _, _ = util.to_double_all(recon_mcnufft, clip=True)
        recon_mcnufft_norm = np.flip(np.abs(recon_mcnufft_norm[160:480, 160:480]), axis=1)[:,:,phase2show]

        recon_a2a_norm, _, _ = util.to_double_all(recon_a2a, clip=True)
        recon_a2a_norm = np.flip(np.abs(recon_a2a_norm[160:480, 160:480]), axis=1)[:,:,phase2show]        

        recon_rare_norm, _, _ = util.to_double_all(recon_rare, clip=True)
        recon_rare_norm = np.flip(np.abs(recon_rare_norm[160:480, 160:480]), axis=1)[:,:,phase2show]
        
        plt.clf()
        plt.subplot(1,3,1)
        plot(recon_mcnufft_norm)
        plt.axis('off')
        plt.title('MCNUFFT')
        plt.subplot(1,3,2)
        plot(recon_a2a_norm)
        plt.title('A2A' )
        plt.axis('off')
        plt.subplot(1,3,3)
        plot(recon_rare_norm)
        plt.title('RARE')
        plt.axis('off')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0,wspace=0)
        plt.show()

if __name__ == '__main__':
    main()