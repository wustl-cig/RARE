# library
import os
import time
import shutil
import warnings
import numpy as np
import scipy.io as sio
from numpy import linalg as LA
from tqdm.auto import tqdm, trange
# scripts
import util

######## Iterative Methods #######

def RARE(dObj, rObj, tau = 0.001, numIter=100, step=100, beta=1e-3, Lipz_total=1, backtracking=True, 
         backtotl=1,  accelerate=False, mode='RED', useNoise=True, is_save=True, save_mat=False, save_path='result', 
         xref=None, xinit=None, clip=False, if_complex='complex', save_iter=5):
    """
    Regularized by artifacts removal methods with switch for RED, PGM, Grad

    ### INPUT:
    dObj           ~ data fidelity term, measurement/forward model
    rObj           ~ regularizer term
    tau            ~ control regularizer strength
    numIter        ~ total number of iterations
    step           ~ step-size
    beta           ~ stoping criterion
    Lipz_total     ~ Lipz value of fowardmodel
    backtracking   ~ backtracking linesearch
    backtotl       ~ tolerance
    accelerate     ~ acceleration or not 
    mode           ~ RED update, PROX, or Grad update
    useNoise.      ~ CNN predict noise or image
    is_save        ~ if true save the reconstruction of each iteration
    save_mat       ~ if save .mat file
    save_path      ~ the save path for is_save
    xref           ~ the CS2000 of the image, for tracking purpose
    if_complex     ~ Use complex number
    ### OUTPUT:
    x     ~ reconstruction of the algorithm
    """

    ##### HELPER FUNCTION #####

    evaluateTol = lambda x, xnext: np.linalg.norm(x.flatten('F')-xnext.flatten('F'))/np.linalg.norm(x.flatten('F'))
    evaluateGx = lambda s_step: 1/Lipz_total * (dObj.grad(s_step, if_complex) + tau * rObj.deartifacts_A2A(s_step,useNoise=useNoise,clip=clip))

    ##### INITIALIZATION #####

    # initialize save foler
    if is_save:
        abs_save_path = os.path.abspath(save_path)
        if os.path.exists(save_path):
            print("Removing '{:}'".format(abs_save_path))
            shutil.rmtree(abs_save_path, ignore_errors=True)
        # make new path
        print("Allocating '{:}'".format(abs_save_path))
        os.makedirs(abs_save_path)

    #initialize info data
    if xref is not None:
        xrefSet = True
        rPSNR = []
    else:
        xrefSet = False

    #log info
    timer = []
    relativeChange = []
    norm_Gs = []
    
    # initialize variables
    if xinit is not None:
        pass
    else:    
        ## xinti plays key role for the reconstruction, not recommend zero initialization
        xinit = dObj.recon_mcnufft if if_complex else np.abs(dObj.recon_mcnufft)
    x = xinit
    s = xinit 
    t = 1.    # controls acceleration
    bar = trange(numIter)

    #Main Loop#
    for indIter in bar:
        timeStart = time.time()
        if mode == 'RED':
            Gs = evaluateGx(s)
            xnext = s - step*Gs
            xnext = np.clip(xnext,0,np.inf) if clip else xnext    # clip to [0, inf]
            norm_Gs.append(LA.norm(Gs.flatten('F')))
        elif mode == 'GRAD': ## GRAD with NN projection, which can be used for warmup
            g = dObj.grad(s,if_complex)
            xnext = s-step*g
            xnext = np.clip(xnext,0,np.inf) if clip else xnext
        timeEnd = time.time() - timeStart
        timer.append(timeEnd)
        if indIter == 0:
            relativeChange.append(np.inf)
        else:
            relativeChange.append(evaluateTol(x, xnext))        
        # ----- backtracking (damping) ------ #
        if backtracking is True:
            G_update = evaluateGx(xnext)
            while LA.norm(G_update.flatten('F')) > LA.norm(Gs.flatten('F')) and step >= backtotl:
                step = beta * step
                xnext = s - step*Gs   #TODO clip to [0, inf]
                G_update = evaluateGx(xnext)

                if step <= backtotl:
                    bar.close()
                    print("Reach to backtotl, stop updating.")
                    return x

        if xrefSet:
            rPSNR.append(util.cal_rPSNR(xref, x))

        # acceleration
        if accelerate:
            tnext = 0.5*(1+np.sqrt(1+4*t*t))
        else:
            tnext = 1
        s = xnext + ((t-1)/tnext)*(xnext-x)

        # update
        t = tnext
        x = xnext

        #save & print
        if is_save and (indIter) % (save_iter) == 0:
            #crop image from 640 to 320, only show the regions of interests.
            x_temp = np.flip(np.abs(xnext[160:480,160:480,:]),axis=1)
            x_temp, _, _ = util.to_double_all(x_temp, clip=True)
            length = int(x_temp.shape[2]/2)
            img_save_1 = x_temp[:,:,0]
            img_save_2 = x_temp[:,:,length]
            for zz in range(1,length):
                img_save_1 = np.concatenate((img_save_1,x_temp[:,:,zz]),axis=1)
                img_save_2 = np.concatenate((img_save_2,x_temp[:,:,zz+length]),axis=1)
            img_save =np.concatenate((img_save_1,img_save_2),axis=0)
            #if save .mat
            if save_mat:
                util.save_mat(xnext, abs_save_path+'/iter_{}.mat'.format(indIter+1))
            #view window    
            img_save = np.clip(img_save,0.03,0.65)
            #save image per iter
            util.save_img(img_save, abs_save_path+'/iter_{}_img.tif'.format(indIter+1))
    bar.close()

    return x