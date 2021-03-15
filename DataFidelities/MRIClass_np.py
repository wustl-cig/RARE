'''
Numpy implementation for free-breathing MRI MCNUFFT,
based on Li Feng & Ricardo Otazo, NYU, 2012.
'''
import os
import math
import glob
import os.path
import numpy as np
import scipy.io as sio

from scipy.sparse import csr_matrix
from scipy.sparse import csc_matrix

from util import *

class MRIClass(object):
    def __init__(self, data_root, slices):
        self.param = self.load_mri(data_root, slices)
        self.y = self.param['param_y']
        self.recon_mcnufft = self.mtimes(self.param, self.y, adjoint=True)
        self.sigSize = self.recon_mcnufft.shape
    
    def size(self):
        sigSize = self.sigSize
        return sigSize

    def eval(self,x):
        z = x - self.y
        d = 0.5 * np.power(np.linalg.norm(self.y.flatten('F')-z.flatten('F')),2)
        return d
    
    def grad(self, x, mode):
        Hx_y = self.mtimes(self.param, x, adjoint=False)
        res = Hx_y - self.param['param_y']
        g = self.mtimes(self.param, res, adjoint=True)
        if mode is 'complex':
            pass
        elif mode is 'real':
            g = g.real
        elif mode is 'imag':
            g = g.imag
        return g

    def mtimes(self, a, bb, adjoint):
        assert a or bb is None,"No Parms Have"
        if adjoint:
            # Multicoil non-Cartesian k-space to Cartesian image domain
            # nufft for each coil and time point
            kx, ky, kc = a['b1'].shape
            kp = bb.shape[3]
            res = np.zeros([kx,ky,kc,kp],dtype=complex)
            for tt in range(kp):
                for ch in range(kc):
                    b = np.multiply(bb[:,:,ch,tt],a['w'][:,:,tt])
                    res[:,:,ch,tt] = np.reshape(self.nufft_adj(b.flatten('F'), a['st']['phase_%d'%(tt)])/np.sqrt(a['imSize'].prod()),(kx,ky),order='F')
            # compensate for undersampling factor
            res = res * a['b1'].shape[0]*np.pi/2/a['w'].shape[1]
            # coil combination for each time point
            ab1_expand = np.expand_dims(a['b1'], axis=3)
            ab1_rep = np.tile(ab1_expand.conjugate(),[1,1,1,kp])
            ab1_abs = np.tile(np.expand_dims(np.sum(np.abs(a['b1'])**2,2),axis=2),[1,1,kp])
            # print(ab1_abs.shape)
            ress = np.sum(np.multiply(res,ab1_rep),2).squeeze()/ab1_abs
        else:
            # Cartesian image to multicoil non-Cartesian k-space
            kx, ky, kc = a['b1'].shape
            kp = bb.shape[2]
            ress = np.zeros([a['dataSize'][0][0], a['dataSize'][0][1],kc,kp],dtype=complex)
            for tt in range(kp):
                for ch in range(kc):
                    res = np.multiply(bb[:,:,tt],a['b1'][:,:,ch])
                    ress[:,:,ch,tt] = np.multiply(np.reshape(self.nufft(res,a['st']['phase_%d'%(tt)])/np.sqrt(a['imSize'].prod()),(a['dataSize'][0][0],a['dataSize'][0][1]),order='F'), a['w'][:,:,tt])
        return ress

    @staticmethod
    def load_mri(data_root,slices):
        path_1 = os.path.join(data_root, 's' + str(slices), 'MCUFFT_Param.mat')
        path_2 = os.path.join(data_root, 's' + str(slices), 'sts')
        mncufft = sio.loadmat(path_1,squeeze_me=False)#, mat_dtype=True, struct_as_record=True)
        #####################
        param_e = {
            "b1": mncufft['b1'],
            "adjoint": mncufft['adjoint'],
            "param_y": mncufft['param_y'],
            "dataSize": mncufft['dataSize'],
            "imSize": mncufft['imSize'],
            "w": mncufft['w'],
        }
        #####################
        sts = {}
        files_sts = glob.glob(os.path.join(path_2, '*.mat'))
        files_sts.sort()
        count = 0
        for name in files_sts:

            st = sio.loadmat(name, squeeze_me=True, mat_dtype=True, struct_as_record=True)

            st_temp = {
                "alpha": st['alpha'],
                "beta": st['beta'],
                "Jd": st['Jd'],
                "kb_alf": st['kb_alf'],
                "kb_m": st['kb_m'],
                "Kd": st['Kd'],
                "kernel": st['kernel'],
                "ktype": st['ktype'],
                "M": st['M'],
                "n_shift": st['n_shift'],
                "Nd": st['Nd'],
                "om": st['om'],
                "p" : st['p'],
                "sn": st['sn'],
                "tol ": st['tol'],
            }
            sts["phase_%d"%(count)] = st_temp
            count = count + 1
        param_e['st'] = sts
        return param_e
        
    @staticmethod
    def nufft_adj(X, st):

        # extract attributes from structure
        Lprod = 0
        Nd = st['Nd']
        Kd = st['Kd']
        Kd = np.array([int(Kd[0]), int(Kd[1])])
        Nd = np.array([640,640])
        dims = X.shape
        assert dims != st['M'], "error size"
        # adjoint of interpolator using precomputed sparse matrix
        if len(dims) >= 2:
            Lprod = np.prod(dims[1:len(dims)])
            X = np.reshape(X, (st['M'],Lprod), order='F') # [M,*L]  
        else:
            X = np.expand_dims(X,axis=1)
            Lprod = 1 # the usual case
        stp = st['p'].transpose().tocsr().conjugate()
        Xk_all = stp.dot(X)#equal to stp.real.dot(X.real) + stp.imag.dot(X.imag)
        x = np.zeros([Kd.prod(), Lprod], dtype=np.complex128, order='F')
        for l1 in range(Lprod):
            Xk = np.reshape(Xk_all[:,l1], (Kd[0],Kd[1],1), order='F').squeeze()
            x[:,l1] = (Kd.prod() * np.fft.ifftn(Xk)).flatten('F')
        x = np.reshape(x,(Kd[0],Kd[1]), order='F')
        if Nd.shape[0] == 2:
            x = x[0:(Nd[0]),0:(Nd[1])]
        x = np.reshape(x, (Nd.prod(), Lprod), order='F')
        snc = np.expand_dims((st['sn'].flatten('F')).conjugate(),axis=1)
        x = np.multiply(x,snc)
        x = np.reshape(x, (Nd[0], Nd[1]), order='F')
        return x

    @staticmethod
    def nufft(x,st):
        
        Nd = st['Nd']
        Kd = st['Kd']
        dims = x.shape
        dd = len(Nd)
        tmp = np.ndim(x)
        #assert np.ndim(x) < dd, "input signal has too few dimensions"
        #assert any(dims[0:dd-1]) != Nd, "input signal has wrong size"

        # the usual case is where L=1, i.e., there is just one
        if np.ndim(x) == dd:
            x = np.multiply(x,st['sn'])
            Xk = np.fft.fftn(x,Kd.astype(np.int64)).flatten('F')
        else: # otherwise, collapse all excess dimensions into just one
            xx = np.reshape(x,(Nd.prod(), np.prod(dims[(dd):])))
            L = xx.shape[1]
            Xk = np.zeros(Kd.prod(),L)
            
            for ll in range(L):
                xl = np.reshape(xx[:,ll],(Nd[0],Nd[1],1), order='F').squeeze()
                xl = np.multiply(xl,st['sn'])
                Xk[:,ll] = np.fft.fftn(xl,(Nd[0],Nd[1],1)).flatten('F')
        
        # interpolate using precomputed sparse matrix
        X = st['p'] * Xk
        if np.ndim(x) > dd:
            X = np.reshape(X, (st.M, dims[(dd+1):]), order='F')
        
        return X
