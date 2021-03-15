'''
New torch implementation for free-breathing 4D MCNUFFT,
based on Li Feng & Ricardo Otazo, NYU, 2012.
Weijie Gan, Jiaming Liu, Mar,2021
'''
from glob import glob
import numpy as np
import math
from time import time
import torch
import scipy.io as sio
import matplotlib.pyplot as plt
import os

from DataFidelities.torch_complex_op import tran_tC, conj_tC_, div_tC, mul_tC, np2tC, reshape_tC, abs_tC, tC2np

is_cuda = True

def dict2cuda(x: dict):
    ret = {}
    for k in x:
        if isinstance(x[k], list):
            x_k = []
            for i in range(x[k].__len__()):
                x_k.append(x[k][i].cuda())

            ret[k] = x_k
        else:
            ret[k] = x[k].cuda()

    return ret

class MRIClass(object):

    def __init__(self, data_root, slices, dtype=torch.float64):
        file_path = os.path.join(data_root, 's' + str(slices))
        self.param, self.y, self.n_coil, self.n_phs = self.load_file(file_path, dtype=dtype)

        self.Kd = 960
        self.Nd = 640
        self.dtype = dtype
        self.recon_mcnufft = tC2np(self.ftran(self.y))
        self.sigSize = self.recon_mcnufft.shape

    def size(self):
        sigSize = self.sigSize
        return sigSize

    def grad(self, x, mode='complex'):
        if not torch.is_tensor(x):
            x_torch = np2tC(x)

        Hx_y = self.fmult(x_torch)
        res = Hx_y - self.y
        g = self.ftran(res)
        g_np = tC2np(g)
        if mode is 'complex':
            pass
        elif mode is 'real':
            g_np = g_np.real
        elif mode is 'imag':
            g_np = g_np.imag
        return g_np

    def ftran(self, x: torch.Tensor) -> torch.Tensor:
        param = self.param

        x = x.to(param['w'].dtype)
        if param['w'].is_cuda:
            x = x.cuda()

        n_spoke = x.shape[1]
        x = mul_tC(x, param['w'].unsqueeze(2))

        y = []
        for index_phs in range(self.n_phs):
            x_i = x[:, :, :, index_phs]
            x_i = tran_tC(x_i, 0, 1)
            x_i = reshape_tC(x_i, [-1, self.n_coil])

            c, d = x_i[..., 0], x_i[..., 1]

            ab_mm = param['p_ftran'][index_phs]
            cd_mm = torch.cat([c, d], -1)
            re_mm = ab_mm.mm(cd_mm)

            ac = re_mm[:self.Kd ** 2, :self.n_coil].unsqueeze(-1)
            bc = re_mm[self.Kd ** 2:, :self.n_coil].unsqueeze(-1)

            ad = re_mm[:self.Kd ** 2, self.n_coil:].unsqueeze(-1)
            bd = re_mm[self.Kd ** 2:, self.n_coil:].unsqueeze(-1)

            y_i_real = ac - bd
            y_i_imag = ad + bc

            y_i = torch.stack([y_i_real, y_i_imag], -1)

            y_i = reshape_tC(y_i, [self.Kd, self.Kd, self.n_coil])
            y_i = tran_tC(y_i, 0, 1)

            y.append(y_i)

        y = torch.stack(y, -2)

        y = y.permute([2, 3, 0, 1, 4])
        y = torch.ifft(y, 2)
        y = y.permute([2, 3, 0, 1, 4])

        y = y[:self.Nd, :self.Nd, :, :]
        C = math.pi / 2 / n_spoke * (self.Kd ** 2)

        y = mul_tC(y, param['sn'].unsqueeze(2) * C)

        b1_expand = param['b1'].unsqueeze(3)
        b1_rep = conj_tC_(b1_expand.clone())
        b1_abs = torch.sum(abs_tC(param['b1']) ** 2, 2).unsqueeze(2)

        y = mul_tC(y, b1_rep)
        y = torch.sum(y, 2)
        y = div_tC(y, b1_abs)

        return y

    def fmult(self, x: torch.Tensor) -> torch.Tensor:
        param = self.param
        n_spoke = param['w'].shape[1]

        x = x.to(param['w'].dtype)
        if param['w'].is_cuda:
            x = x.cuda()

        x = x.unsqueeze(2).repeat([1, 1, self.n_coil, 1, 1])
        x = mul_tC(x, param['b1'].unsqueeze(-2))
        x = mul_tC(x, param['sn'].unsqueeze(2))

        x = x.permute([2, 3, 0, 1, 4])
        x = torch.nn.functional.pad(x, [0, 0, 0, 320, 0, 320])
        x = torch.fft(x, 2)
        x = x.permute([2, 3, 0, 1, 4])

        y = []
        for index_phs in range(self.n_phs):
            x_i = x[:, :, :, index_phs]
            x_i = tran_tC(x_i, 0, 1)
            x_i = reshape_tC(x_i, [-1, self.n_coil])

            c, d = x_i[..., 0], x_i[..., 1]

            ab_mm = param['p_fmult'][index_phs]
            cd_mm = torch.cat([c, d], -1)
            re_mm = ab_mm.mm(cd_mm)

            ac = re_mm[:(n_spoke * self.Nd), :self.n_coil].unsqueeze(-1)
            bc = re_mm[(n_spoke * self.Nd):, :self.n_coil].unsqueeze(-1)

            ad = re_mm[:(n_spoke * self.Nd), self.n_coil:].unsqueeze(-1)
            bd = re_mm[(n_spoke * self.Nd):, self.n_coil:].unsqueeze(-1)

            y_i_real = ac - bd
            y_i_imag = ad + bc

            y_i = torch.stack([y_i_real, y_i_imag], -1)

            y_i = reshape_tC(y_i, [n_spoke, self.Nd, self.n_coil])
            y_i = tran_tC(y_i, 0, 1)

            y.append(y_i)

        y = torch.stack(y, -2)

        y = y / self.Nd
        y = mul_tC(y, param['w'].unsqueeze(2))

        return y

    # noinspection PyArgumentList

    @staticmethod
    def load_file(root_path, cuda=is_cuda, dtype=torch.float64):
        
        mc_coil = sio.loadmat(root_path + '/MCUFFT_Param.mat', squeeze_me=False)

        sts = glob(root_path + '/sts/*.mat')
        sts.sort()

        p_ftran = []
        p_fmult = []
        sn = []
        for path in sts:
            st = sio.loadmat(path, squeeze_me=True, mat_dtype=True, struct_as_record=True)

            p_i = st['p'].copy()
            p_i = p_i.tocoo()

            i_row = torch.LongTensor(p_i.row)
            i_col = torch.LongTensor(p_i.col)
            i = torch.stack([i_row, i_col], 0)

            v = torch.DoubleTensor(p_i.data.real)
            p_real_i = torch.sparse_coo_tensor(i, v).to(dtype)

            v = torch.DoubleTensor(p_i.data.imag)
            p_imag_i = torch.sparse_coo_tensor(i, v).to(dtype)

            p_fmult.append(torch.cat([p_real_i, p_imag_i], 0))

            p_i = st['p'].copy()
            p_i = p_i.transpose().conjugate().tocoo()

            i_row = torch.LongTensor(p_i.row)
            i_col = torch.LongTensor(p_i.col)
            i = torch.stack([i_row, i_col], 0)

            v = torch.DoubleTensor(p_i.data.real)
            p_real_i = torch.sparse_coo_tensor(i, v).to(dtype)

            v = torch.DoubleTensor(p_i.data.imag)
            p_imag_i = torch.sparse_coo_tensor(i, v).to(dtype)

            p_ftran.append(torch.cat([p_real_i, p_imag_i], 0))

            sn.append(st['sn'])

        n_coil = mc_coil['b1'].shape[-1]
        n_phs = mc_coil['w'].shape[-1]

        param = {
            'b1': np2tC(mc_coil['b1']).to(dtype),
            'w': torch.from_numpy(mc_coil['w']).to(dtype),
            'sn': torch.from_numpy(np.stack(sn, -1)).to(dtype),

            'p_ftran': p_ftran,
            'p_fmult': p_fmult,
        }

        y = np2tC(np.ascontiguousarray(mc_coil['param_y'])).to(dtype)

        if cuda:
            param = dict2cuda(param)
            y = y.cuda()

        return param, y, n_coil, n_phs