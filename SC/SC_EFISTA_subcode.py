#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PnP-EFISTA subcode for single-image super-resolution (SC).

Contains the DRUNet network (UNetRes), the DRUNet denoiser wrapper, the
8-fold geometric self-ensemble, the s-fold down-sampling helpers, the LPIPS/FSIM
helper and All_Algo, which runs PnP-FISTA or the proposed PnP-EFISTA for the
observation model y = (x * k) downsampled by sf + n  (Eq. (28) of the paper).

Paper: A. Kumar, K. D. Kumar, S. K. Sahoo, "Plug-and-Play EFISTA for Image
Restoration under High Measurement Noise".
"""

import numpy as np
import torch
import torch.nn as nn
import cv2 as cv
import matplotlib.pyplot as plt
import lpips
from piq import fsim
from scipy.fftpack import fftn, ifftn
from scipy.special import comb

from utils_deblr import utils_image as util
from utils_deblr.utils_deblur1 import psf2otf
from utils_sr import utils_model
from utils_sr import utils_sisr as sr
from utils_sr.utils_torch import img_to_tens, scalar_to_tens
from models import resnet_basicblock1 as B


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class UNetRes(nn.Module):
    def __init__(self, in_nc=1, out_nc=1, nc=[64, 128, 256, 512], nb=4, act_mode='R', downsample_mode='strideconv', upsample_mode='convtranspose'):
        super(UNetRes, self).__init__()

        self.m_head = B.conv(in_nc, nc[0], bias=False, mode='C')

        # downsample
        if downsample_mode == 'avgpool':
            downsample_block = B.downsample_avgpool
        elif downsample_mode == 'maxpool':
            downsample_block = B.downsample_maxpool
        elif downsample_mode == 'strideconv':
            downsample_block = B.downsample_strideconv
        else:
            raise NotImplementedError('downsample mode [{:s}] is not found'.format(downsample_mode))

        self.m_down1 = B.sequential(*[B.ResBlock(nc[0], nc[0], bias=False, mode='C'+act_mode+'C') for _ in range(nb)], downsample_block(nc[0], nc[1], bias=False, mode='2'))
        self.m_down2 = B.sequential(*[B.ResBlock(nc[1], nc[1], bias=False, mode='C'+act_mode+'C') for _ in range(nb)], downsample_block(nc[1], nc[2], bias=False, mode='2'))
        self.m_down3 = B.sequential(*[B.ResBlock(nc[2], nc[2], bias=False, mode='C'+act_mode+'C') for _ in range(nb)], downsample_block(nc[2], nc[3], bias=False, mode='2'))

        self.m_body  = B.sequential(*[B.ResBlock(nc[3], nc[3], bias=False, mode='C'+act_mode+'C') for _ in range(nb)])

        # upsample
        if upsample_mode == 'upconv':
            upsample_block = B.upsample_upconv
        elif upsample_mode == 'pixelshuffle':
            upsample_block = B.upsample_pixelshuffle
        elif upsample_mode == 'convtranspose':
            upsample_block = B.upsample_convtranspose
        else:
            raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))

        self.m_up3 = B.sequential(upsample_block(nc[3], nc[2], bias=False, mode='2'), *[B.ResBlock(nc[2], nc[2], bias=False, mode='C'+act_mode+'C') for _ in range(nb)])
        self.m_up2 = B.sequential(upsample_block(nc[2], nc[1], bias=False, mode='2'), *[B.ResBlock(nc[1], nc[1], bias=False, mode='C'+act_mode+'C') for _ in range(nb)])
        self.m_up1 = B.sequential(upsample_block(nc[1], nc[0], bias=False, mode='2'), *[B.ResBlock(nc[0], nc[0], bias=False, mode='C'+act_mode+'C') for _ in range(nb)])

        self.m_tail = B.conv(nc[0], out_nc, bias=False, mode='C')

    def forward(self, x0):
        x1 = self.m_head(x0)
        x2 = self.m_down1(x1)
        x3 = self.m_down2(x2)
        x4 = self.m_down3(x3)
        x = self.m_body(x4)
        x = self.m_up3(x+x4)
        x = self.m_up2(x+x3)
        x = self.m_up1(x+x2)
        x = self.m_tail(x+x1)

        return x


def drunet_wrapper(x, sigma, net):
    x1 = x.copy()
    xt = img_to_tens(x1).type(torch.FloatTensor).to(device)
    xt = xt[0].permute(0, 3, 1, 2)
    sigmat = scalar_to_tens(sigma).to(device)
    with torch.no_grad():
        x = torch.cat((xt, sigmat.float().repeat(1, 1, xt.shape[2], xt.shape[3])), dim=1)
        yt = utils_model.test_mode(net, x, mode=2, refield=32, min_size=256, modulo=16)
        y2 = util.tensor2uint(yt)
        y2 = y2/255  # normalized
    return y2


def downsamplec(x, sf=3):
    '''s-fold downsampler
    Keeping the upper-left pixel for each distinct sfxsf patch and discarding the others
    x: numpy image, HxWxC
    '''
    st = 0
    return x[st::sf, st::sf, :]


def combine_mean(a, sf):
    '''
    Split a into sf x sf distinct blocks and take their mean
    (the Fourier-domain counterpart of s-fold downsampling).

    Args:
        a: numpy array of shape (H, W, C)
        sf: split factor

    Returns:
        b_mean: numpy array of shape (H//sf, W//sf, C)
    '''
    b_mean = sum(np.vsplit(sum(np.hsplit(a, sf)), sf))/sf**2
    return b_mean


def augment_img(img, mode=0):
    if mode == 0:
        return img
    elif mode == 1:
        return np.flipud(np.rot90(img))
    elif mode == 2:
        return np.flipud(img)
    elif mode == 3:
        return np.rot90(img, k=3)
    elif mode == 4:
        return np.flipud(np.rot90(img, k=2))
    elif mode == 5:
        return np.rot90(img)
    elif mode == 6:
        return np.rot90(img, k=2)
    elif mode == 7:
        return np.flipud(np.rot90(img, k=3))


def compute_lpips_fsim(x_np, y_np, lpips_model, fsim, is_gray=True):
    """
    Compute LPIPS and FSIM between x_np and y_np.

    Args:
        x_np: np.ndarray, estimated image, shape HxW (gray) or HxWx3 (color), values in [0,1].
        y_np: np.ndarray, reference image, same shape as x_np, values in [0,1].
        lpips_model: loaded lpips.LPIPS model.
        fsim: FSIM function accepting (torch_tensor1, torch_tensor2, data_range).
        is_gray: bool, True for gray images, False for color.

    Returns:
        lpips_val (float)
        fsim_val (float)
    """
    if is_gray:
        x_t = torch.from_numpy(x_np.copy()).unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1).to(torch.float32)
        y_t = torch.from_numpy(y_np.copy()).unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1).to(torch.float32)
    else:
        x_t = torch.from_numpy(x_np.copy()).permute(2, 0, 1).unsqueeze(0).to(torch.float32)
        y_t = torch.from_numpy(y_np.copy()).permute(2, 0, 1).unsqueeze(0).to(torch.float32)

    # Normalize [0,1] -> [-1,1]
    x_norm = ((x_t * 2.0) - 1.0).to(device)
    y_norm = ((y_t * 2.0) - 1.0).to(device)
    # Before computing FSIM
    x_t = x_t.to(device).clamp(0.0, 1.0)
    y_t = y_t.to(device).clamp(0.0, 1.0)

    with torch.no_grad():
        lpips_val = lpips_model(x_norm, y_norm).item()
        fsim_val = fsim(x_t, y_t, data_range=1.0).item()

    return lpips_val, fsim_val


# Load LPIPS model once
lpips_model = lpips.LPIPS(net='alex').to(device)
lpips_model.eval()


def All_Algo(X, P, sf, Bobs, MAXITER, nw, p, Sigma, lambda_val, tol, Eta, show_img, Select_algo, model_name, net):
    '''
    X          : ground-truth HR image, HxWx3 in [0,1] (used only for the metrics)
    P          : blur kernel
    sf         : scale factor
    Bobs       : LR observation, (H/sf)x(W/sf)x3
    MAXITER    : number of iterations
    nw         : number of fast-forward gradient steps (order of the weighting matrix W)
    p          : EFISTA noise-scaling parameter
    Sigma      : AWGN standard deviation in [0,1]
    lambda_val : regularization parameter (lambda = lambda_val * Sigma^2)
    tol        : stopping tolerance on |f(x_k) - f(x_k-1)|, -np.inf runs all MAXITER iterations
    Eta        : gradient step size
    Select_algo: 'FISTA' or 'EFISTA'
    '''
    m, n, c = Bobs.shape

    Pbig, Sbig = psf2otf(P, (m*sf, n*sf))
    Sbig = np.repeat(Sbig[:, :, np.newaxis], 3, axis=2)

    Btrans = fftn(Bobs)

    # Initialization: bicubic interpolation of the LR observation
    Interp = cv.resize(Bobs, (Bobs.shape[1]*sf, Bobs.shape[0]*sf), interpolation=cv.INTER_CUBIC)
    Interp = sr.shift_pixel(Interp, sf)
    Y = Interp
    X_iter = Interp

    peaksnr11 = util.calculate_psnr(X_iter*255, X*255, border=0)

    fx_all = []
    PSNR = []
    SSIM = []
    LPIPS = []
    FSIM = []
    out_img = []
    print("#iter              fx-val          PSNR            SSIM              LPIPS              FSIM")
    print("===============\n")

    fx_val = 0.5*np.mean(np.abs(combine_mean(Sbig * fftn(X_iter), sf) - Btrans)**2)
    peaksnr1 = util.calculate_psnr(X_iter*255, X*255., border=0)
    ssimval1 = util.calculate_ssim(X_iter*255, X*255., border=0)
    is_gray = True if X_iter.ndim == 2 else False
    lpips_val, fsim_val = compute_lpips_fsim(X_iter, X, lpips_model, fsim, is_gray=is_gray)

    out_img.append(X_iter)
    fx_all.append(fx_val)
    PSNR.append(peaksnr1)
    SSIM.append(ssimval1)
    LPIPS.append(lpips_val)
    FSIM.append(fsim_val)
    print(f"{0:3d}        {fx_val:15.5f}       {peaksnr1:15.5f}     {ssimval1:15.5f}   {lpips_val:15.5f}    {fsim_val:15.5f}")

    # Weighting matrix W = sum_{k=1}^{nw} C(nw,k) (-Eta H^T H)^(k-1), on the LR grid
    M = Eta * combine_mean(np.abs(Sbig)**2, sf)
    Wbig = np.zeros((m, n, c))
    for k in range(1, nw + 1):
        Wbig += comb(nw, k) * (-1 * M)**(k - 1)

    t_new = 1
    x8 = True

    for i in range(MAXITER):

        X_old = X_iter

        if Select_algo == "FISTA":
            t_old = t_new
            D = combine_mean((Sbig * fftn(Y)), sf) - Btrans
            Y = Y - Eta * np.fft.ifftn(np.tile(D, (sf, sf, 1)) * np.conj(Sbig)).real

            # Forward Augmentation (Before inference)
            if x8:
                Y = augment_img(Y, i % 8)

            if 'drunet' in model_name:
                X_iter = drunet_wrapper(Y, np.sqrt(Eta * lambda_val) * Sigma, net)
            else:
                raise ValueError("Error: select a valid model_name (drunet_color)")

            # Reverse (Backward) Augmentation
            if x8:
                if i % 8 == 3 or i % 8 == 5:
                    X_iter = augment_img(X_iter, 8 - i % 8)
                else:
                    X_iter = augment_img(X_iter, i % 8)

            t_new = (1 + np.sqrt(1 + 4 * t_old**2)) / 2
            Y = X_iter + (t_old - 1) / t_new * (X_iter - X_old)

        elif Select_algo == "EFISTA":
            t_old = t_new

            D = combine_mean((Sbig * fftn(Y)), sf) - Btrans

            # Fast-forward accelerated gradient step
            Y = Y - Eta * np.real(ifftn(np.tile(Wbig * D, (sf, sf, 1)) * np.conj(Sbig)))

            # Forward Augmentation (Before inference)
            if x8:
                Y = augment_img(Y, i % 8)

            # Noise-aware denoising, sigma_d = sqrt(p * Eta * lambda) * Sigma
            if 'drunet' in model_name:
                X_iter = drunet_wrapper(Y, np.sqrt(p * Eta * lambda_val) * Sigma, net)
            else:
                raise ValueError("Error: select a valid model_name (drunet_color)")

            # Reverse (Backward) Augmentation
            if x8:
                if i % 8 == 3 or i % 8 == 5:
                    X_iter = augment_img(X_iter, 8 - i % 8)
                else:
                    X_iter = augment_img(X_iter, i % 8)

            t_new = (1 + np.sqrt(1 + 4 * t_old**2)) / 2
            Y = X_iter + (t_old - 1) / t_new * (X_iter - X_old)

        else:
            raise ValueError("Error: select a valid Select_algo (FISTA or EFISTA)")

        fx_val = 0.5*np.mean(np.abs(combine_mean(Sbig * fftn(X_iter), sf) - Btrans)**2)
        peaksnr1 = util.calculate_psnr(X_iter*255, X*255., border=0)
        ssimval1 = util.calculate_ssim(X_iter*255, X*255., border=0)
        lpips_val, fsim_val = compute_lpips_fsim(X_iter, X, lpips_model, fsim, is_gray=is_gray)

        out_img.append(X_iter)
        fx_all.append(fx_val)
        PSNR.append(peaksnr1)
        SSIM.append(ssimval1)
        LPIPS.append(lpips_val)
        FSIM.append(fsim_val)

        if show_img:
            plt.subplot(1, 2, 1)
            plt.axis('off')
            plt.imshow(np.clip(Interp, 0, 1))
            plt.title('PSNR: {:.2f} for Sigma={}'.format(peaksnr11, Sigma))

            plt.subplot(1, 2, 2)
            plt.axis('off')
            plt.imshow(np.clip(X_iter, 0, 1))
            plt.title('PSNR: {:.2f} Iter: {} algo: {}'.format(peaksnr1, i + 1, Select_algo))
            plt.show()

        print(f"{i+1:3d}        {fx_val:15.5f}        {peaksnr1:15.5f}    {ssimval1:15.5f}  {lpips_val:15.5f}    {fsim_val:15.5f}")

        # Check the stopping criterion based on the data-fidelity value
        if i > 0:
            fun_change = np.abs(fx_all[-1] - fx_all[-2])
            if fun_change < tol:
                print(f"Stopping early at iteration {i+1}")
                break

    return X_iter, fx_all, PSNR, SSIM, LPIPS, FSIM, out_img
