#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 18 13:53:01 2026

@author: dinesh1
"""

import torch
import torch.fft


def splits(a, sf):
    '''split a into sfxsf distinct blocks
    Args:
        a: NxCxWxH
        sf: split factor
    Returns:
        b: NxCx(W/sf)x(H/sf)x(sf^2)
    '''
    b = torch.stack(torch.chunk(a, sf, dim=2), dim=4)
    b = torch.cat(torch.chunk(b, sf, dim=3), dim=4)
    return b

def fft_blur(x, FB):
    """
    Circular convolution using FFT.
    x  : N x C x H x W
    FB : N x 1 x H x W or N x C x H x W, complex
    """
    return torch.real(torch.fft.ifftn(FB * torch.fft.fftn(x, dim=(-2, -1)), dim=(-2, -1)))

def p2o(psf, shape):
    '''
    Convert point-spread function to optical transfer function.
    otf = p2o(psf) computes the Fast Fourier Transform (FFT) of the
    point-spread function (PSF) array and creates the optical transfer
    function (OTF) array that is not influenced by the PSF off-centering.
    Args:
        psf: NxCxhxw
        shape: [H, W]
    Returns:
        otf: NxCxHxWx2
    '''
    otf = torch.zeros(psf.shape[:-2] + shape).type_as(psf)
    otf[...,:psf.shape[2],:psf.shape[3]].copy_(psf)
    for axis, axis_size in enumerate(psf.shape[2:]):
        otf = torch.roll(otf, -int(axis_size / 2), dims=axis+2)
    otf = torch.fft.fftn(otf, dim=(-2,-1))
    #n_ops = torch.sum(torch.tensor(psf.shape).type_as(psf) * torch.log2(torch.tensor(psf.shape).type_as(psf)))
    #otf[..., 1][torch.abs(otf[..., 1]) < n_ops*2.22e-16] = torch.tensor(0).type_as(psf)
    return otf


def upsample(x, sf=3):
    '''s-fold upsampler
    Upsampling the spatial size by filling the new entries with zeros
    x: tensor image, NxCxWxH
    '''
    st = 0
    z = torch.zeros((x.shape[0], x.shape[1], x.shape[2]*sf, x.shape[3]*sf)).type_as(x)
    z[..., st::sf, st::sf].copy_(x)
    return z

def downsample(x, sf=3):
    '''s-fold downsampler
    Keeping the upper-left pixel for each distinct sfxsf patch and discarding the others
    x: tensor image, NxCxWxH
    '''
    st = 0
    return x[..., st::sf, st::sf]

def pre_calculate(x, k, sf):
    '''
    Args:
        x: NxCxHxW, LR input
        k: NxCxhxw
        sf: integer

    Returns:
        FB, FBC, F2B, FBFy
        will be reused during iterations
    '''
    w, h = x.shape[-2:]
    FB = p2o(k, (w*sf, h*sf))
    FBC = torch.conj(FB)
    F2B = torch.pow(torch.abs(FB), 2)
    STy = upsample(x, sf=sf)
    FBFy = FBC*torch.fft.fftn(STy, dim=(-2, -1))
    return FB, FBC, F2B, FBFy


def fft_blur_transpose(x, FBC):
    """
    Transpose blur operation A^T using conjugate OTF.
    """
    return torch.real(torch.fft.ifftn(FBC * torch.fft.fftn(x, dim=(-2, -1)), dim=(-2, -1)))


def conjugate_gradient(Afun, b, x0=None, n_iter=20, tol=1e-6):
    """
    Solve Afun(x) = b using conjugate gradient.
    """
    x = torch.zeros_like(b) if x0 is None else x0.clone()

    r = b - Afun(x)
    p = r.clone()

    rsold = torch.sum(r * r, dim=(-3, -2, -1), keepdim=True)

    for _ in range(n_iter):
        Ap = Afun(p)

        denom = torch.sum(p * Ap, dim=(-3, -2, -1), keepdim=True)
        denom = torch.clamp(denom, min=1e-12)

        step = rsold / denom
        x = x + step * p
        r = r - step * Ap

        rsnew = torch.sum(r * r, dim=(-3, -2, -1), keepdim=True)

        if torch.sqrt(rsnew.max()) < tol:
            break

        beta = rsnew / torch.clamp(rsold, min=1e-12)
        p = r + beta * p
        rsold = rsnew

    return x


def data_solution_weighted_deblur_cg(z, y, FB, FBC, alpha, sigma_map,
                                     cg_iter=20, eps=1e-6,
                                     weighted_penalty=True):
    """
    Weighted x-update for deblurring.

    Solves either:

    Exact weighted HQS:
        (A^T W A + alpha W)x = A^T W y + alpha W z

    or data-weighted only:
        (A^T W A + alpha I)x = A^T W y + alpha z

    Args:
        z: current HQS variable, N x C x H x W
        y: degraded blurry/noisy image, N x C x H x W
        FB: Fourier transform of blur kernel
        FBC: conjugate of FB
        alpha: rho/mu parameter used in your current data_solution
        sigma_map: estimated sigma map, N x 1 x H x W
        cg_iter: number of CG iterations
        weighted_penalty: True matches your current paper equation more closely

    Returns:
        x: updated image
    """

    alpha = alpha.to(device=z.device, dtype=z.dtype)

    sigma_map = sigma_map.to(device=z.device, dtype=z.dtype)

    if sigma_map.shape[1] != 1 and sigma_map.shape[1] != z.shape[1]:
        sigma_map = sigma_map.mean(dim=1, keepdim=True)

    sigma_map = torch.clamp(sigma_map, min=eps)

    W = 1.0 / (sigma_map ** 2)

    # Normalization helps numerical stability.
    # For weighted_penalty=True, this does not change the solution.
    W = W / torch.clamp(W.mean(dim=(-2, -1), keepdim=True), min=eps)

    if weighted_penalty:
        rhs = fft_blur_transpose(W * y, FBC) + alpha * W * z

        def H(v):
            return fft_blur_transpose(W * fft_blur(v, FB), FBC) + alpha * W * v

    else:
        rhs = fft_blur_transpose(W * y, FBC) + alpha * z

        def H(v):
            return fft_blur_transpose(W * fft_blur(v, FB), FBC) + alpha * v

    x = conjugate_gradient(H, rhs, x0=z, n_iter=cg_iter)

    return torch.clamp(x, 0.0, 1.0)