#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test_SC_EFISTA.py -- proposed PnP-EFISTA (DRUNet denoiser) for single-image
super-resolution (scale factor 2).

Reproduces the "Proposed" column of Table 2 (Set3c and SRBSD68, first 8 kernels
of kernels_12, AWGN 10%-25%) and the PnP-EFISTA result of Fig. 4.

Output: restored images, bicubic LR images and one Excel workbook with three
sheets (Avg_Dataset, Avg_Kernel, All_Kernels) in results/SC/.
"""

import os
import sys
import time
import gc
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import hdf5storage
import cv2
import matplotlib.pyplot as plt
from skimage import io
from scipy.fftpack import fftn, ifftn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repository root
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'SC'))

from utils_sr.utils_deblur1 import psf2otf
from utils_sr import utils_image as util
from SC_EFISTA_subcode import All_Algo, UNetRes, downsamplec


def tic():
    return time.time()

def toc(start):
    return time.time() - start

# Paths and Settings
testset = os.path.join(ROOT, 'testsets')
testset_name = 'set3c'           # 'set3c' | 'srbsd68'
select_images = []               # [] = all images; e.g. ['butterfly.png'] for Fig. 4
kernel_folder = 'kernels_12'     # kernels of Zhang et al. (USRNet), first 8 are used
Select_appl = 'super-resolution'
task_current = 'sr'
model_name = 'drunet_color'
model_zoo = os.path.join(ROOT, 'model_zoo')
algorithms = ['EFISTA']          # 'EFISTA' (proposed) | 'FISTA' (PnP-FISTA reference)
results_dir = os.path.join(ROOT, 'results', 'SC')
save_L = True                    # also save the bicubic-upsampled LR observation


# Device Setup
start_time = tic()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Get current time
current_time = datetime.now().strftime("%Y%m%d")

print(f"----Selected Image Application: {Select_appl}----")


# Load Model
if 'drunet' in model_name:
    model = UNetRes(in_nc=4, out_nc=3, nc=[64, 128, 256, 512], nb=4, act_mode='R', downsample_mode="strideconv", upsample_mode="convtranspose")
    model.load_state_dict(torch.load(os.path.join(model_zoo, 'drunet_color.pth'), map_location=device), strict=True)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    model = model.to(device)
else:
    raise ValueError("Error: select a valid model_name (drunet_color)")


# Kernel Load
if 'kernels_12' in kernel_folder:
    kernels = hdf5storage.loadmat(os.path.join(ROOT, 'kernels', 'kernels_12.mat'))['kernels']
elif 'Levin' in kernel_folder:
    kernels = hdf5storage.loadmat(os.path.join(ROOT, 'kernels', 'Levin09.mat'))['kernels']  # 8 kernels


# Algorithm Parameters
MAXITER = 18            # Section 6.2.1
nw = 8                  # number of fast-forward gradient steps
p = 7                   # EFISTA noise-scaling parameter
lambda_val = 0.35       # lambda = 0.35 * Sigma^2
sf = 2                  # scale factor
Eta = 0.9               # gradient step size
var = [0.10, 0.15, 0.20, 0.25]   # AWGN standard deviations (10%, 15%, 20%, 25%)
show_img = False

# Image Files
image_folder = os.path.join(testset, testset_name)
image_files = sorted(os.listdir(image_folder))
if select_images:
    image_files = [f for f in image_files if f in select_images]

# Output Folder
result_name = f"SC_{testset_name}_{task_current}_{kernel_folder}_sf_{sf}_{model_name}_lam_{lambda_val}_P{p}_{'_'.join(algorithms)}_{MAXITER}"
output_folder = os.path.join(results_dir, result_name)
os.makedirs(output_folder, exist_ok=True)

# Initialize accumulators
rows_all_kernel  = []   # every image x kernel x noise row
rows_avg_kernel  = []   # per image x noise (averaged over kernels)
rows_avg_dataset = []   # per noise (averaged over all images x kernels)

# Main Loop
for image_file in image_files:
    image_path = os.path.join(image_folder, image_file)
    X          = io.imread(image_path).astype(float) / 255.0
    X          = util.modcrop(X, sf)
    H, W       = X.shape[:2]
    base_name  = os.path.splitext(image_file)[0]

    for Sigma in var:
        tol = -np.inf    # run all MAXITER iterations

        # per-image accumulators (averaged over kernels)
        img_psnr, img_ssim, img_lpips, img_fsim = [], [], [], []
        img_best_psnr, img_best_ssim            = [], []
        img_best_lpips, img_best_fsim           = [], []

        for k_index in range(8):
            P        = kernels[0, k_index].astype(np.float64)
            ker_name = str(k_index + 1)

            print(f"Image: {base_name}  Kernel: {ker_name}  Noise: {Sigma}  SF: {sf}")

            # Blur (circular convolution) and s-fold downsampling
            k_pad, k_fft = psf2otf(P, [H, W])
            k_fft        = np.repeat(k_fft[:, :, np.newaxis], 3, axis=2)
            B1           = np.real(ifftn(fftn(X) * k_fft))
            B1           = downsamplec(B1, sf)

            for Select_algo in algorithms:
                np.random.seed(0)
                Bobs = B1 + Sigma * np.random.randn(*B1.shape)

                X_out, fun_all, PSNR, SSIM, LPIPS, FSIM, out_img = All_Algo(
                    X, P, sf, Bobs, MAXITER, nw, p, Sigma, lambda_val,
                    tol, Eta, show_img, Select_algo, model_name, net=model
                )

                # Save output image
                out_path = os.path.join(
                    output_folder,
                    f'{base_name}_out_{model_name}_{Select_algo}_sig_{Sigma}_k{ker_name}'
                    f'_psnr{PSNR[-1]:.4f}_ssim{SSIM[-1]:.4f}.png'
                )
                plt.imsave(out_path, np.clip(X_out, 0, 1))
                if save_L:
                    img_L_disp = cv2.resize(Bobs, (Bobs.shape[1] * sf, Bobs.shape[0] * sf),
                                            interpolation=cv2.INTER_CUBIC)
                    psnr_b = util.calculate_psnr(img_L_disp * 255, X * 255, border=0)
                    out_path = os.path.join(
                        output_folder,
                        f'{base_name}_sf{sf}_N{Sigma}_k{ker_name}_{psnr_b:.4f}_LR.png'
                    )
                    plt.imsave(out_path, np.clip(img_L_disp, 0, 1))

                # Best-iteration indices
                best_psnr_idx  = int(np.argmax(PSNR))
                best_ssim_idx  = int(np.argmax(SSIM))
                best_lpips_idx = int(np.argmin(LPIPS))
                best_fsim_idx  = int(np.argmax(FSIM))

                # Sheet 3: every row
                rows_all_kernel.append({
                    'Noise':          Sigma,
                    'Image':          base_name,
                    'Kernel':         ker_name,
                    'SF':             sf,
                    'Algorithm':      Select_algo,
                    # Final-iteration metrics
                    'PSNR':           round(PSNR[-1],  4),
                    'SSIM':           round(SSIM[-1],  4),
                    'LPIPS':          round(LPIPS[-1], 4),
                    'FSIM':           round(FSIM[-1],  4),
                    # Best metrics
                    'BestPSNR':       round(PSNR[best_psnr_idx],   4),
                    'BestPSNR_iter':  best_psnr_idx,
                    'BestSSIM':       round(SSIM[best_ssim_idx],   4),
                    'BestSSIM_iter':  best_ssim_idx,
                    'BestLPIPS':      round(LPIPS[best_lpips_idx], 4),
                    'BestLPIPS_iter': best_lpips_idx,
                    'BestFSIM':       round(FSIM[best_fsim_idx],   4),
                    'BestFSIM_iter':  best_fsim_idx,
                })

                # Accumulate for per-image average
                img_psnr.append(PSNR[-1]);   img_ssim.append(SSIM[-1])
                img_lpips.append(LPIPS[-1]); img_fsim.append(FSIM[-1])
                img_best_psnr.append(PSNR[best_psnr_idx])
                img_best_ssim.append(SSIM[best_ssim_idx])
                img_best_lpips.append(LPIPS[best_lpips_idx])
                img_best_fsim.append(FSIM[best_fsim_idx])

                torch.cuda.empty_cache()
                gc.collect()

        # END kernel loop

        # Sheet 2: per image x noise averaged over kernels
        rows_avg_kernel.append({
            'Noise':         Sigma,
            'Image':         base_name,
            'PSNR-avg':      round(np.mean(img_psnr),       4),
            'SSIM-avg':      round(np.mean(img_ssim),       4),
            'LPIPS-avg':     round(np.mean(img_lpips),      4),
            'FSIM-avg':      round(np.mean(img_fsim),       4),
            'BestPSNR-avg':  round(np.mean(img_best_psnr),  4),
            'BestSSIM-avg':  round(np.mean(img_best_ssim),  4),
            'BestLPIPS-avg': round(np.mean(img_best_lpips), 4),
            'BestFSIM-avg':  round(np.mean(img_best_fsim),  4),
        })

# END image loop

# Sheet 1: per noise averaged over entire dataset
df_all = pd.DataFrame(rows_all_kernel)
for Sigma in var:
    sub = df_all[df_all['Noise'] == Sigma]
    rows_avg_dataset.append({
        'Noise':         Sigma,
        'PSNR-avg':      round(sub['PSNR'].mean(),      4),
        'SSIM-avg':      round(sub['SSIM'].mean(),      4),
        'LPIPS-avg':     round(sub['LPIPS'].mean(),     4),
        'FSIM-avg':      round(sub['FSIM'].mean(),      4),
        'BestPSNR-avg':  round(sub['BestPSNR'].mean(),  4),
        'BestSSIM-avg':  round(sub['BestSSIM'].mean(),  4),
        'BestLPIPS-avg': round(sub['BestLPIPS'].mean(), 4),
        'BestFSIM-avg':  round(sub['BestFSIM'].mean(),  4),
    })

# Autofit helper
def autofit(ws):
    for col in ws.columns:
        width = max((len(str(c.value)) for c in col if c.value), default=8)
        ws.column_dimensions[col[0].column_letter].width = width + 4

# Write single Excel workbook
excel_path = os.path.join(
    output_folder,
    f'SC_EFISTA_{task_current}_{testset_name}_sf{sf}_{current_time}_results.xlsx'
)

with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:

    # Sheet 1 -- dataset avg per noise (values reported in Table 2)
    pd.DataFrame(rows_avg_dataset).to_excel(
        writer, sheet_name='Avg_Dataset', index=False)
    autofit(writer.sheets['Avg_Dataset'])

    # Sheet 2 -- per image x noise averaged over kernels
    pd.DataFrame(rows_avg_kernel).to_excel(
        writer, sheet_name='Avg_Kernel', index=False)
    autofit(writer.sheets['Avg_Kernel'])

    # Sheet 3 -- every image x kernel x noise row
    df_all.to_excel(writer, sheet_name='All_Kernels', index=False)
    autofit(writer.sheets['All_Kernels'])

print(f"\nExcel saved -> {excel_path}")
print("Elapsed time:", toc(start_time), "seconds")
