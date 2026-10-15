# PnP-EFISTA-
Plug-and-Play EFISTA for Image Restoration under High Measurement Noise
# Deep Plug-and-Play Image Restoration

![visitors](https://visitor-badge.glitch.me/badge?page_id=cszn/DPIR) 

[**Avinash Kumar**](https://cszn.github.io/), Koyyada Dinesh, Sujit Kumar Sahoo

 [[paper tpami](https://ieeexplore.ieee.org/abstract/document/9454311)]


Denoising results on BSD68 and Urban100 datasets
----------
|  Dataset    | Noise Level | FFDNet-PSNR(RGB) | FFDNet-PSNR(Y) | **DRUNet-PSNR(RGB)** | **DRUNet-PSNR(Y)** |
|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|
| CBSD68 |  30    | 30.32  | 32.05  |  30.81 | 32.44 |
| CBSD68 |   50   | 27.97  | 29.65  |  28.51 | 30.09 |
| Urban100|  30   | 30.53  | 32.72  |  31.83 | 33.93 |
| Urban100|   50  | 28.05  | 30.09  |  29.61 | 31.57 |

```PSNR(Y) means the PSNR is calculated on the Y channel of YCbCr space.```


Abstract
----------
Image restoration problems, such as image deblurring and super-resolution, remain challenging due to their ill-posed nature, particularly in the presence of high measurement noise. The Fast Iterative Shrinkage-Thresholding Algorithm (FISTA) has been widely used for solving linear inverse problems, with several enhancements and extensions developed over time. In this paper, we propose a Plug-and-Play Enhanced FISTA (PnP-EFISTA) framework that incorporates fast-forward accelerated gradient-descent steps and adaptive denoising to improve restoration. The proposed framework enables the integration of learning-based denoisers without requiring modifications to the underlying restoration model. Extensive experiments on image deblurring and super-resolution demonstrate that PnP-EFISTA is particularly effective at high AWGN levels, achieving competitive or improved Peak Signal-to-Noise Ratio (PSNR) and Structural Similarity Index measure (SSIM) performance compared with state-of-the-art restoration algorithms.

The DRUNet Denoiser (state-of-the-art Gaussian denoising performance!)
----------
* Network architecture

  <img src="figs/denoiser_arch.png" width="800px"/> 

* Grayscale image denoising

  <img src="figs/grayscale_psnr.png" width="800px"/> 

* Color image denoising

  <img src="figs/color_psnr.png" width="375px"/> 
  
|<img align="center" src="figs/test_03_noisy_0750.png" width="320px"/> | <img align="center" src="figs/test_03_usrnet_2355.png" width="320px"/> |
|:---:|:---:|
|<i>(a) Noisy image with noise level 200</i>|<i>(b) Result by the proposed DRUNet denoiser</i>|

**Even trained on noise level range of [0, 50], DRUNet can still perform well on an extremely large unseen noise level of 200.**

Image Deblurring
----------
* Visual comparison

  <img src="figs/deblur_1.png" width="800px"/> 


Single Image Super-Resolution
----------
* Visual comparison

  <img src="figs/sisr_1.png" width="800px"/> 


Color Image Demosaicing
----------
* PSNR

  <img src="figs/demosaic_1.png" width="800px"/> 

* Visual comparison

  <img src="figs/demosaic_2.png" width="800px"/> 



Citation
----------
```BibTex
@article{zhang2021plug,
  title={Plug-and-Play Image Restoration with Deep Denoiser Prior},
  author={Zhang, Kai and Li, Yawei and Zuo, Wangmeng and Zhang, Lei and Van Gool, Luc and Timofte, Radu},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  volume={44},
  number={10},
  pages={6360-6376},
  year={2021}
}
 @inproceedings{zhang2017learning,
   title={Learning Deep CNN Denoiser Prior for Image Restoration},
   author={Zhang, Kai and Zuo, Wangmeng and Gu, Shuhang and Zhang, Lei},
   booktitle={IEEE Conference on Computer Vision and Pattern Recognition},
   pages={3929--3938},
   year={2017},
 }
```
