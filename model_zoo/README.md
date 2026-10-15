# model_zoo

Put the three pretrained networks used by the test scripts in this folder:

| File | Used by | Source |
|---|---|---|
| `drunet_color.pth` | PnP-EFISTA (proposed), DPIR | DRUNet of Zhang et al. (DPIR / KAIR): [direct link](https://github.com/cszn/KAIR/releases/download/v1.0/drunet_color.pth), or the [Google Drive folder](https://drive.google.com/drive/folders/13kfr3qny7S2xwG9h7v95F5mkWs0OmU0D?usp=sharing) listed in the [DPIR model zoo](https://github.com/cszn/DPIR/tree/master/model_zoo) |
| `GSDRUNet.ckpt` | RED-Prox, PnP-SGD | Color gradient-step DRUNet of Hurault et al.: [download](https://plmbox.math.cnrs.fr/f/ab6829cb933743848bef/?dl=1) (from the [GS-PnP repository](https://github.com/samuro95/GSPnP)); save it as `GSDRUNet.ckpt` |
| `model_DWDN.pt` | DWDN | Deep Wiener Deconvolution Network of Dong et al.: [download](https://gitlab.mpi-klsb.mpg.de/jdong/dwdn/-/blob/master/model/model_DWDN.pt) (from the [DWDN repository](https://github.com/dongjxjx/dwdn)) |

Expected layout:

```
model_zoo/
├── drunet_color.pth
├── GSDRUNet.ckpt
└── model_DWDN.pt
```

The weights are not tracked by git (see `.gitignore`); `drunet_color.pth` (~131 MB) and
`GSDRUNet.ckpt` (~204 MB) exceed GitHub's 100 MB file limit.
