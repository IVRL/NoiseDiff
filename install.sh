#!/bin/bash

pip install torch==2.0.1+cu117 -f https://download.pytorch.org/whl/torch_stable.html
pip install torchvision==0.15.2+cu117 -f https://download.pytorch.org/whl/torch_stable.html

pip install \
  tensorboardX \
  rawpy \
  matplotlib \
  opencv-python \
  exifread \
  ema-pytorch \
  denoising_diffusion_pytorch \
  scikit-image \
  -U numba \
  lpips \
  yacs \
  scikit-learn