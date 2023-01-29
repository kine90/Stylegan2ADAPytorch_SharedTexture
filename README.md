# Spout for Python

Based on https://github.com/spiraltechnica/Spout-for-Python

A modified Spout library using Boost::Python to enable Spout texture sharing using Python.
This library is for use with Python 3.5 / 3.6 / 3.7 64bit. Now it will automatically define python version and load appropriate file.

First of all, you should uninstall completely ALL previous NVIDIA CUDA versions. I mean completely.

## Installation

First, you need a working CUDA Stylegan 2 - ADA Pytorch
This is the trickies part

Uninstall all previous NVIDIA CUDA
Go to your environment setup and remove ALL paths, CUDA_HOME, CUDA_PATH etc. cleanly
Delete all files as the uninstallation program left them there !!!
Install Visual Studio 2019
Add this to your PATH:
C:\Program Files (x86)\MicrosoftVisualStudio\2019\Community\VC\Auxiliary\Build\vcvarsx86_amd64.bat
Clean install NVIDIA CUDA 11.1
Check your PATH CUDA_HOME CUDA_PATH pointing to exactly the 11.1 path
Install pytorch (uninstall ALL previous installed one first)
conda install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f
https://download.pytorch.org/whl/torch_stable.html
Install relevant packages
pip install click requests tqdm pyspng ninja imageio-ffmpeg==0.4.3
Check nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2020 NVIDIA Corporation
Built on Tue_Sep_15_19:12:04_Pacific_Daylight_Time_2020
Cuda compilation tools, release 11.1, V11.1.74
Build cuda_11.1.relgpu_drvr455TC455_06.29069683_0
Hope that I did not miss any step. Otherwise, it should work. If it DOES NOT, Ctrl-C immediately and see if the path that is used is 11.0 or 11.1 or whatever earlier version.
P/S: on Windows you may hit a problem with OMP (Initializing libiomp5.dylib, but found libiomp5.dylib already initialized), simply ignore them by adding: os.environ['KMP_DUPLICATE_LIB_OK']='True' to your train.py and training_loop.py


Next, set up an environment using the file you find into folder "conda" and activate it.

Last, find a stylegan 2-ada pytorch pickle, or download one from Nvidia´s repo.
https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/


## Usage

There are two main working versions:

## generate_UDP_v3.py

This version expects to receive random seed and interpolation values from udp and sends back the generated image as shared texture.
For a list of udp commands check udp_commands.txt
Tensor z (with shape[1,512]) is generated using gaussian noise with the random seed received from UDP.
z is then mapped to latent w_samples.
From w_samples an image is generated.
The code blends z and w_using an interpolation factor wich is set trough udp.
Both are useful to smooth the transition of a frame into the previous one. Z interpolation tends to add variety to the transition, while W interpolation results in a more straight transition from image A to B. I usually go for a mix of the two.

## generate_receiveZ_v3.py

This version expects to receive the Z Tensor as a spout texture with resolution 512x1. Other parameters can be controlled trough UDP.
For a list of udp commands check udp_commands.txt
Tensor z is received from Spout. The pixel brightness is converted to float (range 0-1), scaled using the specified sclaed factor, and offset of half of it, as Stylegan expects noise with positive and negative values.
z is then mapped to latent w_samples.
From w_samples an image is generated and sent back as shared texture.
The code blends w_samples using an interpolation factor wich is set trough udp.

TO-DO:
Fully replace spout with shared texture, which is faster.