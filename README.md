# Spout for Python

Based on https://github.com/spiraltechnica/Spout-for-Python

A modified Spout library using Boost::Python to enable Spout texture sharing using Python.
This library is for use with Python 3.5 / 3.6 / 3.7 64bit. Now it will automatically define python version and load appropriate file.

First of all, you should uninstall completely ALL previous NVIDIA CUDA versions. I mean completely.

## Installation

Uninstall all previous NVIDIA CUDA as usual
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




## Using the Library

Watch video use/demo > 
[![](http://img.youtube.com/vi/CmI4zwSAajw/0.jpg)](http://www.youtube.com/watch?v=CmI4zwSAajw "Spout for Python")

```python test.py```
or just check sample code in the test.py
```
# import library
from Library.Spout import Spout

def main() :
    # create spout object
    spout = Spout(silent = True)
    # create receiver
    spout.createReceiver('input')
    # create sender
    spout.createSender('output')

    while True :
        # check on exit
        spout.check()
        # receive data
        data = spout.receive()
        # send data
        spout.send(data)
    
if __name__ == "__main__":
    main()
```

If want multiple receivers/senders, check ```test_mult.py```

## Parameters 
Parameters and arguments for sender and receiver can be checked in the ```Library/Spout.py```

## Requirements

```
pip install -r requirements.txt
```

- pygame
- pyopengl

## Additional
* Allow multiple receivers senders
* Now it can be used as any python library, just few lines of code
* Automatically define the size of receiver and data to send
* Can change receiver size on the go
* Support different receiver/sender imageFormat/type
