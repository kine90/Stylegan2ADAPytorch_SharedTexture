# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Generate images using pretrained network pickle."""

import os
import re
from typing import List, Optional
import torch.multiprocessing as mp
import click
import dnnlib
import numpy as np
import torch
import socket
import legacy
import time
import gfx2cuda

torch.set_grad_enabled(False)
pin_memory = True

######################################################
#####################################################
#########################################################

# UTILITIES ----------------------------------------------------------------------------
def info(arr):
    """Shows statistics and shape information of (lists of) np.arrays/th.tensors
    Args:
        arr (np.array/th.tensor/list): List of or single np.array or th.tensor
    """
    if isinstance(arr, list):
        print([(list(a.shape), f"{a.min():.2f}", f"{a.mean():.2f}", f"{a.max():.2f}") for a in arr])
    else:
        print(list(arr.shape), f"{arr.min():.2f}", f"{arr.mean():.2f}", f"{arr.max():.2f}")

# DATA STRUCTURES----------------------------------------------------------------------------
class gen_par:
    def __init__(self, net, truncation_psi, noise_mode):

        self.model = net
        self.seed = 0
        self.truncation_psi = truncation_psi
        self.z_int = 0
        self.w_int = 0.1
        self.noise_mode = noise_mode
        self.terminate = False

class spout_pars:
    def __init__(self, name, silent):

        self.name = name
        self.silent = silent

# UDP INPUT PROCESS ----------------------------------------------------------------------------
def udp_ops(q, first_gen_pars, udp_in):
    my_pars = first_gen_pars
    print("starting UDP... LISTENING PORT = {}".format(udp_in))
    mysock = socket.socket(socket.AF_INET, # Internet
                          socket.SOCK_DGRAM) # UDP
    mysock.settimeout(0.0001)

    #Bind to receiving ip and port
    try:
        mysock.bind(("127.0.0.1", udp_in))
    except:
        print("Can`t bind listening port!")
    print("done!")
    
    # DEFINE UDP STRINGS DECODING
    def seed():
        my_pars.seed = int(msg)
        print("Received new seed: {}".format(my_pars.seed))
    def trunc():
        my_pars.truncation_psi = (int(msg)/200)-5
        print("Received new Truncation PSI: {}".format(my_pars.truncation_psi))
    def zint():
        my_pars.z_int = int(msg)/2000
        print("Received new Z Interpolation step value: {}".format(my_pars.z_int))
    def wint():
        my_pars.w_int = int(msg)/2000
        print("Received new W Interpolation step value: {}".format(my_pars.w_int))
    def noisemode():
        switch = {
            0: 'const',
            1: 'random',
            2: 'none'
        }
        my_pars.noise_mode = switch.get(int(msg), 'none')
        print("Received new Noise Mode: {}".format(my_pars.noise_mode))
    def askclose():
        my_pars.terminate = True
        print("Received EXIT request")
    
    switch = {
        "seed": seed,
        "trunc": trunc,
        "zint": zint,
        "wint": wint,
        "noisemode": noisemode,
        "terminate": askclose
    }
    
    time.sleep(20)
    # LOOP
    while True:
        time.sleep(0.001)
        try:
            #RECEIVE UDPs
            data, addr = mysock.recvfrom(1024) # buffer size is 1024 bytes
            msg_type, msg = data.decode('utf-8').strip().split("_")
            #DECODE STRING
            switch[msg_type]()
            #PUT PARS INTO QUEUE
            q.put_nowait(my_pars)
        except:
            #Nothing new from udp...
            pass


#----------------------------------------------------------------------------
@click.command()
@click.pass_context
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1.2, show_default=True)
@click.option('--noise-mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), default='const', show_default=True)
@click.option('--udp-in', 'udp_in', type=int, help='UDP listening port', default=5005,show_default=True)


######################################################################################################################
### MAIN #############################################################################################################
 
def main(
    ctx: click.Context,
    network_pkl: str,
    truncation_psi: float,
    noise_mode: str,
    udp_in: int,
):


    now_gen_par = gen_par(network_pkl, truncation_psi, noise_mode)

    q = mp.Queue(1)
    
    up = mp.Process(target=udp_ops, args=(q, now_gen_par, udp_in))
    up.daemon = True
    up.start()

    print('Loading networks from "%s"...' % now_gen_par.model)
    device = torch.device('cuda')
    with dnnlib.util.open_url(now_gen_par.model) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)
        
    with torch.no_grad(): #TORCH NOGRAD SHOULD SPEED THINGS UP
        z = torch.from_numpy(np.random.RandomState(now_gen_par.seed).randn(1, G.z_dim)).to(device)
        oldz = z
        w_samples = G.mapping(oldz, None,  truncation_psi=now_gen_par.truncation_psi)
        oldw_samples = w_samples

        #Generate a first image so we can properly initialize the shared texture
        img = G.synthesis(w_samples, noise_mode=now_gen_par.noise_mode).contiguous().cuda(non_blocking=pin_memory)
        img = (img[0].permute( 1, 2, 0) +0.5).clamp(0, 1).to(dtype=torch.float)
        # synth img TENSOR SIZE IS: torch.Size([1, 3, 1024, 1024])
        alpha = torch.full((1024, 1024, 1), 1.0, dtype = torch.float).cuda(non_blocking=pin_memory).to(device)
        img = torch.cat((img, alpha), 2).contiguous()
        #init ouput texture
    outtex = gfx2cuda.texture(img)

    print("Network and sharedtexture initialized!")
    print("HANDLE = {}".format(outtex.ipc_handle))

    elaps = 0
    while True:
        t0 = time.perf_counter()
        try:
            now_gen_par = q.get_nowait()
        except:
            pass
        if now_gen_par.terminate:
            print("EXITING")
            exit()

        with torch.no_grad(): #TORCH NOGRAD SHOULD SPEED THINGS UP
            z = torch.from_numpy(np.random.RandomState(now_gen_par.seed).randn(1, G.z_dim)).to(device)
            if now_gen_par.z_int != 0:  
                z = ((z * now_gen_par.z_int) + (oldz * (1 - now_gen_par.z_int)))
            oldz = z
            w_samples = G.mapping(z, None,  truncation_psi=now_gen_par.truncation_psi)
            del z
            if now_gen_par.w_int != 0:
                w_samples = ((w_samples * now_gen_par.w_int) + (oldw_samples * (1 - now_gen_par.w_int)))
            oldw_samples = w_samples
            img = G.synthesis(w_samples, noise_mode=now_gen_par.noise_mode).contiguous().cuda(non_blocking=pin_memory)
            del w_samples
            img = (img[0].permute( 1, 2, 0) +0.5).clamp(0, 1).to(dtype=torch.float)
            img = torch.cat((img, alpha), 2).contiguous()
        with outtex:
            outtex.copy_from(img)
        del img
        elaps = 1/(time.perf_counter() - t0)
        print('Seed {} - FPS {} - HANDLE {}'.format(now_gen_par.seed, elaps, outtex.ipc_handle))

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
