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
import pygame
from pygame.locals import *
from Library.Spout import Spout
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

#----------------------------------------------------------------------------
class gen_par:
    def __init__(self, net, truncation_psi, noise_mode, input_handle):

        self.model = net
        self.seed = 0
        self.truncation_psi = truncation_psi
        self.w_int = 0.1
        self.noise_mode = noise_mode
        self.terminate = False
        self.handle = input_handle
        self.amplitude = 4.0

class spout_pars:
    def __init__(self, name, silent):

        self.name = name
        self.silent = silent

#----------------------------------------------------------------------------

#UDP AND INPUT PROCESS
def udp_ops(q, first_gen_pars, udp_in):
    my_pars = first_gen_pars
    print("starting UDP....  LISTENING PORT = {}".format(udp_in))
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
    def trunc():
        my_pars.truncation_psi = (int(msg)/200)-5
        print("Received new Truncation PSI: {}".format(my_pars.truncation_psi))
    def wint():
        my_pars.w_int = int(msg)/2000
        print("Received new W Interpolation step value: {}".format(my_pars.w_int))
    def amplitude():
        my_pars.amplitude = float(int(msg))
        print("Received new amplitude value: {}".format(my_pars.amplitude))
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
        "trunc": trunc,
        "wint": wint,
        "noisemode": noisemode,
        "terminate": askclose,
        "amplitude": amplitude
    }
    

    while True:
        #do things with data
        #print("executing udp ops")
        time.sleep(0.001)
        try:
            data, addr = mysock.recvfrom(1024) # buffer size is 1024 bytes
            #print("received message: %s" % data)
            msg_type, msg = data.decode('utf-8').strip().split("_")
            #print("type: {} msg: {}".format(msg_type, msg))
            switch[msg_type]()
            q.put_nowait(my_pars)
            #print("queue added!")
        except:
            #print("nothing to add....")
            pass


#COMMANDLINE PARAMETERS ----------------------------------------------------------------------------
@click.command()
@click.pass_context
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--noise-mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), default='const', show_default=True)
@click.option('--spout-name', 'spout_name', help='Spout receiver name', default="latent", show_default=True)
@click.option('--spout-window', 'spout_silent',  type=bool, help='Show window for spout output, default hidden', default=True) #to hide window silend should be true
@click.option('--udp-in', 'udp_in', type=int, help='UDP listening port', default=5005,show_default=True)
@click.option('--handle', 'handle', type=int, help='Handle for incoming texture tensor',  required=True)
######################################################################################################################
### MAIN #############################################################################################################
 
def main(
    ctx: click.Context,
    network_pkl: str,
    noise_mode: str,
    spout_name: str,
    spout_silent: bool,
    udp_in: int,
    handle = int,
):
    # LOAD PARS TO DATASTRUCTURES
    now_gen_par = gen_par(network_pkl, 1.0, noise_mode, handle)
    now_spout_pars = spout_pars(spout_name, spout_silent)
    #INIT mpQUEUE FOR UDP INPUTS TRANSFER
    q = mp.Queue(1)
    
    #START UDP LISTENER PROCESS
    up = mp.Process(target=udp_ops, args=(q, now_gen_par, udp_in))
    up.daemon = True
    up.start()
    print("UDP init done")
    print("For info about commands check udp_commands.txt")

    #STYLEGAN2 NETWORK INIT
    print('Loading networks from "%s"...' % now_gen_par.model)
    device = torch.device('cuda')
    with dnnlib.util.open_url(now_gen_par.model) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)
    print("Network init done")

    # GENERATE A WHITE FRAME AND USE IT TO INIT SHARED TEXTURE OUTPUT
    firstframe = torch.full((1024, 1024, 4), 1.0, dtype = torch.float).contiguous().to(device)
    outtex = gfx2cuda.texture(firstframe)
    print("Shared tex init done")
    with outtex:
        outtex.copy_from(firstframe)
    print("HANDLE = {}".format(outtex.ipc_handle))
    del firstframe

    #RECEIVER INIT
    print("starting texture RECEIVER on handle {}".format(now_gen_par.handle))
    try:
        inputtex = gfx2cuda.open_ipc_texture(now_gen_par.handle)
    except:
        print("Failed to acess input texture on specified handle")
        exit()

    z = torch.full((1, 512), 1.0, dtype = torch.float).contiguous()
    with inputtex:
        inputtex.copy_to(z)
    print("First tensor received!")
    print(z.shape())
    print(z)
    #RECEIVE FIRST SPOUT LATENT
    #z = torch.reshape((torch.from_numpy(spoutrcv.receive())[:,:,:1]), (1,512))
    #z = ((z - 127.5) * ( now_gen_par.amplitude / 127.5) ).to(device, dtype = torch.float)
    print("Spout init done, first latent received")

    #GENERATE FIRST W_SAMPLES
    with torch.no_grad(): #TORCH NOGRAD SHOULD SPEED THINGS UP
        w_samples = G.mapping(z, None,  truncation_psi=now_gen_par.truncation_psi)
    oldw_samples = w_samples

    #GENERATE ALPHA CHANNEL LATENT AND STORE IT TO GPU FOR LATER
    alpha = torch.full((1024, 1024, 1), 1.0, dtype = torch.float).cuda(non_blocking=pin_memory).to(device)

    print("All done! Entering loop...")

    #LOOP
    elaps = 0
    while True:
        t0 = time.perf_counter()
           
        # CHECK NEW UDP DATA FROM QUEUE
        try:
            now_gen_par = q.get_nowait()
        except:
            pass

        # CHECK FOR TERMINATE COMMAND
        if now_gen_par.terminate:
            print("Exiting, bye!")
            time.sleep(2)
            exit()

        # RECEIVE LATENT SPOUT TEXTURE
        z = torch.reshape((torch.from_numpy(spoutrcv.receive())[:,:,:1]), (1,512))
        z = ((z - 127.5) * ( now_gen_par.amplitude / 127.5) ).to(device, dtype = torch.float)

        with torch.no_grad(): #TORCH NOGRAD SHOULD SPEED THINGS UP
            #GENERATE W_SAMPLES
            w_samples = G.mapping(z, None,  truncation_psi=now_gen_par.truncation_psi)
            del z
            # INTERPOLAE W
            if now_gen_par.w_int != 0:
                w_samples = ((w_samples * now_gen_par.w_int) + (oldw_samples * (1 - now_gen_par.w_int)))
            oldw_samples = w_samples
            # GENERATE IMAGE
            img = G.synthesis(w_samples, noise_mode=now_gen_par.noise_mode).contiguous().cuda(non_blocking=pin_memory)
            del w_samples
            # PUT IMAGE CHANNELS IN THE PROPER ORDER AND ADD ALPHA CHANNEL
            img = (img[0].permute( 1, 2, 0) +0.5).clamp(0, 1).to(dtype=torch.float)
            img = torch.cat((img, alpha), 2).contiguous()
        # TRANSFER IMAGE TO SHARED TEXTURE
        with outtex:
            outtex.copy_from(img)
        del img
        
        elaps = (time.perf_counter() - t0)
        print("NEW FRAME GENERATED. FPS {} - HANDLE = {}".format((1/(elaps)), outtex.ipc_handle))
        
        

        

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
