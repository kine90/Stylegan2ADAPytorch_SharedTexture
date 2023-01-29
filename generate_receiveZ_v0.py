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
import gfx2cuda
from Library.Spout import Spout
torch.set_grad_enabled(False)

######################################################
#####################################################
#########################################################

#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
def info(arr):
    """Shows statistics and shape information of (lists of) np.arrays/th.tensors
    Args:
        arr (np.array/th.tensor/list): List of or single np.array or th.tensor
    """
    if isinstance(arr, list):
        print([(list(a.shape), f"{a.min():.2f}", f"{a.max():.2f}") for a in arr])
    else:
        print(list(arr.shape), f"{arr.min():.2f}", f"{arr.max():.2f}")

#----------------------------------------------------------------------------
class gen_par:
    def __init__(self, net, truncation_psi, noise_mode):

        self.model = net
        self.seed = 0
        self.truncation_psi = truncation_psi
        self.z_int = 0
        self.w_int = 0.1
        self.noise_mode = noise_mode
        self.terminate = False
        self.amplitude = 4.0

class spout_pars:
    def __init__(self, name, silent):

        self.name = name
        self.silent = silent

#----------------------------------------------------------------------------

#UDP AND INPUT PROCESS
def udp_ops(q, first_gen_pars, ip, udp_in, udp_out):
    my_pars = first_gen_pars
    print("starting UDP.... IP={}, LISTEN= {}, SEND= {}".format(ip, udp_in, udp_out))
    mysock = socket.socket(socket.AF_INET, # Internet
                          socket.SOCK_DGRAM) # UDP
    mysock.settimeout(0.0001)

    #Bind to receiving ip and port
    try:
        mysock.bind((ip, udp_in))
    except:
        print("Can`t bind listening port!")
    print("done!")

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
        "seed": seed,
        "trunc": trunc,
        "zint": zint,
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
    
        
        
        
    


#----------------------------------------------------------------------------

def spout_send(handle, spout_pars, u, device):
    #SPOUT
    print("starting spout SENDER... NAME = {}, SILENT= {}".format(spout_pars.name, spout_pars.silent))
    spoutsnd = Spout(silent = spout_pars.silent , width = 1024, height = 1024 )
    spoutsnd.createSender(name = spout_pars.name)

    while True:
        u.get()
        tex = gfx2cuda.open_ipc_texture(handle)
        pic = torch.ones(1024, 1024, 4).to(device)
        with tex:
            tex.copy_to(pic)
        pic = (pic[:, :, :3]*255).clamp(0, 255).cpu().numpy()
        spoutsnd.send(pic)
        spoutsnd.check()
        time.sleep(0.002)

#----------------------------------------------------------------------------


#----------------------------------------------------------------------------
@click.command()
@click.pass_context
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1.2, show_default=True)
@click.option('--noise-mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), default='const', show_default=True)
@click.option('--spout-name', 'spout_name', help='Spout sender name', default="TorchSpout", show_default=True)
@click.option('--spout-window', 'spout_silent',  type=bool, help='Show window for spout output, default False', default=True) #to hide window silend should be true
@click.option('--udp-in', 'udp_in', type=int, help='UDP listening port', default=5005,show_default=True)
@click.option('--udp-out', 'udp_out', type=int, help='UDP output port', default=5006,show_default=True)
@click.option('--ip', 'ip', help='UDP IP to communicate with, default localhost', default="127.0.0.1",show_default=True)


######################################################################################################################
### MAIN #############################################################################################################
 
def main(
    ctx: click.Context,
    network_pkl: str,
    truncation_psi: float,
    noise_mode: str,
    spout_name: str,
    spout_silent: bool,
    udp_in: int,
    udp_out: int,
    ip: str,
):


    now_gen_par = gen_par(network_pkl, truncation_psi, noise_mode)

    q = mp.Queue(1)
    
    #gp = mp.Process(target=gen_proc, args=(q, now_gen_par, my_spout_pars))
    up = mp.Process(target=udp_ops, args=(q, now_gen_par, ip, udp_in, udp_out))
    up.daemon = True
    up.start()

    #NETWORK
    print('Loading networks from "%s"...' % now_gen_par.model)

    device = torch.device('cuda')
    with dnnlib.util.open_url(now_gen_par.model) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

    print("starting spout RECEIVER... NAME = latent")
    spoutrcv = Spout(silent = True , width = 512, height = 1 )
    spoutrcv.createReceiver(name = "latent")
    time.sleep(2)

    z = torch.reshape((torch.from_numpy(spoutrcv.receive())[:,:,:1]), (1,512))
    z = ((z - 127.5) * ( now_gen_par.amplitude / 127.5) ).to(device, dtype = torch.float)
    print("Received spout data, generating...")

    w_samples = G.mapping(z, None,  truncation_psi=now_gen_par.truncation_psi)
    oldw_samples = w_samples

    img = G.synthesis(w_samples, noise_mode=now_gen_par.noise_mode)
    # synth img TENSOR SIZE IS: torch.Size([1, 3, 1024, 1024])
    print("NETWORK READY")

    img = (img[0].permute( 1, 2, 0) +0.5).clamp(0, 1).to(dtype=torch.float)
    alpha = torch.full((1024, 1024, 1), 1.0, dtype = torch.float).to(device)
    img = torch.cat((img, alpha), 2)#.to(dtype=torch.uint8)
 
    outtex = gfx2cuda.texture(img)
    with outtex:
        outtex.copy_from(img)
    print("SHARED TEXTURE INIT DONE. HANDLE = {}".format(outtex.ipc_handle))

    print("All tests passed. Enetring loop...")

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
        print("UDP check took {}s".format((time.perf_counter() - t0)))
        t0 = time.perf_counter()
        #CHECK SPOUT
        z = torch.reshape((torch.from_numpy(spoutrcv.receive())[:,:,:1]), (1,512))# * (1/127.5) - 128).permute(1,0).to(device, dtype = torch.float)
        z = ((z - 127.5) * ( 4.0 / 127.5) ).to(device, dtype = torch.float)
        print("SPOUT check took {}s".format((time.perf_counter() - t0)))
        t0 = time.perf_counter()
        #GENERATE W SAMPLES
        w_samples = G.mapping(z, None,  truncation_psi=now_gen_par.truncation_psi)
        if now_gen_par.w_int != 0:
            w_samples = ((w_samples * now_gen_par.w_int) + (oldw_samples * (1 - now_gen_par.w_int)))
        oldw_samples = w_samples
        print("W_SAMPLES took {}s".format((time.perf_counter() - t0)))
        t0 = time.perf_counter()
        img = G.synthesis(w_samples, noise_mode=now_gen_par.noise_mode)
        print("IMG SYNTH took {}s".format((time.perf_counter() - t0)))
        t0 = time.perf_counter()
        img = (img[0].permute( 1, 2, 0) +0.5).clamp(0, 1)#.to(dtype=torch.float)
        img = torch.cat((img, alpha), 2)
        print("IMG OPS took {}s".format((time.perf_counter() - t0)))
        t0 = time.perf_counter()
        with outtex:
            outtex.copy_from(img)

        elaps = (time.perf_counter() - t0)
        print("GENERATED FRAME. TOOK {}s - FPS {} - HANDLE = {}".format(elaps, (1/elaps), outtex.ipc_handle))
        

        

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
