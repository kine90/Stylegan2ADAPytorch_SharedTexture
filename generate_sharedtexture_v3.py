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

######################################################
#####################################################
#########################################################

#SETTINGS
udp_timeout = 0.0001


#----------------------------------------------------------------------------


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
        self.inputhandle = 2147487170,

class spout_pars:
    def __init__(self, name, silent):

        self.name = name
        self.silent = silent

#----------------------------------------------------------------------------

#UDP AND INPUT PROCESS
def udp_ops(q, first_gen_pars, socket):
    my_pars = first_gen_pars
    mysock = socket


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

##IMEAGE GENERATOR LOOP
def gen_proc (q, first_gen_pars, spout_pars):
    #NETWORK
    print("Starting generator process....")
    now_gen_par = first_gen_pars

    print('Loading networks from "%s"...' % now_gen_par.model)
    device = torch.device('cuda')
    with dnnlib.util.open_url(now_gen_par.model) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore
    label = torch.zeros([1, G.c_dim], device=device)
    print("Generator load Done!")

    #SPOUT
    print("starting spout... NAME = {}, SILENT= {}".format(spout_pars.name, spout_pars.silent))
    spout = Spout(silent = spout_pars.silent , width = G.img_resolution, height = G.img_resolution )
    spout.createSender(name = spout_pars.name)


    oldz = torch.from_numpy(np.random.RandomState(now_gen_par.seed).randn(1, G.z_dim)).to(device)
    oldw_samples = G.mapping(oldz, None,  truncation_psi=now_gen_par.truncation_psi)

    #Generate a first image so we can properly initialize the shared texture
    w_samples = G.mapping(oldz, None,  truncation_psi=now_gen_par.truncation_psi)
    img = G.synthesis(w_samples, noise_mode=now_gen_par.noise_mode)
    img = (img[0].permute( 1, 2, 0) * 0.5 + 0.5)
    # synth img TENSOR SIZE IS: torch.Size([1, 3, 1024, 1024])
    alpha = torch.ones(1024, 1024, 1).to(device)
    img = torch.cat((img, alpha), 2)
    print(img.dtype)
    print(img.shape)
    #init ouput texture
    outtex = gfx2cuda.texture(img)
    print("Shared Texture Output HANDLE: {}".format(outtex.ipc_handle))
    print("done!")
    spout.send(outtex)
    spout.check()
    



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

        print('Generating image for seed {} Last took {}s. HANDLE {}'.format(now_gen_par.seed, elaps, outtex.ipc_handle))
        z = torch.from_numpy(np.random.RandomState(now_gen_par.seed).randn(1, G.z_dim)).to(device)
        # z TENSOR SIZE IS: torch.Size([1, 512])
        if now_gen_par.z_int != 0:  
            z = ((z * now_gen_par.z_int) + (oldz * (1 - now_gen_par.z_int)))
        oldz = z

        w_samples = G.mapping(z, None,  truncation_psi=now_gen_par.truncation_psi)
        if now_gen_par.w_int != 0:
            w_samples = ((w_samples * now_gen_par.w_int) + (oldw_samples * (1 - now_gen_par.w_int)))
        oldw_samples = w_samples
        
        #img = G(z, label, truncation_psi=now_gen_par.truncation_psi, noise_mode=now_gen_par.noise_mode)
        img = G.synthesis(w_samples, noise_mode=now_gen_par.noise_mode)
        # synth img TENSOR SIZE IS: torch.Size([1, 3, 1024, 1024])
        img = (img[0].permute( 1, 2, 0) * 0.5 + 0.5)
        img = torch.cat((img, alpha), 2)
        with outtex:
            outtex.copy_from(img)
        spout.send(outtex)
        spout.check()

        elaps = time.perf_counter() - t0

#----------------------------------------------------------------------------


#----------------------------------------------------------------------------
@click.command()
@click.pass_context
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--seed', type=int, help='random seed', default=0)
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1.2, show_default=True)
@click.option('--class', 'class_idx', type=int, help='Class label (unconditional if not specified)')
@click.option('--noise-mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), default='const', show_default=True)
@click.option('--projected-w', help='Projection result file', type=str, metavar='FILE')
#@click.option('--outdir', help='Where to save the output images', default="./results", metavar='DIR')
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
    seed: int,
    truncation_psi: float,
    noise_mode: str,
    #outdir: str,
    class_idx: Optional[int],
    projected_w: Optional[str],
    spout_name: str,
    spout_silent: bool,
    udp_in: int,
    udp_out: int,
    ip: str,
):

    now_gen_par = gen_par(network_pkl, truncation_psi, noise_mode)
    my_spout_pars = spout_pars(spout_name, spout_silent )

    ## START UDP
    ####################################################################
    print("starting UDP.... IP={}, LISTEN= {}, SEND= {}".format(ip, udp_in, udp_out))
    sock = socket.socket(socket.AF_INET, # Internet
                          socket.SOCK_DGRAM) # UDP
    sock.settimeout(udp_timeout)

    #Bind to receiving ip and port
    try:
        sock.bind((ip, udp_in))
    except:
        print("Can`t bind listening port!")
    print("done!")


    #START MULTIPLE PROCESSES
    ###################################################################

    q = mp.Queue(1)
    
    #gp = mp.Process(target=gen_proc, args=(q, now_gen_par, my_spout_pars))
    up = mp.Process(target=udp_ops, args=(q, now_gen_par, sock))
    up.daemon = True
    #gp.start()
    #time.sleep(10)
    up.start()
    gen_proc(q, now_gen_par, my_spout_pars)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
