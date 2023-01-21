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
from Library.Spout import Spout

import pygame
from pygame.locals import *


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
        self.latent = np.array([0.0]*512)
        self.z_int = 0
        self.w_int = 0.1
        self.noise_mode = noise_mode
        self.uselatent = False

class spout_pars:
    def __init__(self, name, silent):

        self.name = name
        self.silent = silent


#----------------------------------------------------------------------------

#UDP things



def start_socket(ip, port):
    sock = socket.socket(socket.AF_INET, # Internet
                          socket.SOCK_DGRAM) # UDP
    sock.settimeout(udp_timeout)

    #Bind to receiving ip and port
    try:
        sock.bind((ip, port))
        print("done!")
        return sock
    except:
        print("Can`t bind listening port!")
        pass
    
def udplatent_ops(q, lu, socket):
    mysock = socket
    while True:
        #do things with data
        print("executing udp latent ops")
        #time.sleep(0.2)
        try:
            data, addr = mysock.recvfrom(1024) # buffer size is 1024 bytes
            #print("received message: %s" % data)
            msg_type, msg = data.decode('utf-8').strip().split("_")
            print("type: {} msg: {}".format(msg_type, msg))


def udpsimple_ops(q, lu, first_gen_pars, socket):
    my_pars = first_gen_pars
    mysock = socket
    while True:
        #do things with data
        print("executing udp ops")
        time.sleep(0.2)
        try:
            data, addr = mysock.recvfrom(1024) # buffer size is 1024 bytes
            print("received message: %s" % data)
            msg_type, msg = data.decode('utf-8').strip().split("_")
            print("type: {} msg: {}".format(msg_type, msg))
            if msg_type == "seed":
                my_pars.seed = int(msg)
                my_pars.uselatent = False
                print("Received new seed: {}".format(my_pars.seed))
            if msg_type == "trunc":
                my_pars.truncation_psi = (int(msg)/100000)-5  # one less zero because interval is 10 (-5,+5)
                print("Received new Truncation PSI: {}".format(my_pars.truncation_psi))
            if msg_type == "zint":
                my_pars.z_int = int(msg)/1000000
                print("Received new Z Interpolation step value: {}".format(my_pars.z_int))
            if msg_type == "wint":
                my_pars.w_int = int(msg)/1000000
                print("Received new W Interpolation step value: {}".format(my_pars.w_int))
            if msg_type == "latentA":
                my_pars.latent = np.array((int(msg.split("-"))/1000.0)-0.5)
                my_pars.uselatent = True
                print("Received new latent: {}".format(my_pars.latent))
            if msg_type == "noisemode":
                if int(msg) == 0:
                    my_pars.noise_mode = 'const'
                if int(msg) == 1:
                    my_pars.noise_mode = 'random'
                if int(msg) == 2:
                    my_pars.noise_mode = 'none'
                print("Received new Noise Mode: {}".format(my_pars.noise_mode))
            
            q.put(my_pars)
            print("queue added!!!!!")
        except:
            print("nothing to add....")
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
    print("done!")

    elaps = 0
    while True:
        t0 = time.perf_counter()
        try:
            now_gen_par = q.get_nowait()
        except:
            pass
        
       

        if now_gen_par.uselatent == False:
            z = torch.from_numpy(np.random.RandomState(now_gen_par.seed).randn(1, G.z_dim)).to(device)
            print('Generating image for seed {} ... Last took {}s'.format(now_gen_par.seed, elaps))
        else:
            z = torch.from_numpy(now_gen_par.latent).to(device)
            print('Generating image for latent {} ... Last took {}s'.format(z, elaps))
        if now_gen_par.z_int != 1:  
            z = ((z * now_gen_par.z_int) + (oldz * (1 - now_gen_par.z_int)))
            oldz = z

        w_samples = G.mapping(z, None,  truncation_psi=now_gen_par.truncation_psi)
        if now_gen_par.w_int != 1:
            w_samples = ((w_samples * now_gen_par.w_int) + (oldw_samples * (1 - now_gen_par.w_int)))
            oldw_samples = w_samples
        
        #img = G(z, label, truncation_psi=now_gen_par.truncation_psi, noise_mode=now_gen_par.noise_mode)
        img = G.synthesis(w_samples, noise_mode=now_gen_par.noise_mode)
        img = (img[0].permute( 1, 2, 0) * 127.5 + 128).clamp(0, 255).cpu().numpy()

        spout.send(img)
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
@click.option('--spout-window', 'spout_silent',  type=bool, help='Show window for spout output, default True', default=True) #to hide window silend should be true
@click.option('--udp-in', 'udp_in', type=int, help='UDP listening port', default=5005,show_default=True)
@click.option('--ip', 'ip', help='UDP IP to communicate with, default localhost', default="127.0.0.1",show_default=True)
@click.option('--latentreceiver', 'latent_receiver',  type=bool, help='Start latent receiver, default False', default=False) 
@click.option('--latent-in', 'latent_in', type=int, help='UDP latent listening port', default=5006,show_default=True)
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
    ip: str,
    latent_receiver: bool,
    latent_in: int,

):

    now_gen_par = gen_par(network_pkl, truncation_psi, noise_mode)
    my_spout_pars = spout_pars(spout_name, spout_silent )

    ## START UDP
    ####################################################################
    print("starting UDP simple.... IP={}, LISTEN= {}".format(ip, udp_in))
    simplesocket = start_socket(ip, udp_in)
    if latent_receiver:
        print("starting UDP latent.... IP={}, LISTEN= {}".format(ip, latent_in))
        latentsocket = start_socket(ip, latent_in)


    #START MULTIPLE PROCESSES
    ###################################################################

    q = mp.Queue(1024)
    lu = mp.Queue(4098)
    
    gp = mp.Process(target=gen_proc, args=(q, now_gen_par, my_spout_pars))
    up = mp.Process(target=udp_ops, args=(q, lu, now_gen_par, sock))

    gp.start()
    time.sleep(8)
    up.start()

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
