import numpy as np
from numpy.random import rand
from PIL import Image
import hashlib
import os
import time
import functools
import concurrent.futures

''' Simulating the Ising model and producing Lattice image data by parallel processing
---params---
L : Lattice Size
beta : inverse temperature
mcSteps : Monte Carlo steps for simulation
tempset : number of temperature class
'''

def Metropolis_step(config, L, beta):
    ''' Monte carlo steps using Metropolis algorithm '''
    for i in range(L):
        for j in range(L):            
            a = np.random.randint(0, L)
            b = np.random.randint(0, L)
            s =  config[a, b]
            spinsum = config[(a+1)%L,b] + config[a,(b+1)%L] + config[(a-1)%L,b] + config[a,(b-1)%L]
            dE = 2*s*spinsum
            if dE < 0:
                s *= -1
            elif rand() < np.exp(-beta*dE):
                s *= -1
            config[a, b] = s
    return config


def Random_string(length):
    ''' make random filename for saving config images '''
    buf = ''
    while len(buf) < length:
        buf += hashlib.md5(os.urandom(100)).hexdigest()
    return buf[0:length]


def Making_image(config):
    imData = (config+1)*255//2
    img = Image.fromarray(np.uint8(imData))
    filename = Random_string(np.random.randint(15, 20)) + '.png'
    img.save(filename)
    

def Simulate(imagenum, L, temp):   
    ''' This module simulates the Ising model'''
    np.random.seed(imagenum)
    config = 2*np.random.randint(2, size=(L,L))-1 
    
    mcSteps = 6000
    for i in range(1,mcSteps+1):
        Metropolis_step(config, L, 1.0/temp)
    Making_image(config)
    #if imagenum == 1:  print('T={:.3f} : process id : '.format(temp) + str(os.getpid()))


def arg_parser():
    """
    Create an empty argparse.ArgumentParser.
    """
    import argparse
    return argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)


def main():
    parser = arg_parser()
    parser.add_argument('--lattice_size', help='lattice size', type=int, default=16)
    parser.add_argument('--Tmin', help='minimum temperature', type=float, default=1.0)
    parser.add_argument('--Tmax', help='maximum temperature', type=float, default=5.0)
    parser.add_argument('--tempset', help='number of temp class', type=int, default=100)
    parser.add_argument('--imgset_num', help='number of images for each class', type=int, default=320)

    args = parser.parse_args()
    #args = parser.parse_args(args=['--tempset','2','--imgset_num','4'])

    dir_path = '{}/IMG_{}'.format(os.getcwd(), args.lattice_size)

    T = np.linspace(args.Tmin, args.Tmax, args.tempset)
    try:
        os.mkdir(dir_path)
    except:
        print('Directory already exists')
    
    for level in range(args.tempset):
        try:
            os.mkdir(dir_path + '/Temp{:.2f}_{}'.format(T[level],level))
        except:
            print('Subdirectory already exists')
            continue
        #import pdb; pdb.set_trace()
        os.chdir(dir_path + '/Temp{:.2f}_{}'.format(T[level],level))
        tic = time.time()
        Generate = functools.partial(Simulate, L=args.lattice_size, temp=T[level])
        with concurrent.futures.ProcessPoolExecutor() as executor:
            executor.map(Generate, range(1,args.imgset_num+1), chunksize=40)
        tac = time.time()
        print('{:.3f}'.format(tac-tic))
        os.chdir(dir_path)


if __name__=='__main__':
    main()
