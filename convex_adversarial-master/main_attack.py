import setGPU
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable


import torchvision.transforms as transforms
import torchvision.datasets as datasets

import setproctitle
import argparse

import problems as pblm
from attacks import pgd, svgd
#import cvxpy as cp

import numpy as np

cp2np = lambda x : np.asarray(x.value).T

if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--niters', type=int, default=200)
    parser.add_argument('--epsilon', type=float, default=0.1)
    parser.add_argument('--alpha', type=float, default=0.01)

    parser.add_argument('--mnist', action='store_true')
    parser.add_argument('--svhn', action='store_true')
    parser.add_argument('--har', action='store_true')
    parser.add_argument('--fashion', action='store_true')

    args = parser.parse_args()

    if args.mnist: 
        train_loader, test_loader = pblm.mnist_loaders(args.batch_size)
        model = pblm.mnist_model().cuda()
        # model.load_state_dict(torch.load('icml/mnist_epochs_100_baseline_model.pth'))
        model.load_state_dict(torch.load('../models/mnist.pth'))
    elif args.svhn: 
        train_loader, test_loader = pblm.svhn_loaders(args.batch_size)
        model = pblm.svhn_model().cuda()
        model.load_state_dict(torch.load('svhn_new/svhn_epsilon_0_01_schedule_0_001'))
    elif args.har:
        pass
    elif args.fashion: 
        pass
    else:
        raise ValueError("Need to specify which problem.")
    for p in model.parameters(): 
        p.requires_grad = False

    epsilon = 0.1
    num_classes = model[-1].out_features
    
    pgd(test_loader, model, epsilon, args.niters)
    svgd(test_loader, model, epsilon, args.niters)
    
    

    