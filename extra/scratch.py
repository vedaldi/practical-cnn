# Scratch file

import lab
import math
import numpy as np
import torch
import torch.nn.functional  as F
from matplotlib import pyplot as plt

def tinycnn(x, w, b):
    pad1 = (w.shape[2] - 1) // 2
    rho2 = 3
    pad2 = (rho2 - 1) // 2
    x = F.conv2d(x, w, b, padding=pad1)
    x = F.avg_pool2d(x, rho2, padding=pad2, stride=1)
    return x

# Load a training image and convert to gray-scale
im = lab.imread('data/dots.jpg')
im = im.mean(dim=1)[None,:,:,:]

# Compute the location of black blobs in the image
pos, neg, indices = lab.extract_black_blobs(im)
pos = pos / pos.sum()
neg = neg / neg.sum()

# Preprocess the image by subtracting its mean
im = im - im.mean()
im = lab.imsmooth(im, 3)

num_iterations = 1001
rate = 10
momentum = 0.8
shrinkage = 0.0001
plot_period = 200

with torch.no_grad():
    w = torch.randn(1,1,3,3)
    w = w - w.mean()
    b = torch.Tensor(1)
    b.zero_()

E = []
w.requires_grad_(True)
b.requires_grad_(True)
w_momentum = torch.zeros(w.shape)
b_momentum = torch.zeros(b.shape)

for t in range(num_iterations):

    # Evaluate the CNN and the loss
    y = tinycnn(im, w, b)
    z = (pos * (1 - y).relu() + neg * y.relu()).sum()

    # Track energy
    E.append(z.item() + 0.5 * shrinkage * (w**2).sum().item())

    # Backpropagation
    z.backward()

    # Gradient descent
    with torch.no_grad():
        w_momentum = momentum * w_momentum + (1 - momentum) * (w.grad + shrinkage * w)
        b_momentum = momentum * b_momentum + (1 - momentum) * b.grad
        w -= rate * w_momentum
        b -= 0.1 * rate * b_momentum        
        w.grad.zero_()
        b.grad.zero_()

    # Plotting
    if t % plot_period == 0:
        plt.clf()
        fig = plt.gcf()
        ax1 = fig.add_subplot(1, 3, 1)
        plt.plot(E)
        ax2 = fig.add_subplot(1, 3, 2)
        lab.imsc(w.detach()[0])
        ax3 = fig.add_subplot(1, 3, 3)
        lab.imsc(y.detach()[0])
        plt.plot(indices[1],indices[0],'.')
        plt.pause(0.0001)
