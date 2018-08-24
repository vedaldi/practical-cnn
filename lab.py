import math
import numpy as np
import torch
import torch.nn.functional  as F
from matplotlib import pyplot as plt
from PIL import Image

def t2im(x):
    """Rearrange the N x K x H x W to have shape (NK) x 1 x H x W."""
    return x.reshape(-1, *x.shape[2:])[:,None,:,:]

def imread(file):
    """Read the image `file` as a PyTorch tensor."""
    # Read an example image as a NumPy array
    x = Image.open(file)
    x = np.array(x)
    return torch.tensor(x, dtype=torch.float32).permute(2,0,1)[None,:]/255

def imsc(im, *args, **kwargs):
    """Plot the PyTorch tensor im with timension 3 x H x W or 1 x H x W as an image."""
    if issubclass(im.__class__, torch.Tensor):
        im = im.detach().permute(1,2,0).numpy()
    im = im - im.min()
    im = im / im.max()
    if im.shape[2] == 1:
        im = np.tile(im,[1,1,3])
    #print(im.shape)
    plt.imshow(im, *args, **kwargs)
    ax = plt.gca()
    ax.axis('off')
    ax.set_xticklabels([])
    ax.set_yticklabels([])

def imarraysc(t):
    num = t.shape[0]
    num_cols = math.ceil(math.sqrt(num))
    num_rows = num // num_cols + 1
    fig = plt.gcf()
    plt.clf()
    for i in range(num):
        ax = fig.add_subplot(num_rows, num_cols, i + 1)
        imsc(t[i])        
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    plt.draw()
    plt.pause(0.0001)

def extract_black_blobs(im):
    """Find the dark blobks in an image"""
    with torch.no_grad():
        score = hessiandet(im)
        ismax = (score == F.max_pool2d(score, 3, padding=1, stride=1)) * (score > 3e-4)
        pos = torch.tensor(ismax, dtype=torch.float32)
        neg = 1 - F.max_pool2d(pos, 7, padding=3, stride=1)
        indices = np.where(pos.numpy().squeeze() > 0)

        # neg = 1 - torch.tensor(score > 0, dtype=torch.float32)
        # plt.figure(2)
        # imarraysc(torch.cat((im,pos,neg), 0))
        # plt.figure(3)
        # imsc(torch.cat((im,pos,pos), 1)[0])
        # plt.plot(indices[1],indices[0],'+')
        # plt.pause(100)
    return pos, neg, indices

def imsmooth(im, sigma=2.5):
    """Applies the determinant of Hessian filter to the gray-scale image `im`."""    
    with torch.no_grad():
        m = math.ceil(sigma * 2)    
        u = np.linspace(-m, m, 2*m+1, dtype=np.float32)
        w = np.exp(- 0.5 * (u**2) / (sigma**2))
        w = w.reshape(-1,1) @ w.reshape(1,-1)
        w = w / np.sum(w)
        w = torch.tensor(w).reshape(1,1,*w.shape)
        w = w.expand(w.shape[0], im.shape[1], *w.shape[2:])
        return F.conv2d(im, w, padding=m)

def hessiandet(im):
    """Applies the determinant of Hessian filter to the gray-scale image `im`."""    
    with torch.no_grad():
        # Gaussian filter
        im = imsmooth(im, 2.5)
    
        # Derivative filters
        d11 = torch.tensor([-1, 0, 1], dtype=torch.float32).reshape(1,1,3,1)
        d12 = d11.permute(0,1,3,2)
        d21 = torch.tensor([1, -2, 1], dtype=torch.float32).reshape(1,1,3,1)
        d22 = d21.permute(0,1,3,2)

        im11 = torch.conv2d(im,   d21, padding=[1, 0])
        im22 = torch.conv2d(im,   d22, padding=[0, 1])
        im12 = torch.conv2d(im,   d11, padding=[1, 0])
        im12 = torch.conv2d(im12, d12, padding=[0, 1])
        score = im11 * im22 - im12 * im12
    return score
