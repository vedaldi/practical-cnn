import math
import random
import numpy as np
import time
import torch
import torch.nn as nn
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
    if len(x.shape) == 2:
        x = x[:,:,None]
    return torch.tensor(x, dtype=torch.float32).permute(2,0,1)[None,:]/255

def imsc(im, *args, quiet=False, **kwargs):
    """Plot the PyTorch tensor `im` with dimension 3 x H x W or 1 x H x W as an image."""
    with torch.no_grad():
        im = im - im.min() # make a copy
        im.mul_(1/im.max())
        if not quiet:
            bitmap = im.expand(3, *im.shape[1:]).permute(1,2,0).numpy()
            plt.imshow(bitmap, *args, **kwargs)
            ax = plt.gca()
            ax.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
    return im

def imarraysc(tiles, spacing=0, quiet=False):
    num = tiles.shape[0]
    num_cols = math.ceil(math.sqrt(num))
    num_rows = (num + num_cols - 1) // num_cols
    c = tiles.shape[1]
    h = tiles.shape[2]
    w = tiles.shape[3]
    mosaic = torch.zeros(c,
      h*num_rows + spacing*(num_rows-1),
      w*num_cols + spacing*(num_cols-1))
    for t in range(num):
        u = t % num_cols
        v = t // num_cols
        tile = tiles[t]
        mosaic[0:c,
          v*(h+spacing) : v*(h+spacing)+h,
          u*(w+spacing) : u*(w+spacing)+w] = imsc(tiles[t], quiet=True)
    if not quiet:
        imsc(mosaic)
    return mosaic

def extract_black_blobs(im):
    "Find the dark blobs in an image."
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
    "Apply the determinant of Hessian filter to the gray-scale image `im`."
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

def jitter(x):
    "Apply jitter to the batch of characters `x`."
    with torch.no_grad():
        perm1 = np.random.permutation(x.shape[0])
        perm2 = np.random.permutation(x.shape[0])
        x_ = torch.cat((x[perm1], x[perm2]), 3)
        x_[:,:,:,16:48] = torch.min(x_[:,:,:,16:48], x)
        du = random.randint(-6,6)
        dv = random.randint(-2,2)
        su = [u + du for u in range(16,48)]
        sv = [max(0, min(v + dv, 31)) for v in range(0,32)]
        s =  np.ix_(sv,su)
    return x_
    #return x_[:,:,s[0],s[1]]

def accuracy(prediction, target):
    with torch.no_grad():
        _, predc = prediction.topk(1, 1, True, True)
        correct = predc.t().eq(target).to(torch.float32)
        return correct.mean()

def train_model(model, imdb, batch_size=100, num_epochs=15, jitter=lambda x: x):
    """Train a model using SGD.

    Arguments:
        model {Torch module} -- The model to train.
        imdb {imdb structure} -- The data to train the model from.

    Keyword Arguments:
        batch_size {int} -- Batch size. (default: {100})
        num_epochs {int} -- Number of epochs. (default: {15})
        jitter {function} -- Jitter function (default: {identity})
    """

    print_period = 50
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    train = torch.nonzero(imdb['sets'] == 0).reshape(-1)
    val = torch.nonzero(imdb['sets'] == 1).reshape(-1)

    def get_batch(batch_size, indices):
        t = 0
        while t < len(indices):
            q = min(t + batch_size, len(indices))
            yield indices[t:q]
            t = q

    loss = nn.CrossEntropyLoss()
    train_loss_log = []
    train_acc_log = []
    val_loss_log = []
    val_acc_log = []

    for epoch in range(num_epochs):
        print(f"beginning epoch {epoch} of {num_epochs}")
        train_items = 0
        train_loss = 0
        train_acc = 0
        val_items = 0
        val_loss = 0
        val_acc = 0
        begin = time.clock()
        perm = np.random.permutation(train)
        num_iter = math.ceil(len(perm) / batch_size)
        for iter, batch in enumerate(get_batch(batch_size, perm)):
            # Evaluate network and loss
            x = imdb['images'][batch,]
            c = imdb['labels'][batch,]
            x = jitter(x)

            y = model(x)
            y = y.reshape(y.shape[:2])
            z = loss(y, c)

            # Average
            train_items += len(batch)
            train_loss += len(batch) * z.item()
            train_acc += len(batch) * accuracy(y, c).item()

            # Backprop and SGD
            z.backward()
            optimizer.step()
            model.zero_grad()

            if iter % print_period == 0:
                print(f"epoch: {epoch+1:02d}/{num_epochs:02d}"
                 f" train iter: {iter+1:03d}/{num_iter:03d}"
                 f" speed: {train_items / (time.clock() - begin):.1f} Hz"
                 f" loss: {train_loss/train_items:.2f}"
                 f" acc: {100*train_acc/train_items:.1f}%")

        train_loss_log.append(train_loss / train_items)
        train_acc_log.append(train_acc / train_items)

        num_iter = math.ceil(len(val) / batch_size)
        for iter, batch in enumerate(get_batch(batch_size, val)):
            # Evaluate network and loss
            with torch.no_grad():
                x = imdb['images'][batch,]
                c = imdb['labels'][batch,]
                y = model(x)
                y = y.reshape(y.shape[:2])
                z = loss(y, c)

            # Average
            val_items += len(batch)
            val_loss += len(batch) * z.item()
            val_acc += len(batch) * accuracy(y, c).item()

            if iter % print_period == 0:
                print(f"epoch: {epoch+1:02d}/{num_epochs:02d}"
                 f" val iter: {iter+1:03d}/{num_iter:03d}"
                 f" speed: {train_items / (time.clock() - begin):.1f} Hz"
                 f" loss: {val_loss/val_items:.2f}"
                 f" acc: {100*val_acc/val_items:.1f}%")

        val_loss_log.append(val_loss / val_items)
        val_acc_log.append(val_acc / val_items)

        plt.figure(1)
        plt.clf()
        plt.gcf().add_subplot(1,2,1)
        plt.title('loss')
        plt.plot(train_loss_log)
        plt.plot(val_loss_log,'--')
        plt.legend(('training', 'validation'),)
        plt.gcf().add_subplot(1,2,2)
        plt.title('accuracy')
        plt.plot(train_acc_log)
        plt.plot(val_acc_log,'--')
        plt.legend(('training', 'validation'),)
        plt.pause(1e-5)

def jitter(x):
    "Apply jitter to the batch of characters `x`."
    with torch.no_grad():
        perm1 = np.random.permutation(x.shape[0])
        perm2 = np.random.permutation(x.shape[0])
        x_ = torch.cat((x[perm1], x[perm2]), 3)
        x_[:,:,:,16:48] = torch.min(x_[:,:,:,16:48], x)
        du = random.randint(-6,6)
        dv = random.randint(-2,2)
        su = [u + du for u in range(16,48)]
        sv = [max(0, min(v + dv, 31)) for v in range(0,32)]
        s =  np.ix_(sv,su)
    return x_[:,:,s[0],s[1]]

def decode_predicted_string(y):
    "Decode the prediction of the character CNN into a string"
    _, k = y.max(1)
    return [str(chr(c)) for c in (ord('a') + k).reshape(-1)]

def plot_predicted_string(im, y):
    "Plot the prediction of the character CNN"
    chars = decode_predicted_string(y)
    plt.clf()
    plt.gcf().add_subplot(2,1,1)
    plt.title('predictions')
    imsc(im[0])
    for u, c in enumerate(chars):
        u_ = u * im.shape[3]/len(chars)
        plt.text(u_, 40 + (u % 5) * 15, c)

    plt.gcf().add_subplot(2,1,2)
    plt.title('class scores')
    imsc(y.squeeze()[None,:])
