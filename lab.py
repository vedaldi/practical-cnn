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
    """Rearrange the N x K x H x W to have shape (NK) x 1 x H x W.

    Arguments:
        x {torch.Tensor} -- A N x K x H x W tensor.

    Returns:
        torch.Tensor -- A (NK) x 1 x H x W tensor.
    """
    return x.reshape(-1, *x.shape[2:])[:,None,:,:]

def imread(file):
    """Read the image `file` as a PyTorch tensor.

    Arguments:
        file {str} -- The path to the image.

    Returns:
        torch.Tensor -- The image read as a 3 x H x W tensor in the [0, 1] range.
    """
    # Read an example image as a NumPy array
    x = Image.open(file)
    x = np.array(x)
    if len(x.shape) == 2:
        x = x[:,:,None]
    return torch.tensor(x, dtype=torch.float32).permute(2,0,1)[None,:]/255

def imsc(im, *args, quiet=False, **kwargs):
    """Rescale and plot an image represented as a PyTorch tensor.

     The function scales the input tensor im to the [0 ,1] range.

    Arguments:
        im {torch.Tensor} -- A 3 x H x W or 1 x H x W tensor.

    Keyword Arguments:
        quiet {bool} -- Do not plot. (default: {False})

    Returns:
        torch.Tensor -- The rescaled image tensor.
    """
    handle = None
    with torch.no_grad():
        im = im - im.min() # make a copy
        im.mul_(1/im.max())
        if not quiet:
            bitmap = im.expand(3, *im.shape[1:]).permute(1,2,0).numpy()
            handle = plt.imshow(bitmap, *args, **kwargs)
            ax = plt.gca()
            ax.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
    return im, handle

def imarraysc(tiles, spacing=0, quiet=False):
    """Plot the PyTorch tensor `tiles` with dimesion N x C x H x W as a C x (MH) x (NW) mosaic.

    The range of each image is individually scaled to the range [0, 1].

    Arguments:
        tiles {[type]} -- [description]

    Keyword Arguments:
        spacing {int} -- Thickness of the border (infilled with zeros) around each tile (default: {0})
        quiet {bool} -- Do not plot the mosaic. (default: {False})

    Returns:
        torch.Tensor -- The mosaic as a PyTorch tensor.
    """
    handle = None
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
          u*(w+spacing) : u*(w+spacing)+w] = imsc(tiles[t], quiet=True)[0]
    return imsc(mosaic, quiet=quiet)


def extract_black_blobs(im):
    """Extract the dark blobs from an image.

    Arguments:
        im {torch.Tensor} -- Image as a 1 x 1 x W x H PyTorch tensor.

    Returns:
        torch.Tensor -- An indicator tensor for the pixels centered on a dark blob.
        torch.Tensor -- An indicator tensor for the pixels away from any dark blob.
        torch.Tensor -- A pair of (u,v) tensor with the coordinates of the dark blobs.
    """
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
    "Apply a Gaussian filter of standard deviation `sigma` to the image `im`."
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
    "Apply jitter to the N x 1 x 32 x 32 torch.Tensor `x` representing a batch of character images."
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

def accuracy(prediction, target):
    """Comptute classification accuracy.

    Arguments:
        prediction {torch.Tensor} -- A N x C tensor of prediction scores.
        target {torch.Tensor]} -- A N tensor of ground-truth classes.

    Returns:
        torch.Tensor -- A scalar tensor with the fraction of instances correctly predicted.
    """
    with torch.no_grad():
        _, predc = prediction.topk(1, 1, True, True)
        correct = predc.t().eq(target).to(torch.float32)
        return correct.mean()

def train_model(model, imdb, batch_size=100, num_epochs=15, use_gpu=False, jitter=lambda x: x):
    """Train a model using SGD.

    Arguments:
        model {torch.Module} -- The model to train.
        imdb {dict} -- `imdb` image database with the training data.

    Keyword Arguments:
        batch_size {int} -- Batch size. (default: {100})
        num_epochs {int} -- Number of epochs. (default: {15})
        use_gpu {bool} -- Whether to use the GPU. (default: {False})
        jitter {function} -- Jitter function (default: {identity})

    Returns:
       model {torch.Module} -- The trained model. Might be different from the input.
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

    # Send model to GPU if needed
    device = torch.device("cuda" if use_gpu else "cpu")
    model = model.to(device)

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
            # Get images and labels
            x = imdb['images'][batch,]
            c = imdb['labels'][batch,]

            # Jitter images
            x = jitter(x)

            # Send to GPU if needed
            x = x.to(device)
            c = c.to(device)

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
                # Get images and labels
                x = imdb['images'][batch,]
                c = imdb['labels'][batch,]

                # Send to GPU if needed
                x = x.to(device)
                c = c.to(device)

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

    model = model.to(torch.device("cpu"))
    return model

def decode_predicted_string(y):
    "Decode the C x W tensor `y` character predictions into a string"
    _, k = y.max(1)
    return [str(chr(c)) for c in (ord('a') + k).reshape(-1)]

def plot_predicted_string(im, y):
    "Plot the C x W tensor `y` character predictions overlaying them to the image `im`."
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
