# VGG Convolutional Neural Netwroks Practrical

> Andrea Vedaldi and Andrew Zisserman

This is an [Oxford Visual Geometry Group](http://www.robots.ox.ac.uk/~vgg) computer vision practical, authored by [Andrea Vedaldi](http://www.robots.ox.ac.uk/~vedaldi/) and Andrew Zisserman (Release 2015a).

![cover][1]

*Convolutional neural networks* are an important class of learnable representations applicable, among others, to numerous computer vision problems. Deep CNNs, in particular, hare composed of several layers of processing, each involving linear as well as non-linear operators, that are learned jointly, in an end-to-end manner, to solve a particular tasks. Such methods become the predominant for feature extraction from audiovisual and textual data.

[TOC]

$$
   \newcommand{\bx}{\mathbf{x}}
   \newcommand{\by}{\mathbf{y}}
   \newcommand{\bz}{\mathbf{z}}
   \newcommand{\bw}{\mathbf{w}}
   \newcommand{\cP}{\mathcal{P}}
   \newcommand{\cN}{\mathcal{N}}
   \newcommand{\vc}{\operatorname{vec}}
$$

## Getting started

Read and understand the [requirements and installation instructions](../overview/index.html#installation). The download links for this practical are:

* Code and data: [practical-cnn-2015a.tar.gz](http://www.robots.ox.ac.uk/~vgg/share/practical-cnn-2015a.tar.gz)
* Code only: [practical-cnn-2015a-code-only.tar.gz](http://www.robots.ox.ac.uk/~vgg/share/practical-cnn-2015a-code-only.tar.gz)
* Data only: [practical-cnn-2015a-data-only.tar.gz](http://www.robots.ox.ac.uk/~vgg/share/practical-cnn-2015a-data-only.tar.gz)
* [Git repository](https://github.com/vedaldi/practical-object-cnn) (for lab setters and developers)

After the installation is complete, open and edit the script `exercise1.m` in the MATLAB editor. The script contains commented code and a description for all steps of this exercise, relative to [Part I](#part1) of this document. You can cut and paste this code into the MATLAB window to run it, and will need to modify it as you go through the session. Other files `exercise2.m`, `exercise3.m`, and `exercise4.m` are given for [Part II](#part2), [III](#part3), and [IV](part4).

Each part contains several **Questions** (that require pen and paper) and **Tasks** (that require experimentation or coding) to be answered/completed before proceeding further in the practical.

## Part 1: CNN building blocks {#part1}

### Part 1.1: linear convolution {#part1.1}

A feed-forward neural network can be thought of as the composition of number of functions 
$$
f(\bx) = f_L(\dots f_2(f_1(\bx;\bw_1);\bw_2)\dots),\bw_{L}).
$$
Each function $f_l$ takes as input a datum $\bx_l$ and a parameter vector $\bw_l$ and produces as output a datum $\bx_{l+1}$. While the type and sequence of functions is usually handcrafted, the paramters $\bw=(\bw_1,\dots,\bw_L)$ are *learned from data* in order to solve a target problem, for example classifying images or sounds.

In a *convolutional neural network* data and functions have additional structure. The data $\bx_1,\dots,\bx_n$ are images, sounds, or more in general maps from a lattice[^lattice] to one or more real numbers. In particular, since the rest of the practical will focus on computer vision applications, data will be 2D dimensional arrays of pixels. Formally, each $\bx_i$ will be a $M \times N \times K$ real array of $M \times N$ pixels and $K$ channels per pixels. Hence the first two dimension of the array span space, while the last one channels. Note that only the input $\bx=\bx_1$ of the network is an actual image, while the remaining data are intermediate *feature maps*.

The second property of a CNN is that the functions $f_l$ have a *convolutional structure*. This means that $f_l$ applies to the input map $\bx_l$ an operator that is *local and translation invariant*. Examples of convolutional operators are applying a bank of linear filters to $\bx_l$ or computing its element-wise square. 

In this part we will familiarise ourselves with a number of such convolutional operators. The first one is the regular *linear convolution* by a filter bank. We will start by focusing our attention on a single function relation as follows:
$$
 f: \mathbb{R}^{M\times N\times K} \rightarrow \mathbb{R}^{M' \times N' \times K'},
 \qquad \bx \mapsto \by.
$$
Open the `example1.m` file, select the following part of the code, and execute it in MATLAB (right button > `Evaluate selection` or `Shift+F7`).
```matlab
% Read an example image
x = imread('peppers.png') ;

% Convert to single format
x = im2single(x) ;

% Visualize the input x
figure(1) ; clf ; imagesc(x) 
```
This should display an image of bell peppers in Figure 1:

![bell][2]

Use MATLAB `size` command to obtain the size of the array `x`. Note that the array `x` is converted to *single precision* format. This is because the underlying MatConvNet assumes that data is in single precision.

> **Question.** The third dimension of `x` is 3. Why?

Now we will create a bank 10 of $5 \times 5 \times 3$ filters.
```matlab
% Create a bank of linear filters
w = randn(5,5,3,10,'single') ;
```
The filters are in single precision as well. Note that `w` has four dimensions, packing 10 filters. Note also that each filter is not flat, but rather a volume with three layers. The next step is applying the filter to the image. This uses the `vl_nnconv` function from MatConvNet:
```matlab
% Apply the convolutional operator
y = vl_nnconv(x, w, []) ;
```
**Remark:** You might have noticed that the third argument to the `vl_nnconv` function is the empty matrix `[]`. It can be otherwise used to pass a vector of bias terms to add to the output of each filter.

The variable `y` contains the output of the convolution. Note that the filters are three-dimensional, in the sense that it operates on a map $\bx$ with $K$ channels. Furthermore, there are $K'$ such filters, generating a $K'$ dimensional map $\by$ as follows
$$
y_{i'j'k'} = \sum_{ijk} w_{ijk} x_{i+i',j+j',k,k'}
$$

> **Questions:** Study carefully this expression and answer the following:
> 
> 
>  - Given that the input map $\bx$ has $M \times N \times K$ dimensions and that each of the $K'$ filters has dimension $M_f \times N_f \times K$, what is the dimension of $\by$?
>  - Note that $x$ is indexed by $i+i'$ and $j+j'$, but that there is no plus sign between $k$ and $k'$. Why?

> **Task:** check that the size of the variable `y` matches your calculations.

We can now visualize the output `y` of the convolution. In order to do this, use the [`vl_imarraysc`](http://www.vlfeat.org/matlab/vl_imarraysc.html) function to display an image for each feature channel in `y`:
```matlab
% Visualize the output y
figure(2) ; clf ; vl_imarraysc(y) ; colormap gray ;
```

> **Question:** Study the feature channels obtained. Most will likely contain a strong response in correspondences of edges in the input image `x`. Recall that `w` was obtained by drawing at random numbers from a Gaussian distribution. Can explain this phenomenon?

So far filters preserve the resolution of the input feature map. However, it is often useful to *dowsnample the output*. This can be obtained by using the `stride` option in `vl_nnconv`:
```matlab
% Try again, downsampling the output
y_ds = vl_nnconv(x, w, [], 'stride', 16) ;
figure(3) ; clf ; vl_imarraysc(y_ds) ; colormap gray ;
```
As you should have noticed in a question above, applying a filter to an image or feature map interacts with the boundaries, making the output map smaller by an amount proportional to the size of the filters. If this is undesirable, then the input array can be padded with zeros by using the `pad` option:
```matlab
% Try padding
y_pad = vl_nnconv(x, w, [], 'pad', 4) ;
figure(4) ; clf ; vl_imarraysc(y_pad) ; colormap gray ;
```

> **Task:** Convince yourself that the previous code's output has different boundaries compared to the code that does not use padding. Can you justify the result?

In order to consolidate what learned so far, we will now design a filter by hand.
```matlab
w = [0  1 0 ;
     1 -4 1 ;
     0  1 0 ] ;
w = single(repmat(w, [1, 1, 3])) ;
y_lap = vl_nnconv(x, w, []) ;

figure(5) ; clf ; colormap gray ;
subplot(1,2,1) ; 
imagesc(y_lap) ; title('filter output') ;
subplot(1,2,2) ;
imagesc(-abs(y_lap)) ; title('- abs(filter output)') ;
```
> **Questions:**
> 
> * What filter have we implemented?
> * How are the RGB colour channels processed by this filter?
> * What image structure are detected?

### Part 1.2: non-linear gating {#part1.2}

As we have stated in the introduction, CNNs are obtained by composing several different functions. In addition to the linear filters shown in the [previous part](#part1.1), there are several non-linear convolutonal operators as well.

> **Question:** Some of the functions in a CNN *must* be non-linear. Why?

The simplest non-linearity is obtained by following a linear filter by a *non-linear gating function*, applied identically to each component of a feature map. The simplest such function is the *Rectified Linear Unit (ReLU)*
$$
  y_{ijk} = \max\{0, x_{ijk}\}.
$$
This function is implemented by `vl_relu`; let's try this out:
```matlab
w = single(repmat([1 0 -1], [1, 1, 3])) ;
w = cat(4, w, -w) ;
y = vl_nnconv(x, w, []) ;
z = vl_nnrelu(y) ;

figure(6) ; clf ; colormap gray ;
subplot(1,2,1) ; vl_imarraysc(y) ;
subplot(1,2,2) ; vl_imarraysc(z) ;
```

> **Tasks:**
> 
> * Run the code above and understand what the filter $\bw$ is doing.
> * Explain the final result $\bz$.

### Part 1.3: pooling {#part1.3}

There are several other important operators in a CNN. One of them is *pooling*.  A pooling operator operates on individual feature channels, coalescing nearby feature values into one by the application of a suitable operator. Common choices include max-pooling (using the max operator) or sum-pooling (using summation). For example, *max-pooling* is defined as:
$$
   y_{ijk} = \max \{ y_{i'j'k} : i \leq i' < i+p, j \leq j' < j + p \}
$$
Max pooling is implemented by the `vl_nnpool` function. Try this now:
```matlab
y = vl_nnpool(x, 15) ;
figure(6) ; clf ; imagesc(y) ;
```
> **Question:** look at the resulting image. Can you interpret the result?

The function `vl_nnpool` supports subsampling and padding just like `vl_nnconv`. However, for max-pooling feature maps are padded with the value $-\infty$ instead of 0. Why?

### Part 1.4: normalisation {#part1.4}

Another important CNN building block is channel-wise normalisation. This operator normalises the vector of feature channels at each spatial location in the input map $\bx$. The form of the normalisation operator is actually rather curious:
$$
  y_{ijk'} = \frac{x_{ijk}}{\left(\kappa + \alpha \sum_{k\in G(k')}  x_{ijk}^2\right)^{\beta}}
$$
where $G(k) = \left[k - \lfloor \frac{\rho}{2} \rfloor, k + \lceil \frac{\rho}{2} \rceil\right] \cap \{1, 2, \dots, K\}$ is a group of $\rho$ consecutive feature channels in the input map.

> **Task:** Understand what this operator is doing. How would you set $\kappa$, $\alpha$ and $\beta$ to achieve simple $L^2$ normalisation?

Now let's try this out:
```matlab
rho = 5 ;
kappa = 0 ;
alpha = 1 ;
beta = 0.5 ;
y_nrm = vl_nnnormalize(x, [rho kappa alpha beta]) ;
figure(6) ; clf ; imagesc(y_nrm) ;
```

> **Tasks:** 
> 
> * Inspect the figure just obtained. Can you interpret it?
> * Compute the $L^2$ norm of the feature channels in the output map `y_nrm`. What do you notice? 
> * Explain this result in relation to the particular choice of the parameters $\rho$,  $\kappa$, $\alpha$ and $\beta$.

## Part 2: back-propagation and derivatives

The parameters of a CNN $\bw=(\bw_1,\dots\bw_L)$ should be learned in such a manner that the overall CNN function  $\bz = f(\bx;\bw)$ achieves a desired goal. In some cases,  the goal is to model the distribution of the data, which leads to a *generative objective*. Here, however, we will use $f$ as a *regressor* and obtain it by minimising a *discriminative objective*. In simple terms, we are given:

* examples of the desired input-output relations $(\bx_1,\bz_1), \dots, (\bx_n,\bz_n)$ where $\bx_i$ are input data and $\bz_i$ corresponding output labels;
* and a loss $\ell(\bz,\hat\bz)$ that expresses the penalty for predicting $\hat\bz$ instead of $\bz$.

We use those to write the empirical loss of the CNN $f$ by averaging over the examples:
$$
L(\bw) = \frac{1}{n} \sum_{i=1}^n \ell(\bz_i, f(\bx_i;\bw))
$$
Note that the composition of the function $f$ with the loss $\ell$ can be though of as a CNN with one more layer (called a *loss layer*). Hence, with a slight abuse of notation, in the rest of this part we incorporate the loss in the function $f$ (which therefore is a map $\mathcal{X}\rightarrow\mathbb{R}$) and do not talk about it explicitly anymore.

The simplest algorithm to minimse $L$, and in fact one that is used in practice, is *gradient descent*. The idea is simple: compute the gradient of the objective $L$ at a current solution $\bw^t$ and then update the latter along the direction of fastest descent of $L$:
$$
 \bw^{t+1} = \bw^{t} - \eta_t \frac{\partial f}{\partial \bw}(\bw^t)
$$
where $\eta_t \in \mathbb{R}_+$ is the *learning rate*.

### Part 2.1: theory of backpropagation

The basic computational problem to solve is the calculation of the gradient of the function with respect to the parameter $\bw$. Since $f$ is the composition of several functions, the key ingredient is the *chain rule*:
$$
\begin{align}
 \frac{\partial f}{\partial \bw_l} &= 
  \frac{\partial}{\partial \bw_l}
  f_L(\dots f_2(f_1(\bx;\bw_1);\bw_2)\dots),\bw_{L})
  \\
  &= 
  \frac{\partial \vc f_L}{\partial \vc\bx_L^\top}
  \frac{\partial \vc f_{L-1}}{\partial \vc\bx_{L-1}^\top}
  \dots
  \frac{\partial \vc f_{l+1}}{\partial \vc\bx_{l+1}^\top}
  \frac{\partial \vc f_l}{\partial \bw_l^\top}
\end{align}
$$
The notation requires some explanation. Recall that each function $f_l$ is a map from a $M\times N\times K$ array to a $M' \times N' \times K'$ array. The operator $\vc$ *vectorises* such arrays by stacking their elments in a column vector (the stacking order is arbitrary but conventionally column-major). The symbol $\partial \vc f_l / \partial \vc \bx_l^\top$ then denotes the derivative of a column vector of output variables by a row vector of input variables. Note that $\bw_l$ is already assumed to be a column vector so it does not require explicit vectorisation.

> **Questions:** Make sure you understand the structure of this formula and answer the following:
> 
> * $\partial \vc f_l / \partial \vc \bx_l^\top$ is a matrix. What are its dimensions?
> * The formula can be rewritten with a slightly different notation by replacing the symbols $f_l$ with the symbols $\bx_{l+1}$. If you do so, do you notice any formal cancelation?
> * The formula only includes the derivative symbols. However, these derivatives must be computed at a well defined point. What is this point?

To apply the chain rule we must be able to compute, for each function $l$, its derivative with respect to the parameters $\bw_l$ as well as its input $\bx_l$. While this could be done naively, a problem is the very high dimensionality of the matrices involved in this calculation as these are $M'N'K' \times MNK$ arrays. We will now introduce a ``trick'' that allows reducing this to working with $MNK$ numbers only and which will yield to the *backpropagation algorithm*. 

The key observation is that we ear not after $\partial \vc f_l / \partial \bw_l^\top$ but after $\partial f/ \partial\bw_l^\top$:
$$
 \frac{\partial f}{\partial\bw_l^\top}
 =
 \frac{\partial g_{l+1}}{\partial \vc \bx_{l+1}^\top}
  \frac{\partial \vc f_l}{\partial \bw_l^\top}
$$
where $g_{l+1} = f_L \circ \dots \circ f_{l+1}$ is the ``tail'' of the CNN.

> **Question:** Explain why the diemnsions of the vectors $\partial g_{l+1}/\partial \vc \bx_{l+1}$ and $\partial f/\partial \bw_{l}$ equals the number of elements in $\bx_{l+1}$ and $\bw_l$ respectively. Hence, in particular, the symbol $\partial g_{l+1}/\partial \bx_{l+1}$ (without vectorisation) denotes an array with the same size of $\bx_{l+1}$.
> 
> **Hint:** recall that the last layer is the loss.

Hence the algorithm can focus on computing the derivatives of $g_{l}$ instead of $f_{l}$ which are far lower-dimensional. To see how this can be done iteratively, decompose $g_l$ as:
$$
 \bx_{l}
 \longrightarrow 
 \underset{\displaystyle\underset{\displaystyle\bw_l}{\uparrow}}{\boxed{f_l}} 
 \longrightarrow
 \bx_{l+1}  
 \longrightarrow 
 \boxed{g_{l+1}}
 \longrightarrow
 \bx_{L}
$$
Then the key of the iteration is obtaining the derivatives for layer $l$ given the ones for layer $l+1$:

* Input:
  * the derivative $\partial g_{l+1}/\partial \bx_{l+1}$.
* Output:
  * the derivative $\partial g_{l}/\partial \bx_{l}$
  * the derivative  $\partial g_{l}/\partial \bw_{l}$

> **Question:** Suppose that $f_l$ is the function $\bx_{l+1} = A \bx_{l}$ where $\bx_{l}$ and $\bx_{l+1}$ are column vectors. Suppose that $B = \partial g_{l+1}/\partial \bx_{l+1} $ is given. Derive an expression for $C = \partial g_{l}/\partial \bx_{l}$ and an expression for $D =  \partial g_{l}/\partial \bw_{l}$.

### Part 2.1: Putting backpropagation to the practice

A key feature of MatConvNet and similar neural-netowrk packages is the ability to support back-propagation. In order to do so, lets focus on a single computational block $f$, followed by a function $g$:
$$
 \bx
 \longrightarrow 
 \underset{\displaystyle\underset{\displaystyle\bw}{\uparrow}}{\boxed{f}} 
 \longrightarrow
 \by
 \longrightarrow 
 \boxed{g}
 \longrightarrow
 z
$$
where *$z$ is assumed to be a scalar*. Then each computation block (for example `vl_nnconv` or `vl_nnpool`) can compute $\partial z / \partial \bx$ and $\partial z / \partial \bw$ given as input $\bx$ and $\partial z / \partial \by$. Let's put this to the practice:
```matlab
% Read an example image
x = im2single(imread('peppers.png')) ;

% Create a bank of linear filters and apply them to the image
w = randn(5,5,3,10,'single') ;
y = vl_nnconv(x, w, []) ;

% Create the derivative dz/dy
dzdy = randn(size(y), 'single') ;

% Back-propagation
[dzdx, dzdw] = vl_nnconv(x, w, [], dzdy) ;
```
> **Task:** Run the code above and check the dimensions of `dzdx` and `dzdy`. Does this matches your expectations?

An advantage of this modular view is that new building blocks can be coded and added to the architecture in a simple manner. However, it is easy to make mistakes in the calculation of complex derivatives. Hence, it is a good idea to verify results numerically. Consider the following piece of code:
```matlab
% Check the derivative numerically
ex = randn(size(x), 'single') ;
eta = 0.0001 ;
xp = x + eta * ex  ;
yp = vl_nnconv(xp, w, []) ;

dzdx_empirical = sum(dzdy(:) .* (yp(:) - y(:)) / eta) ;
dzdx_computed = sum(dzdx(:) .* ex(:)) ;

fprintf(...
  'der: empirical: %f, computed: %f, error: %.2f %%\n', ...
  dzdx_empirical, dzdx_computed, ...
  abs(1 - dzdx_empirical/dzdx_computed)*100) ;
```

> **Questions:**
> 
> * What is the meaning f `ex` in the code above? 
> * What are the derivatives `dzdx_empirical` and `dzdx_comptued`?
>
> **Tasks:**
> 
> *  Run the code and convince yourself that `vl_nnconv` derivatives is (probably) correct.
> * Create a new version of this code to test the derivative calculation with respect to $\bw$.

We are now ready to build our first elementary CNN, composed of just two layers, and to compute its derivatives:
```matlab
% Parameters of the CNN
w1 = randn(5,5,3,10,'single') ;
rho2 = 10 ;

% Run the CNN forward
x1 = im2single(imread('peppers.png')) ;
x2 = vl_nnconv(x1, w1, []) ;
x3 = vl_nnpool(x2, rho2) ;

% Create the derivative dz/dx3
dzdx3 = randn(size(x3), 'single') ;

% Run the CNN backward
dzdx2 = vl_nnpool(x2, rho2, dzdx3) ;
[dzdx1, dzdw1] = vl_nnconv(x1, w1, [], dzdx2) ;
```
> **Question:** Note that the last derivative in the CNN is `dzdx3`. Here, for the sake of the example, this derivative is initalized randomly. In a practical application, what would this derivative represent?

We can now use the same technique as before to check that the derivative computed through back-propagation are correct.

```matlab
% Check the derivative numerically
ew1 = randn(size(w1), 'single') ;
eta = 0.0001 ;
w1p = w1 + eta * ew1  ;

x1p = x1 ;
x2p = vl_nnconv(x1p, w1p, []) ;
x3p = vl_nnpool(x2p, rho2) ;

dzdw1_empirical = sum(dzdx3(:) .* (x3p(:) - x3(:)) / eta) ;
dzdw1_computed = sum(dzdw1(:) .* ew1(:)) ;

fprintf(...
  'der: empirical: %f, computed: %f, error: %.2f %%\n', ...
  dzdw1_empirical, dzdw1_computed, ...
  abs(1 - dzdw1_empirical/dzdw1_computed)*100) ;
```
 
## Part 3: learning a tiny CNN

In this part we will learn a very simple CNN. The CNN is composed of exactly two layers: a convolutional layear and a max-pooling layer:
$$
\bx_2 = W * \bx_1 + b, \qquad \bx_3 = \operatorname{maxpool}_\rho \bx_2.
$$
$W$ contains a single $3\times 3$ square filter, so that $b$ is a scalar. and the input image $\bx=\bx_1$ has a single channel.
> **Task**
> 
> * Open the file `tinycnn.m` and inspect the code. Convince yourself that the code computes the CNN just described.
> * Look at the paddings used in the code. If the input image $\bx_1$ has dimensions $M\times N$, what is the dimension of the output feature map $\bx_3$?
> 

In the rest of the section we will learn the CNN parameters in order to extract blob-like structures from images.

### Part 3.1: training data and labels

The first step is to load the image `data/dots.jpg` and to use the supplied `extractBlackBlobs` function to extract all the black dots in the image.
```matlab
% Load an image
im = rgb2gray(im2single(imread('data/dots.jpg'))) ;

% Compute the location of black blobs in the image
[pos,neg] = extractBlackBlobs(im) ;
```
The arrays `pos` and `neg` contain now pixel labels and  will be used as *annotations* for the supervised training of the CNN. These annotations can be visualised as follows:
```matlab
figure(1) ; clf ; 
subplot(1,3,1) ; imagesc(im) ; axis equal ; title('image') ;
subplot(1,3,2) ; imagesc(pos) ; axis equal ; title('positive points (blob centres)') ;
subplot(1,3,3) ; imagesc(neg) ; axis equal ; title('negative points (not a blob)') ;
colormap gray ; 
```

> **Task:** Inspect `pos` and `neg` and convince yourself that:
>
> - `pos` contains a single `true` value in correspondence of each blob centre;
> - `neg` contains a `true` value for each pixel sufficiently far away from a blob.
> 
> Are there pixels for which both `pos` and `neg` evaluate to false?

### Part 3.2: image preprocessing

Before we attempt to train the CNN, the image is pre-processed to remove its mean value. It is also smoothed by applying a Gaussian kernel of standard deviation 3 pixels:
```matlab
% Pre-smooth the image
im = vl_imsmooth(im,3) ;

% Subtract median value
im = im - median(im(:)) ;
```
We will come back to this preprocessing steps later.

### Part 3.3: learning with gradient descent

We will now setup a learning problem to learn $W$ and $b$ to detect black blobs in images. Recall that the CNN computes for each image pixel $(u,v)$ a score $f(\bx;\bw,b)_{(u,v)}$. We would like this score to be:

* at least as large as 1 for any pixel that is marked as a blob centre (`pos` or $(u,v)\in\cP$) and
* at most zero for any pixel that is marked as being far away from a blob (`neg` or $(u,v)\in\cN$).

We do so by defining and then optimising the following objective function:
$$
E(\bw,b) = 
\frac{\lambda}{2}\|\bw\|^2
+
\frac{1}{|\cP|}\sum_{(u,v) \in \cP}
\max\{0, 1 - f(\bx;\bw,b)_{(u,v)}\}
+
\frac{1}{|\cN|}\sum_{(u,v) \in \cN}
\max\{0, f(\bx;\bw,b)_{(u,v)}\}.
$$

> **Questions:**
> 
> - What can you say about the score of each pixel if $\lambda=0$ and $E(\bw,b) =0$?
> - Note that the objective enforces a *marging* between the scores of the positive and negative pixels. How much is this margin?

We can now train the CNN by minimising the objective function with respect to $\bw$ and $b$. We do so by using an algorithm called *gradient descent with momentum*.  Given the current solution $(\bw_t,b_t)$ and update it , this is updated to $(\bw_{t+1},b_t)$ by following the direction of fastest descent as given by the negative gradient $-\nabla E(\bw_t,b_t)$ of the objective. However, gradient updates are smoothed by considering a *momentum* term $(\bar\bw_{t}, \bar\mu_t)$, yielding the update equations
$$
 \bar\bw_{t+1} \leftarrow \mu \bar\bw_t + \eta \frac{\partial E}{\partial \bw_t},
 \qquad
 \bw_{t+1} \leftarrow \bw_{t} - \bar\bw_t.
$$
and similarly for the bias term. Here $\mu$ is the *momentum rate* and $\eta$ the *learning rate*.

> **Questions:**
>
> - Explain why the momentum rate must be smaller than 1. What is the effect of having a momentum rate close to 1?
> - The learning rate establishes how fast the algorithm will try to minimise the objective function. Can you see any problem with a large learning rate?

The parameters of the algorithm are set as follows:
```Malta
numIterations = 500 ;
rate = 5 ;
momentum = 0.9 ;
shrinkRate = 0.0001 ;
plotPeriod = 10 ;
```

> **Tasks:**
> 
> - Inspect the code in the file  `exercise3.m`. Convince yourself that the code is implementing the algorithm described above. Pay particular attention at the forward and backward passes as well as at how the objective function and its derivatives are computed.
> - Run the algorithm and observe the results. Then answer the following questions:
>   * The learned filter should resemble the discretisation of a well-known differential operator. Which one? 
>   * What is the average of the filter values compared to the average of the absolute values?
> - Run the algorithm again and observe the evolution of the histograms of the score of the positive and negative pixels in relation to the values 0 and 1. Answer the following:
>   * Is the objective function minimised monotonically?
>   * As the histograms evolve, can you identify at least two "phases" in the optimisation?
>   * Once converged, do the score distribute in the manner that you would expect?
>
> **Hint:** the `plotPeriod` option can be changed to plot the diagnostic figure with a higher or lower frequency; this can singificantly affect the speed of the algorithm.

### Part 3.4: Experimenting with the tiny CNN

In this part we will experiment with several variants of the network just learned. First, we study the effect of the image smoothing:

> **Task:** Train again the tiny CNN *without smoothing the input image in prerprocessing*. Anser the following questions:
> 
>   * Is the learned filter very different from the one learned before?
>   * If so, can you figure out what "went wrong"?
>   * Look carefully at the output of the first layer, magnifying with th loupe tool. Is the maximal filter response attained in the middle of each blob?
>
> **Hint:** The Laplacian of Gaussian operator responds maximally at the centre of a blob only if the latter matches the blob size. Relate this fact to the combination of pre-smoothing the image and applying the learned $3\times 3$ filter.

Now restore the smoothing but switch off subtracting the median from the input image.

> **Task:** Train again the tiny CNN *without subtracting the median value in prerprocessing*. Anser the following questions:
>
>   * Does the algorithm converge?
>   * Reduce a hundred-fold the learning are and increase the maximum number of iterations by an equal amount. Does it get better?
>   * Explain why adding a constant to the input image can have such a dramatic effect on the performance of the optimisation.
>
> **Hint:** What constraint should the filter $\bw$ satisfy if the filter output should be zero when (i) the input image is zero or (ii) the input image is a large constant? Do you think that ti would be easy for gradient descent to enforce (ii) at all times?

What you have just witnessed is actually a fairly general principle: centering the data usually makes learning problems much better conditioned.

Now we will explore several parameters in the algorithms:

> **Task:** Restore the preprocessing as given in `experiment4.m`.  Try the following:
>
> * Try increasing th learning rate `eta`. Can you achieve a better value of the energy in the 500 iterations?
> * Disable momentum by setting `momentum = 0`. Now try to beat the result obtained above by choosing `eta`. Can you succeed?

Finally, consider the regularization effect of shrinking:

> **Task:** Restore the learning rate and momentum as given in `experiment4.m`. Then increase the shrkincage factor tenfold and a hundred-fold.
> 
> - What is the effect on the convergence speed?
> - What is the effect on the final value of the total objective function and of the average loss part of it?

## Part 4:


[^lattice]: A two-dimensional *lattice* is a discrete grid embedded in $R^2$, similar for example to a checkerboard.
