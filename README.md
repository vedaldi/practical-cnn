Convolutional neural network practical
======================================

> A computer vision practical by the Oxford Visual Geometry group,
> authored by Andrea Vedaldi and Andrew Zisserman.

Start from `doc/instructions.html`.

Package contents
----------------

The practical consists of four exercises, organized in the following
files:

* `exercise1.m` -- Part 1: CNN fundamentals
* `exercise2.m` -- Part 2: Derivatives and backpropagation
* `exercise3.m` -- Part 3: Learning a tiny CNN
* `exercise4.m` -- Part 4: Learning a CNN to recognize characters
* `exercise5.m` -- Part V: Train your own object detector

The practical runs in MATLAB and uses
[MatConvNet](http://www.vlfeat.org/matconvnet) and
[VLFeat](http://www.vlfeat.org). This package contains the following
MATLAB functions:

* `extractBlackBlobs.m`: extract black blobs from an image.
* `tinycnn.m`: implements a very simple CNN.
* `setup.m`: setup MATLAB environment.

Appendix: Installing from scratch
---------------------------------

The practical requires both VLFeat and MatConvNet. VLFeat comes with
pre-built binaries, but MatConvNet does not.

1. From Bash, run `./extras/download.sh`. This will download the
   German Street Sign Benchmark data and VLFeat.
2. From MATLAB, run `addpath extras ; prepareLabData.m`.

Changes
-------

* *2014a* - Initial edition

License
-------

    Copyright (c) 2011-13 Andrea Vedaldi

    Permission is hereby granted, free of charge, to any person
    obtaining a copy of this software and associated documentation
    files (the "Software"), to deal in the Software without
    restriction, including without limitation the rights to use, copy,
    modify, merge, publish, distribute, sublicense, and/or sell copies
    of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be
    included in all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
    EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
    MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
    NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
    HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
    WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
    DEALINGS IN THE SOFTWARE.
Empty
