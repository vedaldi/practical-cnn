Convolutional neural network practical
======================================

A computer vision practical by the Oxford Visual Geometry group,
authored by Andrea Vedaldi and Andrew Zisserman.

This practical requires PyTorch and a number of other Python dependencies, such as MatPlotLib, NumPy, and PIL. The easiest way of installing all that is to install [Anaconda](https://www.anaconda.com/download/) for Python 3.6 and then to use the following commands:

      conda env create -f practical.yaml
      conda activate practical
      jupyter notebook practical.ipynb

Package contents
----------------

The practical consist of a iPython notebook and a supporting file:

* `practical.ipynb` -- The Python Jupyter practical
* `lab.py` -- Supporting Python code

The distribution also ships with some data and maintenance code.

Appendix: Installing from scratch
---------------------------------

The practical requires both VLFeat and MatConvNet. VLFeat comes with
pre-built binaries, but MatConvNet does not.

1. Set the current directory to the practical base directory.
2. Run `./extra/genfonts.sh`. This will download the Google Fonts
  and extract them as PNG files in `./extra/fonts`.
3. Run `./extra/genstring.sh`. This will create
  `data/sentence-lato.png`.
4. Run `conda activate practical ; python -m extra.pack_fonts`. This will create `data/charsdb.pth`.
5. Run `conda activate practical ; python -m extra.pretrain_models`. This will create `data/model*.pth` and `data/alexnet.pth`.

Changes
-------

* *2018a* - Rewrite for PyTorch.
* *2017a* - Removes dependency on VLFeat and upgrades MatConvNet.
* *2015a* - Initial edition.

License
-------

    Copyright (c) 2015 Andrea Vedaldi
    
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
