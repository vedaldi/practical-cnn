Convolutional neural network practical (PyTorch version)
========================================================

A computer vision practical by the Oxford Visual Geometry group,
authored by Andrea Vedaldi and Andrew Zisserman.

Running the practical on your local machine
-------------------------------------------

*   Install the appropriate PyTorch environment. The easiest way is to install [Anaconda](https://www.anaconda.com/download/) or [Miniconda](https://conda.io/miniconda.html) and then crate the environment `practical` using the provided `pratical.yaml` configuration file, as follows:

        conda env create -f practical.yaml

    This step needs to be run only once.

*   Activate the PyTorch environment for the current session:

        conda activate practical

*   Run the pratical

        jupyter notebook practical.yaml

Appendix: Installing from scratch
---------------------------------

*   Install and setup PyTorch as explained above.

*   For **developers**, it is also recommended to install the `nbstripout` command:
        
        nbstripout --install

    In this manner Git can strip the output data from the iPython notebook before checking in.

*   For bootstrapping the data for the practical follow these steps:

        cd practical-directory
        ./extra/genfonts.sh              # Download the Google Fonts to ./extra/fonts
        ./extra/genstring.sh             # Create data/sentence-lato.png
        python3 -m extra.pack_fonts      # Create data/charsdb.pth
        python3 -m extra.pretrain_models # Create data/model*.pth and data/alexnet.pth


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
2. Install the Conda environment as explained above, then activate it with
   `conda activate practical`.
2. Run `./extra/genfonts.sh`. This will download the Google Fonts
  and extract them as PNG files in `./extra/fonts`.
3. Run `./extra/genstring.sh`. This will create
  `data/sentence-lato.png`.
4. Run `python -m extra.pack_fonts`. This will create `data/charsdb.pth`.
5. Run `python -m extra.pretrain_models`. This will create `data/model*.pth` and `data/alexnet.pth`.

For developers, it is also recommended to install the `nbstripout` command 
in git using `nbstripout --install`.

Changes
-------

* *2018a* - PyTorch version.
* *2017a* - Removes dependency on VLFeat and upgrades MatConvNet.
* *2015a* - Initial edition.

License
-------

    Copyright (c) 2015-18 Andrea Vedaldi and Andrew Zisserman.
    
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
