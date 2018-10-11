# Build practical

name ?= practical-cnn
ver ?= pytorch-2018a

code=\
README.md \
practical.ipynb \
practical.yaml \
lab.py

data=\
data/cover.png \
data/alexnet.pth \
data/charscnn_jitter.pth \
data/charscnn.pth \
data/charsdb.pth \
data/dots.jpg \
data/imnet_classes.json \
data/peppers.png \
data/erc.jpg \
data/epsrc.png \
data/sentence-lato.png

include extra/practical/Makefile

clean: nbstripout

.PHONY: nbstripout
nbstripout:
	nbstripout -f practical.ipynb
