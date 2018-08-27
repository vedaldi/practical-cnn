# Build practical

name ?= practical-cnn
ver ?= pytorch-2018a

code=\
practica.pipynb \
lab.py \
README.md

doc=\
doc/images \
doc/instructions.html

data=\
data/alexnet.pth \
data/charscnn_jitter.pth \
data/charscnn.pth \
data/charsdb.pth \
data/dots.jpg \
data/imnet_classes.json \
data/peppers.png \
data/sentence-lato.png

include extra/practical/Makefile

clean: nbstripout

.PHONY: nbstripout
nbstripout:
	nbstripout -f practical.ipynb
