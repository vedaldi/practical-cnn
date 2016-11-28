#!/bin/bash
if test ! -e vlfeat
then
    wget http://www.vlfeat.org/download/vlfeat-0.9.20-bin.tar.gz --output-document=data/vlfeat.tar.gz --continue
    tar xzvf data/vlfeat.tar.gz
    mv vlfeat-0.9.20 vlfeat
fi

if test ! -e matconvnet
then
    wget http://www.vlfeat.org/sandbox-matconvnet/download/matconvnet-1.0-beta23.tar.gz \
         --output-document=data/matconvnet.tar.gz --continue
    tar xzvf data/matconvnet.tar.gz
    mv matconvnet-1.0-beta23 matconvnet
fi

if test ! -e imagenet-vgg-verydeep-16.mat
then
    wget http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-16.mat \
        --output-document=data/imagenet-vgg-verydeep-16.mat --continue
fi

