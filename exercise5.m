function exercise5()
% EXERCISE5   Part 5 of the VGG CNN practical

setup ;

% -------------------------------------------------------------------------
% Part 5.1: load a pretrained model
% -------------------------------------------------------------------------

net = load('data/imagenet-vgg-verydeep-16.mat') ;
vl_simplenn_display(net) ;

% -------------------------------------------------------------------------
% Part 5.2: use the model to classify an image
% -------------------------------------------------------------------------

% obtain and preprocess an image
im = imread('peppers.png') ;
im_ = single(im) ; % note: 255 range
im_ = imresize(im_, net.normalization.imageSize(1:2)) ;
im_ = im_ - net.normalization.averageImage ;

% run the CNN
res = vl_simplenn(net, im_) ;

% show the classification result
scores = squeeze(gather(res(end).x)) ;
[bestScore, best] = max(scores) ;
figure(1) ; clf ; imagesc(im) ; axis image ;
title(sprintf('%s (%d), score %.3f',...
net.classes.description{best}, best, bestScore)) ;
