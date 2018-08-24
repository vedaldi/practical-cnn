function ims = imsmooth(im,sigma)
% IMSMOOTH  Smooth an image using a Gaussian filter
%    IMS = IMSMOOTH(IM,SIGMA) applies a Gaussian filter of standard
%    deviation SIGMA to the image.

w = ceil(sigma*3.5) ;
h = exp(-0.5*((-w:w)/sigma).^2) ; h = h / sum(h) ;
ims = conv2(h,h,im,'same') ;
