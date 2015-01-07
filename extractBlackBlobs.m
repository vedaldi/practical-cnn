function [features, background] = extractImageBlobs(im)
% EXTRACTIMAGEBLOBS
[f,~,info] = vl_covdet(im, 'doubleimage', true, 'method', 'hessian') ;
ok = info.peakScores > 0.02 ;
idx = sub2ind(size(im), round(f(2,ok)), round(f(1,ok))) ;
features = false([size(im,1) size(im,2)]) ;
features(idx) = true ;
background = ~imdilate(features, strel('disk', 5, 0)) ;
