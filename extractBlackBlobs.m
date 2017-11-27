function [features, background] = extractBlackBlobs(im)
% EXTRACTIMAGEBLOBS  Find the dark blobs in the images

f = hessiandet(im) ;
ok = f(3,:) > 0.0006 ;
idx = sub2ind(size(im), round(f(2,ok)), round(f(1,ok))) ;
features = false([size(im,1) size(im,2)]) ;
features(idx) = true ;
background = ~imdilate(features, strel('disk', 5, 0)) ;

b = 5 ;
features([1:b, end-b:end],:) = 0 ;
features(:,[1:b, end-b:end]) = 0 ;
background([1:b, end-b:end],:) = 0 ;
background(:,[1:b, end-b:end]) = 0 ;

function f = hessiandet(im)
% HESSIANDET  Basic Hessian detector
%   F = HESSIANDET(IM) runs a basic implementation of the Hessian
%   detector on the gray-scale image IM.

ims = imsmooth(im,2.5) ;

d2 = [1 -2 1];
d = [-1 0 1]/2 ;
im11 = conv2(1,d2,ims,'same') ;
im22 = conv2(d2,1,ims,'same') ;
im12 = conv2(d,d,ims,'same') ;
score = im11.*im22 - im12.*im12 ;

points = (islocalmax(score,1) .* islocalmax(score,2)) + ...
         (islocalmin(score,1) .* islocalmin(score,2)) ;
points = points .* (abs(score) > 0.0006) ;

scores = score(find(points)) ;
[i,j] = find(points) ;
f = [j(:),i(:),scores(:)]' ;

if 0
  clf ;subplot(1,2,1); imagesc(im) ; hold on ; colormap gray ;
  plot(f(1,:),f(2,:),'r.') ;
  subplot(1,2,2) ; imagesc(abs(score).^.25) ;
  keyboard
end

function m = islocalmax(x,dim)
m  = (circshift(x,1,dim) < x) & (circshift(x,-1,dim) < x) ;

function m = islocalmin(x,dim)
m = (circshift(x,1,dim) > x) & (circshift(x,-1,dim) > x) ;
