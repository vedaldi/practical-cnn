% -------------------------------------------------------------------------
% Part 3: Learning a simple CNN
% -------------------------------------------------------------------------

setup ;

% Load an image and compute its edge map
im_ = rgb2gray(im2single(imread('data/dots.jpg'))) ;
im = [im_, im_ ; flipud(im_), flipud(im_)'] ;
im = im_ ;
[features,background] = extractImageBlobs(im) ;
b = 1 ;
features = features(1+b:end-b,1+b:end-b) ;
background = background(1+b:end-b,1+b:end-b) ;

im = vl_imsmooth(im,3) ;
im = im - mean(im(:)) ;

figure(1) ; clf ; 
subplot(1,3,1) ; imagesc(im) ; axis equal ;
subplot(1,3,2) ; imagesc(features) ; axis equal ;
subplot(1,3,3) ; imagesc(background) ; axis equal ;
colormap gray ; 

% Initialize the network
w = randn(3, 3, 1) ;
%w = w * 10 ;
if 0
w =[...
 0  1 0 ;
 1 -4 1 ;
0  1 0 ] * 100;
end
w = single(w - mean(w(:))) ;
b = single(0) ;
dzdw_momentum = zeros('like', w) ;
dzdb_momentum = zeros('like', b) ;

% SGD parameters
T = 500 ;
eta = 5 ;
momentum = 0.9 ;
energy = zeros(1, T) ;
shrink = 0.000005 ;
wn = 2 ;
y = zeros(size(features),'single') ;
y(features) = +1 ;
y(background) = -1 ;
pos = features ;
neg = background ;
ignore = y(:) == 0 ;

% SGD with momentum
for t = 1:T
  
  if t > 100, eta=1;end
  % Forward pass
  res = edgecnn(im, w, b) ;
  
  % Loss  
  z = y .* (res.x4 - 1) ;
  E(t) = ...
    mean(max(0, 1 - res.x4(pos))) + ...
    wn*mean(max(0, res.x4(neg))) ;
  dzdx4 = ...
    - single(res.x4 < 1 & pos) / sum(pos(:)) + ...
    + wn*single(res.x4 > 0 & neg) / sum(neg(:)) ; 

  % Backward pass
  res = edgecnn(im, w, b, dzdx4) ;
  
  % Gradient step
  dzdw_momentum = momentum * dzdw_momentum + res.dzdw ;  
  dzdb_momentum = momentum * dzdb_momentum + res.dzdb ; 
  w = (1 - shrink) * w - eta * dzdw_momentum ;
  b = b - 0.1 * eta * dzdb_momentum ;
  
  w(:)'
  b
  
  % Plots
  if mod(t-1, 1) == 0
    fp = res.x4 > 0 & y < 0 ;
    fn = res.x4 < 1 & y > 0 ;
    tn = res.x4 <= 0 & y < 0 ;
    tp = res.x4 >= 1 & y > 0 ;
    err = cat(3, fp|fn , tp|tn, y==0) ;
    
    figure(2) ; clf ; colormap gray ;
    subplot(2,3,1) ; imagesc(res.x4) ; axis equal ; title('network output') ;
    subplot(2,3,2) ; plot(1:t, E(1:t)) ; grid on ; title('objective') ;
    subplot(2,3,3) ; vl_imarraysc(w) ; title('filter slices') ; axis equal ;
    subplot(2,3,4) ; imagesc(res.x2) ; title('first layer output') ; axis equal ;
    subplot(2,3,5) ; image(err) ;
    subplot(2,3,6) ;
    [h,x]=hist(res.x4(pos(:)),30) ;plot(x,h/max(h),'g') ;
    hold on ;
    [h,x]=hist(res.x2(neg(:)),30) ;plot(x,h/max(h),'r') ;
  end
end