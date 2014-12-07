% -------------------------------------------------------------------------
% Part 3: Learning a simple CNN
% -------------------------------------------------------------------------

setup ;

% Load an image and compute its edge map
im = im2single(imread('peppers.png')) ;
[edges,fill] = extractImageEdges(im) ;
figure(1) ; clf ; colormap gray ; 
subplot(1,3,1) ; imagesc(im) ; axis equal ;
subplot(1,3,2) ; imagesc(edges) ; axis equal ;
subplot(1,3,3) ; imagesc(fill) ; axis equal ;

% Initialize the network
w = randn(3, 3, 3, 'single') ;
b = single(1) ;
dzdw_momentum = zeros('like', w) ;
dzdb_momentum = zeros('like', b) ;

% SGD parameters
T = 500 ;
eta = 0.005 ;
momentum = 0.9 ;
energy = zeros(1, T) ;
y = zeros(size(edges),'single') ;
y(edges) = +1 ;
y(fill) = -1 ;

% SGD with momentum
for t = 1:T
  % Forward pass
  res = edgecnn(im, w, b) ;
  
  % Loss  
  z = y .* res.x4 ;
  E(t) = mean(max(0, 1 - z(:))) ;
  dzdx4 = - y .* (z < 1) / numel(z) ;
  
  % Backward pass
  res = edgecnn(im, w, b, dzdx4) ;
  
  % Gradient step
  dzdw_momentum = momentum * dzdw_momentum + res.dzdw ;  
  dzdb_momentum = momentum * dzdb_momentum + res.dzdb ; 
  w = w - eta * dzdw_momentum ;
  b = b - eta * dzdb_momentum ;
  
  % Plots
  if mod(t-1, 20) == 0 
    figure(2) ; clf ; colormap gray ;
    subplot(2,2,1) ; imagesc(res.x4) ; axis equal ; title('network output') ;
    subplot(2,2,2) ; plot(1:t, E(1:t)) ; grid on ; title('objective') ;
    subplot(2,2,3) ; vl_imarraysc(w) ; title('filter slices') ; axis equal ;
    subplot(2,2,4) ; imagesc(res.x2) ; title('first layer output') ; axis equal ;
  end
end