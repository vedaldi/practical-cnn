setup ;

% -------------------------------------------------------------------------
% Part 2.1: Linear convolution derivatives
% -------------------------------------------------------------------------

% Read an example image
x = im2single(imread('peppers.png')) ;

% Create a bank of linear filters and apply them to the image
w = randn(5,5,3,10,'single') ;
y = vl_nnconv(x, w, []) ;

% Create the derivative dz/dy
dzdy = randn(size(y), 'single') ;

% Back-propagation
[dzdx, dzdw] = vl_nnconv(x, w, [], dzdy) ;

% Check the derivative numerically
ex = randn(size(x), 'single') ;
eta = 0.0001 ;
xp = x + eta * ex  ;
yp = vl_nnconv(xp, w, []) ;

dzdx_empirical = sum(dzdy(:) .* (yp(:) - y(:)) / eta) ;
dzdx_computed = sum(dzdx(:) .* ex(:)) ;

fprintf(...
  'der: empirical: %f, computed: %f, error: %.2f %%\n', ...
  dzdx_empirical, dzdx_computed, ...
  abs(1 - dzdx_empirical/dzdx_computed)*100) ;

% -------------------------------------------------------------------------
% Part 2.2: Back-propagation
% -------------------------------------------------------------------------

% Parameters of the CNN
w1 = randn(5,5,3,10,'single') ;
rho2 = 10 ;

% Run the CNN forward
x1 = im2single(imread('peppers.png')) ;
x2 = vl_nnconv(x1, w1, []) ;
x3 = vl_nnpool(x2, rho2) ;

% Create the derivative dz/dx3
dzdx3 = randn(size(x3), 'single') ;

% Run the CNN backward
dzdx2 = vl_nnpool(x2, rho2, dzdx3) ;
[dzdx1, dzdw1] = vl_nnconv(x1, w1, [], dzdx2) ;

% Check the derivative numerically
ew1 = randn(size(w1), 'single') ;
eta = 0.0001 ;
w1p = w1 + eta * ew1  ;

x1p = x1 ;
x2p = vl_nnconv(x1p, w1p, []) ;
x3p = vl_nnpool(x2p, rho2) ;

dzdw1_empirical = sum(dzdx3(:) .* (x3p(:) - x3(:)) / eta) ;
dzdw1_computed = sum(dzdw1(:) .* ew1(:)) ;

fprintf(...
  'der: empirical: %f, computed: %f, error: %.2f %%\n', ...
  dzdw1_empirical, dzdw1_computed, ...
  abs(1 - dzdw1_empirical/dzdw1_computed)*100) ;
