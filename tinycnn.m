function res = tinycnn(x, w, b, dzdy)
% TINYCNN  A very simple CNN
%   RES = TINYCNN(X, W, B) evaluates a CNN with two layers: linear
%   filtering and max pooling. W is a QxQ filter and B its (scalar) bias
%   and X a MxN input image.
%
%   RES = TINYCNN(X, W, B, DZDY) backpropagates the CNN loss derivative DZDY
%   thorugh the network.
%
%   RES.X1, RES.X2, and RES.X3 contain the input of the first and second
%   CNN layers and of the CNN loss. RES.DZDX1, RES.DZDX2, and RES.DZDX3
%   contain the corresponding derivatives. RES.DZDW and RES.DZDB contain
%   the derivatives of the loss with respect to the parameters W and B.

% Author: Andrea Vedaldi

% Paramters of the layers
pad1 = ([size(w,1) size(w,1) size(w,2) size(w,2)] - 1) / 2 ;
rho2 = 3 ;
pad2 = (rho2 - 1) / 2 ;

% Forward pass
res.x1 = x ;
res.x2 = vl_nnconv(res.x1, w, b, 'pad', pad1) ;
res.x3 = vl_nnpool(res.x2, rho2, 'pad', pad2) ;

% Backward pass (only if passed output derivative)
if nargin > 3
  res.dzdx3 = dzdy ;
  res.dzdx2 = vl_nnpool(res.x2, rho2, res.dzdx3, 'pad', pad2) ;
  [res.dzdx1, res.dzdw, res.dzdb] = ...
    vl_nnconv(res.x1, w, b, res.dzdx2, 'pad', pad1) ;
end
