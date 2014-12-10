function res = edgecnn(x, w, b, dzdy)
% EDGECNN  A very simple CNN that could be used to detect edges in an image

pad1 = ([size(w,1) size(w,1) size(w,2) size(w,2)] - 1) / 2 ;
pad1 = 0 ;
rho3 = 1 ;
pad3 = (rho3 - 1) / 2 ;

% Forward pass
res.x1 = x ;
res.x2 = vl_nnconv(res.x1, w, b, 'pad', pad1) ;
%res.x3 = abs(res.x2) ;
res.x3 = res.x2 ;
res.x4 = vl_nnpool(res.x3, rho3, 'pad', pad3) ;

% Backward pass (only if passed output derivative)
if nargin > 3
  res.dzdx4 = dzdy ;
  res.dzdx3 = vl_nnpool(res.x3, rho3, res.dzdx4, 'pad', pad3) ;
  %res.dzdx2 = res.dzdx3 .* sign(res.x2) ;
  res.dzdx2 = res.dzdx3 ;
  [res.dzdx1, res.dzdw, res.dzdb] = ...
    vl_nnconv(res.x1, w, b, res.dzdx2, 'pad', pad1) ;
end
