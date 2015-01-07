setup ;

% -------------------------------------------------------------------------
% Part 1.1: Linear convolution
% -------------------------------------------------------------------------

% Read an example image
x = imread('peppers.png') ;

% Convert to single format
x = im2single(x) ;

% Visualize the input x
figure(1) ; clf ; imagesc(x) ;

% Create a bank of linear filters
w = randn(5,5,3,10,'single') ;

% Apply the convolutional operator
y = vl_nnconv(x, w, []) ;

% Visualize the output y
figure(2) ; clf ; vl_imarraysc(y) ; colormap gray ;

% Try again, downsampling the output
y_ds = vl_nnconv(x, w, [], 'stride', 16) ;
figure(3) ; clf ; vl_imarraysc(y_ds) ; colormap gray ;

% Try padding
y_pad = vl_nnconv(x, w, [], 'pad', 4) ;
figure(4) ; clf ; vl_imarraysc(y_pad) ; colormap gray ;

% Manually design a filter
w = [0  1 0 ;
     1 -4 1 ;
     0  1 0 ] ;
w = single(repmat(w, [1, 1, 3])) ;
y_lap = vl_nnconv(x, w, []) ;
figure(5) ; clf ; colormap gray ;
subplot(1,2,1) ; imagesc(y_lap) ; title('filter output') ;
subplot(1,2,2) ; imagesc(-abs(y_lap)) ; title('- abs(filter output)') ;


% -------------------------------------------------------------------------
% Part 1.2: Non-linear gating (ReLU)
% -------------------------------------------------------------------------

w = single(repmat([1 0 -1], [1, 1, 3])) ;
w = cat(4, w, -w) ;
y = vl_nnconv(x, w, []) ;
z = vl_nnrelu(y) ;

figure(6) ; clf ; colormap gray ;
subplot(1,2,1) ; vl_imarraysc(y) ;
subplot(1,2,2) ; vl_imarraysc(z) ;

% -------------------------------------------------------------------------
% Part 1.2: Pooling
% -------------------------------------------------------------------------

y = vl_nnpool(x, 15) ;
figure(7) ; clf ; imagesc(y) ;

% -------------------------------------------------------------------------
% Part 1.3: Normalization
% -------------------------------------------------------------------------

rho = 5 ;
kappa = 0 ;
alpha = 1 ;
beta = 0.5 ;
y_nrm = vl_nnnormalize(x, [rho kappa alpha beta]) ;
figure(8) ; clf ; imagesc(y_nrm) ;
