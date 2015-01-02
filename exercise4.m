function exercise4()
% EXERCISE4   Part 4 of the VGG CNN practical

setup ;

% -------------------------------------------------------------------------
% Part 4.1: prepare data
% -------------------------------------------------------------------------

% Load character dataset
imdb = load('data/charsdb.mat') ;

% Visualize some of the data
figure(1) ; clf ; colormap gray ;
subplot(1,2,1) ;
vl_imarraysc(imdb.images.data(:,:,imdb.images.label==1 & imdb.images.set==1)) ;
axis equal ;
title('training chars for ''a''') ;

subplot(1,2,2) ;
vl_imarraysc(imdb.images.data(:,:,imdb.images.label==1 & imdb.images.set==2)) ;
axis equal ;
title('validation chars for ''a''') ;

% -------------------------------------------------------------------------
% Part 4.2: initialize a CNN architecture
% -------------------------------------------------------------------------

net = initializeCharacterCNN() ;

% -------------------------------------------------------------------------
% Part 4.3: train and evaluate the model
% -------------------------------------------------------------------------

trainOpts.batchSize = 100 ;
trainOpts.numEpochs = 100 ;
trainOpts.continue = true ;
trainOpts.useGpu = false ;
trainOpts.learningRate = 0.001 ;
trainOpts.numEpochs = 15 ;
trainOpts.expDir = 'data/chars-experiment' ;

% Take the average image out
imageMean = mean(imdb.images.data(:)) ;
imdb.images.data = imdb.images.data - imageMean ;
%bsxfun(@minus, imdb.images.data, mean(imdb.images.data,3)) ;

% Convert to a GPU array if needed
if trainOpts.useGpu
  imdb.images.data = gpuArray(imdb.images.data) ;
end

% Call training function in MatConvNet
[net,info] = cnn_train(net, imdb, @getBatch, trainOpts) ;

% Move the CNN back to CPU if it was trained on GPU
if trainOpts.useGpu
  net = vl_simplenn_move(net, 'cpu') ;
end

% Save the result for later use
net.layers(end) = [] ;
net.imageMean = imageMean ;
save('data/chars-experiment/charscnn.mat', '-struct', 'net') ;

% -------------------------------------------------------------------------
% Part 4.4: visualize the learned filters
% -------------------------------------------------------------------------

figure(2) ; clf ; colormap gray ;
vl_imarraysc(squeeze(net.layers{1}.filters),'spacing',2)
axis equal ;
title('filters in the first layer') ;

% -------------------------------------------------------------------------
% Part 4.5: train with jitter
% -------------------------------------------------------------------------

trainOpts.expDir = 'data/chars-jit-experiment' ;

% Initlialize a new network
net = initializeCharacterCNN() ;

% Call training function in MatConvNet
[net,info] = cnn_train(net, imdb, @getBatchWithJitter, trainOpts) ;

% Move the CNN back to CPU if it was trained on GPU
if trainOpts.useGpu
  net = vl_simplenn_move(net, 'cpu') ;
end

% Save the result for later use
net.layers(end) = [] ;
net.imageMean = imageMean ;
save('data/chars-experiment/charscnn-jit.mat', '-struct', 'net') ;

% -------------------------------------------------------------------------
% Part 4.4: apply the model
% -------------------------------------------------------------------------

% Load network and remove the last layer (loss)
% net = load('data/chars-experiment/charscnn.mat') ;
net = load('data/chars-experiment/charscnn-jit.mat') ;

if 1
  im = im2single(rgb2gray(imread('data/string.png'))) ;
  im = 256 * (im - net.imageMean) ;
else
  im = 256 * imdb.images.data(:,:,2) ;
  im = 256 * reshape(imdb.images.data(:,:,1:10),32,[]) ;
end
%im = 256 * (im - net.imageMean) ;

res = vl_simplenn(net, im) ;

% decode
for i=1:size(res(end).x,2)
  [score(i),pred(i)] = max(squeeze(res(end).x(1,i,:))) ;
end

fprintf('interpretation: %s\n', imdb.meta.classes(pred)) ;
keyboard

% --------------------------------------------------------------------
function [im, labels] = getBatch(imdb, batch)
% --------------------------------------------------------------------
im = imdb.images.data(:,:,batch) ;
im = 256 * reshape(im, 32, 32, 1, []) ;
labels = imdb.images.label(1,batch) ;

% --------------------------------------------------------------------
function [im, labels] = getBatchWithJitter(imdb, batch)
% --------------------------------------------------------------------
im = imdb.images.data(:,:,batch) ;
labels = imdb.images.label(1,batch) ;

n = numel(batch) ;
train = find(imdb.images.set == 1) ;

sel = randperm(numel(train), n) ;
im1 = imdb.images.data(:,:,sel) ;

sel = randperm(numel(train), n) ;
im2 = imdb.images.data(:,:,sel) ;

ctx = [im1(:,17:32,:) im2(:,1:16,:)] ;

im = min(im, ctx) ;
im = 256 * reshape(im, 32, 32, 1, []) ;
