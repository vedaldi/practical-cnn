function decodeCharacters(net, imdb, im, res)
% DECODECHARACTERS(NET, IMDB, IM, RES) visualizes the characters
% recognized by a Character CNN (see initializeCharacterCNN()) in a 32
% x W image of a sentence.

for i=1:size(res(end).x,2)
  [score(i),pred(i)] = max(squeeze(res(end).x(1,i,:))) ;
end
fprintf('%s: decode chars: %s\n', mfilename, imdb.meta.classes(pred)) ;

clf ; colormap gray ;
subplot(2,1,1) ;
imagesc(im) ; axis equal off ;
ylim([0 50]) ; hold on ;
x = linspace(16, size(im,2)-16, numel(pred)) ;
for i=1:numel(x)
  text(x(i), 40, imdb.meta.classes(pred(i)), 'fontsize', 10, 'color', 'b') ;
end
title('input image and predicted characters (in blue)') ;

subplot(2,1,2) ;
imagesc(squeeze(res(end).x)') ;
axis equal tight ;
title('character scores') ;
