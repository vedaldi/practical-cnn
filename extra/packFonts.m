function packFonts()
% run extra/genfont.sh first

fonts = dir(fullfile('extra','fonts')) ;
fonts = {fonts([fonts.isdir]).name} ;
fonts(ismember(fonts,{'.','..'})) = [] ;
chars = 'a':'z' ;

im = cell(numel(fonts), numel(chars)) ;
labels = cell(numel(fonts), numel(chars)) ;

for i = 1:numel(fonts)
  for j = 1:numel(chars)
    [p,cmap] = imread(fullfile('extra', 'fonts', fonts{i}, [chars(j) '.png'])) ;
    im{i,j} = im2single(ind2gray(p,cmap)) ;
    labels{i,j} = j ;
  end
end

imdb.meta.classes = chars ;
imdb.meta.sets = {'train', 'val'} ;
imdb.meta.fonts = fonts ;
imdb.images.id = 1:numel(im) ;
imdb.images.data = cat(3, im{:}) ;
imdb.images.label = cat(2, labels{:}) ;

% Create training and validation sets
sets = ones(numel(fonts), numel(chars)) ;
sets(end-349:end,:) = 2 ;
imdb.images.set = sets(:)' ;

save('data/charsdb.mat', '-struct', 'imdb') ;
