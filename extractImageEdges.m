function [edges, fill] = extractImageEdges(im)
edges = edge(rgb2gray(im),'canny') ;
fill = ~imdilate(edges, strel('disk', 5, 0)) ;