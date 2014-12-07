% PREPARELABDATA

% --------------------------------------------------------------------
%                                                      Download VLFeat
% --------------------------------------------------------------------

if ~exist('vlfeat', 'dir')
  from = 'http://www.vlfeat.org/download/vlfeat-0.9.18-bin.tar.gz' ;
  fprintf('Downloading vlfeat from %s\n', from) ;
  untar(from, 'data') ;
  movefile('data/vlfeat-0.9.18', 'vlfeat') ;
end

setup ;
