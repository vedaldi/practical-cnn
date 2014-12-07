
run matconvnet/matlab/vl_setupnn ;
try
  vl_nnconv(single(1),single(1),[]) ;
catch
  warning('VL_NNCONV() does not seem to be compiled. Trying to compile now.') ;
  vl_compilenn2('enableGpu', false, 'verbose', true) ;
end

run vlfeat/toolbox/vl_setup ;