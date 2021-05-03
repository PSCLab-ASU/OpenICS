%    COMPILE compiles the c files and generates mex files.
%
% Removed verbose options (-v) in all commands

if exist('OCTAVE_VERSION', 'builtin')
  mkoctfile --mex -DOCTAVE_MEX_FILE ../mex/mdwt.c   ../lib/src/dwt.c   ../lib/src/init.c ../lib/src/platform.c -I../lib/inc -o omdwt.mex
  mkoctfile --mex -DOCTAVE_MEX_FILE ../mex/midwt.c  ../lib/src/idwt.c  ../lib/src/init.c ../lib/src/platform.c -I../lib/inc -o omidwt.mex
  mkoctfile --mex -DOCTAVE_MEX_FILE ../mex/mrdwt.c  ../lib/src/rdwt.c  ../lib/src/init.c ../lib/src/platform.c -I../lib/inc -o omrdwt.mex
  mkoctfile --mex -DOCTAVE_MEX_FILE ../mex/mirdwt.c ../lib/src/irdwt.c ../lib/src/init.c ../lib/src/platform.c -I../lib/inc -o omirdwt.mex
else
  x = computer();
  if (x(length(x)-1:length(x)) == '64')
    mex -largeArrayDims ../mex/mdwt.c   ../lib/src/dwt.c   ../lib/src/init.c ../lib/src/platform.c -I../lib/inc -outdir ../bin
    mex -largeArrayDims ../mex/midwt.c  ../lib/src/idwt.c  ../lib/src/init.c ../lib/src/platform.c -I../lib/inc -outdir ../bin
    mex -largeArrayDims ../mex/mrdwt.c  ../lib/src/rdwt.c  ../lib/src/init.c ../lib/src/platform.c -I../lib/inc -outdir ../bin
    mex -largeArrayDims ../mex/mirdwt.c ../lib/src/irdwt.c ../lib/src/init.c ../lib/src/platform.c -I../lib/inc -outdir ../bin
  else
    mex -compatibleArrayDims ../mex/mdwt.c   ../lib/src/dwt.c   ../lib/src/init.c ../lib/src/platform.c -I../lib/inc -outdir ../bin
    mex -compatibleArrayDims ../mex/midwt.c  ../lib/src/idwt.c  ../lib/src/init.c ../lib/src/platform.c -I../lib/inc -outdir ../bin
    mex -compatibleArrayDims ../mex/mrdwt.c  ../lib/src/rdwt.c  ../lib/src/init.c ../lib/src/platform.c -I../lib/inc -outdir ../bin
    mex -compatibleArrayDims ../mex/mirdwt.c ../lib/src/irdwt.c ../lib/src/init.c ../lib/src/platform.c -I../lib/inc -outdir ../bin
  end
end
