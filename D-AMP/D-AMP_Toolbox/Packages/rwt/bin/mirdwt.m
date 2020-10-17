function [x,L] = mirdwt(yl,yh,h,L)
%    function [x,L] = mirdwt(yl,yh,h,L);
% 
%    Function computes the inverse redundant discrete wavelet
%    transform x  for a 1D or 2D input signal. (Redundant means here
%    that the sub-sampling after each stage of the forward transform
%    has been omitted.) yl contains the lowpass and yl the highpass
%    components as computed, e.g., by mrdwt. In the case of a 2D
%    signal, the ordering in
%    yh is [lh hl hh lh hl ... ] (first letter refers to row, second
%    to column filtering).  
%
%    Input:
%       yl : lowpass component
%       yh : highpass components
%       h  : scaling filter
%       L  : number of levels. In the case of a 1D signal, 
%            length(yl) must  be divisible by 2^L;
%            in the case of a 2D signal, the row and
%            the column dimension must be divisible by 2^L.
%   
%    Output:
%	     x : finite length 1D or 2D signal
%	     L : number of levels
%
%  HERE'S AN EASY WAY TO RUN THE EXAMPLES:
%  Cut-and-paste the example you want to run to a new file 
%  called ex.m, for example. Delete out the % at the beginning 
%  of each line in ex.m (Can use search-and-replace in your editor
%  to replace it with a space). Type 'ex' in matlab and hit return.
%
%
%    Example 1:
%    xin = makesig('Leopold',8);
%    h = daubcqf(4,'min');
%    L = 1;
%    [yl,yh,L] = mrdwt(xin,h,L);
%    [x,L] = mirdwt(yl,yh,h,L)
%    x = 0.0000 1.0000 0.0000 -0.0000 0 0 0 -0.0000
%    L = 1
%  
%    Example 2:  
%    load lena;
%    h = daubcqf(4,'min');
%    L = 2;
%    [ll_lev2,yh,L] = mrdwt(lena,h,L); % lena is a 256x256 matrix
%    N = 256;
%    lh_lev1 = yh(:,1:N); 
%    hl_lev1 = yh(:,N+1:2*N); 
%    hh_lev1 = yh(:,2*N+1:3*N);
%    lh_lev2 = yh(:,3*N+1:4*N); 
%    hl_lev2 = yh(:,4*N+1:5*N); 
%    hh_lev2 = yh(:,5*N+1:6*N);
%    figure; colormap(gray); imagesc(lena); title('Original Image');
%    figure; colormap(gray); imagesc(ll_lev2); title('LL Level 2');
%    figure; colormap(gray); imagesc(hh_lev2); title('HH Level 2');
%    figure; colormap(gray); imagesc(hl_lev2); title('HL Level 2');
%    figure; colormap(gray); imagesc(lh_lev2); title('LH Level 2');
%    figure; colormap(gray); imagesc(hh_lev1); title('HH Level 1');
%    figure; colormap(gray); imagesc(hl_lev2); title('HL Level 1');
%    figure; colormap(gray); imagesc(lh_lev2); title('LH Level 1');
%    [lena_Hat,L] = mirdwt(ll_lev2,yh,h,L);
%    figure; colormap(gray); imagesc(lena_Hat); 
%                            title('Reconstructed Image');
%
%    See also: mdwt, midwt, mrdwt
%
%    Warning! min(size(yl))/2^L should be greater than length(h)
%
%Author: Markus Lang  <lang@jazz.rice.edu>
if exist('OCTAVE_VERSION', 'builtin')
  yl = yl * 1.0;
  yh = yh * 1.0;
  if (exist('L'))
    [x,L] = omirdwt(yl,yh,h,L);
  else  
    [x,L] = omirdwt(yl,yh,h);
  end
else
  error('You must compile wavelet toolbox before use')
end
