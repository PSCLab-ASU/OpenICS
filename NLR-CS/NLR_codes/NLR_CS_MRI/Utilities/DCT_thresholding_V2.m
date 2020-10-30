function  im_out   =  DCT_thresholding_V2( im, par, D,t0 )

b        =  par.win;
s        =  par.step;
b2       =  b*b;
[h  w]   =  size(im);

N       =  h-b+1;
M       =  w-b+1;
r       =  [1:s:N];
r       =  [r r(end)+1:N];
c       =  [1:s:M];
c       =  [c c(end)+1:M];

N       =  length(r);
M       =  length(c);
L       =  N*M;
X       =  zeros(b2, L);

% For the Y component
k    =  0;
for i  = 1:b
    for j  = 1:b
        k    =  k+1;
        blk  =  im(r-1+i,c-1+j);
        X(k,:) =  blk(:)';
    end
end
% t0    =   par.t0;
% Y     =   D'*soft( D*X, t0 );
Y     =   D'*Group_soft( D*X, t0 );


% Output the processed image
im_out   =  zeros(h,w);
im_wei   =  zeros(h,w);
k        =  0;
for i  = 1:b
    for j  = 1:b
        k    =  k+1;
        im_out(r-1+i,c-1+j)  =  im_out(r-1+i,c-1+j) + reshape( Y(k,:)', [N M]);
        im_wei(r-1+i,c-1+j)  =  im_wei(r-1+i,c-1+j) + 1;       
        
%         im_out(i:h-b+i,j:w-b+j)  =  im_out(i:h-b+i,j:w-b+j) + reshape( Y(k,:)', [N M]);
%         im_wei(i:h-b+i,j:w-b+j)  =  im_wei(i:h-b+i,j:w-b+j) + 1;
    end
end

im_out  =  im_out./(im_wei+eps);
return;


% Group sparsity 
function  Y  =  Group_soft( X, tau )
Y  =  X .* ( soft( abs(X), tau )./abs(X) );
return;

