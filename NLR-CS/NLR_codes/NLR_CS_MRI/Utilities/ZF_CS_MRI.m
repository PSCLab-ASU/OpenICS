function  [Rec_im, PSNR, SNR]  =  ZF_CS_MRI ( data, rate, L, s_model )
randn('state',0);

I            =   data;
I            =   I./max( abs(I(:)) );
ori_im       =   I.*255;
imwrite( abs(ori_im)./max(abs(ori_im(:))), 'Ori_im.tif' );   

par.h        =   size(ori_im, 1);
par.w        =   size(ori_im, 2);     
[A, At, P]   =   Compressive_sensing(ori_im, rate, L, s_model);
par.y        =   A(ori_im(:));
par.picks    =   P;

mask=zeros(size(ori_im));
mask(P)=1;
imwrite(fftshift(mask),'t.tif');

par.ori_im      =   ori_im;
[Rec_im PSNR, SNR]   =   Zero_filling( par, A, At );  
Rec_im          =   Rec_im./255;
return;



function [A, At, P]   =  Compressive_sensing(im, rate, L, model)
if model==1 || model==2
    [A, At, P]    =  Random_Sensing(im, rate, model);
elseif model==3
    [A, At, P]    =  Radial_Line_Sensing(im, L);
end
return;


function [A, At, P]   =  Random_Sensing( im, rate, model )
rand('seed',0);
[h w ch]     =    size(im);

if model==1
    N            =    h*w;
    K            =    round( N*rate );
    P            =    randperm(N)';
    q            =    randperm(N/2-1)+1;
    OMEGA        =    q(1:ceil(K/2))';
    A            =    @(z) A_f(z, OMEGA, P);
    At           =    @(z) At_f(z, N, OMEGA, P);
    P            =    OMEGA;
else    
    if rate==0.2
        factor   =    4.427;
    elseif rate==0.25
        factor   =    4;
    else
        factor   =    sqrt(1/rate)*2;
    end
    picks        =    RandMask_rect( double(h/factor), double(w/factor), h, w);
    A            =    @(z) A_fhp(z, picks, h, w);
    At           =    @(z) At_fhp(z, picks, h, w);
    P            =    picks;
end
return;


function [A, At, P]   =  Radial_Line_Sensing( im, L )
rand('seed',0);
[h w ch]            =    size(im);

if 1
    aperture=(pi/180)*180;    % aperture encompassing all scanning angles (aperture<pi is limited angle)
    direction=(pi/180)*0;     % direction of the scanning beam (middle axis)
else
    aperture=(pi/180)*90;     % aperture encompassing all scanning angles (aperture<pi is limited angle)
    direction=(pi/180)*45;    % direction of the scanning beam (middle axis)    
end
S       =    LineMaskLimitedAngle(L,h,aperture,direction);
P       =    find(S);
A       =    @(z) A_fhp(z, P, h, w);
At      =    @(z) At_fhp(z, P, h, w);
return


function [Rec_im, PSNR, SNR]  =  Zero_filling( par, A, At )
y          =   par.y;
Rec_im     =   reshape(At( y ), size(par.ori_im));
PSNR       =   csnr( abs(Rec_im),abs( par.ori_im), 0, 0 );
SNR        =   cal_snr( abs(Rec_im),abs( par.ori_im), 0, 0 );
return;
