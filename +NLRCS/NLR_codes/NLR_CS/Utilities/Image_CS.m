function  [Rec_im, PSNR, SSIM]    =  Image_CS( ori_im, s_model, rate, L)
randn('state',0);
rand('state',0);

par               =    Set_parameters(rate, L, s_model);
par.s_model       =    s_model;

[A, At, P]        =    Compressive_sensing(ori_im, rate, L, s_model);
par.picks         =    P;
par.y             =    A(ori_im(:));
par.ori_im        =    ori_im;   % For computing PSNR only
disp( sprintf('The sensing rate: %f\n', length(P)/(prod(size(ori_im)))) );

[Rec_im PSNR SSIM]      =   NLR_CS_Reconstruction( par, A, At ); 
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
    OMEGA        =    q(1:ceil(K))';
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
