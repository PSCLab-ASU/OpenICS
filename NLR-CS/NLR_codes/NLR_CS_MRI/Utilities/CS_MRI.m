function  [Rec_im, PSNR, SNR]  =  CS_MRI ( I, rate, L, s_model )
randn('state',0);
ori_im       =   I.*255;
[A, At, P]   =   Compressive_sensing(ori_im, rate, L, s_model);
[h,w]        =   size(ori_im);
rate         =   length(P)/(w*h);
disp( sprintf('CS-MRI rate: %3.2f\n', rate) );

par          =   Set_parameters(rate, L, s_model);
par.h        =   size(ori_im, 1);
par.w        =   size(ori_im, 2);     

par.y        =   A(ori_im(:));
par.picks    =   P;

par.ori_im              =     ori_im;   % For computing the PSNR only
[Rec_im PSNR]           =     NLR_CS_MRI( par, A, At );  
Rec_im                  =     Rec_im./255;
return;


function [A, At, P]   =  Compressive_sensing(im, rate, L, model)
if model==1 
    [A, At, P]    =  Random_Sensing(im, rate);
elseif model==2
    [A, At, P]    =  Radial_Line_Sensing(im, L);
end
return;


function [A, At, P]   =  Random_Sensing( im, rate )
rand('seed',0);
[h w ch]     =    size(im);

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

