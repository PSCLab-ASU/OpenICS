function [A, At]   =  Random_Sensing( im, rate )
rand('seed',0);

[h w ch]     =    size(im);
N            =    h*w;
K            =    round( N*rate );
P            =    randperm(N)';
q            =    randperm(N/2-1)+1;
OMEGA        =    q(1:K/2)';
A            =    @(z) A_f(z, OMEGA, P);
At           =    @(z) At_f(z, N, OMEGA, P);
return