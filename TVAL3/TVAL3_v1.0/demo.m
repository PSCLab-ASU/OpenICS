% This simple demo examines if TVAL3 works normally. Please try more demos
% in the "Demos" directory, which would show users what TVAL3 is capable of.
% 
% I: 64x64 phantom (real, two-dimentional)
% A: random matrix without normality and orthogonality (real)
% f: observation with/without noise (real)
%
% Written by: Chengbo Li
% Advisor: Prof. Yin Zhang and Wotao Yin
% CAAM department, Rice University
% 05/21/2009

clear; close all;
path(path,genpath(pwd));
fullscreen = get(0,'ScreenSize');

% problem size
n = 64;
ratio = .3;
p = n; q = n; % p x q is the size of image
m = round(ratio*n^2);

% sensing matrix
A = rand(m,p*q)-.5;

% original image
I = phantom(n);
nrmI = norm(I,'fro');
figure('Name','TVAL3','Position',...
    [fullscreen(1) fullscreen(2) fullscreen(3) fullscreen(4)]);
subplot(121); imshow(I,[]);
title('Original phantom','fontsize',18); drawnow;

% observation
f = A*I(:);
favg = mean(abs(f));

% add noise
f = f + .00*favg*randn(m,1);


%% Run TVAL3
clear opts
opts.mu = 2^8;
opts.beta = 2^5;
opts.tol = 1E-3;
opts.maxit = 300;
opts.TVnorm = 1;
opts.nonneg = true;

t = cputime;
[U, out] = TVAL3(A,f,p,q,opts);
t = cputime - t;


subplot(122); 
imshow(U,[]);
title('Recovered by TVAL3','fontsize',18);
xlabel(sprintf(' %2d%% measurements \n Rel-Err: %4.2f%%, CPU: %4.2fs ',ratio*100,norm(U-I,'fro')/nrmI*100,t),'fontsize',16);
