% At_fhp.m
%
% Adjoint of At_fhp (2D Fourier half plane measurements).
%
% Usage: x = At_fhp(b, OMEGA, n)
%
% b - K vector = [mean; real part(OMEGA); imag part(OMEGA)]
%
% OMEGA - K/2-1 vector denoting which Fourier coefficients to use
%         (the real and imag parts of each freq are kept).
%
% n - Image is nxn pixels
%
% x - N vector
%
% Written by: Justin Romberg, Caltech
% Created: October 2005
% Email: jrom@acm.caltech.edu
%

function x = At_fhp(y, OMEGA, h, w)

K = length(y);

fx = zeros(h,w);

fx(1,1) = y(1);
% fx(OMEGA) = (1/sqrt(2))*(y(2:(K+1)/2) + i*y((K+3)/2:K));
fx(OMEGA) = y(2:(K+1)/2) + i*y((K+3)/2:K);

% % Modified by W. Dong on Nov. 28 2011
% fx(OMEGA)  = sqrt(2)*(y(1:K/2) + i*y((K+2)/2:K));

n = round(sqrt(h*w));
%x = reshape(real(n*ifft2(fx)), h*w, 1);
% x = reshape(n*ifft2(fx), h*w, 1);
x = reshape(ifft2(fx), h*w, 1);
