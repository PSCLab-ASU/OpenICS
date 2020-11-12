% L1_A_fhp.m
%
% Takes measurements in the upper half-plane of the 2D Fourier transform.
%
% Usage: b = L1_A_fhp(x, OMEGA)
%
% x - N vector
%
% b - K vector = [mean; real part(OMEGA); imag part(OMEGA)]
%
% OMEGA - K/2-1 vector denoting which Fourier coefficients to use
%         (the real and imag parts of each freq are kept).
%
% Written by: Justin Romberg, Caltech
% Created: October 2005
% Email: jrom@acm.caltech.edu
%
% Renamed to L1_A_fhp to distinguish from NLR-CS implementations

function y = L1_A_fhp(x, OMEGA)

n = round(sqrt(length(x)));

yc = 1/n*fft2(reshape(x,n,n));
y = [yc(1,1); sqrt(2)*real(yc(OMEGA)); sqrt(2)*imag(yc(OMEGA))];

