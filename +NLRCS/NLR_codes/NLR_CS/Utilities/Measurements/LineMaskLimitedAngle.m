% LineMaskLimitedAngle.m
%
% Returns the indicator of the domain in 2D Fourier space for the
% specified line geometry.
% Usage :  [S,Mh,mi,mhi] = LineMaskLimitedAngle(L,n,aperture,direction)
%
% n : size of domain
% L : number of lines
% aperture : aperture of fan
% direction : central direction of fan
%
% Alessandro Foi, Tampere University of Technology (2006-2011)
%
% Based on the original file LineMask.m written by Justin Romberg (rev. 12/2/2004) as included in the L1magic package (http://www.acm.caltech.edu/l1magic/)

function S = LineMaskLimitedAngle(L,n,aperture,direction)

if nargin<3
    aperture=pi;
end
if nargin<4
    direction=0;
end

if (pi-aperture)>(aperture/L)
    thc = linspace(-direction-aperture/2, -direction+aperture/2, L);
else
    thc = linspace(-direction-pi/2, -direction+pi/2-pi/L, L);
end

thc=mod(thc,pi);

S = zeros(n);

% full mask
for ll = 1:L
    if ((thc(ll) <= pi/4) || (thc(ll) > 3*pi/4))
        yr = round(tan(thc(ll))*(-n/2+1:n/2-1)+n/2+1);
        for nn = 1:n-1
            S(yr(nn),nn+1) = 1;
        end
    else
        xc = round(cot(thc(ll))*(-n/2+1:n/2-1)+n/2+1);
        for nn = 1:n-1
            S(nn+1,xc(nn)) = 1;
        end
    end
end

S = ifftshift(S);
