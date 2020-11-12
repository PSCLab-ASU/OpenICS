function [picks,P] = selectPF(tol_h, tol_w, h, w)

half_h     =   round( h/2 );
half_w     =   round( w/2 );
[I,J]      =   meshgrid(1:w, 1:h);
P          =   abs(I - half_h*(rand(h, w) + .5)) < tol_h & ...
               abs(J - half_w*(rand(h, w) + .5)) < tol_w;

P(half_h+1:h, :)          =   0;
P(half_h:half_h+1, :)     =   1;
P(:,half_w:half_w+1)      =   1;
P                         =   ifftshift(P);
P(1,1)                    =   1;
picks                     =   find(P);