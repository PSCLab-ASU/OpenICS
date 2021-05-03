function  par  =  Set_parameters(rate, L, s_model)
par.win        =    6;    % Patch size
par.nblk       =    45;    
par.step       =    min(6, par.win-1);

%-------------------------------------------------------
% The random sampling pattern used in L1-magic software
%-------------------------------------------------------
% added s_model 2 to parameters
if s_model==1 || s_model == 2
    par.K0     =   3;
    par.K      =   18;
    par.c0     =   0.49;
    
    if rate<=0.05
        par.t0         =   3.8;            %  Threshold for DCT reconstruction     
        par.nSig       =   4.66;           %  4.45          
        par.c0         =   0.6;            %  0.6     Threshold for warm start step
        par.c1         =   2.2;            %  1.96    Threshold for weighted SVT
            
    elseif rate<=0.1                      
        par.t0         =   2.4;     
        par.nSig       =   3.25;         
        par.c1         =   1.55;   
        
    elseif rate<=0.15                 
        par.t0         =   1.8;     
        par.nSig       =   2.65;         
        par.c1         =   1.35;  
        
    elseif rate<=0.2           
        par.t0         =   1.4;     
        par.nSig       =   2.35;         
        par.c1         =   1.32;  
        
    elseif rate<=0.25
        par.t0         =   1.0;     
        par.nSig       =   2.1;  
        par.c1         =   1.15;  
        
    elseif rate<=0.3
        par.t0         =   0.8; 
        par.nSig       =   1.8;      
        par.c1         =   0.9;
        
    else
        par.t0         =   0.8; 
        par.nSig       =   1.4;  
        par.c1         =   0.75; 
    end    
    
%------------------------------------------------
% Pseudo radial lines sampling patterns
%------------------------------------------------
elseif s_model==3                       
    par.K0             =   8;
    par.K              =   33;
    
    if L<=20
        par.t0         =   4.5;     %   Threshold for DCT reconstruction     
        par.nSig       =   8.8;         
        par.c0         =   0.68;        
        par.c1         =   2.7; 
        
    elseif L<=35
        par.t0         =   2.8;    
        par.nSig       =   6.9;        
        par.c0         =   0.75;       
        par.c1         =   1.8;
        
    elseif L<=50                          
        par.t0         =   3.5;          
        par.nSig       =   5.15;      
        par.c0         =   0.6;
        par.c1         =   0.85; 
        
    elseif L<=65
        par.t0         =   3.0;          
        par.nSig       =   4.3;         
        par.c0         =   0.58;
        par.c1         =   0.72; 

    elseif L<=80                          
        par.t0         =   2.5;          
        par.nSig       =   3.7;          
        par.c0         =   0.55;           
        par.c1         =   0.65; 
        
    else                                 
        par.t0         =   1.5;
        par.nSig       =   3.35;        
        par.c0         =   0.55;
        par.c1         =   0.64;
    end        
end

return;