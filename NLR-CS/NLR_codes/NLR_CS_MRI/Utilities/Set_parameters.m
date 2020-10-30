function  par  =  Set_parameters(rate, L, s_model)
par.win      =   6;
par.nblk     =   43;    
par.step     =   min(6, par.win-1);
par.K0       =   5;
if s_model==1 
    if rate>=0.38
        par.K     =   20;
    else
        par.K     =   30;
    end
    
    if rate <= 0.052 
        par.lamada     =   0.2; 
        par.t0         =   2.0;   % Parameters for DCT recovery
        
        par.nSig       =   6.8; 
        par.c0         =   2.1;
        par.c1         =   6.0;
        
    elseif rate<=0.1  
        par.lamada     =   0.9;
        par.t0         =   5.5;
        
        par.nSig       =   6.4;
        par.c0         =   1.9;
        par.c1         =   5.7;
    
    elseif rate<=0.22   
        par.lamada     =   0.9;
        par.t0         =   5.2; 
        
        par.nSig       =   4.0;
        par.c0         =   1.1;
        par.c1         =   3.0;
        
    elseif rate<=0.32             
        par.lamada     =   0.7;      
        par.t0         =   0.6;     
        
        par.nSig       =   3.3; 
        par.c0         =   0.5;  
        par.c1         =   2.0;
        
    elseif rate<=0.42  
        par.lamada     =   0.7;
        par.t0         =   0.5; 
    
        par.nSig       =   2.5;   
        par.c0         =   0.43;
        par.c1         =   1.7;   
    elseif rate<=0.52 
        par.lamada     =   0.4;
        par.t0         =   0.25; 
    
        par.nSig       =   2.3;  
        par.c0         =   0.35;   
        par.c1         =   1.6;  
        
    end
    
elseif s_model==2
    if rate<=0.2
        par.K     =   18;
    else
        par.K     =   12;
    end
    
    if rate <= 0.052    %  L <= 15
        par.lamada     =   0.2; 
        par.t0         =   2.0;   % Parameters for DCT recovery
        
        par.nSig       =   6.4; 
        par.c0         =   2.1;
        par.c1         =   6.6;
        
    elseif rate<=0.1  % L <= 25
        par.lamada     =   0.9;
        par.t0         =   5.5;
        
        par.nSig       =   8.4;
        par.c0         =   1.6;
        par.c1         =   5.8; 
    
    elseif rate<=0.2    %  L <= 35
        par.lamada     =   0.9;
        par.t0         =   5.2; 
        
        par.nSig       =   5.8;   
        par.c0         =   0.69; 
        par.c1         =   3.6;  
        
    elseif rate<=0.3   % L <= 45
        par.lamada     =   0.7;      
        par.t0         =   0.6;     
        
        par.nSig       =   2.8;  
        par.c0         =   0.35;  
        par.c1         =   1.75;  
        
    elseif rate<=0.4   % L <= 55
        par.lamada     =   0.7;
        par.t0         =   0.5; 
    
        par.nSig       =   2.4;
        par.c0         =   0.3;  
        par.c1         =   1.8; 
        
    else  % rate<=0.51  % L <= 65
        par.lamada     =   0.4;
        par.t0         =   0.25; 
    
        par.nSig       =   2.6; 
        par.c0         =   0.34;  
        par.c1         =   1.8;
        
    end     
end

return;