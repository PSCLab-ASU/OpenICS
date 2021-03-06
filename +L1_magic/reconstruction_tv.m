% reconstruction_tv.m
% 
% Uses TV (total variance) reconstruction methods from l1-magic toolbox.
% 
% Usage: [x_hat,specifics,runtime] = reconstruction_tv(x,y,img_dims,A,At,specifics)
%
% x - nx1 vector, original signal
%
% y - mx1 vector, observations
%
% img_dims - Vector denoting the size of the original image. [c,w,h]
%
% A - Function handle to sensing method
%
% At - Function handle to transposed sensing method
%
% specifics - struct that may contain the following arguments:
%
%   constraint - The constraint to use. 'eq', 'qc', or 'dantzig'.
%       Default: 'eq'
%
%   lbtol - The tolerance of the log-barrier algorithm, which terminates
%           when duality gap <= lbtol.
%       Default: 1e-2
%
%   mu - Factor by which to increase the barrier constant per iteration.
%       Default: 2
%
%   lintol - The tolerance of the linear equation solving algorithm. Either
%            Conjugated Gradients or Symmetric LQ method.
%       Default: 1e-8
%
%   linmaxiter - The max iterations of CG or SymmLQ.
%       Default: 200
%
%   epsilon - The constraint relaxation parameter, AKA allowed error. Only
%             applied on 'qc' and 'dantzig' constraints.
%       Default: 5e-3
%
%   normalization - Whether the image should be normalized. May help make
%                   the image sparser and improve reconstruction accuracy.
%       Default: false
%

function [x_hat,specifics,runtime] = reconstruction_tv(x, y, img_dims, A, At, specifics)

    % set default values
    if ~isfield(specifics, 'constraint')
        specifics.constraint = 'eq';
    end
    
    if ~isfield(specifics, 'lbtol')
        specifics.lbtol = 1e-2;
    end
    
    if ~isfield(specifics, 'mu')
        specifics.mu = 2;
    end
    
    if ~isfield(specifics, 'lintol')
        specifics.lintol = 1e-8;
    end
    
    if ~isfield(specifics, 'linmaxiter')
        specifics.linmaxiter = 200;
    end
    
    if ~isfield(specifics, 'normalization')
        specifics.normalization = false;
    end
    
    % all constraints besides 'eq' require an epsilon argument
    if ~strcmp(specifics.constraint, 'eq') && ~isfield(specifics, 'epsilon')
        specifics.epsilon = 5e-3;
    end
    
    % apply normalization to image
    if specifics.normalization
        x_norm = norm(x(:));
        x = x / x_norm;
        x_mean = mean(x(:));
        x = x - x_mean;
        y = A(x(:));
    end

    x0 = At(y); % lowest energy initial point
    time0 = clock;
    
    if strcmp(specifics.constraint, 'eq')
        x_hat = tveq_logbarrier(x0, A, At, y, specifics.lbtol, specifics.mu, specifics.lintol, specifics.linmaxiter);
    elseif strcmp(specifics.constraint, 'qc')
        x_hat = tvqc_logbarrier(x0, A, At, y, specifics.epsilon, specifics.lbtol, specifics.mu, specifics.lintol, specifics.linmaxiter);
    elseif strcmp(specifics.constraint, 'dantzig')
        x_hat = tvdantzig_logbarrier(x0, A, At, y, specifics.epsilon, specifics.lbtol, specifics.mu, specifics.lintol, specifics.linmaxiter);
    end
    
    runtime = etime(clock, time0);
    fprintf('Total elapsed time = %f secs\n\n', runtime);
    
    % invert normalization
    if specifics.normalization
        x_hat = x_hat + x_mean;
        x_hat = x_hat * x_norm;
    end
end
