% reconstruction_l1.m
% 
% Uses L1 reconstruction methods from l1-magic toolbox.
% 
% Usage: x_hat = reconstruction_l1(x,y,img_dims,A,At,specifics)
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
%   constraint - The constraint to use. 'eq', 'qc', 'dantzig', or 'decode'.
%       Default: 'eq'
%
%   pdtol - The tolerance of the primal-dual algorithm, which terminates
%           when duality gap <= pdtol. Applied on all constraints except
%           'qc'.
%       Default: 1e-3
%
%   pdmaxiter - The max iterations of primal-dual algorithm. Applied on all
%               constraints except 'qc'.
%       Default: 50
%
%   cgtol - The tolerance of the Conjugated Gradients algorithm.
%       Default: 1e-8
%
%   cgmaxiter - The max iterations of CG.
%       Default: 200
%
%   epsilon - The constraint relaxation parameter, AKA allowed error. Only
%             applied on 'qc' and 'dantzig' constraints.
%       Default: 5e-3
%
%   lbtol - The tolerance of the log-barrier algorithm, which terminates
%           when duality gap <= lbtol. Only applied on 'qc' constraint.
%       Default: 1e-3
%
%   mu - Factor by which to increase the barrier constant per iteration.
%        Only applied on 'qc' constraint.
%       Default: 10
%
%   normalization - Whether the image should be normalized. May help make
%                   the image sparser and improve reconstruction accuracy.
%       Default: false
%

function x_hat = reconstruction_l1(x, y, img_dims, A, At, specifics)
    % set default values
    if ~isfield(specifics, 'constraint')
        specifics.constraint = 'eq';
    end
    
    if ~isfield(specifics, 'cgtol')
        specifics.cgtol = 1e-8;
    end
    
    if ~isfield(specifics, 'cgmaxiter')
        specifics.cgmaxiter = 200;
    end
    
    if ~isfield(specifics, 'normalization')
        specifics.normalization = false;
    end
    
    % 'qc' and 'dantzig' constraints require an epsilon argument
    if strcmp(specifics.constraint, 'qc') || strcmp(specifics.constraint, 'dantzig') && ~isfield(specifics, 'epsilon')
        specifics.epsilon = 5e-3;
    end
    
    % 'qc' constraint requires lbtol and mu arguments
    if strcmp(specifics.constraint, 'qc')
        if ~isfield(specifics, 'lbtol')
            specifics.lbtol = 1e-3;
        end

        if ~isfield(specifics, 'mu')
            specifics.mu = 10;
        end
    else
    % other constraints require pdtol and pdmaxiter arguments
        if ~isfield(specifics, 'pdtol')
            specifics.pdtol = 1e-3;
        end

        if ~isfield(specifics, 'pdmaxiter')
            specifics.pdmaxiter = 50;
        end
    end
    
    % apply normalization to image
    if specifics.normalization
        x_norm = norm(x(:));
        x = x / x_norm;
        x_mean = mean(x(:));
        x = x - x_mean;
        y = A(x(:));
    end
    
    x0 = At(y);
    time0 = clock;

    if strcmp(specifics.constraint, 'eq')
        x_hat = l1eq_pd(x0, A, At, y, specifics.pdtol, specifics.pdmaxiter, specifics.cgtol, specifics.cgmaxiter);
    elseif strcmp(specifics.constraint, 'qc')
        x_hat = l1qc_logbarrier(x0, A, At, y, specifics.epsilon, specifics.lbtol, specifics.mu, specifics.cgtol, specifics.cgmaxiter);
    elseif strcmp(specifics.constraint, 'dantzig')
        x_hat = l1dantzig_pd(x0, A, At, y, specifics.epsilon, specifics.pdtol, specifics.pdmaxiter, specifics.cgtol, specifics.cgmaxiter);
    elseif strcmp(specifics.constraint, 'decode')
        x_hat = l1decode_pd(x0, A, At, y, specifics.pdtol, specifics.pdmaxiter, specifics.cgtol, specifics.cgmaxiter);
    end
    
    fprintf('Total elapsed time = %f secs\n\n', etime(clock,time0));
    
    % invert normalization
    if specifics.normalization
        x_hat = x_hat + x_mean;
        x_hat = x_hat * x_norm;
    end
end
