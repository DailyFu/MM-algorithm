function [beta, history] = MM_Logistic(Y, X, rho, maxiter, beps, veps)
%% This MATLB function is used to solve Logistic regression by MM algorithm.
% Syntax: [beta, history] = MM_Lasso(Y, X, rho, maxiter, beps, veps)
% INPUT ARGUMENTS:
% Y: n*1 matrix, response vector, n is observations(sample size).
% X: n*p matrix, covariates matrix, p is dimension of covariates.
% rho: relax parameter in the algorithm to enable the algorithm more
% stable.
% maxiter: maximum iterative times in algorithm. optional argument with
% default 1000.
% beps: a scalar, tolerance sup norm error of beta. ooptional argument with
% default 1e-5.
% veps: a scalar, tolerance of error of objective value. ooptional argument with
% default 1e-8. 
%
if(~exist('rho', 'var') || isempty(rho))
    rho=1;
end
if(~exist('maxiter', 'var'))
    maxiter = 1000;
end
if(~exist('beps', 'var'))
    beps = 1e-9;
end
if(~exist('veps', 'var'))
    veps = 1e-12;
end
%
[~, p] = size(X);
beta = ones(p,1);
db = Inf;
objdif = Inf;
objv = Inf;
k = 1;
while db > beps && k <= maxiter && objdif > veps
    betatmp = beta;
    xb = X*beta;
    H = 1./ (1+exp(-xb));
    M = (H.^2) ./ exp(xb);
    S = 2*X'*(repmat(M, 1, p).*X);
    C = sum((repmat(Y,1,p) + repmat(2*M.*xb,1,p) - repmat(H,1,p) ).*X, 1);
    beta = S\ C';
    beta = rho*beta + (1-rho)*betatmp;
    
    db = max(abs(beta-betatmp));
    history.errl2 = db; % sup norm error of |beta0- betatmp|
    history.objvalue = objfun(Y, X, beta); % objective value
    objdif = abs(objv - history.objvalue); % difference of objective values
    history.objdif = objdif;
    objv = history.objvalue;
    history.iter = k; % iteration times
    fprintf('current iteration is %d \n', k);
    k = k+1;
end

function objvalue = objfun(Y, X, beta) % negative loglikelihood
xb = X*beta;
objvalue = sum(log(1+exp(xb))) - sum(Y.*xb);

