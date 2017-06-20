function [beta, history] = MM_Lasso(Y, X, lambda, rho, maxiter, plarge, beps, veps)
%% This MATLB function is used to solve Lasso by MM algorithm.
% Syntax: [beta, history] = MM_Lasso(Y, X, lambda, rho, maxiter, plarge, beps, veps)
% INPUT ARGUMENTS:
% Y: n*1 matrix, response vector, n is observations(sample size).
% X: n*p matrix, covariates matrix, p is dimension of covariates.
% lambda: a scalar greater than 0, tuning parameter in lasso.
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
if(~exist('plarge', 'var'))
    plarge = 0;
end
if(~exist('beps', 'var'))
    beps = 1e-5;
end
if(~exist('veps', 'var'))
    veps = 1e-8;
end
%
[n, p] = size(X);
beta = zeros(p,1)+1e-3;
k = 1;
eps1 = 1e-6;
db = Inf;
objv = Inf;
objdif = Inf;
XtX = 2*(X'*X);
C = 2*sum(X.* repmat(Y,1,p))';
while db > beps && k <= maxiter && objdif > veps
    betatmp = beta;
    if(plarge)
        Lambda_inv = diag(beta/lambda);
        invM = invsum(X, Lambda_inv, n);
        beta = invM * C;
    else
        beta = (XtX + diag(lambda ./ abs(betatmp+eps1)))\ C; % eps1 ensure non NaN.
    end
    beta = rho*beta + (1-rho)*betatmp; % relaxation
    
    db = max(abs(beta-betatmp));
    history.errl2 = db; % sup norm error of |beta0- betatmp|
    history.objvalue = objfun(Y, X, beta, lambda); % objective value
    objdif = abs(objv - history.objvalue); % difference of objective values
    history.objdif = objdif;
    objv = history.objvalue;
    history.iter = k; % iteration times
    fprintf('current iteration is %d \n', k);
    k = k+1;
end

function objvalue = objfun(Y, X, beta, lambda)
objvalue = sum((Y- X*beta).^2) + lambda*sum(abs(beta));

function invM = invsum(X, Lambda_inv,n)
tmp = Lambda_inv*X';
invM = Lambda_inv - (tmp)*((X*tmp + 1/2*speye(n,n))\ tmp');


