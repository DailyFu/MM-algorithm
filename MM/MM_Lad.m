function [beta, history] = MM_Lad(Y, X, rho, maxiter, beps, veps)
%% This MATLB function is used to solve Least absolute distance regression by MM algorithm.
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
    beps = 1.5e-5;
end
if(~exist('veps', 'var'))
    veps = 2.5e-6;
end
%
[n, p] = size(X);
beta = zeros(p,1)+1e-3;
k = 1;
db = Inf;
objv = Inf;
objdif = Inf;

while db > beps && k < maxiter && objdif > veps
    betatmp = beta;
    H_total = zeros(p, p,n);
    denom = abs(Y-X*beta);
    Denom = (repmat(denom, 1, p)+ 1e-5)';  % ensure algorithm's stability.
    c_num = repmat(Y, 1, p).* X; 
    for j =1:p
        H_total(j,:,:) = (repmat(X(:,j), 1, p).* X)' ./ Denom;
    end
    H = sum(H_total,3);
    C = sum(c_num' ./ Denom, 2); % beter result can be obtained with eps stability.
    beta = H \ C;
    beta = rho*beta + (1-rho)*betatmp; % relaxation
    
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

function objvalue = objfun(Y, X, beta)
objvalue = sum(abs(Y- X*beta));

