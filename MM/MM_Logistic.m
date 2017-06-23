function [beta, history] = MM_Logistic(Y, X, rho, maxiter, beps, veps, method)
%% This MATLB function is used to solve Logistic regression by MM algorithm.
% Syntax: [beta, history] = MM_Lasso(Y, X, rho, maxiter, beps, veps, method)
% INPUT ARGUMENTS:
% Y: n*1 matrix, response vector, n is observations(sample size).
% X: n*p matrix, covariates matrix, p is dimension of covariates.
% rho: relax parameter in the algorithm to enable the algorithm more
% stable.
% maxiter: maximum iterative times in algorithm. optional argument with
% default 1000.
% beps: a scalar, tolerance sup norm error of beta. ooptional argument with
% default 1e-11.
% veps: a scalar, tolerance of error of objective value. ooptional argument with
% default 1e-15. 
% method: a string, specific MM algorithm, currently support two methods,
% 'upbound' and 'convexsupport', default as upbound.
% Author : Liu Wei. Email: weidliu321@163.com
% Institute: Center of Statistical Research and School of Statistics,
% Southwestern University of Finance and Economics, Chengdu,Sichuan, China
% Date: 2017/06/18
% Update Date: 2017/06/23

if(~exist('rho', 'var') || isempty(rho))
    rho=1;
end
if(~exist('maxiter', 'var') || isempty(maxiter))
    maxiter = 1000;
end
if(~exist('beps', 'var') || isempty(beps))
    beps = 1e-11;
end
if(~exist('veps', 'var') || isempty(veps))
    veps = 1e-15;
end
if(~exist('method', 'var') || isempty(method))
    method= 'upbound';
end
%
[~, p] = size(X);
Xinf = 4*((X'*X)\X'); %cashe.
beta = ones(p,1);
db = Inf;
objdif = Inf;
objv = Inf;
k = 1;
while db > beps && k <= maxiter && objdif > veps
    betatmp = beta;
    xb = X*betatmp;
    H = 1./ (1+exp(-xb));
    switch method
        case 'upbound'
            beta = betatmp + Xinf*(Y-H);
        case 'convexsupport'
            M = (H.^2) ./ exp(xb);
            S = 2*X'*(repmat(M, 1, p).*X);
            C = sum((repmat(Y,1,p) + repmat(2*M.*xb,1,p) - repmat(H,1,p) ).*X, 1);
            beta = S\ C';
    end
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

