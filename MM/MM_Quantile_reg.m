function [beta, history] = MM_Quantile_reg(Y, X, q, rho, maxiter, beps, veps)
%% This MATLB function is used to solve Quantile regression by MM algorithm.
% Syntax: [beta, history] = MM_Lasso(Y, X, rho, maxiter, beps, veps)
% INPUT ARGUMENTS:
% Y: n*1 matrix, response vector, n is observations(sample size).
% X: n*p matrix, covariates matrix, p is dimension of covariates.
% q: a scalar, specific quantile, default as 0.5, median.
% rho: relax parameter in the algorithm to enable the algorithm more
% stable, default as 1.
% maxiter: maximum iterative times in algorithm. optional argument with
% default 1000.
% beps: a scalar, tolerance sup norm error of beta. ooptional argument with
% default 1e-5.
% veps: a scalar, tolerance of error of objective value. ooptional argument with
% default 1e-8. 
% Author : Liu Wei. Email: weidliu321@163.com
% Institute: Center of Statistical Research and School of Statistics,
% Southwestern University of Finance and Economics, Chengdu,Sichuan, China
% Date: 2017/06/23
if(~exist('q', 'var') || isempty(q))
    q = 0.5;
end
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
    veps = 1e-6;
end
%
[n, p] = size(X);
YX = repmat(Y,1,p) .* X; % cashe
C2 = sum((1-2*q)*X, 1);
C1tmp = YX;
% initialize
beta = ones(p,1)-0.5;
objv = Inf;
k = 1;
db = Inf;
objdif = Inf;
while db >= beps && objdif>=veps && k<=maxiter
    betatmp = beta;
    H_total = zeros(p, p,n);
    denom = abs(Y-X*beta);
    Denom = (repmat(denom, 1, p)+ 1e-5)';  % ensure algorithm's stability.
    for j =1:p
        H_total(j,:,:) = (repmat(X(:,j), 1, p).* X)' ./ Denom;
    end
    H = sum(H_total,3);
    C1 = sum(C1tmp ./ Denom');
    beta = H \ (C1+C2)';
    beta = rho*beta + (1- rho)*betatmp; % relaxation
     % output
    db = max(abs(beta-betatmp));
    history.err_sup = db; % sup norm error of |beta0- betatmp|
    history.objvalue = objfun(Y, X, beta, q); % objective value
    objdif = abs(objv - history.objvalue); % difference of objective values
    history.objdif = objdif;
    objv = history.objvalue;
    history.iter = k; % iteration times
    fprintf('Current Iteration is %d \n', k);
    k = k + 1;
end

function objvalue = objfun(Y, X, beta, q)
objvalue = sum(veefun(q,Y- X*beta));

function quant = veefun(q, theta) % quantile regression loss function 
quant = q*theta.*(theta>=0) - (1-q)*theta.*(theta<0);
