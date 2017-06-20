% Case I: p is small and we can easily inverse p*p matrix.
clear;
n = 500;
p = 30;
X = randn(n, p);
epsi = randn(n, 1);
beta0 = zeros(p,1);
beta0(1:10) = 2;
Y = X*beta0 + epsi;

% % MM algorithm
% lambda = 1;
% % initialize
% beta = zeros(p,1)+ 1;
% k = 1;
% db = Inf;
% betatmp = beta;
% beps = 1e-10;
% XtX = 2*(X'*X);
% C = 2*sum(X.* repmat(Y,1,p))';
% while db > beps
%    betatmp = beta;
%    beta = (XtX + diag(lambda ./ abs(betatmp)))\ C;
%    db = norm(beta-betatmp, 2);
%    fprintf('current iteration is %d \n', k);
%    k = k+1;
% end
lambda = 1;
[beta, history] = MM_Lasso(Y, X, lambda, [], 400, 1, 1e-5)
[beta(1:20) beta0(1:20)]

% Case 2: p is large, we use a matrix inverse lemma to inverse p*p matrix.
clear;
n = 500;
p = 3000;
X = randn(n, p);
epsi = randn(n, 1);
beta0 = zeros(p,1);
beta0(1:10) = 2;
Y = X*beta0 + epsi;
lambda = 1;
beta = MM_Lasso(Y, X, lambda, [], 400, 1, 1e-5)
beta = MM_Lasso(Y, X, lambda, [], 400, 0, 1e-5)

[beta(1:20) beta0(1:20)]