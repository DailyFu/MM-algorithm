% Case I: p is small and we can easily inverse p*p matrix.
clear;
n = 500;
p = 30;
X = mvnrnd(zeros(1,p),toeplitz(0.5.^(0:p-1)),n);
epsi = randn(n, 1);
beta0 = ones(p,1);
Y = X*beta0 + epsi;

[beta, history] = MM_Lad(Y, X, 0.8, 300)
[beta, beta0]