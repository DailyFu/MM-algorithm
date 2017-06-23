% Quantile regression based on MM algorithm.
clear;
n = 500;
p = 30;
X = mvnrnd(zeros(1,p),toeplitz(0.5.^(0:p-1)),n);
epsi = randn(n, 1);
beta0 = ones(p,1);
Y = X*beta0 + epsi;

q = 0.3; % median regression.
[beta, history] = MM_Quantile_reg(Y, X, q)
[beta beta0]