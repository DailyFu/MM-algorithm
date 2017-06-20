clear;
rng(0)
n = 100;
p = 5;
X = mvnrnd(zeros(1,p), toeplitz(10*0.5.^(0:p-1)), n);
beta0 = ones(p,1)*2;
P = 1./ (1+exp(-X*beta0));
Y = binornd(1,P,n,1);
% MM algorithm

[beta, history] = MM_Logistic(Y,X)
[beta, beta0]
