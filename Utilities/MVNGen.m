function [seed,X] = MVNGen(seed,mus,sigma)
% takes in a seed, a row vector of means, and a sigma matrix
% returns the next seed and a vector of X's from the 
% multivariate normal distribution

%find dimension of input mean vector
dim=size(mus,2);

%perform Cholesky decomposition
C=chol(sigma,'lower');

% generate required random numbers (u's)
for i=1:dim
    [seed,u]=u16807d(seed);
    us(i)=u;
end;

% convert u's to iid standard normal z's
% via Beasley-Springer-Moro code supplied by Dr. Hunter
zs = InvNorm(us);

% transform iid Standard z's to MVN z's
X = C*transpose(zs) + transpose(mus);
end