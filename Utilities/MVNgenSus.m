function [iseedV,X]=MVNgenSus(iseedV,muvect,sigmat)
%FUNCTION INPUTS:
% muvect is a column vector containing the mean values to generate
% sigmat is a symmetric PSD matrix; if not, code finds the closest PSD
%       matrix and generates from that instead.
% iseed is the seed for mrg32k3a. NOTICE IT IS A VECTOR OF SIX SEEDS.
%FUNCTION OUTPUTS:
% iseed is the next seed
% X is a column vector; realization of MVN (muvect, sigmat)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% PERFORM ERROR CHECKS

% GET THE DIMENSION OF MU VECTOR
dim1 = size(muvect,1);
dim2 = size(muvect,2);
mydim = max(dim1,dim2);
if dim1>1 && dim2>1
    fprintf('ERROR in MVNgen: mu vector is a matrix.\n')
elseif dim1==1 && dim2>1
    muvect = muvect'; %TAKE TRANSPOSE
    fprintf('WARNING in MVNgen: mu vector is a row, took transpose.\n')
end

%See if sigmat is Square. If not, generate an error.

[r,c] = size(sigmat);
if r ~= c
  error('ERROR in MVNgen: Sigma must be a square matrix.')
end

% CHECK IF mySig IS PSD, and if not, get something close
[~,p]=chol(sigmat);
if p>0
    fprintf('WARNING in MVNgen: Sigma is not PD. Generating nearest PSD Matrix for use in MVNgen.');
    sigmatgen = nearestSPD(sigmat);
elseif p==0
    sigmatgen = sigmat;
end

%% GENERATE MVN RANDOM VECTOR

% DO CHOLESKY DECOMPOSITION
C = chol(sigmatgen,'lower');

%INITIALIZE Z VECTOR
Z = zeros(mydim,1);

%GENERATE mydim STANDARD NORMALS
for i=1:mydim
    [iseedV,U]=mrg32k3a(iseedV);
    Z(i) = InvNormBSM(U);
end
    
X = C*Z + muvect;

