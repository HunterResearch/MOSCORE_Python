function [iseedV,testprob] = ProblemStructRandCov(objvals,maxvar,iseedV) 

%This function takes in the objective values (perhaps randomly generated) 
% and the maximum variance allowed. It outputs the test problem in the 
% correct structure format, with covariance matrices and inverse covariance matrix.
% NOTE ASSUMPTIONS BELOW.

%INPUTS: objvals is a numsys by numobj matrix containing objective function
%       values of all the systems. maxvar is a scalar containing the desired 
%       maximum variance for each objective. iseedV is the 6-valued random
%       seed used in Eric's comparison algorithm.
%OUTPUTS: testprob is a structure array containing all the information
%       required for the test problem.

% February 20, 2018: Generalized a previous version to randomly generate
% covariances for each system/objectives. (commented out as of 3/17/18)

% March 17, 2018: Changed how random covariance matrix is formed, using suggestion 
% from: https://math.stackexchange.com/questions/357980/how-to-generate-random-symmetric-positive-definite-matrices-using-matlab

%Find out how many systems and objectives there are.
[numsys,numobj] = size(objvals);


covCell = cell(numsys,2); % Assign the covariance matrices to a cell

rng(iseedV(1));
for i=1:numsys
    [iseedV,u] = mrg32k3a(iseedV);
    var = 1 + u*(maxvar-1);
    B=rand(numobj,numobj);
    C=rand(numobj,numobj);
    D=B-C;
    c=0.5*(D+D');
    for j=1:numobj
        c(j,j)=0;
    end
    cov=c + var*eye(numobj);

    % CHECK IF allequal IS PSD, and if not, get something close
    [~,p]=chol(cov);
    if p>0
        fprintf('WARNING in ProblemStruct: Cov Matrix is not PD. Using nearest PSD Matrix.');
        cov = nearestSPD(cov);
    end
    covCell{i,1}=cov;
    covCell{i,2}=inv(cov);
end

% for i=1:numsys
%     [iseedV,u] = mrg32k3a(iseedV);
%     var = 1 + u*(maxvar-1);
%     cov = eye(numobj)*var;
%     for a=1:numobj
%         for b=(a+1):numobj
%             [iseedV,u2] = mrg32k3a(iseedV);
%             corr = -1 + 2*u2;
%             cov(a,b)=corr*var;
%             cov(b,a)=corr*var;
%         end
%     end
%      
%     % CHECK IF allequal IS PSD, and if not, get something close
%     [~,p]=chol(cov);
%     if p>0
%         fprintf('WARNING in ProblemStruct: Cov Matrix is not PD. Using nearest PSD Matrix.');
%         cov = nearestSPD(cov);
%     end
%     covCell{i,1}=cov;
%     covCell{i,2}=inv(cov);
% end

% STORE INFORMATION IN A STRUCTURED ARRAY 
objCell = num2cell(objvals',1)'; %transpose the mean values; make it a cell; transpose it back

testprob = struct('num',[],'obj',objCell,'cov',covCell(:,1),'invcov',covCell(:,2));

% LABEL THE SYSTEMS 
for i=1:numsys
    testprob(i).num = i;
end


