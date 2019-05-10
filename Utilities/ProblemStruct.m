function [testprob] = ProblemStruct(objvals,objvar,objcorr) 

%This function takes in the objective values (perhaps randomly generated) 
% and the desired objective correlation. It outputs the test problem in the 
% correct structure format, with covariance matrices and inverse covariance matrix.
% NOTE ASSUMPTIONS BELOW.

%INPUTS: objvals is a numsys by numobj matrix containing objective function
%       values of all the systems. objcorr is a scalar containing the desired 
%       correlatin between the objectives. 
%OUTPUTS: testprob is a structure array containing all the information
%       required for the test problem.

% April 15, 2016: Right now, this code assumes all the objectives have the same
%correlation. (E.g., if objcorr is .5, then the correlation between
%objectives 1 and 2, 1 and 3, and 2 and 3 are .5. Thus may be somewhat
%unrealistic for high-dimensional problems, so this code should probably be
%generalized at a later date.


%Find out how many systems and objectives there are.
[numsys,numobj] = size(objvals);

% DETERMINE THE COVARIANCE MATRICES: for now, make them equal across all
% systems with variance objvar and equal correlation across all objectives.
allequal = eye(numobj)*objvar+(ones(numobj)-eye(numobj))*objcorr*objvar; % MAKE SURE DIMENSIONS MATCH numobj

% CHECK IF allequal IS PSD, and if not, get something close
[~,p]=chol(allequal);
if p>0
    fprintf('WARNING in ProblemStruct: Cov Matrix is not PD. Using nearest PSD Matrix.');
    allequal = nearestSPD(allequal);
end

covCell = cell(numsys,2); % Assign the covariance matrices to a cell
for i=1:numsys
    covCell{i,1}=allequal;
    covCell{i,2}=inv(allequal);
end

% STORE INFORMATION IN A STRUCTURED ARRAY 
objCell = num2cell(objvals',1)'; %transpose the mean values; make it a cell; transpose it back

testprob = struct('num',[],'obj',objCell,'cov',covCell(:,1),'invcov',covCell(:,2));

% LABEL THE SYSTEMS 
for i=1:numsys
    testprob(i).num = i;
end


