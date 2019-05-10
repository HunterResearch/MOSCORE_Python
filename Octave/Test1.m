% test Octave

%add Eric's paths
pardir = fileparts(pwd);
addpath(fullfile(pardir,'RateCalcs'));
addpath(genpath(fullfile(pardir,'Utilities')));
%addpath('/Users/applegae/Dropbox/Research/CodeEric');
%addpath(genpath('/Users/applegae/Dropbox/Research/CodeEric/Utilities'));
%addpath('/Users/applegae/Dropbox/Research/CodeEric/Mathematica');
%addpath('/Users/applegae/Dropbox/Research/CodeEric/Prod_Star');

[objVals,covs,settingsSeq] = loadMOCBA('ind');
%[objVals,covs,settingsSeq] = load2DExample(2, 'neg')
numsys=size(objVals,1);

% CREATE EstObj for Eric's code
EstObj = struct('num',[],'obj',[],'cov',[],'Nsofar',[],'asofar',[],'curr_alloc',[]);

% CONVERT Guy's inputs to Eric's form
for sys = 1:numsys
   EstObj(sys).obj = objVals(sys,:)';
   EstObj(sys).cov = covs(:,:,sys);
   EstObj(sys).Nsofar = 1;
   EstObj(sys).asofar = 1/numsys;
   EstObj(sys).num = sys;
end

fprintf('Starting Allocation...\n');
[C] = Prod_Allocation_iS2(EstObj,settingsSeq);
