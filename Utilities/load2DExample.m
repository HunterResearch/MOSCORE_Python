function [ objVals,covs,settingsSeq ] = load2DExample( instance, cov_type )
%LOAD2DEXAMPLE Summary of this function goes here
%   Detailed explanation goes here

    load('testproblems20.mat');
    if instance == 1
        objind = 31;
        objVals = objs{objind};
    elseif instance == 2
        objind = 57;
        objVals = objs{objind};
    end
    num_sys = size(objVals,1);
    if strcmpi(cov_type,'ind')
        c=0;
        covs=repmat([[1 c];[c 1]],1,1,num_sys);
    elseif strcmpi(cov_type,'neg')
        c=-0.8;
        covs=repmat([[1 c];[c 1]],1,1,num_sys);
    elseif strcmpi(cov_type,'pos')
        c= 0.8;
        covs=repmat([[1 c];[c 1]],1,1,num_sys);
    end
    
    
    settingsSeq = struct();
    settingsSeq.fmincon.TolCon=1e-12;
    settingsSeq.fmincon.TolX = 1e-12;
    settingsSeq.fmincon.TolFun = 1e-12;
    
    settingsSeq.fmincon.Display = 'off';
    settingsSeq.fmincon.Algorithm = 'sqp';
    settingsSeq.fmincon.MaxIter=5000;
    settingsSeq.fmincon.MaxFunEvals = 30000;
    settingsSeq.fmincon.MaxFunEval = 30000;
    
    settingsSeq.sequential.n_pilot = 5;
    settingsSeq.sequential.n_between=1;
    settingsSeq.sequential.n_total= 2500;
    settingsSeq.sequential.min_sample = 1e-8 * ones(num_sys,1);
    settingsSeq.sequential.seed = 12345;
end

