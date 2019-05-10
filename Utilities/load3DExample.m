function [ objVals, covs, settingsSeq ] = load3DExample( id ,corr )
%LOADEXAMPLE Summary of this function goes here
%   Detailed explanation goes here

    load('sphereProblems.mat');
    
    if id == 1
        objVals = objVals1;
    elseif id == 2 
        objVals = objVals2;
    end
    if strcmpi(corr,'neg')
        covs = covsNeg;
    elseif strcmpi(corr,'pos')
        covs = covsPos;
    elseif strcmpi(corr,'ind')
        covs = covsInd;
    end
    
    % settings for FMINCON
    settingsSeq = struct();
    settingsSeq.fmincon.TolCon=1e-12;
    settingsSeq.fmincon.TolX = 1e-12;
    settingsSeq.fmincon.TolFun = 1e-12;

    settingsSeq.fmincon.Display = 'off';
    settingsSeq.fmincon.Algorithm = 'sqp';
    settingsSeq.fmincon.MaxIter=5000;
    settingsSeq.fmincon.MaxFunEvals = 30000;
    settingsSeq.fmincon.MaxFunEval = 30000;

end

