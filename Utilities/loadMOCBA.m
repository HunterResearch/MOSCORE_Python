function [ objVals,covs,settingsSeq ] = loadMOCBA( covtype )
%LOADMOCBA Summary of this function goes here
%   Detailed explanation goes here
 

    fid = fopen('general_25.txt');
C = textscan(fid,'%f%f%f%f%f%f','HeaderLines',0);
fclose(fid);
objVals = 1.0*[C{1} C{3} C{5}];

covs = zeros(3,3,25);
for i=1:25
    for j=1:3
        covs(j,j,i) = C{2*j}(i)^2;
    end
end

if strcmpi(covtype,'ind')
elseif strcmpi(covtype,'pos')
    cp = [ [64 0.4*8*8 0.4*8*8]; [ 0.4*8*8 64 0.4*8*8 ]; [0.4*8*8 0.4*8*8 64]];
    covs = repmat(cp,1,1,25);
elseif strcmpi(covtype,'neg')
    cn = [ [64 -0.4*8*8 -0.4*8*8]; [ -0.4*8*8 64 -0.4*8*8 ]; [-0.4*8*8 -0.4*8*8 64]];
    covs = repmat(cn,1,1,25);
end

settingsSeq.fmincon.MaxIter = 5000;
settingsSeq.fmincon.MaxFunEvals = 30000;
settingsSeq.fmincon.Display= 'off';
settingsSeq.fmincon.TolCon = 1e-10;
settingsSeq.fmincon.TolX = 1e-10;
settingsSeq.fmincon.TolFun = 1e-10;
settingsSeq.SCORE.SolverTol = 1e-10;
settingsSeq.sequential.n_pilot = 5;
settingsSeq.sequential.n_between=10;
settingsSeq.sequential.n_total=2000;
settingsSeq.sequential.seed=12345;
settingsSeq.sequential.min_sample = 1e-8;

end

