function [rate] = CalcBFRate(allocation,EstObj)
%% ALLOCATION DETERMINATION

numsys = numel(EstObj); %get the number of systems
numobj = numel(EstObj(1).obj);

%calculate observed Pareto set
[EstObj] = CalcParetoEA(EstObj); %THIS CODE ADDS THE pareto AND obj1 FIELDS TO EstObj

% find indices of paretos and non-paretos
paretos=[];
alphas=zeros(numsys,1);
num_par = 0;
for i=1:numsys
    if EstObj(i).paretonum<Inf
        paretos=[paretos; EstObj(i).obj'];
        num_par = num_par + 1;
    end
    alphas(i)=allocation(EstObj(i).num);
end

%Compute kappas
v=1:numobj;
k=num_par;
Kappa = PermsRep(v,k);

alphas = [alphas; 0];
% Determine Rate Constraints (with respect to alphas)
[rate] = Prod_Rates_BF_Comp(alphas,EstObj,Kappa,num_par);

