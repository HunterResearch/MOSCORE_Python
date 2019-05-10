function [rate,Erate,Irate] = CalcMCRate(allocation,EstObj)
%% ALLOCATION DETERMINATION

numsys = numel(EstObj); %get the number of systems
numobj = numel(EstObj(1).obj);

%calculate observed Pareto set
[EstObj] = CalcParetoEA(EstObj); %THIS CODE ADDS THE pareto AND obj1 FIELDS TO EstObj

% find indices of paretos and non-paretos, find phantoms
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
[phantoms]=FindPhants(paretos);
phantoms=sortrows(phantoms);
num_phants=size(phantoms,1);

% FIND PARETO INDICES OF PHANTOMS 
Phantoms=ones(num_phants,numobj)*Inf;
for i=1:num_phants
    for j=1:numobj
       for k=1:num_par
           if paretos(k,j)==phantoms(i,j)
               Phantoms(i,j) = k;
           end
       end
    end
end

alphas = [alphas; 0];
% Determine Rate Constraints (with respect to alphas)
[rates,Erate,Irate] = Prod_Rates_True_Comp(alphas,EstObj,Phantoms,num_par);
rate= min(-rates);
