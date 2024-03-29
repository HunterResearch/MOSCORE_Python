function [C] = Prod_Allocation_TrueBF(EstObj,settingsSeq)
%% ALLOCATION DETERMINATION

numsys = numel(EstObj); %get the number of systems
numobj = numel(EstObj(1).obj);

%calculate observed Pareto set
[EstObj] = CalcParetoEA(EstObj); %THIS CODE ADDS THE pareto AND obj1 FIELDS TO EstObj

% find indices of paretos and non-paretos
paretos=[];
num_par = 0;
for i=1:numsys
    if EstObj(i).paretonum<Inf
        paretos=[paretos; EstObj(i).obj'];
        num_par = num_par + 1;
    end
end

%Compute kappas
v=1:numobj;
k=num_par;
Kappa = PermsRep(v,k);

% Determine Rate Constraints (with respect to alphas)
Constraints = @(alphas)Prod_Rates_BF(alphas,EstObj,Kappa,num_par);


% Solve for alphas
    % start at equal allocation and aux. variable z =0 
    a0 = [(1/numsys)*ones(numsys,1);1e-50];
    
%     lower_bound = zeros(num_par+2,1);
    lower_bound = ones(numsys+1,1)*1e-50;
    upper_bound = Inf(numsys+1,1);
    
    alphAeq = [ones(1,numsys),0];
    alphbeq = 1.0;

% settings for FMINCON
settingsSeq = struct();
settingsSeq.fmincon.TolCon=1e-12;
settingsSeq.fmincon.TolX = 1e-08;
settingsSeq.fmincon.TolFun = 1e-14;
settingsSeq.fmincon.MaxFunEvals = 500;
settingsSeq.fmincon.Display = 'off';
settingsSeq.fmincon.Algorithm = 'sqp';
settingsSeq.fmincon.MaxIter = 5000;

    % objective value for problem Q
    scaling = 1.0;
    calcobjOuter = @(alphas) calcobjOuterScaled(alphas,scaling,numsys);
    
    
   optionsOUTER = optimset('Algorithm','sqp', ...
    'GradObj','on','GradConstr','on','FinDiffType','forward', ...
    'MaxIter',settingsSeq.fmincon.MaxIter,'MaxFunEvals',settingsSeq.fmincon.MaxFunEvals, 'display',settingsSeq.fmincon.Display,...
    'TolCon',settingsSeq.fmincon.TolCon,'TolFun',settingsSeq.fmincon.TolFun,'TolX',settingsSeq.fmincon.TolX,'DerivativeCheck','off');

%     [a1,fval,exitcode,output,~] = fmincon(calcobjOuter,a0,[],[],alphAeq,alphbeq,lower_bound,upper_bound,Constraints,optionsOUTER);

% first pass, find solution when StepTol = 1e-8
    [a1,fval,exitcode,output,lam] = fmincon(calcobjOuter,a0,[],[],alphAeq,alphbeq,lower_bound,upper_bound,Constraints,optionsOUTER);
% find value of rate and re-run with current solution as warm-start
currfval=fval;
a0=a1;
if output.stepsize<1e-18
    stopind=1;
else 
    stopind=0;
end

TolX=settingsSeq.fmincon.TolX/100;
while stopind==0
% restrict FunEvals and MaxIter to run a few at a time
% decrease StepTolerance and see if it stops before MaxFunEvals or MaxIter
optionsOUTER = optimset('Algorithm','sqp', ...
    'GradObj','on','GradConstr','on','FinDiffType','forward', ...
    'MaxIter',100,'MaxFunEvals',250, 'display',settingsSeq.fmincon.Display,...
    'TolCon',settingsSeq.fmincon.TolCon,'TolFun',settingsSeq.fmincon.TolFun,'TolX',TolX,'DerivativeCheck','off');
  [a1,fval,exitcode,output,lam] = fmincon(calcobjOuter,a0,[],[],alphAeq,alphbeq,lower_bound,upper_bound,Constraints,optionsOUTER);
  exp=floor(log10(abs(currfval))); 
  if abs(fval-currfval)<(10^(exp))/10000 || output.stepsize<1e-18
       stopind=1;
   end
   if output.iterations<100 || output.funcCount<250
       TolX=TolX/100;
   end
   currfval=fval;
    a0=a1;
end

    alloc = a1(1:(end-1));
    z = a1(end);
   
% alloc(abs(alloc) < 1e-10) = 0;
alloc(alloc < 0) = 0;
    
    C = [EstObj(:).num ; alloc'];
    C = sortrows(C');
    
end    

function [objvalOuter,objGrad] = calcobjOuterScaled(alphas,c,numsys)
        objvalOuter = -c*alphas(numsys+1); 
        objGrad = [zeros(numsys,1);-c];
end