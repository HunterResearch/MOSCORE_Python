function [C] = Prod_Allocation_iS2(EstObj,settingsSeq)
%% ALLOCATION DETERMINATION

numsys = numel(EstObj); %get the number of systems
numobj = numel(EstObj(1).obj);

%calculate observed Pareto set
[EstObj] = CalcParetoEA(EstObj); %THIS CODE ADDS THE pareto AND obj1 FIELDS TO EstObj

% find indices of paretos and non-paretos, find phantoms
paretos=[];
% parlab=[];
num_par = 0;
for i=1:numsys
    if EstObj(i).paretonum<Inf
        paretos=[paretos; EstObj(i).obj'];
%         parlab = [parlab; i];
        num_par = num_par + 1;
    end
end
% tic;
[phantoms]=FindPhants(paretos);
phantoms=sortrows(phantoms);
num_phants=size(phantoms,1);
% phant_time=toc;
% fprintf('Time to find %d phantoms: %5.3f \n',num_phants,phant_time);

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

% stime=tic;
% Determine closest non-Paretos to phantoms (SCORE calculation)
% [Jstar,Lambdas] = Prod_SCORE(Phantoms,EstObj,numsys,num_par);
[Jstar,Lambdas] = Prod_SCORE_iS(Phantoms,EstObj,numsys,num_par);

% Determine closest Paretos to Paretos (MCE_SCORE calculation)
% [Mstar] = Prod_SCORE_MCE(EstObj,num_par);
[Mstar] = Prod_SCORE_MCE_iS(EstObj,num_par);

% scortime=toc(stime);
% fprintf('Time to compute Scores: %6.4f \n',scortime);

% Determine Rate Constraints (with respect to alphas)
Constraints = @(alphas)Prod_Rates_iS(alphas,EstObj,Phantoms,num_par,Mstar,Jstar,Lambdas);


% Solve for alphas
    % start at equal allocation and aux. variable z =0 
    a0 = [(1/(2*num_par))*ones(num_par,1);0.5;1e-50];
    
%     lower_bound = zeros(num_par+2,1);
    lower_bound = ones(num_par+2,1)*1e-50;
    upper_bound = Inf(num_par+2,1);
    
    alphAeq = [ones(1,num_par+1),0];
    alphbeq = 1.0;
    
    
    % objective value for problem Q
    scaling = 1.0;
    calcobjOuter = @(alphas) calcobjOuterScaled(alphas,scaling,num_par);

% settings for FMINCON
settingsSeq = struct();
settingsSeq.fmincon.TolCon=1e-12;
settingsSeq.fmincon.TolX = 1e-08;
settingsSeq.fmincon.TolFun = 1e-14;
% settingsSeq.fmincon.MaxFunEvals = 500;
settingsSeq.fmincon.MaxFunEvals = 100;
settingsSeq.fmincon.Display = 'off';
settingsSeq.fmincon.Algorithm = 'sqp';
settingsSeq.fmincon.MaxIter = 5000;

optionsOUTER = optimset('Algorithm','sqp', ...
'GradObj','on','GradConstr','on','FinDiffType','central',...
'MaxIter',settingsSeq.fmincon.MaxIter,'MaxFunEvals',settingsSeq.fmincon.MaxFunEvals, 'display',settingsSeq.fmincon.Display,...
'TolCon',settingsSeq.fmincon.TolCon,'TolFun',settingsSeq.fmincon.TolFun,'TolX',settingsSeq.fmincon.TolX,'DerivativeCheck','off');

% tic;


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
    'GradObj','on','GradConstr','on','FinDiffType','central', ...
    'MaxIter',100,'MaxFunEvals',200, 'display',settingsSeq.fmincon.Display,...
    'TolCon',settingsSeq.fmincon.TolCon,'TolFun',settingsSeq.fmincon.TolFun,'TolX',TolX,'DerivativeCheck','off');
    [a1,fval,exitcode,output,lam] = fmincon(calcobjOuter,a0,[],[],alphAeq,alphbeq,lower_bound,upper_bound,Constraints,optionsOUTER);
    exp=floor(log10(abs(currfval))); 
    if abs(fval-currfval)<(10^(exp))/10000 || output.stepsize<1e-18
       stopind=1;
    end
    if output.iterations<100 || output.funcCount<250
       TolX=max(TolX/100,1e-18);
    end
    currfval=fval;
    a0=a1;
end

alloc = zeros(numsys,1);
for i=1:num_par
    alloc(i) = a1(i);
end

num_dom=numsys-num_par;
for j_ind=1:num_dom
    j = num_par+j_ind;
    alloc(j)= Lambdas(j_ind)*a1(num_par+1);
end
    
% alloc(abs(alloc) < 1e-10) = 0;
alloc(alloc < 0) = 0;
    
C = [EstObj(:).num ; alloc'];
C = sortrows(C');

% ftime=toc;
% fprintf('Time to compute allocation: %10.4f \n',ftime);
end    

function [objvalOuter,objGrad] = calcobjOuterScaled(alphas,c,num_par)
        objvalOuter = -c*alphas(num_par+2); 
        objGrad = [zeros(num_par+1,1);-c];
end