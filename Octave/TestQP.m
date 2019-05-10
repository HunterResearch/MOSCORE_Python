% test Octave

%add Eric's paths
pardir = fileparts(pwd);
addpath(fullfile(pardir,'RateCalcs'));
addpath(genpath(fullfile(pardir,'Utilities')));

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

%% Inputs to test MCE
I=1;
J=3;
aI=EstObj(I).asofar;
aJ=EstObj(J).asofar;
Iobj=EstObj(I).obj;
Jobj=EstObj(J).obj;
Isig=EstObj(I).cov;
Jsig=EstObj(J).cov;
fprintf('Determine MCE Rate...\n');
[curr_rate,GradI,GradJ] = qpMCE(aI,aJ,Iobj,Isig,Jobj,Jsig)

%% Inputs to test SCORE and MCI
PH = 1;
NP = 9;

cl=struct2cell(EstObj);
VobjInf=ones(numobj,1)*Inf;

PHind = Phantoms(PH,:); %Pareto indices for phantom
% data for phantom
PHobjs=VobjInf;
for b=1:numobj
    if PHind(b)<Inf
        Psys=PHind(b); % Pareto system number
        Pobjs = cl{2,Psys}; % Pareto objectives
        PHobjs(b) = Pobjs(b);
    end
end
ObjJ = cl{2,NP};
CovJ = cl{3,NP};

fprintf('Determine SCORE...\n');
[score,binds]=qpSCORE(PHobjs,ObjJ,CovJ)

fprintf('Determine MCI...\n');
ph_ind = 6;
j_ind = 16;

tol = 1e-12;
alphas = [(1/(2*num_par))*ones(num_par,1);0.5;1e-50];
slack_ind = length(alphas)-1;
z_ind = length(alphas);
z = alphas(z_ind); %get current objective value
cnt=100;
ObjZeros = zeros(numobj,1);
objlist=1:numobj;

lambda_j = 1;
alph_j = EstObj(j_ind).asofar;
objVal_j = cl{2,j_ind};
cov_j = cl{3,j_ind};
PHind = Phantoms(ph_ind,:); %Pareto indices for phantom

% data for phantom
PHobjs=ObjZeros;
PHvars=ObjZeros;
PHalphs=ObjZeros;
phidx=objlist;
alphcnt=0;
idxcnt=numobj;
for b=1:numobj
    if PHind(b)<Inf
        Psys=PHind(b); % Pareto system number
        Pobjs = cl{2,Psys}; % Pareto objectives
        Pcovs = cl{3,Psys}; % Pareto covariances
        PHobjs(b) = Pobjs(b);
        PHvars(b) = Pcovs(b,b);
        if alphas(Psys) < tol
            PHalphs(b)=0;
            alphcnt=alphcnt+1;
        else
            PHalphs(b) = alphas(Psys);
        end
    else
        phidx(phidx==b)=[];
        idxcnt=idxcnt-1;
    end
end

% any way to speed up the vector extraction?
objJ = objVal_j(phidx);
covJ = cov_j(phidx,phidx);

phobjs = PHobjs(phidx);
phvars = PHvars(phidx);
alphs = PHalphs(phidx);

[RateMF,mgrads] = qpMCI(alph_j,objJ,covJ,alphs,phobjs,phvars);
GradI=mgrads(1);
mgrads(1)=[];

mfgrads=ObjZeros;
mfgrads(phidx)=mgrads;
MFGrads(1,:)=[slack_ind i+cnt -lambda_j*GradI];
for o=1:numobj
    grad = mfgrads(o);
    if grad<tol
        grad=0;
    end
    MFGrads(o+1,:)=[PHind(o) i+cnt -grad];
end
MFGrads(numobj+2,:) = [z_ind  i+cnt 1.0] ;
RateMF
MFGrads
