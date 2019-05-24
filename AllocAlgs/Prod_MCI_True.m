function [MCIRates,MCI_grad_indices] = Prod_MCI_True(alphas,EstObj,Phantoms,num_par,cnt,numobj,z,z_ind)

% tol = 1e-9;
tol = 1e-12;

numphants = size(Phantoms,1); %get the number of phantoms
numsys = numel(EstObj); %get the number of total systems
numnpar = numsys - num_par; %calc the number of non-Paretos
numMCI = numnpar*numphants; %in Problem Q, the number of MCI is each combo of non-pareto and phantom

MCIRates = zeros(numMCI,1);
MCI_grad_indices =zeros(numMCI*(numobj+2),3); % over-allocate for speed, then reduce

%pre-compute for speed
GradMat = zeros(numobj+2,3);
ObjZeros = zeros(numobj,1);
objlist=1:numobj;

% try for speed increase
cl=struct2cell(EstObj);

% cycle through each non-Pareto/phantom combo 
i=0;
for j=1:numnpar 
   
    % data for dominated system
    j_ind = num_par + j;
    alph_j = alphas(j_ind); 
    objVal_j = cl{2,j_ind};
    cov_j = cl{3,j_ind};
    
    for l=1:numphants
        
        i=i+1; % update index of current MCI constraint

        % data for phantom
        PHind = Phantoms(l,:); %Pareto indices for phantom

        MFGrads=GradMat;

        if abs(alph_j) <= tol
%         if abs(alph_j) <= 0
            RateMF = 0;
            MFGrads(1,:)=[j_ind i+cnt 0];
            for o=1:numobj
                MFGrads(o+1,:)=[PHind(o) i+cnt 0];
            end 
            MFGrads(numobj+2,:) = [z_ind  i+cnt 1.0] ;
        else
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
                    if alphas(Psys) <= tol
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

            if alphcnt==idxcnt
                RateMF = 0;
                GradI = 0;
                mgrads=zeros(size(phidx,2),1);
                for o=1:size(phidx,2)
                    mgrads(o)=(0.5)*(objJ(o)-phobjs(o))^2/phvars(o);
                end 
            else
                sz=size(phidx,2);
                if sz==1
                    [RateMF,GradI,mgrads]=MCI1d(alph_j,alphs,objJ,covJ,phobjs,phvars);
                elseif sz==2
                    [RateMF,GradI,mgrads]=MCI2d(alph_j,alphs,objJ,covJ,phobjs,phvars);
                elseif sz==3
                    [RateMF,GradI,mgrads]=MCI3d(alph_j,alphs,objJ,covJ,phobjs,phvars);
                else
                    [RateMF,mgrads] = qpMCI(alph_j,objJ,covJ,alphs,phobjs,phvars);
                    GradI=mgrads(1);
                    mgrads(1)=[];
                end
            end
            
            mfgrads=ObjZeros;
            mfgrads(phidx)=mgrads;
          
            MFGrads(1,:)=[j_ind i+cnt -1*GradI];
            for o=1:numobj
                grad = mfgrads(o);
                if grad<tol
                    grad=0;
                end
                MFGrads(o+1,:)=[PHind(o) i+cnt -grad];
            end 
            MFGrads(numobj+2,:) = [z_ind  i+cnt 1.0] ;
        end

%         pRate=RateMF;
%         kap1=PHobjs(1);
%         kap2=PHobjs(2);
%         kap3=PHobjs(3);
%         fprintf('Phantom [%d %d %d]: %5.10f \n',kap1,kap2,kap3,pRate);
        
        MCIRates(i) = z - RateMF;
        MCI_grad_indices(numobj*(i-1)+2*i-1:numobj*i+2*i,:)=MFGrads;
    end
end
MCI_grad_indices(MCI_grad_indices(:,1)==Inf,:)=[];



