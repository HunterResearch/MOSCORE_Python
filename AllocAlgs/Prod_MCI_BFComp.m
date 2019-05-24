function [minRate] = Prod_MCI_BFComp(alphas,EstObj,Kappa,num_par,z)

tol = 1e-12;
opts = optimset('Algorithm','interior-point-convex','Display','off','TolX',1e-12,'TolFun',1e-12);

numsys = numel(EstObj); %get the number of total systems
numkap = size(Kappa,1);
numnpar = numsys - num_par; %calc the number of non-Paretos

%pre-compute for speed
ParZeros = zeros(num_par,1);

% try for speed increase
cl=struct2cell(EstObj);

minRate=Inf;
for j=1:numnpar 
   
    % data for dominated system
    j_ind = num_par + j;
    alphJ = alphas(j_ind); 
    objValJ = cl{2,j_ind};
    covJ = cl{3,j_ind};
    
    for l=1:numkap

        kap = Kappa(l,:); %objective indices for kappa vector

        %determine which objectives are 'playing' 
        PHobjs=ParZeros;
        PHvars=ParZeros;
        PHalphs=ParZeros;
        objidx=[];
        for p=1:num_par
            obj=kap(p);
            Pobjs = cl{2,p}; % Pareto objectives
            Pcovs = cl{3,p}; % Pareto covariances
            PHobjs(p) = Pobjs(obj);
            PHvars(p) = Pcovs(obj,obj);
            if alphas(p) <= tol
                PHalphs(p)=0;
            else
                PHalphs(p) = alphas(p);
            end
           objidx=[objidx, obj];
        end
        objidx=unique(objidx);

        % reduce non-Pareto info to only objectives that are 'playing'
        % 05/22 - NO, keep full j objectives and matrix, anything that
        % doesn't match an objective in a kappa vector will be
        % unconstrained
        objVal_j = objValJ;
        cov_j = covJ;
%         objVal_j = objValJ(objidx);
%         cov_j = covJ(objidx,objidx);
        qpnumobj=size(objVal_j,1);
        invcov_j = cov_j\eye(qpnumobj);
        aI = alphJ;
        
        H = aI*invcov_j;
        f = -1*aI*invcov_j*objVal_j;

        A = [];
        b = zeros(num_par,1);

        for p=1:num_par
            H = blkdiag(H, PHalphs(p)*(PHvars(p))^(-1));
            f = [f; -1*PHalphs(p)*(PHvars(p))^(-1)*PHobjs(p)];
            Avec=zeros(qpnumobj+num_par,1)';
            obj=kap(p);
%             oidx=find(objidx==obj);
            Avec(obj)=1;
            Avec(qpnumobj+p)=-1;
            A = [A; Avec];                
        end 

        H=(H+H')/2; % safeguard against non-symmetry due to floating points

        [x_star,~] = quadprog(H,f,A,b,[],[],[],[],[],opts);

%         if isequal(kap,[1 1 1 1 1]) && j==1
%             teststop=1;
%         end
        
        %compute rate using x_star
        Rate = 0.5*aI*transpose(objVal_j-x_star(1:qpnumobj))*invcov_j*(objVal_j-x_star(1:qpnumobj));
        for o=1:num_par
            Rate = Rate + 0.5*PHalphs(o)*transpose(x_star(qpnumobj+o)-PHobjs(o))*(PHvars(o))^(-1)*(x_star(qpnumobj+o)-PHobjs(o));
        end 
        Rate = z - Rate;
        
%         pRate=-Rate;
%         kap1=kap(1);
%         kap2=kap(2);
%         fprintf('Kappa [%d %d]: %5.10f \n',kap1,kap2,pRate);
        
%         if j==2 
%             if kap == [2 2 1 1 1]
%                 fprintf('RATE at 22111: %3.20f \n',-Rate);
%             end
%         end
        
        if -Rate<=minRate
            minRate=-Rate;
%             fprintf('New Min Rate: %3.20f, at non-Pareto %d and kappa vector \n',minRate,j);
%             minRate
%             kap
%             objValJ
        end 
    end
end




