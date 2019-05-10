function [MCIRates,MCI_grad_indices] = Prod_MCI_BF(alphas,EstObj,Kappa,num_par,z,z_ind,cnt)

tol = 1e-12;
opts = optimset('Algorithm','interior-point-convex','Display','off','TolX',1e-12,'TolFun',1e-12);

numsys = numel(EstObj); %get the number of total systems
numkap = size(Kappa,1); % get the number of kappas
numnpar = numsys - num_par; %calc the number of non-Paretos
numMCI = numnpar*numkap; %in Problem Q, the number of MCI is each combo of non-pareto and kappa

MCIRates = zeros(numMCI,1);
MCI_grad_indices =zeros(numMCI*(num_par+2),3); % pre-allocate for speed

%pre-compute for speed
GradMat = zeros(num_par+2,3);
ParZeros = zeros(num_par,1);

% try for speed increase
cl=struct2cell(EstObj);

i=0;
for j=1:numnpar 
   
    % data for dominated system
    j_ind = num_par + j;
    alphJ = alphas(j_ind); 
    objValJ = cl{2,j_ind};
    covJ = cl{3,j_ind};
    
    for l=1:numkap

        i=i+1; % update index of current MCI constraint
        
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
        objVal_j = objValJ(objidx);
        cov_j = covJ(objidx,objidx);
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
            oidx=find(objidx==obj);
            Avec(oidx)=1; %#ok<FNDSB>
            Avec(qpnumobj+p)=-1;
            A = [A; Avec];                
        end 

        H=(H+H')/2; % safeguard against non-symmetry due to floating points

        [x_star,~] = quadprog(H,f,A,b,[],[],[],[],[],opts);

        %compute rate and gradients using x_star
        Grads=GradMat;
        Rate = 0.5*aI*transpose(objVal_j-x_star(1:qpnumobj))*invcov_j*(objVal_j-x_star(1:qpnumobj));
        gradj = 0.5*transpose(x_star(1:qpnumobj)-objVal_j)*invcov_j*(x_star(1:qpnumobj)-objVal_j);
        Grads(1,:)=[j_ind i+cnt -1*gradj];
        for o=1:num_par
            Rate = Rate + 0.5*PHalphs(o)*transpose(x_star(qpnumobj+o)-PHobjs(o))*(PHvars(o))^(-1)*(x_star(qpnumobj+o)-PHobjs(o));
            gradj = 0.5*transpose(x_star(qpnumobj+o)-PHobjs(o))*(PHvars(o))^(-1)*(x_star(qpnumobj+o)-PHobjs(o));
            Grads(1+o,:)=[o i+cnt -1*gradj];
        end 
        Grads(num_par+2,:)=[z_ind  i+cnt 1.0] ;

        MCIRates(i) = z - Rate;
        MCI_grad_indices(num_par*(i-1)+2*i-1:num_par*i+2*i,:)=Grads;
        
    end
end




