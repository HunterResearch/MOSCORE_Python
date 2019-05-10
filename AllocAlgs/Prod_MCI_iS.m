function [MCIRates,MCI_grad_indices] = Prod_MCI_iS(alphas,Lambdas,Jstar,EstObj,Phantoms,num_par,cnt,numobj,z,z_ind)

tol = 1e-50;

numMCI = size(Jstar,1);
MCIRates = zeros(numMCI,1);
slack_ind = length(alphas)-1;
MCI_grad_indices =zeros(numMCI*(numobj+2),3); % over-allocate for speed, then reduce

%pre-compute for speed
GradMat = zeros(numobj+2,3);
ObjZeros = zeros(numobj,1);
ObjInf = ones(numobj,1)*Inf;

% try for speed increase
cl=struct2cell(EstObj);

for i=1:numMCI % cycle through each non-Pareto/phantom combo from SCORE calcs
    j_ind = Jstar(i,1);

    % data for dominated system
    lambda_j = Lambdas(j_ind-num_par);
    aj = lambda_j*alphas(slack_ind); 
    objj = cl{2,j_ind};
    covj = cl{3,j_ind};
    
    % data for phantom
    ph_ind = Jstar(i,2); %phantom number
    PHind = Phantoms(ph_ind,:); %Pareto indices for phantom
    
    MFGrads=GradMat;
    
    if abs(aj) < tol
        RateMF = 0;
        MFGrads(1,:)=[slack_ind i+cnt 0];
        for o=1:numobj
            MFGrads(o+1,:)=[PHind(o) i+cnt 0];
        end 
        MFGrads(numobj+2,:) = [z_ind  i+cnt 1.0] ;
    else
        % data for phantom
        PHobjs=ObjInf;
        PHvars=ObjZeros;
        PHalphs=ObjZeros;
        for b=1:numobj
            if PHind(b)<Inf
                Psys=PHind(b); % Pareto system number
                Pobjs = cl{2,Psys}; % Pareto objectives
                Pcovs = cl{3,Psys}; % Pareto covariances
                PHobjs(b) = Pobjs(b);
                PHvars(b) = Pcovs(b,b);
                PHalphs(b) = alphas(Psys);
            end
        end
           
        RateMF=0;
        Gradj=0;
        for m=1:numobj
            if objj(m)>PHobjs(m)
                RateMF=RateMF+(PHalphs(m)*aj*(PHobjs(m)-objj(m))^2)/(2*(aj*PHvars(m) + PHalphs(m)*covj(m,m)));
                Gradj=Gradj+ (PHalphs(m)^2*covj(m,m)*(PHobjs(m)-objj(m))^2)/(2*(aj*PHvars(m) + PHalphs(m)*covj(m,m))^2);
                
                grad = (aj^2*PHvars(m)*(PHobjs(m)-objj(m))^2)/(2*(aj*PHvars(m) + PHalphs(m)*covj(m,m))^2);
                MFGrads(m+1,:)=[PHind(m) i+cnt -grad];
            else
                MFGrads(m+1,:)=[Inf i+cnt 0];
            end
        end      
        MFGrads(1,:)=[slack_ind i+cnt -lambda_j*Gradj];
        MFGrads(numobj+2,:) = [z_ind  i+cnt 1.0] ;
     end

    MCIRates(i) = z - RateMF;
    MCI_grad_indices(numobj*(i-1)+2*i-1:numobj*i+2*i,:)=MFGrads;
end
MCI_grad_indices(MCI_grad_indices(:,1)==Inf,:)=[];



