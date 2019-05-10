function [Jstar,Lambdas] = Prod_SCORE_iS(Phantoms,EstObj,numsys,num_par)

% determine sizes of inputs
[num_phants,numobj]=size(Phantoms);

% pre-compute for speed
VobjInf=ones(numobj,1)*Inf;
VobjOnes=ones(numobj,1);
VobjZeros=zeros(numobj,1);

% try for speed increase
cl=struct2cell(EstObj);

% pre-allocate result matrices
Scores = zeros(num_phants,(numsys-num_par));
% Jstar = end result storage: non-Pareto system number, phantom system number
Jstar = zeros(num_phants*numobj,2); % over-allocate for speed, then reduce

for i=1:num_phants % cycle through each phantom
    
    PHind = Phantoms(i,:); %Pareto indices for phantom
    % data for phantom
    PHobjs=VobjInf;
    for b=1:numobj
        if PHind(b)<Inf
            Psys=PHind(b); % Pareto system number
            Pobjs = cl{2,Psys}; % Pareto objectives            
            PHobjs(b) = Pobjs(b);
        end
    end
  
    Jcomps=VobjInf;
    Jidx=VobjInf;

    % pair with each non-Pareto
    for j=num_par+1:numsys 
        % get objectives and covariances for non-Pareto system j
        ObjJ = cl{2,j};
        CovJ = cl{3,j};
        
        % determine SCORE between non-Pareto and phantom
        score=0;
        binds=VobjInf;
        for m=1:numobj
            if ObjJ(m)>PHobjs(m)
                score=score+(PHobjs(m)-ObjJ(m))^2/(2*CovJ(m,m));
                binds(m)=1;
            end
        end      
        Jcurr=binds*score;
        
     
        Scores(i,j-num_par) = score;

        % Determine if this non-Pareto is closer than another
        for m=1:numobj
            if Jcurr(m)<Jcomps(m)
                Jcomps(m)=Jcurr(m);
                Jidx(m)=j;
            end
        end
    end
    
    Lidx=VobjOnes*i;
    Jstar_tmp=[Jidx Lidx];
    Jstar((numobj*(i-1)+1):i*numobj,:)=Jstar_tmp;
end

InvScores = 1./min(Scores);
Lambdas = InvScores/sum(InvScores);
Jstar(Jstar(:,1)==Inf,:)=[]; % reduce rows
Jstar = unique(Jstar,'rows');

end