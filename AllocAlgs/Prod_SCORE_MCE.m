function [Mstar] = Prod_SCORE_MCE(EstObj,num_par)

% determine sizes of inputs
numobj = numel(EstObj(1).obj);

% pre-compute for speed
VobjInf=ones(numobj,1)*Inf;
VobjOnes=ones(numobj,1);

% try for speed increase
cl=struct2cell(EstObj);

% pre-allocate result matrices
Scores = zeros(num_par,num_par);
AllScores = zeros(num_par*(num_par-1),1);
% Mstar = end result storage: Pareto 1 system number, Pareto 2 system number
Mstar = zeros(num_par*numobj,2); % over-allocate for speed, then reduce

cnt=0;
for i=1:num_par % cycle through each Pareto 2
    
    % get Pareto 2 objectives from EstObj 
    objI = cl{2,i};
    
    Jcomps=VobjInf;
    Jidx=VobjInf;

    % pair with each Pareto 1
    for j=1:num_par
        if i~=j
            % get objectives and covariances for Pareto system j
            objJ = cl{2,j};
            covJ = cl{3,j};

            % determine SCORE between Paretos
            if numobj==1
                [score,binds]=SCORE1d(objI,objJ,covJ); 
            elseif numobj==2
                [score,binds]=SCORE2d(objI,objJ,covJ); 
            elseif numobj==3
                [score,binds]=SCORE3d(objI,objJ,covJ);
            else
                [score,binds]=qpSCORE(objI,objJ,covJ);
            end

            Scores(i,j) = score;
            %add to list of scores to take median later
            cnt=cnt+1;
            AllScores(cnt)=score;

            Jcurr=binds;

            % Determine if this Pareto 1 is closer than another
            for m=1:numobj
                if Jcurr(m)<Jcomps(m)
                    Jcomps(m)=Jcurr(m);
                    Jidx(m)=j;
                end
            end
            
        end
    end
    
    Lidx=VobjOnes*i;
    Jstar_tmp=[Jidx Lidx];
    Mstar((numobj*(i-1)+1):i*numobj,:)=Jstar_tmp;
end

Mstar(Mstar(:,1)==Inf,:)=[]; % reduce rows
% switch the systems so that the rates are calculated in the opposite roles
MstarB = [Mstar(:,2), Mstar(:,1)];
Mstar=[Mstar; MstarB];

% add pairs of systems where the SCORE < percentile(all SCOREs)
% MedScore=median(AllScores);
MedScore=prctile(AllScores,25);
for a=1:num_par
    for b=1:num_par
        if a~=b && Scores(a,b)<=MedScore
            Mstar=[Mstar; b, a];
            Mstar=[Mstar; a, b];
        end
    end
end

Mstar = unique(Mstar,'rows');

end