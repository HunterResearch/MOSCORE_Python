function [Jstar,Lambdas] = Prod_SCORE(Phantoms,EstObj,numsys,num_par)

% determine sizes of inputs
[num_phants,numobj]=size(Phantoms);

% pre-compute for speed
VobjInf=ones(numobj,1)*Inf;
VobjOnes=ones(numobj,1);
o=ones(numobj,numobj);
objlist=1:numobj;

% try for speed increase
cl=struct2cell(EstObj);

% pre-allocate result matrices
Scores = zeros(num_phants,(numsys-num_par));
% Jstar = end result storage: non-Pareto system number, phantom system number
Jstar = zeros(num_phants*numobj,2); % over-allocate for speed, then reduce

for i=1:num_phants % cycle through each phantom
    
    % get phantom objectives from EstObj 
    phidx=objlist;
        % NEW WAY (vectorization)
    idx2=Phantoms(i,phidx);
    phidx=find(phidx.*(idx2~=Inf));
    obidx=idx2(idx2~=Inf);
%     o(:,phidx)=[EstObj(obidx).obj];
    o(:,phidx)=[cl{2,obidx}];
    dia=diag(o);
    phobj=dia(phidx);

        % OLD WAY with FOR loop
%     phant_obj=VobjInf;
%     for k=1:numobj
%         idx = Phantoms(i,k); 
%         if idx==Inf
%             phant_obj(k)=Inf;
%             phidx(phidx==k)=[];
%         else
%             tmp1 = EstObj(idx).obj;
%             phant_obj(k) = tmp1(k);
%         end
%     end
%     phobj=phant_obj(phidx);

    sz=size(phidx,2);
    
    Jcomps=VobjInf;
    Jidx=VobjInf;

    % pair with each non-Pareto
    for j=num_par+1:numsys 
        % get objectives and covariances for non-Pareto system j
        ObjJ = cl{2,j};
%         ObjJ = EstObj(j).obj;
        objJ = ObjJ(phidx);
        CovJ = cl{3,j};
%         CovJ = EstObj(j).cov;
        covJ = CovJ(phidx,phidx);
        
        % determine SCORE between non-Pareto and phantom
        if sz==1
            [score,binds]=SCORE1d(phobj,objJ,covJ); 
        elseif sz==2
            [score,binds]=SCORE2d(phobj,objJ,covJ); 
        elseif sz==3
            [score,binds]=SCORE3d(phobj,objJ,covJ);
        else
            [score,binds]=qpSCORE(phobj,objJ,covJ);
        end

        Jcurr=VobjInf;
        Jcurr(phidx)=binds;
        
        Scores(i,j-num_par) = score;

        % Determine if this non-Pareto is closer than another

        % VECTORIZATION (did not work to reduce time)
%         d=Jcurr2(objs)<Jcomps2(objs);
%         r=objs(d);
%         Jcomps2(r)=Jcurr2(r);
%         Jidx2(r)=j;

        
        % FOR LOOP (faster than vectorization, surprisingly)
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
% Jstar=Jstar([sum(diff(Jstar)==0,2); 0]~=2,:); %use diff approach instead of 'unique' function for speed

end