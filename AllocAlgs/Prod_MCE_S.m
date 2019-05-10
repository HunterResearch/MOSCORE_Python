function [MCERates,MCEGrads] = Prod_MCE_S(alphas,EstObj,Mstar,numobj,z,z_ind)

tol = 1e-12;

% try for speed increase
cl=struct2cell(EstObj);

% Determine MCE Rates
numMCE = size(Mstar,1);
MCERates=zeros(numMCE,1);
MCEGrads=zeros(numMCE,9);

cnt = 0;

for k=1:numMCE % cycle through each Pareto combo from SCORE_MCE calcs
    i = Mstar(k,1); %pareto 1
    j = Mstar(k,2); %pareto 2
    
    cnt = cnt + 1;

    if( abs(alphas(i))<= tol || abs(alphas(j))<= tol)
        Rate = z;
        GradAMF = 0;
        GradBMF = 0;
    else
        obji = cl{2,i};
        covi = cl{3,i};
        objj = cl{2,j};
        covj = cl{3,j};
        if numobj==2
            [Rate,GradAMF,GradBMF]=MCE2d(alphas(i),alphas(j),obji,covi,objj,covj);
        elseif numobj==3
            [Rate,GradAMF,GradBMF]=MCE3d(alphas(i),alphas(j),obji,covi,objj,covj);
        else
            [Rate,GradAMF,GradBMF] = qpMCE(alphas(i),alphas(j),obji,covi,objj,covj);
        end
        Rate = z - Rate;
    end

    MCERates(cnt)=Rate;         
    MCEGrads(cnt,:)=[i cnt -1.0*GradAMF j cnt -1.0*GradBMF z_ind cnt 1]; 
end
MCEGrads = [ MCEGrads(:,1:3);MCEGrads(:,4:6);MCEGrads(:,7:9)];