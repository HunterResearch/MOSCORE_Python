function [MCERates,MCEGrads] = Prod_MCE_iS(alphas,EstObj,Mstar,numobj,z,z_ind)

tol = 1e-50;

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
        
        Rate=0;
        GradAMF = 0;
        GradBMF = 0;
        for m=1:numobj
            if obji(m)>objj(m)
                Rate=Rate+(alphas(i)*alphas(j)*(obji(m)-objj(m))^2)/(2*(alphas(j)*covi(m,m) + alphas(i)*covj(m,m)));
                GradAMF=GradAMF+(alphas(j)^2*covi(m,m)*(obji(m)-objj(m))^2)/(2*(alphas(j)*covi(m,m) + alphas(i)*covj(m,m))^2);
                GradBMF=GradBMF+(alphas(i)^2*covj(m,m)*(obji(m)-objj(m))^2)/(2*(alphas(j)*covi(m,m) + alphas(i)*covj(m,m))^2);
            end
        end      
        Rate = z - Rate;
    end

    MCERates(cnt)=Rate;         
    MCEGrads(cnt,:)=[i cnt -1.0*GradAMF j cnt -1.0*GradBMF z_ind cnt 1]; 
end
MCEGrads = [ MCEGrads(:,1:3);MCEGrads(:,4:6);MCEGrads(:,7:9)];