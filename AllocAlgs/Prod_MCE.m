function [MCERates,MCEGrads] = Prod_MCE(alphas,EstObj,num_par,numobj,z,z_ind)

tol = 1e-12;

% try for speed increase
cl=struct2cell(EstObj);

% Determine MCE Rates
numMCE = (num_par-1)*(num_par);
MCERates=zeros(numMCE,1);

MCEGrads=zeros(numMCE,9);
cnt = 0;
for i=1:num_par % cycle through each Pareto
    for j=1:num_par % pair with each other other Pareto
        if i~=j
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
    end
end
MCEGrads = [ MCEGrads(:,1:3);MCEGrads(:,4:6);MCEGrads(:,7:9)];