function [minRate] = Prod_MCE_BFComp(alphas,EstObj,num_par,numobj,z)

tol = 1e-12;

% try for speed increase
cl=struct2cell(EstObj);

minRate=Inf;

% cnt = 0;
for i=1:num_par % cycle through each Pareto
    for j=1:num_par % pair with each other other Pareto
        if i~=j
%             cnt = cnt + 1;

            if( abs(alphas(i))<= tol || abs(alphas(j))<= tol)
                Rate = z;
            else
                obji = cl{2,i};
                covi = cl{3,i};
                objj = cl{2,j};
                covj = cl{3,j};
                if numobj==2
                    [Rate,~,~]=MCE2d(alphas(i),alphas(j),obji,covi,objj,covj);
                elseif numobj==3
                    [Rate,~,~]=MCE3d(alphas(i),alphas(j),obji,covi,objj,covj);
                else
                    [Rate,~,~] = qpMCE(alphas(i),alphas(j),obji,covi,objj,covj);
                end
                Rate = z - Rate;
            end
        
            if -Rate<=minRate
                minRate=-Rate;
%                 minRate
            end     
        end
    end
end
