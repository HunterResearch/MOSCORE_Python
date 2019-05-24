function [ rate] = Prod_Rates_BF_Comp(alphas,EstObj,Kappa,num_par)
    
% numsys = numel(EstObj); %get the number of systems
numobj = numel(EstObj(1).obj);

z_ind = length(alphas);
z = alphas(z_ind); %get current objective value

% Determine min MCE Rate
[Erate] = Prod_MCE_BFComp(alphas,EstObj,num_par,numobj,z);

% Determine min MCI Rate
% [Irate] = Prod_MCI_BFComp_bad(alphas,EstObj,Kappa,num_par,z);
[Irate] = Prod_MCI_BFComp(alphas,EstObj,Kappa,num_par,z);

% determine min overall rate
rate = min(Erate,Irate);
% if Erate<Irate
%     fprintf('MCE Rate is lowest \n')
% else
%     fprintf('MCI Rate is lowest \n')
% end

end