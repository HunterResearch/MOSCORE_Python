% function [ rates, eqcons, gradRates, eqconsgrad,grad_indices] = Prod_Rates_True_Comp(alphas,EstObj,Phantoms,num_par)
function [ rates, Erate,Irate] = Prod_Rates_True_Comp(alphas,EstObj,Phantoms,num_par)
    
numobj=size(Phantoms,2);
z_ind = length(alphas);
z = alphas(z_ind); %get current objective value

% Determine MCE Rates
[MCERates,~] = Prod_MCE_True(alphas,EstObj,num_par,numobj,z,z_ind);
Erate=min(-MCERates);

% Determine MCI Rates
cnt = size(MCERates,1);
[MCIRates,~] = Prod_MCI_True(alphas,EstObj,Phantoms,num_par,cnt,numobj,z,z_ind);
Irate=min(-MCIRates);

% Assemble everything together
rates = [MCERates; MCIRates];

end