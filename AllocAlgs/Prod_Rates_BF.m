function [ rates, eqcons, gradRates, eqconsgrad,grad_indices] = Prod_Rates_BF(alphas,EstObj,Kappa,num_par)
    
% numsys = numel(EstObj); %get the number of systems
numobj = numel(EstObj(1).obj);

z_ind = length(alphas);
z = alphas(z_ind); %get current objective value

% Determine MCE Rates
[MCERates,MCEGrads] = Prod_MCE_True(alphas,EstObj,num_par,numobj,z,z_ind);

% Determine MCI Rates
cnt = size(MCERates,1);
[MCIRates,MCIGrads] = Prod_MCI_BF(alphas,EstObj,Kappa,num_par,z,z_ind,cnt);


% Assemble everything together
rates = [MCERates; MCIRates];

grad_indices = [MCEGrads;MCIGrads];
gradRates = sparse(grad_indices(:,1),grad_indices(:,2),grad_indices(:,3));

eqcons = [];
eqconsgrad=[];

end