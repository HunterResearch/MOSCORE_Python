function [ rates, eqcons, gradRates, eqconsgrad] = Prod_Rates(alphas,EstObj,Phantoms,num_par,Jstar,Lambdas)

numobj=size(Phantoms,2);
z_ind = length(alphas);
z = alphas(z_ind); %get current objective value

% Determine MCE Rates
[MCERates,MCEGrads] = Prod_MCE(alphas,EstObj,num_par,numobj,z,z_ind);

% Determine MCI Rates
cnt = size(MCERates,1);
[MCIRates,MCIGrads] = Prod_MCI(alphas,Lambdas,Jstar,EstObj,Phantoms,num_par,cnt,numobj,z,z_ind);

% Assemble everything together
rates = [MCERates; MCIRates];

grad_indices = [MCEGrads;MCIGrads];
gradRates = sparse(grad_indices(:,1),grad_indices(:,2),grad_indices(:,3));

eqcons = [];
eqconsgrad=[];

end