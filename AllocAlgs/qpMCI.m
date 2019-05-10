function [curr_rate,Grads] = qpMCI(aI,Iobj,Isig,alph,Lobj,Lvar)

Lvar(Lvar==0)=1e-100; % make a zero variance (if it happens) a very small 
%positive number to ensure quadprog works

numobj=size(Iobj,1);

objVal_j = Iobj;
cov_j = Isig;
invcov_j = cov_j\eye(numobj);

H = aI*invcov_j;
f = -1*aI*invcov_j*objVal_j;
A = eye(numobj);
b=ones(numobj,1)*Inf;

% data for phantom
% go through objectives of phantom and add to quadprog inputs
Indic=[]; %indices of objectives that are involved in rate
for p=1:numobj
    Avec=zeros(numobj,1);
    if Lobj(p)<Inf
        Indic=[Indic p]; %add objective index to list
        Avec(p)=-1;
        H = blkdiag(H, alph(p)*(Lvar(p))^(-1));
        f = [f; -1*alph(p)*(Lvar(p))^(-1)*Lobj(p)];
        A = [A Avec];
        b(p) = 0;            
    end
end
    
H=(H+H')/2; % safeguard against non-symmetry due to floating points

opts = optimset('Algorithm','interior-point-convex','Display','off','TolX',1e-12,'TolFun',1e-12);
[x_star,~] = quadprog(H,f,A,b,[],[],[],[],[],opts);

%compute rate and gradients using x_star
Grads=[];
curr_rate = 0.5*aI*transpose(objVal_j-x_star(1:numobj))*invcov_j*(objVal_j-x_star(1:numobj));
gradj = 0.5*transpose(x_star(1:numobj)-objVal_j)*invcov_j*(x_star(1:numobj)-objVal_j);
Grads=[Grads; gradj];
for o=1:size(Indic,2)
    idxo=Indic(o);
    curr_rate = curr_rate + 0.5*alph(idxo)*transpose(x_star(numobj+o)-Lobj(idxo))*(Lvar(idxo))^(-1)*(x_star(numobj+o)-Lobj(idxo));
    gradj = 0.5*transpose(x_star(numobj+o)-Lobj(idxo))*(Lvar(idxo))^(-1)*(x_star(numobj+o)-Lobj(idxo));
    Grads=[Grads; gradj];
end 