function [score,binds]=qpSCORE(phobj,objJ,covJ)
opts = optimset('Algorithm','interior-point-convex','Display','off','TolX',1e-12,'TolFun',1e-12);
        
numobj=size(phobj,1);

invcovj = (covJ)\eye(numobj);
H = invcovj;
f = -1*invcovj*objJ;
A = eye(numobj);
b = phobj;

H=(H+H')/2; % safeguard against non-symmetry due to floating points
[x_star,~,~,~,Lam] = quadprog(H,f,A,b,[],[],[],[],[],opts);
        

score = 0.5*[objJ-x_star]'*invcovj*[objJ-x_star];
      
% DETERMINE WHICH COMPONENT THE SCORE IS CALCULATED ON
tol = 1e-6;

Lamda = Lam.ineqlin;
Lamda(Lamda(:,1)>tol,:)=1;
Lamda(Lamda(:,1)<=tol,:)=Inf;
binds = score*Lamda;
        