function [score,binds]=qpSCORE(phobj,objJ,covJ)

numobj=size(phobj,1);

invcovj = (covJ)\eye(numobj);
H = invcovj;
f = -1*invcovj*objJ;
A = eye(numobj);
b = phobj;

H=(H+H')/2; % safeguard against non-symmetry due to floating points

if isOctave()
  x0 = objJ;
  %fprintf('Using Octave qp \n')
  [x_star,obj,info,lambda] = qp(x0,H,f,[],[],[],[],[],A,b);
  x_star
  info
  lambda
  Lamda = lambda;
else
  opts = optimset('Algorithm','interior-point-convex','Display','off','TolX',1e-12,'TolFun',1e-12);
  [x_star,~,~,~,Lam] = quadprog(H,f,A,b,[],[],[],[],[],opts);
  Lamda = Lam.ineqlin;
end

score = 0.5*[objJ-x_star]'*invcovj*[objJ-x_star];

% DETERMINE WHICH COMPONENT THE SCORE IS CALCULATED ON
tol = 1e-6;

Lamda(Lamda(:,1)>tol,:)=1;
Lamda(Lamda(:,1)<=tol,:)=Inf;
binds = score*Lamda;
