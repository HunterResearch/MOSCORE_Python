function [curr_rate,GradI,GradJ] = qpMCE(aI,aJ,Iobj,Isig,Jobj,Jsig)

numobj=size(Iobj,1);

invcovi = (Isig)\eye(numobj);
invcovj = (Jsig)\eye(numobj);
H = blkdiag(aI*invcovi,aJ*invcovj);
f = [-1*aI*invcovi*Iobj; -1*aJ*invcovj*Jobj];
A = [-eye(numobj),eye(numobj)];
b = zeros(numobj,1);

H=(H+H')/2; % safeguard against non-symmetry due to floating points
if isOctave()
  x0 = [Iobj; Jobj];
  %fprintf('Using Octave qp \n')
  [x_star,obj,info,lambda] = qp(x0,H,f,[],[],[],[],[],A,b);
else
  opts = optimset('Algorithm','interior-point-convex','Display','off','TolX',1e-12,'TolFun',1e-12);
  [x_star,~] = quadprog(H,f,A,b,[],[],[],[],[],opts);
end

curr_rate = 0.5*aI*transpose(x_star(1:numobj)-Iobj)*invcovi*(x_star(1:numobj)-Iobj)+0.5*aJ*transpose(x_star((numobj+1):end)-Jobj)*invcovj*(x_star((numobj+1):end)-Jobj);
GradI= 0.5*transpose(x_star(1:numobj)-Iobj)*invcovi*(x_star(1:numobj)-Iobj);
GradJ= 0.5*transpose(x_star((numobj+1):end)-Jobj)*invcovj*(x_star((numobj+1):end)-Jobj);
