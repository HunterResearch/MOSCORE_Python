function [curr_rate,GradI,Grads]=MCI2d(aI,alphs,Iobj,Isig,Lobjs,Lsigs)
 
a1=alphs(1);
a2=alphs(2);
i1=Iobj(1);
i2=Iobj(2);
l1=Lobjs(1);
l2=Lobjs(2);
covi11=Isig(1,1);
covi12=Isig(1,2);
covi22=Isig(2,2);
Var1=Lsigs(1);
Var2=Lsigs(2);
 
cond11_1=a1*aI*(a2*covi12*(l2-i2)+(i1-l1)*(a2*covi22+aI*Var2)) ...
  *(aI*Var1*(a2*covi22+aI*Var2)+a1*((-1)*a2*covi12^2+a2*covi11* ...
  covi22+aI*covi11*Var2))^(-1);
cond11_2=a2*aI*(a1*covi12*(l1-i1)+(i2-l2)*(a1*covi11+aI*Var1)) ...
  *(aI*Var1*(a2*covi22+aI*Var2)+a1*((-1)*a2*covi12^2+a2*covi11* ...
  covi22+aI*covi11*Var2))^(-1);
cond10_1=a1*aI*(i1-l1)*(a1*covi11+aI*Var1)^(-1);
cond10_2=-i2+l2+a1*covi12*(i1-l1)*(a1*covi11+aI*Var1)^(-1);
cond01_1=-i1+l1+a2*covi12*(i2-l2)*(a2*covi22+aI*Var2)^(-1);
cond01_2=a2*aI*(i2-l2)*(a2*covi22+aI*Var2)^(-1);
 
if (0 < cond11_1 && 0 < cond11_2 ) 
   curr_rate=(1/2)*aI*(2*a1*a2*covi12*(l1-i1)*(i2-l2)+...
  a2*(i2-l2)^2*(a1*covi11+aI*Var1)+a1*(i1-l1)^2*(a2*covi22+aI* ...
  Var2))*(aI*Var1*(a2*covi22+aI*Var2)+a1*((-1)*a2*covi12^2+a2* ...
  covi11*covi22+aI*covi11*Var2))^(-1);
   GradI=(1/2)*(aI*Var1*(a2*covi22+aI*Var2)+a1*((-1)*a2*covi12^2+a2* ...
  covi11*covi22+aI*covi11*Var2))^(-2)*(a1^2*a2^2*(covi12^2+(-1) ...
  *covi11*covi22)*(i2-l2)*(2*covi12*(i1-l1)+covi11*( ...
  l2-i2))+a2^2*aI*(i2-l2)^2*Var1*((-2)*a1*covi12^2+2* ...
  a1*covi11*covi22+aI*covi22*Var1)+2*a1*a2*aI^2*covi12*(i1-l1)...
  *(i2-l2)*Var1*Var2+a1^2*(i1-l1)^2*(a2^2* ...
  covi22*((-1)*covi12^2+covi11*covi22)+(-2)*a2*aI*(covi12^2+(-1)* ...
  covi11*covi22)*Var2+aI^2*covi11*Var2^2));
   Grads=[(1/2)*aI^2*Var1*(a2*(covi22*(i1-l1)+covi12*(l2-i2))+ ...
  aI*(i1-l1)*Var2)^2*(aI*Var1*(a2*covi22+aI*Var2)+a1*((-1) ...
  *a2*covi12^2+a2*covi11*covi22+aI*covi11*Var2))^(-2),(1/2)* ...
  aI^2*(a1*(covi12*(i1-l1)+covi11*(l2-i2))+aI*(l2-i2)...
  *Var1)^2*Var2*(aI*Var1*(a2*covi22+aI*Var2)+a1*((-1)*a2* ...
  covi12^2+a2*covi11*covi22+aI*covi11*Var2))^(-2)];
 
elseif (0 < cond10_1 && 0 <= cond10_2 ) || (0 < cond11_1 && 0 == cond11_2 ) 
   curr_rate=a1*aI*(i1-l1)^2*(2*a1*covi11+2*aI*Var1)^(-1);
   GradI=(1/2)*a1^2*covi11*(i1-l1)^2*(a1*covi11+aI*Var1)^(-2);
   Grads=[(1/2)*aI^2*(i1-l1)^2*Var1*(a1*covi11+aI*Var1)^(-2),0];
 
elseif (0 <= cond01_1 && 0 < cond01_2 ) || (0 == cond11_1 && 0 < cond11_2 ) 
   curr_rate=a2*aI*(i2-l2)^2*(2*a2*covi22+2*aI*Var2)^(-1);
   GradI=(1/2)*a2^2*covi22*(i2-l2)^2*(a2*covi22+aI*Var2)^(-2);
   Grads=[0,(1/2)*aI^2*(i2-l2)^2*Var2*(a2*covi22+aI*Var2)^(-2)];

else
%    	fprintf('Calling QuadProg (MCI - 2) \n');
   	[curr_rate,Grads]=qpMCI(aI,Iobj,Isig,alphs,Lobjs,Lsigs);
   	GradI=Grads(1);
   	Grads(1)=[];
end
 
% if curr_rate==Inf
%    	fprintf('Infinite MCI Looping \n');
%    	pause;
% end
