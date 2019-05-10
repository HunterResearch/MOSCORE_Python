function [curr_rate,GradI,Grads]=MCI1d(aI,alphs,Iobj,Isig,Lobjs,Lsigs)
 
a1=alphs(1);
i1=Iobj(1);
l1=Lobjs(1);
covi11=Isig(1,1);
Var1=Lsigs(1);
  
cond1_1=a1*aI*(i1-l1)/(a1*covi11+aI*Var1);
 
if (0 < cond1_1 ) 
   curr_rate=a1*aI*(i1-l1)^2/(2*a1*covi11+2*aI*Var1);
   GradI=(1/2)*a1^2*covi11*(i1-l1)^2*(a1*covi11+aI*Var1)^(-2);
   Grads=(1/2)*aI^2*(i1-l1)^2*Var1*(a1*covi11+aI*Var1)^(-2);

else
%     fprintf('Calling QuadProg (MCI - 1) \n');
   	[curr_rate,Grads]=qpMCI(aI,Iobj,Isig,alphs,Lobjs,Lsigs);
   	GradI=Grads(1);
   	Grads(1)=[];
end
 
% if curr_rate==Inf
%    	fprintf('Infinite MCI Looping \n');
%    	pause;
% end