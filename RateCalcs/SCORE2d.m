function [curr_rate,Binds]=SCORE2d(Gobj,Jobj,CovJ)
 
g1=Gobj(1);
g2=Gobj(2);
j1=Jobj(1);
j2=Jobj(2);
covj11=CovJ(1,1);
covj12=CovJ(1,2);
covj22=CovJ(2,2);
 
cond11_1=(covj22*(g1-j1)+covj12*(j2-g2))/(covj12^2-covj11*covj22);
cond11_2=(covj12*(j1-g1)+covj11*(g2-j2))/(covj12^2-covj11*covj22);
cond10_1=(j1-g1)/covj11;
cond10_2=g2-j2+covj12*(j1-g1)/covj11;
cond01_1=g1-j1+covj12*(j2-g2)/covj22;
cond01_2=(j2-g2)/covj22;

  
if (0 < cond11_1 && 0 < cond11_2 ) 
  curr_rate=(covj22*(g1-j1)^2+(2*covj12*(j1-g1)+covj11*(g2-j2))*(g2-j2))/(2*covj11*covj22-2*covj12^2);
  Binds=[curr_rate; curr_rate];
elseif (0 < cond10_1 && 0 <= cond10_2 ) || (0 < cond11_1 && 0 == cond11_2 ) 
  curr_rate=(1/2)*covj11^(-1)*(g1-j1)^2;
  Binds=[curr_rate; Inf];
elseif (0 <= cond01_1 && 0 < cond01_2 ) || (0 == cond11_1 && 0 < cond11_2 ) 
  curr_rate=(1/2)*covj22^(-1)*(g2-j2)^2;
  Binds=[Inf; curr_rate];
else
    disp('SCORE Infinite');
    [curr_rate,Binds]=qpSCORE(Gobj,Jobj,CovJ);
end
