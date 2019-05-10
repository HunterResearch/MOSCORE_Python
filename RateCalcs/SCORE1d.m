function [curr_rate,Binds]=SCORE1d(Gobj,Jobj,CovJ)
 
g1=Gobj(1);
j1=Jobj(1);
covj11=CovJ(1,1);
 
curr_rate=(1/2)*(g1-j1)^2/covj11;
Binds=[curr_rate];

