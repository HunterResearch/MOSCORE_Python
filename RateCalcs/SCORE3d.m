function [curr_rate,Binds]=SCORE3d(Gobj,Jobj,CovJ)
 
g1=Gobj(1);
g2=Gobj(2);
g3=Gobj(3);
j1=Jobj(1);
j2=Jobj(2);
j3=Jobj(3);
covj11=CovJ(1,1);
covj12=CovJ(1,2);
covj22=CovJ(2,2);
covj13=CovJ(1,3);
covj23=CovJ(2,3);
covj33=CovJ(3,3);
 
cond111_1=(covj13^2*covj22+(-2)*covj12*covj13*covj23+covj12^2*covj33+ ...
  covj11*(covj23^2+(-1)*covj22*covj33))^(-1)*(covj22*covj33*(g1-j1)+...
  covj23^2*(j1-g1)+covj13*covj23*(g2-j2)+covj12* ...
  covj33*(j2-g2)+covj12*covj23*(g3-j3)+covj13*covj22*(j3-g3));
cond111_2=(covj13^2*covj22+(-2)*covj12*covj13*covj23+covj12^2*covj33+ ...
  covj11*(covj23^2+(-1)*covj22*covj33))^(-1)*(covj13*covj23*(g1-j1)+...
  covj12*covj33*(j1-g1)+covj11*covj33*(g2-j2)+ ...
  covj13^2*(j2-g2)+covj12*covj13*(g3-j3)+covj11*covj23*(j3-g3));
cond111_3=(covj13^2*covj22+(-2)*covj12*covj13*covj23+covj12^2*covj33+ ...
  covj11*(covj23^2+(-1)*covj22*covj33))^(-1)*(covj12*covj23*(g1-j1)+...
  covj13*covj22*(j1-g1)+covj12*covj13*(g2-j2)+ ...
  covj11*covj23*(j2-g2)+covj11*covj22*(g3-j3)+covj12^2*(j3-g3));
cond110_1=(covj12^2+(-1)*covj11*covj22)^(-1)*(covj22*g1+(-1)*covj12*g2+(-1)*covj22*j1+covj12*j2);
cond110_2=(covj12^2+(-1)*covj11*covj22)^(-1)*(covj12*(j1-g1)+covj11*(g2-j2));
cond110_3=g3-j3+(covj12^2+(-1)*covj11*covj22)^(-1)*(covj23*(covj12*(j1-g1)+covj11*(g2-j2))+...
    covj13*(covj22*(g1-j1)+covj12*(j2-g2)));
cond101_1=(covj13^2+(-1)*covj11*covj33)^(-1)*(covj33*g1+(-1)*covj13*g3+(-1)*covj33*j1+covj13*j3);
cond101_2=g2+(covj13^2+(-1)*covj11*covj33)^(-1)*(covj12*covj33*(g1-j1)+...
    (-1)*covj13^2*j2+covj11*(covj23*g3+covj33*j2+(-1)*covj23*j3) ...
  +covj13*(covj23*(j1-g1)+covj12*(j3-g3)));
cond101_3=(covj13^2+(-1)*covj11*covj33)^(-1)*(covj13*(j1-g1)+covj11*(g3-j3));
cond100_1=covj11^(-1)*(j1-g1);
cond100_2=g2-j2+covj11^(-1)*covj12*(j1-g1);
cond100_3=g3-j3+covj11^(-1)*covj13*(j1-g1);
cond011_1=g1+(covj23^2+(-1)*covj22*covj33)^(-1)*((-1)*covj23^2*j1+covj22* ...
  covj33*j1+covj13*(covj23*(j2-g2)+covj22*(g3-j3))+ ...
  covj12*(covj33*g2+(-1)*covj23*g3+(-1)*covj33*j2+covj23*j3));
cond011_2=(covj23^2+(-1)*covj22*covj33)^(-1)*(covj33*g2+(-1)*covj23*g3+(-1)*covj33*j2+covj23*j3);
cond011_3=(covj23^2+(-1)*covj22*covj33)^(-1)*(covj23*(j2-g2)+covj22*(g3-j3));
cond010_1=g1+(-1)*covj22^(-1)*(covj12*g2+covj22*j1+(-1)*covj12*j2);
cond010_2=covj22^(-1)*(j2-g2);
cond010_3=g3-j3+covj22^(-1)*covj23*(j2-g2);
cond001_1=g1+(-1)*covj33^(-1)*(covj13*g3+covj33*j1+(-1)*covj13*j3);
cond001_2=g2+(-1)*covj33^(-1)*(covj23*g3+covj33*j2+(-1)*covj23*j3);
cond001_3=covj33^(-1)*(j3-g3);
 
if (0 < cond111_1 && 0 < cond111_2 && 0 < cond111_3 ) 
    curr_rate=(1/2)*(covj13^2*covj22+(-2)*covj12*covj13*covj23+covj12^2* ...
      covj33+covj11*(covj23^2-covj22*covj33))^(-1)*((covj23^2-covj22*covj33)*(g1-j1)^2+...
      ((-2)*covj13*covj23+2*covj12*covj33)*(g1-j1)*(g2-j2)+(covj13^2-covj11*covj33)* ...
      (g2-j2)^2+2*covj11*covj23*(g2-j2)*(g3-j3)+ ...
      covj12^2*(g3-j3)^2+(((-2)*covj13*covj22+2*covj12*covj23)*( ...
      g1-j1)+2*covj12*covj13*(g2-j2)+covj11*covj22*(g3-j3))*(j3-g3));
    Binds=[ curr_rate; curr_rate; curr_rate];
    
elseif (0 < cond110_1 && 0 < cond110_2 && 0 <= cond110_3 ) || (0 < cond111_1 && 0 < cond111_2 && 0 == cond111_3 ) 
    curr_rate=((-2)*covj12^2+2*covj11*covj22)^(-1)*(covj22*(g1-j1)^2+( ...
      2*covj12*(j1-g1)+covj11*(g2-j2))*(g2-j2));
    Binds=[ curr_rate; curr_rate; Inf];
    
elseif (0 < cond101_1 && 0 <= cond101_2 && 0 < cond101_3 ) || (0 < cond111_1 && 0 == cond111_2 && 0 < cond111_3 ) 
    curr_rate=((-2)*covj13^2+2*covj11*covj33)^(-1)*(covj33*(g1-j1)^2+( ...
      2*covj13*(j1-g1)+covj11*(g3-j3))*(g3-j3));
    Binds=[ curr_rate; Inf; curr_rate];

elseif (0 < cond100_1 && 0 <= cond100_2 && 0 <= cond100_3 ) || (0 < cond111_1 && 0 == cond111_2 && 0 == cond111_3 ) || (0 < cond110_1 && 0 == cond110_2 && 0 <= cond110_3 ) || (0 < cond101_1 && 0 <= cond101_2 && 0 == cond101_3 ) 
    curr_rate=(1/2)*covj11^(-1)*(g1-j1)^2;
    Binds=[ curr_rate; Inf; Inf];

elseif (0 <= cond011_1 && 0 < cond011_2 && 0 < cond011_3 ) || (0 == cond111_1 && 0 < cond111_2 && 0 < cond111_3 ) 
    curr_rate=((-2)*covj23^2+2*covj22*covj33)^(-1)*(covj33*(g2-j2)^2+( ...
      2*covj23*(j2-g2)+covj22*(g3-j3))*(g3-j3));
    Binds=[ Inf; curr_rate; curr_rate];

elseif (0 <= cond010_1 && 0 < cond010_2 && 0 <= cond010_3 ) || (0 == cond111_1 && 0 < cond111_2 && 0 == cond111_3 ) || (0 == cond110_1 && 0 < cond110_2 && 0 <= cond110_3 ) || (0 <= cond011_1 && 0 < cond011_2 && 0 == cond011_3 ) 
    curr_rate=(1/2)*covj22^(-1)*(g2-j2)^2;
    Binds=[ Inf; curr_rate; Inf];

elseif (0 <= cond001_1 && 0 <= cond001_2 && 0 < cond001_3 ) || (0 == cond111_1 && 0 == cond111_2 && 0 < cond111_3 ) || (0 == cond101_1 && 0 <= cond101_2 && 0 < cond101_3 ) || (0 <= cond011_1 && 0 == cond011_2 && 0 < cond011_3 ) 
    curr_rate=(1/2)*covj33^(-1)*(g3-j3)^2;
    Binds=[ Inf; Inf; curr_rate];

else
    disp('SCORE Infinite');
    [curr_rate,Binds]=qpSCORE(Gobj,Jobj,CovJ);
end
