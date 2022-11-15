#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 21:21:48 2019

@author: nathangeldner
"""
import scipy.linalg as linalg
from cvxopt import matrix, solvers
import numpy as np


def MCI_1d(aI, Iobj, Isig, alphs, Lobjs, Lsigs):
    """calculates phantom MCI rates

    Parameters
    ----------
    aI: float, allocation to system I
    Iobj: numpy arra, objectives of system I (yes I know there's no need for the matrix in 1d, but this way it's consistent across cases - Nathan)
    Isig: 2d numpy array, covariance matrix of objectives for system i
    alphs: numpy array of allocations of the paretos from which the phantom pareto at hand is derived
    Lobjs: numpy array, phantom pareto objectives
    Lsigs: numpy array, phantom pareto variances (phantom objectives are treated as independent)
            
    Returns
    -------
    curr_rate:MCI event decay rate (phantom approximation) given aI and alphs
    gradI: gradient of curr_rate wrt system I
    Grads: gradient of curr_rate wrt alphs"""
            
        
        
    a1=alphs[0]
    i1=Iobj[0]
    l1=Lobjs[0]
    covi11=Isig[0,0]
    Var1=Lsigs[0]
    
    cond1_1=a1*aI*(i1-l1)/(a1*covi11+aI*Var1)
    
    if 0 < cond1_1 :
        curr_rate=a1*aI*(i1-l1)**2/(2*a1*covi11+2*aI*Var1)
        GradI=(1/2)*a1**2*covi11*(i1-l1)**2*(a1*covi11+aI*Var1)**(-2)
        Grads=np.array([(1/2)*aI**2*(i1-l1)**2*Var1*(a1*covi11+aI*Var1)**(-2)])
        
    else:
        curr_rate,GradI, Grads= MCI_four_d_plus(aI,Iobj,Isig,alphs,Lobjs,Lsigs)

        
    return curr_rate, GradI, Grads

def MCI_2d(aI, Iobj, Isig, alphs, Lobjs, Lsigs):
    """calculates phantom MCI rates

    parameters
    ----------
    aI: float, allocation to system I
    Iobj: numpy arra, objectives of system I (yes I know there's no need for the matrix in 1d, but this way it's consistent across cases - Nathan)
    Isig: 2d numpy array, covariance matrix of objectives for system i
    alphs: numpy array of allocations of the paretos from which the phantom pareto at hand is derived
    Lobjs: numpy array, phantom pareto objectives
    Lsigs: numpy array, phantom pareto variances (phantom objectives are treated as independent)

    returns
    -------
    curr_rate:MCI event decay rate (phantom approximation) given aI and alphs
    gradI: gradient of curr_rate wrt system I
    Grads: gradient of curr_rate wrt alphs"""
    a1=alphs[0]
    a2=alphs[1]
    i1=Iobj[0]
    i2=Iobj[1]
    l1=Lobjs[0]
    l2=Lobjs[1]
    covi11=Isig[0,0]
    covi12=Isig[0,1]
    covi22=Isig[1,1]
    Var1=Lsigs[0]
    Var2=Lsigs[1]
    
    cond11_1=a1*aI*(a2*covi12*(l2-i2)+(i1-l1)*(a2*covi22+aI*Var2)) \
  *(aI*Var1*(a2*covi22+aI*Var2)+a1*((-1)*a2*covi12**2+a2*covi11* \
  covi22+aI*covi11*Var2))**(-1)
    cond11_2=a2*aI*(a1*covi12*(l1-i1)+(i2-l2)*(a1*covi11+aI*Var1)) \
  *(aI*Var1*(a2*covi22+aI*Var2)+a1*((-1)*a2*covi12**2+a2*covi11* \
  covi22+aI*covi11*Var2))**(-1)
    cond10_1=a1*aI*(i1-l1)*(a1*covi11+aI*Var1)**(-1)
    cond10_2=-i2+l2+a1*covi12*(i1-l1)*(a1*covi11+aI*Var1)**(-1)
    cond01_1=-i1+l1+a2*covi12*(i2-l2)*(a2*covi22+aI*Var2)**(-1)
    cond01_2=a2*aI*(i2-l2)*(a2*covi22+aI*Var2)**(-1)
    
    if (0 < cond11_1 and 0< cond11_2):
        curr_rate=(1/2)*aI*(2*a1*a2*covi12*(l1-i1)*(i2-l2)+\
  a2*(i2-l2)**2*(a1*covi11+aI*Var1)+a1*(i1-l1)**2*(a2*covi22+aI* \
  Var2))*(aI*Var1*(a2*covi22+aI*Var2)+a1*((-1)*a2*covi12**2+a2* \
  covi11*covi22+aI*covi11*Var2))**(-1)
        GradI=(1/2)*(aI*Var1*(a2*covi22+aI*Var2)+a1*((-1)*a2*covi12**2+a2* \
  covi11*covi22+aI*covi11*Var2))**(-2)*(a1**2*a2**2*(covi12**2+(-1) \
  *covi11*covi22)*(i2-l2)*(2*covi12*(i1-l1)+covi11*( \
  l2-i2))+a2**2*aI*(i2-l2)**2*Var1*((-2)*a1*covi12**2+2* \
  a1*covi11*covi22+aI*covi22*Var1)+2*a1*a2*aI**2*covi12*(i1-l1)\
  *(i2-l2)*Var1*Var2+a1**2*(i1-l1)**2*(a2**2* \
  covi22*((-1)*covi12**2+covi11*covi22)+(-2)*a2*aI*(covi12**2+(-1)* \
  covi11*covi22)*Var2+aI**2*covi11*Var2**2))
        Grads=np.array([(1/2)*aI**2*Var1*(a2*(covi22*(i1-l1)+covi12*(l2-i2))+ \
  aI*(i1-l1)*Var2)**2*(aI*Var1*(a2*covi22+aI*Var2)+a1*((-1) \
  *a2*covi12**2+a2*covi11*covi22+aI*covi11*Var2))**(-2),(1/2)* \
  aI**2*(a1*(covi12*(i1-l1)+covi11*(l2-i2))+aI*(l2-i2)\
  *Var1)**2*Var2*(aI*Var1*(a2*covi22+aI*Var2)+a1*((-1)*a2* \
  covi12**2+a2*covi11*covi22+aI*covi11*Var2))**(-2)])
    
    elif (0 < cond10_1 and 0<= cond10_2) or (0<cond11_1 and 0==cond11_2):
        curr_rate=a1*aI*(i1-l1)**2*(2*a1*covi11+2*aI*Var1)**(-1)
        GradI=(1/2)*a1**2*covi11*(i1-l1)**2*(a1*covi11+aI*Var1)**(-2)
        Grads=np.array([(1/2)*aI**2*(i1-l1)**2*Var1*(a1*covi11+aI*Var1)**(-2),0])
        
    elif (0<= cond01_1 and 0<cond01_2) or (0==cond11_1 and 0<cond11_2):
        curr_rate=a2*aI*(i2-l2)**2*(2*a2*covi22+2*aI*Var2)**(-1)
        GradI=(1/2)*a2**2*covi22*(i2-l2)**2*(a2*covi22+aI*Var2)**(-2)
        Grads=np.array([0,(1/2)*aI**2*(i2-l2)**2*Var2*(a2*covi22+aI*Var2)**(-2)])
    else:
        curr_rate,GradI, Grads= MCI_four_d_plus(aI,Iobj,Isig,alphs,Lobjs,Lsigs)

        
    return curr_rate, GradI, Grads

def MCI_3d(aI, Iobj, Isig, alphs, Lobjs, Lsigs):
    """calculates phantom MCI rates

    parameters
    ----------
    aI: float, allocation to system I
    Iobj: numpy arra, objectives of system I (yes I know there's no need for the matrix in 1d, but this way it's consistent across cases - Nathan)
    Isig: 2d numpy array, covariance matrix of objectives for system i
    alphs: numpy array of allocations of the paretos from which the phantom pareto at hand is derived
    Lobjs: numpy array, phantom pareto objectives
    Lsigs: numpy array, phantom pareto variances (phantom objectives are treated as independent)

    returns
    -------
    curr_rate:MCI event decay rate (phantom approximation) given aI and alphs
    gradI: gradient of curr_rate wrt system I
    Grads: gradient of curr_rate wrt alphs"""

    a1=alphs[0]
    a2=alphs[1]
    a3=alphs[2]
    i1=Iobj[0]
    i2=Iobj[1]
    i3=Iobj[2]
    l1=Lobjs[0]
    l2=Lobjs[1]
    l3=Lobjs[2]
    covi11=Isig[0,0]
    covi12=Isig[0,1]
    covi22=Isig[1,1]
    covi13=Isig[0,2]
    covi23=Isig[1,2]
    covi33=Isig[2,2]
    Var1=Lsigs[0]
    Var2=Lsigs[1]
    Var3=Lsigs[2]
    
    cond111_1=a1*aI*(a2*a3*covi23**2*(l1-i1)+a2*a3*covi13*covi23*(i2-l2)\
  +a2*a3*covi12*covi23*(i3-l3)+a3*covi13*(l3-i3) \
  *(a2*covi22+aI*Var2)+a2*covi12*(l2-i2)*(a3*covi33+aI*Var3) \
  +(i1-l1)*(a2*covi22+aI*Var2)*(a3*covi33+aI*Var3))*((-1)* \
  a1*a3*(a2*(covi13**2*covi22+(-2)*covi12*covi13*covi23+covi11* \
  covi23**2+covi12**2*covi33+(-1)*covi11*covi22*covi33)+aI*( \
  covi13**2+(-1)*covi11*covi33)*Var2)+a1*aI*((-1)*a2*covi12**2+a2* \
  covi11*covi22+aI*covi11*Var2)*Var3+aI*Var1*(aI*Var2*(a3*covi33+ \
  aI*Var3)+a2*((-1)*a3*covi23**2+a3*covi22*covi33+aI*covi22*Var3)) \
  )**(-1)
    cond111_2=a2*aI*(a1*a3*covi13*covi23*(i1-l1)+a1*a3*covi13**2*(l2-i2)\
  +a1*a3*covi12*covi13*(i3-l3)+a3*covi23*(l3-i3) \
  *(a1*covi11+aI*Var1)+a1*covi12*(l1-i1)*(a3*covi33+aI*Var3) \
  +(i2-l2)*(a1*covi11+aI*Var1)*(a3*covi33+aI*Var3))*((-1)* \
  a1*a3*(a2*(covi13**2*covi22+(-2)*covi12*covi13*covi23+covi11* \
  covi23**2+covi12**2*covi33+(-1)*covi11*covi22*covi33)+aI*( \
  covi13**2+(-1)*covi11*covi33)*Var2)+a1*aI*((-1)*a2*covi12**2+a2* \
  covi11*covi22+aI*covi11*Var2)*Var3+aI*Var1*(aI*Var2*(a3*covi33+ \
  aI*Var3)+a2*((-1)*a3*covi23**2+a3*covi22*covi33+aI*covi22*Var3)) \
  )**(-1)
    cond111_3=a3*aI*(a1*a2*covi12*covi23*(i1-l1)+a1*a2*covi12*covi13*( \
  i2-l2)+a1*a2*covi12**2*(l3-i3)+a2*covi23*(l2-i2) \
  *(a1*covi11+aI*Var1)+a1*covi13*(l1-i1)*(a2*covi22+aI*Var2) \
  +(i3-l3)*(a1*covi11+aI*Var1)*(a2*covi22+aI*Var2))*((-1)* \
  a1*a3*(a2*(covi13**2*covi22+(-2)*covi12*covi13*covi23+covi11* \
  covi23**2+covi12**2*covi33+(-1)*covi11*covi22*covi33)+aI*( \
  covi13**2+(-1)*covi11*covi33)*Var2)+a1*aI*((-1)*a2*covi12**2+a2* \
  covi11*covi22+aI*covi11*Var2)*Var3+aI*Var1*(aI*Var2*(a3*covi33+ \
  aI*Var3)+a2*((-1)*a3*covi23**2+a3*covi22*covi33+aI*covi22*Var3)) \
  )**(-1)
    cond110_1=a1*aI*(a2*(covi22*(i1-l1)+covi12*(l2-i2))+\
  aI*(i1-l1)*Var2)*(aI*Var1*(a2*covi22+aI*Var2)+a1*((-1)*a2*covi12**2+ \
  a2*covi11*covi22+aI*covi11*Var2))**(-1)
    cond110_2=a2*aI*(a1*(covi12*(l1-i1)+covi11*(i2-l2))+ \
  aI*(i2-l2)*Var1)*(aI*Var1*(a2*covi22+aI*Var2)+a1*((-1)*a2*covi12**2+ \
  a2*covi11*covi22+aI*covi11*Var2))**(-1)
    cond110_3=l3+(-1)*(a1*a2*((-1)*covi12**2*i3+covi12*covi23*(i1-l1)+ \
  covi13*(covi22*(l1-i1)+covi12*(i2-l2))+covi11*((-1)* \
  covi23*i2+covi22*i3+covi23*l2))+a2*aI*(covi22*i3+ \
  covi23*(l2-i2))*Var1+aI*(a1*(covi11*i3+covi13*(l1-i1))+aI*i3*Var1) \
  *Var2)*(aI*Var1*(a2*covi22+aI*Var2)+a1*((-1)*a2*covi12**2+a2* \
  covi11*covi22+aI*covi11*Var2))**(-1)
    cond101_1=a1*aI*(a3*(covi33*(i1-l1)+covi13*(l3-i3))+ \
  aI*(i1-l1)*Var3)*(aI*Var1*(a3*covi33+aI*Var3)+a1*((-1)*a3*covi13**2+ \
  a3*covi11*covi33+aI*covi11*Var3))**(-1)
    cond101_2=l2+(-1)*(a1*a3*((-1)*covi13**2*i2+covi12*covi33*(l1-i1)+ \
  covi13*(covi23*(i1-l1)+covi12*(i3-l3))+covi11*(covi33* \
  i2+(-1)*covi23*i3+covi23*l3))+a3*aI*(covi33*i2+ \
  covi23*(l3-i3))*Var1+aI*(a1*(covi11*i2+covi12*(l1-i1))+aI*i2*Var1)* \
  Var3)*(aI*Var1*(a3*covi33+aI*Var3)+a1*((-1)*a3*covi13**2+a3* \
  covi11*covi33+aI*covi11*Var3))**(-1)
    cond101_3=a3*aI*(a1*(covi13*(l1-i1)+covi11*(i3-l3))+ \
  aI*(i3-l3)*Var1)*(aI*Var1*(a3*covi33+aI*Var3)+a1*((-1)*a3*covi13**2+ \
  a3*covi11*covi33+aI*covi11*Var3))**(-1)
    cond100_1=a1*aI*(i1-l1)*(a1*covi11+aI*Var1)**(-1)
    cond100_2=-i2+l2+a1*covi12*(i1-l1)*(a1*covi11+aI*Var1)**(-1)
    cond100_3=-i3+l3+a1*covi13*(i1-l1)*(a1*covi11+aI*Var1)**(-1)
    cond011_1=l1+(-1)*(a2*a3*((-1)*covi23**2*i1+covi12*covi33*(l2-i2)+ \
  covi23*(covi13*(i2-l2)+covi12*(i3-l3))+covi22*(covi33* \
  i1+(-1)*covi13*i3+covi13*l3))+a3*aI*(covi33*i1+ \
  covi13*(l3-i3))*Var2+aI*(a2*(covi22*i1+covi12*(l2-i2))+aI*i1*Var2)* \
  Var3)*(aI*Var2*(a3*covi33+aI*Var3)+a2*((-1)*a3*covi23**2+a3* \
  covi22*covi33+aI*covi22*Var3))**(-1)
    cond011_2=a2*aI*(a3*(covi33*(i2-l2)+covi23*(l3-i3))+ \
  aI*(i2-l2)*Var3)*(aI*Var2*(a3*covi33+aI*Var3)+a2*((-1)*a3*covi23**2+ \
  a3*covi22*covi33+aI*covi22*Var3))**(-1)
    cond011_3=a3*aI*(a2*(covi23*(l2-i2)+covi22*(i3-l3))+ \
  aI*(i3-l3)*Var2)*(aI*Var2*(a3*covi33+aI*Var3)+a2*((-1)*a3*covi23**2+ \
  a3*covi22*covi33+aI*covi22*Var3))**(-1)
    cond010_1=-i1+l1+a2*covi12*(i2-l2)*(a2*covi22+aI*Var2)**(-1)
    cond010_2=a2*aI*(i2-l2)*(a2*covi22+aI*Var2)**(-1)
    cond010_3=-i3+l3+a2*covi23*(i2-l2)*(a2*covi22+aI*Var2)**(-1)
    cond001_1=-i1+l1+a3*covi13*(i3-l3)*(a3*covi33+aI*Var3)**(-1)
    cond001_2=-i2+l2+a3*covi23*(i3-l3)*(a3*covi33+aI*Var3)**(-1)
    cond001_3=a3*aI*(i3-l3)*(a3*covi33+aI*Var3)**(-1)
    if 0 < cond111_1 and 0 < cond111_2 and 0 < cond111_3:
        curr_rate=aI*(2*a1*(a2*a3*(covi13**2*covi22+(-2)*covi12*covi13*covi23+ \
  covi11*covi23**2+covi12**2*covi33+(-1)*covi11*covi22*covi33)+a2* \
  aI*(covi12**2+(-1)*covi11*covi22)*Var3+(-1)*aI*Var2*((-1)*a3* \
  covi13**2+a3*covi11*covi33+aI*covi11*Var3))+(-2)*aI*Var1*(aI* \
  Var2*(a3*covi33+aI*Var3)+a2*((-1)*a3*covi23**2+a3*covi22*covi33+ \
  aI*covi22*Var3)))**(-1)*(2*a1*a2*a3*covi12*covi13*(i2-l2) \
  *(l3-i3)+2*a2*a3*covi23*(i2-l2)*(i3-l3)*(a1* \
  covi11+aI*Var1)+2*a1*a3*(l1-i1)*(a2*covi12*covi33*(l2-i2)\
  +(-1)*covi13*(i3-l3)*(a2*covi22+aI*Var2))+(-1)*a3*( \
  i3-l3)**2*(aI*Var1*(a2*covi22+aI*Var2)+a1*((-1)*a2* \
  covi12**2+a2*covi11*covi22+aI*covi11*Var2))+2*a1*a2*(i1-l1) \
  *(a3*covi23*(covi13*(l2-i2)+covi12*(l3-i3))+aI* \
  covi12*(i2-l2)*Var3)+(-1)*a1*(i1-l1)**2*(aI*Var2*( \
  a3*covi33+aI*Var3)+a2*((-1)*a3*covi23**2+a3*covi22*covi33+aI* \
  covi22*Var3))+(i2-l2)**2*(a1*a2*a3*(covi13**2+(-1)*covi11* \
  covi33)+(-1)*a2*aI*(a3*covi33*Var1+a1*covi11*Var3+aI*Var1*Var3) \
  ))
        GradI=(1/2)*((-1)*a1*a3*(a2*(covi13**2*covi22+(-2)*covi12*covi13* \
  covi23+covi11*covi23**2+covi12**2*covi33+(-1)*covi11*covi22*covi33) \
  +aI*(covi13**2+(-1)*covi11*covi33)*Var2)+a1*aI*((-1)*a2* \
  covi12**2+a2*covi11*covi22+aI*covi11*Var2)*Var3+aI*Var1*(aI* \
  Var2*(a3*covi33+aI*Var3)+a2*((-1)*a3*covi23**2+a3*covi22*covi33+ \
  aI*covi22*Var3)))**(-2)*(aI**2*Var1**2*(a3**2*aI**2*covi33*(i3-l3)**2*Var2**2 \
  +(-2)*a2*a3*aI*(i3-l3)*Var2*(a3*( \
  covi23**2+(-1)*covi22*covi33)*(i3-l3)+aI*covi23*(l2-i2) \
  *Var3)+a2**2*(a3**2*(covi23**2+(-1)*covi22*covi33)*((-1)*covi33* \
  (i2-l2)**2+(i3-l3)*(2*covi23*i2+(-1)*covi22*i3+(-2)* \
  covi23*l2+covi22*l3))+(-2)*a3*aI*(covi23**2+(-1)*covi22*covi33)* \
  (i2-l2)**2*Var3+aI**2*covi22*(i2-l2)**2*Var3**2))+(-2)* \
  a1*aI*Var1*(a3*aI**2*(i3-l3)*Var2**2*(a3*(covi13**2+(-1)* \
  covi11*covi33)*(i3-l3)+aI*covi13*(l1-i1)*Var3)+a2*aI* \
  Var2*(a3**2*(covi13*(covi23*(covi33*(i1-l1)*(i2-l2)+( \
  -3)*covi12*(i3-l3)**2)+(-1)*covi23**2*(i1-l1)*(i3-l3)\
  +covi12*covi33*(i2-l2)*(i3-l3))+covi13**2*(covi23* \
  (l2-i2)+2*covi22*(i3-l3))*(i3-l3)+covi12**2* \
  covi33*(i3-l3)**2+2*covi11*(covi23**2+(-1)*covi22*covi33)*( \
  i3-l3)**2+(-1)*covi12*covi33*(i1-l1)*(covi33*i2+(-1)* \
  covi23*i3+(-1)*covi33*l2+covi23*l3))+2*a3*aI*(covi13*(covi23*( \
  i1-l1)*(i2-l2)+(-1)*(covi22*i1+(-1)*covi12*i2+(-1)* \
  covi22*l1+covi12*l2)*(i3-l3))+(-1)*covi11*covi23*(i2-l2)*(i3-l3)\
  +(-1)*covi12*(i1-l1)*(covi33*i2+(-1)* \
  covi23*i3+(-1)*covi33*l2+covi23*l3))*Var3+(-1)*aI**2*covi12*(i1-l1)*(i2-l2)*Var3**2)\
  +a2**2*(a3**2*(covi13**2*covi22+( \
  -2)*covi12*covi13*covi23+covi12**2*covi33+covi11*(covi23**2+(-1)* \
  covi22*covi33))*(covi33*(i2-l2)**2+(2*covi23*(l2-i2)+ \
  covi22*(i3-l3))*(i3-l3))+a3*aI*(covi13**2*covi22*(i2+( \
  -1)*l2)**2+2*covi11*(covi23**2+(-1)*covi22*covi33)*(i2-l2) \
  **2+(-1)*covi12*covi23*(i1-l1)*(covi23*i2+(-1)*covi22*i3+( \
  -1)*covi23*l2+covi22*l3)+covi12**2*(i2-l2)*(2*covi33*i2+( \
  -1)*covi23*i3+(-2)*covi33*l2+covi23*l3)+covi13*((-3)*covi12* \
  covi23*(i2-l2)**2+(-1)*covi22**2*(i1-l1)*(i3-l3)+ \
  covi22*(i2-l2)*(covi23*i1+covi12*i3+(-1)*covi23*l1+(-1)* \
  covi12*l3)))*Var3+aI**2*(covi12**2+(-1)*covi11*covi22)*(i2-l2)**2*Var3**2))\
  +a1**2*(aI**2*Var2**2*(a3**2*(covi13**2+(-1)* \
  covi11*covi33)*((-1)*covi33*(i1-l1)**2+(i3-l3)*(2* \
  covi13*i1+(-1)*covi11*i3+(-2)*covi13*l1+covi11*l3))+(-2)*a3*aI* \
  (covi13**2+(-1)*covi11*covi33)*(i1-l1)**2*Var3+aI**2*covi11* \
  (i1-l1)**2*Var3**2)+2*a2*aI*Var2*(a3**2*(covi13**2*covi22+( \
  -2)*covi12*covi13*covi23+covi12**2*covi33+covi11*(covi23**2+(-1)* \
  covi22*covi33))*((-1)*covi33*(i1-l1)**2+(i3-l3)*(2* \
  covi13*(i1-l1)+covi11*(l3-i3)))+(-1)*a3*aI*(2* \
  covi12**2*covi33*(i1-l1)**2+covi13**2*(i1-l1)*(2* \
  covi22*(i1-l1)+covi12*(l2-i2))+covi11*(i1-l1)*( \
  covi23**2*(i1-l1)+2*covi22*covi33*(l1-i1)+covi12* \
  covi23*(i3-l3))+covi13*(covi11*covi23*(i1-l1)*(i2-l2)\
  +covi12*((-3)*covi23*(i1-l1)**2+covi11*(i2-l2)*( \
  i3-l3))+(-1)*covi12**2*(i1-l1)*(i3-l3))+(-1)* \
  covi11**2*covi23*(i2-l2)*(i3-l3))*Var3+(-1)*aI**2*( \
  covi12**2+(-1)*covi11*covi22)*(i1-l1)**2*Var3**2)+a2**2*( \
  a3**2*(covi13**2*covi22+(-2)*covi12*covi13*covi23+covi12**2* \
  covi33+covi11*(covi23**2+(-1)*covi22*covi33))*(2*covi12*covi33* \
  i1*i2+covi13**2*i2**2+(-1)*covi11*covi33*i2**2+(-2)*covi12* \
  covi13*i2*i3+covi12**2*i3**2+covi23**2*(i1-l1)**2+(-2)* \
  covi12*covi33*i2*l1+(-2)*covi12*covi33*i1*l2+(-2)*covi13**2* \
  i2*l2+2*covi11*covi33*i2*l2+2*covi12*covi13*i3*l2+2*covi12* \
  covi33*l1*l2+covi13**2*l2**2+(-1)*covi11*covi33*l2**2+2*covi23*( \
  (-1)*covi13*(i1-l1)*(i2-l2)+(-1)*(covi12*i1+(-1)* \
  covi11*i2+(-1)*covi12*l1+covi11*l2)*(i3-l3))+(-2)*covi12*( \
  covi12*i3+covi13*(l2-i2))*l3+covi12**2*l3**2+(-1)*covi22*( \
  covi33*(i1-l1)**2+(i3-l3)*((-2)*covi13*i1+covi11*i3+2* \
  covi13*l1+(-1)*covi11*l3)))+(-2)*a3*aI*(covi13**2*covi22+(-2)* \
  covi12*covi13*covi23+covi12**2*covi33+covi11*(covi23**2+(-1)* \
  covi22*covi33))*(covi22*(i1-l1)**2+(2*covi12*(l1-i1)+ \
  covi11*(i2-l2))*(i2-l2))*Var3+aI**2*(covi12**2+(-1)* \
  covi11*covi22)*((-1)*covi22*(i1-l1)**2+(i2-l2)*(2* \
  covi12*(i1-l1)+covi11*(l2-i2)))*Var3**2)))
        Grads=np.array([(1/2)*aI**2*Var1*(a2*a3*(covi23**2*(l1-i1)+covi12*covi33* \
  (l2-i2)+covi23*(covi13*(i2-l2)+covi12*(i3-l3))+ \
  covi22*(covi33*i1+(-1)*covi13*i3+(-1)*covi33*l1+covi13*l3))+a3* \
  aI*(covi33*(i1-l1)+covi13*(l3-i3))*Var2+aI*(a2*( \
  covi22*(i1-l1)+covi12*(l2-i2))+aI*(i1-l1)*Var2)* \
  Var3)**2*((-1)*a1*a3*(a2*(covi13**2*covi22+(-2)*covi12*covi13* \
  covi23+covi11*covi23**2+covi12**2*covi33+(-1)*covi11*covi22*covi33) \
  +aI*(covi13**2+(-1)*covi11*covi33)*Var2)+a1*aI*((-1)*a2* \
  covi12**2+a2*covi11*covi22+aI*covi11*Var2)*Var3+aI*Var1*(aI* \
  Var2*(a3*covi33+aI*Var3)+a2*((-1)*a3*covi23**2+a3*covi22*covi33+ \
  aI*covi22*Var3)))**(-2),(1/2)*aI**2*Var2*(a1*a3*(covi12*covi33* \
  (l1-i1)+covi13**2*(l2-i2)+covi13*(covi23*(i1-l1)+ \
  covi12*(i3-l3))+covi11*(covi33*i2+(-1)*covi23*i3+(-1)* \
  covi33*l2+covi23*l3))+a3*aI*(covi33*(i2-l2)+covi23*((-1)* \
  i3+l3))*Var1+aI*(a1*(covi12*(l1-i1)+covi11*(i2-l2))+ \
  aI*(i2+(-1)*l2)*Var1)*Var3)**2*((-1)*a1*a3*(a2*(covi13**2* \
  covi22+(-2)*covi12*covi13*covi23+covi11*covi23**2+covi12**2*covi33+ \
  (-1)*covi11*covi22*covi33)+aI*(covi13**2+(-1)*covi11*covi33)* \
  Var2)+a1*aI*((-1)*a2*covi12**2+a2*covi11*covi22+aI*covi11*Var2) \
  *Var3+aI*Var1*(aI*Var2*(a3*covi33+aI*Var3)+a2*((-1)*a3* \
  covi23**2+a3*covi22*covi33+aI*covi22*Var3)))**(-2),(1/2)*aI**2*( \
  a1*a2*(covi12*covi23*(i1-l1)+covi13*(covi22*(l1-i1)+ \
  covi12*(i2-l2))+covi12**2*(l3-i3)+covi11*((-1)*covi23* \
  i2+covi22*i3+covi23*l2+(-1)*covi22*l3))+a2*aI*(covi23*((-1)*i2+ \
  l2)+covi22*(i3-l3))*Var1+aI*(a1*(covi13*(l1-i1)+ \
  covi11*(i3-l3))+aI*(i3-l3)*Var1)*Var2)**2*Var3*((-1)* \
  a1*a3*(a2*(covi13**2*covi22+(-2)*covi12*covi13*covi23+covi11* \
  covi23**2+covi12**2*covi33+(-1)*covi11*covi22*covi33)+aI*( \
  covi13**2+(-1)*covi11*covi33)*Var2)+a1*aI*((-1)*a2*covi12**2+a2* \
  covi11*covi22+aI*covi11*Var2)*Var3+aI*Var1*(aI*Var2*(a3*covi33+ \
  aI*Var3)+a2*((-1)*a3*covi23**2+a3*covi22*covi33+aI*covi22*Var3)) \
  )**(-2)])
    elif (0 < cond110_1 and 0 < cond110_2 and 0 <= cond110_3 ) or (0 < cond111_1 and 0 < cond111_2 and 0 == cond111_3 ):
        curr_rate=(1/2)*aI*(2*a1*a2*covi12*(l1-i1)*(i2-l2)+\
  a2*(i2-l2)**2*(a1*covi11+aI*Var1)+a1*(i1-l1)**2*(a2*covi22+aI* \
  Var2))*(aI*Var1*(a2*covi22+aI*Var2)+a1*((-1)*a2*covi12**2+a2* \
  covi11*covi22+aI*covi11*Var2))**(-1)
        GradI=(1/2)*(aI*Var1*(a2*covi22+aI*Var2)+a1*((-1)*a2*covi12**2+a2* \
  covi11*covi22+aI*covi11*Var2))**(-2)*(a1**2*a2**2*(covi12**2+(-1) \
  *covi11*covi22)*(i2-l2)*(2*covi12*(i1-l1)+covi11*(l2-i2))\
  +a2**2*aI*(i2-l2)**2*Var1*((-2)*a1*covi12**2+2* \
  a1*covi11*covi22+aI*covi22*Var1)+2*a1*a2*aI**2*covi12*(i1-l1)\
  *(i2-l2)*Var1*Var2+a1**2*(i1-l1)**2*(a2**2* \
  covi22*((-1)*covi12**2+covi11*covi22)+(-2)*a2*aI*(covi12**2+(-1)* \
  covi11*covi22)*Var2+aI**2*covi11*Var2**2))
        Grads=np.array([(1/2)*aI**2*Var1*(a2*(covi22*(i1-l1)+covi12*(l2-i2))+ \
  aI*(i1-l1)*Var2)**2*(aI*Var1*(a2*covi22+aI*Var2)+a1*((-1) \
  *a2*covi12**2+a2*covi11*covi22+aI*covi11*Var2))**(-2),(1/2)* \
  aI**2*(a1*(covi12*(i1-l1)+covi11*(l2-i2))+\
  aI*(l2-i2)*Var1)**2*Var2*(aI*Var1*(a2*covi22+aI*Var2)+a1*((-1)*a2* \
  covi12**2+a2*covi11*covi22+aI*covi11*Var2))**(-2),0])

    elif (0 < cond101_1 and 0 <= cond101_2 and 0 < cond101_3 ) or (0 < cond111_1 and 0 == cond111_2 and 0 < cond111_3 ):
        curr_rate=(1/2)*aI*(2*a1*a3*covi13*(l1-i1)*(i3-l3)+\
  a3*(i3-l3)**2*(a1*covi11+aI*Var1)+a1*(i1-l1)**2*(a3*covi33+aI* \
  Var3))*(aI*Var1*(a3*covi33+aI*Var3)+a1*((-1)*a3*covi13**2+a3* \
  covi11*covi33+aI*covi11*Var3))**(-1)
        GradI=(1/2)*(aI*Var1*(a3*covi33+aI*Var3)+a1*((-1)*a3*covi13**2+a3* \
  covi11*covi33+aI*covi11*Var3))**(-2)*(a1**2*a3**2*(covi13**2+(-1) \
  *covi11*covi33)*(i3-l3)*(2*covi13*(i1-l1)+covi11*(l3-i3))\
  +a3**2*aI*(i3-l3)**2*Var1*((-2)*a1*covi13**2+2* \
  a1*covi11*covi33+aI*covi33*Var1)+2*a1*a3*aI**2*covi13\
  *(i1-l1)*(i3-l3)*Var1*Var3+a1**2*(i1-l1)**2*(a3**2* \
  covi33*((-1)*covi13**2+covi11*covi33)+(-2)*a3*aI*(covi13**2+(-1)* \
  covi11*covi33)*Var3+aI**2*covi11*Var3**2))
        Grads=np.array([(1/2)*aI**2*Var1*(a3*(covi33*(i1-l1)+covi13*(l3-i3))+ \
  aI*(i1-l1)*Var3)**2*(aI*Var1*(a3*covi33+aI*Var3)+a1*((-1) \
  *a3*covi13**2+a3*covi11*covi33+aI*covi11*Var3))**(-2),0,(1/2)* \
  aI**2*(a1*(covi13*(i1-l1)+covi11*(l3-i3))+\
  aI*(l3-i3)*Var1)**2*Var3*(aI*Var1*(a3*covi33+aI*Var3)+a1*((-1)*a3* \
  covi13**2+a3*covi11*covi33+aI*covi11*Var3))**(-2)])


    elif (0 < cond100_1 and 0 <= cond100_2 and 0 <= cond100_3 ) or (0 < cond111_1 and 0 == cond111_2 and 0 == cond111_3 ) or (0 < cond110_1 and 0 == cond110_2 and 0 <= cond110_3 ) or (0 < cond101_1 and 0 <= cond101_2 and 0 == cond101_3 ):
        curr_rate=a1*aI*(i1-l1)**2*(2*a1*covi11+2*aI*Var1)**(-1)
        GradI=(1/2)*a1**2*covi11*(i1-l1)**2*(a1*covi11+aI*Var1)**(-2)
        Grads=np.array([(1/2)*aI**2*(i1-l1)**2*Var1*(a1*covi11+aI*Var1)**(-2),0,0])
    
    elif (0 <= cond011_1 and 0 < cond011_2 and 0 < cond011_3 ) or (0 == cond111_1 and 0 < cond111_2 and 0 < cond111_3 ):
        curr_rate=(1/2)*aI*(2*a2*a3*covi23*(l2-i2)*(i3-l3)+\
  a3*(i3-l3)**2*(a2*covi22+aI*Var2)+a2*(i2-l2)**2*(a3*covi33+aI* \
  Var3))*(aI*Var2*(a3*covi33+aI*Var3)+a2*((-1)*a3*covi23**2+a3* \
  covi22*covi33+aI*covi22*Var3))**(-1)
        GradI=(1/2)*(aI*Var2*(a3*covi33+aI*Var3)+a2*((-1)*a3*covi23**2+a3* \
  covi22*covi33+aI*covi22*Var3))**(-2)*(a2**2*a3**2*(covi23**2+(-1) \
  *covi22*covi33)*(i3-l3)*(2*covi23*(i2-l2)+covi22*(l3-i3))\
  +a3**2*aI*(i3-l3)**2*Var2*((-2)*a2*covi23**2+2* \
  a2*covi22*covi33+aI*covi33*Var2)+2*a2*a3*aI**2*covi23*(i2-l2)*\
  (i3-l3)*Var2*Var3+a2**2*(i2-l2)**2*(a3**2* \
  covi33*((-1)*covi23**2+covi22*covi33)+(-2)*a3*aI*(covi23**2+(-1)* \
  covi22*covi33)*Var3+aI**2*covi22*Var3**2))
        Grads=np.array([0,(1/2)*aI**2*Var2*(a3*(covi33*(i2-l2)+covi23*(l3-i3) \
  )+aI*(i2-l2)*Var3)**2*(aI*Var2*(a3*covi33+aI*Var3)+a2*(( \
  -1)*a3*covi23**2+a3*covi22*covi33+aI*covi22*Var3))**(-2),(1/2)* \
  aI**2*(a2*(covi23*(i2-l2)+covi22*(l3-i3))+\
  aI*(l3-i3)*Var2)**2*Var3*(aI*Var2*(a3*covi33+aI*Var3)+a2*((-1)*a3* \
  covi23**2+a3*covi22*covi33+aI*covi22*Var3))**(-2)])
    
    elif (0 <= cond010_1 and 0 < cond010_2 and 0 <= cond010_3 ) or (0 == cond111_1 and 0 < cond111_2 and 0 == cond111_3 ) or (0 == cond110_1 and 0 < cond110_2 and 0 <= cond110_3 ) or (0 <= cond011_1 and 0 < cond011_2 and 0 == cond011_3 ):
        curr_rate=a2*aI*(i2-l2)**2*(2*a2*covi22+2*aI*Var2)**(-1)
        GradI=(1/2)*a2**2*covi22*(i2-l2)**2*(a2*covi22+aI*Var2)**(-2)
        Grads=np.array([0,(1/2)*aI**2*(i2-l2)**2*Var2*(a2*covi22+aI*Var2)**(-2),0])
        
    elif (0 <= cond001_1 and 0 <= cond001_2 and 0 < cond001_3 ) or (0 == cond111_1 and 0 == cond111_2 and 0 < cond111_3 ) or (0 == cond101_1 and 0 <= cond101_2 and 0 < cond101_3 ) or (0 <= cond011_1 and 0 == cond011_2 and 0 < cond011_3 ):
        curr_rate=a3*aI*(i3-l3)**2*(2*a3*covi33+2*aI*Var3)**(-1)
        GradI=(1/2)*a3**2*covi33*(i3-l3)**2*(a3*covi33+aI*Var3)**(-2)
        Grads=np.array([0,0,(1/2)*aI**2*(i3-l3)**2*Var3*(a3*covi33+aI*Var3)**(-2)])
        
    else:
        curr_rate,GradI, Grads= MCI_four_d_plus(aI,Iobj,Isig,alphs,Lobjs,Lsigs)
        
    return curr_rate, GradI, Grads


def MCI_four_d_plus(alpha_j, obj_j, cov_j, phantom_alphas, phantom_obj, phantom_var):
    """calculates phantom MCI rates
    The input variable names and the return variable names are not the same as 2d and 3d (Ziyu)
    
    parameters
    ----------
    aI: float, allocation to system I
    Iobj: numpy arra, objectives of system I (yes I know there's no need for the matrix in 1d, but this way it's consistent across cases - Nathan)
    Isig: 2d numpy array, covariance matrix of objectives for system i
    alphs: numpy array of allocations of the paretos from which the phantom pareto at hand is derived
    Lobjs: numpy array, phantom pareto objectives
    Lsigs: numpy array, phantom pareto variances (phantom objectives are treated as independent)

    returns
    -------
    curr_rate:MCI event decay rate (phantom approximation) given aI and alphs
    gradI: gradient of curr_rate wrt system I
    Grads: gradient of curr_rate wrt alphs"""
    
    #zero var could break the optimizer, replace with tiny var
    phantom_var[phantom_var==0] = 10**-100
    
    n_obj = len(obj_j)
    
    inv_cov_j = np.linalg.inv(cov_j)
    
    #TODO vectorize?
    
    P =  alpha_j*inv_cov_j
    
    q = -1*alpha_j*inv_cov_j @ obj_j
    
    G = np.identity(n_obj)
    
    h = np.ones(n_obj)*np.inf
    
    #add phantom data, looping through objectives
    
    indices = []
    
    for p in range(n_obj):
        G_vector = np.zeros([n_obj,1])
        if phantom_obj[p]< np.inf:
            indices = indices + [p]
            G_vector[p] = -1.0
            P = linalg.block_diag(P, phantom_alphas[p]*phantom_var[p]**-1)
            q = np.append(q, -1*phantom_alphas[p]*(phantom_var[p]**-1)*phantom_obj[p])
            G = np.append(G,G_vector,axis=1)
            h[p] = 0
    
    P = matrix( (P + P.transpose())/2)
    
    q = matrix(q)
    
    G = matrix(G)
    
    h = matrix(h)
    
    solvers.options['show_progress'] = False 
    x_star = np.array(solvers.qp(P,q,G,h)['x']).flatten()

    rate = 0.5*alpha_j*(obj_j - x_star[0:n_obj]).transpose() @ inv_cov_j @ (obj_j - x_star[0:n_obj])
    
    grad_j = 0.5*( x_star[0:n_obj]-obj_j).transpose() @ inv_cov_j @ ( x_star[0:n_obj]-obj_j)
    

    phantom_grads = np.zeros(n_obj)
    
      
    
    for o in range(len(indices)):
        ind = indices[o]
        rate =  rate + 0.5*phantom_alphas[ind]*(x_star[n_obj + o] - phantom_obj[ind]).transpose() * (phantom_var[ind]**-1) *(x_star[n_obj + o] - phantom_obj[ind])
        phantom_grads[o] = 0.5*(x_star[n_obj + o] - phantom_obj[ind]).transpose() * (phantom_var[ind]**-1) *(x_star[n_obj + o] - phantom_obj[ind])
        
    return rate, grad_j, phantom_grads






