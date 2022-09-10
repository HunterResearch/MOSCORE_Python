#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 18:23:56 2019

@author: nathangeldner
"""
import scipy.linalg as linalg
from cvxopt import matrix, solvers
import numpy as np

def MCE_2d(aI,aJ,Iobj,Isig,Jobj,Jsig, inv_var_i, inv_var_j):
    """calculates MCE constraint values and gradients
    
    parameters
    ----------
    aI: float, allocation to system i
    aJ: float, allocation to system j
    Iobj: numpy array, objective vals of system i
    Isig: 2d numpy array, covariance matrix of objectives of system i
    Jobj: numpy array, objective vals of system j
    Jsig: 2d numpy array, covariance martrix of objectives of system j
    inv_var_i: 2d numpy array, inverse of Isig (precomputed for efficiency)
    inv_var_j: 2d numpy array, inverse of Jsig (precomputed for efficiency)
            
    returns
    -------
    curr_rate: float, decay rate of MCE event between systems i and j
    grad_i: numpy array, gradient of rate wrt alpha_i
    grad_j: numpy array, gradient of rate wrt alpha_j"""
                
    i1=Iobj[0]
    i2=Iobj[1]
    j1=Jobj[0]
    j2=Jobj[1]
    covj11=Jsig[0,0]
    covj12=Jsig[0,1]
    covj22=Jsig[1,1]
    covi11=Isig[0,0]
    covi12=Isig[0,1]
    covi22=Isig[1,1]
    
    cond11_1=aI*aJ*(aJ**2*((-1)*covi12**2+covi11*covi22)+aI*aJ*(covi22* \
                    covj11+(-2)*covi12*covj12+covi11*covj22)+aI**2*((-1)*covj12**2+ \
                           covj11*covj22))**(-1)*((aJ*covi22+aI*covj22)*(j1-i1)+(aJ* \
                                           covi12+aI*covj12)*(i2-j2))
    
    cond11_2=aI*aJ*(aJ**2*((-1)*covi12**2+covi11*covi22)+aI*aJ*(covi22* \
                    covj11+(-2)*covi12*covj12+covi11*covj22)+aI**2*((-1)*covj12**2+ \
                           covj11*covj22))**(-1)*((aJ*covi12+aI*covj12)*(i1-j1)+(aJ* \
                                           covi11+aI*covj11)*(j2-i2))
    cond10_1=aI*aJ*(aJ*covi11+aI*covj11)**(-1)*(j1-i1)
    cond10_2=i2-j2+(aJ*covi11+aI*covj11)**(-1)*(aJ*covi12+aI*covj12)*(j1-i1)
    cond01_1=i1-j1+(aJ*covi12+aI*covj12)*(aJ*covi22+aI*covj22)**(-1)*(j2-i2)
    cond01_2=aI*aJ*(aJ*covi22+aI*covj22)**(-1)*(j2-i2)
    
    if 0 < cond11_1 and 0 < cond11_2:
        curr_rate=(1/2)*aI*aJ*(aJ**2*((-1)*covi12**2+covi11*covi22)+aI*aJ*( \
                   covi22*covj11+(-2)*covi12*covj12+covi11*covj22)+aI**2*((-1)* \
                                 covj12**2+covj11*covj22))**(-1)*((aJ*covi22+aI*covj22)*(i1-j1)**2+\
        2*(aJ*covi12+aI*covj12)*(j1-i1)*(i2-j2)+(aJ*covi11+aI*covj11)*(i2-j2)**2)
        
        GradI=(1/2)*aJ**2*(aJ**2*(covi12**2+(-1)*covi11*covi22)+(-1)*aI*aJ*( \
               covi22*covj11+(-2)*covi12*covj12+covi11*covj22)+aI**2*(covj12**2+( \
                             -1)*covj11*covj22))**(-2)*(aJ**2*(covi12**2+(-1)*covi11*covi22)*( \
                             (-1)*covi22*(i1-j1)**2+(i2-j2)*(2*covi12*i1+(-1)* \
                              covi11*i2+(-2)*covi12*j1+covi11*j2))+(-2)*aI*aJ*(covi12**2+(-1)* \
                              covi11*covi22)*(covj22*(i1-j1)**2+(i2-j2)*((-2)* \
                                              covj12*i1+covj11*i2+2*covj12*j1+(-1)*covj11*j2))+aI**2*(covi22*( \
                                                                              covj12*(i1-j1)+covj11*(j2-i2))**2+(covj22*(i1-j1)+ \
                                                                                     covj12*(j2-i2))*(2*covi12*((-1)*covj12*i1+covj11*i2+ \
                                                                                             covj12*j1+(-1)*covj11*j2)+covi11*(covj22*i1+(-1)*covj12*i2+(-1)* \
                                                                                                       covj22*j1+covj12*j2))))
                             
        GradJ=(1/2)*aI**2*(aJ**2*(covi12**2+(-1)*covi11*covi22)+(-1)*aI*aJ*( \
               covi22*covj11+(-2)*covi12*covj12+covi11*covj22)+aI**2*(covj12**2+( \
                             -1)*covj11*covj22))**(-2)*((-2)*aI*aJ*(covj12**2+(-1)*covj11* \
                             covj22)*(covi22*(i1-j1)**2+(2*covi12*(j1-i1)+covi11*( \
                                      i2-j2))*(i2-j2))+aI**2*(covj12**2+(-1)*covj11*covj22)*( \
                                      (-1)*covj22*(i1-j1)**2+(i2-j2)*(2*covj12*(i1-j1)+ \
                                       covj11*(j2-i2)))+aJ**2*(covi22**2*covj11*(i1-j1)**2+(-2) \
                                       *covi11*covi12*(covj22*(i1-j1)+covj12*(i2-j2))*(i2-j2)+\
                                       covi11**2*covj22*(i2-j2)**2+covi12**2*(covj22*(i1-j1)**2+\
                                                         (i2-j2)*(2*covj12*i1+covj11*i2+(-2)*covj12*j1+( \
                                                          -1)*covj11*j2))+(-2)*covi22*(i1-j1)*(covi11*covj12*(j2-i2)+\
                                                          covi12*(covj12*i1+covj11*i2+(-1)*covj12*j1+(-1)*covj11*j2))))
                                    
    elif (0 < cond10_1 and 0<= cond10_2) or (0 < cond11_1 and cond11_2 ==0):
        curr_rate=aI*aJ*(2*aJ*covi11+2*aI*covj11)**(-1)*(i1-j1)**2
        GradI=(1/2)*aJ**2*covi11*(aJ*covi11+aI*covj11)**(-2)*(i1-j1)**2
        GradJ=(1/2)*aI**2*covj11*(aJ*covi11+aI*covj11)**(-2)*(i1-j1)**2
        
    elif (0<=cond01_1 and 0< cond01_2) or (cond11_1 == 0 and 0<cond11_2):
        curr_rate=aI*aJ*(2*aJ*covi22+2*aI*covj22)**(-1)*(i2-j2)**2
        GradI=(1/2)*aJ**2*covi22*(aJ*covi22+aI*covj22)**(-2)*(i2-j2)**2
        GradJ=(1/2)*aI**2*covj22*(aJ*covi22+aI*covj22)**(-2)*(i2-j2)**2
        
    else:
        print('Calling Quardprog (MCE-2)')
        curr_rate, GradI, GradJ = MCE_four_d_plus(aI, aJ, Iobj, inv_var_i, Jobj, inv_var_j, 2)
            
    return curr_rate, GradI, GradJ

def MCE_3d(aI,aJ,Iobj,Isig,Jobj,Jsig, inv_var_i, inv_var_j):
    """calculates MCE constraint values and gradients

    parameters
    ----------
    aI: float, allocation to system i
    aJ: float, allocation to system j
    Iobj: numpy array, objective vals of system i
    Isig: 2d numpy array, covariance matrix of objectives of system i
    Jobj: numpy array, objective vals of system j
    Jsig: 2d numpy array, covariance martrix of objectives of system j
    inv_var_i: 2d numpy array, inverse of Isig (precomputed for efficiency)
    inv_var_j: 2d numpy array, inverse of Jsig (precomputed for efficiency)

    returns
    -------
    curr_rate: float, decay rate of MCE event between systems i and j
    grad_i: numpy array, gradient of rate wrt alpha_i
    grad_j: numpy array, gradient of rate wrt alpha_j"""
    
    i1=Iobj[0]
    i2=Iobj[1]
    i3=Iobj[2]
    j1=Jobj[0]
    j2=Jobj[1]
    j3=Jobj[2]
    covj11=Jsig[0,0]
    covj12=Jsig[0,1]
    covj22=Jsig[1,1]
    covj13=Jsig[0,2]
    covj23=Jsig[1,2]
    covj33=Jsig[2,2]
    covi11=Isig[0,0]
    covi12=Isig[0,1]
    covi22=Isig[1,1]
    covi13=Isig[0,2]
    covi23=Isig[1,2]
    covi33=Isig[2,2]
    
    cond111_1=aI*aJ*(aJ**3*(covi13**2*covi22+(-2)*covi12*covi13*covi23+ \
                            covi12**2*covi33+covi11*(covi23**2+(-1)*covi22*covi33))+aI**2*aJ*( \
                                                    (-2)*covi23*covj12*covj13+covi22*covj13**2+2*covi13*covj13* \
                                                    covj22+covi33*(covj12**2+(-1)*covj11*covj22)+2*covi23*covj11* \
                                                    covj23+(-2)*covi13*covj12*covj23+(-2)*covi12*covj13*covj23+ \
                                                    covi11*covj23**2+(-1)*(covi22*covj11+(-2)*covi12*covj12+covi11* \
                                                                      covj22)*covj33)+aI*aJ**2*(covi23**2*covj11+2*covi12*covi33* \
                                                                      covj12+covi13**2*covj22+(-1)*covi11*covi33*covj22+(-2)*covi12* \
                                                                      covi13*covj23+(-2)*covi23*(covi13*covj12+covi12*covj13+(-1)* \
                                                                                     covi11*covj23)+covi12**2*covj33+(-1)*covi22*(covi33*covj11+(-2)* \
                                                                                                                     covi13*covj13+covi11*covj33))+aI**3*(covj13**2*covj22+(-2)*covj12* \
                                                                                     covj13*covj23+covj12**2*covj33+covj11*(covj23**2+(-1)*covj22* \
                                                                                                                            covj33)))**(-1)*(aJ**2*(covi23**2*(j1-i1)+covi12*covi33*(j2-i2)+\
                                                                                                                            covi23*(covi13*(i2-j2)+covi12*(i3-j3))+covi22*( \
                                                                                                                                   covi33*i1+(-1)*covi13*i3+(-1)*covi33*j1+covi13*j3))+aI*aJ*( \
                                                                                                                                   covi22*covj33*i1+covi13*covj23*i2+(-1)*covi12*covj33*i2+(-1)* \
                                                                                                                                   covi22*covj13*i3+(-1)*covi13*covj22*i3+covi12*covj23*i3+(-1)* \
                                                                                                                                   covi22*covj33*j1+(-1)*covi13*covj23*j2+covi12*covj33*j2+covi33*( \
                                                                                                                                                    covj22*(i1-j1)+covj12*(j2-i2))+covi22*covj13*j3+covi13* \
                                                                                                                                                    covj22*j3+(-1)*covi12*covj23*j3+covi23*((-2)*covj23*i1+covj13* \
                                                                                                                                                               i2+covj12*i3+2*covj23*j1+(-1)*covj13*j2+(-1)*covj12*j3))+aI**2*( \
                                                                                                                                                               covj23**2*(j1-i1)+covj12*covj33*(j2-i2)+covj23*(covj13* \
                                                                                                                                                                         (i2-j2)+covj12*(i3-j3))+covj22*(covj33*i1+(-1)*covj13* \
                                                                                                                                                                         i3+(-1)*covj33*j1+covj13*j3)))
    cond111_2=aI*aJ*(aJ**3*(covi13**2*covi22+(-2)*covi12*covi13*covi23+ \
                            covi12**2*covi33+covi11*(covi23**2+(-1)*covi22*covi33))+aI**2*aJ*( \
                                                    (-2)*covi23*covj12*covj13+covi22*covj13**2+2*covi13*covj13* \
                                                    covj22+covi33*(covj12**2+(-1)*covj11*covj22)+2*covi23*covj11* \
                                                    covj23+(-2)*covi13*covj12*covj23+(-2)*covi12*covj13*covj23+ \
                                                    covi11*covj23**2+(-1)*(covi22*covj11+(-2)*covi12*covj12+covi11* \
                                                                      covj22)*covj33)+aI*aJ**2*(covi23**2*covj11+2*covi12*covi33* \
                                                                      covj12+covi13**2*covj22+(-1)*covi11*covi33*covj22+(-2)*covi12* \
                                                                      covi13*covj23+(-2)*covi23*(covi13*covj12+covi12*covj13+(-1)* \
                                                                                     covi11*covj23)+covi12**2*covj33+(-1)*covi22*(covi33*covj11+(-2)* \
                                                                                                                     covi13*covj13+covi11*covj33))+aI**3*(covj13**2*covj22+(-2)*covj12* \
                                                                                     covj13*covj23+covj12**2*covj33+covj11*(covj23**2+(-1)*covj22* \
                                                                                                                            covj33)))**(-1)*(aJ**2*(covi12*covi33*(j1-i1)+covi13**2*(j2-i2)+\
                                                                                                                            covi13*(covi23*(i1-j1)+covi12*(i3-j3))+covi11*( \
                                                                                                                                   covi33*i2+(-1)*covi23*i3+(-1)*covi33*j2+covi23*j3))+aI*aJ*( \
                                                                                                                                   covi13*covj23*i1+(-1)*covi12*covj33*i1+(-2)*covi13*covj13*i2+ \
                                                                                                                                   covi11*covj33*i2+covi13*covj12*i3+covi12*covj13*i3+(-1)*covi11* \
                                                                                                                                   covj23*i3+(-1)*covi13*covj23*j1+covi12*covj33*j1+covi33*(covj12* \
                                                                                                                                             (j1-i1)+covj11*(i2-j2))+2*covi13*covj13*j2+(-1)* \
                                                                                                                                             covi11*covj33*j2+(-1)*covi13*covj12*j3+(-1)*covi12*covj13*j3+ \
                                                                                                                                             covi11*covj23*j3+covi23*(covj13*i1+(-1)*covj11*i3+(-1)*covj13* \
                                                                                                                                                                      j1+covj11*j3))+aI**2*(covj12*covj33*(j1-i1)+covj13**2*(j2-i2)+\
                                                                                                                                                                      covj13*(covj23*(i1-j1)+covj12*(i3-j3))+covj11*( \
                                                                                                                                                                             covj33*i2+(-1)*covj23*i3+(-1)*covj33*j2+covj23*j3)))
    cond111_3=aI*aJ*(aJ**3*(covi13**2*covi22+(-2)*covi12*covi13*covi23+ \
                            covi12**2*covi33+covi11*(covi23**2+(-1)*covi22*covi33))+aI**2*aJ*( \
                                                    (-2)*covi23*covj12*covj13+covi22*covj13**2+2*covi13*covj13* \
                                                    covj22+covi33*(covj12**2+(-1)*covj11*covj22)+2*covi23*covj11* \
                                                    covj23+(-2)*covi13*covj12*covj23+(-2)*covi12*covj13*covj23+ \
                                                    covi11*covj23**2+(-1)*(covi22*covj11+(-2)*covi12*covj12+covi11* \
                                                                      covj22)*covj33)+aI*aJ**2*(covi23**2*covj11+2*covi12*covi33* \
                                                                      covj12+covi13**2*covj22+(-1)*covi11*covi33*covj22+(-2)*covi12* \
                                                                      covi13*covj23+(-2)*covi23*(covi13*covj12+covi12*covj13+(-1)* \
                                                                                     covi11*covj23)+covi12**2*covj33+(-1)*covi22*(covi33*covj11+(-2)* \
                                                                                                                     covi13*covj13+covi11*covj33))+aI**3*(covj13**2*covj22+(-2)*covj12* \
                                                                                     covj13*covj23+covj12**2*covj33+covj11*(covj23**2+(-1)*covj22* \
                                                                                                                            covj33)))**(-1)*(aJ**2*(covi12*covi23*(i1-j1)+covi13*( \
                                                                                                                                       covi22*(j1-i1)+covi12*(i2-j2))+covi12**2*(j3-i3)+ \
                                                                                                                            covi11*((-1)*covi23*i2+covi22*i3+covi23*j2+(-1)*covi22*j3))+aI* \
                                                                                                                            aJ*((-1)*covi13*covj22*i1+covi12*covj23*i1+covi13*covj12*i2+ \
                                                                                                                                covi12*covj13*i2+(-1)*covi11*covj23*i2+(-2)*covi12*covj12*i3+ \
                                                                                                                                covi11*covj22*i3+covi13*covj22*j1+(-1)*covi12*covj23*j1+(-1)* \
                                                                                                                                covi13*covj12*j2+(-1)*covi12*covj13*j2+covi11*covj23*j2+covi23*( \
                                                                                                                                                 covj12*(i1-j1)+covj11*(j2-i2))+2*covi12*covj12*j3+(-1) \
                                                                                                                                                 *covi11*covj22*j3+covi22*((-1)*covj13*i1+covj11*i3+covj13*j1+( \
                                                                                                                                                                           -1)*covj11*j3))+aI**2*(covj12*covj23*(i1-j1)+covj13*( \
                                                                                                                                                 covj22*(j1-i1)+covj12*(i2-j2))+covj12**2*(j3-i3)+ \
                                                                                                                                                 covj11*((-1)*covj23*i2+covj22*i3+covj23*j2+(-1)*covj22*j3)))
    cond110_1=aI*aJ*(aJ**2*((-1)*covi12**2+covi11*covi22)+aI*aJ*(covi22* \
                     covj11+(-2)*covi12*covj12+covi11*covj22)+aI**2*((-1)*covj12**2+ \
                            covj11*covj22))**(-1)*((aJ*covi22+aI*covj22)*(j1-i1)+(aJ* \
                                             covi12+aI*covj12)*(i2-j2))
    
    cond110_2=aI*aJ*(aJ**2*((-1)*covi12**2+covi11*covi22)+aI*aJ*(covi22* \
                     covj11+(-2)*covi12*covj12+covi11*covj22)+aI**2*((-1)*covj12**2+ \
                            covj11*covj22))**(-1)*((aJ*covi12+aI*covj12)*(i1-j1)+(aJ* \
                                             covi11+aI*covj11)*(j2-i2))
    
    cond110_3=(aJ**2*(covi12**2+(-1)*covi11*covi22)+(-1)*aI*aJ*(covi22*covj11+( \
               -2)*covi12*covj12+covi11*covj22)+aI**2*(covj12**2+(-1)*covj11* \
    covj22))**(-1)*((aJ*covi13+aI*covj13)*(aJ*covi22+aI*covj22)*(i1-j1)+\
    (aJ*covi12+aI*covj12)*(aJ*covi23+aI*covj23)*(j1-i1) \
    +(aJ*covi11+aI*covj11)*(aJ*covi23+aI*covj23)*(i2-j2)+(aJ* \
     covi12+aI*covj12)*(aJ*covi13+aI*covj13)*(j2-i2)+(aJ*covi12+ \
                       aI*covj12)**2*(i3-j3)+(aJ*covi11+aI*covj11)*(aJ*covi22+aI* \
                                     covj22)*(j3-i3))
    
    cond101_1=aI*aJ*(aJ**2*((-1)*covi13**2+covi11*covi33)+aI*aJ*(covi33* \
                     covj11+(-2)*covi13*covj13+covi11*covj33)+aI**2*((-1)*covj13**2+ \
                            covj11*covj33))**(-1)*((aJ*covi33+aI*covj33)*(j1-i1)+(aJ* \
                                             covi13+aI*covj13)*(i3-j3))
    
    cond101_2=(aJ**2*(covi13**2+(-1)*covi11*covi33)+(-1)*aI*aJ*(covi33*covj11+( \
               -2)*covi13*covj13+covi11*covj33)+aI**2*(covj13**2+(-1)*covj11* \
    covj33))**(-1)*((aJ*covi12+aI*covj12)*(aJ*covi33+aI*covj33)*(i1-j1)+\
    (aJ*covi13+aI*covj13)*(aJ*covi23+aI*covj23)*(j1-i1) \
    +(aJ*covi13+aI*covj13)**2*(i2-j2)+(aJ*covi11+aI*covj11)*( \
     aJ*covi33+aI*covj33)*(j2-i2)+(aJ*covi11+aI*covj11)*(aJ* \
                          covi23+aI*covj23)*(i3-j3)+(aJ*covi12+aI*covj12)*(aJ*covi13+ \
                                            aI*covj13)*(j3-i3))
    
    cond101_3=aI*aJ*(aJ**2*((-1)*covi13**2+covi11*covi33)+aI*aJ*(covi33* \
                     covj11+(-2)*covi13*covj13+covi11*covj33)+aI**2*((-1)*covj13**2+ \
                            covj11*covj33))**(-1)*((aJ*covi13+aI*covj13)*(i1-j1)+(aJ* \
                                             covi11+aI*covj11)*(j3-i3))
    
    cond100_1=aI*aJ*(aJ*covi11+aI*covj11)**(-1)*(j1-i1)
    cond100_2=i2+(aJ*covi11+aI*covj11)**(-1)*(aJ*covi12+aI*covj12)*(j1-i1)-j2
    cond100_3=i3+(aJ*covi11+aI*covj11)**(-1)*(aJ*covi13+aI*covj13)*(j1-i1)-j3
    
    cond011_1=(aJ**2*(covi23**2+(-1)*covi22*covi33)+(-1)*aI*aJ*(covi33*covj22+( \
               -2)*covi23*covj23+covi22*covj33)+aI**2*(covj23**2+(-1)*covj22* \
    covj33))**(-1)*((aJ*covi23+aI*covj23)**2*(i1-j1)+(aJ*covi22+ \
              aI*covj22)*(aJ*covi33+aI*covj33)*(j1-i1)+(aJ*covi12+aI* \
                         covj12)*(aJ*covi33+aI*covj33)*(i2-j2)+(aJ*covi13+aI*covj13) \
    *(aJ*covi23+aI*covj23)*(j2-i2)+(aJ*covi13+aI*covj13)*(aJ* \
     covi22+aI*covj22)*(i3-j3)+(aJ*covi12+aI*covj12)*(aJ*covi23+ \
                       aI*covj23)*(j3-i3))
    
    cond011_2=aI*aJ*(aJ**2*((-1)*covi23**2+covi22*covi33)+aI*aJ*(covi33* \
                     covj22+(-2)*covi23*covj23+covi22*covj33)+aI**2*((-1)*covj23**2+ \
                            covj22*covj33))**(-1)*((aJ*covi33+aI*covj33)*(j2-i2)+(aJ* \
                                             covi23+aI*covj23)*(i3-j3))
    
    cond011_3=aI*aJ*(aJ**2*((-1)*covi23**2+covi22*covi33)+aI*aJ*(covi33* \
                     covj22+(-2)*covi23*covj23+covi22*covj33)+aI**2*((-1)*covj23**2+ \
                            covj22*covj33))**(-1)*((aJ*covi23+aI*covj23)*(i2-j2)+(aJ* \
                                             covi22+aI*covj22)*(j3-i3))
    
    cond010_1=i1+(-1)*j1+(aJ*covi12+aI*covj12)*(aJ*covi22+aI*covj22)**(-1)*(j2-i2)
    cond010_2=aI*aJ*(aJ*covi22+aI*covj22)**(-1)*(j2-i2)
    cond010_3=i3+(aJ*covi22+aI*covj22)**(-1)*(aJ*covi23+aI*covj23)*(j2-i2)-j3
    cond001_1=i1-j1+(aJ*covi13+aI*covj13)*(aJ*covi33+aI*covj33)**(-1)*(j3-i3)
    cond001_2=i2-j2+(aJ*covi23+aI*covj23)*(aJ*covi33+aI*covj33)**(-1)*(j3-i3)
    cond001_3=aI*aJ*(aJ*covi33+aI*covj33)**(-1)*(j3-i3)
    
    if 0 < cond111_1 and 0<cond111_2 and 0< cond111_3:
        curr_rate=(1/2)*aI*aJ*(aJ**3*(covi13**2*covi22+(-2)*covi12*covi13*covi23+ \
                               covi12**2*covi33+covi11*(covi23**2+(-1)*covi22*covi33))+aI**2*aJ*( \
                                                       (-2)*covi23*covj12*covj13+covi22*covj13**2+2*covi13*covj13* \
                                                       covj22+covi33*(covj12**2+(-1)*covj11*covj22)+2*covi23*covj11* \
                                                       covj23+(-2)*covi13*covj12*covj23+(-2)*covi12*covj13*covj23+ \
                                                       covi11*covj23**2+(-1)*(covi22*covj11+(-2)*covi12*covj12+covi11* \
                                                                         covj22)*covj33)+aI*aJ**2*(covi23**2*covj11+2*covi12*covi33* \
                                                                         covj12+covi13**2*covj22+(-1)*covi11*covi33*covj22+(-2)*covi12* \
                                                                         covi13*covj23+(-2)*covi23*(covi13*covj12+covi12*covj13+(-1)* \
                                                                                        covi11*covj23)+covi12**2*covj33+(-1)*covi22*(covi33*covj11+(-2)* \
                                                                                                                        covi13*covj13+covi11*covj33))+aI**3*(covj13**2*covj22+(-2)*covj12* \
                                                                                        covj13*covj23+covj12**2*covj33+covj11*(covj23**2+(-1)*covj22* \
                                                                                                                               covj33)))**(-1)*((-1)*(aJ**2*((-1)*covi23**2+covi22*covi33)+aI* \
                                                                                                                               aJ*(covi33*covj22+(-2)*covi23*covj23+covi22*covj33)+aI**2*((-1)* \
                                                                                                                                   covj23**2+covj22*covj33))*(i1+(-1)*j1)**2+(aJ**2*(covi13**2+(-1)* \
                                                                                                                                                             covi11*covi33)+(-1)*aI*aJ*(covi33*covj11+(-2)*covi13*covj13+ \
                                                                                                                                                                            covi11*covj33)+aI**2*(covj13**2+(-1)*covj11*covj33))*(i2+(-1)*j2) \
                                                                                                                               **2+2*(i1+(-1)*j1)*(((-1)*(aJ*covi13+aI*covj13)*(aJ*covi23+aI* \
                                                                                                                                          covj23)+aI*aJ*covi12*covj33)*(i2+(-1)*j2)+(-1)*(aJ*(aJ*covi12* \
                                                                                                                                                                       covi23+aI*covi23*covj12+(-1)*aI*covi13*covj22)+aI**2*covj12* \
                                                                                                                               covj23)*(i3+(-1)*j3))+2*(aJ**2*((-1)*covi12*covi13+covi11*covi23) \
                                                                                                                               +aI*aJ*(covi23*covj11+covi11*covj23)+aI**2*((-1)*covj12*covj13+ \
                                                                                                                                       covj11*covj23))*(i2+(-1)*j2)*(i3+(-1)*j3)+2*aI*aJ*(covi13* \
                                                                                                                                       covj12+covi12*covj13)*((-1)*i2+j2)*(i3+(-1)*j3)+(-1)*(aJ**2*((-1) \
                                                                                                                                                              *covi12**2+covi11*covi22)+aI*aJ*(covi22*covj11+(-2)*covi12* \
                                                                                                                                                                                              covj12+covi11*covj22)+aI**2*((-1)*covj12**2+covj11*covj22))*(i3+( \
                                                                                                                                                                                                                          -1)*j3)**2+2*((-1)*i1+j1)*((-1)*(aJ**2*covi12*covi33+aI**2* \
                                                                                                                                                                                                                                       covj12*covj33)*(i2+(-1)*j2)+aI*aJ*covi33*covj12*((-1)*i2+j2)+( \
                                                                                                                                                                                                                                                      -1)*(aJ**2*covi13*covi22+aI**2*covj13*covj22+(-1)*aI*aJ*covi12* \
                                                                                                                                                                                                                                                          covj23)*(i3+(-1)*j3)+aI*aJ*covi22*covj13*((-1)*i3+j3)))
                                                                                                                               
        GradI=(1/2)*(covi13**2*covi22+(-2)*covi12*covi13*covi23+covi12**2* \
           covi33+covi11*(covi23**2+(-1)*covi22*covi33))**(-1)*(2*covi12* \
                         covi33*i1*i2+covi13**2*i2**2+(-1)*covi11*covi33*i2**2+(-2)* \
                         covi12*covi13*i2*i3+covi12**2*i3**2+(-2)*covi12*covi33*(aJ**3*( \
                                                             covi13**2*covi22+(-2)*covi12*covi13*covi23+covi12**2*covi33+ \
                                                             covi11*(covi23**2+(-1)*covi22*covi33))+aI**2*aJ*((-2)*covi23* \
                                                                    covj12*covj13+covi22*covj13**2+2*covi13*covj13*covj22+covi33*( \
                                                                                                                                  covj12**2+(-1)*covj11*covj22)+2*covi23*covj11*covj23+(-2)*covi13* \
                                                                                                                                  covj12*covj23+(-2)*covi12*covj13*covj23+covi11*covj23**2+(-1)*( \
                                                                                                                                                covi22*covj11+(-2)*covi12*covj12+covi11*covj22)*covj33)+aI*aJ**2* \
                                                                                                                                  (covi23**2*covj11+2*covi12*covi33*covj12+covi13**2*covj22+(-1)* \
                                                                                                                                   covi11*covi33*covj22+(-2)*covi12*covi13*covj23+(-2)*covi23*( \
                                                                                                                                                        covi13*covj12+covi12*covj13+(-1)*covi11*covj23)+covi12**2*covj33+( \
                                                                                                                                                                                    -1)*covi22*(covi33*covj11+(-2)*covi13*covj13+covi11*covj33))+ \
                                                                                                                                   aI**3*(covj13**2*covj22+(-2)*covj12*covj13*covj23+covj12**2* \
                                                                                                                                          covj33+covj11*(covj23**2+(-1)*covj22*covj33)))**(-1)*i2*(aI**3*( \
                                                                                                                                                        covj13**2*covj22+(-2)*covj12*covj13*covj23+covj12**2*covj33+ \
                                                                                                                                                        covj11*(covj23**2+(-1)*covj22*covj33))*i1+aJ**3*(covi13**2*covi22+ \
                                                                                                                                                               (-2)*covi12*covi13*covi23+covi12**2*covi33+covi11*(covi23**2+(-1)* \
                                                                                                                                                                covi22*covi33))*j1+aI**2*aJ*(covi22*covj13**2*i1+covi13*covj13* \
                                                                                                                                                                covj22*i1+covi33*(covj12**2+(-1)*covj11*covj22)*i1+(-1)*covi13* \
                                                                                                                                                                covj12*covj23*i1+(-1)*covi12*covj13*covj23*i1+2*covi23*((-1)* \
                                                                                                                                                                                  covj12*covj13+covj11*covj23)*i1+(-1)*covi22*covj11*covj33*i1+ \
                                                                                                                                                                                  covi12*covj12*covj33*i1+covi13*covj12*covj13*i2+(-1)*covi12* \
                                                                                                                                                                                  covj13**2*i2+(-1)*covi13*covj11*covj23*i2+covi11*covj13*covj23* \
                                                                                                                                                                                  i2+covi12*covj11*covj33*i2+(-1)*covi11*covj12*covj33*i2+(-1)* \
                                                                                                                                                                                  covi13*covj12**2*i3+covi12*covj12*covj13*i3+covi13*covj11* \
                                                                                                                                                                                  covj22*i3+(-1)*covi11*covj13*covj22*i3+(-1)*covi12*covj11* \
                                                                                                                                                                                  covj23*i3+covi11*covj12*covj23*i3+covi13*covj13*covj22*j1+(-1)* \
                                                                                                                                                                                  covi13*covj12*covj23*j1+(-1)*covi12*covj13*covj23*j1+covi11* \
                                                                                                                                                                                  covj23**2*j1+covi12*covj12*covj33*j1+(-1)*covi11*covj22*covj33* \
                                                                                                                                                                                  j1+(-1)*covi13*covj12*covj13*j2+covi12*covj13**2*j2+covi13* \
                                                                                                                                                                                  covj11*covj23*j2+(-1)*covi11*covj13*covj23*j2+(-1)*covi12* \
                                                                                                                                                                                  covj11*covj33*j2+covi11*covj12*covj33*j2+(covi13*covj12**2+(-1)* \
                                                                                                                                                                                                                            covi12*covj12*covj13+(-1)*covi13*covj11*covj22+covi11*covj13* \
                                                                                                                                                                                                                            covj22+covi12*covj11*covj23+(-1)*covi11*covj12*covj23)*j3)+aI* \
                                                                                                                                                                                  aJ**2*(covi23**2*covj11*i1+covi12*covi33*covj12*i1+covi12* \
                                                                                                                                                                                         covi33*covj11*i2+covi13**2*covj12*i2+(-1)*covi11*covi33*covj12* \
                                                                                                                                                                                         i2+(-1)*covi12*covi13*covj13*i2+(-1)*covi12*covi13*covj12*i3+ \
                                                                                                                                                                                         covi12**2*covj13*i3+covi12*covi33*covj12*j1+covi13**2*covj22*j1+( \
                                                                                                                                                                                                                                                          -1)*covi11*covi33*covj22*j1+(-2)*covi12*covi13*covj23*j1+ \
                                                                                                                                                                                                                                                          covi12**2*covj33*j1+(-1)*covi12*covi33*covj11*j2+(-1)*covi13**2* \
                                                                                                                                                                                                                                                          covj12*j2+covi11*covi33*covj12*j2+covi12*covi13*covj13*j2+ \
                                                                                                                                                                                                                                                          covi12*covi13*covj12*j3+(-1)*covi12**2*covj13*j3+(-1)*covi23*( \
                                                                                                                                                                                                                                                                                  covi13*(covj12*(i1+j1)+covj11*(i2+(-1)*j2))+covi12*(covj13*(i1+j1) \
                                                                                                                                                                                                                                                                                          +covj11*(i3+(-1)*j3))+covi11*((-1)*covj13*i2+(-1)*covj12*i3+(-2) \
                                                                                                                                                                                                                                                                                                  *covj23*j1+covj13*j2+covj12*j3))+covi22*((-1)*covi33*covj11*i1+ \
                                                                                                                                                                                                                                                                                          covi13*(covj13*(i1+j1)+covj11*(i3+(-1)*j3))+(-1)*covi11*(covj13* \
                                                                                                                                                                                                                                                                                                 i3+covj33*j1+(-1)*covj13*j3))))+(-2)*covi12*covi33*(aJ**3*( \
                                                                                                                                                                                                                                                                                          covi13**2*covi22+(-2)*covi12*covi13*covi23+covi12**2*covi33+ \
                                                                                                                                                                                                                                                                                          covi11*(covi23**2+(-1)*covi22*covi33))+aI**2*aJ*((-2)*covi23* \
                                                                                                                                                                                                                                                                                                 covj12*covj13+covi22*covj13**2+2*covi13*covj13*covj22+covi33*( \
                                                                                                                                                                                                                                                                                                                                                               covj12**2+(-1)*covj11*covj22)+2*covi23*covj11*covj23+(-2)*covi13* \
                                                                                                                                                                                                                                                                                                                                                               covj12*covj23+(-2)*covi12*covj13*covj23+covi11*covj23**2+(-1)*( \
                                                                                                                                                                                                                                                                                                                                                                             covi22*covj11+(-2)*covi12*covj12+covi11*covj22)*covj33)+aI*aJ**2* \
                                                                                                                                                                                                                                                                                                                                                               (covi23**2*covj11+2*covi12*covi33*covj12+covi13**2*covj22+(-1)* \
                                                                                                                                                                                                                                                                                                                                                                covi11*covi33*covj22+(-2)*covi12*covi13*covj23+(-2)*covi23*( \
                                                                                                                                                                                                                                                                                                                                                                                     covi13*covj12+covi12*covj13+(-1)*covi11*covj23)+covi12**2*covj33+( \
                                                                                                                                                                                                                                                                                                                                                                                                                 -1)*covi22*(covi33*covj11+(-2)*covi13*covj13+covi11*covj33))+ \
                                                                                                                                                                                                                                                                                                                                                                aI**3*(covj13**2*covj22+(-2)*covj12*covj13*covj23+covj12**2* \
                                                                                                                                                                                                                                                                                                                                                                       covj33+covj11*(covj23**2+(-1)*covj22*covj33)))**(-1)*i1*(aI**3*( \
                                                                                                                                                                                                                                                                                                                                                                                     covj13**2*covj22+(-2)*covj12*covj13*covj23+covj12**2*covj33+ \
                                                                                                                                                                                                                                                                                                                                                                                     covj11*(covj23**2+(-1)*covj22*covj33))*i2+aJ**3*(covi13**2*covi22+ \
                                                                                                                                                                                                                                                                                                                                                                                            (-2)*covi12*covi13*covi23+covi12**2*covi33+covi11*(covi23**2+(-1)* \
                                                                                                                                                                                                                                                                                                                                                                                             covi22*covi33))*j2+aI*aJ**2*(covi12*covi33*covj22*i1+covi12* \
                                                                                                                                                                                                                                                                                                                                                                                             covi33*covj12*i2+covi13**2*covj22*i2+(-1)*covi11*covi33*covj22* \
                                                                                                                                                                                                                                                                                                                                                                                             i2+(-1)*covi12*covi13*covj23*i2+(-1)*covi12*covi13*covj22*i3+ \
                                                                                                                                                                                                                                                                                                                                                                                             covi12**2*covj23*i3+(-1)*covi12*covi33*covj22*j1+covi12*covi33* \
                                                                                                                                                                                                                                                                                                                                                                                             covj12*j2+(-1)*covi12*covi13*covj23*j2+covi12**2*covj33*j2+ \
                                                                                                                                                                                                                                                                                                                                                                                             covi23**2*(covj12*(i1+(-1)*j1)+covj11*j2)+covi12*covi13*covj22* \
                                                                                                                                                                                                                                                                                                                                                                                             j3+(-1)*covi12**2*covj23*j3+covi23*((-1)*covi13*(covj22*(i1+(-1) \
                                                                                                                                                                                                                                                                                                                                                                                                                                  *j1)+covj12*(i2+j2))+covi11*(covj23*(i2+j2)+covj22*(i3+(-1)*j3))+ \
                                                                                                                                                                                                                                                                                                                                                                                             covi12*((-1)*covj23*i1+(-1)*covj12*i3+covj23*j1+(-2)*covj13*j2+ \
                                                                                                                                                                                                                                                                                                                                                                                                     covj12*j3))+(-1)*covi22*(covi33*(covj12*(i1+(-1)*j1)+covj11*j2)+ \
                                                                                                                                                                                                                                                                                                                                                                                                     covi13*((-1)*covj23*i1+(-1)*covj12*i3+covj23*j1+(-2)*covj13*j2+ \
                                                                                                                                                                                                                                                                                                                                                                                                             covj12*j3)+covi11*(covj23*i3+covj33*j2+(-1)*covj23*j3)))+aI**2* \
                                                                                                                                                                                                                                                                                                                                                                                                     aJ*((-1)*covi12*covj23**2*i1+covi12*covj22*covj33*i1+covi33* \
                                                                                                                                                                                                                                                                                                                                                                                                         covj12**2*i2+(-1)*covi33*covj11*covj22*i2+2*covi13*covj13* \
                                                                                                                                                                                                                                                                                                                                                                                                         covj22*i2+(-2)*covi13*covj12*covj23*i2+(-1)*covi12*covj13* \
                                                                                                                                                                                                                                                                                                                                                                                                         covj23*i2+covi11*covj23**2*i2+covi12*covj12*covj33*i2+(-1)* \
                                                                                                                                                                                                                                                                                                                                                                                                         covi11*covj22*covj33*i2+(-1)*covi12*covj13*covj22*i3+covi12* \
                                                                                                                                                                                                                                                                                                                                                                                                         covj12*covj23*i3+covi12*covj23**2*j1+(-1)*covi12*covj22*covj33* \
  j1+(-1)*covi12*covj13*covj23*j2+covi12*covj12*covj33*j2+covi12* \
  covj13*covj22*j3+(-1)*covi12*covj12*covj23*j3+covi23*(covj12* \
  covj23*(i1+(-1)*j1)+(-1)*covj13*(covj22*(i1+(-1)*j1)+covj12*(i2+ \
  j2))+covj11*(covj23*(i2+j2)+covj22*(i3+(-1)*j3))+covj12**2*((-1)* \
  i3+j3))+covi22*(covj12*covj33*((-1)*i1+j1)+covj13**2*j2+covj13*( \
  covj23*(i1+(-1)*j1)+covj12*(i3+(-1)*j3))+(-1)*covj11*(covj23*i3+ \
  covj33*j2+(-1)*covj23*j3))))+(-2)*covi13**2*(aJ**3*(covi13**2* \
  covi22+(-2)*covi12*covi13*covi23+covi12**2*covi33+covi11*( \
  covi23**2+(-1)*covi22*covi33))+aI**2*aJ*((-2)*covi23*covj12* \
  covj13+covi22*covj13**2+2*covi13*covj13*covj22+covi33*(covj12**2+( \
  -1)*covj11*covj22)+2*covi23*covj11*covj23+(-2)*covi13*covj12* \
  covj23+(-2)*covi12*covj13*covj23+covi11*covj23**2+(-1)*(covi22* \
  covj11+(-2)*covi12*covj12+covi11*covj22)*covj33)+aI*aJ**2*( \
  covi23**2*covj11+2*covi12*covi33*covj12+covi13**2*covj22+(-1)* \
  covi11*covi33*covj22+(-2)*covi12*covi13*covj23+(-2)*covi23*( \
  covi13*covj12+covi12*covj13+(-1)*covi11*covj23)+covi12**2*covj33+( \
  -1)*covi22*(covi33*covj11+(-2)*covi13*covj13+covi11*covj33))+ \
  aI**3*(covj13**2*covj22+(-2)*covj12*covj13*covj23+covj12**2* \
  covj33+covj11*(covj23**2+(-1)*covj22*covj33)))**(-1)*i2*(aI**3*( \
  covj13**2*covj22+(-2)*covj12*covj13*covj23+covj12**2*covj33+ \
  covj11*(covj23**2+(-1)*covj22*covj33))*i2+aJ**3*(covi13**2*covi22+ \
  (-2)*covi12*covi13*covi23+covi12**2*covi33+covi11*(covi23**2+(-1)* \
  covi22*covi33))*j2+aI*aJ**2*(covi12*covi33*covj22*i1+covi12* \
  covi33*covj12*i2+covi13**2*covj22*i2+(-1)*covi11*covi33*covj22* \
  i2+(-1)*covi12*covi13*covj23*i2+(-1)*covi12*covi13*covj22*i3+ \
  covi12**2*covj23*i3+(-1)*covi12*covi33*covj22*j1+covi12*covi33* \
  covj12*j2+(-1)*covi12*covi13*covj23*j2+covi12**2*covj33*j2+ \
  covi23**2*(covj12*(i1+(-1)*j1)+covj11*j2)+covi12*covi13*covj22* \
  j3+(-1)*covi12**2*covj23*j3+covi23*((-1)*covi13*(covj22*(i1+(-1) \
  *j1)+covj12*(i2+j2))+covi11*(covj23*(i2+j2)+covj22*(i3+(-1)*j3))+ \
  covi12*((-1)*covj23*i1+(-1)*covj12*i3+covj23*j1+(-2)*covj13*j2+ \
  covj12*j3))+(-1)*covi22*(covi33*(covj12*(i1+(-1)*j1)+covj11*j2)+ \
  covi13*((-1)*covj23*i1+(-1)*covj12*i3+covj23*j1+(-2)*covj13*j2+ \
  covj12*j3)+covi11*(covj23*i3+covj33*j2+(-1)*covj23*j3)))+aI**2* \
  aJ*((-1)*covi12*covj23**2*i1+covi12*covj22*covj33*i1+covi33* \
  covj12**2*i2+(-1)*covi33*covj11*covj22*i2+2*covi13*covj13* \
  covj22*i2+(-2)*covi13*covj12*covj23*i2+(-1)*covi12*covj13* \
  covj23*i2+covi11*covj23**2*i2+covi12*covj12*covj33*i2+(-1)* \
  covi11*covj22*covj33*i2+(-1)*covi12*covj13*covj22*i3+covi12* \
  covj12*covj23*i3+covi12*covj23**2*j1+(-1)*covi12*covj22*covj33* \
  j1+(-1)*covi12*covj13*covj23*j2+covi12*covj12*covj33*j2+covi12* \
  covj13*covj22*j3+(-1)*covi12*covj12*covj23*j3+covi23*(covj12* \
  covj23*(i1+(-1)*j1)+(-1)*covj13*(covj22*(i1+(-1)*j1)+covj12*(i2+ \
  j2))+covj11*(covj23*(i2+j2)+covj22*(i3+(-1)*j3))+covj12**2*((-1)* \
  i3+j3))+covi22*(covj12*covj33*((-1)*i1+j1)+covj13**2*j2+covj13*( \
  covj23*(i1+(-1)*j1)+covj12*(i3+(-1)*j3))+(-1)*covj11*(covj23*i3+ \
  covj33*j2+(-1)*covj23*j3))))+2*covi11*covi33*(aJ**3*(covi13**2* \
  covi22+(-2)*covi12*covi13*covi23+covi12**2*covi33+covi11*( \
  covi23**2+(-1)*covi22*covi33))+aI**2*aJ*((-2)*covi23*covj12* \
  covj13+covi22*covj13**2+2*covi13*covj13*covj22+covi33*(covj12**2+( \
  -1)*covj11*covj22)+2*covi23*covj11*covj23+(-2)*covi13*covj12* \
  covj23+(-2)*covi12*covj13*covj23+covi11*covj23**2+(-1)*(covi22* \
  covj11+(-2)*covi12*covj12+covi11*covj22)*covj33)+aI*aJ**2*( \
  covi23**2*covj11+2*covi12*covi33*covj12+covi13**2*covj22+(-1)* \
  covi11*covi33*covj22+(-2)*covi12*covi13*covj23+(-2)*covi23*( \
  covi13*covj12+covi12*covj13+(-1)*covi11*covj23)+covi12**2*covj33+( \
  -1)*covi22*(covi33*covj11+(-2)*covi13*covj13+covi11*covj33))+ \
  aI**3*(covj13**2*covj22+(-2)*covj12*covj13*covj23+covj12**2* \
  covj33+covj11*(covj23**2+(-1)*covj22*covj33)))**(-1)*i2*(aI**3*( \
  covj13**2*covj22+(-2)*covj12*covj13*covj23+covj12**2*covj33+ \
  covj11*(covj23**2+(-1)*covj22*covj33))*i2+aJ**3*(covi13**2*covi22+ \
  (-2)*covi12*covi13*covi23+covi12**2*covi33+covi11*(covi23**2+(-1)* \
  covi22*covi33))*j2+aI*aJ**2*(covi12*covi33*covj22*i1+covi12* \
  covi33*covj12*i2+covi13**2*covj22*i2+(-1)*covi11*covi33*covj22* \
  i2+(-1)*covi12*covi13*covj23*i2+(-1)*covi12*covi13*covj22*i3+ \
  covi12**2*covj23*i3+(-1)*covi12*covi33*covj22*j1+covi12*covi33* \
  covj12*j2+(-1)*covi12*covi13*covj23*j2+covi12**2*covj33*j2+ \
  covi23**2*(covj12*(i1+(-1)*j1)+covj11*j2)+covi12*covi13*covj22* \
  j3+(-1)*covi12**2*covj23*j3+covi23*((-1)*covi13*(covj22*(i1+(-1) \
  *j1)+covj12*(i2+j2))+covi11*(covj23*(i2+j2)+covj22*(i3+(-1)*j3))+ \
  covi12*((-1)*covj23*i1+(-1)*covj12*i3+covj23*j1+(-2)*covj13*j2+ \
  covj12*j3))+(-1)*covi22*(covi33*(covj12*(i1+(-1)*j1)+covj11*j2)+ \
  covi13*((-1)*covj23*i1+(-1)*covj12*i3+covj23*j1+(-2)*covj13*j2+ \
  covj12*j3)+covi11*(covj23*i3+covj33*j2+(-1)*covj23*j3)))+aI**2* \
  aJ*((-1)*covi12*covj23**2*i1+covi12*covj22*covj33*i1+covi33* \
  covj12**2*i2+(-1)*covi33*covj11*covj22*i2+2*covi13*covj13* \
  covj22*i2+(-2)*covi13*covj12*covj23*i2+(-1)*covi12*covj13* \
  covj23*i2+covi11*covj23**2*i2+covi12*covj12*covj33*i2+(-1)* \
  covi11*covj22*covj33*i2+(-1)*covi12*covj13*covj22*i3+covi12* \
  covj12*covj23*i3+covi12*covj23**2*j1+(-1)*covi12*covj22*covj33* \
  j1+(-1)*covi12*covj13*covj23*j2+covi12*covj12*covj33*j2+covi12* \
  covj13*covj22*j3+(-1)*covi12*covj12*covj23*j3+covi23*(covj12* \
  covj23*(i1+(-1)*j1)+(-1)*covj13*(covj22*(i1+(-1)*j1)+covj12*(i2+ \
  j2))+covj11*(covj23*(i2+j2)+covj22*(i3+(-1)*j3))+covj12**2*((-1)* \
  i3+j3))+covi22*(covj12*covj33*((-1)*i1+j1)+covj13**2*j2+covj13*( \
  covj23*(i1+(-1)*j1)+covj12*(i3+(-1)*j3))+(-1)*covj11*(covj23*i3+ \
  covj33*j2+(-1)*covj23*j3))))+2*covi12*covi13*(aJ**3*(covi13**2* \
  covi22+(-2)*covi12*covi13*covi23+covi12**2*covi33+covi11*( \
  covi23**2+(-1)*covi22*covi33))+aI**2*aJ*((-2)*covi23*covj12* \
  covj13+covi22*covj13**2+2*covi13*covj13*covj22+covi33*(covj12**2+( \
  -1)*covj11*covj22)+2*covi23*covj11*covj23+(-2)*covi13*covj12* \
  covj23+(-2)*covi12*covj13*covj23+covi11*covj23**2+(-1)*(covi22* \
  covj11+(-2)*covi12*covj12+covi11*covj22)*covj33)+aI*aJ**2*( \
  covi23**2*covj11+2*covi12*covi33*covj12+covi13**2*covj22+(-1)* \
  covi11*covi33*covj22+(-2)*covi12*covi13*covj23+(-2)*covi23*( \
  covi13*covj12+covi12*covj13+(-1)*covi11*covj23)+covi12**2*covj33+( \
  -1)*covi22*(covi33*covj11+(-2)*covi13*covj13+covi11*covj33))+ \
  aI**3*(covj13**2*covj22+(-2)*covj12*covj13*covj23+covj12**2* \
  covj33+covj11*(covj23**2+(-1)*covj22*covj33)))**(-1)*i3*(aI**3*( \
  covj13**2*covj22+(-2)*covj12*covj13*covj23+covj12**2*covj33+ \
  covj11*(covj23**2+(-1)*covj22*covj33))*i2+aJ**3*(covi13**2*covi22+ \
  (-2)*covi12*covi13*covi23+covi12**2*covi33+covi11*(covi23**2+(-1)* \
  covi22*covi33))*j2+aI*aJ**2*(covi12*covi33*covj22*i1+covi12* \
  covi33*covj12*i2+covi13**2*covj22*i2+(-1)*covi11*covi33*covj22* \
  i2+(-1)*covi12*covi13*covj23*i2+(-1)*covi12*covi13*covj22*i3+ \
  covi12**2*covj23*i3+(-1)*covi12*covi33*covj22*j1+covi12*covi33* \
  covj12*j2+(-1)*covi12*covi13*covj23*j2+covi12**2*covj33*j2+ \
  covi23**2*(covj12*(i1+(-1)*j1)+covj11*j2)+covi12*covi13*covj22* \
  j3+(-1)*covi12**2*covj23*j3+covi23*((-1)*covi13*(covj22*(i1+(-1) \
  *j1)+covj12*(i2+j2))+covi11*(covj23*(i2+j2)+covj22*(i3+(-1)*j3))+ \
  covi12*((-1)*covj23*i1+(-1)*covj12*i3+covj23*j1+(-2)*covj13*j2+ \
  covj12*j3))+(-1)*covi22*(covi33*(covj12*(i1+(-1)*j1)+covj11*j2)+ \
  covi13*((-1)*covj23*i1+(-1)*covj12*i3+covj23*j1+(-2)*covj13*j2+ \
  covj12*j3)+covi11*(covj23*i3+covj33*j2+(-1)*covj23*j3)))+aI**2* \
  aJ*((-1)*covi12*covj23**2*i1+covi12*covj22*covj33*i1+covi33* \
  covj12**2*i2+(-1)*covi33*covj11*covj22*i2+2*covi13*covj13* \
  covj22*i2+(-2)*covi13*covj12*covj23*i2+(-1)*covi12*covj13* \
  covj23*i2+covi11*covj23**2*i2+covi12*covj12*covj33*i2+(-1)* \
  covi11*covj22*covj33*i2+(-1)*covi12*covj13*covj22*i3+covi12* \
  covj12*covj23*i3+covi12*covj23**2*j1+(-1)*covi12*covj22*covj33* \
  j1+(-1)*covi12*covj13*covj23*j2+covi12*covj12*covj33*j2+covi12* \
  covj13*covj22*j3+(-1)*covi12*covj12*covj23*j3+covi23*(covj12* \
  covj23*(i1+(-1)*j1)+(-1)*covj13*(covj22*(i1+(-1)*j1)+covj12*(i2+ \
  j2))+covj11*(covj23*(i2+j2)+covj22*(i3+(-1)*j3))+covj12**2*((-1)* \
  i3+j3))+covi22*(covj12*covj33*((-1)*i1+j1)+covj13**2*j2+covj13*( \
  covj23*(i1+(-1)*j1)+covj12*(i3+(-1)*j3))+(-1)*covj11*(covj23*i3+ \
  covj33*j2+(-1)*covj23*j3))))+2*covi12*covi33*(aJ**3*(covi13**2* \
  covi22+(-2)*covi12*covi13*covi23+covi12**2*covi33+covi11*( \
  covi23**2+(-1)*covi22*covi33))+aI**2*aJ*((-2)*covi23*covj12* \
  covj13+covi22*covj13**2+2*covi13*covj13*covj22+covi33*(covj12**2+( \
  -1)*covj11*covj22)+2*covi23*covj11*covj23+(-2)*covi13*covj12* \
  covj23+(-2)*covi12*covj13*covj23+covi11*covj23**2+(-1)*(covi22* \
  covj11+(-2)*covi12*covj12+covi11*covj22)*covj33)+aI*aJ**2*( \
  covi23**2*covj11+2*covi12*covi33*covj12+covi13**2*covj22+(-1)* \
  covi11*covi33*covj22+(-2)*covi12*covi13*covj23+(-2)*covi23*( \
  covi13*covj12+covi12*covj13+(-1)*covi11*covj23)+covi12**2*covj33+( \
  -1)*covi22*(covi33*covj11+(-2)*covi13*covj13+covi11*covj33))+ \
  aI**3*(covj13**2*covj22+(-2)*covj12*covj13*covj23+covj12**2* \
  covj33+covj11*(covj23**2+(-1)*covj22*covj33)))**(-2)*(aI**3*( \
  covj13**2*covj22+(-2)*covj12*covj13*covj23+covj12**2*covj33+ \
  covj11*(covj23**2+(-1)*covj22*covj33))*i1+aJ**3*(covi13**2*covi22+ \
  (-2)*covi12*covi13*covi23+covi12**2*covi33+covi11*(covi23**2+(-1)* \
  covi22*covi33))*j1+aI**2*aJ*(covi22*covj13**2*i1+covi13*covj13* \
  covj22*i1+covi33*(covj12**2+(-1)*covj11*covj22)*i1+(-1)*covi13* \
  covj12*covj23*i1+(-1)*covi12*covj13*covj23*i1+2*covi23*((-1)* \
  covj12*covj13+covj11*covj23)*i1+(-1)*covi22*covj11*covj33*i1+ \
  covi12*covj12*covj33*i1+covi13*covj12*covj13*i2+(-1)*covi12* \
  covj13**2*i2+(-1)*covi13*covj11*covj23*i2+covi11*covj13*covj23* \
  i2+covi12*covj11*covj33*i2+(-1)*covi11*covj12*covj33*i2+(-1)* \
  covi13*covj12**2*i3+covi12*covj12*covj13*i3+covi13*covj11* \
  covj22*i3+(-1)*covi11*covj13*covj22*i3+(-1)*covi12*covj11* \
  covj23*i3+covi11*covj12*covj23*i3+covi13*covj13*covj22*j1+(-1)* \
  covi13*covj12*covj23*j1+(-1)*covi12*covj13*covj23*j1+covi11* \
  covj23**2*j1+covi12*covj12*covj33*j1+(-1)*covi11*covj22*covj33* \
  j1+(-1)*covi13*covj12*covj13*j2+covi12*covj13**2*j2+covi13* \
  covj11*covj23*j2+(-1)*covi11*covj13*covj23*j2+(-1)*covi12* \
  covj11*covj33*j2+covi11*covj12*covj33*j2+(covi13*covj12**2+(-1)* \
  covi12*covj12*covj13+(-1)*covi13*covj11*covj22+covi11*covj13* \
  covj22+covi12*covj11*covj23+(-1)*covi11*covj12*covj23)*j3)+aI* \
  aJ**2*(covi23**2*covj11*i1+covi12*covi33*covj12*i1+covi12* \
  covi33*covj11*i2+covi13**2*covj12*i2+(-1)*covi11*covi33*covj12* \
  i2+(-1)*covi12*covi13*covj13*i2+(-1)*covi12*covi13*covj12*i3+ \
  covi12**2*covj13*i3+covi12*covi33*covj12*j1+covi13**2*covj22*j1+( \
  -1)*covi11*covi33*covj22*j1+(-2)*covi12*covi13*covj23*j1+ \
  covi12**2*covj33*j1+(-1)*covi12*covi33*covj11*j2+(-1)*covi13**2* \
  covj12*j2+covi11*covi33*covj12*j2+covi12*covi13*covj13*j2+ \
  covi12*covi13*covj12*j3+(-1)*covi12**2*covj13*j3+(-1)*covi23*( \
  covi13*(covj12*(i1+j1)+covj11*(i2+(-1)*j2))+covi12*(covj13*(i1+j1) \
  +covj11*(i3+(-1)*j3))+covi11*((-1)*covj13*i2+(-1)*covj12*i3+(-2) \
  *covj23*j1+covj13*j2+covj12*j3))+covi22*((-1)*covi33*covj11*i1+ \
  covi13*(covj13*(i1+j1)+covj11*(i3+(-1)*j3))+(-1)*covi11*(covj13* \
  i3+covj33*j1+(-1)*covj13*j3))))*(aI**3*(covj13**2*covj22+(-2)* \
  covj12*covj13*covj23+covj12**2*covj33+covj11*(covj23**2+(-1)* \
  covj22*covj33))*i2+aJ**3*(covi13**2*covi22+(-2)*covi12*covi13* \
  covi23+covi12**2*covi33+covi11*(covi23**2+(-1)*covi22*covi33))*j2+ \
  aI*aJ**2*(covi12*covi33*covj22*i1+covi12*covi33*covj12*i2+ \
  covi13**2*covj22*i2+(-1)*covi11*covi33*covj22*i2+(-1)*covi12* \
  covi13*covj23*i2+(-1)*covi12*covi13*covj22*i3+covi12**2*covj23* \
  i3+(-1)*covi12*covi33*covj22*j1+covi12*covi33*covj12*j2+(-1)* \
  covi12*covi13*covj23*j2+covi12**2*covj33*j2+covi23**2*(covj12*( \
  i1+(-1)*j1)+covj11*j2)+covi12*covi13*covj22*j3+(-1)*covi12**2* \
  covj23*j3+covi23*((-1)*covi13*(covj22*(i1+(-1)*j1)+covj12*(i2+j2) \
  )+covi11*(covj23*(i2+j2)+covj22*(i3+(-1)*j3))+covi12*((-1)* \
  covj23*i1+(-1)*covj12*i3+covj23*j1+(-2)*covj13*j2+covj12*j3))+( \
  -1)*covi22*(covi33*(covj12*(i1+(-1)*j1)+covj11*j2)+covi13*((-1)* \
  covj23*i1+(-1)*covj12*i3+covj23*j1+(-2)*covj13*j2+covj12*j3)+ \
  covi11*(covj23*i3+covj33*j2+(-1)*covj23*j3)))+aI**2*aJ*((-1)* \
  covi12*covj23**2*i1+covi12*covj22*covj33*i1+covi33*covj12**2*i2+( \
  -1)*covi33*covj11*covj22*i2+2*covi13*covj13*covj22*i2+(-2)* \
  covi13*covj12*covj23*i2+(-1)*covi12*covj13*covj23*i2+covi11* \
  covj23**2*i2+covi12*covj12*covj33*i2+(-1)*covi11*covj22*covj33* \
  i2+(-1)*covi12*covj13*covj22*i3+covi12*covj12*covj23*i3+covi12* \
  covj23**2*j1+(-1)*covi12*covj22*covj33*j1+(-1)*covi12*covj13* \
  covj23*j2+covi12*covj12*covj33*j2+covi12*covj13*covj22*j3+(-1)* \
  covi12*covj12*covj23*j3+covi23*(covj12*covj23*(i1+(-1)*j1)+(-1)* \
  covj13*(covj22*(i1+(-1)*j1)+covj12*(i2+j2))+covj11*(covj23*(i2+j2) \
  +covj22*(i3+(-1)*j3))+covj12**2*((-1)*i3+j3))+covi22*(covj12* \
  covj33*((-1)*i1+j1)+covj13**2*j2+covj13*(covj23*(i1+(-1)*j1)+ \
  covj12*(i3+(-1)*j3))+(-1)*covj11*(covj23*i3+covj33*j2+(-1)* \
  covj23*j3))))+covi13**2*(aJ**3*(covi13**2*covi22+(-2)*covi12* \
  covi13*covi23+covi12**2*covi33+covi11*(covi23**2+(-1)*covi22* \
  covi33))+aI**2*aJ*((-2)*covi23*covj12*covj13+covi22*covj13**2+2* \
  covi13*covj13*covj22+covi33*(covj12**2+(-1)*covj11*covj22)+2* \
  covi23*covj11*covj23+(-2)*covi13*covj12*covj23+(-2)*covi12* \
  covj13*covj23+covi11*covj23**2+(-1)*(covi22*covj11+(-2)*covi12* \
  covj12+covi11*covj22)*covj33)+aI*aJ**2*(covi23**2*covj11+2* \
  covi12*covi33*covj12+covi13**2*covj22+(-1)*covi11*covi33*covj22+( \
  -2)*covi12*covi13*covj23+(-2)*covi23*(covi13*covj12+covi12* \
  covj13+(-1)*covi11*covj23)+covi12**2*covj33+(-1)*covi22*(covi33* \
  covj11+(-2)*covi13*covj13+covi11*covj33))+aI**3*(covj13**2*covj22+( \
  -2)*covj12*covj13*covj23+covj12**2*covj33+covj11*(covj23**2+(-1)* \
  covj22*covj33)))**(-2)*(aI**3*(covj13**2*covj22+(-2)*covj12* \
  covj13*covj23+covj12**2*covj33+covj11*(covj23**2+(-1)*covj22* \
  covj33))*i2+aJ**3*(covi13**2*covi22+(-2)*covi12*covi13*covi23+ \
  covi12**2*covi33+covi11*(covi23**2+(-1)*covi22*covi33))*j2+aI* \
  aJ**2*(covi12*covi33*covj22*i1+covi12*covi33*covj12*i2+ \
  covi13**2*covj22*i2+(-1)*covi11*covi33*covj22*i2+(-1)*covi12* \
  covi13*covj23*i2+(-1)*covi12*covi13*covj22*i3+covi12**2*covj23* \
  i3+(-1)*covi12*covi33*covj22*j1+covi12*covi33*covj12*j2+(-1)* \
  covi12*covi13*covj23*j2+covi12**2*covj33*j2+covi23**2*(covj12*( \
  i1+(-1)*j1)+covj11*j2)+covi12*covi13*covj22*j3+(-1)*covi12**2* \
  covj23*j3+covi23*((-1)*covi13*(covj22*(i1+(-1)*j1)+covj12*(i2+j2) \
  )+covi11*(covj23*(i2+j2)+covj22*(i3+(-1)*j3))+covi12*((-1)* \
  covj23*i1+(-1)*covj12*i3+covj23*j1+(-2)*covj13*j2+covj12*j3))+( \
  -1)*covi22*(covi33*(covj12*(i1+(-1)*j1)+covj11*j2)+covi13*((-1)* \
  covj23*i1+(-1)*covj12*i3+covj23*j1+(-2)*covj13*j2+covj12*j3)+ \
  covi11*(covj23*i3+covj33*j2+(-1)*covj23*j3)))+aI**2*aJ*((-1)* \
  covi12*covj23**2*i1+covi12*covj22*covj33*i1+covi33*covj12**2*i2+( \
  -1)*covi33*covj11*covj22*i2+2*covi13*covj13*covj22*i2+(-2)* \
  covi13*covj12*covj23*i2+(-1)*covi12*covj13*covj23*i2+covi11* \
  covj23**2*i2+covi12*covj12*covj33*i2+(-1)*covi11*covj22*covj33* \
  i2+(-1)*covi12*covj13*covj22*i3+covi12*covj12*covj23*i3+covi12* \
  covj23**2*j1+(-1)*covi12*covj22*covj33*j1+(-1)*covi12*covj13* \
  covj23*j2+covi12*covj12*covj33*j2+covi12*covj13*covj22*j3+(-1)* \
  covi12*covj12*covj23*j3+covi23*(covj12*covj23*(i1+(-1)*j1)+(-1)* \
  covj13*(covj22*(i1+(-1)*j1)+covj12*(i2+j2))+covj11*(covj23*(i2+j2) \
  +covj22*(i3+(-1)*j3))+covj12**2*((-1)*i3+j3))+covi22*(covj12* \
  covj33*((-1)*i1+j1)+covj13**2*j2+covj13*(covj23*(i1+(-1)*j1)+ \
  covj12*(i3+(-1)*j3))+(-1)*covj11*(covj23*i3+covj33*j2+(-1)* \
  covj23*j3))))**2+(-1)*covi11*covi33*(aJ**3*(covi13**2*covi22+(-2) \
  *covi12*covi13*covi23+covi12**2*covi33+covi11*(covi23**2+(-1)* \
  covi22*covi33))+aI**2*aJ*((-2)*covi23*covj12*covj13+covi22* \
  covj13**2+2*covi13*covj13*covj22+covi33*(covj12**2+(-1)*covj11* \
  covj22)+2*covi23*covj11*covj23+(-2)*covi13*covj12*covj23+(-2)* \
  covi12*covj13*covj23+covi11*covj23**2+(-1)*(covi22*covj11+(-2)* \
  covi12*covj12+covi11*covj22)*covj33)+aI*aJ**2*(covi23**2*covj11+ \
  2*covi12*covi33*covj12+covi13**2*covj22+(-1)*covi11*covi33* \
  covj22+(-2)*covi12*covi13*covj23+(-2)*covi23*(covi13*covj12+ \
  covi12*covj13+(-1)*covi11*covj23)+covi12**2*covj33+(-1)*covi22*( \
  covi33*covj11+(-2)*covi13*covj13+covi11*covj33))+aI**3*(covj13**2* \
  covj22+(-2)*covj12*covj13*covj23+covj12**2*covj33+covj11*( \
  covj23**2+(-1)*covj22*covj33)))**(-2)*(aI**3*(covj13**2*covj22+(-2) \
  *covj12*covj13*covj23+covj12**2*covj33+covj11*(covj23**2+(-1)* \
  covj22*covj33))*i2+aJ**3*(covi13**2*covi22+(-2)*covi12*covi13* \
  covi23+covi12**2*covi33+covi11*(covi23**2+(-1)*covi22*covi33))*j2+ \
  aI*aJ**2*(covi12*covi33*covj22*i1+covi12*covi33*covj12*i2+ \
  covi13**2*covj22*i2+(-1)*covi11*covi33*covj22*i2+(-1)*covi12* \
  covi13*covj23*i2+(-1)*covi12*covi13*covj22*i3+covi12**2*covj23* \
  i3+(-1)*covi12*covi33*covj22*j1+covi12*covi33*covj12*j2+(-1)* \
  covi12*covi13*covj23*j2+covi12**2*covj33*j2+covi23**2*(covj12*( \
  i1+(-1)*j1)+covj11*j2)+covi12*covi13*covj22*j3+(-1)*covi12**2* \
  covj23*j3+covi23*((-1)*covi13*(covj22*(i1+(-1)*j1)+covj12*(i2+j2) \
  )+covi11*(covj23*(i2+j2)+covj22*(i3+(-1)*j3))+covi12*((-1)* \
  covj23*i1+(-1)*covj12*i3+covj23*j1+(-2)*covj13*j2+covj12*j3))+( \
  -1)*covi22*(covi33*(covj12*(i1+(-1)*j1)+covj11*j2)+covi13*((-1)* \
  covj23*i1+(-1)*covj12*i3+covj23*j1+(-2)*covj13*j2+covj12*j3)+ \
  covi11*(covj23*i3+covj33*j2+(-1)*covj23*j3)))+aI**2*aJ*((-1)* \
  covi12*covj23**2*i1+covi12*covj22*covj33*i1+covi33*covj12**2*i2+( \
  -1)*covi33*covj11*covj22*i2+2*covi13*covj13*covj22*i2+(-2)* \
  covi13*covj12*covj23*i2+(-1)*covi12*covj13*covj23*i2+covi11* \
  covj23**2*i2+covi12*covj12*covj33*i2+(-1)*covi11*covj22*covj33* \
  i2+(-1)*covi12*covj13*covj22*i3+covi12*covj12*covj23*i3+covi12* \
  covj23**2*j1+(-1)*covi12*covj22*covj33*j1+(-1)*covi12*covj13* \
  covj23*j2+covi12*covj12*covj33*j2+covi12*covj13*covj22*j3+(-1)* \
  covi12*covj12*covj23*j3+covi23*(covj12*covj23*(i1+(-1)*j1)+(-1)* \
  covj13*(covj22*(i1+(-1)*j1)+covj12*(i2+j2))+covj11*(covj23*(i2+j2) \
  +covj22*(i3+(-1)*j3))+covj12**2*((-1)*i3+j3))+covi22*(covj12* \
  covj33*((-1)*i1+j1)+covj13**2*j2+covj13*(covj23*(i1+(-1)*j1)+ \
  covj12*(i3+(-1)*j3))+(-1)*covj11*(covj23*i3+covj33*j2+(-1)* \
  covj23*j3))))**2+aJ**2*covi23**2*(aJ**3*(covi13**2*covi22+(-2)* \
  covi12*covi13*covi23+covi12**2*covi33+covi11*(covi23**2+(-1)* \
  covi22*covi33))+aI**2*aJ*((-2)*covi23*covj12*covj13+covi22* \
  covj13**2+2*covi13*covj13*covj22+covi33*(covj12**2+(-1)*covj11* \
  covj22)+2*covi23*covj11*covj23+(-2)*covi13*covj12*covj23+(-2)* \
  covi12*covj13*covj23+covi11*covj23**2+(-1)*(covi22*covj11+(-2)* \
  covi12*covj12+covi11*covj22)*covj33)+aI*aJ**2*(covi23**2*covj11+ \
  2*covi12*covi33*covj12+covi13**2*covj22+(-1)*covi11*covi33* \
  covj22+(-2)*covi12*covi13*covj23+(-2)*covi23*(covi13*covj12+ \
  covi12*covj13+(-1)*covi11*covj23)+covi12**2*covj33+(-1)*covi22*( \
  covi33*covj11+(-2)*covi13*covj13+covi11*covj33))+aI**3*(covj13**2* \
  covj22+(-2)*covj12*covj13*covj23+covj12**2*covj33+covj11*( \
  covj23**2+(-1)*covj22*covj33)))**(-2)*(aJ**2*(covi13**2*covi22+(-2) \
  *covi12*covi13*covi23+covi12**2*covi33+covi11*(covi23**2+(-1)* \
  covi22*covi33))*(i1+(-1)*j1)+aI*aJ*(covi13**2*(covj22*(i1+(-1)* \
  j1)+covj12*((-1)*i2+j2))+covi12*(covi33*(covj12*(i1+(-1)*j1)+ \
  covj11*((-1)*i2+j2))+covi23*(covj13*((-1)*i1+j1)+covj11*(i3+(-1)* \
  j3)))+covi12**2*(covj33*(i1+(-1)*j1)+covj13*((-1)*i3+j3))+covi13*( \
  covi23*(covj12*((-1)*i1+j1)+covj11*(i2+(-1)*j2))+covi22*(covj13* \
  i1+(-1)*covj11*i3+(-1)*covj13*j1+covj11*j3)+covi12*((-2)*covj23* \
  i1+covj13*i2+covj12*i3+2*covj23*j1+(-1)*covj13*j2+(-1)*covj12* \
  j3))+covi11*(covi33*(covj22*((-1)*i1+j1)+covj12*(i2+(-1)*j2))+ \
  covi23*(2*covj23*i1+(-1)*covj13*i2+(-1)*covj12*i3+(-2)*covj23* \
  j1+covj13*j2+covj12*j3)+covi22*((-1)*covj33*i1+covj13*i3+covj33* \
  j1+(-1)*covj13*j3)))+aI**2*(covi13*(covj12*covj23*((-1)*i1+j1)+ \
  covj13*(covj22*(i1+(-1)*j1)+covj12*((-1)*i2+j2))+covj12**2*(i3+( \
  -1)*j3)+covj11*(covj23*i2+(-1)*covj22*i3+(-1)*covj23*j2+covj22* \
  j3))+covi12*(covj12*covj33*(i1+(-1)*j1)+covj13**2*(i2+(-1)*j2)+ \
  covj11*((-1)*covj33*i2+covj23*i3+covj33*j2+(-1)*covj23*j3)+ \
  covj13*(covj23*((-1)*i1+j1)+covj12*((-1)*i3+j3)))+covi11*( \
  covj23**2*(i1+(-1)*j1)+covj12*covj33*(i2+(-1)*j2)+covj22*((-1)* \
  covj33*i1+covj13*i3+covj33*j1+(-1)*covj13*j3)+covj23*(covj13*(( \
  -1)*i2+j2)+covj12*((-1)*i3+j3)))))**2+covi12**2*(aJ**3*(covi13**2* \
  covi22+(-2)*covi12*covi13*covi23+covi12**2*covi33+covi11*( \
  covi23**2+(-1)*covi22*covi33))+aI**2*aJ*((-2)*covi23*covj12* \
  covj13+covi22*covj13**2+2*covi13*covj13*covj22+covi33*(covj12**2+( \
  -1)*covj11*covj22)+2*covi23*covj11*covj23+(-2)*covi13*covj12* \
  covj23+(-2)*covi12*covj13*covj23+covi11*covj23**2+(-1)*(covi22* \
  covj11+(-2)*covi12*covj12+covi11*covj22)*covj33)+aI*aJ**2*( \
  covi23**2*covj11+2*covi12*covi33*covj12+covi13**2*covj22+(-1)* \
  covi11*covi33*covj22+(-2)*covi12*covi13*covj23+(-2)*covi23*( \
  covi13*covj12+covi12*covj13+(-1)*covi11*covj23)+covi12**2*covj33+( \
  -1)*covi22*(covi33*covj11+(-2)*covi13*covj13+covi11*covj33))+ \
  aI**3*(covj13**2*covj22+(-2)*covj12*covj13*covj23+covj12**2* \
  covj33+covj11*(covj23**2+(-1)*covj22*covj33)))**(-2)*(aI**3*( \
  covj13**2*covj22+(-2)*covj12*covj13*covj23+covj12**2*covj33+ \
  covj11*(covj23**2+(-1)*covj22*covj33))*i3+aJ**3*(covi13**2*covi22+ \
  (-2)*covi12*covi13*covi23+covi12**2*covi33+covi11*(covi23**2+(-1)* \
  covi22*covi33))*j3+aI*aJ**2*((-1)*covi22*covi33*covj13*i1+ \
  covi12*covi33*covj23*i1+covi13*covi22*covj33*i1+covi12*covi33* \
  covj13*i2+covi13**2*covj23*i2+(-1)*covi11*covi33*covj23*i2+(-1)* \
  covi12*covi13*covj33*i2+covi13*covi22*covj13*i3+(-1)*covi12* \
  covi13*covj23*i3+covi12**2*covj33*i3+(-1)*covi11*covi22*covj33* \
  i3+covi22*covi33*covj13*j1+(-1)*covi12*covi33*covj23*j1+(-1)* \
  covi13*covi22*covj33*j1+(-1)*covi12*covi33*covj13*j2+(-1)* \
  covi13**2*covj23*j2+covi11*covi33*covj23*j2+covi12*covi13* \
  covj33*j2+((-1)*covi22*covi33*covj11+2*covi12*covi33*covj12+ \
  covi13*covi22*covj13+covi13**2*covj22+(-1)*covi11*covi33*covj22+( \
  -1)*covi12*covi13*covj23)*j3+covi23**2*(covj13*(i1+(-1)*j1)+ \
  covj11*j3)+covi23*(covi13*((-1)*covj13*i2+covj23*((-1)*i1+j1)+ \
  covj13*j2+(-2)*covj12*j3)+(-1)*covi12*(covj33*(i1+(-1)*j1)+ \
  covj13*(i3+j3))+covi11*(covj33*(i2+(-1)*j2)+covj23*(i3+j3))))+ \
  aI**2*aJ*((-1)*covi13*covj23**2*i1+covi13*covj22*covj33*i1+ \
  covi13*covj13*covj23*i2+(-1)*covi13*covj12*covj33*i2+covi22* \
  covj13**2*i3+covi13*covj13*covj22*i3+(-1)*covi13*covj12*covj23* \
  i3+(-2)*covi12*covj13*covj23*i3+covi11*covj23**2*i3+(-1)*covi22* \
  covj11*covj33*i3+2*covi12*covj12*covj33*i3+(-1)*covi11*covj22* \
  covj33*i3+covi13*covj23**2*j1+(-1)*covi13*covj22*covj33*j1+(-1)* \
  covi13*covj13*covj23*j2+covi13*covj12*covj33*j2+covi13*covj13* \
  covj22*j3+(-1)*covi13*covj12*covj23*j3+covi33*(covj12*covj23*( \
  i1+(-1)*j1)+covj13*(covj22*((-1)*i1+j1)+covj12*(i2+(-1)*j2))+ \
  covj12**2*j3+(-1)*covj11*(covj23*i2+(-1)*covj23*j2+covj22*j3))+ \
  covi23*(covj12*covj33*((-1)*i1+j1)+covj13**2*((-1)*i2+j2)+covj13* \
  (covj23*(i1+(-1)*j1)+(-1)*covj12*(i3+j3))+covj11*(covj33*(i2+(-1) \
  *j2)+covj23*(i3+j3)))))**2+(-2)*covi12*(aJ**3*(covi13**2*covi22+( \
  -2)*covi12*covi13*covi23+covi12**2*covi33+covi11*(covi23**2+(-1)* \
  covi22*covi33))+aI**2*aJ*((-2)*covi23*covj12*covj13+covi22* \
  covj13**2+2*covi13*covj13*covj22+covi33*(covj12**2+(-1)*covj11* \
  covj22)+2*covi23*covj11*covj23+(-2)*covi13*covj12*covj23+(-2)* \
  covi12*covj13*covj23+covi11*covj23**2+(-1)*(covi22*covj11+(-2)* \
  covi12*covj12+covi11*covj22)*covj33)+aI*aJ**2*(covi23**2*covj11+ \
  2*covi12*covi33*covj12+covi13**2*covj22+(-1)*covi11*covi33* \
  covj22+(-2)*covi12*covi13*covj23+(-2)*covi23*(covi13*covj12+ \
  covi12*covj13+(-1)*covi11*covj23)+covi12**2*covj33+(-1)*covi22*( \
  covi33*covj11+(-2)*covi13*covj13+covi11*covj33))+aI**3*(covj13**2* \
  covj22+(-2)*covj12*covj13*covj23+covj12**2*covj33+covj11*( \
  covj23**2+(-1)*covj22*covj33)))**(-2)*(aI**3*(covj13**2*covj22+(-2) \
  *covj12*covj13*covj23+covj12**2*covj33+covj11*(covj23**2+(-1)* \
  covj22*covj33))*i3+aJ**3*(covi13**2*covi22+(-2)*covi12*covi13* \
  covi23+covi12**2*covi33+covi11*(covi23**2+(-1)*covi22*covi33))*j3+ \
  aI*aJ**2*((-1)*covi22*covi33*covj13*i1+covi12*covi33*covj23*i1+ \
  covi13*covi22*covj33*i1+covi12*covi33*covj13*i2+covi13**2* \
  covj23*i2+(-1)*covi11*covi33*covj23*i2+(-1)*covi12*covi13* \
  covj33*i2+covi13*covi22*covj13*i3+(-1)*covi12*covi13*covj23*i3+ \
  covi12**2*covj33*i3+(-1)*covi11*covi22*covj33*i3+covi22*covi33* \
  covj13*j1+(-1)*covi12*covi33*covj23*j1+(-1)*covi13*covi22* \
  covj33*j1+(-1)*covi12*covi33*covj13*j2+(-1)*covi13**2*covj23*j2+ \
  covi11*covi33*covj23*j2+covi12*covi13*covj33*j2+((-1)*covi22* \
  covi33*covj11+2*covi12*covi33*covj12+covi13*covi22*covj13+ \
  covi13**2*covj22+(-1)*covi11*covi33*covj22+(-1)*covi12*covi13* \
  covj23)*j3+covi23**2*(covj13*(i1+(-1)*j1)+covj11*j3)+covi23*( \
  covi13*((-1)*covj13*i2+covj23*((-1)*i1+j1)+covj13*j2+(-2)* \
  covj12*j3)+(-1)*covi12*(covj33*(i1+(-1)*j1)+covj13*(i3+j3))+ \
  covi11*(covj33*(i2+(-1)*j2)+covj23*(i3+j3))))+aI**2*aJ*((-1)* \
  covi13*covj23**2*i1+covi13*covj22*covj33*i1+covi13*covj13* \
  covj23*i2+(-1)*covi13*covj12*covj33*i2+covi22*covj13**2*i3+ \
  covi13*covj13*covj22*i3+(-1)*covi13*covj12*covj23*i3+(-2)* \
  covi12*covj13*covj23*i3+covi11*covj23**2*i3+(-1)*covi22*covj11* \
  covj33*i3+2*covi12*covj12*covj33*i3+(-1)*covi11*covj22*covj33* \
  i3+covi13*covj23**2*j1+(-1)*covi13*covj22*covj33*j1+(-1)*covi13* \
  covj13*covj23*j2+covi13*covj12*covj33*j2+covi13*covj13*covj22* \
  j3+(-1)*covi13*covj12*covj23*j3+covi33*(covj12*covj23*(i1+(-1)* \
  j1)+covj13*(covj22*((-1)*i1+j1)+covj12*(i2+(-1)*j2))+covj12**2*j3+ \
  (-1)*covj11*(covj23*i2+(-1)*covj23*j2+covj22*j3))+covi23*( \
  covj12*covj33*((-1)*i1+j1)+covj13**2*((-1)*i2+j2)+covj13*(covj23* \
  (i1+(-1)*j1)+(-1)*covj12*(i3+j3))+covj11*(covj33*(i2+(-1)*j2)+ \
  covj23*(i3+j3)))))*(aI**3*covi12*(covj13**2*covj22+(-2)*covj12* \
  covj13*covj23+covj12**2*covj33+covj11*(covj23**2+(-1)*covj22* \
  covj33))*i3+aJ**3*(covi13**2*covi22+(-2)*covi12*covi13*covi23+ \
  covi12**2*covi33+covi11*(covi23**2+(-1)*covi22*covi33))*(covi12* \
  i3+covi13*((-1)*i2+j2))+aI**2*aJ*(covi12*((-2)*covi23*covj12* \
  covj13+covi22*covj13**2+covi33*(covj12**2+(-1)*covj11*covj22)+2* \
  covi23*covj11*covj23+(-2)*covi12*covj13*covj23+covi11*covj23**2+( \
  -1)*(covi22*covj11+(-2)*covi12*covj12+covi11*covj22)*covj33)*i3+ \
  covi13*(covi23*(covj12*covj23*(i1+(-1)*j1)+covj13*(covj22*((-1)* \
  i1+j1)+covj12*(i2+(-1)*j2))+covj12**2*((-1)*i3+j3)+covj11*((-1)* \
  covj23*i2+covj22*i3+covj23*j2+(-1)*covj22*j3))+covi22*(covj12* \
  covj33*((-1)*i1+j1)+covj13**2*((-1)*i2+j2)+covj13*(covj23*(i1+(-1) \
  *j1)+covj12*(i3+(-1)*j3))+covj11*(covj33*i2+(-1)*covj23*i3+(-1)* \
  covj33*j2+covj23*j3))+covi12*(covj23**2*((-1)*i1+j1)+covj12* \
  covj33*((-1)*i2+j2)+covj23*(covj13*(i2+(-1)*j2)+(-1)*covj12*(i3+ \
  j3))+covj22*(covj33*(i1+(-1)*j1)+covj13*(i3+j3)))))+aI*aJ**2*( \
  covi12*(covi23**2*covj11+2*covi12*covi33*covj12+(-1)*covi11* \
  covi33*covj22+covi23*((-2)*covi12*covj13+2*covi11*covj23)+ \
  covi12**2*covj33+(-1)*covi22*(covi33*covj11+covi11*covj33))*i3+ \
  covi13**2*(covi23*(covj22*((-1)*i1+j1)+covj12*(i2+(-1)*j2))+ \
  covi22*(covj23*i1+(-2)*covj13*i2+covj12*i3+(-1)*covj23*j1+2* \
  covj13*j2+(-1)*covj12*j3)+covi12*(covj23*i2+(-1)*covj23*j2+ \
  covj22*j3))+covi13*(covi23**2*(covj12*(i1+(-1)*j1)+covj11*((-1)* \
  i2+j2))+covi23*(covi12*((-1)*covj23*i1+2*covj13*i2+(-3)*covj12* \
  i3+covj23*j1+(-2)*covj13*j2+covj12*j3)+covi11*((-1)*covj23*i2+ \
  covj22*i3+covj23*j2+(-1)*covj22*j3))+covi22*(2*covi12*covj13*i3+ \
  covi33*((-1)*covj12*i1+covj11*i2+covj12*j1+(-1)*covj11*j2)+ \
  covi11*(covj33*i2+(-1)*covj23*i3+(-1)*covj33*j2+covj23*j3))+(-1) \
  *covi12*(covi33*((-1)*covj22*i1+covj12*i2+covj22*j1+(-1)* \
  covj12*j2)+covi12*(covj33*(i2+(-1)*j2)+covj23*(i3+j3))))))+2* \
  aJ**2*covi23*(aJ**3*(covi13**2*covi22+(-2)*covi12*covi13*covi23+ \
  covi12**2*covi33+covi11*(covi23**2+(-1)*covi22*covi33))+aI**2*aJ*( \
  (-2)*covi23*covj12*covj13+covi22*covj13**2+2*covi13*covj13* \
  covj22+covi33*(covj12**2+(-1)*covj11*covj22)+2*covi23*covj11* \
  covj23+(-2)*covi13*covj12*covj23+(-2)*covi12*covj13*covj23+ \
  covi11*covj23**2+(-1)*(covi22*covj11+(-2)*covi12*covj12+covi11* \
  covj22)*covj33)+aI*aJ**2*(covi23**2*covj11+2*covi12*covi33* \
  covj12+covi13**2*covj22+(-1)*covi11*covi33*covj22+(-2)*covi12* \
  covi13*covj23+(-2)*covi23*(covi13*covj12+covi12*covj13+(-1)* \
  covi11*covj23)+covi12**2*covj33+(-1)*covi22*(covi33*covj11+(-2)* \
  covi13*covj13+covi11*covj33))+aI**3*(covj13**2*covj22+(-2)*covj12* \
  covj13*covj23+covj12**2*covj33+covj11*(covj23**2+(-1)*covj22* \
  covj33)))**(-2)*((-1)*covi13*(aJ**2*(covi13**2*covi22+(-2)* \
  covi12*covi13*covi23+covi12**2*covi33+covi11*(covi23**2+(-1)* \
  covi22*covi33))*(i1+(-1)*j1)+aI*aJ*(covi13**2*(covj22*(i1+(-1)* \
  j1)+covj12*((-1)*i2+j2))+covi12*(covi33*(covj12*(i1+(-1)*j1)+ \
  covj11*((-1)*i2+j2))+covi23*(covj13*((-1)*i1+j1)+covj11*(i3+(-1)* \
  j3)))+covi12**2*(covj33*(i1+(-1)*j1)+covj13*((-1)*i3+j3))+covi13*( \
  covi23*(covj12*((-1)*i1+j1)+covj11*(i2+(-1)*j2))+covi22*(covj13* \
  i1+(-1)*covj11*i3+(-1)*covj13*j1+covj11*j3)+covi12*((-2)*covj23* \
  i1+covj13*i2+covj12*i3+2*covj23*j1+(-1)*covj13*j2+(-1)*covj12* \
  j3))+covi11*(covi33*(covj22*((-1)*i1+j1)+covj12*(i2+(-1)*j2))+ \
  covi23*(2*covj23*i1+(-1)*covj13*i2+(-1)*covj12*i3+(-2)*covj23* \
  j1+covj13*j2+covj12*j3)+covi22*((-1)*covj33*i1+covj13*i3+covj33* \
  j1+(-1)*covj13*j3)))+aI**2*(covi13*(covj12*covj23*((-1)*i1+j1)+ \
  covj13*(covj22*(i1+(-1)*j1)+covj12*((-1)*i2+j2))+covj12**2*(i3+( \
  -1)*j3)+covj11*(covj23*i2+(-1)*covj22*i3+(-1)*covj23*j2+covj22* \
  j3))+covi12*(covj12*covj33*(i1+(-1)*j1)+covj13**2*(i2+(-1)*j2)+ \
  covj11*((-1)*covj33*i2+covj23*i3+covj33*j2+(-1)*covj23*j3)+ \
  covj13*(covj23*((-1)*i1+j1)+covj12*((-1)*i3+j3)))+covi11*( \
  covj23**2*(i1+(-1)*j1)+covj12*covj33*(i2+(-1)*j2)+covj22*((-1)* \
  covj33*i1+covj13*i3+covj33*j1+(-1)*covj13*j3)+covj23*(covj13*(( \
  -1)*i2+j2)+covj12*((-1)*i3+j3)))))*(aJ**2*(covi13**2*covi22+(-2)* \
  covi12*covi13*covi23+covi12**2*covi33+covi11*(covi23**2+(-1)* \
  covi22*covi33))*(i2+(-1)*j2)+aI*aJ*(covi23**2*(covj12*((-1)*i1+ \
  j1)+covj11*(i2+(-1)*j2))+covi23*(covi13*(covj22*(i1+(-1)*j1)+ \
  covj12*((-1)*i2+j2))+covi12*(covj23*i1+(-2)*covj13*i2+covj12*i3+( \
  -1)*covj23*j1+2*covj13*j2+(-1)*covj12*j3)+covi11*(covj23*i2+(-1) \
  *covj22*i3+(-1)*covj23*j2+covj22*j3))+covi22*(covi33*(covj12*( \
  i1+(-1)*j1)+covj11*((-1)*i2+j2))+covi13*((-1)*covj23*i1+2* \
  covj13*i2+(-1)*covj12*i3+covj23*j1+(-2)*covj13*j2+covj12*j3)+ \
  covi11*((-1)*covj33*i2+covj23*i3+covj33*j2+(-1)*covj23*j3))+ \
  covi12*(covi33*(covj22*((-1)*i1+j1)+covj12*(i2+(-1)*j2))+covi13*( \
  (-1)*covj23*i2+covj22*i3+covj23*j2+(-1)*covj22*j3)+covi12*( \
  covj33*i2+(-1)*covj23*i3+(-1)*covj33*j2+covj23*j3)))+aI**2*( \
  covi23*(covj12*covj23*((-1)*i1+j1)+covj13*(covj22*(i1+(-1)*j1)+ \
  covj12*((-1)*i2+j2))+covj12**2*(i3+(-1)*j3)+covj11*(covj23*i2+(-1) \
  *covj22*i3+(-1)*covj23*j2+covj22*j3))+covi22*(covj12*covj33*(i1+ \
  (-1)*j1)+covj13**2*(i2+(-1)*j2)+covj11*((-1)*covj33*i2+covj23*i3+ \
  covj33*j2+(-1)*covj23*j3)+covj13*(covj23*((-1)*i1+j1)+covj12*(( \
  -1)*i3+j3)))+covi12*(covj23**2*(i1+(-1)*j1)+covj12*covj33*(i2+(-1) \
  *j2)+covj22*((-1)*covj33*i1+covj13*i3+covj33*j1+(-1)*covj13*j3)+ \
  covj23*(covj13*((-1)*i2+j2)+covj12*((-1)*i3+j3)))))+(-1)*(aJ**2*( \
  covi13**2*covi22+(-2)*covi12*covi13*covi23+covi12**2*covi33+ \
  covi11*(covi23**2+(-1)*covi22*covi33))*(i3+(-1)*j3)+aI*aJ*((-1)* \
  covi12*covi33*covj23*i1+(-1)*covi12*covi33*covj13*i2+(-1)* \
  covi13**2*covj23*i2+covi11*covi33*covj23*i2+covi12*covi13* \
  covj33*i2+2*covi12*covi33*covj12*i3+covi13**2*covj22*i3+(-1)* \
  covi11*covi33*covj22*i3+(-1)*covi12*covi13*covj23*i3+covi12* \
  covi33*covj23*j1+covi12*covi33*covj13*j2+covi13**2*covj23*j2+(-1) \
  *covi11*covi33*covj23*j2+(-1)*covi12*covi13*covj33*j2+ \
  covi23**2*(covj13*((-1)*i1+j1)+covj11*(i3+(-1)*j3))+(-2)*covi12* \
  covi33*covj12*j3+(-1)*covi13**2*covj22*j3+covi11*covi33*covj22* \
  j3+covi12*covi13*covj23*j3+covi23*(covi13*(covj13*i2+(-2)* \
  covj12*i3+covj23*(i1+(-1)*j1)+(-1)*covj13*j2+2*covj12*j3)+ \
  covi12*(covj33*i1+(-1)*covj13*i3+(-1)*covj33*j1+covj13*j3)+ \
  covi11*((-1)*covj33*i2+covj23*i3+covj33*j2+(-1)*covj23*j3))+ \
  covi22*(covi13*(covj33*((-1)*i1+j1)+covj13*(i3+(-1)*j3))+covi33*( \
  covj13*(i1+(-1)*j1)+covj11*((-1)*i3+j3))))+aI**2*(covi33*(covj12* \
  covj23*((-1)*i1+j1)+covj13*(covj22*(i1+(-1)*j1)+covj12*((-1)*i2+ \
  j2))+covj12**2*(i3+(-1)*j3)+covj11*(covj23*i2+(-1)*covj22*i3+(-1) \
  *covj23*j2+covj22*j3))+covi23*(covj12*covj33*(i1+(-1)*j1)+ \
  covj13**2*(i2+(-1)*j2)+covj11*((-1)*covj33*i2+covj23*i3+covj33* \
  j2+(-1)*covj23*j3)+covj13*(covj23*((-1)*i1+j1)+covj12*((-1)*i3+ \
  j3)))+covi13*(covj23**2*(i1+(-1)*j1)+covj12*covj33*(i2+(-1)*j2)+ \
  covj22*((-1)*covj33*i1+covj13*i3+covj33*j1+(-1)*covj13*j3)+ \
  covj23*(covj13*((-1)*i2+j2)+covj12*((-1)*i3+j3)))))*(aJ**2*( \
  covi13**2*covi22+(-2)*covi12*covi13*covi23+covi12**2*covi33+ \
  covi11*(covi23**2+(-1)*covi22*covi33))*(covi12*(i1+(-1)*j1)+ \
  covi11*((-1)*i2+j2))+aI*aJ*(covi12**2*((-1)*covi23*covj13*i1+( \
  -2)*covi13*covj23*i1+covi13*covj13*i2+(-1)*covi11*covj33*i2+ \
  covi23*covj11*i3+covi13*covj12*i3+covi11*covj23*i3+covi23* \
  covj13*j1+2*covi13*covj23*j1+(-1)*covi13*covj13*j2+covi11* \
  covj33*j2+covi33*(covj12*(i1+(-1)*j1)+covj11*((-1)*i2+j2))+(-1)*( \
  covi23*covj11+covi13*covj12+covi11*covj23)*j3)+covi12**3*(covj33*( \
  i1+(-1)*j1)+covj13*((-1)*i3+j3))+covi12*(covi13**2*(covj22*(i1+( \
  -1)*j1)+covj12*((-1)*i2+j2))+covi11*(covi22*(covj33*((-1)*i1+j1)+ \
  covj13*(i3+(-1)*j3))+covi23*(covj13*i2+(-2)*covj12*i3+covj23*(i1+ \
  (-1)*j1)+(-1)*covj13*j2+2*covj12*j3))+covi13*(covi23*(covj12*(( \
  -1)*i1+j1)+covj11*(i2+(-1)*j2))+covi22*(covj13*i1+(-1)*covj11*i3+ \
  (-1)*covj13*j1+covj11*j3)+covi11*(covj23*i2+(-1)*covj22*i3+(-1)* \
  covj23*j2+covj22*j3)))+covi11*(covi23**2*(covj12*(i1+(-1)*j1)+ \
  covj11*((-1)*i2+j2))+covi23*(covi13*(covj22*((-1)*i1+j1)+covj12*( \
  i2+(-1)*j2))+covi11*(covj23*((-1)*i2+j2)+covj22*(i3+(-1)*j3)))+ \
  covi22*(covi33*(covj12*((-1)*i1+j1)+covj11*(i2+(-1)*j2))+covi13*( \
  covj23*i1+(-2)*covj13*i2+covj12*i3+(-1)*covj23*j1+2*covj13*j2+( \
  -1)*covj12*j3)+covi11*(covj33*i2+(-1)*covj23*i3+(-1)*covj33*j2+ \
  covj23*j3))))+aI**2*(covi12*covi13*(covj12*covj23*((-1)*i1+j1)+ \
  covj13*(covj22*(i1+(-1)*j1)+covj12*((-1)*i2+j2))+covj12**2*(i3+( \
  -1)*j3)+covj11*(covj23*i2+(-1)*covj22*i3+(-1)*covj23*j2+covj22* \
  j3))+covi12**2*(covj12*covj33*(i1+(-1)*j1)+covj13**2*(i2+(-1)*j2)+ \
  covj11*((-1)*covj33*i2+covj23*i3+covj33*j2+(-1)*covj23*j3)+ \
  covj13*(covj23*((-1)*i1+j1)+covj12*((-1)*i3+j3)))+covi11*(covi23* \
  (covj12*covj23*(i1+(-1)*j1)+covj13*(covj22*((-1)*i1+j1)+covj12*( \
  i2+(-1)*j2))+covj12**2*((-1)*i3+j3)+covj11*((-1)*covj23*i2+ \
  covj22*i3+covj23*j2+(-1)*covj22*j3))+covi22*(covj12*covj33*((-1) \
  *i1+j1)+covj13**2*((-1)*i2+j2)+covj13*(covj23*(i1+(-1)*j1)+ \
  covj12*(i3+(-1)*j3))+covj11*(covj33*i2+(-1)*covj23*i3+(-1)* \
  covj33*j2+covj23*j3))))))+(-1)*aJ**2*covi22*(aJ**3*(covi13**2* \
  covi22+(-2)*covi12*covi13*covi23+covi12**2*covi33+covi11*( \
  covi23**2+(-1)*covi22*covi33))+aI**2*aJ*((-2)*covi23*covj12* \
  covj13+covi22*covj13**2+2*covi13*covj13*covj22+covi33*(covj12**2+( \
  -1)*covj11*covj22)+2*covi23*covj11*covj23+(-2)*covi13*covj12* \
  covj23+(-2)*covi12*covj13*covj23+covi11*covj23**2+(-1)*(covi22* \
  covj11+(-2)*covi12*covj12+covi11*covj22)*covj33)+aI*aJ**2*( \
  covi23**2*covj11+2*covi12*covi33*covj12+covi13**2*covj22+(-1)* \
  covi11*covi33*covj22+(-2)*covi12*covi13*covj23+(-2)*covi23*( \
  covi13*covj12+covi12*covj13+(-1)*covi11*covj23)+covi12**2*covj33+( \
  -1)*covi22*(covi33*covj11+(-2)*covi13*covj13+covi11*covj33))+ \
  aI**3*(covj13**2*covj22+(-2)*covj12*covj13*covj23+covj12**2* \
  covj33+covj11*(covj23**2+(-1)*covj22*covj33)))**(-2)*(covi33*((-1) \
  *aJ**2*(covi13**2*covi22+(-2)*covi12*covi13*covi23+covi12**2* \
  covi33+covi11*(covi23**2+(-1)*covi22*covi33))*(i1+(-1)*j1)+aI**2*( \
  covi11*(covj23**2*((-1)*i1+j1)+covj12*covj33*((-1)*i2+j2)+covj23* \
  (covj13*(i2+(-1)*j2)+covj12*(i3+(-1)*j3))+covj22*(covj33*i1+(-1)* \
  covj13*i3+(-1)*covj33*j1+covj13*j3))+covi13*(covj12*covj23*(i1+( \
  -1)*j1)+covj13*(covj22*((-1)*i1+j1)+covj12*(i2+(-1)*j2))+ \
  covj12**2*((-1)*i3+j3)+covj11*((-1)*covj23*i2+covj22*i3+covj23* \
  j2+(-1)*covj22*j3))+covi12*(covj12*covj33*((-1)*i1+j1)+covj13**2* \
  ((-1)*i2+j2)+covj13*(covj23*(i1+(-1)*j1)+covj12*(i3+(-1)*j3))+ \
  covj11*(covj33*i2+(-1)*covj23*i3+(-1)*covj33*j2+covj23*j3)))+aI* \
  aJ*(covi13**2*(covj22*((-1)*i1+j1)+covj12*(i2+(-1)*j2))+ \
  covi12**2*(covj33*((-1)*i1+j1)+covj13*(i3+(-1)*j3))+covi13*( \
  covi23*(covj12*(i1+(-1)*j1)+covj11*((-1)*i2+j2))+covi22*((-1)* \
  covj13*i1+covj11*i3+covj13*j1+(-1)*covj11*j3)+covi12*(2*covj23* \
  i1+(-1)*covj13*i2+(-1)*covj12*i3+(-2)*covj23*j1+covj13*j2+ \
  covj12*j3))+covi11*(covi33*(covj22*(i1+(-1)*j1)+covj12*((-1)*i2+ \
  j2))+covi23*((-2)*covj23*i1+covj13*i2+covj12*i3+2*covj23*j1+(-1) \
  *covj13*j2+(-1)*covj12*j3)+covi22*(covj33*i1+(-1)*covj13*i3+(-1) \
  *covj33*j1+covj13*j3))+covi12*(covi33*(covj12*((-1)*i1+j1)+ \
  covj11*(i2+(-1)*j2))+covi23*(covj13*(i1+(-1)*j1)+covj11*((-1)*i3+ \
  j3)))))**2+(-1)*(aJ**2*(covi13**2*covi22+(-2)*covi12*covi13* \
  covi23+covi12**2*covi33+covi11*(covi23**2+(-1)*covi22*covi33))*(i3+ \
  (-1)*j3)+aI*aJ*((-1)*covi12*covi33*covj23*i1+(-1)*covi12* \
  covi33*covj13*i2+(-1)*covi13**2*covj23*i2+covi11*covi33*covj23* \
  i2+covi12*covi13*covj33*i2+2*covi12*covi33*covj12*i3+covi13**2* \
  covj22*i3+(-1)*covi11*covi33*covj22*i3+(-1)*covi12*covi13* \
  covj23*i3+covi12*covi33*covj23*j1+covi12*covi33*covj13*j2+ \
  covi13**2*covj23*j2+(-1)*covi11*covi33*covj23*j2+(-1)*covi12* \
  covi13*covj33*j2+covi23**2*(covj13*((-1)*i1+j1)+covj11*(i3+(-1)* \
  j3))+(-2)*covi12*covi33*covj12*j3+(-1)*covi13**2*covj22*j3+ \
  covi11*covi33*covj22*j3+covi12*covi13*covj23*j3+covi23*(covi13*( \
  covj13*i2+(-2)*covj12*i3+covj23*(i1+(-1)*j1)+(-1)*covj13*j2+2* \
  covj12*j3)+covi12*(covj33*i1+(-1)*covj13*i3+(-1)*covj33*j1+ \
  covj13*j3)+covi11*((-1)*covj33*i2+covj23*i3+covj33*j2+(-1)* \
  covj23*j3))+covi22*(covi13*(covj33*((-1)*i1+j1)+covj13*(i3+(-1)* \
  j3))+covi33*(covj13*(i1+(-1)*j1)+covj11*((-1)*i3+j3))))+aI**2*( \
  covi33*(covj12*covj23*((-1)*i1+j1)+covj13*(covj22*(i1+(-1)*j1)+ \
  covj12*((-1)*i2+j2))+covj12**2*(i3+(-1)*j3)+covj11*(covj23*i2+(-1) \
  *covj22*i3+(-1)*covj23*j2+covj22*j3))+covi23*(covj12*covj33*(i1+ \
  (-1)*j1)+covj13**2*(i2+(-1)*j2)+covj11*((-1)*covj33*i2+covj23*i3+ \
  covj33*j2+(-1)*covj23*j3)+covj13*(covj23*((-1)*i1+j1)+covj12*(( \
  -1)*i3+j3)))+covi13*(covj23**2*(i1+(-1)*j1)+covj12*covj33*(i2+(-1) \
  *j2)+covj22*((-1)*covj33*i1+covj13*i3+covj33*j1+(-1)*covj13*j3)+ \
  covj23*(covj13*((-1)*i2+j2)+covj12*((-1)*i3+j3)))))*(aJ**2*( \
  covi13**2*covi22+(-2)*covi12*covi13*covi23+covi12**2*covi33+ \
  covi11*(covi23**2+(-1)*covi22*covi33))*(2*covi13*(i1+(-1)*j1)+ \
  covi11*((-1)*i3+j3))+aI**2*(2*covi13**2*(covj12*covj23*((-1)*i1+ \
  j1)+covj13*(covj22*(i1+(-1)*j1)+covj12*((-1)*i2+j2))+covj12**2*( \
  i3+(-1)*j3)+covj11*(covj23*i2+(-1)*covj22*i3+(-1)*covj23*j2+ \
  covj22*j3))+covi11*(covi33*(covj12*covj23*(i1+(-1)*j1)+covj13*( \
  covj22*((-1)*i1+j1)+covj12*(i2+(-1)*j2))+covj12**2*((-1)*i3+j3)+ \
  covj11*((-1)*covj23*i2+covj22*i3+covj23*j2+(-1)*covj22*j3))+ \
  covi23*(covj12*covj33*((-1)*i1+j1)+covj13**2*((-1)*i2+j2)+covj13* \
  (covj23*(i1+(-1)*j1)+covj12*(i3+(-1)*j3))+covj11*(covj33*i2+(-1)* \
  covj23*i3+(-1)*covj33*j2+covj23*j3)))+covi13*(2*covi12*(covj12* \
  covj33*(i1+(-1)*j1)+covj13**2*(i2+(-1)*j2)+covj11*((-1)*covj33* \
  i2+covj23*i3+covj33*j2+(-1)*covj23*j3)+covj13*(covj23*((-1)*i1+ \
  j1)+covj12*((-1)*i3+j3)))+covi11*(covj23**2*(i1+(-1)*j1)+covj12* \
  covj33*(i2+(-1)*j2)+covj22*((-1)*covj33*i1+covj13*i3+covj33*j1+( \
  -1)*covj13*j3)+covj23*(covj13*((-1)*i2+j2)+covj12*((-1)*i3+j3)))) \
  )+aI*aJ*(2*covi13**3*(covj22*(i1+(-1)*j1)+covj12*((-1)*i2+j2))+ \
  covi13**2*((-4)*covi12*covj23*i1+2*covi12*covj13*i2+covi11* \
  covj23*i2+2*covi12*covj12*i3+(-1)*covi11*covj22*i3+4*covi12* \
  covj23*j1+2*covi23*(covj12*((-1)*i1+j1)+covj11*(i2+(-1)*j2))+(-2) \
  *covi12*covj13*j2+(-1)*covi11*covj23*j2+(-2)*covi12*covj12*j3+ \
  covi11*covj22*j3+2*covi22*(covj13*i1+(-1)*covj11*i3+(-1)* \
  covj13*j1+covj11*j3))+covi13*(2*covi12**2*(covj33*(i1+(-1)*j1)+ \
  covj13*((-1)*i3+j3))+covi11*(2*covi33*(covj22*((-1)*i1+j1)+ \
  covj12*(i2+(-1)*j2))+3*covi23*(covj23*i1+(-1)*covj13*i2+(-1)* \
  covj23*j1+covj13*j2)+covi22*((-1)*covj33*i1+covj13*i3+covj33*j1+( \
  -1)*covj13*j3))+covi12*(2*covi33*(covj12*(i1+(-1)*j1)+covj11*(( \
  -1)*i2+j2))+2*covi23*((-1)*covj13*i1+covj11*i3+covj13*j1+(-1)* \
  covj11*j3)+covi11*((-1)*covj33*i2+covj23*i3+covj33*j2+(-1)* \
  covj23*j3)))+covi11*(covi23**2*(covj13*(i1+(-1)*j1)+covj11*((-1)* \
  i3+j3))+covi33*(covi22*(covj13*((-1)*i1+j1)+covj11*(i3+(-1)*j3))+ \
  covi12*(covj23*i1+covj13*i2+(-2)*covj12*i3+(-1)*covj23*j1+(-1)* \
  covj13*j2+2*covj12*j3)+covi11*((-1)*covj23*i2+covj22*i3+covj23* \
  j2+(-1)*covj22*j3))+covi23*(covi12*(covj33*((-1)*i1+j1)+covj13*( \
  i3+(-1)*j3))+covi11*(covj33*(i2+(-1)*j2)+covj23*((-1)*i3+j3))))))) \
  )
        GradJ=(1/2)*(covj13**2*covj22+(-2)*covj12*covj13*covj23+covj12**2* \
  covj33+covj11*(covj23**2+(-1)*covj22*covj33))**(-1)*(aJ**3*( \
  covi13**2*covi22+(-2)*covi12*covi13*covi23+covi12**2*covi33+ \
  covi11*(covi23**2+(-1)*covi22*covi33))+aI**2*aJ*((-2)*covi23* \
  covj12*covj13+covi22*covj13**2+2*covi13*covj13*covj22+covi33*( \
  covj12**2+(-1)*covj11*covj22)+2*covi23*covj11*covj23+(-2)*covi13* \
  covj12*covj23+(-2)*covi12*covj13*covj23+covi11*covj23**2+(-1)*( \
  covi22*covj11+(-2)*covi12*covj12+covi11*covj22)*covj33)+aI*aJ**2* \
  (covi23**2*covj11+2*covi12*covi33*covj12+covi13**2*covj22+(-1)* \
  covi11*covi33*covj22+(-2)*covi12*covi13*covj23+(-2)*covi23*( \
  covi13*covj12+covi12*covj13+(-1)*covi11*covj23)+covi12**2*covj33+( \
  -1)*covi22*(covi33*covj11+(-2)*covi13*covj13+covi11*covj33))+ \
  aI**3*(covj13**2*covj22+(-2)*covj12*covj13*covj23+covj12**2* \
  covj33+covj11*(covj23**2+(-1)*covj22*covj33)))**(-2)*(2*covj12* \
  covj33*(aJ**3*(covi13**2*covi22+(-2)*covi12*covi13*covi23+ \
  covi12**2*covi33+covi11*(covi23**2+(-1)*covi22*covi33))+aI**2*aJ*( \
  (-2)*covi23*covj12*covj13+covi22*covj13**2+2*covi13*covj13* \
  covj22+covi33*(covj12**2+(-1)*covj11*covj22)+2*covi23*covj11* \
  covj23+(-2)*covi13*covj12*covj23+(-2)*covi12*covj13*covj23+ \
  covi11*covj23**2+(-1)*(covi22*covj11+(-2)*covi12*covj12+covi11* \
  covj22)*covj33)+aI*aJ**2*(covi23**2*covj11+2*covi12*covi33* \
  covj12+covi13**2*covj22+(-1)*covi11*covi33*covj22+(-2)*covi12* \
  covi13*covj23+(-2)*covi23*(covi13*covj12+covi12*covj13+(-1)* \
  covi11*covj23)+covi12**2*covj33+(-1)*covi22*(covi33*covj11+(-2)* \
  covi13*covj13+covi11*covj33))+aI**3*(covj13**2*covj22+(-2)*covj12* \
  covj13*covj23+covj12**2*covj33+covj11*(covj23**2+(-1)*covj22* \
  covj33)))**2*j1*j2+covj13**2*(aJ**3*(covi13**2*covi22+(-2)* \
  covi12*covi13*covi23+covi12**2*covi33+covi11*(covi23**2+(-1)* \
  covi22*covi33))+aI**2*aJ*((-2)*covi23*covj12*covj13+covi22* \
  covj13**2+2*covi13*covj13*covj22+covi33*(covj12**2+(-1)*covj11* \
  covj22)+2*covi23*covj11*covj23+(-2)*covi13*covj12*covj23+(-2)* \
  covi12*covj13*covj23+covi11*covj23**2+(-1)*(covi22*covj11+(-2)* \
  covi12*covj12+covi11*covj22)*covj33)+aI*aJ**2*(covi23**2*covj11+ \
  2*covi12*covi33*covj12+covi13**2*covj22+(-1)*covi11*covi33* \
  covj22+(-2)*covi12*covi13*covj23+(-2)*covi23*(covi13*covj12+ \
  covi12*covj13+(-1)*covi11*covj23)+covi12**2*covj33+(-1)*covi22*( \
  covi33*covj11+(-2)*covi13*covj13+covi11*covj33))+aI**3*(covj13**2* \
  covj22+(-2)*covj12*covj13*covj23+covj12**2*covj33+covj11*( \
  covj23**2+(-1)*covj22*covj33)))**2*j2**2+(-1)*covj11*covj33*( \
  aJ**3*(covi13**2*covi22+(-2)*covi12*covi13*covi23+covi12**2* \
  covi33+covi11*(covi23**2+(-1)*covi22*covi33))+aI**2*aJ*((-2)* \
  covi23*covj12*covj13+covi22*covj13**2+2*covi13*covj13*covj22+ \
  covi33*(covj12**2+(-1)*covj11*covj22)+2*covi23*covj11*covj23+(-2) \
  *covi13*covj12*covj23+(-2)*covi12*covj13*covj23+covi11*covj23**2+ \
  (-1)*(covi22*covj11+(-2)*covi12*covj12+covi11*covj22)*covj33)+aI* \
  aJ**2*(covi23**2*covj11+2*covi12*covi33*covj12+covi13**2*covj22+( \
  -1)*covi11*covi33*covj22+(-2)*covi12*covi13*covj23+(-2)*covi23*( \
  covi13*covj12+covi12*covj13+(-1)*covi11*covj23)+covi12**2*covj33+( \
  -1)*covi22*(covi33*covj11+(-2)*covi13*covj13+covi11*covj33))+ \
  aI**3*(covj13**2*covj22+(-2)*covj12*covj13*covj23+covj12**2* \
  covj33+covj11*(covj23**2+(-1)*covj22*covj33)))**2*j2**2+(-2)* \
  covj12*covj13*(aJ**3*(covi13**2*covi22+(-2)*covi12*covi13*covi23+ \
  covi12**2*covi33+covi11*(covi23**2+(-1)*covi22*covi33))+aI**2*aJ*( \
  (-2)*covi23*covj12*covj13+covi22*covj13**2+2*covi13*covj13* \
  covj22+covi33*(covj12**2+(-1)*covj11*covj22)+2*covi23*covj11* \
  covj23+(-2)*covi13*covj12*covj23+(-2)*covi12*covj13*covj23+ \
  covi11*covj23**2+(-1)*(covi22*covj11+(-2)*covi12*covj12+covi11* \
  covj22)*covj33)+aI*aJ**2*(covi23**2*covj11+2*covi12*covi33* \
  covj12+covi13**2*covj22+(-1)*covi11*covi33*covj22+(-2)*covi12* \
  covi13*covj23+(-2)*covi23*(covi13*covj12+covi12*covj13+(-1)* \
  covi11*covj23)+covi12**2*covj33+(-1)*covi22*(covi33*covj11+(-2)* \
  covi13*covj13+covi11*covj33))+aI**3*(covj13**2*covj22+(-2)*covj12* \
  covj13*covj23+covj12**2*covj33+covj11*(covj23**2+(-1)*covj22* \
  covj33)))**2*j2*j3+covj12**2*(aJ**3*(covi13**2*covi22+(-2)* \
  covi12*covi13*covi23+covi12**2*covi33+covi11*(covi23**2+(-1)* \
  covi22*covi33))+aI**2*aJ*((-2)*covi23*covj12*covj13+covi22* \
  covj13**2+2*covi13*covj13*covj22+covi33*(covj12**2+(-1)*covj11* \
  covj22)+2*covi23*covj11*covj23+(-2)*covi13*covj12*covj23+(-2)* \
  covi12*covj13*covj23+covi11*covj23**2+(-1)*(covi22*covj11+(-2)* \
  covi12*covj12+covi11*covj22)*covj33)+aI*aJ**2*(covi23**2*covj11+ \
  2*covi12*covi33*covj12+covi13**2*covj22+(-1)*covi11*covi33* \
  covj22+(-2)*covi12*covi13*covj23+(-2)*covi23*(covi13*covj12+ \
  covi12*covj13+(-1)*covi11*covj23)+covi12**2*covj33+(-1)*covi22*( \
  covi33*covj11+(-2)*covi13*covj13+covi11*covj33))+aI**3*(covj13**2* \
  covj22+(-2)*covj12*covj13*covj23+covj12**2*covj33+covj11*( \
  covj23**2+(-1)*covj22*covj33)))**2*j3**2+aI**2*covj23**2*(aI**2*( \
  covj13**2*covj22+(-2)*covj12*covj13*covj23+covj12**2*covj33+ \
  covj11*(covj23**2+(-1)*covj22*covj33))*(i1+(-1)*j1)+aI*aJ*( \
  covi22*covj13**2*i1+covi13*covj13*covj22*i1+(-1)*covi13*covj12* \
  covj23*i1+(-1)*covi12*covj13*covj23*i1+(-1)*covi22*covj11* \
  covj33*i1+covi12*covj12*covj33*i1+covi13*covj12*covj13*i2+(-1)* \
  covi12*covj13**2*i2+(-1)*covi13*covj11*covj23*i2+covi11*covj13* \
  covj23*i2+covi12*covj11*covj33*i2+(-1)*covi11*covj12*covj33*i2+( \
  -1)*covi13*covj12**2*i3+covi12*covj12*covj13*i3+covi13*covj11* \
  covj22*i3+(-1)*covi11*covj13*covj22*i3+(-1)*covi12*covj11* \
  covj23*i3+covi11*covj12*covj23*i3+covi33*(covj12**2+(-1)*covj11* \
  covj22)*(i1+(-1)*j1)+(-2)*covi23*(covj12*covj13+(-1)*covj11* \
  covj23)*(i1+(-1)*j1)+(-1)*covi22*covj13**2*j1+(-1)*covi13* \
  covj13*covj22*j1+covi13*covj12*covj23*j1+covi12*covj13*covj23* \
  j1+covi22*covj11*covj33*j1+(-1)*covi12*covj12*covj33*j1+(-1)* \
  covi13*covj12*covj13*j2+covi12*covj13**2*j2+covi13*covj11* \
  covj23*j2+(-1)*covi11*covj13*covj23*j2+(-1)*covi12*covj11* \
  covj33*j2+covi11*covj12*covj33*j2+(covi13*covj12**2+(-1)*covi12* \
  covj12*covj13+(-1)*covi13*covj11*covj22+covi11*covj13*covj22+ \
  covi12*covj11*covj23+(-1)*covi11*covj12*covj23)*j3)+aJ**2*( \
  covi12*covi33*covj12*i1+covi12*covi33*covj11*i2+covi13**2* \
  covj12*i2+(-1)*covi11*covi33*covj12*i2+(-1)*covi12*covi13* \
  covj13*i2+(-1)*covi12*covi13*covj12*i3+covi12**2*covj13*i3+ \
  covi23**2*covj11*(i1+(-1)*j1)+(-1)*covi12*covi33*covj12*j1+(-1)* \
  covi12*covi33*covj11*j2+(-1)*covi13**2*covj12*j2+covi11*covi33* \
  covj12*j2+covi12*covi13*covj13*j2+covi12*covi13*covj12*j3+(-1)* \
  covi12**2*covj13*j3+covi22*(covi33*covj11*((-1)*i1+j1)+covi11* \
  covj13*((-1)*i3+j3)+covi13*(covj13*i1+covj11*i3+(-1)*covj13*j1+( \
  -1)*covj11*j3))+covi23*(covi13*(covj12*((-1)*i1+j1)+covj11*((-1) \
  *i2+j2))+covi12*((-1)*covj13*i1+(-1)*covj11*i3+covj13*j1+covj11* \
  j3)+covi11*(covj13*i2+covj12*i3+(-1)*covj13*j2+(-1)*covj12*j3)))) \
  **2+(-2)*covj12*covj33*(aJ**3*(covi13**2*covi22+(-2)*covi12* \
  covi13*covi23+covi12**2*covi33+covi11*(covi23**2+(-1)*covi22* \
  covi33))+aI**2*aJ*((-2)*covi23*covj12*covj13+covi22*covj13**2+2* \
  covi13*covj13*covj22+covi33*(covj12**2+(-1)*covj11*covj22)+2* \
  covi23*covj11*covj23+(-2)*covi13*covj12*covj23+(-2)*covi12* \
  covj13*covj23+covi11*covj23**2+(-1)*(covi22*covj11+(-2)*covi12* \
  covj12+covi11*covj22)*covj33)+aI*aJ**2*(covi23**2*covj11+2* \
  covi12*covi33*covj12+covi13**2*covj22+(-1)*covi11*covi33*covj22+( \
  -2)*covi12*covi13*covj23+(-2)*covi23*(covi13*covj12+covi12* \
  covj13+(-1)*covi11*covj23)+covi12**2*covj33+(-1)*covi22*(covi33* \
  covj11+(-2)*covi13*covj13+covi11*covj33))+aI**3*(covj13**2*covj22+( \
  -2)*covj12*covj13*covj23+covj12**2*covj33+covj11*(covj23**2+(-1)* \
  covj22*covj33)))*j2*(aI**3*(covj13**2*covj22+(-2)*covj12*covj13* \
  covj23+covj12**2*covj33+covj11*(covj23**2+(-1)*covj22*covj33))*i1+ \
  aJ**3*(covi13**2*covi22+(-2)*covi12*covi13*covi23+covi12**2* \
  covi33+covi11*(covi23**2+(-1)*covi22*covi33))*j1+aI**2*aJ*( \
  covi22*covj13**2*i1+covi13*covj13*covj22*i1+covi33*(covj12**2+(-1) \
  *covj11*covj22)*i1+(-1)*covi13*covj12*covj23*i1+(-1)*covi12* \
  covj13*covj23*i1+2*covi23*((-1)*covj12*covj13+covj11*covj23)*i1+ \
  (-1)*covi22*covj11*covj33*i1+covi12*covj12*covj33*i1+covi13* \
  covj12*covj13*i2+(-1)*covi12*covj13**2*i2+(-1)*covi13*covj11* \
  covj23*i2+covi11*covj13*covj23*i2+covi12*covj11*covj33*i2+(-1)* \
  covi11*covj12*covj33*i2+(-1)*covi13*covj12**2*i3+covi12*covj12* \
  covj13*i3+covi13*covj11*covj22*i3+(-1)*covi11*covj13*covj22*i3+( \
  -1)*covi12*covj11*covj23*i3+covi11*covj12*covj23*i3+covi13* \
  covj13*covj22*j1+(-1)*covi13*covj12*covj23*j1+(-1)*covi12* \
  covj13*covj23*j1+covi11*covj23**2*j1+covi12*covj12*covj33*j1+(-1) \
  *covi11*covj22*covj33*j1+(-1)*covi13*covj12*covj13*j2+covi12* \
  covj13**2*j2+covi13*covj11*covj23*j2+(-1)*covi11*covj13*covj23* \
  j2+(-1)*covi12*covj11*covj33*j2+covi11*covj12*covj33*j2+(covi13* \
  covj12**2+(-1)*covi12*covj12*covj13+(-1)*covi13*covj11*covj22+ \
  covi11*covj13*covj22+covi12*covj11*covj23+(-1)*covi11*covj12* \
  covj23)*j3)+aI*aJ**2*(covi23**2*covj11*i1+covi12*covi33*covj12* \
  i1+covi12*covi33*covj11*i2+covi13**2*covj12*i2+(-1)*covi11* \
  covi33*covj12*i2+(-1)*covi12*covi13*covj13*i2+(-1)*covi12* \
  covi13*covj12*i3+covi12**2*covj13*i3+covi12*covi33*covj12*j1+ \
  covi13**2*covj22*j1+(-1)*covi11*covi33*covj22*j1+(-2)*covi12* \
  covi13*covj23*j1+covi12**2*covj33*j1+(-1)*covi12*covi33*covj11* \
  j2+(-1)*covi13**2*covj12*j2+covi11*covi33*covj12*j2+covi12* \
  covi13*covj13*j2+covi12*covi13*covj12*j3+(-1)*covi12**2*covj13* \
  j3+(-1)*covi23*(covi13*(covj12*(i1+j1)+covj11*(i2+(-1)*j2))+ \
  covi12*(covj13*(i1+j1)+covj11*(i3+(-1)*j3))+covi11*((-1)*covj13* \
  i2+(-1)*covj12*i3+(-2)*covj23*j1+covj13*j2+covj12*j3))+covi22*(( \
  -1)*covi33*covj11*i1+covi13*(covj13*(i1+j1)+covj11*(i3+(-1)*j3))+ \
  (-1)*covi11*(covj13*i3+covj33*j1+(-1)*covj13*j3))))+(-2)*covj12* \
  covj33*(aJ**3*(covi13**2*covi22+(-2)*covi12*covi13*covi23+ \
  covi12**2*covi33+covi11*(covi23**2+(-1)*covi22*covi33))+aI**2*aJ*( \
  (-2)*covi23*covj12*covj13+covi22*covj13**2+2*covi13*covj13* \
  covj22+covi33*(covj12**2+(-1)*covj11*covj22)+2*covi23*covj11* \
  covj23+(-2)*covi13*covj12*covj23+(-2)*covi12*covj13*covj23+ \
  covi11*covj23**2+(-1)*(covi22*covj11+(-2)*covi12*covj12+covi11* \
  covj22)*covj33)+aI*aJ**2*(covi23**2*covj11+2*covi12*covi33* \
  covj12+covi13**2*covj22+(-1)*covi11*covi33*covj22+(-2)*covi12* \
  covi13*covj23+(-2)*covi23*(covi13*covj12+covi12*covj13+(-1)* \
  covi11*covj23)+covi12**2*covj33+(-1)*covi22*(covi33*covj11+(-2)* \
  covi13*covj13+covi11*covj33))+aI**3*(covj13**2*covj22+(-2)*covj12* \
  covj13*covj23+covj12**2*covj33+covj11*(covj23**2+(-1)*covj22* \
  covj33)))*j1*(aI**3*(covj13**2*covj22+(-2)*covj12*covj13*covj23+ \
  covj12**2*covj33+covj11*(covj23**2+(-1)*covj22*covj33))*i2+aJ**3*( \
  covi13**2*covi22+(-2)*covi12*covi13*covi23+covi12**2*covi33+ \
  covi11*(covi23**2+(-1)*covi22*covi33))*j2+aI*aJ**2*(covi12* \
  covi33*covj22*i1+covi12*covi33*covj12*i2+covi13**2*covj22*i2+(-1) \
  *covi11*covi33*covj22*i2+(-1)*covi12*covi13*covj23*i2+(-1)* \
  covi12*covi13*covj22*i3+covi12**2*covj23*i3+(-1)*covi12*covi33* \
  covj22*j1+covi12*covi33*covj12*j2+(-1)*covi12*covi13*covj23*j2+ \
  covi12**2*covj33*j2+covi23**2*(covj12*(i1+(-1)*j1)+covj11*j2)+ \
  covi12*covi13*covj22*j3+(-1)*covi12**2*covj23*j3+covi23*((-1)* \
  covi13*(covj22*(i1+(-1)*j1)+covj12*(i2+j2))+covi11*(covj23*(i2+j2) \
  +covj22*(i3+(-1)*j3))+covi12*((-1)*covj23*i1+(-1)*covj12*i3+ \
  covj23*j1+(-2)*covj13*j2+covj12*j3))+(-1)*covi22*(covi33*( \
  covj12*(i1+(-1)*j1)+covj11*j2)+covi13*((-1)*covj23*i1+(-1)* \
  covj12*i3+covj23*j1+(-2)*covj13*j2+covj12*j3)+covi11*(covj23*i3+ \
  covj33*j2+(-1)*covj23*j3)))+aI**2*aJ*((-1)*covi12*covj23**2*i1+ \
  covi12*covj22*covj33*i1+covi33*covj12**2*i2+(-1)*covi33*covj11* \
  covj22*i2+2*covi13*covj13*covj22*i2+(-2)*covi13*covj12*covj23* \
  i2+(-1)*covi12*covj13*covj23*i2+covi11*covj23**2*i2+covi12* \
  covj12*covj33*i2+(-1)*covi11*covj22*covj33*i2+(-1)*covi12* \
  covj13*covj22*i3+covi12*covj12*covj23*i3+covi12*covj23**2*j1+(-1) \
  *covi12*covj22*covj33*j1+(-1)*covi12*covj13*covj23*j2+covi12* \
  covj12*covj33*j2+covi12*covj13*covj22*j3+(-1)*covi12*covj12* \
  covj23*j3+covi23*(covj12*covj23*(i1+(-1)*j1)+(-1)*covj13*( \
  covj22*(i1+(-1)*j1)+covj12*(i2+j2))+covj11*(covj23*(i2+j2)+covj22* \
  (i3+(-1)*j3))+covj12**2*((-1)*i3+j3))+covi22*(covj12*covj33*((-1) \
  *i1+j1)+covj13**2*j2+covj13*(covj23*(i1+(-1)*j1)+covj12*(i3+(-1)* \
  j3))+(-1)*covj11*(covj23*i3+covj33*j2+(-1)*covj23*j3))))+(-2)* \
  covj13**2*(aJ**3*(covi13**2*covi22+(-2)*covi12*covi13*covi23+ \
  covi12**2*covi33+covi11*(covi23**2+(-1)*covi22*covi33))+aI**2*aJ*( \
  (-2)*covi23*covj12*covj13+covi22*covj13**2+2*covi13*covj13* \
  covj22+covi33*(covj12**2+(-1)*covj11*covj22)+2*covi23*covj11* \
  covj23+(-2)*covi13*covj12*covj23+(-2)*covi12*covj13*covj23+ \
  covi11*covj23**2+(-1)*(covi22*covj11+(-2)*covi12*covj12+covi11* \
  covj22)*covj33)+aI*aJ**2*(covi23**2*covj11+2*covi12*covi33* \
  covj12+covi13**2*covj22+(-1)*covi11*covi33*covj22+(-2)*covi12* \
  covi13*covj23+(-2)*covi23*(covi13*covj12+covi12*covj13+(-1)* \
  covi11*covj23)+covi12**2*covj33+(-1)*covi22*(covi33*covj11+(-2)* \
  covi13*covj13+covi11*covj33))+aI**3*(covj13**2*covj22+(-2)*covj12* \
  covj13*covj23+covj12**2*covj33+covj11*(covj23**2+(-1)*covj22* \
  covj33)))*j2*(aI**3*(covj13**2*covj22+(-2)*covj12*covj13*covj23+ \
  covj12**2*covj33+covj11*(covj23**2+(-1)*covj22*covj33))*i2+aJ**3*( \
  covi13**2*covi22+(-2)*covi12*covi13*covi23+covi12**2*covi33+ \
  covi11*(covi23**2+(-1)*covi22*covi33))*j2+aI*aJ**2*(covi12* \
  covi33*covj22*i1+covi12*covi33*covj12*i2+covi13**2*covj22*i2+(-1) \
  *covi11*covi33*covj22*i2+(-1)*covi12*covi13*covj23*i2+(-1)* \
  covi12*covi13*covj22*i3+covi12**2*covj23*i3+(-1)*covi12*covi33* \
  covj22*j1+covi12*covi33*covj12*j2+(-1)*covi12*covi13*covj23*j2+ \
  covi12**2*covj33*j2+covi23**2*(covj12*(i1+(-1)*j1)+covj11*j2)+ \
  covi12*covi13*covj22*j3+(-1)*covi12**2*covj23*j3+covi23*((-1)* \
  covi13*(covj22*(i1+(-1)*j1)+covj12*(i2+j2))+covi11*(covj23*(i2+j2) \
  +covj22*(i3+(-1)*j3))+covi12*((-1)*covj23*i1+(-1)*covj12*i3+ \
  covj23*j1+(-2)*covj13*j2+covj12*j3))+(-1)*covi22*(covi33*( \
  covj12*(i1+(-1)*j1)+covj11*j2)+covi13*((-1)*covj23*i1+(-1)* \
  covj12*i3+covj23*j1+(-2)*covj13*j2+covj12*j3)+covi11*(covj23*i3+ \
  covj33*j2+(-1)*covj23*j3)))+aI**2*aJ*((-1)*covi12*covj23**2*i1+ \
  covi12*covj22*covj33*i1+covi33*covj12**2*i2+(-1)*covi33*covj11* \
  covj22*i2+2*covi13*covj13*covj22*i2+(-2)*covi13*covj12*covj23* \
  i2+(-1)*covi12*covj13*covj23*i2+covi11*covj23**2*i2+covi12* \
  covj12*covj33*i2+(-1)*covi11*covj22*covj33*i2+(-1)*covi12* \
  covj13*covj22*i3+covi12*covj12*covj23*i3+covi12*covj23**2*j1+(-1) \
  *covi12*covj22*covj33*j1+(-1)*covi12*covj13*covj23*j2+covi12* \
  covj12*covj33*j2+covi12*covj13*covj22*j3+(-1)*covi12*covj12* \
  covj23*j3+covi23*(covj12*covj23*(i1+(-1)*j1)+(-1)*covj13*( \
  covj22*(i1+(-1)*j1)+covj12*(i2+j2))+covj11*(covj23*(i2+j2)+covj22* \
  (i3+(-1)*j3))+covj12**2*((-1)*i3+j3))+covi22*(covj12*covj33*((-1) \
  *i1+j1)+covj13**2*j2+covj13*(covj23*(i1+(-1)*j1)+covj12*(i3+(-1)* \
  j3))+(-1)*covj11*(covj23*i3+covj33*j2+(-1)*covj23*j3))))+2* \
  covj11*covj33*(aJ**3*(covi13**2*covi22+(-2)*covi12*covi13*covi23+ \
  covi12**2*covi33+covi11*(covi23**2+(-1)*covi22*covi33))+aI**2*aJ*( \
  (-2)*covi23*covj12*covj13+covi22*covj13**2+2*covi13*covj13* \
  covj22+covi33*(covj12**2+(-1)*covj11*covj22)+2*covi23*covj11* \
  covj23+(-2)*covi13*covj12*covj23+(-2)*covi12*covj13*covj23+ \
  covi11*covj23**2+(-1)*(covi22*covj11+(-2)*covi12*covj12+covi11* \
  covj22)*covj33)+aI*aJ**2*(covi23**2*covj11+2*covi12*covi33* \
  covj12+covi13**2*covj22+(-1)*covi11*covi33*covj22+(-2)*covi12* \
  covi13*covj23+(-2)*covi23*(covi13*covj12+covi12*covj13+(-1)* \
  covi11*covj23)+covi12**2*covj33+(-1)*covi22*(covi33*covj11+(-2)* \
  covi13*covj13+covi11*covj33))+aI**3*(covj13**2*covj22+(-2)*covj12* \
  covj13*covj23+covj12**2*covj33+covj11*(covj23**2+(-1)*covj22* \
  covj33)))*j2*(aI**3*(covj13**2*covj22+(-2)*covj12*covj13*covj23+ \
  covj12**2*covj33+covj11*(covj23**2+(-1)*covj22*covj33))*i2+aJ**3*( \
  covi13**2*covi22+(-2)*covi12*covi13*covi23+covi12**2*covi33+ \
  covi11*(covi23**2+(-1)*covi22*covi33))*j2+aI*aJ**2*(covi12* \
  covi33*covj22*i1+covi12*covi33*covj12*i2+covi13**2*covj22*i2+(-1) \
  *covi11*covi33*covj22*i2+(-1)*covi12*covi13*covj23*i2+(-1)* \
  covi12*covi13*covj22*i3+covi12**2*covj23*i3+(-1)*covi12*covi33* \
  covj22*j1+covi12*covi33*covj12*j2+(-1)*covi12*covi13*covj23*j2+ \
  covi12**2*covj33*j2+covi23**2*(covj12*(i1+(-1)*j1)+covj11*j2)+ \
  covi12*covi13*covj22*j3+(-1)*covi12**2*covj23*j3+covi23*((-1)* \
  covi13*(covj22*(i1+(-1)*j1)+covj12*(i2+j2))+covi11*(covj23*(i2+j2) \
  +covj22*(i3+(-1)*j3))+covi12*((-1)*covj23*i1+(-1)*covj12*i3+ \
  covj23*j1+(-2)*covj13*j2+covj12*j3))+(-1)*covi22*(covi33*( \
  covj12*(i1+(-1)*j1)+covj11*j2)+covi13*((-1)*covj23*i1+(-1)* \
  covj12*i3+covj23*j1+(-2)*covj13*j2+covj12*j3)+covi11*(covj23*i3+ \
  covj33*j2+(-1)*covj23*j3)))+aI**2*aJ*((-1)*covi12*covj23**2*i1+ \
  covi12*covj22*covj33*i1+covi33*covj12**2*i2+(-1)*covi33*covj11* \
  covj22*i2+2*covi13*covj13*covj22*i2+(-2)*covi13*covj12*covj23* \
  i2+(-1)*covi12*covj13*covj23*i2+covi11*covj23**2*i2+covi12* \
  covj12*covj33*i2+(-1)*covi11*covj22*covj33*i2+(-1)*covi12* \
  covj13*covj22*i3+covi12*covj12*covj23*i3+covi12*covj23**2*j1+(-1) \
  *covi12*covj22*covj33*j1+(-1)*covi12*covj13*covj23*j2+covi12* \
  covj12*covj33*j2+covi12*covj13*covj22*j3+(-1)*covi12*covj12* \
  covj23*j3+covi23*(covj12*covj23*(i1+(-1)*j1)+(-1)*covj13*( \
  covj22*(i1+(-1)*j1)+covj12*(i2+j2))+covj11*(covj23*(i2+j2)+covj22* \
  (i3+(-1)*j3))+covj12**2*((-1)*i3+j3))+covi22*(covj12*covj33*((-1) \
  *i1+j1)+covj13**2*j2+covj13*(covj23*(i1+(-1)*j1)+covj12*(i3+(-1)* \
  j3))+(-1)*covj11*(covj23*i3+covj33*j2+(-1)*covj23*j3))))+2* \
  covj12*covj13*(aJ**3*(covi13**2*covi22+(-2)*covi12*covi13*covi23+ \
  covi12**2*covi33+covi11*(covi23**2+(-1)*covi22*covi33))+aI**2*aJ*( \
  (-2)*covi23*covj12*covj13+covi22*covj13**2+2*covi13*covj13* \
  covj22+covi33*(covj12**2+(-1)*covj11*covj22)+2*covi23*covj11* \
  covj23+(-2)*covi13*covj12*covj23+(-2)*covi12*covj13*covj23+ \
  covi11*covj23**2+(-1)*(covi22*covj11+(-2)*covi12*covj12+covi11* \
  covj22)*covj33)+aI*aJ**2*(covi23**2*covj11+2*covi12*covi33* \
  covj12+covi13**2*covj22+(-1)*covi11*covi33*covj22+(-2)*covi12* \
  covi13*covj23+(-2)*covi23*(covi13*covj12+covi12*covj13+(-1)* \
  covi11*covj23)+covi12**2*covj33+(-1)*covi22*(covi33*covj11+(-2)* \
  covi13*covj13+covi11*covj33))+aI**3*(covj13**2*covj22+(-2)*covj12* \
  covj13*covj23+covj12**2*covj33+covj11*(covj23**2+(-1)*covj22* \
  covj33)))*j3*(aI**3*(covj13**2*covj22+(-2)*covj12*covj13*covj23+ \
  covj12**2*covj33+covj11*(covj23**2+(-1)*covj22*covj33))*i2+aJ**3*( \
  covi13**2*covi22+(-2)*covi12*covi13*covi23+covi12**2*covi33+ \
  covi11*(covi23**2+(-1)*covi22*covi33))*j2+aI*aJ**2*(covi12* \
  covi33*covj22*i1+covi12*covi33*covj12*i2+covi13**2*covj22*i2+(-1) \
  *covi11*covi33*covj22*i2+(-1)*covi12*covi13*covj23*i2+(-1)* \
  covi12*covi13*covj22*i3+covi12**2*covj23*i3+(-1)*covi12*covi33* \
  covj22*j1+covi12*covi33*covj12*j2+(-1)*covi12*covi13*covj23*j2+ \
  covi12**2*covj33*j2+covi23**2*(covj12*(i1+(-1)*j1)+covj11*j2)+ \
  covi12*covi13*covj22*j3+(-1)*covi12**2*covj23*j3+covi23*((-1)* \
  covi13*(covj22*(i1+(-1)*j1)+covj12*(i2+j2))+covi11*(covj23*(i2+j2) \
  +covj22*(i3+(-1)*j3))+covi12*((-1)*covj23*i1+(-1)*covj12*i3+ \
  covj23*j1+(-2)*covj13*j2+covj12*j3))+(-1)*covi22*(covi33*( \
  covj12*(i1+(-1)*j1)+covj11*j2)+covi13*((-1)*covj23*i1+(-1)* \
  covj12*i3+covj23*j1+(-2)*covj13*j2+covj12*j3)+covi11*(covj23*i3+ \
  covj33*j2+(-1)*covj23*j3)))+aI**2*aJ*((-1)*covi12*covj23**2*i1+ \
  covi12*covj22*covj33*i1+covi33*covj12**2*i2+(-1)*covi33*covj11* \
  covj22*i2+2*covi13*covj13*covj22*i2+(-2)*covi13*covj12*covj23* \
  i2+(-1)*covi12*covj13*covj23*i2+covi11*covj23**2*i2+covi12* \
  covj12*covj33*i2+(-1)*covi11*covj22*covj33*i2+(-1)*covi12* \
  covj13*covj22*i3+covi12*covj12*covj23*i3+covi12*covj23**2*j1+(-1) \
  *covi12*covj22*covj33*j1+(-1)*covi12*covj13*covj23*j2+covi12* \
  covj12*covj33*j2+covi12*covj13*covj22*j3+(-1)*covi12*covj12* \
  covj23*j3+covi23*(covj12*covj23*(i1+(-1)*j1)+(-1)*covj13*( \
  covj22*(i1+(-1)*j1)+covj12*(i2+j2))+covj11*(covj23*(i2+j2)+covj22* \
  (i3+(-1)*j3))+covj12**2*((-1)*i3+j3))+covi22*(covj12*covj33*((-1) \
  *i1+j1)+covj13**2*j2+covj13*(covj23*(i1+(-1)*j1)+covj12*(i3+(-1)* \
  j3))+(-1)*covj11*(covj23*i3+covj33*j2+(-1)*covj23*j3))))+2* \
  covj12*covj33*(aI**3*(covj13**2*covj22+(-2)*covj12*covj13*covj23+ \
  covj12**2*covj33+covj11*(covj23**2+(-1)*covj22*covj33))*i1+aJ**3*( \
  covi13**2*covi22+(-2)*covi12*covi13*covi23+covi12**2*covi33+ \
  covi11*(covi23**2+(-1)*covi22*covi33))*j1+aI**2*aJ*(covi22* \
  covj13**2*i1+covi13*covj13*covj22*i1+covi33*(covj12**2+(-1)* \
  covj11*covj22)*i1+(-1)*covi13*covj12*covj23*i1+(-1)*covi12* \
  covj13*covj23*i1+2*covi23*((-1)*covj12*covj13+covj11*covj23)*i1+ \
  (-1)*covi22*covj11*covj33*i1+covi12*covj12*covj33*i1+covi13* \
  covj12*covj13*i2+(-1)*covi12*covj13**2*i2+(-1)*covi13*covj11* \
  covj23*i2+covi11*covj13*covj23*i2+covi12*covj11*covj33*i2+(-1)* \
  covi11*covj12*covj33*i2+(-1)*covi13*covj12**2*i3+covi12*covj12* \
  covj13*i3+covi13*covj11*covj22*i3+(-1)*covi11*covj13*covj22*i3+( \
  -1)*covi12*covj11*covj23*i3+covi11*covj12*covj23*i3+covi13* \
  covj13*covj22*j1+(-1)*covi13*covj12*covj23*j1+(-1)*covi12* \
  covj13*covj23*j1+covi11*covj23**2*j1+covi12*covj12*covj33*j1+(-1) \
  *covi11*covj22*covj33*j1+(-1)*covi13*covj12*covj13*j2+covi12* \
  covj13**2*j2+covi13*covj11*covj23*j2+(-1)*covi11*covj13*covj23* \
  j2+(-1)*covi12*covj11*covj33*j2+covi11*covj12*covj33*j2+(covi13* \
  covj12**2+(-1)*covi12*covj12*covj13+(-1)*covi13*covj11*covj22+ \
  covi11*covj13*covj22+covi12*covj11*covj23+(-1)*covi11*covj12* \
  covj23)*j3)+aI*aJ**2*(covi23**2*covj11*i1+covi12*covi33*covj12* \
  i1+covi12*covi33*covj11*i2+covi13**2*covj12*i2+(-1)*covi11* \
  covi33*covj12*i2+(-1)*covi12*covi13*covj13*i2+(-1)*covi12* \
  covi13*covj12*i3+covi12**2*covj13*i3+covi12*covi33*covj12*j1+ \
  covi13**2*covj22*j1+(-1)*covi11*covi33*covj22*j1+(-2)*covi12* \
  covi13*covj23*j1+covi12**2*covj33*j1+(-1)*covi12*covi33*covj11* \
  j2+(-1)*covi13**2*covj12*j2+covi11*covi33*covj12*j2+covi12* \
  covi13*covj13*j2+covi12*covi13*covj12*j3+(-1)*covi12**2*covj13* \
  j3+(-1)*covi23*(covi13*(covj12*(i1+j1)+covj11*(i2+(-1)*j2))+ \
  covi12*(covj13*(i1+j1)+covj11*(i3+(-1)*j3))+covi11*((-1)*covj13* \
  i2+(-1)*covj12*i3+(-2)*covj23*j1+covj13*j2+covj12*j3))+covi22*(( \
  -1)*covi33*covj11*i1+covi13*(covj13*(i1+j1)+covj11*(i3+(-1)*j3))+ \
  (-1)*covi11*(covj13*i3+covj33*j1+(-1)*covj13*j3))))*(aI**3*( \
  covj13**2*covj22+(-2)*covj12*covj13*covj23+covj12**2*covj33+ \
  covj11*(covj23**2+(-1)*covj22*covj33))*i2+aJ**3*(covi13**2*covi22+ \
  (-2)*covi12*covi13*covi23+covi12**2*covi33+covi11*(covi23**2+(-1)* \
  covi22*covi33))*j2+aI*aJ**2*(covi12*covi33*covj22*i1+covi12* \
  covi33*covj12*i2+covi13**2*covj22*i2+(-1)*covi11*covi33*covj22* \
  i2+(-1)*covi12*covi13*covj23*i2+(-1)*covi12*covi13*covj22*i3+ \
  covi12**2*covj23*i3+(-1)*covi12*covi33*covj22*j1+covi12*covi33* \
  covj12*j2+(-1)*covi12*covi13*covj23*j2+covi12**2*covj33*j2+ \
  covi23**2*(covj12*(i1+(-1)*j1)+covj11*j2)+covi12*covi13*covj22* \
  j3+(-1)*covi12**2*covj23*j3+covi23*((-1)*covi13*(covj22*(i1+(-1) \
  *j1)+covj12*(i2+j2))+covi11*(covj23*(i2+j2)+covj22*(i3+(-1)*j3))+ \
  covi12*((-1)*covj23*i1+(-1)*covj12*i3+covj23*j1+(-2)*covj13*j2+ \
  covj12*j3))+(-1)*covi22*(covi33*(covj12*(i1+(-1)*j1)+covj11*j2)+ \
  covi13*((-1)*covj23*i1+(-1)*covj12*i3+covj23*j1+(-2)*covj13*j2+ \
  covj12*j3)+covi11*(covj23*i3+covj33*j2+(-1)*covj23*j3)))+aI**2* \
  aJ*((-1)*covi12*covj23**2*i1+covi12*covj22*covj33*i1+covi33* \
  covj12**2*i2+(-1)*covi33*covj11*covj22*i2+2*covi13*covj13* \
  covj22*i2+(-2)*covi13*covj12*covj23*i2+(-1)*covi12*covj13* \
  covj23*i2+covi11*covj23**2*i2+covi12*covj12*covj33*i2+(-1)* \
  covi11*covj22*covj33*i2+(-1)*covi12*covj13*covj22*i3+covi12* \
  covj12*covj23*i3+covi12*covj23**2*j1+(-1)*covi12*covj22*covj33* \
  j1+(-1)*covi12*covj13*covj23*j2+covi12*covj12*covj33*j2+covi12* \
  covj13*covj22*j3+(-1)*covi12*covj12*covj23*j3+covi23*(covj12* \
  covj23*(i1+(-1)*j1)+(-1)*covj13*(covj22*(i1+(-1)*j1)+covj12*(i2+ \
  j2))+covj11*(covj23*(i2+j2)+covj22*(i3+(-1)*j3))+covj12**2*((-1)* \
  i3+j3))+covi22*(covj12*covj33*((-1)*i1+j1)+covj13**2*j2+covj13*( \
  covj23*(i1+(-1)*j1)+covj12*(i3+(-1)*j3))+(-1)*covj11*(covj23*i3+ \
  covj33*j2+(-1)*covj23*j3))))+covj13**2*(aI**3*(covj13**2*covj22+( \
  -2)*covj12*covj13*covj23+covj12**2*covj33+covj11*(covj23**2+(-1)* \
  covj22*covj33))*i2+aJ**3*(covi13**2*covi22+(-2)*covi12*covi13* \
  covi23+covi12**2*covi33+covi11*(covi23**2+(-1)*covi22*covi33))*j2+ \
  aI*aJ**2*(covi12*covi33*covj22*i1+covi12*covi33*covj12*i2+ \
  covi13**2*covj22*i2+(-1)*covi11*covi33*covj22*i2+(-1)*covi12* \
  covi13*covj23*i2+(-1)*covi12*covi13*covj22*i3+covi12**2*covj23* \
  i3+(-1)*covi12*covi33*covj22*j1+covi12*covi33*covj12*j2+(-1)* \
  covi12*covi13*covj23*j2+covi12**2*covj33*j2+covi23**2*(covj12*( \
  i1+(-1)*j1)+covj11*j2)+covi12*covi13*covj22*j3+(-1)*covi12**2* \
  covj23*j3+covi23*((-1)*covi13*(covj22*(i1+(-1)*j1)+covj12*(i2+j2) \
  )+covi11*(covj23*(i2+j2)+covj22*(i3+(-1)*j3))+covi12*((-1)* \
  covj23*i1+(-1)*covj12*i3+covj23*j1+(-2)*covj13*j2+covj12*j3))+( \
  -1)*covi22*(covi33*(covj12*(i1+(-1)*j1)+covj11*j2)+covi13*((-1)* \
  covj23*i1+(-1)*covj12*i3+covj23*j1+(-2)*covj13*j2+covj12*j3)+ \
  covi11*(covj23*i3+covj33*j2+(-1)*covj23*j3)))+aI**2*aJ*((-1)* \
  covi12*covj23**2*i1+covi12*covj22*covj33*i1+covi33*covj12**2*i2+( \
  -1)*covi33*covj11*covj22*i2+2*covi13*covj13*covj22*i2+(-2)* \
  covi13*covj12*covj23*i2+(-1)*covi12*covj13*covj23*i2+covi11* \
  covj23**2*i2+covi12*covj12*covj33*i2+(-1)*covi11*covj22*covj33* \
  i2+(-1)*covi12*covj13*covj22*i3+covi12*covj12*covj23*i3+covi12* \
  covj23**2*j1+(-1)*covi12*covj22*covj33*j1+(-1)*covi12*covj13* \
  covj23*j2+covi12*covj12*covj33*j2+covi12*covj13*covj22*j3+(-1)* \
  covi12*covj12*covj23*j3+covi23*(covj12*covj23*(i1+(-1)*j1)+(-1)* \
  covj13*(covj22*(i1+(-1)*j1)+covj12*(i2+j2))+covj11*(covj23*(i2+j2) \
  +covj22*(i3+(-1)*j3))+covj12**2*((-1)*i3+j3))+covi22*(covj12* \
  covj33*((-1)*i1+j1)+covj13**2*j2+covj13*(covj23*(i1+(-1)*j1)+ \
  covj12*(i3+(-1)*j3))+(-1)*covj11*(covj23*i3+covj33*j2+(-1)* \
  covj23*j3))))**2+(-1)*covj11*covj33*(aI**3*(covj13**2*covj22+(-2) \
  *covj12*covj13*covj23+covj12**2*covj33+covj11*(covj23**2+(-1)* \
  covj22*covj33))*i2+aJ**3*(covi13**2*covi22+(-2)*covi12*covi13* \
  covi23+covi12**2*covi33+covi11*(covi23**2+(-1)*covi22*covi33))*j2+ \
  aI*aJ**2*(covi12*covi33*covj22*i1+covi12*covi33*covj12*i2+ \
  covi13**2*covj22*i2+(-1)*covi11*covi33*covj22*i2+(-1)*covi12* \
  covi13*covj23*i2+(-1)*covi12*covi13*covj22*i3+covi12**2*covj23* \
  i3+(-1)*covi12*covi33*covj22*j1+covi12*covi33*covj12*j2+(-1)* \
  covi12*covi13*covj23*j2+covi12**2*covj33*j2+covi23**2*(covj12*( \
  i1+(-1)*j1)+covj11*j2)+covi12*covi13*covj22*j3+(-1)*covi12**2* \
  covj23*j3+covi23*((-1)*covi13*(covj22*(i1+(-1)*j1)+covj12*(i2+j2) \
  )+covi11*(covj23*(i2+j2)+covj22*(i3+(-1)*j3))+covi12*((-1)* \
  covj23*i1+(-1)*covj12*i3+covj23*j1+(-2)*covj13*j2+covj12*j3))+( \
  -1)*covi22*(covi33*(covj12*(i1+(-1)*j1)+covj11*j2)+covi13*((-1)* \
  covj23*i1+(-1)*covj12*i3+covj23*j1+(-2)*covj13*j2+covj12*j3)+ \
  covi11*(covj23*i3+covj33*j2+(-1)*covj23*j3)))+aI**2*aJ*((-1)* \
  covi12*covj23**2*i1+covi12*covj22*covj33*i1+covi33*covj12**2*i2+( \
  -1)*covi33*covj11*covj22*i2+2*covi13*covj13*covj22*i2+(-2)* \
  covi13*covj12*covj23*i2+(-1)*covi12*covj13*covj23*i2+covi11* \
  covj23**2*i2+covi12*covj12*covj33*i2+(-1)*covi11*covj22*covj33* \
  i2+(-1)*covi12*covj13*covj22*i3+covi12*covj12*covj23*i3+covi12* \
  covj23**2*j1+(-1)*covi12*covj22*covj33*j1+(-1)*covi12*covj13* \
  covj23*j2+covi12*covj12*covj33*j2+covi12*covj13*covj22*j3+(-1)* \
  covi12*covj12*covj23*j3+covi23*(covj12*covj23*(i1+(-1)*j1)+(-1)* \
  covj13*(covj22*(i1+(-1)*j1)+covj12*(i2+j2))+covj11*(covj23*(i2+j2) \
  +covj22*(i3+(-1)*j3))+covj12**2*((-1)*i3+j3))+covi22*(covj12* \
  covj33*((-1)*i1+j1)+covj13**2*j2+covj13*(covj23*(i1+(-1)*j1)+ \
  covj12*(i3+(-1)*j3))+(-1)*covj11*(covj23*i3+covj33*j2+(-1)* \
  covj23*j3))))**2+(-2)*covj12*(aJ**3*(covi13**2*covi22+(-2)* \
  covi12*covi13*covi23+covi12**2*covi33+covi11*(covi23**2+(-1)* \
  covi22*covi33))*covj12*j3+aI**3*(covj13**2*covj22+(-2)*covj12* \
  covj13*covj23+covj12**2*covj33+covj11*(covj23**2+(-1)*covj22* \
  covj33))*(covj13*(i2+(-1)*j2)+covj12*j3)+aI**2*aJ*((-1)*covi12* \
  covj13*covj23**2*i1+covi12*covj13*covj22*covj33*i1+covi33* \
  covj12**2*covj13*i2+(-1)*covi33*covj11*covj13*covj22*i2+2* \
  covi13*covj13**2*covj22*i2+(-2)*covi13*covj12*covj13*covj23*i2+( \
  -1)*covi12*covj13**2*covj23*i2+covi11*covj13*covj23**2*i2+ \
  covi12*covj12*covj13*covj33*i2+(-1)*covi11*covj13*covj22* \
  covj33*i2+(-1)*covi12*covj13**2*covj22*i3+covi12*covj12*covj13* \
  covj23*i3+covi12*covj13*covj23**2*j1+(-1)*covi12*covj13*covj22* \
  covj33*j1+covi22*covj13*((-1)*covj11*covj23*i3+covj13*(covj12* \
  i3+covj23*(i1+(-1)*j1))+covj12*covj33*((-1)*i1+j1))+(-1)*covi33* \
  covj12**2*covj13*j2+covi33*covj11*covj13*covj22*j2+(-2)*covi13* \
  covj13**2*covj22*j2+2*covi13*covj12*covj13*covj23*j2+covi12* \
  covj13**2*covj23*j2+(-1)*covi11*covj13*covj23**2*j2+(-1)*covi12* \
  covj12*covj13*covj33*j2+covi11*covj13*covj22*covj33*j2+covi23* \
  covj13*((-1)*covj12**2*i3+covj12*covj23*(i1+(-1)*j1)+covj11*( \
  covj23*i2+covj22*i3+(-1)*covj23*j2)+covj13*(covj22*((-1)*i1+j1)+ \
  covj12*((-1)*i2+j2)))+(-1)*covi23*(covj12**2*covj13+covj11* \
  covj13*covj22+(-2)*covj11*covj12*covj23)*j3+covi22*covj11*( \
  covj13*covj23+(-1)*covj12*covj33)*j3+(covi12*covj13**2*covj22+ \
  covi33*(covj12**3+(-1)*covj11*covj12*covj22)+(-3)*covi12*covj12* \
  covj13*covj23+covi11*covj12*covj23**2+2*covi13*covj12*(covj13* \
  covj22+(-1)*covj12*covj23)+covj12*(2*covi12*covj12+(-1)*covi11* \
  covj22)*covj33)*j3)+aI*aJ**2*(covi23*covj13*((-1)*covi12*( \
  covj23*i1+covj12*i3+(-1)*covj23*j1)+covi11*(covj23*i2+covj22*i3+( \
  -1)*covj23*j2)+covi13*(covj22*((-1)*i1+j1)+covj12*((-1)*i2+j2)))+ \
  covj13*(covi12**2*covj23*i3+covi22*(covi13*covj23*i1+covi13* \
  covj12*i3+(-1)*covi11*covj23*i3+(-1)*covi13*covj23*j1+covi33* \
  covj12*((-1)*i1+j1))+(covi13**2+(-1)*covi11*covi33)*covj22*(i2+( \
  -1)*j2)+covi12*(covi33*(covj22*i1+covj12*i2+(-1)*covj22*j1+(-1)* \
  covj12*j2)+(-1)*covi13*(covj23*i2+covj22*i3+(-1)*covj23*j2)))+( \
  -1)*covi23*(2*covi13*covj12**2+covi12*covj12*covj13+covi11* \
  covj13*covj22+(-2)*covi11*covj12*covj23)*j3+((covi13**2+(-1)* \
  covi11*covi33)*covj12*covj22+covi12*(2*covi33*covj12**2+covi13* \
  covj13*covj22+(-2)*covi13*covj12*covj23)+covi12**2*((-1)*covj13* \
  covj23+covj12*covj33)+covi22*((-1)*covi33*covj11*covj12+covi13* \
  covj12*covj13+covi11*covj13*covj23+(-1)*covi11*covj12*covj33))* \
  j3+covi23**2*covj12*(covj13*(i1+(-1)*j1)+covj11*j3)))*(aI**3*( \
  covj13**2*covj22+(-2)*covj12*covj13*covj23+covj12**2*covj33+ \
  covj11*(covj23**2+(-1)*covj22*covj33))*i3+aJ**3*(covi13**2*covi22+ \
  (-2)*covi12*covi13*covi23+covi12**2*covi33+covi11*(covi23**2+(-1)* \
  covi22*covi33))*j3+aI*aJ**2*((-1)*covi22*covi33*covj13*i1+ \
  covi12*covi33*covj23*i1+covi13*covi22*covj33*i1+covi12*covi33* \
  covj13*i2+covi13**2*covj23*i2+(-1)*covi11*covi33*covj23*i2+(-1)* \
  covi12*covi13*covj33*i2+covi13*covi22*covj13*i3+(-1)*covi12* \
  covi13*covj23*i3+covi12**2*covj33*i3+(-1)*covi11*covi22*covj33* \
  i3+covi22*covi33*covj13*j1+(-1)*covi12*covi33*covj23*j1+(-1)* \
  covi13*covi22*covj33*j1+(-1)*covi12*covi33*covj13*j2+(-1)* \
  covi13**2*covj23*j2+covi11*covi33*covj23*j2+covi12*covi13* \
  covj33*j2+((-1)*covi22*covi33*covj11+2*covi12*covi33*covj12+ \
  covi13*covi22*covj13+covi13**2*covj22+(-1)*covi11*covi33*covj22+( \
  -1)*covi12*covi13*covj23)*j3+covi23**2*(covj13*(i1+(-1)*j1)+ \
  covj11*j3)+covi23*(covi13*((-1)*covj13*i2+covj23*((-1)*i1+j1)+ \
  covj13*j2+(-2)*covj12*j3)+(-1)*covi12*(covj33*(i1+(-1)*j1)+ \
  covj13*(i3+j3))+covi11*(covj33*(i2+(-1)*j2)+covj23*(i3+j3))))+ \
  aI**2*aJ*((-1)*covi13*covj23**2*i1+covi13*covj22*covj33*i1+ \
  covi13*covj13*covj23*i2+(-1)*covi13*covj12*covj33*i2+covi22* \
  covj13**2*i3+covi13*covj13*covj22*i3+(-1)*covi13*covj12*covj23* \
  i3+(-2)*covi12*covj13*covj23*i3+covi11*covj23**2*i3+(-1)*covi22* \
  covj11*covj33*i3+2*covi12*covj12*covj33*i3+(-1)*covi11*covj22* \
  covj33*i3+covi13*covj23**2*j1+(-1)*covi13*covj22*covj33*j1+(-1)* \
  covi13*covj13*covj23*j2+covi13*covj12*covj33*j2+covi13*covj13* \
  covj22*j3+(-1)*covi13*covj12*covj23*j3+covi33*(covj12*covj23*( \
  i1+(-1)*j1)+covj13*(covj22*((-1)*i1+j1)+covj12*(i2+(-1)*j2))+ \
  covj12**2*j3+(-1)*covj11*(covj23*i2+(-1)*covj23*j2+covj22*j3))+ \
  covi23*(covj12*covj33*((-1)*i1+j1)+covj13**2*((-1)*i2+j2)+covj13* \
  (covj23*(i1+(-1)*j1)+(-1)*covj12*(i3+j3))+covj11*(covj33*(i2+(-1) \
  *j2)+covj23*(i3+j3)))))+covj12**2*(aI**3*(covj13**2*covj22+(-2)* \
  covj12*covj13*covj23+covj12**2*covj33+covj11*(covj23**2+(-1)* \
  covj22*covj33))*i3+aJ**3*(covi13**2*covi22+(-2)*covi12*covi13* \
  covi23+covi12**2*covi33+covi11*(covi23**2+(-1)*covi22*covi33))*j3+ \
  aI*aJ**2*((-1)*covi22*covi33*covj13*i1+covi12*covi33*covj23*i1+ \
  covi13*covi22*covj33*i1+covi12*covi33*covj13*i2+covi13**2* \
  covj23*i2+(-1)*covi11*covi33*covj23*i2+(-1)*covi12*covi13* \
  covj33*i2+covi13*covi22*covj13*i3+(-1)*covi12*covi13*covj23*i3+ \
  covi12**2*covj33*i3+(-1)*covi11*covi22*covj33*i3+covi22*covi33* \
  covj13*j1+(-1)*covi12*covi33*covj23*j1+(-1)*covi13*covi22* \
  covj33*j1+(-1)*covi12*covi33*covj13*j2+(-1)*covi13**2*covj23*j2+ \
  covi11*covi33*covj23*j2+covi12*covi13*covj33*j2+((-1)*covi22* \
  covi33*covj11+2*covi12*covi33*covj12+covi13*covi22*covj13+ \
  covi13**2*covj22+(-1)*covi11*covi33*covj22+(-1)*covi12*covi13* \
  covj23)*j3+covi23**2*(covj13*(i1+(-1)*j1)+covj11*j3)+covi23*( \
  covi13*((-1)*covj13*i2+covj23*((-1)*i1+j1)+covj13*j2+(-2)* \
  covj12*j3)+(-1)*covi12*(covj33*(i1+(-1)*j1)+covj13*(i3+j3))+ \
  covi11*(covj33*(i2+(-1)*j2)+covj23*(i3+j3))))+aI**2*aJ*((-1)* \
  covi13*covj23**2*i1+covi13*covj22*covj33*i1+covi13*covj13* \
  covj23*i2+(-1)*covi13*covj12*covj33*i2+covi22*covj13**2*i3+ \
  covi13*covj13*covj22*i3+(-1)*covi13*covj12*covj23*i3+(-2)* \
  covi12*covj13*covj23*i3+covi11*covj23**2*i3+(-1)*covi22*covj11* \
  covj33*i3+2*covi12*covj12*covj33*i3+(-1)*covi11*covj22*covj33* \
  i3+covi13*covj23**2*j1+(-1)*covi13*covj22*covj33*j1+(-1)*covi13* \
  covj13*covj23*j2+covi13*covj12*covj33*j2+covi13*covj13*covj22* \
  j3+(-1)*covi13*covj12*covj23*j3+covi33*(covj12*covj23*(i1+(-1)* \
  j1)+covj13*(covj22*((-1)*i1+j1)+covj12*(i2+(-1)*j2))+covj12**2*j3+ \
  (-1)*covj11*(covj23*i2+(-1)*covj23*j2+covj22*j3))+covi23*( \
  covj12*covj33*((-1)*i1+j1)+covj13**2*((-1)*i2+j2)+covj13*(covj23* \
  (i1+(-1)*j1)+(-1)*covj12*(i3+j3))+covj11*(covj33*(i2+(-1)*j2)+ \
  covj23*(i3+j3)))))**2+(-2)*aI**2*covj23*(covj13*(aI**2*( \
  covj13**2*covj22+(-2)*covj12*covj13*covj23+covj12**2*covj33+ \
  covj11*(covj23**2+(-1)*covj22*covj33))*(i1+(-1)*j1)+aI*aJ*( \
  covi22*covj13**2*i1+covi13*covj13*covj22*i1+(-1)*covi13*covj12* \
  covj23*i1+(-1)*covi12*covj13*covj23*i1+(-1)*covi22*covj11* \
  covj33*i1+covi12*covj12*covj33*i1+covi13*covj12*covj13*i2+(-1)* \
  covi12*covj13**2*i2+(-1)*covi13*covj11*covj23*i2+covi11*covj13* \
  covj23*i2+covi12*covj11*covj33*i2+(-1)*covi11*covj12*covj33*i2+( \
  -1)*covi13*covj12**2*i3+covi12*covj12*covj13*i3+covi13*covj11* \
  covj22*i3+(-1)*covi11*covj13*covj22*i3+(-1)*covi12*covj11* \
  covj23*i3+covi11*covj12*covj23*i3+covi33*(covj12**2+(-1)*covj11* \
  covj22)*(i1+(-1)*j1)+(-2)*covi23*(covj12*covj13+(-1)*covj11* \
  covj23)*(i1+(-1)*j1)+(-1)*covi22*covj13**2*j1+(-1)*covi13* \
  covj13*covj22*j1+covi13*covj12*covj23*j1+covi12*covj13*covj23* \
  j1+covi22*covj11*covj33*j1+(-1)*covi12*covj12*covj33*j1+(-1)* \
  covi13*covj12*covj13*j2+covi12*covj13**2*j2+covi13*covj11* \
  covj23*j2+(-1)*covi11*covj13*covj23*j2+(-1)*covi12*covj11* \
  covj33*j2+covi11*covj12*covj33*j2+(covi13*covj12**2+(-1)*covi12* \
  covj12*covj13+(-1)*covi13*covj11*covj22+covi11*covj13*covj22+ \
  covi12*covj11*covj23+(-1)*covi11*covj12*covj23)*j3)+aJ**2*( \
  covi12*covi33*covj12*i1+covi12*covi33*covj11*i2+covi13**2* \
  covj12*i2+(-1)*covi11*covi33*covj12*i2+(-1)*covi12*covi13* \
  covj13*i2+(-1)*covi12*covi13*covj12*i3+covi12**2*covj13*i3+ \
  covi23**2*covj11*(i1+(-1)*j1)+(-1)*covi12*covi33*covj12*j1+(-1)* \
  covi12*covi33*covj11*j2+(-1)*covi13**2*covj12*j2+covi11*covi33* \
  covj12*j2+covi12*covi13*covj13*j2+covi12*covi13*covj12*j3+(-1)* \
  covi12**2*covj13*j3+covi22*(covi33*covj11*((-1)*i1+j1)+covi11* \
  covj13*((-1)*i3+j3)+covi13*(covj13*i1+covj11*i3+(-1)*covj13*j1+( \
  -1)*covj11*j3))+covi23*(covi13*(covj12*((-1)*i1+j1)+covj11*((-1) \
  *i2+j2))+covi12*((-1)*covj13*i1+(-1)*covj11*i3+covj13*j1+covj11* \
  j3)+covi11*(covj13*i2+covj12*i3+(-1)*covj13*j2+(-1)*covj12*j3)))) \
  *(aI**2*(covj13**2*covj22+(-2)*covj12*covj13*covj23+covj12**2* \
  covj33+covj11*(covj23**2+(-1)*covj22*covj33))*(i2+(-1)*j2)+aJ**2*( \
  covi12*covi33*covj22*i1+covi12*covi33*covj12*i2+covi13**2* \
  covj22*i2+(-1)*covi11*covi33*covj22*i2+(-1)*covi12*covi13* \
  covj23*i2+(-1)*covi12*covi13*covj22*i3+covi12**2*covj23*i3+ \
  covi23**2*covj12*(i1+(-1)*j1)+(-1)*covi12*covi33*covj22*j1+(-1)* \
  covi12*covi33*covj12*j2+(-1)*covi13**2*covj22*j2+covi11*covi33* \
  covj22*j2+covi12*covi13*covj23*j2+covi12*covi13*covj22*j3+(-1)* \
  covi12**2*covj23*j3+covi22*(covi33*covj12*((-1)*i1+j1)+covi11* \
  covj23*((-1)*i3+j3)+covi13*(covj23*i1+covj12*i3+(-1)*covj23*j1+( \
  -1)*covj12*j3))+covi23*(covi13*(covj22*((-1)*i1+j1)+covj12*((-1) \
  *i2+j2))+covi12*((-1)*covj23*i1+(-1)*covj12*i3+covj23*j1+covj12* \
  j3)+covi11*(covj23*i2+covj22*i3+(-1)*covj23*j2+(-1)*covj22*j3)))+ \
  aI*aJ*((-1)*covi12*covj23**2*i1+covi12*covj22*covj33*i1+covi33* \
  covj12**2*i2+(-1)*covi33*covj11*covj22*i2+2*covi13*covj13* \
  covj22*i2+(-2)*covi13*covj12*covj23*i2+(-1)*covi12*covj13* \
  covj23*i2+covi11*covj23**2*i2+covi12*covj12*covj33*i2+(-1)* \
  covi11*covj22*covj33*i2+(-1)*covi12*covj13*covj22*i3+covi12* \
  covj12*covj23*i3+covi12*covj23**2*j1+(-1)*covi12*covj22*covj33* \
  j1+(-1)*covi33*covj12**2*j2+covi33*covj11*covj22*j2+(-2)*covi13* \
  covj13*covj22*j2+2*covi13*covj12*covj23*j2+covi12*covj13* \
  covj23*j2+(-1)*covi11*covj23**2*j2+(-1)*covi12*covj12*covj33*j2+ \
  covi11*covj22*covj33*j2+covi12*covj13*covj22*j3+(-1)*covi12* \
  covj12*covj23*j3+covi22*(covj12*covj33*((-1)*i1+j1)+covj13*( \
  covj23*(i1+(-1)*j1)+covj12*(i3+(-1)*j3))+covj11*covj23*((-1)*i3+ \
  j3))+covi23*(covj12*covj23*(i1+(-1)*j1)+covj13*(covj22*((-1)*i1+ \
  j1)+covj12*((-1)*i2+j2))+covj12**2*((-1)*i3+j3)+covj11*(covj23*i2+ \
  covj22*i3+(-1)*covj23*j2+(-1)*covj22*j3))))+(aI**2*(covj13**2* \
  covj22+(-2)*covj12*covj13*covj23+covj12**2*covj33+covj11*( \
  covj23**2+(-1)*covj22*covj33))*(i3+(-1)*j3)+aJ**2*(covi12*covi33* \
  covj23*i1+covi12*covi33*covj13*i2+covi13**2*covj23*i2+(-1)* \
  covi11*covi33*covj23*i2+(-1)*covi12*covi13*covj33*i2+(-1)* \
  covi12*covi13*covj23*i3+covi12**2*covj33*i3+covi23**2*covj13*(i1+ \
  (-1)*j1)+(-1)*covi12*covi33*covj23*j1+(-1)*covi12*covi33* \
  covj13*j2+(-1)*covi13**2*covj23*j2+covi11*covi33*covj23*j2+ \
  covi12*covi13*covj33*j2+covi12*covi13*covj23*j3+(-1)*covi12**2* \
  covj33*j3+covi22*(covi33*covj13*((-1)*i1+j1)+covi11*covj33*((-1) \
  *i3+j3)+covi13*(covj33*i1+covj13*i3+(-1)*covj33*j1+(-1)*covj13* \
  j3))+covi23*(covi13*(covj23*((-1)*i1+j1)+covj13*((-1)*i2+j2))+ \
  covi12*((-1)*covj33*i1+(-1)*covj13*i3+covj33*j1+covj13*j3)+ \
  covi11*(covj33*i2+covj23*i3+(-1)*covj33*j2+(-1)*covj23*j3)))+aI* \
  aJ*((-1)*covi13*covj23**2*i1+covi13*covj22*covj33*i1+covi13* \
  covj13*covj23*i2+(-1)*covi13*covj12*covj33*i2+covi22*covj13**2* \
  i3+covi13*covj13*covj22*i3+(-1)*covi13*covj12*covj23*i3+(-2)* \
  covi12*covj13*covj23*i3+covi11*covj23**2*i3+(-1)*covi22*covj11* \
  covj33*i3+2*covi12*covj12*covj33*i3+(-1)*covi11*covj22*covj33* \
  i3+covi13*covj23**2*j1+(-1)*covi13*covj22*covj33*j1+(-1)*covi13* \
  covj13*covj23*j2+covi13*covj12*covj33*j2+covi33*(covj13*(covj22* \
  ((-1)*i1+j1)+covj12*(i2+(-1)*j2))+covj23*(covj12*(i1+(-1)*j1)+ \
  covj11*((-1)*i2+j2)))+((-1)*covi22*covj13**2+(-1)*covi13*covj13* \
  covj22+covi13*covj12*covj23+2*covi12*covj13*covj23+(-1)*covi11* \
  covj23**2+covi22*covj11*covj33+(-2)*covi12*covj12*covj33+covi11* \
  covj22*covj33)*j3+covi23*(covj12*covj33*((-1)*i1+j1)+covj13**2*(( \
  -1)*i2+j2)+covj11*(covj33*i2+covj23*i3+(-1)*covj33*j2+(-1)* \
  covj23*j3)+covj13*(covj23*(i1+(-1)*j1)+covj12*((-1)*i3+j3)))))*( \
  aI**2*(covj13**2*covj22+(-2)*covj12*covj13*covj23+covj12**2* \
  covj33+covj11*(covj23**2+(-1)*covj22*covj33))*(covj12*(i1+(-1)*j1) \
  +covj11*((-1)*i2+j2))+aI*aJ*(covi22*covj12*covj13**2*i1+covi13* \
  covj12*covj13*covj22*i1+(-1)*covi13*covj12**2*covj23*i1+(-1)* \
  covi22*covj11*covj13*covj23*i1+(-1)*covi12*covj12*covj13* \
  covj23*i1+covi12*covj11*covj23**2*i1+covi12*covj12**2*covj33*i1+( \
  -1)*covi12*covj11*covj22*covj33*i1+covi13*covj12**2*covj13*i2+( \
  -1)*covi12*covj12*covj13**2*i2+(-2)*covi13*covj11*covj13* \
  covj22*i2+covi13*covj11*covj12*covj23*i2+covi12*covj11*covj13* \
  covj23*i2+covi11*covj12*covj13*covj23*i2+(-1)*covi11*covj11* \
  covj23**2*i2+(-1)*covi11*covj12**2*covj33*i2+covi11*covj11* \
  covj22*covj33*i2+(-1)*covi13*covj12**3*i3+(-1)*covi22*covj11* \
  covj12*covj13*i3+covi12*covj12**2*covj13*i3+covi13*covj11* \
  covj12*covj22*i3+covi12*covj11*covj13*covj22*i3+(-1)*covi11* \
  covj12*covj13*covj22*i3+covi22*covj11**2*covj23*i3+(-2)*covi12* \
  covj11*covj12*covj23*i3+covi11*covj12**2*covj23*i3+(-1)*covi22* \
  covj12*covj13**2*j1+(-1)*covi13*covj12*covj13*covj22*j1+covi13* \
  covj12**2*covj23*j1+covi22*covj11*covj13*covj23*j1+covi12* \
  covj12*covj13*covj23*j1+(-1)*covi12*covj11*covj23**2*j1+(-1)* \
  covi12*covj12**2*covj33*j1+covi12*covj11*covj22*covj33*j1+(-1)* \
  covi13*covj12**2*covj13*j2+covi12*covj12*covj13**2*j2+2*covi13* \
  covj11*covj13*covj22*j2+(-1)*covi13*covj11*covj12*covj23*j2+(-1) \
  *covi12*covj11*covj13*covj23*j2+(-1)*covi11*covj12*covj13* \
  covj23*j2+covi11*covj11*covj23**2*j2+covi11*covj12**2*covj33*j2+( \
  -1)*covi11*covj11*covj22*covj33*j2+covi33*(covj12**2+(-1)* \
  covj11*covj22)*(covj12*(i1+(-1)*j1)+covj11*((-1)*i2+j2))+(covi13* \
  (covj12**3+(-1)*covj11*covj12*covj22)+covj13*(covi11*covj12* \
  covj22+(-1)*covi12*(covj12**2+covj11*covj22))+covj12*(2*covi12* \
  covj11+(-1)*covi11*covj12)*covj23+covi22*covj11*(covj12*covj13+( \
  -1)*covj11*covj23))*j3+covi23*(covj11*covj12*(covj23*(i1+(-1)* \
  j1)+covj13*(i2+(-1)*j2))+covj12**2*(2*covj13*((-1)*i1+j1)+covj11* \
  (i3+(-1)*j3))+covj11*(covj13*covj22*(i1+(-1)*j1)+covj11*((-1)* \
  covj23*i2+(-1)*covj22*i3+covj23*j2+covj22*j3))))+aJ**2*(covi12*( \
  covi33*covj12**2+(-1)*covi23*covj12*covj13+(-1)*covi33*covj11* \
  covj22+covi23*covj11*covj23)*(i1+(-1)*j1)+covi13**2*(covj12**2+(-1) \
  *covj11*covj22)*(i2+(-1)*j2)+covi12**2*(covj12*covj13+(-1)* \
  covj11*covj23)*(i3+(-1)*j3)+covi13*((-1)*covi23*(covj12**2+(-1)* \
  covj11*covj22)*(i1+(-1)*j1)+covi22*(covj12*covj13+(-1)*covj11* \
  covj23)*(i1+(-1)*j1)+covi12*(covj12*covj13*((-1)*i2+j2)+ \
  covj12**2*((-1)*i3+j3)+covj11*(covj23*i2+covj22*i3+(-1)*covj23* \
  j2+(-1)*covj22*j3)))+covi11*((-1)*covi33*(covj12**2+(-1)*covj11* \
  covj22)*(i2+(-1)*j2)+(-1)*covi22*(covj12*covj13+(-1)*covj11* \
  covj23)*(i3+(-1)*j3)+covi23*(covj12*covj13*(i2+(-1)*j2)+ \
  covj12**2*(i3+(-1)*j3)+covj11*((-1)*covj23*i2+(-1)*covj22*i3+ \
  covj23*j2+covj22*j3))))))+(-1)*aI**2*covj22*(covj33*((-1)*aI**2* \
  (covj13**2*covj22+(-2)*covj12*covj13*covj23+covj12**2*covj33+ \
  covj11*(covj23**2+(-1)*covj22*covj33))*(i1+(-1)*j1)+aI*aJ*((-1)* \
  covi22*covj13**2*i1+(-1)*covi13*covj13*covj22*i1+covi13*covj12* \
  covj23*i1+covi12*covj13*covj23*i1+covi22*covj11*covj33*i1+(-1)* \
  covi12*covj12*covj33*i1+(-1)*covi13*covj12*covj13*i2+covi12* \
  covj13**2*i2+covi13*covj11*covj23*i2+(-1)*covi11*covj13*covj23* \
  i2+(-1)*covi12*covj11*covj33*i2+covi11*covj12*covj33*i2+covi13* \
  covj12**2*i3+(-1)*covi12*covj12*covj13*i3+(-1)*covi13*covj11* \
  covj22*i3+covi11*covj13*covj22*i3+covi12*covj11*covj23*i3+(-1)* \
  covi11*covj12*covj23*i3+(-1)*covi33*(covj12**2+(-1)*covj11* \
  covj22)*(i1+(-1)*j1)+2*covi23*(covj12*covj13+(-1)*covj11*covj23) \
  *(i1+(-1)*j1)+covi22*covj13**2*j1+covi13*covj13*covj22*j1+(-1)* \
  covi13*covj12*covj23*j1+(-1)*covi12*covj13*covj23*j1+(-1)* \
  covi22*covj11*covj33*j1+covi12*covj12*covj33*j1+covi13*covj12* \
  covj13*j2+(-1)*covi12*covj13**2*j2+(-1)*covi13*covj11*covj23*j2+ \
  covi11*covj13*covj23*j2+covi12*covj11*covj33*j2+(-1)*covi11* \
  covj12*covj33*j2+((-1)*covi13*covj12**2+covi12*covj12*covj13+ \
  covi13*covj11*covj22+(-1)*covi11*covj13*covj22+(-1)*covi12* \
  covj11*covj23+covi11*covj12*covj23)*j3)+aJ**2*((-1)*covi12* \
  covi33*covj12*i1+(-1)*covi12*covi33*covj11*i2+(-1)*covi13**2* \
  covj12*i2+covi11*covi33*covj12*i2+covi12*covi13*covj13*i2+ \
  covi12*covi13*covj12*i3+(-1)*covi12**2*covj13*i3+covi12*covi33* \
  covj12*j1+covi23**2*covj11*((-1)*i1+j1)+covi12*covi33*covj11*j2+ \
  covi13**2*covj12*j2+(-1)*covi11*covi33*covj12*j2+(-1)*covi12* \
  covi13*covj13*j2+(-1)*covi12*covi13*covj12*j3+covi12**2*covj13* \
  j3+covi22*(covi33*covj11*(i1+(-1)*j1)+covi11*covj13*(i3+(-1)*j3)+ \
  covi13*((-1)*covj13*i1+(-1)*covj11*i3+covj13*j1+covj11*j3))+ \
  covi23*(covi13*(covj12*(i1+(-1)*j1)+covj11*(i2+(-1)*j2))+covi12*( \
  covj13*i1+covj11*i3+(-1)*covj13*j1+(-1)*covj11*j3)+covi11*((-1)* \
  covj13*i2+(-1)*covj12*i3+covj13*j2+covj12*j3))))**2+(-1)*(aI**2*( \
  covj13**2*covj22+(-2)*covj12*covj13*covj23+covj12**2*covj33+ \
  covj11*(covj23**2+(-1)*covj22*covj33))*(i3+(-1)*j3)+aJ**2*( \
  covi12*covi33*covj23*i1+covi12*covi33*covj13*i2+covi13**2* \
  covj23*i2+(-1)*covi11*covi33*covj23*i2+(-1)*covi12*covi13* \
  covj33*i2+(-1)*covi12*covi13*covj23*i3+covi12**2*covj33*i3+ \
  covi23**2*covj13*(i1+(-1)*j1)+(-1)*covi12*covi33*covj23*j1+(-1)* \
  covi12*covi33*covj13*j2+(-1)*covi13**2*covj23*j2+covi11*covi33* \
  covj23*j2+covi12*covi13*covj33*j2+covi12*covi13*covj23*j3+(-1)* \
  covi12**2*covj33*j3+covi22*(covi33*covj13*((-1)*i1+j1)+covi11* \
  covj33*((-1)*i3+j3)+covi13*(covj33*i1+covj13*i3+(-1)*covj33*j1+( \
  -1)*covj13*j3))+covi23*(covi13*(covj23*((-1)*i1+j1)+covj13*((-1) \
  *i2+j2))+covi12*((-1)*covj33*i1+(-1)*covj13*i3+covj33*j1+covj13* \
  j3)+covi11*(covj33*i2+covj23*i3+(-1)*covj33*j2+(-1)*covj23*j3)))+ \
  aI*aJ*((-1)*covi13*covj23**2*i1+covi13*covj22*covj33*i1+covi13* \
  covj13*covj23*i2+(-1)*covi13*covj12*covj33*i2+covi22*covj13**2* \
  i3+covi13*covj13*covj22*i3+(-1)*covi13*covj12*covj23*i3+(-2)* \
  covi12*covj13*covj23*i3+covi11*covj23**2*i3+(-1)*covi22*covj11* \
  covj33*i3+2*covi12*covj12*covj33*i3+(-1)*covi11*covj22*covj33* \
  i3+covi13*covj23**2*j1+(-1)*covi13*covj22*covj33*j1+(-1)*covi13* \
  covj13*covj23*j2+covi13*covj12*covj33*j2+covi33*(covj13*(covj22* \
  ((-1)*i1+j1)+covj12*(i2+(-1)*j2))+covj23*(covj12*(i1+(-1)*j1)+ \
  covj11*((-1)*i2+j2)))+((-1)*covi22*covj13**2+(-1)*covi13*covj13* \
  covj22+covi13*covj12*covj23+2*covi12*covj13*covj23+(-1)*covi11* \
  covj23**2+covi22*covj11*covj33+(-2)*covi12*covj12*covj33+covi11* \
  covj22*covj33)*j3+covi23*(covj12*covj33*((-1)*i1+j1)+covj13**2*(( \
  -1)*i2+j2)+covj11*(covj33*i2+covj23*i3+(-1)*covj33*j2+(-1)* \
  covj23*j3)+covj13*(covj23*(i1+(-1)*j1)+covj12*((-1)*i3+j3)))))*( \
  aI**2*(covj13**2*covj22+(-2)*covj12*covj13*covj23+covj12**2* \
  covj33+covj11*(covj23**2+(-1)*covj22*covj33))*(2*covj13*(i1+(-1)* \
  j1)+covj11*((-1)*i3+j3))+aI*aJ*(2*covi22*covj13**3*i1+2*covi13* \
  covj13**2*covj22*i1+(-2)*covi13*covj12*covj13*covj23*i1+(-2)* \
  covi12*covj13**2*covj23*i1+covi13*covj11*covj23**2*i1+(-2)* \
  covi22*covj11*covj13*covj33*i1+2*covi12*covj12*covj13*covj33* \
  i1+(-1)*covi13*covj11*covj22*covj33*i1+2*covi13*covj12* \
  covj13**2*i2+(-2)*covi12*covj13**3*i2+(-3)*covi13*covj11*covj13* \
  covj23*i2+2*covi11*covj13**2*covj23*i2+covi13*covj11*covj12* \
  covj33*i2+2*covi12*covj11*covj13*covj33*i2+(-2)*covi11*covj12* \
  covj13*covj33*i2+(-2)*covi13*covj12**2*covj13*i3+(-1)*covi22* \
  covj11*covj13**2*i3+2*covi12*covj12*covj13**2*i3+covi13*covj11* \
  covj13*covj22*i3+(-2)*covi11*covj13**2*covj22*i3+covi13*covj11* \
  covj12*covj23*i3+2*covi11*covj12*covj13*covj23*i3+(-1)*covi11* \
  covj11*covj23**2*i3+covi22*covj11**2*covj33*i3+(-2)*covi12* \
  covj11*covj12*covj33*i3+covi11*covj11*covj22*covj33*i3+(-2)* \
  covi22*covj13**3*j1+(-2)*covi13*covj13**2*covj22*j1+2*covi13* \
  covj12*covj13*covj23*j1+2*covi12*covj13**2*covj23*j1+(-1)* \
  covi13*covj11*covj23**2*j1+2*covi22*covj11*covj13*covj33*j1+(-2) \
  *covi12*covj12*covj13*covj33*j1+covi13*covj11*covj22*covj33*j1+ \
  (-2)*covi13*covj12*covj13**2*j2+2*covi12*covj13**3*j2+3*covi13* \
  covj11*covj13*covj23*j2+(-2)*covi11*covj13**2*covj23*j2+(-1)* \
  covi13*covj11*covj12*covj33*j2+(-2)*covi12*covj11*covj13* \
  covj33*j2+2*covi11*covj12*covj13*covj33*j2+covi33*(2*covj12**2* \
  covj13*(i1+(-1)*j1)+covj11*(covj13*covj22*((-1)*i1+j1)+covj11* \
  covj23*(i2+(-1)*j2))+covj11*covj12*((-1)*covj23*i1+(-1)*covj13* \
  i2+covj23*j1+covj13*j2))+(-1)*((-1)*covi22*covj11*covj13**2+2* \
  covi12*covj12*covj13**2+(-2)*covi11*covj13**2*covj22+2*covi11* \
  covj12*covj13*covj23+(-1)*covi11*covj11*covj23**2+covi13*((-2)* \
  covj12**2*covj13+covj11*covj13*covj22+covj11*covj12*covj23)+ \
  covj11*(covi22*covj11+(-2)*covi12*covj12+covi11*covj22)*covj33)* \
  j3+covi23*(covj12*(covj11*covj33*(i1+(-1)*j1)+4*covj13**2*((-1)* \
  i1+j1)+covj11*covj13*(i3+(-1)*j3))+covj11*(3*covj13*covj23*(i1+( \
  -1)*j1)+covj13**2*(i2+(-1)*j2)+covj11*((-1)*covj33*i2+(-1)* \
  covj23*i3+covj33*j2+covj23*j3))))+aJ**2*(2*covi12*covi33*covj12* \
  covj13*i1+(-1)*covi12*covi33*covj11*covj23*i1+covi12*covi33* \
  covj11*covj13*i2+2*covi13**2*covj12*covj13*i2+(-2)*covi11* \
  covi33*covj12*covj13*i2+(-2)*covi12*covi13*covj13**2*i2+(-1)* \
  covi13**2*covj11*covj23*i2+covi11*covi33*covj11*covj23*i2+ \
  covi12*covi13*covj11*covj33*i2+(-2)*covi12*covi13*covj12* \
  covj13*i3+2*covi12**2*covj13**2*i3+covi12*covi13*covj11*covj23* \
  i3+(-1)*covi12**2*covj11*covj33*i3+covi23**2*covj11*covj13*(i1+( \
  -1)*j1)+(-2)*covi12*covi33*covj12*covj13*j1+covi12*covi33* \
  covj11*covj23*j1+(-1)*covi12*covi33*covj11*covj13*j2+(-2)* \
  covi13**2*covj12*covj13*j2+2*covi11*covi33*covj12*covj13*j2+2* \
  covi12*covi13*covj13**2*j2+covi13**2*covj11*covj23*j2+(-1)* \
  covi11*covi33*covj11*covj23*j2+(-1)*covi12*covi13*covj11* \
  covj33*j2+covi22*(covi33*covj11*covj13*((-1)*i1+j1)+covi13*(2* \
  covj13**2*(i1+(-1)*j1)+covj11*covj33*((-1)*i1+j1)+covj11*covj13*( \
  i3+(-1)*j3))+(-1)*covi11*(2*covj13**2+(-1)*covj11*covj33)*(i3+( \
  -1)*j3))+covi12*(2*covi13*covj12*covj13+(-2)*covi12*covj13**2+( \
  -1)*covi13*covj11*covj23+covi12*covj11*covj33)*j3+covi23*( \
  covi13*(2*covj12*covj13*((-1)*i1+j1)+covj11*(covj23*i1+(-1)* \
  covj13*i2+(-1)*covj23*j1+covj13*j2))+covi12*(covj11*covj33*(i1+( \
  -1)*j1)+2*covj13**2*((-1)*i1+j1)+covj11*covj13*((-1)*i3+j3))+ \
  covi11*(2*covj13**2*(i2+(-1)*j2)+2*covj12*covj13*(i3+(-1)*j3)+ \
  covj11*((-1)*covj33*i2+(-1)*covj23*i3+covj33*j2+covj23*j3))))))) 
    elif (0 < cond110_1 and 0<cond110_2 and 0<=cond110_3) or (0< cond111_1 and 0< cond111_2 and 0==cond111_3):
        curr_rate=(1/2)*aI*aJ*(aJ**2*((-1)*covi12**2+covi11*covi22)+aI*aJ*( \
  covi22*covj11+(-2)*covi12*covj12+covi11*covj22)+aI**2*((-1)* \
  covj12**2+covj11*covj22))**(-1)*((aJ*covi22+aI*covj22)*(i1-j1)**2+\
  2*(aJ*covi12+aI*covj12)*(j1-i1)*(i2-j2)+(aJ*covi11+aI*covj11)*(i2-j2)**2)
      
        GradI=(1/2)*aJ**2*(aJ**2*(covi12**2+(-1)*covi11*covi22)+(-1)*aI*aJ*( \
  covi22*covj11+(-2)*covi12*covj12+covi11*covj22)+aI**2*(covj12**2+( \
  -1)*covj11*covj22))**(-2)*(aJ**2*(covi12**2+(-1)*covi11*covi22)*( \
  (-1)*covi22*(i1-j1)**2+(i2-j2)*(2*covi12*i1+(-1)* \
  covi11*i2+(-2)*covi12*j1+covi11*j2))+(-2)*aI*aJ*(covi12**2+(-1)* \
  covi11*covi22)*(covj22*(i1-j1)**2+(i2-j2)*((-2)* \
  covj12*i1+covj11*i2+2*covj12*j1+(-1)*covj11*j2))+aI**2*(covi22*( \
  covj12*(i1-j1)+covj11*(j2-i2))**2+(covj22*(i1-j1)+ \
  covj12*(j2-i2))*(2*covi12*((-1)*covj12*i1+covj11*i2+ \
  covj12*j1+(-1)*covj11*j2)+covi11*(covj22*i1+(-1)*covj12*i2+(-1)* \
  covj22*j1+covj12*j2))))
  
        GradJ=(1/2)*aI**2*(aJ**2*(covi12**2+(-1)*covi11*covi22)+(-1)*aI*aJ*( \
  covi22*covj11+(-2)*covi12*covj12+covi11*covj22)+aI**2*(covj12**2+( \
  -1)*covj11*covj22))**(-2)*((-2)*aI*aJ*(covj12**2+(-1)*covj11* \
  covj22)*(covi22*(i1+(-1)*j1)**2+(2*covi12*((-1)*i1+j1)+covi11*( \
  i2+(-1)*j2))*(i2+(-1)*j2))+aI**2*(covj12**2+(-1)*covj11*covj22)*( \
  (-1)*covj22*(i1+(-1)*j1)**2+(i2+(-1)*j2)*(2*covj12*(i1+(-1)*j1)+ \
  covj11*((-1)*i2+j2)))+aJ**2*(covi22**2*covj11*(i1+(-1)*j1)**2+(-2) \
  *covi11*covi12*(covj22*(i1+(-1)*j1)+covj12*(i2+(-1)*j2))*(i2+( \
  -1)*j2)+covi11**2*covj22*(i2+(-1)*j2)**2+covi12**2*(covj22*(i1+( \
  -1)*j1)**2+(i2+(-1)*j2)*(2*covj12*i1+covj11*i2+(-2)*covj12*j1+( \
  -1)*covj11*j2))+(-2)*covi22*(i1+(-1)*j1)*(covi11*covj12*((-1)* \
  i2+j2)+covi12*(covj12*i1+covj11*i2+(-1)*covj12*j1+(-1)*covj11*j2) \
  )))
    elif (0 < cond101_1 and 0<= cond101_2 and 0< cond101_3) or (0<cond111_1 and 0==cond111_2 and 0<cond111_3):
        curr_rate=(1/2)*aI*aJ*(aJ**2*((-1)*covi13**2+covi11*covi33)+aI*aJ*( \
  covi33*covj11+(-2)*covi13*covj13+covi11*covj33)+aI**2*((-1)* \
  covj13**2+covj11*covj33))**(-1)*((aJ*covi33+aI*covj33)*(i1+(-1)* \
  j1)**2+2*(aJ*covi13+aI*covj13)*((-1)*i1+j1)*(i3+(-1)*j3)+(aJ* \
  covi11+aI*covj11)*(i3+(-1)*j3)**2)
        GradI=(1/2)*aJ**2*(aJ**2*(covi13**2+(-1)*covi11*covi33)+(-1)*aI*aJ*( \
  covi33*covj11+(-2)*covi13*covj13+covi11*covj33)+aI**2*(covj13**2+( \
  -1)*covj11*covj33))**(-2)*(aJ**2*(covi13**2+(-1)*covi11*covi33)*( \
  (-1)*covi33*(i1+(-1)*j1)**2+(i3+(-1)*j3)*(2*covi13*i1+(-1)* \
  covi11*i3+(-2)*covi13*j1+covi11*j3))+(-2)*aI*aJ*(covi13**2+(-1)* \
  covi11*covi33)*(covj33*(i1+(-1)*j1)**2+(i3+(-1)*j3)*((-2)* \
  covj13*i1+covj11*i3+2*covj13*j1+(-1)*covj11*j3))+aI**2*(covi33*( \
  covj13*(i1+(-1)*j1)+covj11*((-1)*i3+j3))**2+(covj33*(i1+(-1)*j1)+ \
  covj13*((-1)*i3+j3))*(2*covi13*((-1)*covj13*i1+covj11*i3+ \
  covj13*j1+(-1)*covj11*j3)+covi11*(covj33*i1+(-1)*covj13*i3+(-1)* \
  covj33*j1+covj13*j3))))
        GradJ=(1/2)*aI**2*(aJ**2*(covi13**2+(-1)*covi11*covi33)+(-1)*aI*aJ*( \
  covi33*covj11+(-2)*covi13*covj13+covi11*covj33)+aI**2*(covj13**2+( \
  -1)*covj11*covj33))**(-2)*((-2)*aI*aJ*(covj13**2+(-1)*covj11* \
  covj33)*(covi33*(i1+(-1)*j1)**2+(2*covi13*((-1)*i1+j1)+covi11*( \
  i3+(-1)*j3))*(i3+(-1)*j3))+aI**2*(covj13**2+(-1)*covj11*covj33)*( \
  (-1)*covj33*(i1+(-1)*j1)**2+(i3+(-1)*j3)*(2*covj13*(i1+(-1)*j1)+ \
  covj11*((-1)*i3+j3)))+aJ**2*(covi33**2*covj11*(i1+(-1)*j1)**2+(-2) \
  *covi11*covi13*(covj33*(i1+(-1)*j1)+covj13*(i3+(-1)*j3))*(i3+( \
  -1)*j3)+covi11**2*covj33*(i3+(-1)*j3)**2+covi13**2*(covj33*(i1+( \
  -1)*j1)**2+(i3+(-1)*j3)*(2*covj13*i1+covj11*i3+(-2)*covj13*j1+( \
  -1)*covj11*j3))+(-2)*covi33*(i1+(-1)*j1)*(covi11*covj13*((-1)* \
  i3+j3)+covi13*(covj13*i1+covj11*i3+(-1)*covj13*j1+(-1)*covj11*j3) \
  )))

    elif (0 < cond100_1 and 0<=cond100_2 and 0<= cond100_3) or (0<cond111_1 and 0==cond111_2 and 0==cond111_3) or (0< cond110_1 and 0== cond110_2 and 0<= cond110_3) or (0< cond101_1 and 0<=cond101_2 and 0==cond101_3):
        curr_rate=aI*aJ*(2*aJ*covi11+2*aI*covj11)**(-1)*(i1+(-1)*j1)**2
        GradI=(1/2)*aJ**2*covi11*(aJ*covi11+aI*covj11)**(-2)*(i1+(-1)*j1)**2
        GradJ=(1/2)*aI**2*covj11*(aJ*covi11+aI*covj11)**(-2)*(i1+(-1)*j1)**2
    elif (0<= cond011_1 and 0< cond011_2 and 0<cond011_3) or (0 == cond111_1 and 0< cond111_2 and 0<cond111_3):
        curr_rate=(1/2)*aI*aJ*(aJ**2*((-1)*covi23**2+covi22*covi33)+aI*aJ*( \
  covi33*covj22+(-2)*covi23*covj23+covi22*covj33)+aI**2*((-1)* \
  covj23**2+covj22*covj33))**(-1)*((aJ*covi33+aI*covj33)*(i2+(-1)* \
  j2)**2+2*(aJ*covi23+aI*covj23)*((-1)*i2+j2)*(i3+(-1)*j3)+(aJ* \
  covi22+aI*covj22)*(i3+(-1)*j3)**2)
        GradI=(1/2)*aJ**2*(aJ**2*(covi23**2+(-1)*covi22*covi33)+(-1)*aI*aJ*( \
  covi33*covj22+(-2)*covi23*covj23+covi22*covj33)+aI**2*(covj23**2+( \
  -1)*covj22*covj33))**(-2)*(aJ**2*(covi23**2+(-1)*covi22*covi33)*( \
  (-1)*covi33*(i2+(-1)*j2)**2+(i3+(-1)*j3)*(2*covi23*i2+(-1)* \
  covi22*i3+(-2)*covi23*j2+covi22*j3))+(-2)*aI*aJ*(covi23**2+(-1)* \
  covi22*covi33)*(covj33*(i2+(-1)*j2)**2+(i3+(-1)*j3)*((-2)* \
  covj23*i2+covj22*i3+2*covj23*j2+(-1)*covj22*j3))+aI**2*(covi33*( \
  covj23*(i2+(-1)*j2)+covj22*((-1)*i3+j3))**2+(covj33*(i2+(-1)*j2)+ \
  covj23*((-1)*i3+j3))*(2*covi23*((-1)*covj23*i2+covj22*i3+ \
  covj23*j2+(-1)*covj22*j3)+covi22*(covj33*i2+(-1)*covj23*i3+(-1)* \
  covj33*j2+covj23*j3))))
        GradJ=(1/2)*aI**2*(aJ**2*(covi23**2+(-1)*covi22*covi33)+(-1)*aI*aJ*( \
  covi33*covj22+(-2)*covi23*covj23+covi22*covj33)+aI**2*(covj23**2+( \
  -1)*covj22*covj33))**(-2)*((-2)*aI*aJ*(covj23**2+(-1)*covj22* \
  covj33)*(covi33*(i2+(-1)*j2)**2+(2*covi23*((-1)*i2+j2)+covi22*( \
  i3+(-1)*j3))*(i3+(-1)*j3))+aI**2*(covj23**2+(-1)*covj22*covj33)*( \
  (-1)*covj33*(i2+(-1)*j2)**2+(i3+(-1)*j3)*(2*covj23*(i2+(-1)*j2)+ \
  covj22*((-1)*i3+j3)))+aJ**2*(covi33**2*covj22*(i2+(-1)*j2)**2+(-2) \
  *covi22*covi23*(covj33*(i2+(-1)*j2)+covj23*(i3+(-1)*j3))*(i3+( \
  -1)*j3)+covi22**2*covj33*(i3+(-1)*j3)**2+covi23**2*(covj33*(i2+( \
  -1)*j2)**2+(i3+(-1)*j3)*(2*covj23*i2+covj22*i3+(-2)*covj23*j2+( \
  -1)*covj22*j3))+(-2)*covi33*(i2+(-1)*j2)*(covi22*covj23*((-1)* \
  i3+j3)+covi23*(covj23*i2+covj22*i3+(-1)*covj23*j2+(-1)*covj22*j3) \
  )))
    elif (0 <= cond010_1 and 0 < cond010_2 and 0 <= cond010_3 ) or (0 == cond111_1 and 0 < cond111_2 and 0 == cond111_3 ) or (0 == cond110_1 and 0 < cond110_2 and 0 <= cond110_3 ) or (0 <= cond011_1 and 0 < cond011_2 and 0 == cond011_3 ):
        curr_rate=aI*aJ*(2*aJ*covi22+2*aI*covj22)**(-1)*(i2+(-1)*j2)**2
        GradI=(1/2)*aJ**2*covi22*(aJ*covi22+aI*covj22)**(-2)*(i2+(-1)*j2)**2
        GradJ=(1/2)*aI**2*covj22*(aJ*covi22+aI*covj22)**(-2)*(i2+(-1)*j2)**2
    elif (0 <= cond001_1 and 0 <= cond001_2 and 0 < cond001_3 ) or (0 == cond111_1 and 0 == cond111_2 and 0 < cond111_3 ) or (0 == cond101_1 and 0 <= cond101_2 and 0 < cond101_3 ) or (0 <= cond011_1 and 0 == cond011_2 and 0 < cond011_3 ):
        curr_rate=aI*aJ*(2*aJ*covi33+2*aI*covj33)**(-1)*(i3+(-1)*j3)**2
        GradI=(1/2)*aJ**2*covi33*(aJ*covi33+aI*covj33)**(-2)*(i3+(-1)*j3)**2
        GradJ=(1/2)*aI**2*covj33*(aJ*covi33+aI*covj33)**(-2)*(i3+(-1)*j3)**2
    else:
        print('Calling Quardprog (MCE-2)')
        curr_rate, GradI, GradJ = MCE_four_d_plus(aI, aJ, Iobj, inv_var_i, Jobj, inv_var_j, 3)
    return curr_rate, GradI, GradJ

def MCE_four_d_plus(alpha_i, alpha_j, obj_i, inv_var_i, obj_j, inv_var_j, n_obj):
    """calculates MCE constraint values and gradients\n
    The input variable names and the return variable names are not the same as 2d and 3d (Ziyu)

    parameters
    ----------
    aI: float, allocation to system i
    aJ: float, allocation to system j
    Iobj: numpy array, objective vals of system i
    Isig: 2d numpy array, covariance matrix of objectives of system i
    Jobj: numpy array, objective vals of system j
    Jsig: 2d numpy array, covariance martrix of objectives of system j
    inv_var_i: 2d numpy array, inverse of Isig (precomputed for efficiency)
    inv_var_j: 2d numpy array, inverse of Jsig (precomputed for efficiency)

    returns
    -------
    rate: float, decay rate of MCE event between systems i and j
    grad_i: numpy array, gradient of rate wrt alpha_i
    grad_j: numpy array, gradient of rate wrt alpha_j"""
    
    #this comes almost straight from the 
    P = linalg.block_diag(alpha_i*inv_var_i, alpha_j*inv_var_j)
    
    
    
    q = matrix(-1*np.append(alpha_i * inv_var_i @ obj_i, alpha_j * inv_var_j@ obj_j))
    
    G = matrix(np.append(-1*np.identity(n_obj),np.identity(n_obj),axis=1))
    
    
    h = matrix(np.zeros(n_obj))
    
    

    P = matrix((P + P.transpose())/2)
    
    
    

    
    x_star = np.array(solvers.qp(P,q,G,h)['x']).flatten()
    

    

    
    rate = 0.5*alpha_i*np.transpose(x_star[0:n_obj] - obj_i) @ inv_var_i @(x_star[0:n_obj]-obj_i) +\
    0.5*alpha_j*(x_star[n_obj:] - obj_j) @ inv_var_j @(x_star[n_obj:]-obj_j)
    
    grad_i = 0.5*(x_star[0:n_obj] - obj_i) @ inv_var_i @ (x_star[0:n_obj] - obj_i)
    
    grad_j = 0.5*(x_star[n_obj:] - obj_j) @ inv_var_j @ (x_star[n_obj:]-obj_j)
    

    
    return rate, grad_i, grad_j