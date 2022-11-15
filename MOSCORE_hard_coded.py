#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 17:55:52 2019

@author: nathangeldner
"""
import scipy.linalg as linalg
from cvxopt import matrix, solvers
import numpy as np






def SCORE_1d(Gobj,Jobj, CovJ):
    """parameters:
                Gobj: numpy array, objective vals of a pareto system or a phantom
                Jobj: numpy array, objective vals of a pareto system or a non-pareto
                CovJ: 2d numpy array, covariance matrix of J
        output: 
            curr_rate: float (not sure what it represents, the score calcs confused me 3 years ago - Nathan)
            Binds: numpy array (ditto - Nathan)"""
                
                
    g1=Gobj[0]
    j1=Jobj[0]
    covj11=CovJ[0,0]
    
    curr_rate=(1/2)*(g1-j1)**2/covj11
    Binds=np.array([curr_rate])
    
    return curr_rate, Binds

def SCORE_2d(Gobj,Jobj, CovJ):
    g1=Gobj[0]
    g2=Gobj[1]
    j1=Jobj[0]
    j2=Jobj[1]
    covj11=CovJ[0,0]
    covj12=CovJ[0,1]
    covj22=CovJ[1,1]
    
    cond11_1=(covj22*(g1-j1)+covj12*(j2-g2))/(covj12**2-covj11*covj22)
    cond11_2=(covj12*(j1-g1)+covj11*(g2-j2))/(covj12**2-covj11*covj22)
    cond10_1=(j1-g1)/covj11
    cond10_2=g2-j2+covj12*(j1-g1)/covj11
    cond01_1=g1-j1+covj12*(j2-g2)/covj22
    cond01_2=(j2-g2)/covj22
    
    if 0 < cond11_1 and 0 < cond11_2 :
        curr_rate=(covj22*(g1-j1)**2+(2*covj12*(j1-g1)+covj11*(g2-j2))*(g2-j2))/(2*covj11*covj22-2*covj12**2)
        Binds=np.array([curr_rate, curr_rate])
    elif (0 < cond10_1 and 0 <= cond10_2 ) or (0 < cond11_1 and 0 == cond11_2 ):
        curr_rate=(1/2)*covj11**(-1)*(g1-j1)**2
        Binds=np.array([curr_rate, np.inf])
    elif (0 <= cond01_1 and 0 < cond01_2 ) or (0 == cond11_1 and 0 < cond11_2 ):
        curr_rate=(1/2)*covj22**(-1)*(g2-j2)**2
        Binds=np.array([np.inf, curr_rate])
    else:
        print('SCORE Infinite')
        curr_rate, Binds = score_four_d_plus(Gobj, Jobj, CovJ)
    return curr_rate, Binds

def SCORE_3d(Gobj,Jobj, CovJ):
    g1=Gobj[0]
    g2=Gobj[1]
    g3=Gobj[2]
    j1=Jobj[0]
    j2=Jobj[1]
    j3=Jobj[2]
    covj11=CovJ[0,0]
    covj12=CovJ[0,1]
    covj22=CovJ[1,1]
    covj13=CovJ[0,2]
    covj23=CovJ[1,2]
    covj33=CovJ[2,2]
    
    cond111_1=(covj13**2*covj22+(-2)*covj12*covj13*covj23+covj12**2*covj33+ \
  covj11*(covj23**2+(-1)*covj22*covj33))**(-1)*(covj22*covj33*(g1-j1)+\
  covj23**2*(j1-g1)+covj13*covj23*(g2-j2)+covj12* \
  covj33*(j2-g2)+covj12*covj23*(g3-j3)+covj13*covj22*(j3-g3))
    cond111_2=(covj13**2*covj22+(-2)*covj12*covj13*covj23+covj12**2*covj33+ \
  covj11*(covj23**2+(-1)*covj22*covj33))**(-1)*(covj13*covj23*(g1-j1)+\
  covj12*covj33*(j1-g1)+covj11*covj33*(g2-j2)+ \
  covj13**2*(j2-g2)+covj12*covj13*(g3-j3)+covj11*covj23*(j3-g3))
    cond111_3=(covj13**2*covj22+(-2)*covj12*covj13*covj23+covj12**2*covj33+ \
  covj11*(covj23**2+(-1)*covj22*covj33))**(-1)*(covj12*covj23*(g1-j1)+\
  covj13*covj22*(j1-g1)+covj12*covj13*(g2-j2)+ \
  covj11*covj23*(j2-g2)+covj11*covj22*(g3-j3)+covj12**2*(j3-g3))
    cond110_1=(covj12**2+(-1)*covj11*covj22)**(-1)*(covj22*g1+(-1)*covj12*g2+(-1)*covj22*j1+covj12*j2)
    cond110_2=(covj12**2+(-1)*covj11*covj22)**(-1)*(covj12*(j1-g1)+covj11*(g2-j2))
    cond110_3=g3-j3+(covj12**2+(-1)*covj11*covj22)**(-1)*(covj23*(covj12*(j1-g1)+covj11*(g2-j2))+\
    covj13*(covj22*(g1-j1)+covj12*(j2-g2)))
    cond101_1=(covj13**2+(-1)*covj11*covj33)**(-1)*(covj33*g1+(-1)*covj13*g3+(-1)*covj33*j1+covj13*j3)
    cond101_2=g2+(covj13**2+(-1)*covj11*covj33)**(-1)*(covj12*covj33*(g1-j1)+\
    (-1)*covj13**2*j2+covj11*(covj23*g3+covj33*j2+(-1)*covj23*j3) \
  +covj13*(covj23*(j1-g1)+covj12*(j3-g3)))
    cond101_3=(covj13**2+(-1)*covj11*covj33)**(-1)*(covj13*(j1-g1)+covj11*(g3-j3))
    cond100_1=covj11**(-1)*(j1-g1)
    cond100_2=g2-j2+covj11**(-1)*covj12*(j1-g1)
    cond100_3=g3-j3+covj11**(-1)*covj13*(j1-g1)
    cond011_1=g1+(covj23**2+(-1)*covj22*covj33)**(-1)*((-1)*covj23**2*j1+covj22* \
  covj33*j1+covj13*(covj23*(j2-g2)+covj22*(g3-j3))+ \
  covj12*(covj33*g2+(-1)*covj23*g3+(-1)*covj33*j2+covj23*j3))
    cond011_2=(covj23**2+(-1)*covj22*covj33)**(-1)*(covj33*g2+(-1)*covj23*g3+(-1)*covj33*j2+covj23*j3)
    cond011_3=(covj23**2+(-1)*covj22*covj33)**(-1)*(covj23*(j2-g2)+covj22*(g3-j3))
    cond010_1=g1+(-1)*covj22**(-1)*(covj12*g2+covj22*j1+(-1)*covj12*j2)
    cond010_2=covj22**(-1)*(j2-g2)
    cond010_3=g3-j3+covj22**(-1)*covj23*(j2-g2)
    cond001_1=g1+(-1)*covj33**(-1)*(covj13*g3+covj33*j1+(-1)*covj13*j3)
    cond001_2=g2+(-1)*covj33**(-1)*(covj23*g3+covj33*j2+(-1)*covj23*j3)
    cond001_3=covj33**(-1)*(j3-g3)
        
    if (0 < cond111_1 and 0 < cond111_2 and 0 < cond111_3 ):
        curr_rate=(1/2)*(covj13**2*covj22+(-2)*covj12*covj13*covj23+covj12**2* \
      covj33+covj11*(covj23**2-covj22*covj33))**(-1)*((covj23**2-covj22*covj33)*(g1-j1)**2+\
      ((-2)*covj13*covj23+2*covj12*covj33)*(g1-j1)*(g2-j2)+(covj13**2-covj11*covj33)* \
      (g2-j2)**2+2*covj11*covj23*(g2-j2)*(g3-j3)+ \
      covj12**2*(g3-j3)**2+(((-2)*covj13*covj22+2*covj12*covj23)*( \
      g1-j1)+2*covj12*covj13*(g2-j2)+covj11*covj22*(g3-j3))*(j3-g3))
        Binds=np.array([ curr_rate, curr_rate, curr_rate])
        
    elif (0 < cond110_1 and 0 < cond110_2 and 0 <= cond110_3 ) or (0 < cond111_1 and 0 < cond111_2 and 0 == cond111_3 ):
        curr_rate=((-2)*covj12**2+2*covj11*covj22)**(-1)*(covj22*(g1-j1)**2+( \
      2*covj12*(j1-g1)+covj11*(g2-j2))*(g2-j2))
        Binds=np.array([ curr_rate, curr_rate, np.inf])
        
    elif (0 < cond101_1 and 0 <= cond101_2 and 0 < cond101_3 ) or (0 < cond111_1 and 0 == cond111_2 and 0 < cond111_3 ):
        curr_rate=((-2)*covj13**2+2*covj11*covj33)**(-1)*(covj33*(g1-j1)**2+( \
      2*covj13*(j1-g1)+covj11*(g3-j3))*(g3-j3))
        Binds=np.array([ curr_rate, np.inf, curr_rate])
        
    elif (0 < cond100_1 and 0 <= cond100_2 and 0 <= cond100_3 ) or (0 < cond111_1 and 0 == cond111_2 and 0 == cond111_3 ) or (0 < cond110_1 and 0 == cond110_2 and 0 <= cond110_3 ) or (0 < cond101_1 and 0 <= cond101_2 and 0 == cond101_3 ): 
        curr_rate=(1/2)*covj11**(-1)*(g1-j1)**2
        Binds=np.array([ curr_rate, np.inf, np.inf])
        
    elif (0 <= cond011_1 and 0 < cond011_2 and 0 < cond011_3 ) or (0 == cond111_1 and 0 < cond111_2 and 0 < cond111_3 ):
        curr_rate=((-2)*covj23**2+2*covj22*covj33)**(-1)*(covj33*(g2-j2)**2+( \
      2*covj23*(j2-g2)+covj22*(g3-j3))*(g3-j3))
        Binds=np.array([ np.inf, curr_rate, curr_rate])
        
    elif (0 <= cond010_1 and 0 < cond010_2 and 0 <= cond010_3 ) or (0 == cond111_1 and 0 < cond111_2 and 0 == cond111_3 ) or (0 == cond110_1 and 0 < cond110_2 and 0 <= cond110_3 ) or (0 <= cond011_1 and 0 < cond011_2 and 0 == cond011_3 ):
        curr_rate=(1/2)*covj22**(-1)*(g2-j2)**2
        Binds=np.array([ np.inf, curr_rate, np.inf])
        
    elif (0 <= cond001_1 and 0 <= cond001_2 and 0 < cond001_3 ) or (0 == cond111_1 and 0 == cond111_2 and 0 < cond111_3 ) or (0 == cond101_1 and 0 <= cond101_2 and 0 < cond101_3 ) or (0 <= cond011_1 and 0 == cond011_2 and 0 < cond011_3 ):
        curr_rate=(1/2)*covj33**(-1)*(g3-j3)**2
        Binds=np.array([ np.inf, np.inf, curr_rate])
        
    else:
        print('SCORE Infinite')
        curr_rate, Binds = score_four_d_plus(Gobj, Jobj, CovJ)
        
    return curr_rate, Binds
    
        
        
    
def score_four_d_plus(phantom_objectives,obj_j, cov_j):
    
    opts = {'abstol':10**-12, 'reltol': 10**-12, 'show_progress': False,# 'feastol': 10**-12\
            }

    
    n_obj = len(phantom_objectives)
    
    inv_cov_j = np.linalg.inv(cov_j)
    
    P = matrix(inv_cov_j)
    
    q = matrix(-1*inv_cov_j@obj_j)
    
    G = matrix(np.identity(n_obj))
    
    h = matrix(phantom_objectives)
    
    solvers.options['show_progress'] = False
    res = solvers.qp(P,q,G,h, options=opts)
    
    x_star = np.array(res['x']).flatten()
    
    #apologies for the stylistic inconsistency here, lambda without a capital
    #is for function handles
    Lambda = np.array(res['z']).flatten()

    score = 0.5*np.transpose(obj_j-x_star) @ inv_cov_j @ (obj_j-x_star)
    
    #tol = 10**-6
    #original matlab code set tol to 10**-6, but the lagrange multipliers are slightly different
    #so I'm going to see what happens if I set it to 10**-5
    tol = 10**-6
    
    #print(Lambda)
    
    Lambda[Lambda>tol] = 1
    
    Lambda[Lambda<=tol] = np.inf
    
    
    #gonna confess here, I'm not certain what this does
    binds = score*Lambda
    
    return score, binds
    
    
        
        
        
        
        
        
        