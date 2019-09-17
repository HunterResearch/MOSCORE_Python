#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  1 19:44:22 2019

@author: nathangeldner
"""

from scipy import io
import numpy as np
from allocate import allocate
from utils import create_allocation_problem

size_ind = 0


data = io.loadmat('FixedProblems3D10.mat')
corrs = data['corrs']
objs = data['objs']
z_vec = []
for prob_ind in range(10):
    if prob_ind not in [5]:
        my_objs = {}
        for i in range(len(objs[prob_ind,size_ind])):
            my_objs[i] = list(objs[prob_ind,size_ind][i])
        corr = corrs[prob_ind,size_ind]
        n_sys = len(my_objs)
        n_obj = len(my_objs[0])
        cov = np.identity(n_obj) + (np.ones([n_obj,n_obj])  - np.identity(n_obj))*corr
        covs = {}
        for i in range(n_sys):
            covs[i] = cov
        problem = create_allocation_problem(my_objs, covs)
        allocation, z = allocate('Brute Force', problem)
        print(z)
        #print(allocation)
        z_vec.append(z)
    