#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 13 21:19:58 2019

@author: nathangeldner
"""
import numpy as np
import pymoso.chnutils as utils
import itertools as it
import scipy.optimize as opt

def calc_bf_allocation(systems, warm_start = None):
    
    n_obj = len(systems["obj"][0])
    
    n_systems = len(systems["obj"])
    
    systems["pareto_indices"]= list(utils.get_nondom(systems["obj"]))
    
    systems["non_pareto_indices"] = [system for system in range(n_systems) if system not in systems["pareto_indices"]]
    


    
    #TODO compute kappa vectors
    
    #