#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 17:42:23 2019

@author: nathangeldner
"""
from iscore_allocation import iscore_allocation
from score_allocation import score_allocation
from phantom_allocation import calc_phantom_allocation
from brute_force_allocation import calc_bf_allocation



def allocate(method, systems, warm_start = None):
    """generate a non-sequential simulation allocation for the MORS problem
    
    Arguments
    ---------
    
    method: str
        chosen allocation method. Options are "iSCORE", "SCORE", "Phantom", and "Brute Force"
        
    systems: dict with keys 'obj', 'var', 'inv_var', 'pareto_indices', 'non_pareto_indices'
            systems['obj'] is a dictionary of numpy arrays, indexed by system number,
                each of which corresponds to the objective values of a system
            systems['var'] is a dictionary of 2d numpy arrays, indexed by system number,
                each of which corresponds to the covariance matrix of a system
            systems['inv_var'] is a dictionary of 2d numpy, indexed by system number,
                each of which corresponds to the inverse covariance matrix of a system
            systems['pareto_indices'] is a list of pareto systems ordered by the first objective
            systems['non_pareto_indices'] is a list of non-pareto systems ordered by the first objective
    
    warm_start: list of float
        an initial simulation allocation which sets the starting point for determining the optimal allocation.
        Length must be equal to the number of systems. 
        
    Returns:
        outs:tuple
            outs[0] is a list of float of length equal to the number of systems. Provies the
                estimated optimal simulation allocation
            outs[1] is the estimated rate of convergence
                """
        
    if warm_start is not None and len(warm_start)!= len(systems['obj']):
        raise ValueError("Length of warm_start must be equal to the number of systems")
        
    if method == "iSCORE":
        return iscore_allocation(systems, warm_start = warm_start)
    elif method == "SCORE":
        return score_allocation_smart(systems, warm_start = warm_start)
    elif method == "Phantom":
        return phantom_allocation_smart(systems, warm_start = warm_start)
    elif method == "Brute Force":
        return bf_allocation_smart(systems, warm_start = warm_start)
    else:
        raise ValueError("Invalid method selected. Valid methods are iSCORE, SCORE, Phantom, and Brute Force")
    
        

def score_allocation_smart(systems, warm_start = None):
    """generate a non-sequential simulation allocation for the MORS problem
    using the SCORE method
    
    Arguments
    ---------
        
    systems: dict with keys 'obj', 'var', 'inv_var', 'pareto_indices', 'non_pareto_indices'
            systems['obj'] is a dictionary of numpy arrays, indexed by system number,
                each of which corresponds to the objective values of a system
            systems['var'] is a dictionary of 2d numpy arrays, indexed by system number,
                each of which corresponds to the covariance matrix of a system
            systems['inv_var'] is a dictionary of 2d numpy, indexed by system number,
                each of which corresponds to the inverse covariance matrix of a system
            systems['pareto_indices'] is a list of pareto systems ordered by the first objective
            systems['non_pareto_indices'] is a list of non-pareto systems ordered by the first objective
    
    warm_start: list of float
        an initial simulation allocation which sets the starting point for determining the optimal allocation.
        Length must be equal to the number of systems. 
        
    Returns:
        outs:tuple
            outs[0] is a list of float of length equal to the number of systems. Provies the
                estimated optimal simulation allocation
            outs[1] is the estimated rate of convergence
                """
    if len(systems['obj'][0]) > 3:
        warm_start = iscore_allocation(systems, warm_start = warm_start)[0]
    return score_allocation(systems, warm_start = warm_start)

def phantom_allocation_smart(systems, warm_start = None):
    """generate a non-sequential simulation allocation for the MORS problem
    using the Phantom method
    
    Arguments
    ---------
        
    systems: dict with keys 'obj', 'var', 'inv_var', 'pareto_indices', 'non_pareto_indices'
            systems['obj'] is a dictionary of numpy arrays, indexed by system number,
                each of which corresponds to the objective values of a system
            systems['var'] is a dictionary of 2d numpy arrays, indexed by system number,
                each of which corresponds to the covariance matrix of a system
            systems['inv_var'] is a dictionary of 2d numpy, indexed by system number,
                each of which corresponds to the inverse covariance matrix of a system
            systems['pareto_indices'] is a list of pareto systems ordered by the first objective
            systems['non_pareto_indices'] is a list of non-pareto systems ordered by the first objective
    
    warm_start: list of float
        an initial simulation allocation which sets the starting point for determining the optimal allocation.
        Length must be equal to the number of systems. 
        
    Returns:
        outs:tuple
            outs[0] is a list of float of length equal to the number of systems. Provies the
                estimated optimal simulation allocation
            outs[1] is the estimated rate of convergence
    """
    if len(systems['obj'][0]) > 3:
        warm_start = iscore_allocation(systems, warm_start = warm_start)[0]
    return calc_phantom_allocation(systems, warm_start = warm_start)

def bf_allocation_smart(systems, warm_start = None):
    """generate a non-sequential simulation allocation for the MORS problem
    using the Brute Force method
    
    Arguments
    ---------
        
    systems: dict with keys 'obj', 'var', 'inv_var', 'pareto_indices', 'non_pareto_indices'
            systems['obj'] is a dictionary of numpy arrays, indexed by system number,
                each of which corresponds to the objective values of a system
            systems['var'] is a dictionary of 2d numpy arrays, indexed by system number,
                each of which corresponds to the covariance matrix of a system
            systems['inv_var'] is a dictionary of 2d numpy, indexed by system number,
                each of which corresponds to the inverse covariance matrix of a system
            systems['pareto_indices'] is a list of pareto systems ordered by the first objective
            systems['non_pareto_indices'] is a list of non-pareto systems ordered by the first objective
    
    warm_start: list of float
        an initial simulation allocation which sets the starting point for determining the optimal allocation.
        Length must be equal to the number of systems. 
        
    Returns:
        outs:tuple
            outs[0] is a list of float of length equal to the number of systems. Provies the
                estimated optimal simulation allocation
            outs[1] is the estimated rate of convergence
    """
    if len(systems['obj'][0]) > 3:
        warm_start = iscore_allocation(systems, warm_start = warm_start)[0]
    return calc_bf_allocation(systems, warm_start = warm_start)
