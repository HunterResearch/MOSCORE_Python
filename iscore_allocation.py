#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 05:30:30 2019

@author: nathangeldner
"""

import numpy as np
import pymoso.chnutils as utils
import itertools as it
import scipy.optimize as opt
import scipy.linalg as linalg
from cvxopt import matrix, solvers
from phantom_allocation import find_phantoms
from brute_force_allocation import hessian_zero


def iscore_allocation(systems,warm_start = None):
    """Calculates ISCORE Allocation given a set of systems and optional warm start
    
    Parameters
    ----------
        Systems : dict
            ``"obj"``
            a dictionary of objective value (float) tuples  keyed by system number
            
            ``"var"``
            a dictionary of objective covariance matrices (numpy matrices) keyed by system number
            
            ``"pareto_indices"``
            a list of integer system numbers of estimated pareto systems ordered by first objective value
            
            ``"non_pareto_indices"``
            a list of integer system numbers of estimated non-parety systems ordered by first objective value
            
        warm_start: numpy array of length equal to the number of system, which sums to 1
        
        
    Returns:
        
        out_tuple:
            out_tuple[0]: the estimated optimal allocation of simulation runs assuming that estimated objectives and variances are true\n
            
            out_tuple[1]: the estimated convergence rate associatedc with the optimal allocation"""
    
    n_obj = len(systems['obj'][0])
    
    n_systems = len(systems["obj"])
        
    num_par = len(systems["pareto_indices"])
    
    pareto_array  = np.zeros([num_par,n_obj])
    
    for i in range(num_par):
        pareto_array[i,:] = systems['obj'][systems['pareto_indices'][i]]
    
    phantom_values = find_phantoms(pareto_array,n_obj,num_par)
    
    
    
   #sort the phantoms. the commented part doesn't give different results, but this
    #makes the constraint ordering identical to that of the matlab code, which you'll want for debugging
    for i in range(n_obj):
        phantom_values = phantom_values[(phantom_values[:,n_obj-1-i]).argsort(kind='mergesort')]
    #phantom_values = phantom_values[(phantom_values[:,0]).argsort()]
    
    
    
    
    
    n_phantoms = len(phantom_values)
    
    #TODO: consider using something other than inf as a placeholder. 
    #unfortunately, inf is a float in numpy, and arrays must be homogenous
    #and floats don't automatically cast to ints for indexing leading to an error
    #right now we're casting as ints for indexing, but that's a little gross
    phantoms = np.ones([n_phantoms,n_obj])*np.inf
    
    
    
    
    #TODO vectorize?
    for i in range(n_phantoms):
        for j in range(n_obj):
            for k in range(num_par):
                if pareto_array[k,j] == phantom_values[i,j]:
                    phantoms[i,j] = k
    
                    
    j_star, lambdas = calc_iSCORE(phantoms, systems, n_systems, num_par, n_obj, n_phantoms)  
    
    m_star = calc_iSCORE_MCE(systems,num_par,n_obj)
    
    
    
    
    iscore_constraints_wrapper.last_alphas = None
    
    constraint_values = lambda x: iscore_constraints_wrapper(x, systems,phantoms, num_par, m_star, j_star, lambdas,n_obj,  n_systems)[0]
    constraint_jacobian = lambda x: iscore_constraints_wrapper(x, systems,phantoms, num_par, m_star, j_star, lambdas,n_obj,  n_systems)[1]
   
    
    nonlinear_constraint = opt.NonlinearConstraint(constraint_values,-np.inf,0.0,jac=constraint_jacobian,keep_feasible=False)
    
    my_bounds = [(10**-50,np.inf) for i in range(num_par+1)] + [(0.0,np.inf)]
    
    if warm_start is None:
        warm_start = np.array([1/(2*num_par)]*num_par + [0.5] + [0])
    else:
        warm_start = np.append(warm_start[0:num_par], [sum(warm_start[num_par:]), 0])
    

    equality_constraint_array =np.ones(num_par + 2)
    
    equality_constraint_array[-1] = 0
    
    equality_constraint_bound = 1.0
    
    equality_constraint = opt.LinearConstraint(equality_constraint_array, \
                                               equality_constraint_bound, \
                                               equality_constraint_bound)
    
    #print(len(constraint_jacobian(warm_start)))
                
    #print(systems)
    res = opt.minimize(objective_function,warm_start, method='trust-constr', jac=True, hess = hessian_zero,\
                       bounds = my_bounds,\
                       constraints = [equality_constraint,\
                                      nonlinear_constraint],\
                       options = {'gtol': 10**-12, 'xtol': 10**-12, 'maxiter': 10000})
    
    stop_flag = 1
    if res.status ==0:
        stop_flag = 0
        
    while stop_flag == 0:
        print("looping")
        
        res = opt.minimize(objective_function,res.x, method='trust-constr', jac=True, hess = hessian_zero,\
                       bounds = my_bounds,\
                       constraints = [equality_constraint,\
                                      nonlinear_constraint]\
                                      )
        if res.status !=0:
            stop_flag = 1
    
        
    #print(res.constr_violation)
    #print(res.message)
    
    out_alphas = res.x
    

    alloc = np.zeros(n_systems)
    alloc[systems['pareto_indices']] = out_alphas[0:num_par]
    
    for j in range(n_systems-num_par):
        alloc[systems['non_pareto_indices'][j]] = lambdas[j]*out_alphas[-2]

    
    return alloc, out_alphas[-1]

def iscore_constraints(alphas, systems,phantoms, num_par, m_star, j_star, lambdas, n_obj, n_systems):
    """parameters:
            alphas: numpy array of length n_systems + 1 consisting of allocation for each system and estimated convergence rate
            systems : dict

                ``"obj"``
                A dictionary of numpy arrays, indexed by system number,each of which corresponds to the objective values of a system.

                ``"var"``
                A dictionary of 2d numpy arrays, indexed by system number,each of which corresponds to the covariance matrix of a system.

                ``"inv_var"``
                A dictionary of 2d numpy, indexed by system number,each of which corresponds to the inverse covariance matrix of a system.

                ``"pareto_indices"``
                A list of pareto systems ordered by the first objective.

                ``"non_pareto_indices"``
                A list of non-pareto systems ordered by the first objective.
            phantoms: numpy matrix with n_obj columns and an arbitrary number of rows, where each element is a pareto system number. Each row corresponds to a phantom pareto system - pareto system number n in column j implies that the phantom pareto has the same value in objective j as pareto system n
            num_par: integer, number of estimated pareto systems
            m_star: numpy matrix
            j_star: numpy_matrix
            lambdas: numpy_array
            n_obj: number of systems
            n_systems: integer, number of total systems

    Returns
    -------

        rates: numpy array, giving the value of z(estimated convergence rate) minus the convergence rate upper bound associated with each constraint\n
        jacobian: 2d numy array, giving the jacobian of the rates with respect to the vector alpha (including the final element z)
 """
    tol = 10**-50
    
    alphas[0:-1][alphas[0:-1] < tol] = 0
    MCE_rates, MCE_grads = MCE_iscore_rates(alphas, systems, m_star, n_obj, n_systems)
    
    MCI_rates, MCI_grads = MCI_iscore_rates(alphas, lambdas, j_star, systems, phantoms, num_par, n_obj, n_systems)
    
    rates = np.append(MCE_rates,MCI_rates,axis=0)
    
    grads = np.append(MCE_grads, MCI_grads, axis=0)
    
    return rates, grads

def MCI_iscore_rates(alphas, lambdas, j_star, systems, phantoms, num_par, n_obj, n_systems):
    """calculates the MCI Phantom rate constraint values and jacobian

    Parameters
    ----------
            alphas:  numpy array of length n_systems + 1 consisting of allocation for each system and estimated convergence rate
            lambdas: numpy array
            j_star: numpy  matrix
            systems : dict

                ``"obj"``
                A dictionary of numpy arrays, indexed by system number,each of which corresponds to the objective values of a system.

                ``"var"``
                A dictionary of 2d numpy arrays, indexed by system number,each of which corresponds to the covariance matrix of a system.

                ``"inv_var"``
                A dictionary of 2d numpy, indexed by system number,each of which corresponds to the inverse covariance matrix of a system.

                ``"pareto_indices"``
                A list of pareto systems ordered by the first objective.

                ``"non_pareto_indices"``
                A list of non-pareto systems ordered by the first objective.
            phantoms: numpy matrix with n_obj columns and an arbitrary number of rows, where each element is a pareto system number. Each row corresponds to a phantom pareto system - pareto system number n in column j implies that the phantom pareto has the same value in objective j as pareto system n
            num_par: integer, number of estimated pareto systems
            n_systems: integer, number of total systems
            n_obj: integer, number of objectives

    Returns
    -------
    MCE_rates: numpy array, giving the value of z(estimated convergence rate) minus the convergence rate upper bound associated with each MCE constraint\n
    MCE_grads: 2d numy array, giving the jacobian of the MCE constraint values with respect to the vector alpha (including the final element z)"""
    
    tol = 10**-50
    
    n_MCI = len(j_star)
    
    MCI_rates = np.zeros(n_MCI)
    MCI_grads = np.zeros([n_MCI, len(alphas)])
    
    for i in range(n_MCI):
        j = int(j_star[i,0])
        j_ind = systems['non_pareto_indices'][j]
        lambda_j = lambdas[j]
        
        alpha_j = lambda_j* alphas[-2]
        
        obj_j = systems['obj'][j_ind]
        cov_j = systems['var'][j_ind]
        
        phantom_ind = int(j_star[i,1])
        phantom_pareto_nums = phantoms[phantom_ind,:]
        
        if alpha_j < tol:
            MCI_rates[i] = alphas[-1]
            MCI_grads[i,-1] = 1
            #all the grads aside from z are zero here, already zero from initialization
        else:
            
            phantom_objectives = np.ones(n_obj)*np.inf
            phantom_vars = np.zeros(n_obj)
            phantom_alphas = np.zeros(n_obj)
            
            for b in range(n_obj):
                if phantom_pareto_nums[b] < np.inf:
                    phantom_pareto_num = int(phantom_pareto_nums[b])
                    phantom_pareto_ind = systems['pareto_indices'][phantom_pareto_num]
                    phantom_objectives[b] = systems['obj'][phantom_pareto_ind][b]
                    phantom_vars[b] = systems['var'][phantom_pareto_ind][b,b]
                    phantom_alphas[b] = alphas[phantom_pareto_num]
                    
            rate = 0
            grad_j = 0
            for m in range(n_obj):
                if obj_j[m] > phantom_objectives[m]:
                    
                    rate = rate + (phantom_alphas[m]*alpha_j * (phantom_objectives[m] - obj_j[m])**2)/(2*(alpha_j *phantom_vars[m] + phantom_alphas[m]*cov_j[m,m]))
                    
                    grad_j = grad_j + (phantom_alphas[m]**2 * cov_j[m,m] *(phantom_objectives[m]-obj_j[m])**2)/(2*(alpha_j*phantom_vars[m] + phantom_alphas[m]*cov_j[m,m])**2)
                    
                    grad = (alpha_j**2 * phantom_vars[m] *(phantom_objectives[m]-obj_j[m])**2)/(2*(alpha_j*phantom_vars[m] + phantom_alphas[m]*cov_j[m,m])**2)
                    
                    MCI_grads[i,int(phantom_pareto_nums[m])] = -1.0*grad
                    
            MCI_rates[i] = alphas[-1]-rate
            MCI_grads[i,-1] = 1
            MCI_grads[i,-2] = -1.0*lambda_j*grad_j
            
    return MCI_rates, MCI_grads


def MCE_iscore_rates(alphas, systems, M_star, n_obj, n_systems):
    """calculates the MCE iscore rate constraint values and jacobian

    Parameters
    ----------
    alphas:  numpy array of length n_systems + 1 consisting of allocation for each system and estimated convergence rate
    systems : dict
        ``"obj"``
        A dictionary of numpy arrays, indexed by system number,each of which corresponds to the objective values of a system.

        ``"var"``
        A dictionary of 2d numpy arrays, indexed by system number,each of which corresponds to the covariance matrix of a system.

        ``"inv_var"``
        A dictionary of 2d numpy, indexed by system number,each of which corresponds to the inverse covariance matrix of a system.

        ``"pareto_indices"``
        A list of pareto systems ordered by the first objective.

        ``"non_pareto_indices"``
        A list of non-pareto systems ordered by the first objective.
    M_star: numpy array
    n_obj: integer, number of objectives
    n_systems: integer, number of total systems


    Returns
    -------
    MCE_rates: numpy array, giving the value of z(estimated convergence rate) minus the convergence rate upper bound associated with each MCE constraint\n
    MCE_grads: 2d numy array, giving the jacobian of the MCE constraint values with respect to the vector alpha (including the final element z)
            """
    tol = 10**-50
    
    n_MCE = len(M_star)
    
    MCE_rates = np.zeros(n_MCE)
    
    MCE_grads = np.zeros([n_MCE,len(alphas)])
    
    for k in range(n_MCE):
        i =int(M_star[k,0])
        j = int(M_star[k,1])
        
        
        if (alphas[i] < tol or alphas[j] < tol):
            rate = alphas[-1]
            grad_i = 0
            grad_j = 0
        else:
            i_ind = systems['pareto_indices'][i]
            j_ind = systems['pareto_indices'][j]
            obj_i = systems['obj'][i_ind]
            cov_i = systems['var'][i_ind]
            obj_j = systems['obj'][j_ind]
            cov_j = systems['var'][j_ind]
            
            rate = 0
            grad_i = 0
            grad_j = 0
            
            #TODO vectorize
            for m in range(n_obj):
                if obj_i[m] > obj_j[m]:
                    rate = rate + (alphas[i]*alphas[j]*(obj_i[m]-obj_j[m])**2)/(2*(alphas[j]*cov_i[m,m] + alphas[i]*cov_j[m,m]))
                    grad_i = grad_i + (alphas[j]**2 * cov_i[m,m]*(obj_i[m]-obj_j[m])**2)/(2*(alphas[j]*cov_i[m,m] + alphas[i]*cov_j[m,m])**2)
                    grad_j = grad_j + (alphas[i]**2 * cov_j[m,m]*(obj_i[m]-obj_j[m])**2)/(2*(alphas[j]*cov_i[m,m] + alphas[i]*cov_j[m,m])**2)
            
            rate = alphas[-1] - rate
            
        MCE_rates[k] = rate
        MCE_grads[k,i] = -1*grad_i
        MCE_grads[k,j] = -1*grad_j
        MCE_grads[k,-1] = 1.0
    return MCE_rates, MCE_grads
            

def iscore_constraints_wrapper(alphas, systems,phantoms, num_par, m_star, j_star, lambdas, n_obj, n_systems):
    """scipy optimization methods don't directly support simultaneous computation
    of constraint values and their gradients. Additionally, it only considers one constraint and its gradient
    at a time, as separate functions.

    parameters
    ----------
        alphas:  numpy array of length n_systems + 1 consisting of allocation for each system and estimated convergence rate
        systems : dict
            ``"obj"``
            A dictionary of numpy arrays, indexed by system number,each of which corresponds to the objective values of a system.

            ``"var"``
            A dictionary of 2d numpy arrays, indexed by system number,each of which corresponds to the covariance matrix of a system.

            ``"inv_var"``
            A dictionary of 2d numpy, indexed by system number,each of which corresponds to the inverse covariance matrix of a system.

            ``"pareto_indices"``
            A list of pareto systems ordered by the first objective.

            ``"non_pareto_indices"``
            A list of non-pareto systems ordered by the first objective.
        phantoms: numpy matrix with n_obj columns and an arbitrary number of rows, where each element is a pareto system number. Each row corresponds to a phantom pareto system - pareto system number n in column j implies that the phantom pareto has the same value in objective j as pareto system n
        num_par: integer, number of estimated pareto systems
        n_obj: number of systems
        n_systems: integer, number of total systems

    Returns
    -------
    rates: numpy array, giving the value of z(estimated convergence rate) minus the convergence rate upper bound associated with each constraint\n
    jacobian: 2d numy array, giving the jacobian of the rates with respect to the vector alpha (including the final element z)
    """

    if all(alphas == iscore_constraints_wrapper.last_alphas):
        return iscore_constraints_wrapper.last_outputs
    else:
        
        rates, jacobian = iscore_constraints(alphas, systems,phantoms, num_par, m_star, j_star, lambdas, n_obj, n_systems)
        
        iscore_constraints_wrapper.last_alphas = alphas
        iscore_constraints_wrapper.last_outputs = rates, jacobian
        
        return rates, jacobian
    
    

def objective_function(alphas):
    """Parameters
    ----------
    alphas:  numpy array of length n_systems + 1 consisting of allocation for each system and estimated convergence rate\n

    """
    
def calc_iSCORE_MCE(systems, num_par, n_obj):
    """calculates the iSCORE for MCE constraints

    parameters
    ----------
        systems : dict
            ``"obj"``
            A dictionary of numpy arrays, indexed by system number,each of which corresponds to the objective values of a system.

            ``"var"``
            A dictionary of 2d numpy arrays, indexed by system number,each of which corresponds to the covariance matrix of a system.

            ``"inv_var"``
            A dictionary of 2d numpy, indexed by system number,each of which corresponds to the inverse covariance matrix of a system.

            ``"pareto_indices"``
            A list of pareto systems ordered by the first objective.

            ``"non_pareto_indices"``
            A list of non-pareto systems ordered by the first objective.
        num_par: integer, number of pareto systems
        n_obj: integer, number of objectives

    Returns
    -------
    M_star: a numpy matrix (sorry, I don't remember what's going on here - Nathan)

        """
    scores = np.zeros([num_par,num_par])
    all_scores = np.zeros(num_par*(num_par-1))
    M_star = np.zeros([num_par*n_obj,2])
    
    count = 0
    
    for i in range(num_par):
        i_ind = systems['pareto_indices'][i]
        obj_i = systems['obj'][i_ind]
        
        j_comps = np.ones(n_obj)*np.inf
        j_inds = np.ones(n_obj)*np.inf
        
        for j in range(num_par):
            if i != j:
                j_ind = systems['pareto_indices'][j]
                
                obj_j = systems['obj'][j_ind]
                cov_j = systems['var'][j_ind]
                
                score = 0
                binds = np.ones(n_obj)*np.inf
                for m in range(n_obj):
                    if obj_j[m]>obj_i[m]:
                        score = score + (obj_i[m]-obj_j[m])**2/(2*cov_j[m,m])
                        binds[m] = 1
                        
                j_current = binds*score
                
                scores[i,j] = score
                
                all_scores[count] = score
                
                count = count + 1
                #TODO vectorize?
                for m in range(n_obj):
                    if j_current[m] < j_comps[m]:
                        j_comps[m] = j_current[m]
                        j_inds[m] = j
                        
        L_inds = np.ones(n_obj)*i
        M_star[n_obj*i:n_obj*(i+1),:] = np.vstack((j_inds,L_inds)).T
    
                    
    #TODO not sure why we're doing this, but we remove the rows of M_star where
    #thie first column is infinity
    M_star = M_star[M_star[:,0]<np.inf,:]
    #switch columns and append
    M_star_b = M_star[:,[1,0]]
    
    M_star = np.append(M_star,M_star_b,axis=0)
    
    #add pairs of systems where SCORE < percentile(all scores)
    score_percentile = np.percentile(all_scores,25)
    
    for a in range(num_par):
        for b in range(num_par):
            if a!=b and scores[a,b]<=score_percentile:

                M_star = np.append(M_star,[[a,b]],axis=0)
                M_star = np.append(M_star,[[b,a]],axis=0)
                
    M_star = np.unique(M_star,axis=0)
    return M_star
    
    
def calc_iSCORE(phantoms, systems, n_systems, num_par, n_obj, n_phantoms):
    """calculates the iSCORE for MCI constraints

    parameters
    ----------
    phantoms: numpy matrix with n_obj columns and an arbitrary number of rows, where each element is a pareto system number. Each row corresponds to a phantom pareto system - pareto system number n in column j implies that the phantom pareto has the same value in objective j as pareto system n
    systems : dict
        ``"obj"``
        A dictionary of numpy arrays, indexed by system number,each of which corresponds to the objective values of a system.

        ``"var"``
        A dictionary of 2d numpy arrays, indexed by system number,each of which corresponds to the covariance matrix of a system.

        ``"inv_var"``
        A dictionary of 2d numpy, indexed by system number,each of which corresponds to the inverse covariance matrix of a system.

        ``"pareto_indices"``
        A list of pareto systems ordered by the first objective.

        ``"non_pareto_indices"``
        A list of non-pareto systems ordered by the first objective.
    n_systems: number of systems
    num_par: integer, number of pareto systems
    n_obj: integer, number of objectives
    n_phantoms: number of phantom systems

    Returns
    -------
    j_star: a numpy matrix (sorry, I don't remember what's going on here - Nathan)\n
    lambdas: a numpy array (ditto)

        """
    
    scores = np.zeros([n_phantoms, n_systems-num_par])
    j_star = np.zeros([n_phantoms*n_obj,2])
    
    for i in range(n_phantoms):
        phantom_indices = phantoms[i,:]
        phantom_objs = np.ones(n_obj)*np.inf
        
        for b in range(n_obj):
            if phantom_indices[b] < np.inf:
                pareto_num = int(phantom_indices[b])
                pareto_system = systems['pareto_indices'][pareto_num]
                phantom_objs[b] = systems['obj'][pareto_system][b]
        
        j_comps = np.ones(n_obj)*np.inf
        j_inds = np.ones(n_obj)*np.inf
        
        for j in range(n_systems-num_par):
            j_ind = systems['non_pareto_indices'][j]
            obj_j = systems['obj'][j_ind]
            cov_j = systems['var'][j_ind]
            
            score = 0
            binds = np.ones(n_obj)*np.inf
            for m in range(n_obj):
                if obj_j[m] > phantom_objs[m]:
                    score = score + (phantom_objs[m] - obj_j[m])**2/(2*cov_j[m,m])
                    binds[m] = 1
            
            j_current = binds*score
            
            scores[i,j] = score
            
            #determine if this non-pareto is closer than another
            #TODO vectorize
            for m in range(n_obj):
                if j_current[m] < j_comps[m]:
                    j_comps[m] = j_current[m]
                    j_inds[m] = j
                    
        L_indices = np.ones(n_obj)*i
        j_star[i*n_obj:(i+1)*n_obj,:] = np.vstack((j_inds,L_indices)).T
        
    inv_scores = 1/np.minimum.reduce(scores)
    lambdas = inv_scores/sum(inv_scores)
    
    j_star = j_star[j_star[:,0]<np.inf,:]
    
    j_star = np.unique(j_star,axis=0)
    
    return j_star, lambdas
    
    
    