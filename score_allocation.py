#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 18:07:50 2019

@author: nathangeldner
"""


import numpy as np
import pymoso.chnutils as utils
import itertools as it
import scipy.optimize as opt

from cvxopt import matrix, solvers

from phantom_allocation import find_phantoms
from brute_force_allocation import hessian_zero
from MCI_hard_coded import MCI_1d, MCI_2d, MCI_3d, MCI_four_d_plus
from MCE_hard_coded import MCE_2d, MCE_3d, MCE_four_d_plus
from score_calc_hard_coded import SCORE_1d, SCORE_2d, SCORE_3d, score_four_d_plus


# solvers.options['show_progress'] = False


def score_allocation(systems,warm_start = None):
    """Calculates Phantom Allocation given a set of systems and optional warm start
    
    Parameters:
        
        Systems: Dictionary with following keys and values
            'obj': a dictionary of objective value (float) tuples  keyed by system number
            'var': a dictionary of objective covariance matrices (numpy matrices) keyed by system number
            'pareto_indices': a list of integer system numbers of estimated pareto systems ordered by first objective value
            'non_pareto_indices': a list o finteger system numbers of estimated non-parety systems ordered by first objective value
            
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
    #also, inf doesn't cast to intmax if you cast as int, it ends up being very negative for some reason
    phantoms = np.ones([n_phantoms,n_obj])*np.inf
    
    
    
    
    #TODO vectorize?
    for i in range(n_phantoms):
        for j in range(n_obj):
            for k in range(num_par):
                if pareto_array[k,j] == phantom_values[i,j]:
                    phantoms[i,j] = k
                    
    j_star, lambdas = calc_SCORE(phantoms, systems, n_systems, num_par, n_obj, n_phantoms)  
    
    m_star = calc_SCORE_MCE(systems,num_par,n_obj)
    
    
    
    score_constraints_wrapper.last_alphas = None
    
    constraint_values = lambda x: score_constraints_wrapper(x, systems,phantoms, num_par, m_star, j_star, lambdas,n_obj,  n_systems)[0]
    constraint_jacobian = lambda x: score_constraints_wrapper(x, systems,phantoms, num_par, m_star, j_star, lambdas,n_obj,  n_systems)[1]
   
    
    
    nonlinear_constraint = opt.NonlinearConstraint(constraint_values,-np.inf,0.0,jac=constraint_jacobian,keep_feasible=False)
    
    my_bounds = [(10**-12,1.0) for i in range(num_par+1)] + [(0.0,np.inf)]
    
    if warm_start is None:
        warm_start = np.array([1/(2*num_par)]*num_par + [0.5] + [0])
    else:
        warm_start = np.append(warm_start[0:num_par], [sum(warm_start[num_par:]), 0])

    equality_constraint_array =np.ones(num_par + 2)
    
    equality_constraint_array[-1] = 0.0
    
    equality_constraint_bound = 1.0
    
    equality_constraint = opt.LinearConstraint(equality_constraint_array, \
                                               equality_constraint_bound, \
                                               equality_constraint_bound)
                
    res = opt.minimize(objective_function,warm_start, method='trust-constr', jac=True, hess = hessian_zero,\
                       bounds = my_bounds,\
                       constraints = [equality_constraint,\
                                      #bound_constraint, \
                                      nonlinear_constraint]\
                       )
    
    stop_flag = 1
    if res.status ==0:
        stop_flag = 0
        
    while stop_flag == 0:
        
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
    #sort output by system number
    alloc = np.zeros(n_systems)
    alloc[systems['pareto_indices']] = out_alphas[0:num_par]
    
    for j in range(n_systems-num_par):
        alloc[systems['non_pareto_indices'][j]] = lambdas[j]*out_alphas[-2]

    
    return alloc, out_alphas[-1]
        
        
        
def objective_function(alphas):
    """We want to maximize the convergence rate, so the objective function is -1 times the convergence rate
    and the gradient thereof is zero with respect to alphas and -1 with respect to the convergence rate

    parameters
    ----------
    alphas:  numpy array of length n_systems + 1 consisting of allocation for each system and estimated convergence rate,
            gradient = np.zeros(len(alphas)),
            gradient[-1] = -1.0,
             return -1.0*alphas[-1],gradient
    """

    gradient = np.zeros(len(alphas))
    gradient[-1] = -1
    return -1*alphas[-1],gradient

    
def score_constraints(alphas, systems,phantoms, num_par, m_star, j_star, lambdas, n_obj, n_systems):
    """parameters:
            alphas: numpy array of length n_systems + 1 consisting of allocation for each system and estimated convergence rate
            systems: dict, as described under calc_bf_allocation()
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
    tol = 10**-12
    alphas[0:-1][alphas[0:-1] < tol] = 0
    MCE_rates, MCE_grads = MCE_score_rates(alphas, systems, m_star, n_obj, n_systems)
    
    MCI_rates, MCI_grads = MCI_score_rates(alphas, lambdas, j_star, systems, phantoms, num_par, n_obj, n_systems)
    
    rates = np.append(MCE_rates,MCI_rates,axis=0)
    
    grads = np.append(MCE_grads, MCI_grads, axis=0)
    
    return rates, grads

def MCI_score_rates(alphas, lambdas, j_star, systems, phantoms, num_par, n_obj, n_systems):
    """calculates the MCI Phantom rate constraint values and jacobian
    
    Parameters
    ----------
            alphas:  numpy array of length n_systems + 1 consisting of allocation for each system and estimated convergence rate
            lambdas: numpy array
            j_star: numpy  matrix
            systems: dict, as described under calc_bf_allocation()
            phantoms: numpy matrix with n_obj columns and an arbitrary number of rows, where each element is a pareto system number. Each row corresponds to a phantom pareto system - pareto system number n in column j implies that the phantom pareto has the same value in objective j as pareto system n
            num_par: integer, number of estimated pareto systems
            n_systems: integer, number of total systems
            n_obj: integer, number of objectives

    Returns
    -------
    MCE_rates: numpy array, giving the value of z(estimated convergence rate) minus the convergence rate upper bound associated with each MCE constraint\n
    MCE_grads: 2d numy array, giving the jacobian of the MCE constraint values with respect to the vector alpha (including the final element z)"""
      
    tol = 10**-12
    
    n_MCI = len(j_star)
    
    MCI_rates = np.zeros(n_MCI)
    MCI_grads = np.zeros([n_MCI, len(alphas)])
    
    
    for i in range(n_MCI):
        j = int(j_star[i,0])
        j_ind = systems['non_pareto_indices'][j]
        lambda_j = lambdas[j]
        alpha_j = lambda_j * alphas[-2]
        obj_j = systems['obj'][j_ind]
        cov_j = systems['var'][j_ind]
        
        phantom_ind = int(j_star[i,1])
        phantom_pareto_inds = phantoms[phantom_ind,:]
        
        if alpha_j < tol:
            #rate is 0, returned rate is z, grads wrt anything but z is left as zero
            MCI_rates[i] = alphas[-1]
            MCI_grads[i,-1] = 1
        else:
            
            phantom_objectives = np.zeros(n_obj)
            phantom_vars = np.zeros(n_obj)
            phantom_alphas = np.zeros(n_obj)
            objectives_playing = np.array(range(n_obj))
            alpha_zeros = 0
            n_objectives_playing = n_obj
            
            for b in range(n_obj):
                if phantom_pareto_inds[b] < np.inf:
                    pareto_system_num = int(phantom_pareto_inds[b])
                    pareto_system_ind = systems['pareto_indices'][pareto_system_num]
                    phantom_objectives[b] = systems['obj'][pareto_system_ind][b]
                    phantom_vars[b] = systems['var'][pareto_system_ind][b,b]
                    if alphas[pareto_system_num] < tol:
                        phantom_alphas[b] = 0
                        alpha_zeros +=1
                    else:
                        phantom_alphas[b] = alphas[pareto_system_num]
                else:
                    n_objectives_playing -=1
               
            objectives_playing = objectives_playing[phantom_pareto_inds<np.inf]
            
            obj_j = obj_j[objectives_playing]
            cov_j = cov_j[np.ix_(objectives_playing,objectives_playing)]
            
            phantom_objectives = phantom_objectives[objectives_playing]
            phantom_vars = phantom_vars[objectives_playing]
            phantom_alphas = phantom_alphas[objectives_playing]
            
            if alpha_zeros ==  n_objectives_playing:
                MCI_rates[i] = alphas[-1]
                MCI_grads[i,phantom_pareto_inds[phantom_pareto_inds<np.inf].astype(int)] = -0.5*((obj_j-phantom_objectives)**2)/phantom_vars
                MCI_grads[i,-1] = 1
            else:
                #TODO hard code solutions for < 4 objectives
                length = len(objectives_playing)
                if length ==1:
                    rate, grad_j, phantom_grads = MCI_1d(alpha_j, obj_j, cov_j, phantom_alphas, phantom_objectives, phantom_vars)
                elif length ==2:
                    rate, grad_j, phantom_grads = MCI_2d(alpha_j, obj_j, cov_j, phantom_alphas, phantom_objectives, phantom_vars)
                elif length ==3:
                    rate, grad_j, phantom_grads = MCI_3d(alpha_j, obj_j, cov_j, phantom_alphas, phantom_objectives, phantom_vars)
                else:
                    rate, grad_j, phantom_grads = MCI_four_d_plus(alpha_j, obj_j, cov_j, phantom_alphas, phantom_objectives, phantom_vars)
                
                MCI_rates[i] = alphas[-1]-rate
                phantom_grads[phantom_grads<tol] = 0
                MCI_grads[i, phantom_pareto_inds[phantom_pareto_inds<np.inf].astype(int)]  = -1.0*phantom_grads
                MCI_grads[i, -2] = -1*lambda_j*grad_j
                MCI_grads[i,-1] = 1
                
    return MCI_rates, MCI_grads
                
        
        

def MCE_score_rates(alphas, systems, M_star, n_obj, n_systems):
    """calculates the MCE score rate constraint values and jacobian
    
    Parameters
    ----------
    alphas:  numpy array of length n_systems + 1 consisting of allocation for each system and estimated convergence rate
    systems: dict, as described under calc_bf_allocation()
    M_star: numpy array
    n_systems: integer, number of total systems
    n_obj: integer, number of objectives

    Returns
    -------
    MCE_rates: numpy array, giving the value of z(estimated convergence rate) minus the convergence rate upper bound associated with each MCE constraint\n
    MCE_grads: 2d numy array, giving the jacobian of the MCE constraint values with respect to the vector alpha (including the final element z)
            """
    
    tol = 10**-12
    
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
            inv_cov_i = systems['inv_var'][i_ind]
            obj_j = systems['obj'][j_ind]
            inv_cov_j = systems['inv_var'][j_ind]
            
            #TODO hard code solutions for <4 dimensions
            if n_obj == 2:
                rate, grad_i, grad_j = MCE_2d(alphas[i],alphas[j],obj_i,systems["var"][i_ind]\
                                                              ,obj_j,systems["var"][j_ind], inv_cov_i\
                                                              , inv_cov_j)
            elif n_obj ==3:
                rate, grad_i, grad_j = MCE_3d(alphas[i],alphas[j],obj_i,systems["var"][i_ind]\
                                                              ,obj_j,systems["var"][j_ind], inv_cov_i\
                                                              , inv_cov_j)
            else:
                rate, grad_i, grad_j = MCE_four_d_plus(alphas[i], alphas[j], obj_i, inv_cov_i, obj_j, inv_cov_j, n_obj)
            
            
            rate = alphas[-1]-rate
            
        MCE_rates[k] = rate
        MCE_grads[k,i] = -1*grad_i
        MCE_grads[k,j] = -1*grad_j
        MCE_grads[k,-1] = 1.0
        
    return MCE_rates, MCE_grads
    
    
    
def score_constraints_wrapper(alphas, systems,phantoms, num_par, m_star, j_star, lambdas, n_obj, n_systems):
    """scipy optimization methods don't directly support simultaneous computation
    of constraint values and their gradients. Additionally, it only considers one constraint and its gradient
    at a time, as separate functions. Thus we check whether we're looking at the same alphas
    as the last call, and if so return the same output
    
    parameters:
            alphas:  numpy array of length n_systems + 1 consisting of allocation for each system and estimated convergence rate
            systems: dict, as described under calc_bf_allocation()
            phantoms: numpy matrix with n_obj columns and an arbitrary number of rows, where each element is a pareto system number. Each row corresponds to a phantom pareto system - pareto system number n in column j implies that the phantom pareto has the same value in objective j as pareto system n
            num_par: integer, number of estimated pareto systems
            n_obj: number of systems
            n_systems: integer, number of total systems
            
    Returns
    -------
    rates: numpy array, giving the value of z(estimated convergence rate) minus the convergence rate upper bound associated with each constraint\n
    jacobian: 2d numy array, giving the jacobian of the rates with respect to the vector alpha (including the final element z)
            
    """
    if all(alphas == score_constraints_wrapper.last_alphas):
        return score_constraints_wrapper.last_outputs
    else:
        
        rates, jacobian = score_constraints(alphas, systems,phantoms, num_par, m_star, j_star, lambdas, n_obj, n_systems)
        
        score_constraints_wrapper.last_alphas = alphas
        score_constraints_wrapper.last_outputs = rates, jacobian
        
        return rates, jacobian
    
    
    
def calc_SCORE_MCE(systems,num_par,n_obj):
    """calculates the SCORE for MCE constraints
    
    parameters:
            systems:  systems: dict, as described under calc_score_allocation()
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
                
                #TODO hard code solutions for <4 objectives
                
                if n_obj ==1:
                    score, binds = SCORE_1d(obj_i, obj_j, cov_j)
                elif n_obj ==2:
                        score, binds = SCORE_2d(obj_i, obj_j, cov_j)
                elif n_obj ==3:
                    score, binds = SCORE_3d(obj_i, obj_j, cov_j)
                else:
                    score, binds = score, binds = score_four_d_plus(obj_i, obj_j, cov_j)
                
                scores[i,j] = score
                
                all_scores[count] = score
                
                count = count+1
                
                j_current = binds
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
        
        
        
        
    


def calc_SCORE(phantoms, systems, n_systems, num_par, n_obj, n_phantoms):
    """calculates the SCORE for MCI constraints
    
    parameters
    ----------
            phantoms: numpy matrix with n_obj columns and an arbitrary number of rows, where each element is a pareto system number. Each row corresponds to a phantom pareto system - pareto system number n in column j implies that the phantom pareto has the same value in objective j as pareto system n
            systems: dict, as described under calc_score_allocation()
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
    j_star = np.zeros([n_phantoms*n_obj, 2])
    #note: the matlab code pre-computes several vectors for speed here
    #which I instead initialize individually
    #this is because initializing vector v at this point, setting v_instance = v
    #and then modifying v_instance would modify v, which is undesireable
    
    #loop over phantoms
    for i in range(n_phantoms):
        
        #phantom_pareto_inds refers to the pareto system number from which each phantom objective
        #gets its value. We drop objectives that are infinity in the phantom
        #and keep track of the rest of the objectives in objectives_playing
        phantom_pareto_nums = phantoms[i,:]
        objectives_playing = np.array(range(n_obj))[phantom_pareto_nums<np.inf]
        phantom_pareto_nums = phantom_pareto_nums[phantom_pareto_nums<np.inf]
        #pareto_phantom_inds refers to the actual system indices
        
        
        
        
        
        n_obj_playing = len(objectives_playing)
        
        phantom_objectives = np.zeros(n_obj_playing)
        #extract the objectives for the phantoms
        for obj in range(n_obj_playing):
            phantom_pareto_ind = systems['pareto_indices'][int(phantom_pareto_nums[obj])]
            phantom_objectives[obj] = systems['obj'][phantom_pareto_ind][objectives_playing[obj]]
        
        j_comps = np.ones(n_obj)*np.inf
        j_indices = np.ones(n_obj)*np.inf
        size = len(objectives_playing)
        for j in range(n_systems-num_par):
            j_ind = systems['non_pareto_indices'][j]
            obj_j = systems['obj'][j_ind][objectives_playing]
            cov_j = systems['var'][j_ind][np.ix_(objectives_playing,objectives_playing)]
            
            #TODO hard code solutions for 1, 2, 3 objectives
            if size ==1:
                score, binds = SCORE_1d(phantom_objectives, obj_j, cov_j)
            elif size ==2:
                score, binds = SCORE_2d(phantom_objectives, obj_j, cov_j)
            elif size ==3:
                score, binds = SCORE_3d(phantom_objectives, obj_j, cov_j)
            else:
                score, binds = score_four_d_plus(phantom_objectives, obj_j, cov_j)
            
            j_current = np.ones(n_obj)*np.inf
            j_current[objectives_playing] = binds
            scores[i,j] = score
            
            for m in range(n_obj):
                if j_current[m] < j_comps[m]:
                    j_comps[m] = j_current[m]
                    j_indices[m] = j
        
        L_indices = np.ones(n_obj)*i
        
        #for every constraint (with n_obj rows in J_star), we want the non-pareto index
        #per objective and the phantom index
        #TODO consider instead having one row per constraint, one column per objective
        #and separate matrices (arrays) for J indices and L indices. Or actually we wouldn't
        #need L indices I think because the J matrix would then be ordered as such


        j_star[n_obj*i:n_obj*(i+1),:] = np.vstack((j_indices,L_indices)).T
        
    #inv_scores is the inverse of the minimum of each column in scores, resulting in one
    #value per non-pareto system
    inv_scores = 1/np.minimum.reduce(scores)
    Lambdas = inv_scores/sum(inv_scores)
    
    #TODO not sure why we're doing this, but we remove the rows of J_star where
    #thie first column is infinity
    j_star = j_star[j_star[:,0]<np.inf,:]
    
    j_star = np.unique(j_star,axis=0)
    
    return j_star, Lambdas
    
        
            
       
    
    

    
    
    
    