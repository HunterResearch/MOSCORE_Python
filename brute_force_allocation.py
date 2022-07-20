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
import scipy.linalg as linalg
from cvxopt import matrix, solvers
from MCE_hard_coded import MCE_2d, MCE_3d, MCE_four_d_plus

solvers.options['show_progress'] = False

def calc_bf_allocation(systems, warm_start = None):
    """Calculates Brute Force Allocation given a set of systems and optional warm start
    
    Parameters:
        
        Systems: Dictionary with following keys and values
            'obj': a dictionary of objective value (float) tuples  keyed by system number
            'var': a dictionary of objective covariance matrices (numpy matrices) keyed by system number
            'pareto_indices': a list of integer system numbers of estimated pareto systems ordered by first objective value
            'non_pareto_indices': a list o finteger system numbers of estimated non-parety systems ordered by first objective value
            
        warm_start: numpy array of length equal to the number of system, which sums to 1
        
        
    Returns:
        
        out_tuple:
            out_tuple[0]: the estimated optimal allocation of simulation runs assuming that estimated objectives and variances are true
            out_tuple[1]: the estimated convergence rate associatedc with the optimal allocation
            """
    
    #extract number of objectives
    n_obj = len(systems["obj"][0])
    #extract number of systems
    n_systems = len(systems["obj"])
    #extract number of pareto systems    
    num_par = len(systems["pareto_indices"])
    

    
    v = range(n_obj)
    #kappa here is a list of tuples. each tuple is of length num_par with elements
    #corresponding to objective indices. 
    #to exclude a pareto, a non-pareto must dominate the pareto with number equal to the kappa index
    #along objective equal to the kappa value, for some kappa.
    kappa = list(it.product(v,repeat=num_par))
    
    #we don't have a good way to pass in constraint values and gradients simultaneously
    #so this is to help with our workaround for that. described under brute_force_constraitns_wrapper
    brute_force_constraints_wrapper.last_alphas = None
    
    #lambda is for function handles. We define a callable for the constraint values, and another for the constraint
    #jacobian
    constraint_values = lambda x: brute_force_constraints_wrapper(x, systems, kappa, num_par,n_obj, n_systems)[0]
    
    constraint_jacobian = lambda x: brute_force_constraints_wrapper(x, systems, kappa, num_par,n_obj, n_systems)[1]
    #define nonlinear constraint object for the optimizer. will not work if we switch away from 
    #trust_constr, but the syntax isn't that different if we do.
    nonlinear_constraint = opt.NonlinearConstraint(constraint_values,-np.inf,0.0,jac=constraint_jacobian,keep_feasible=False)
    
    #define bounds on alpha vaues and z (the last element of our decision variable array)
    my_bounds = [(10**-12,1.0) for i in range(n_systems)] + [(0.0,np.inf)]
    
    if warm_start is None:
        warm_start = np.array([1.0/n_systems]*n_systems +[0])
    else:
        warm_start = np.append(warm_start, 0)
    
    
    #The objective is -z, and its derivative wrt z is -1
    obj_function = lambda x: objective_function(x)
    
    #set sum of alphas (not z) to 1
    equality_constraint_array =np.ones(n_systems + 1)
    
    equality_constraint_array[-1] = 0.0
    
    equality_constraint_bound = 1.0
    
    equality_constraint = opt.LinearConstraint(equality_constraint_array, \
                                               equality_constraint_bound, \
                                               equality_constraint_bound)
    

    
    res = opt.minimize(obj_function,warm_start, method='trust-constr', jac=True, hess = hessian_zero,\
                       bounds = my_bounds,\
                       constraints = [equality_constraint,\
                                      #bound_constraint, \
                                      nonlinear_constraint]\
                       )
    
    stop_flag = 1
    if res.status ==0:
        stop_flag = 0
        
    while stop_flag == 0:
        print('cycling')
        res = opt.minimize(objective_function,res.x, method='trust-constr', jac=True, hess = hessian_zero,\
                       bounds = my_bounds,\
                       constraints = [equality_constraint,\
                                      nonlinear_constraint]\
                                      )
        if res.status !=0:
            stop_flag = 1
    
    #print(res.constr_violation)
    #print(res.message)
    

    

    #res.x includes alphas and z. to return only alphas, subset with res.x[0:-1]
    return res.x[0:-1], res.x[-1]
#hessian of the objective function is just a matrix of zeros
def hessian_zero(alphas):
    """returns a square matrix of zeros with len(alpha) rows and columns"""
    return np.zeros([len(alphas),len(alphas)])

    

def objective_function(alphas):
    """We want to maximize the convergence rate, so the objective function is -1 times the convergence rate
    and the gradient thereof is zero with respect to alphas and -1 with respect to the convergence rate"""
    gradient = np.zeros(len(alphas))
    gradient[-1] = -1
    return -1*alphas[-1],gradient


def brute_force_constraints_wrapper(alphas, systems, kappa, num_par,n_obj, n_systems):
    """scipy optimization methods don't directly support simultaneous computation
    of constraint values and their gradients. Additionally, it only considers one constraint and its gradient
    at a time, as separate functions. Thus we check whether we're looking at the same alphas
    as the last call, and if so return the same output
    
    parameters:
            alphas: numpy array of length n_systems + 1 consisting of allocation for each system and estimated convergence rate
            systems: dict, as described under calc_bf_allocation()
            kappa: numpy list (length n_obj^num_par) of tuples (length num_par), such that each tuple indicates that an MCI event may occur if a non-pareto dominates pareto i in objective tuple[i] for all i in range(num_par)
            num_par: integer, number of estimated pareto systems
            n_obj: integer, number of objectives
            n_systems: integer, number of total systems
            
    output:
            rates: numpy array, giving the value of z(estimated convergence rate) minus the convergence rate upper bound associated with each constraint
            jacobian: 2d numy array, giving the jacobian of the rates with respect to the vector alpha (including the final element z)
            
    """
    if all(alphas == brute_force_constraints_wrapper.last_alphas):
        return brute_force_constraints_wrapper.last_outputs
    else:
        
        rates, jacobian = brute_force_constraints(alphas,systems,kappa, num_par, n_obj, n_systems)
        
        brute_force_constraints_wrapper.last_alphas = alphas
        brute_force_constraints_wrapper.last_outputs = rates, jacobian
        
        return rates, jacobian
    
def brute_force_constraints(alphas, systems, kappa, num_par,n_obj, n_systems):
    """calculates MCE constraints and MCI constraints on the convergence rate and appends them together, where the value of each constraint is equal to z, the total rate estimator, minus the rate associated with each possible MCI and MCE event, each of which serves as an upper bound on the total rate
    
    parameters:
            alphas: numpy array of length n_systems + 1 consisting of allocation for each system and estimated convergence rate
            systems: dict, as described under calc_bf_allocation()
            kappa: numpy list (length n_obj^num_par) of tuples (length num_par), such that each tuple indicates that an MCI event may occur if a non-pareto dominates pareto i in objective tuple[i] for all i in range(num_par)
            num_par: integer, number of estimated pareto systems
            n_obj: integer, number of objectives
            n_systems: integer, number of total systems
            
    output:
            rates: numpy array, giving the value of z(estimated convergence rate) minus the convergence rate upper bound associated with each constraint
            jacobian: 2d numy array, giving the jacobian of the constraint values with respect to the vector alpha (including the final element z)
            
    """
      
    #get MCE constraint values and gradients
    MCE_rates, MCE_grads = MCE_brute_force_rates(alphas, systems, num_par,n_systems, n_obj)
    #get MCI constraint values and gradients
    MCI_rates, MCI_grads = MCI_brute_force_rates(alphas, systems, kappa, num_par, n_systems, n_obj)
    #put em all together
    rates = np.append(MCE_rates,MCI_rates,axis=0)
    
    grads = np.append(MCE_grads,MCI_grads,axis=0)
    
    
    return rates, grads
        
        
def MCE_brute_force_rates(alphas, systems, num_par,n_systems, n_obj):
    """calculates the MCE brute force rate constraint values and jacobian
    
    parameters:
            alphas:  numpy array of length n_systems + 1 consisting of allocation for each system and estimated convergence rate
            systems: dict, as described under calc_bf_allocation()
            num_par: integer, number of estimated pareto systems
            n_systems: integer, number of total systems
            n_obj: integer, number of objectives
    output:
            MCE_rates: numpy array, giving the value of z(estimated convergence rate) minus the convergence rate upper bound associated with each MCE constraint
            MCE_grads: 2d numy array, giving the jacobian of the MCE constraint values with respect to the vector alpha (including the final element z)
            """
    
    
    #negative alphas break the quadratic optimizer called below, and alphas that
    #are too small may give us numerical precision issues
    tol = 10**-12
    alphas[0:-1][alphas[0:-1]<=tol] = 0
    

    
    #there's an MCE constraint for every non-diagonal element of a paretos by paretos matrix
    n_MCE = num_par*(num_par-1)
    MCE_rates = np.zeros(n_MCE)
    MCE_grads = np.zeros([n_MCE, n_systems + 1])
    
    #assign value of 1 to d constraint_val / d z
    MCE_grads[:,n_systems] = 1
    
    count = 0
    
    for i in systems['pareto_indices']:
        for j in systems['pareto_indices']:
            if i!=j:
                

                if alphas[i]<=tol or alphas[j]<=tol:
                    #it can be shown that if either alpha is zero, the rate is zero and the derivatives are zero
                    #constraint values are z - rate, so set "rate" here to z

                    rate = alphas[-1]
                    d_rate_d_i = 0
                    d_rate_d_j = 0
                    
                else:
                    if n_obj == 2:
                        rate, d_rate_d_i, d_rate_d_j = MCE_2d(alphas[i],alphas[j],systems["obj"][i],systems["var"][i]\
                                                              ,systems["obj"][j],systems["var"][j], systems["inv_var"][i]\
                                                              , systems["inv_var"][j])
                    elif n_obj ==3:
                        rate, d_rate_d_i, d_rate_d_j = MCE_3d(alphas[i],alphas[j],systems["obj"][i],systems["var"][i]\
                                                              ,systems["obj"][j],systems["var"][j], systems["inv_var"][i]\
                                                              , systems["inv_var"][j])
                    else:
                        rate, d_rate_d_i, d_rate_d_j = MCE_four_d_plus(alphas[i],alphas[j],systems["obj"][i],\
                                                                   systems["inv_var"][i] ,systems['obj'][j],\
                                                                   systems['inv_var'][j],n_obj)
                        
                    
                    
                    rate = alphas[-1] - rate
                

                
                MCE_rates[count] = rate
                MCE_grads[count,i] = -1.0*d_rate_d_i
                MCE_grads[count,j] = -1.0*d_rate_d_j
                
                count = count+1
                
    
                
    return MCE_rates, MCE_grads
                
            

    
    
        
        
        
def MCI_brute_force_rates(alphas, systems, kappa, num_par, n_systems, n_obj):
    """calculates the MCE brute force rate constraint values and jacobian
    
    parameters:
            alphas:  numpy array of length n_systems + 1 consisting of allocation for each system and estimated convergence rate
            systems: dict, as described under calc_bf_allocation()
            kappa: numpy list (length n_obj^num_par) of tuples (length num_par), such that each tuple indicates that an MCI event may occur if a non-pareto dominates pareto i in objective tuple[i] for all i in range(num_par)
            num_par: integer, number of estimated pareto systems
            n_systems: integer, number of total systems
            n_obj: integer, number of objectives
    output:
            MCI_rates: numpy array, giving the value of z(estimated convergence rate) minus the convergence rate upper bound associated with each MCI constraint
            MCI_grads: 2d numy array, giving the jacobian of the MCI constraint values with respect to the vector alpha (including the final element z)
            """

    
    tol = 10**-12
    
    n_kap = len(kappa)
    
    n_non_par = n_systems-num_par
    #we have one constraint for every non-pareto for every kappa vector
    n_MCI = n_non_par *n_kap
    
    MCI_rates = np.zeros(n_MCI)
    
    MCI_grad = np.zeros([n_MCI, n_systems + 1])
    
    MCI_grad[:,n_systems] = 1
    
    #set alphas of  systems to zero if they are less than tol
    #however, zeros break the optimizer, instead use tiny value
    alphas[0:-1][alphas[0:-1]<=tol] = tol
    
    
    count = 0
    
    for j in systems['non_pareto_indices']:
        
        obj_j = systems['obj'][j]
        inv_var_j = systems['inv_var'][j]
        
        #TODO see if this actually belongs, I put it because it was what we were doing in the phantom and it
        #keeps the quadratic optimization from breaking if we don't set zero alphas to tol
        for kap in kappa:
            if False:#alphas[j]<= tol:
                #the rate and gradients are zero, only have to worry about gradient wrt z since 
                #we initialize with zero
                #MCI_grad[count,-1] = 1
                #?
                None
                
            else:
        #initialize objectives and variances
                relevant_objectives = np.zeros(num_par)
                relevant_variances = np.zeros(num_par)
                
                for p in range(num_par):
                    #get the actual index of the pareto system
                    pareto_system_ind = systems['pareto_indices'][p]
                    #extract variances and objective values
                    relevant_objectives[p] = systems['obj'][pareto_system_ind][kap[p]]
                    relevant_variances[p] = systems['var'][pareto_system_ind][kap[p],kap[p]]
                #get the alpha of the pareto system    
                pareto_alphas = alphas[systems['pareto_indices']]
                    
    
                
                
                
                #quardatic optimization step
                P = linalg.block_diag(alphas[j]*inv_var_j,np.diag(pareto_alphas*(1/relevant_variances)))
                
                q = matrix(-1 * np.append(alphas[j]*inv_var_j @ obj_j, \
                                          pareto_alphas * 1/relevant_variances *relevant_objectives))
                
                G_left_side = np.zeros([num_par,n_obj])
                
                G_left_side[range(num_par),kap] = 1
                
                
                
       
                G = matrix(np.append(G_left_side, -1*np.identity(num_par),axis=1))
                
                h = matrix(np.zeros(num_par))
                
                P = matrix((P + P.transpose())/2)
                
                x_star = np.array(solvers.qp(P,q,G,h)['x']).flatten()
                
                rate = 0.5*alphas[j]*(obj_j-x_star[0:n_obj])@ inv_var_j @(obj_j - x_star[0:n_obj]) +\
                0.5*np.sum(pareto_alphas* (x_star[n_obj:] - relevant_objectives)*(1/relevant_variances)*(x_star[n_obj:]-relevant_objectives))
                
                MCI_grad[count,j] = -1.0*0.5*(obj_j-x_star[0:n_obj])@ inv_var_j @(obj_j - x_star[0:n_obj])
                
                MCI_grad[count,systems['pareto_indices']] = -1.0*0.5*(x_star[n_obj:] - relevant_objectives)*(1/relevant_variances)*(x_star[n_obj:]-relevant_objectives)
    
                
                MCI_rates[count] = alphas[-1] - rate
            
            count = count + 1
            
    return MCI_rates, MCI_grad
            


    
        
        
        
        
        
        
        
        
        