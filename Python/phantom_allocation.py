#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 27 19:25:06 2019

@author: nathangeldner
"""

import numpy as np
import pymoso.chnutils as utils
import itertools as it
import scipy.optimize as opt
import scipy.linalg as linalg
from cvxopt import matrix, solvers
from brute_force_allocation import MCE_brute_force_rates, objective_function, hessian_zero


solvers.options['show_progress'] = False

def calc_phantom_allocation(systems, warm_start=None):
    #every comment from the brute_force_allocation top level function applies here, but I'm too tired right now to copy them over
    n_obj = len(systems["obj"][0])
    
    n_systems = len(systems["obj"])
        
    num_par = len(systems["pareto_indices"])
    #get array of pareto values for the phantom finder
    pareto_array  = np.zeros([num_par,n_obj])
    
    for i in range(num_par):
        pareto_array[i,:] = systems['obj'][systems['pareto_indices'][i]]
    #get the phantoms
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
                    phantoms[i,j] = systems['pareto_indices'][k]
    
                    
    phantom_constraints_wrapper.last_alphas = None
    
    constraint_values = lambda x: phantom_constraints_wrapper(x, systems, phantoms, num_par,n_obj, n_systems)[0]
    
    constraint_jacobian = lambda x: phantom_constraints_wrapper(x, systems, phantoms, num_par,n_obj, n_systems)[1]
    
    nonlinear_constraint = opt.NonlinearConstraint(constraint_values,-np.inf,0.0,jac=constraint_jacobian,keep_feasible=False)
    
    my_bounds = [(10**-12,1.0) for i in range(n_systems)] + [(0.0,np.inf)]
    
    if warm_start is None:
        warm_start = np.array([1.0/n_systems]*n_systems +[0.0])
    
    
    obj_function = lambda x: objective_function(x,n_systems)
    
    
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
    
    print(res.constr_violation)
    print(res.message)
    

    

    
    return res.x


def phantom_constraints_wrapper(alphas, systems, phantoms, num_par,n_obj, n_systems):
    """scipy optimization methods don't directly support simultaneous computation
    of constraint values and their gradients. Additionally, it only considers one constraint and its gradient
    at a time, as separate functions. Thus we check whether we're looking at the same alphas
    as the last call, and if so return the same output"""
    if all(alphas == phantom_constraints_wrapper.last_alphas):
        return phantom_constraints_wrapper.last_outputs
    else:
        
        rates, jacobian = phantom_constraints(alphas,systems,phantoms, num_par, n_obj, n_systems)
        
        phantom_constraints_wrapper.last_alphas = alphas
        phantom_constraints_wrapper.last_outputs = rates, jacobian
        
        return rates, jacobian
    
def phantom_constraints(alphas, systems, phantoms, num_par,n_obj, n_systems):
    
    
    MCE_rates, MCE_grads = MCE_brute_force_rates(alphas, systems, num_par,n_systems, n_obj)
    
    MCI_rates, MCI_grads = MCI_phantom_rates(alphas, systems, phantoms, num_par, n_systems, n_obj)
    
    rates = np.append(MCE_rates,MCI_rates,axis=0)
    
    grads = np.append(MCE_grads,MCI_grads,axis=0)
    
    
    return rates, grads

def MCI_phantom_rates(alphas,systems,phantoms,num_par,n_systems,n_obj):
    tol = 10**-12
    
    n_phantoms = len(phantoms)
    
    n_nonpar = n_systems - num_par
    
    n_MCI = n_nonpar*n_phantoms
    
    MCI_rates = np.zeros(n_MCI)
    
    MCI_grads = np.zeros([n_MCI,n_systems+1])
    
    count = 0
    
    alphas[0:-1][alphas[0:-1]<=tol] = 0
    
    for j in systems['non_pareto_indices']:
        
        for l in range(n_phantoms):
            
            #get the pareto indices corresponding to phantom l
            phantom_indices = phantoms[l,:]
            
            if alphas[j]<= tol:
                #the rate and gradients are zero, only have to worry about gradient wrt z since 
                #we initialize with zero
                MCI_grads[count,-1] = 1
            else:
                phantom_obj = np.zeros(n_obj)
                phantom_var = np.zeros(n_obj)
                phantom_alphas = np.zeros(n_obj)
                
                phantom_objectives = np.array(range(n_obj))
                phantom_objective_count = n_obj
                
                alpha_zeros = 0
                
                
                
                for b in range(n_obj):
                    if phantom_indices[b]<np.inf:
                        
                        pareto_system = int(phantom_indices[b])
                        
                        phantom_obj[b] = systems['obj'][pareto_system][b]
                        phantom_var[b] = systems['var'][pareto_system][b,b]
                        
                        if alphas[pareto_system] <=tol:
                            phantom_alphas[b] = 0
                            alpha_zeros = alpha_zeros + 1
                        else:
                            phantom_alphas[b] = alphas[pareto_system]
                    else:
                        phantom_objective_count -=1
                        
                
                phantom_objectives = phantom_objectives[phantom_indices<np.inf]
                
                    
                        
                obj_j = systems['obj'][j][phantom_objectives]
                #only want covariances for the phantom objectives, np.ix_ allows us to subset nicely that way
                cov_j = systems['var'][j][np.ix_(phantom_objectives,phantom_objectives)]
                
                
                
                phantom_obj = phantom_obj[phantom_objectives]
                phantom_var = phantom_var[phantom_objectives]
                phantom_alphas = phantom_alphas[phantom_objectives]
                
                if alpha_zeros == phantom_objective_count:
                    rate = 0
                    grad_j = 0
                    phantom_grads = 0.5*((obj_j-phantom_obj)**2)/phantom_var
                    
                    #note: floats equal to ints don't get automatically converted when used for indices, need to convert
                    MCI_grads[count,phantom_indices[phantom_indices<np.inf].astype(int)] = -1.0*phantom_grads
                    MCI_grads[count, -1] = 1
                    
                    MCI_rates[count] = alphas[-1]-rate
                    
                else:
                    #TODO hard code solutions for 1-3 objectives
                    rate, grad_j, phantom_grads = MCI_four_d_plus(alphas[j], obj_j, cov_j, phantom_alphas, phantom_obj, phantom_var)


                    MCI_grads[count,phantom_indices[phantom_indices<np.inf].astype(int)] = -1.0*phantom_grads
                    MCI_grads[count, -1] = 1
                    MCI_grads[count, j] = -1.0*grad_j
                    MCI_rates[count] = alphas[-1]-rate
                    
            
                
            count = count + 1
            
    return MCI_rates, MCI_grads
                
    
def MCI_four_d_plus(alpha_j, obj_j, cov_j, phantom_alphas, phantom_obj, phantom_var):
    
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
    
    x_star = np.array(solvers.qp(P,q,G,h)['x']).flatten()
    
    rate = 0.5*alpha_j*(obj_j - x_star[0:n_obj]).transpose() @ inv_cov_j @ (obj_j - x_star[0:n_obj])
    
    grad_j = 0.5*( x_star[0:n_obj]-obj_j).transpose() @ inv_cov_j @ ( x_star[0:n_obj]-obj_j)
    

    phantom_grads = np.zeros(n_obj)
    
      
    
    for o in range(len(indices)):
        ind = indices[o]
        rate =  rate + 0.5*phantom_alphas[ind]*(x_star[n_obj + o] - phantom_obj[ind]).transpose() * (phantom_var[ind]**-1) *(x_star[n_obj + o] - phantom_obj[ind])
        phantom_grads[o] = 0.5*(x_star[n_obj + o] - phantom_obj[ind]).transpose() * (phantom_var[ind]**-1) *(x_star[n_obj + o] - phantom_obj[ind])
        
    return rate, grad_j, phantom_grads
        
def find_phantoms(paretos,n_obj,num_par):
    
    
    max_y = np.inf
    
    phantoms = np.zeros([1,n_obj])
    
    v = range(n_obj)
    
    for b in v:
        #get all subsets of size b of our objective dimensions
        
        T = list(it.combinations(v,b+1))
        

        
        for dims in T:
            temp_paretos = paretos[:,dims]
            
            
            
            temp_paretos = np.unique(temp_paretos,axis=0)
            
            
            
            #only want points which are paretos in our b dimensional subspace
            temp_paretos = temp_paretos[is_pareto_efficient(temp_paretos,return_mask = False),:]
            
            #do the sweep
            phants = sweep(temp_paretos)
            phan = np.ones([len(phants),n_obj])*max_y
            
            phan[:,dims] = phants
            
            phantoms= np.append(phantoms,phan,axis=0)
    phantoms = phantoms[1:,:]
    
    return phantoms
        
            
def sweep(paretos):
    
    n_obj = len(paretos[0,:])
    
    num_par = len(paretos)
    
    if n_obj ==1:
        return paretos[[paretos[:,0].argmin()],:]
    else:
        phantoms = np.zeros([1,n_obj])
        #python indexes from 0, so the index for our last dimension is n_obj-1
        d = n_obj-1
        
        #other dimesions
        T = range(d)
        
        #sort paretos by dimension d in descending order
        paretos = paretos[(-paretos[:,d]).argsort(),:]
        
        #sweep dimension d from max to min, projecting into dimensions T
        for i in range(num_par-(n_obj-1)): #need at least n_obj-1 Paretos to get a phantom
            max_y = paretos[i,d]
            temp_paretos = paretos[i+1:num_par,:] #paretos following the current max pareto
            temp_paretos = temp_paretos[:,T] #remove dimension d/ project onto dimensions d
            temp_paretos = temp_paretos[is_pareto_efficient(temp_paretos,return_mask=False),:] #find pareto front
            phants = sweep(temp_paretos) #find phantoms from hyperplane passing through pareto i along T
            
            #of phantom candidates, include only those which are dominated by pareto i
            
            #TODO vectorize?
            for j in T:

                non_dom = phants[:,j]>paretos[i,j]
                phants = phants[non_dom,:]
                
            phan = np.ones([len(phants),n_obj])*max_y
            phan[:,T] = phants
            phantoms = np.append(phantoms,phan,axis=0)
        #remove the row we used to initialize phantoms    
        phantoms = phantoms[1:,:]
        return phantoms
            
            
            
        
        
        
def is_pareto_efficient(costs, return_mask = True):
    """
    PyMOSO's pareto finder works with dictionaries, which is great for the rest
    of the code here but not great for finding phantom paretos. We use this instead
    from stackexchange thread https://stackoverflow.com/questions/32791911/fast-calculation-of-pareto-front-in-python
    user 851699, Peter
    Thanks Peter.
    As with all code posted to stackexchange since Feb 2016, this funciton is covered under the MIT open source license
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :param return_mask: True to return a mask
    :return: An array of indices of pareto-efficient points.
        If return_mask is True, this will be an (n_points, ) boolean array
        Otherwise it will be a (n_efficient_points, ) integer array of indices.
    """
    is_efficient = np.arange(costs.shape[0])
    n_points = costs.shape[0]
    next_point_index = 0  # Next index in the is_efficient array to search for
    while next_point_index<len(costs):
        nondominated_point_mask = np.any(costs<costs[next_point_index], axis=1)
        nondominated_point_mask[next_point_index] = True
        is_efficient = is_efficient[nondominated_point_mask]  # Remove dominated points
        costs = costs[nondominated_point_mask]
        next_point_index = np.sum(nondominated_point_mask[:next_point_index])+1
    if return_mask:
        is_efficient_mask = np.zeros(n_points, dtype = bool)
        is_efficient_mask[is_efficient] = True
        return is_efficient_mask
    else:
        return is_efficient
        
    
            
            