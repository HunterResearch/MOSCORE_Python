#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 19:32:50 2019

@author: nathangeldner
"""
import numpy as np
import scipy as sp
import pymoso.chnutils as moso_utils


from utils import nearestSPD, create_allocation_problem
from base import MORS_Problem


class Random_MORS_Problem(MORS_Problem):
    """MORS_Problem subclass used for example problems.
    
    Attributes
    ----------
    n_obj : int
        number of objectives

    systems : list
        list of systems with associated x's (if applicable)

    n_systems : int
        number of systems

    true_means : list
        true perfomances of all systems

    true_covs : list
        true covariance matrices of all systems

    true_paretos_mask : list
        a mask indicating whether each system is a Pareto system or not

    true_paretos : list
        list of indicies of true Pareto systems

    n_pareto_systems : int
        number of Pareto systems

    sample_sizes : list
        sample sizes for each system

    sums : list
        sums of observed objectives

    sums_of_products : list
        sums of products of pairs of observed objectives

    sample_means : list
        sample means of objectives for each system

    sample_covs : list
        sample variance-covariance matrices of objectives for each system

    rng : MRG32k3a object
        random number generator to use for simulating replications

    rng_states : list
        states of random number generators (i.e., substream) for each system
    """
    def __init__(self):
        super().__init__()

    def g(self, x):
        """Perform a single replication at a given system.
        Obtain a noisy estimate of its objectives.

        Parameters
        ----------
        x : tuple
            tuple of values (possibly non-numerical) of inputs
            characterizing the simulatable system

        Returns
        -------
        obj : tuple
            tuple of estimates of the objectives
        """
        system_idx = self.systems.index(x)
        obj = self.rng.mvnormalvariate(mean_vec=self.obj[system_idx],
                                       cov=self.var[system_idx],
                                       factorized=False)
        return tuple(obj)


def create_random_problem_fixed_pareto(n_systems, n_obj, n_paretos, sigma=1, corr=None, center=100, radius=6, minsep=0.0001):
    """Randomly create a MORS problem with a fixed number of pareto systems.

    Notes
    -----
    Uses numpy random number generator.
    For reproducability, users may set seed prior to running.
    
    Parameters
    ----------
    n_systems : int
        number of systems to create
            
    n_obj : int
        number of objectives for the problem
            
    n_paretos : int
        number of pareto systems for the problem
            
    sigma : float or int
        variance of each objective
            
    corr : float or int
        correlation of each objective
            
    center : float or int
        coordinate (for all objectives) of the center of the sphere on which we generate objective values
            
    radius : float or int
        radius of the sphere on which we generate objective values
            
    minsep : float or int
        minimum separation between a pareto and any other point
        
    Returns
    -------
    problem : dict
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
    """    
    # Generate pareto systems.
    Paretos = {}
    for i in range(n_paretos):
        ind = 0
        # Loop until new pareto point is added.
        while ind == 0:
            y = np.random.multivariate_normal([0] * n_obj, np.identity(n_obj))  # Generate standard normal vector.
            z = y / np.linalg.norm(y)  # Normalize to put on the face of unit sphere.
            if sum(z < 0) == n_obj:  # If all values are negative (i.e., in negative orthant), continue.
                y = z * radius + np.ones(n_obj) * center
                if i == 0:  # If this is the first point, add it to the list.
                    Paretos[i] = y
                    ind = 1  # Exit loop.
                else:
                    ind2 = 0
                    # Compare to other pareto systems.
                    for j in range(i):
                        d = sum(abs(Paretos[j] - y) < minsep)  # Check whether it's too close.
                        if d > 0:
                            ind2 = 1  # If it's too close, don't use it.
                            break
                    if ind2 == 0:
                        Paretos[i] = y
                        ind = 1  # Exit loop.

    # Generate non-pareto systems.
    n_non_paretos = n_systems - n_paretos    
    NonParetos = {}
    for i in range(n_non_paretos):
        ind = 0
        while ind == 0:  # Do until new non-pareto is added.
            # Generate point in sphere with radius rad.
            X = np.random.multivariate_normal([1] * n_obj, np.identity(n_obj))
            s2 = sum(X**2)
            X = X * radius * (sp.special.gammainc(s2 / 2, 1 / 2)**(1 / 1)) / np.sqrt(s2)
            y = X + center  # Move to center.
            ind2 = 0
            ind3 = 0
            for j in range(n_paretos):  # Compare to pareto systems.
                d = sum(abs(Paretos[j] - y) < minsep)  # Check if it's close to a pareto system.
                if d > 0:
                    ind2 = 1
                    break
                d2 = sum(y - Paretos[j] > 0)  # Check that it's dominated by a pareto system.
                if d2 == n_obj:
                    ind3 += 1
            if ind2 == 0 and ind3 > 0:
                NonParetos[i + n_paretos] = y
                ind = 1  # Exit loop.
    objectives = {**Paretos, **NonParetos}
    
    if corr == None:
        p = False
        while not p:
            rho = np.random.uniform(low=-1, high=1)
            covar = rho*sigma*sigma
            cp = (np.ones(n_obj)-np.identity(n_obj))*covar + np.identity(n_obj)*sigma**2
            #check if it's positive semi-definite
            p = np.all(np.linalg.eigvals(cp)>0) and np.all(cp-cp.T==0)
    else:
        rho = corr
        covar = rho*sigma*sigma
        cp = (np.ones(n_obj)-np.identity(n_obj))*covar + np.identity(n_obj)*sigma**2
        #check if it's positive semi-definite
        p = np.all(np.linalg.eigvals(cp)>0) and np.all(cp-cp.T==0)
        if not p:
            print("input correlation value results in non-SPD covariance matrix. Finding nearest SPD matrix")
            cp = nearestSPD(cp)
    variances = {}
    for i in range(n_systems):
        variances[i] = cp
        objectives[i] = list(objectives[i])
        
    return create_allocation_problem(objectives, variances)

# class RandomSequentialProblem(Oracle):
#     """Oracle subclass used for example problems
    
#     Attributes
#     ----------
#     obj: dict of lists of float
#         mean objective values for each system, indexed by system number
        
#     var: dict of numpy arrays
#         true variance structure of each system, indexed by system number
        
#     rng: pymoso.prng.MRG32k3a object
    
#     crnflag: bool
#         indicates whether common random numbers is turned on or off
        
#     simpar: int
#         number of parallel processes to be produced when taking simulation replications
        
#     num_obj: number of objectives
    
#     n_systems: number of systems
        
#     """
#     def __init__(self, obj, var, rng, crnflag=False, simpar=1):
#         self.num_obj = len(obj[0])
#         self.n_systems = len(obj)
#         self.obj = obj
#         self.var = var
#         super().__init__(self.n_systems, rng, crnflag=crnflag, simpar = simpar)
        
#     def mvnGen(self, mu, var, rng):
#         """produces draws from multivariate normal using self.rng
        
#         Arguments
#         ---------
#         mu: list of float
#             mean vector
            
#         var: numpy array
#             covariance matrix
            
#         """
#         Z = np.array([rng.gauss(mu[i],var[i,i]**0.5) for i in range(len(mu))])
#         c = np.linalg.cholesky(var)
#         X = c @ Z + mu
#         return X
        
#     def g(self, x, rng):
#         mu = np.array(self.obj[x])
#         var = self.var[x]
#         return tuple(self.mvnGen(mu,var, rng))

# def rand_problem_fixed(n_systems, n_obj, n_paretos, sigma = 1, corr = None, center = 100, radius = 6, minsep = 0.0001):
#     """randomly creates a problem to test allocation algorithms with a fixed number of pareto systems
#     note: uses numpy's random number generator. For reproducability, users may set seed prior to running.
    
#     Arguments
#     ---------
#         n_systems : int
#                 number of systems to create
                
#         n_obj: int
#                 number of objectives for the problem
                
#         n_paretos: int
#                 number of pareto systems for the problem
                
#         sigma: float or int
#                 variance of each objective
                
#         corr: float or int
#                 correlation of each objective
                
#         center: float or int
#                 coordinate (for all objectives) of the center of the sphere on which we generate objective values
                
#         radius: float or int
#                 radius of the sphere on which we generate objective values
                
#         minsep: float or int
#                 minimum separation between a pareto and any other point
        
                
#     Returns
#     -------
#         problem: dict
                
#             problem['obj'] is a dictionary of numpy arrays, indexed by system number,
#                 each of which corresponds to the objective values of a system
#             problem['var'] is a dictionary of 2d numpy arrays, indexed by system number,
#                 each of which corresponds to the covariance matrix of a system
#             problem['inv_var'] is a dictionary of 2d numpy, indexed by system number,
#                 each of which corresponds to the inverse covariance matrix of a system
#             problem['pareto_indices'] is a list of pareto systems ordered by the first objective
#             problem['non_pareto_indices'] is a list of non-pareto systems ordered by the first objective"""
        
#     #generate pareto systems
    
#     Y = {}
    
#     for i in range(n_paretos):
#         ind = 0
#         #loop until pareto point is added
#         while ind==0:
#             y = np.random.multivariate_normal([0]*n_obj, np.identity(n_obj)) #generate standard normal vector
#             z = y/np.linalg.norm(y) #normalize to put on sphere face
#             if sum(z<0) == n_obj: #if all values are negative, continue
#                 y = z*radius + np.ones(n_obj)*center
#                 if i == 0: #if this is the first point, add it to the list
#                     Y[i] = y
#                     ind = 1 #exit loop
#                 else:
#                     ind2 = 0
#                     #compare to other paretos
#                     for j in range(i):
#                         d = sum(abs(Y[j] - y)<minsep) #check whether it's too close
                        
#                         if d>0:
#                             ind2 = 1 #if it's too close, don't use it
#                             break
#                     if ind2 ==0:
#                         Y[i] = y
#                         ind = 1 #exit loop
                        
#     n_non_paretos = n_systems - n_paretos
    
#     #generate non-paertos
#     NP = {}
#     for i in range(n_non_paretos):
#         ind = 0
#         while ind == 0: #do until new non-pareto is added
#             #generate point in sphere with radius rad
#             X = np.random.multivariate_normal([1]*n_obj, np.identity(n_obj))
#             s2 = sum(X**2)
#             X = X * radius* (sp.special.gammainc(s2/2, 1/2)**(1/1))/np.sqrt(s2)
#             y = X + center # move to center
#             ind2 = 0
#             ind3 = 0
#             for j in range(n_paretos):#compare to paretos
#                 d = sum(abs(Y[j] - y) < minsep) #check if its close to a pareto
#                 if d>0:
#                     ind2 = 1
#                     break
#                 d2 = sum(y-Y[j]>0) #check that it is dominated by pareto
#                 if d2==n_obj:
#                     ind3+= 1
#             if ind2==0 and ind3>0:
#                 NP[i+n_paretos] = y
#                 ind = 1 #exit loop
#     objectives = {**Y, **NP}
    
#     if corr == None:
#         p = False
#         while not p:
#             rho = np.random.uniform(low=-1, high=1)
#             covar = rho*sigma*sigma
#             cp = (np.ones(n_obj)-np.identity(n_obj))*covar + np.identity(n_obj)*sigma**2
#             #check if it's positive semi-definite
#             p = np.all(np.linalg.eigvals(cp)>0) and np.all(cp-cp.T==0)
#     else:
#         rho = corr
#         covar = rho*sigma*sigma
#         cp = (np.ones(n_obj)-np.identity(n_obj))*covar + np.identity(n_obj)*sigma**2
#         #check if it's positive semi-definite
#         p = np.all(np.linalg.eigvals(cp)>0) and np.all(cp-cp.T==0)
#         if not p:
#             print("input correlation value results in non-SPD covariance matrix. Finding nearest SPD matrix")
#             cp = nearestSPD(cp)
#     variances = {}
#     for i in range(n_systems):
#         variances[i] = cp
#         objectives[i] = list(objectives[i])
        
#     return create_allocation_problem(objectives, variances)
    
    
                
def rand_problem(n_systems, n_obj, sigma = 1, corr = None, center = 100, radius = 6, minsep = 0.0001):
    """randomly creates a problem to test allocation algorithms with random number of pareto systems
    note: uses numpy's random number generator. For reproducability, users may set seed prior to running.
    
    Arguments
    ---------
        n_systems : int
                number of systems to create
                
        n_obj: int
                number of objectives for the problem
                
        sigma: float or int
                variance of each objective
                
        corr: float or int
                correlation of each objective
                
        center: float or int
                coordinate (for all objectives) of the center of the sphere on which we generate objective values
                
        radius: float or int
                radius of the sphere on which we generate objective values
                
        minsep: float or int
                minimum separation between a pareto and any other point
        
                
    Returns
    -------
        problem: dict
                
            problem['obj'] is a dictionary of numpy arrays, indexed by system number,
                each of which corresponds to the objective values of a system
            problem['var'] is a dictionary of 2d numpy arrays, indexed by system number,
                each of which corresponds to the covariance matrix of a system
            problem['inv_var'] is a dictionary of 2d numpy, indexed by system number,
                each of which corresponds to the inverse covariance matrix of a system
            problem['pareto_indices'] is a list of pareto systems ordered by the first objective
            problem['non_pareto_indices'] is a list of pareto systems ordered by the first objective
        """
        
    X = {}
    for i in range(n_systems):
        #generate point in sphere with radius rad
        X[i] = np.random.multivariate_normal([1]*n_obj, np.identity(n_obj))
        s2 = sum(X[i]**2)
        X[i] = X[i] * radius* (sp.special.gammainc(s2/2, n_obj/2)**(1/n_obj))/np.sqrt(s2)
        X[i] = list(X[i] + center) # move to center
    if minsep > 0:
        #find paretos
        
        paretos = list(moso_utils.get_nondom(X))
        non_paretos = [point for point in range(n_systems) if point not in paretos]
        
        #a point will be designated bad if its too close to a pareto
        bads = []
        
        for par1_ind in range(len(paretos)):
            for par2_ind in range(par1_ind):
                par1 = paretos[par1_ind]
                par2 = paretos[par2_ind]
            #no need to make a comparison if one of the points is bad
                if par2 not in bads and par1 not in bads:
                    # a pareto is bad if it's within minsep of a non-bad pareto along any objective
                    d = sum([abs(X[par1][obj] - X[par2][obj] ) < minsep for obj in range(n_obj)])
                    if d > 0:
                        bads.append(par2)
                    
        #remove duplicates                
        bads = list(set(bads))
        #remove bads from paretos                
        paretos = [par for par in paretos if par not in bads ]
        
        
        for par in paretos:
            for nonpar in non_paretos:
                
                d = sum([abs(X[nonpar][obj] - X[par][obj] ) < minsep for obj in range(n_obj)])
                if d> 0:
                    bads.append(nonpar)
        
        for point in bads:
            ind = 0
            while ind ==0:
                X[point] = np.random.multivariate_normal([1]*n_obj, np.identity(n_obj))
                s2 = sum(X[point]**2)
                X[point] = X[point] * radius* (sp.special.gammainc(s2/2, n_obj/2)**(1/n_obj))/np.sqrt(s2)
                X[point] = list(X[point] + center) # move to center
                ind2 = 0
                ind3 = 0
                for par in paretos: #compare to each pareto
                    d = sum([abs(X[point][obj] - X[par][obj] ) < minsep for obj in range(n_obj)]) #check minimum separation
                    d2 = sum([X[point][obj] - X[par][obj] > 0 for obj in range(n_obj)])#make sure new point is not pareto
                    if d>0:
                        ind2 = ind2+1
                    if d2 ==n_obj:
                        ind3 = ind3+1
                if ind2==0 and ind3>0:
                    ind = 1 #exit loop if we're outside of minsep of all paretos and dominated by any pareto
                        
                    
           
            
    objectives = X
    
    if corr == None:
        p = False
        while not p:
            rho = np.random.uniform(low=-1, high=1)
            covar = rho*sigma*sigma
            cp = (np.ones(n_obj)-np.identity(n_obj))*covar + np.identity(n_obj)*sigma**2
            #check if it's positive semi-definite
            p = np.all(np.linalg.eigvals(cp)>0) and np.all(cp-cp.T==0)
    else:
        rho = corr
        covar = rho*sigma*sigma
        cp = (np.ones(n_obj)-np.identity(n_obj))*covar + np.identity(n_obj)*sigma**2
        #check if it's positive semi-definite
        p = np.all(np.linalg.eigvals(cp)>0) and np.all(cp-cp.T==0)
        if not p:
            print("input correlation value results in non-SPD covariance matrix. Finding nearest PSD matrix")
            cp = nearestSPD(cp)
    variances = {}
    for i in range(n_systems):
        variances[i] = cp
        objectives[i] = list(objectives[i])
        
    return create_allocation_problem(objectives, variances)
            
def create_mocba_problem(covtype):
    """generates the MOCBA 25 example problem
    
    Arguments
    ---------
    covtype: str
        "ind" sets objectives to be independent
        "pos" sets all objectives to have parwise correlation of 0.4
        "neg" sets all objectives to have pairwise correlation of -0.4
        
    Returns
    -------
        problem: dict
                
            problem['obj'] is a dictionary of numpy arrays, indexed by system number,
                each of which corresponds to the objective values of a system
            problem['var'] is a dictionary of 2d numpy arrays, indexed by system number,
                each of which corresponds to the covariance matrix of a system
            problem['inv_var'] is a dictionary of 2d numpy, indexed by system number,
                each of which corresponds to the inverse covariance matrix of a system
            problem['pareto_indices'] is a list of pareto systems ordered by the first objective
            problem['non_pareto_indices'] is a list of pareto systems ordered by the first objective
    """
    n_obj = 3
    obj = {0: [8, 36, 60], 1: [12, 32, 52], 2: [14, 38, 54], 3: [16, 46, 48], 4: [4, 42, 56],\
           5: [18,40,62], 6: [10,44,58], 7: [20,34,64], 8: [22,28,68], 9: [24,40,62],\
           10: [26,38,64], 11: [28,40,66], 12: [30,42,62], 13: [32,44,64], 14: [26,40,66],\
           15: [28,42,64], 16: [32,38,66], 17: [30,40,62], 18: [34,42,64], 19: [26,44,60],\
           20: [28,38,66], 21: [32,40,62], 22: [30,46,64], 23: [32,44,66], 24: [30,40,64]}
    covs = {}
    
    if covtype == "ind":
        cov = np.identity(n_obj) * 8
    elif covtype == "pos":
        cov = np.array([[64, 0.4*8*8, 0.4*8*8], [0.4*8*8,64,0.4*8*8], [0.4*8*8,0.4*8*8,64]])
    elif covtype == "neg":
        cov = np.array([[64, -0.4*8*8, -0.4*8*8], [-0.4*8*8,64,-0.4*8*8], [-0.4*8*8,-0.4*8*8,64]])
    else:
        raise ValueError("Invalid covtype. Valid choices are ind, pos, and neg.")
    
    for key in obj.keys():
        covs[key] = cov
    return create_allocation_problem(obj, covs)


def create_test_problem_2():
    """generates test problem 2 from **insert citation**
    
    Returns
    -------
    problem: dict
                
            problem['obj'] is a dictionary of numpy arrays, indexed by system number,
                each of which corresponds to the objective values of a system
            problem['var'] is a dictionary of 2d numpy arrays, indexed by system number,
                each of which corresponds to the covariance matrix of a system
            problem['inv_var'] is a dictionary of 2d numpy, indexed by system number,
                each of which corresponds to the inverse covariance matrix of a system
            problem['pareto_indices'] is a list of pareto systems ordered by the first objective
            problem['non_pareto_indices'] is a list of pareto systems ordered by the first objective
    """
    obj_array = np.genfromtxt('TP2_Objs.csv', delimiter = ',')
    obj = {}
    for i in range(len(obj_array[:,0])):
        obj[i] = list(obj_array[i,:])
        
    covs = {}
    for key in obj.keys():
        covs[key] = np.identity(len(obj[key]))
    
    return create_allocation_problem(obj,covs)
        
        
    

def allocation_to_sequential(allocation_problem, rng, crnflag = False, simpar = 1):
    """creates an oracle object that produces a multivariate normal objective value
    with the "true" mean and variance structure provided
    
    Arguments
    ---------
    
    allocation_problem: dict
            allocation_problem['obj'] is a dictionary of numpy arrays, indexed by system number,
                each of which corresponds to the objective values of a system
            allocation_problem['var'] is a dictionary of 2d numpy arrays, indexed by system number,
                each of which corresponds to the covariance matrix of a system
            allocation_problem['inv_var'] is a dictionary of 2d numpy, indexed by system number,
                each of which corresponds to the inverse covariance matrix of a system
            allocation_problem['pareto_indices'] is a list of pareto systems ordered by the first objective
            allocation_problem['non_pareto_indices'] is a list of pareto systems ordered by the first objective
            
    rng: a pymoso.prng.MRG32k3a object
    
    crnflag: bool
        if true, the oracle will utilize common random numbers
    simpar: int
        the number of parallel processes used in taking simulation replications
        
    Returns
    -------
    
    allocation_problem: dict
        identical to input
        
    orc: MORS RandomSequential Problem object
        inherits from Oracle
    """
    
    orc = RandomSequentialProblem(allocation_problem["obj"], allocation_problem["var"], rng, crnflag = crnflag, simpar = simpar)
    
    
    return allocation_problem, orc




  
  
  
  
  
  
  
  
  
  
  

