#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 18:08:02 2019

@author: nathangeldner
"""
import numpy as np
from pymoso.prng.mrg32k3a import MRG32k3a, get_next_prnstream, bsm, mrg32k3a, jump_substream
import pymoso.chnutils as utils



from allocate import allocate
import time
from utils import calc_phantom_rate, create_allocation_problem, _mp_objmethod

import copy
import multiprocessing as mp




def solve(problem_class, solver_class, n_0, budget, method, seed = (42, 42, 42, 42, 42, 42), delta=15, time_budget = 604800, metrics = False, \
          pareto_true = None, phantom_rates = False,\
              alloc_prob_true = None, crnflag = False, simpar = 1):
    
    my_problem = problem_class(MRG32k3a(x = seed), crnflag = crnflag, simpar = simpar)
    my_solver_prn = get_next_prnstream(my_problem.rng.get_seed(), False)
    
    my_solver = MORS_solver(my_problem, my_solver_prn)
    
    return my_solver.solve( n_0, budget, method, delta=delta, time_budget = time_budget,\
                           metrics = metrics, pareto_true = pareto_true, phantom_rates = phantom_rates,\
                           alloc_prob_true = alloc_prob_true)
    
    
def solve2(my_problem, solver_class, n_0, budget, method, seed = (42, 42, 42, 42, 42, 42), delta=15, time_budget = 604800, metrics = False, \
          pareto_true = None, phantom_rates = False,\
              alloc_prob_true = None, crnflag = False, simpar = 1):
    
    my_solver_prn = get_next_prnstream(my_problem.rng.get_seed(), False)
    
    my_solver = MORS_solver(my_problem, my_solver_prn)
    
    return my_solver.solve( n_0, budget, method, delta=delta, time_budget = time_budget,\
                           metrics = metrics, pareto_true = pareto_true, phantom_rates = phantom_rates,\
                           alloc_prob_true = alloc_prob_true)


class Oracle(object):
    """class for implementation of black-box simulations for use in MORS problems
    
    Attributes
    ----------
    n_systems: int
                the number of systems defined in the oracle
    
    crnflag: bool
                Indicates whether common random numbers is turned on or off. Defaults to off
    
    rng: prng.MRG32k3a object
                A prng.MRG32k3a object, a pseudo-random number generator
                    used by the oracle to simulate objective values.
    
    random_state: tuple or list of tuples
                If crnflag = False, a tuple of length 2. The first item is a tuple of int, which is 
                    32k3a seed. The second is random.Random state.
                If crnflag = True, a list of such tuples with length  n_systems
    
    simpar: int
                the number of processes to use during simulation. Defaults to 1
                
    num_obj: int
                The number of objectives returned by g
    
    Parameters
    ----------
    n_systems: int
            the number of systems which are accepted by g
        
    rng: prng.MRG32k3a object
                
    """
    
    def __init__(self, n_systems, rng, crnflag=False, simpar = 1):
        
        self.rng = rng
        self.crnflag = crnflag
        if crnflag == False:
            self.random_state = rng.getstate()
        if crnflag == True:
            self.random_state = [rng.getstate()]*n_systems
        self.n_systems = n_systems
        self.simpar = simpar
        
        super().__init__()
        
    def nextobs(self):
        state = self.random_state
        self.rng.setstate(state)
        jump_substream(self.rng)
        self.random_state = self.rng.getstate()
        
    def crn_nextobs(self, system):
        state = self.random_state[system]
        self.rng.setstate(state)
        jump_substream(self.rng)
        self.random_state[system] = self.rng.getstate()
    
    def crn_setobs(self, system):
        state = self.random_state[system]
        self.rng.setstate(state)
        
        
   
        
        
    
    def bump(self, x):
        """takes in x, a list of systems to simulate (allowing for repetition), and 
        returns simulated objective values
        
        Parameters
        ----------
        x: tuple of ints
                    tuple of systems to simulate
                    
        m: tuple of ints
                    tuple of number of requested replicates for each system in x
                    
        returns
        -------
        objs: dictionary of lists of tuples of float
                    objs[system] is a list with length equal to the number of repetitions of system in x
                    containing tuples of simulated objective values
                    """
                    
        objs = {}
        n_replicates = len(x)
        for system in set(x):
            objs[system] = []
        
        
        if n_replicates < 1:
            print('--* Error: Number of replications must be at least 1. ')
            return
        
        if n_replicates ==1:
            if self.crnflag ==True:
                self.crn_setobs(x[0])
                obs = self.g(x[0],self.rng)
                objs[x[0]].append(obs)
                self.crn_nextobs(x[0])
                
            else:
                obs = self.g(x,self.rng)
                objs[x[0]].append(obs)
                self.next_obs()
        else:
            if self.simpar == 1:
                if self.crnflag == True:
                    for system in x:
                        self.crn_setobs(system)
                        obs = self.g(system,self.rng)
                        objs[system].append(obs)
                        self.crn_nextobs(system)
                else:
                    for system in x:
                        obs = self.g(system,self.rng)
                        objs[system].append(obs)
                        self.nextobs()
            else:
                sim_old = self.simpar
                nproc = self.simpar
                if self.simpar > n_replicates:
                    nproc = n_replicates
                replicates_per_proc = [int(n_replicates/nproc) for i in range(nproc)]
                for i in range(n_replicates % nproc):
                    replicates_per_proc[i] +=1
                chunk_args = [[]]*nproc
                args_chunked = 0
                for i in range(nproc):
                    chunk_args[i] = x[args_chunked:(args_chunked +replicates_per_proc[i])]
                    args_chunked+= replicates_per_proc[i]
                    #turn off simpar so the workers will be doing the serial version
                self.simpar = 1
                #need to send a different oracle object to each worker with the correct
                #random state
                oracle_list = []
                if self.crnflag == True:
                    for i in range(nproc-1):
                        new_oracle = copy.deepcopy(self)
                        oracle_list.append(new_oracle)
                        for system in chunk_args[i]:
                            self.crn_nextobs(system)
                    oracle_list.append(self)
                else:
                    for i in range(nproc-1):
                        new_oracle = copy.deepcopy(self)
                        oracle_list.append(new_oracle)
                        for system in chunk_args[i]:
                            self.nextobs()
                    oracle_list.append(self)
                
                pres = []
                with mp.Pool(nproc) as p:
                    for i in range(nproc):
                        pres.append(p.apply_async(_mp_objmethod, (oracle_list[i], 'bump', [chunk_args[i]])))
                    for i in range(nproc):
                        res = pres[i].get()
                        for system in res.keys():
                            objs[system] = objs[system] + res[system]
                            
                
                self.simpar = sim_old
        return objs
                    
           


    
class MORS_Tester(object):
    """
    Stores data for testing MORS algorithms.
    
    Attributes
    ----------
    ranorc: an oracle class, usually an implementation of MyProblem
    
    solution: list of int
                a list of pareto systems
                
    problem_struct: None or dictionary with keys 'obj', 'var', 'inv_var', 'pareto_indices', 'non_pareto_indices'
            problem['obj'] is a dictionary of numpy arrays, indexed by system number,
                each of which corresponds to the objective values of a system
            problem['var'] is a dictionary of 2d numpy arrays, indexed by system number,
                each of which corresponds to the covariance matrix of a system
            problem['inv_var'] is a dictionary of 2d numpy, indexed by system number,
                each of which corresponds to the inverse covariance matrix of a system
            problem['pareto_indices'] is a list of pareto systems ordered by the first objective
            problem['non_pareto_indices'] is a list of non-pareto systems ordered by the first objective
    """
    
    def __init__(self, problem, solution, problem_struct = None):
        
        self.ranorc = problem
        self.solution = solution
        self.problem_struct = problem_struct
        
    def aggregate_metrics(self,solver_outputs):
        """takes in runtime metrics produced by the macroreplications in testsolve
        and calculates empirical misclassification rates
        
        Arguments
        ---------
        
        solver_outputs: list of dictionaries
            each dictionary must contain the following keys:
                MCI_bool: a list of booleans, each of which indicates whether a misclassification by inclusion
                    event occurred at a given point in the sequential solver macroreplication
                MCE_bool: a list of booleans, each of which indicates whether a misclassification by exclusion
                    event occurred at a given point in the sequential solver macroreplication
                MC_bool: a list of booleans, each of which indicates whether a misclassification
                    event occurred at a given point in the sequential solver macroreplication
                    
        Returns
        -------
        
        rates: dict
            rates['MCI_rate']: list of float
                empirical MCI rate at a given point across sequential solver macroreplications
            rates['MCE_rate']: list of float
                empirical MCE rate at a given point across sequential solver macroreplications
            rates['MC_rate']: list of float
                empirical MC rate at a given point across sequential solver macroreplications
        
        """
        MCI_all = []
        MCE_all = []
        MC_all = []
        for output in solver_outputs:
            MCI_all.append(output['MCI_bool'])
            MCE_all.append(output['MCE_bool'])
            MC_all.append(output['MC_bool'])
            
        MCI_all = np.array(MCI_all)
        MCE_all = np.array(MCE_all)
        MC_all = np.array(MC_all)
        MCI_rate = [np.mean(MCI_all[:,i]) for i in range(MCI_all.shape[1])]
        MCE_rate = [np.mean(MCE_all[:,i]) for i in range(MCE_all.shape[1])]
        MC_rate = [np.mean(MC_all[:,i]) for i in range(MC_all.shape[1])]
        
        
        rates = {'MCI_rate': MCI_rate, 'MCE_rate': MCE_rate, 'MC_rate': MC_rate}
        return rates
    
class MORS_solver(object):
    """Solver object for sequential MORS problems
    
    Attributes
    ----------
    
    orc: MOSCORE Oracle object
        the simulation oracle to be optimized
    
    num_calls: int
        the number of function calls to the oracle thus far
        
    num_obj: int
        the number of objectives
        
    n_systems: int
        the number of systems accepted by the oracle
        
    solver_prn: MRG32k3a object
        a random number generator, defined in pymoso.prng.mrg32k3a, in the pymoso package
    
    """
    
    def __init__(self,orc, solver_prn):
        self.orc = orc
        self.num_calls = 0
        self.num_obj = self.orc.num_obj
        self.n_systems = self.orc.n_systems
        self.solver_prn = solver_prn
        super().__init__()
        
        
    def solve(self, n_0, budget, method, delta=15, time_budget = 604800, metrics = False, pareto_true = None, phantom_rates = False,\
              alloc_prob_true = None):
        """function for sequential MORS solver.
        
        Arguments
        ---------
        n_0: int
            initial allocation to each system, necessary to make an initial estimate of the objective
            values and covariance matrics. Must be greater than or equal to n_obj plus 1 to guarantee
            positive definite covariance matrices
        
        budget: int
            simulation allocation budget. After running n_0 simulation replications of each system, the function
            will take additional replications in increments of delta until the budget is exceeded
            
        method: str
            chosen allocation method. Options are "iSCORE", "SCORE", "Phantom", and "Brute Force"
            
        delta: int
            the number of simulation replications taken before re-evaluating the allocation
            
        time_budget: int or float
            before each new allocation evaluation, if the time budget is exceeded the solver will terminate.
            
        metrics: bool
            when metrics is True, the function returns a dictionary metrics_out, containing
            runtime metrics pertaining to the performance of the solver 
            
        pareto_true: list or tuple of int
            if the true pareto frontier is known, it can be provided to the solver. Has no effect if
            metrics is False. If metrics is true and pareto_true is provided, metrics_out will 
            contain information pertaining to misclassifications during the solver run. This is
            not recommended outside of the testsolve function
            
        phantom_rates: bool
            if phantom_rates is True, and alloc_prob_true is provided, metrics_out will include the phantom rate
            calculated at each allocation
            
        alloc_prob_true: dict
            requires the following structure
            alloc_prob_true['obj'] is a dictionary of numpy arrays, indexed by system number,
                each of which corresponds to the objective values of a system
            alloc_prob_true['var'] is a dictionary of 2d numpy arrays, indexed by system number,
                each of which corresponds to the covariance matrix of a system
            alloc_prob_true['inv_var'] is a dictionary of 2d numpy, indexed by system number,
                each of which corresponds to the inverse covariance matrix of a system
            alloc_prob_true['pareto_indices'] is a list of pareto systems ordered by the first objective
            alloc_prob_true['non_pareto_indices'] is a list of non-pareto systems ordered by the first objective
            
        Returns
        -------
        outs: dict
            outs['paretos']: list of indices of pareto systems
            outs['objectives']: dictionary, keyed by system index, of lists containing objective values
            outs['variances']: dictionary, keyed by system index, of covariance matrices as numpy arrays
            outs['alpha_hat']: list of float, final simulation allocation by system
            outs['sample_sizes']: list of int, final sample size for each system
            
        metrics_out: dict (optional)
            metrics_out['alpha_hats']: list of lists of float, the simulation allocation selected at each step in the solver
            metrics_out['alpha_bars']: list of lists of float, the portion of the simulation budget that has been allocated
                to each system at each step in the solver
            metrics_out['paretos']: list of lists of int, the estimated pareto frontier at each step in the solver
            metrics_out['MCI_bool']: list of Bool, indicating whether an misclassification by inclusion
                occured at each step in the solver
            metrics_out['MCE_bool']: list of Bool, indicating whether an misclassification by exclusion
                occured at each step in the solver
            metrics_out['MC_bool']: list of Bool, indicating whether an misclassification
                occured at each step in the solver
            metrics_out['percent_false_exclusion']: list of float, the portion of true pareto systems 
                which are falsely excluded at each step in the solver
                
            metrics_out['percent_false_inclusion']: list of float, the portion of true non-pareto systems
                which are falsely included at each step in the solver
            metrics_out['percent_misclassification']: list of float, the portion of systems which are
                misclassified at each step in the solver

            """
        
        
        #hit each system with n_0
        t_0 = time.time()
        
        if phantom_rates == True and alloc_prob_true is not None and pareto_true is None:
            pareto_true = alloc_prob_true['pareto_indices']
        
        if n_0 < self.num_obj + 1:
            raise ValueError("n_0 has to be greater than or equal to n_obj plus 1 to guarantee positive definite\
                  covariance matrices")
        objectives = {}
        variances = {}
        sum_samples = {}
        sum_samples_squared = {}
        warm_start = None
        n_systems = self.n_systems
        alpha_hat = [1/n_systems]*n_systems
        
        initial_bump_args = list(range(n_systems))*n_0
        initial_bump = self.orc.bump(initial_bump_args)
        sample_size = [n_0]*n_systems
        
        for i in range(n_systems):
            sample_objectives = np.array(initial_bump[i])
            sum_samples[i] = np.array([[0]*self.orc.num_obj]).T
            sum_samples_squared[i] = np.array([[0]*self.orc.num_obj]*self.orc.num_obj)
            
            for sample in range(sample_objectives.shape[0]):
                sum_samples[i] = sum_samples[i] + sample_objectives[sample:sample+1,:].T
                sum_samples_squared[i] = sum_samples_squared[i] + sample_objectives[sample:sample+1,:].T @\
                    sample_objectives[sample:sample+1,:]
            objectives[i] = sum_samples[i] / sample_size[i]
            variances[i] = sum_samples_squared[i] / sample_size[i] - objectives[i] @ objectives[i].T
            objectives[i] = tuple(objectives[i].flatten())
            
        if metrics == True:
            pareto_est = list(utils.get_nondom(objectives))
            
            metrics_out = {'alpha_hats': [alpha_hat], 'alpha_bars': [[size/sum(sample_size) for size in sample_size]], \
                           'paretos': [pareto_est]}
            if pareto_true is not None:
                MCI_bool = any([pareto_est_system not in pareto_true for pareto_est_system in pareto_est])
                MCE_bool = any([pareto_true_system not in pareto_est for pareto_true_system in pareto_true])
                MC_bool = MCI_bool or MCE_bool
                p = len(pareto_true)
                n_correct_select = sum([pareto_est_system in pareto_true for pareto_est_system in pareto_est])
                n_false_select = sum([pareto_est_system not in pareto_true for pareto_est_system in pareto_est])
                percent_false_exclusion = (p - n_correct_select)/p
                percent_false_inclusion = n_false_select / (n_systems - p)
                percent_misclassification = ((p - n_correct_select) + n_false_select)/n_systems
                
                metrics_out['MCI_bool'] = [MCI_bool]
                metrics_out['MCE_bool'] = [MCE_bool]
                metrics_out['MC_bool'] = [MC_bool]
                metrics_out['percent_false_exclusion'] = [percent_false_exclusion]
                metrics_out['percent_false_inclusion'] = [percent_false_inclusion]
                metrics_out['percent_misclassification'] = [percent_misclassification]
                
                if phantom_rates == True:
                    if alloc_prob_true is None:
                        raise ValueError("alloc_prob_true argument is required for calculation of phantom rates")
                    else:
                        metrics_out['phantom_rate'] = [calc_phantom_rate([size/sum(sample_size) for size in sample_size],\
                                   alloc_prob_true)]
            
                         
        self.num_calls = n_0 * n_systems
        t_1 = time.time()
        while self.num_calls <= budget and (t_1-t_0) <= time_budget:
            
            #get allocation distribution
            allocation_problem = create_allocation_problem(objectives,variances)
            alpha_hat = allocate(method, allocation_problem, warm_start = warm_start)[0]
            
            warm_start = alpha_hat
            systems_to_sample = self.solver_prn.choices(range(n_systems), weights = alpha_hat, k=delta)
            
            my_bump = self.orc.bump(systems_to_sample)
            
            
            for system in set(systems_to_sample):
                sample_objectives = np.array(my_bump[system])
                
                for sample in range(sample_objectives.shape[0]):
                    sum_samples[i] = sum_samples[i] + sample_objectives[sample:sample+1,:].T
                    sum_samples_squared[i] = sum_samples_squared[i] + sample_objectives[sample:sample+1,:].T @\
                        sample_objectives[sample:sample+1,:]
                        
                objectives[i] = sum_samples[i] / sample_size[i]
                variances[i] = sum_samples_squared[i] / sample_size[i] - objectives[i] @ objectives[i].T
                objectives[i] = tuple(objectives[i].flatten())
            for system in systems_to_sample:
                sample_size[system]+=1
                self.num_calls +=1
            
            if metrics == True:
                pareto_est = list(utils.get_nondom(objectives))
                
                metrics_out['alpha_hats'].append(alpha_hat)
                metrics_out['alpha_bars'].append([size/sum(sample_size) for size in sample_size])
                metrics_out['paretos'].append(pareto_est)
                
                if pareto_true is not None:
                    p = len(pareto_true)
                    n_correct_select = sum([pareto_est_system in pareto_true for pareto_est_system in pareto_est])
                    n_false_select = sum([pareto_est_system not in pareto_true for pareto_est_system in pareto_est])
                    percent_false_exclusion = (p - n_correct_select)/p
                    percent_false_inclusion = n_false_select / (n_systems - p)
                    percent_misclassification = ((p - n_correct_select) + n_false_select)/n_systems
                    MCI_bool = any([pareto_est_system not in pareto_true for pareto_est_system in pareto_est])
                    MCE_bool = any([pareto_true_system not in pareto_est for pareto_true_system in pareto_true])
                    MC_bool = MCI_bool or MCE_bool
                    metrics_out['MCI_bool'].append(MCI_bool)
                    metrics_out['MCE_bool'].append(MCE_bool)
                    metrics_out['MC_bool'].append(MC_bool)
                    metrics_out['percent_misclassification'].append(percent_misclassification)
                    metrics_out['percent_false_inclusion'].append(percent_false_inclusion)
                    metrics_out['percent_false_exclusion'].append(percent_false_exclusion)
                    
                    if phantom_rates == True:
                        if alloc_prob_true is None:
                            raise ValueError("alloc_prob_true argument is required for calculation of phantom rates")
                        else:
                            metrics_out['phantom_rate'] = calc_phantom_rate([size/sum(sample_size) for size in sample_size],\
                                       alloc_prob_true)
            t_1 = time.time()
                            
                    
                
        
        
        outs = {}
        outs['paretos'] = list(utils.get_nondom(objectives))
        outs['objectives'] = objectives
        outs['variances'] = variances
        outs['alpha_hat'] = alpha_hat
        outs['sample_sizes'] = sample_size
        
        if metrics == True:
            return outs, metrics_out
        else:
            return outs
        