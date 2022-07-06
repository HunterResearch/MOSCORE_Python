#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Summary
-------
Provide class definitions for MORS Problems and Solvers and pairings.

Listing
-------
MORS_Problem : class
MORS_Solver : class
record_metrics : function
MORS_Tester : class
"""

import numpy as np
# import time
# import copy
# import multiprocessing as mp

# from pymoso.prng.mrg32k3a import MRG32k3a, get_next_prnstream, bsm, mrg32k3a, jump_substream
import pymoso.chnutils as utils
from mrg32k3a import MRG32k3a

from allocate import allocate
from utils import create_allocation_problem, is_pareto_efficient  # calc_phantom_rate, _mp_objmethod


class MORS_Problem(object):
    """Class for multi-objective ranking-and-selection problem.

    Attributes
    ---------
    n_obj: int
        number of objectives
    systems: list
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
        self.n_systems = len(self.systems)
        # If performances are known, determine which systems are Pareto efficient.
        if self.true_means is not None:
            self.true_paretos_mask = is_pareto_efficient(costs=np.array(self.true_means), return_mask=True)
            self.true_paretos = [idx for idx in range(self.n_systems) if self.true_paretos_mask[idx]]
            self.n_pareto_systems = len(self.true_paretos)
        # Initialize sample statistics.
        self.reset_statistics()

    def attach_rng(self, rng):
        """Attach random number generator to MORS_Problem object.

        Parameters
        ----------
        rng : MRG32k3a object
            random number generator to use for simulating replications

        Returns
        -------
        None
        """
        self.rng = rng

    def reset_statistics(self):
        """Reset sample statistics for all systems.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.sample_sizes = [0 for _ in range(self.n_systems)]
        self.sums = [[0 for _ in range(self.n_obj)] for _ in range(self.n_systems)]
        self.sums_of_products = [[[0 for _ in range(self.n_obj)] for _ in range(self.n_obj)] for _ in range(self.n_systems)]
        self.sample_means = [[None for _ in range(self.n_obj)] for _ in range(self.n_systems)]
        self.sample_covs = [[[None for _ in range(self.n_obj)] for _ in range(self.n_obj)] for _ in range(self.n_systems)]

    def update_statistics(self, system_indices, objs):
        """Update statistics for systems given a new batch of simulation outputs.

        Parameters
        ----------
        system_indices : list
            list of indices of systems to simulate (allows repetition)
        objs : list
            list of estimates of objectives returned by reach replication
        """
        if len(system_indices) != len(objs):
            print("Length of results must equal length of list of simulated systems.")
            return
        for idx in range(len(system_indices)):
            system_idx = system_indices[idx]
            # Increase sample size.
            self.sample_sizes[system_idx] += 1
            # Add outputs to running sums and recompute sample means.
            for obj_idx in range(self.n_obj):
                self.sums[system_idx][obj_idx] += objs[idx][obj_idx]
                self.sample_means[system_idx][obj_idx] = self.sums[system_idx][obj_idx] / self.sample_sizes[system_idx]
            # Add outputs to running sums of products and recompute sample variance-covariance matrix.
            for obj_idx1 in range(self.n_obj):
                for obj_idx2 in range(self.n_obj):
                    self.sums_of_products[system_idx][obj_idx1][obj_idx2] += objs[idx][obj_idx1] * objs[idx][obj_idx2]
                    if self.sample_sizes[system_idx] > 1:
                        # From https://www.randomservices.org/random/sample/Covariance.html,
                        #   sample cov (x, y) = n / (n-1) * [sample mean (x*y) - sample mean (x) * sample mean (y)]
                        self.sample_covs[system_idx][obj_idx1][obj_idx2] = self.sample_sizes[system_idx] / (self.sample_sizes[system_idx] - 1) * \
                            (self.sums_of_products[system_idx][obj_idx1][obj_idx2] / self.sample_sizes[system_idx] - self.sample_means[system_idx][obj_idx1] * self.sample_means[system_idx][obj_idx2])
        # TO DO: Make more efficient by only recomputing stats outside the for loop.

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
        raise NotImplementedError

    def bump(self, system_indices):
        """Obtain replications from a list of systems.

        Parameters
        ----------
        system_indices : list
            list of indices of systems to simulate (allows repetition)

        Returns
        -------
        objs : list
            list of estimates of objectives returned by reach replication
        """
        objs = []
        for system_idx in system_indices:
            # Re-seed random number generator.
            self.rng.seed(self.rng_states[system_idx])
            # Simulate a replication and record outputs.
            objs.append(self.g(x=self.systems[system_idx]))
            # Advance rng to start of next subsubstream and record new state.
            self.rng.advance_subsubstream()
            self.rng_states[system_idx] = self.rng._current_state
        return objs


class MORS_Solver(object):
    """Class for multi-objective ranking-and-selection solver.

    Attributes
    ----------
    rng : mrg32k3a.MRG32k3a object
        random number generator to use for allocating samples across systems
    budget : int
        total budgeted number of simulation replications.
    n0 : int
        common initial sample size for each system. Necessary to make an initial estimate
        of the objective values and variance-covariance matrix. Must be greater
        than or equal to the number of objectives plus 1 to guarantee positive-definite
        sample variance-covariance matrix
    delta : int
        incremental number of simulation replications taken before re-optimizing the allocation
    allocation_rule : str
        chosen allocation method. Options are "iSCORE", "SCORE", "Phantom", and "Brute Force"
    crn_across_solns : bool
        indicates whether CRN are used when sampling across systems (True) or not (False)
    """
    def __init__(self, budget, n0, delta, allocation_rule, crn_across_solns):
        self.budget = budget
        self.n0 = n0
        self.delta = delta
        self.allocation_rule = allocation_rule
        self.crn_across_solns = crn_across_solns

    def attach_rng(self, rng):
        """Attach random number generator to MORS_Solver object.

        Parameters
        ----------
        rng : MRG32k3a object
            random number generator to use for simulating replications

        Returns
        -------
        None
        """
        self.rng = rng

    def solve(self, problem):
        """Solve a given MORS problem - one macroreplication.

        Parameters
        ----------
        problem : base.MORS_Problem object
            multi-objective R&S problem to solve

        Returns
        -------
        outputs : dict
            outputs['alpha_hat'] : list of float, final simulation allocation by system
            outputs['paretos'] : list of indices of estimated pareto systems at termination
            outputs['objectives'] : dictionary, keyed by system index, of lists containing estimated objective values at termination
            outputs['variances'] : dictionary, keyed by system index, of estimated covariance matrices as numpy arrays at termination
            outputs['sample_sizes'] : list of int, final sample size for each system
        metrics : dict (optional)
            metrics['alpha_hats'] : list of lists of float, the simulation allocation selected at each step in the solver
            metrics['alpha_bars'] : list of lists of float, the portion of the simulation budget that has been allocated
                to each system at each step in the solver
            metrics['paretos'] : list of lists of int, the estimated pareto frontier at each step in the solver
            metrics['MCI_bool'] : list of Bool, indicating whether an misclassification by inclusion
                occured at each step in the solver
            metrics['MCE_bool'] : list of Bool, indicating whether an misclassification by exclusion
                occured at each step in the solver
            metrics['MC_bool'] : list of Bool, indicating whether an misclassification
                occured at each step in the solver
            metrics['percent_false_exclusion'] : list of float, the portion of true pareto systems
                that are falsely excluded at each step in the solver
            metrics['percent_false_inclusion'] : list of float, the portion of true non-pareto systems
                that are falsely included at each step in the solver
            metrics['percent_misclassification'] : list of float, the portion of systems that are
                misclassified at each step in the solver
        """
        if self.n0 < problem.n_obj + 1:
            raise ValueError("n0 has to be greater than or equal to n_obj plus 1 to guarantee positive definite\
                  sample variance-covariance matrices.")
        warm_start = None

        # Initialize summary statistics tracked over time:
        metrics = {'alpha_hats': [],
                   'alpha_bars': [],
                   'paretos': [],
                   'MCI_bool': [],
                   'MCE_bool': [],
                   'MC_bool': [],
                   'percent_false_exclusion': [],
                   'percent_false_inclusion': [],
                   'percent_misclassification': []
                   }

        # Simulate n0 replications at each system. Initial allocation is equal.
        alpha_hat = np.array([1 / problem.n_systems for _ in range(problem.n_systems)])
        system_indices = list(range(problem.n_systems)) * self.n0
        objs = problem.bump(system_indices=system_indices)
        problem.update_statistics(system_indices=system_indices, objs=objs)
        budget_expended = self.n0 * problem.n_systems

        # Record summary statistics tracked over time.
        record_metrics(metrics, problem, alpha_hat)

        # Expend rest of the budget in batches of size delta.
        while budget_expended < self.budget:

            # Get distribution for sampling allocation
            obj_vals = {idx: problem.sample_means[idx] for idx in range(problem.n_systems)}
            obj_vars = {idx: np.array(problem.sample_covs[idx]) for idx in range(problem.n_systems)}
            allocation_problem = create_allocation_problem(obj_vals=obj_vals, obj_vars=obj_vars)
            alpha_hat = allocate(self.allocation_rule, allocation_problem, warm_start=warm_start)[0]
            warm_start = alpha_hat

            # Sequentially choose systems to simulate by drawing independently from allocation distribution.
            systems_to_sample = self.rng.choices(population=range(problem.n_systems),
                                                 weights=alpha_hat,
                                                 k=self.delta
                                                 )
            # Repeated systems are possible.

            # Simulate selected systems.
            objs = problem.bump(system_indices=systems_to_sample)
            problem.update_statistics(system_indices=systems_to_sample, objs=objs)
            budget_expended = sum(problem.sample_sizes)

            # Record summary statistics tracked over time.
            record_metrics(metrics, problem, alpha_hat)

        # Record summary statistics upon termination.
        outputs = {}
        outputs['alpha_hat'] = alpha_hat
        outputs['paretos'] = metrics['paretos'][-1]  # -1 corresponds to termination.
        outputs['objectives'] = problem.sample_means
        outputs['variances'] = problem.sample_covs
        outputs['sample_sizes'] = problem.sample_sizes

        return outputs, metrics


def record_metrics(metrics, problem, alpha_hat):
    """
    Record summary statistics tracked over time.

    Parameters
    ----------
    metrics : dict
            metrics['alpha_hats']: list of lists of float, the simulation allocation selected at each step in the solver
            metrics['alpha_bars']: list of lists of float, the portion of the simulation budget that has been allocated
                to each system at each step in the solver
            metrics['paretos']: list of lists of int, the estimated pareto frontier at each step in the solver
            metrics['MCI_bool']: list of Bool, indicating whether an misclassification by inclusion
                occured at each step in the solver
            metrics['MCE_bool']: list of Bool, indicating whether an misclassification by exclusion
                occured at each step in the solver
            metrics['MC_bool']: list of Bool, indicating whether an misclassification
                occured at each step in the solver
            metrics['percent_false_exclusion']: list of float, the portion of true pareto systems
                which are falsely excluded at each step in the solver
            metrics['percent_false_inclusion']: list of float, the portion of true non-pareto systems
                which are falsely included at each step in the solver
            metrics['percent_misclassification']: list of float, the portion of systems which are
                misclassified at each step in the solver
    problem : base.MORS_Problem object
        multi-objective R&S problem to solve
    alpha_hat : list
        allocation recommended by allocation rule

    Returns
    -------
    metrics : dict
            metrics['alpha_hats'] : list of np arrays of float, the simulation allocation selected at each step in the solver
            metrics['alpha_bars'] : list of np arrays of float, the portion of the simulation budget that has been allocated
                to each system at each step in the solver
            metrics['paretos'] : list of lists of int, the estimated pareto frontier at each step in the solver
            metrics['MCI_bool'] : list of Bool, indicating whether an misclassification by inclusion
                occured at each step in the solver
            metrics['MCE_bool'] : list of Bool, indicating whether an misclassification by exclusion
                occured at each step in the solver
            metrics['MC_bool'] : list of Bool, indicating whether an misclassification
                occured at each step in the solver
            metrics['percent_false_exclusion'] : list of float, the portion of true pareto systems
                which are falsely excluded at each step in the solver
            metrics['percent_false_inclusion'] : list of float, the portion of true non-pareto systems
                which are falsely included at each step in the solver
            metrics['percent_misclassification'] : list of float, the portion of systems which are
                misclassified at each step in the solver
    """
    # Record recommended and empirical allocation proportions.
    metrics['alpha_hats'].append(alpha_hat)
    metrics['alpha_bars'].append(np.array([size / sum(problem.sample_sizes) for size in problem.sample_sizes]))

    # Record systems that look Pareto efficient.
    est_obj_vals = {idx: problem.sample_means[idx] for idx in range(problem.n_systems)}
    est_pareto = list(utils.get_nondom(est_obj_vals))
    metrics['paretos'].append(est_pareto)

    # If true objectives are known, compute error statistics.
    if problem.true_means is not None:

        # Compute and record booleans for MCI, MCE, MC.
        MCI_bool = any([est_pareto_system not in problem.true_paretos for est_pareto_system in est_pareto])
        MCE_bool = any([true_pareto_system not in est_pareto for true_pareto_system in problem.true_paretos])
        metrics['MCI_bool'].append(MCI_bool)
        metrics['MCE_bool'].append(MCE_bool)
        metrics['MC_bool'].append(MCI_bool or MCE_bool)

        # Compute and record proportions of systems misclassified in different ways.
        n_correct_select = sum([est_pareto_system in problem.true_paretos for est_pareto_system in est_pareto])
        n_false_select = sum([est_pareto_system not in problem.true_paretos for est_pareto_system in est_pareto])
        metrics['percent_false_exclusion'].append((problem.n_pareto_systems - n_correct_select) / problem.n_pareto_systems)
        metrics['percent_false_inclusion'].append(n_false_select / (problem.n_systems - problem.n_pareto_systems))
        metrics['percent_misclassification'].append(((problem.n_pareto_systems - n_correct_select) + n_false_select) / problem.n_systems)

    return metrics


class MORS_Tester(object):
    """
    A pairing of a MORS solver and problem for running experiments

    Attributes
    ----------
    solver : base.MORS_Solver object
        multi-objective RS solver
    problem : base.MORS_Problem object
        multi-objective RS problem
    n_macroreps : int
        number of macroreplications run
    all_outputs : list of dict
        list of terminal statistics from each macroreplication
    all_metrics : list of dict
        list of statistics over time from each macroreplication
    rates : dict
        rates['MCI_rate'] : list of float
            empirical MCI rate at a given point across sequential solver macroreplications
        rates['MCE_rate'] : list of float
            empirical MCE rate at a given point across sequential solver macroreplications
        rates['MC_rate'] : list of float
            empirical MC rate at a given point across sequential solver macroreplications
        rates['avg_percent_false_exclusion'] : list of float
            the average proportion of true pareto systems that are falsely excluded at each step in the solver
        rates['avg_percent_false_inclusion'] : list of float
            the average proportion of true non-pareto systems that are falsely included at each step in the solver
        rates['avg_percent_misclassification'] : list of float
            the proportion of systems that are misclassified at each step in the solver
    """
    def __init__(self, solver, problem):
        self.problem = problem
        self.solver = solver
        # Initialize tracking of statistics for each macroreplication.
        self.all_outputs = []
        self.all_metrics = []

    def setup_rng_states(self):
        """Setup rng states for each system based on whether solver uses CRN.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        if self.solver.crn_across_solns:
            # Using CRN --> all systems based on common substream
            rng_states = [self.problem.rng._current_state for _ in range(self.problem.n_systems)]
        else:
            # Not using CRN --> each system uses different substream
            rng_states = []
            for _ in range(self.problem.n_systems):
                rng_states.append(self.problem.rng._current_state)
                self.problem.rng.advance_substream()
        self.problem.rng_states = rng_states

    def run(self, n_macroreps):
        """Run n_macroreps of the solver on the problem.

        Parameters
        ----------
        n_macroreps : int
            number of macroreplications run

        Returns
        -------
        None
        """
        self.n_macroreps = n_macroreps
        # Create, initialize, and attach random number generators.
        #       Stream 0: solver rng
        #       Streams 1, 2, ...: sampling on macroreplication 1, 2, ...
        #           Substreams 0, 1, 2: sampling at system 1, 2, ...
        #               Subsubstreams 0, 1, 2: sampling replication 1, 2, ...
        solver_rng = MRG32k3a()  # Stream 0
        self.solver.attach_rng(solver_rng)
        problem_rng = MRG32k3a(s_ss_sss_index=[1, 0, 0])  # Stream 1
        self.problem.attach_rng(problem_rng)
        self.setup_rng_states()
        # Run n_macroreps of the solver on the problem and record results.
        for mrep in range(self.n_macroreps):
            print(f"Running macroreplication {mrep + 1} of {self.n_macroreps}.")
            outputs, metrics = self.solver.solve(problem=self.problem)
            self.all_outputs.append(outputs)
            self.all_metrics.append(metrics)
            # Reset sample statistics
            self.problem.reset_statistics()
            # Advance random number generators in preparation for next macroreplication.
            self.solver.rng.advance_substream()  # Not strictly necessary.
            self.problem.rng.advance_stream()
            self.setup_rng_states()
        # Aggregate metrics across macroreplications.
        self.aggregate_metrics()

    def aggregate_metrics(self):
        """Aggregate run-time statistics over macroreplications, e.g., calculate
        empirical misclassification rates.

        Arguments
        ---------
        None

        Returns
        -------
        None
        """
        # Number of budget times at which statistics are recorded.
        n_budgets = len(self.all_metrics[0]["MCI_bool"])
        # TODO: Write helper function that does the double list comprehension and takes
        # the key as an argument.
        # Calculate misclassification rates over time (aggregated over macroreplications).
        MCI_rate = [np.mean([self.all_metrics[macro_idx]["MCI_bool"][budget_idx] for macro_idx in range(self.n_macroreps)]) for budget_idx in range(n_budgets)]
        MCE_rate = [np.mean([self.all_metrics[macro_idx]["MCE_bool"][budget_idx] for macro_idx in range(self.n_macroreps)]) for budget_idx in range(n_budgets)]
        MC_rate = [np.mean([self.all_metrics[macro_idx]["MC_bool"][budget_idx] for macro_idx in range(self.n_macroreps)]) for budget_idx in range(n_budgets)]
        avg_percent_false_exclusion = [np.mean([self.all_metrics[macro_idx]["percent_false_exclusion"][budget_idx] for macro_idx in range(self.n_macroreps)]) for budget_idx in range(n_budgets)]
        avg_percent_false_inclusion = [np.mean([self.all_metrics[macro_idx]["percent_false_inclusion"][budget_idx] for macro_idx in range(self.n_macroreps)]) for budget_idx in range(n_budgets)]
        avg_percent_misclassification = [np.mean([self.all_metrics[macro_idx]["percent_misclassification"][budget_idx] for macro_idx in range(self.n_macroreps)]) for budget_idx in range(n_budgets)]
        self.rates = {'MCI_rate': MCI_rate,
                      'MCE_rate': MCE_rate,
                      'MC_rate': MC_rate,
                      'avg_percent_false_exclusion': avg_percent_false_exclusion,
                      'avg_percent_false_inclusion': avg_percent_false_inclusion,
                      'avg_percent_misclassification': avg_percent_misclassification
                      }

# START OLD CODE
# """
# Created on Sun Sep 15 18:08:02 2019

# @author: nathangeldner
# """

# def solve(problem_class, solver_class, n_0, budget, method, seed = (42, 42, 42, 42, 42, 42), delta=15, time_budget = 604800, metrics = False, \
#           pareto_true = None, phantom_rates = False,\
#               alloc_prob_true = None, crnflag = False, simpar = 1):
    
#     my_problem = problem_class(MRG32k3a(x = seed), crnflag = crnflag, simpar = simpar)
#     my_solver_prn = get_next_prnstream(my_problem.rng.get_seed(), False)
    
#     my_solver = MORS_solver(my_problem, my_solver_prn)
    
#     return my_solver.solve( n_0, budget, method, delta=delta, time_budget = time_budget,\
#                            metrics = metrics, pareto_true = pareto_true, phantom_rates = phantom_rates,\
#                            alloc_prob_true = alloc_prob_true)
    
    
# def solve2(my_problem, solver_class, n_0, budget, method, seed = (42, 42, 42, 42, 42, 42), delta=15, time_budget = 604800, metrics = False, \
#           pareto_true = None, phantom_rates = False,\
#               alloc_prob_true = None, crnflag = False, simpar = 1):
    
#     my_solver_prn = get_next_prnstream(my_problem.rng.get_seed(), False)
    
#     my_solver = MORS_solver(my_problem, my_solver_prn)
    
#     return my_solver.solve( n_0, budget, method, delta=delta, time_budget = time_budget,\
#                            metrics = metrics, pareto_true = pareto_true, phantom_rates = phantom_rates,\
#                            alloc_prob_true = alloc_prob_true)


# class Oracle(object):
#     """class for implementation of black-box simulations for use in MORS problems
    
#     Attributes
#     ----------
#     n_systems: int
#                 the number of systems defined in the oracle
    
#     crnflag: bool
#                 Indicates whether common random numbers is turned on or off. Defaults to off
    
#     rng: prng.MRG32k3a object
#                 A prng.MRG32k3a object, a pseudo-random number generator
#                     used by the oracle to simulate objective values.
    
#     random_state: tuple or list of tuples
#                 If crnflag = False, a tuple of length 2. The first item is a tuple of int, which is 
#                     32k3a seed. The second is random.Random state.
#                 If crnflag = True, a list of such tuples with length  n_systems
    
#     simpar: int
#                 the number of processes to use during simulation. Defaults to 1
                
#     num_obj: int
#                 The number of objectives returned by g
    
#     Parameters
#     ----------
#     n_systems: int
#             the number of systems which are accepted by g
        
#     rng: prng.MRG32k3a object
                
#     """
    
#     def __init__(self, n_systems, rng, crnflag=False, simpar = 1):
        
#         self.rng = rng
#         self.crnflag = crnflag
#         if crnflag == False:
#             self.random_state = rng.getstate()
#         if crnflag == True:
#             self.random_state = [rng.getstate()]*n_systems
#         self.n_systems = n_systems
#         self.simpar = simpar
        
#         super().__init__()
        
#     def nextobs(self):
#         state = self.random_state
#         self.rng.setstate(state)
#         jump_substream(self.rng)
#         self.random_state = self.rng.getstate()
        
#     def crn_nextobs(self, system):
#         state = self.random_state[system]
#         self.rng.setstate(state)
#         jump_substream(self.rng)
#         self.random_state[system] = self.rng.getstate()
    
#     def crn_setobs(self, system):
#         state = self.random_state[system]
#         self.rng.setstate(state)
        
        
   
        
        
    
#     def bump(self, x):
#         """takes in x, a list of systems to simulate (allowing for repetition), and 
#         returns simulated objective values
        
#         Parameters
#         ----------
#         x: tuple of ints
#                     tuple of systems to simulate
                    
#         m: tuple of ints
#                     tuple of number of requested replicates for each system in x
                    
#         returns
#         -------
#         objs: dictionary of lists of tuples of float
#                     objs[system] is a list with length equal to the number of repetitions of system in x
#                     containing tuples of simulated objective values
#                     """
                    
#         objs = {}
#         n_replicates = len(x)
#         for system in set(x):
#             objs[system] = []
        
        
#         if n_replicates < 1:
#             print('--* Error: Number of replications must be at least 1. ')
#             return
        
#         if n_replicates ==1:
#             if self.crnflag ==True:
#                 self.crn_setobs(x[0])
#                 obs = self.g(x[0],self.rng)
#                 objs[x[0]].append(obs)
#                 self.crn_nextobs(x[0])
                
#             else:
#                 obs = self.g(x,self.rng)
#                 objs[x[0]].append(obs)
#                 self.next_obs()
#         else:
#             if self.simpar == 1:
#                 if self.crnflag == True:
#                     for system in x:
#                         self.crn_setobs(system)
#                         obs = self.g(system,self.rng)
#                         objs[system].append(obs)
#                         self.crn_nextobs(system)
#                 else:
#                     for system in x:
#                         obs = self.g(system,self.rng)
#                         objs[system].append(obs)
#                         self.nextobs()
#             else:
#                 sim_old = self.simpar
#                 nproc = self.simpar
#                 if self.simpar > n_replicates:
#                     nproc = n_replicates
#                 replicates_per_proc = [int(n_replicates/nproc) for i in range(nproc)]
#                 for i in range(n_replicates % nproc):
#                     replicates_per_proc[i] +=1
#                 chunk_args = [[]]*nproc
#                 args_chunked = 0
#                 for i in range(nproc):
#                     chunk_args[i] = x[args_chunked:(args_chunked +replicates_per_proc[i])]
#                     args_chunked+= replicates_per_proc[i]
#                     #turn off simpar so the workers will be doing the serial version
#                 self.simpar = 1
#                 #need to send a different oracle object to each worker with the correct
#                 #random state
#                 oracle_list = []
#                 if self.crnflag == True:
#                     for i in range(nproc-1):
#                         new_oracle = copy.deepcopy(self)
#                         oracle_list.append(new_oracle)
#                         for system in chunk_args[i]:
#                             self.crn_nextobs(system)
#                     oracle_list.append(self)
#                 else:
#                     for i in range(nproc-1):
#                         new_oracle = copy.deepcopy(self)
#                         oracle_list.append(new_oracle)
#                         for system in chunk_args[i]:
#                             self.nextobs()
#                     oracle_list.append(self)
                
#                 pres = []
#                 with mp.Pool(nproc) as p:
#                     for i in range(nproc):
#                         pres.append(p.apply_async(_mp_objmethod, (oracle_list[i], 'bump', [chunk_args[i]])))
#                     for i in range(nproc):
#                         res = pres[i].get()
#                         for system in res.keys():
#                             objs[system] = objs[system] + res[system]
                            
                
#                 self.simpar = sim_old
#         return objs
                    
           


    
# class MORS_tester(object):
#     """
#     Stores data for testing MORS algorithms.
    
#     Attributes
#     ----------
#     ranorc: an oracle class, usually an implementation of MyProblem
    
#     solution: list of int
#                 a list of pareto systems
                
#     problem_struct: None or dictionary with keys 'obj', 'var', 'inv_var', 'pareto_indices', 'non_pareto_indices'
#             problem['obj'] is a dictionary of numpy arrays, indexed by system number,
#                 each of which corresponds to the objective values of a system
#             problem['var'] is a dictionary of 2d numpy arrays, indexed by system number,
#                 each of which corresponds to the covariance matrix of a system
#             problem['inv_var'] is a dictionary of 2d numpy, indexed by system number,
#                 each of which corresponds to the inverse covariance matrix of a system
#             problem['pareto_indices'] is a list of pareto systems ordered by the first objective
#             problem['non_pareto_indices'] is a list of non-pareto systems ordered by the first objective
#     """
    
#     def __init__(self, problem, solution, problem_struct = None):
        
#         self.ranorc = problem
#         self.solution = solution
#         self.problem_struct = problem_struct
        
#     def aggregate_metrics(self,solver_outputs):
#         """takes in runtime metrics produced by the macroreplications in testsolve
#         and calculates empirical misclassification rates
        
#         Arguments
#         ---------
        
#         solver_outputs: list of dictionaries
#             each dictionary must contain the following keys:
#                 MCI_bool: a list of booleans, each of which indicates whether a misclassification by inclusion
#                     event occurred at a given point in the sequential solver macroreplication
#                 MCE_bool: a list of booleans, each of which indicates whether a misclassification by exclusion
#                     event occurred at a given point in the sequential solver macroreplication
#                 MC_bool: a list of booleans, each of which indicates whether a misclassification
#                     event occurred at a given point in the sequential solver macroreplication
                    
#         Returns
#         -------
        
#         rates: dict
#             rates['MCI_rate']: list of float
#                 empirical MCI rate at a given point across sequential solver macroreplications
#             rates['MCE_rate']: list of float
#                 empirical MCE rate at a given point across sequential solver macroreplications
#             rates['MC_rate']: list of float
#                 empirical MC rate at a given point across sequential solver macroreplications
        
#         """
#         MCI_all = []
#         MCE_all = []
#         MC_all = []
#         for output in solver_outputs:
#             MCI_all.append(output['MCI_bool'])
#             MCE_all.append(output['MCE_bool'])
#             MC_all.append(output['MC_bool'])
            
#         MCI_all = np.array(MCI_all)
#         MCE_all = np.array(MCE_all)
#         MC_all = np.array(MC_all)
#         MCI_rate = [np.mean(MCI_all[:,i]) for i in range(MCI_all.shape[1])]
#         MCE_rate = [np.mean(MCE_all[:,i]) for i in range(MCE_all.shape[1])]
#         MC_rate = [np.mean(MC_all[:,i]) for i in range(MC_all.shape[1])]
        
        
#         rates = {'MCI_rate': MCI_rate, 'MCE_rate': MCE_rate, 'MC_rate': MC_rate}
#         return rates
    
# class MORS_solver(object):
#     """Solver object for sequential MORS problems
    
#     Attributes
#     ----------
    
#     orc: MOSCORE Oracle object
#         the simulation oracle to be optimized
    
#     num_calls: int
#         the number of function calls to the oracle thus far
        
#     num_obj: int
#         the number of objectives
        
#     n_systems: int
#         the number of systems accepted by the oracle
        
#     solver_prn: MRG32k3a object
#         a random number generator, defined in pymoso.prng.mrg32k3a, in the pymoso package
    
#     """
    
#     def __init__(self,orc, solver_prn):
#         self.orc = orc
#         self.num_calls = 0
#         self.num_obj = self.orc.num_obj
#         self.n_systems = self.orc.n_systems
#         self.solver_prn = solver_prn
#         super().__init__()
        
        
#     def solve(self, n_0, budget, method, delta=15, time_budget = 604800, metrics = False, pareto_true = None, phantom_rates = False,\
#               alloc_prob_true = None):
#         """function for sequential MORS solver.
        
#         Arguments
#         ---------
#         n_0: int
#             initial allocation to each system, necessary to make an initial estimate of the objective
#             values and covariance matrics. Must be greater than or equal to n_obj plus 1 to guarantee
#             positive definite covariance matrices
        
#         budget: int
#             simulation allocation budget. After running n_0 simulation replications of each system, the function
#             will take additional replications in increments of delta until the budget is exceeded
            
#         method: str
#             chosen allocation method. Options are "iSCORE", "SCORE", "Phantom", and "Brute Force"
            
#         delta: int
#             the number of simulation replications taken before re-evaluating the allocation
            
#         time_budget: int or float
#             before each new allocation evaluation, if the time budget is exceeded the solver will terminate.
            
#         metrics: bool
#             when metrics is True, the function returns a dictionary metrics_out, containing
#             runtime metrics pertaining to the performance of the solver 
            
#         pareto_true: list or tuple of int
#             if the true pareto frontier is known, it can be provided to the solver. Has no effect if
#             metrics is False. If metrics is true and pareto_true is provided, metrics_out will 
#             contain information pertaining to misclassifications during the solver run. This is
#             not recommended outside of the testsolve function
            
#         phantom_rates: bool
#             if phantom_rates is True, and alloc_prob_true is provided, metrics_out will include the phantom rate
#             calculated at each allocation
            
#         alloc_prob_true: dict
#             requires the following structure
#             alloc_prob_true['obj'] is a dictionary of numpy arrays, indexed by system number,
#                 each of which corresponds to the objective values of a system
#             alloc_prob_true['var'] is a dictionary of 2d numpy arrays, indexed by system number,
#                 each of which corresponds to the covariance matrix of a system
#             alloc_prob_true['inv_var'] is a dictionary of 2d numpy, indexed by system number,
#                 each of which corresponds to the inverse covariance matrix of a system
#             alloc_prob_true['pareto_indices'] is a list of pareto systems ordered by the first objective
#             alloc_prob_true['non_pareto_indices'] is a list of non-pareto systems ordered by the first objective
            
#         Returns
#         -------
#         outs: dict
#             outs['paretos']: list of indices of pareto systems
#             outs['objectives']: dictionary, keyed by system index, of lists containing objective values
#             outs['variances']: dictionary, keyed by system index, of covariance matrices as numpy arrays
#             outs['alpha_hat']: list of float, final simulation allocation by system
#             outs['sample_sizes']: list of int, final sample size for each system
            
#         metrics_out: dict (optional)
#             metrics_out['alpha_hats']: list of lists of float, the simulation allocation selected at each step in the solver
#             metrics_out['alpha_bars']: list of lists of float, the portion of the simulation budget that has been allocated
#                 to each system at each step in the solver
#             metrics_out['paretos']: list of lists of int, the estimated pareto frontier at each step in the solver
#             metrics_out['MCI_bool']: list of Bool, indicating whether an misclassification by inclusion
#                 occured at each step in the solver
#             metrics_out['MCE_bool']: list of Bool, indicating whether an misclassification by exclusion
#                 occured at each step in the solver
#             metrics_out['MC_bool']: list of Bool, indicating whether an misclassification
#                 occured at each step in the solver
#             metrics_out['percent_false_exclusion']: list of float, the portion of true pareto systems 
#                 which are falsely excluded at each step in the solver
                
#             metrics_out['percent_false_inclusion']: list of float, the portion of true non-pareto systems
#                 which are falsely included at each step in the solver
#             metrics_out['percent_misclassification']: list of float, the portion of systems which are
#                 misclassified at each step in the solver

#             """
        
        
#         #hit each system with n_0
#         t_0 = time.time()
        
#         if phantom_rates == True and alloc_prob_true is not None and pareto_true is None:
#             pareto_true = alloc_prob_true['pareto_indices']
        
#         if n_0 < self.num_obj + 1:
#             raise ValueError("n_0 has to be greater than or equal to n_obj plus 1 to guarantee positive definite\
#                   covariance matrices")
#         objectives = {}
#         variances = {}
#         sum_samples = {}
#         sum_samples_squared = {}
#         warm_start = None
#         n_systems = self.n_systems
#         alpha_hat = [1/n_systems]*n_systems
        
#         initial_bump_args = list(range(n_systems))*n_0
#         initial_bump = self.orc.bump(initial_bump_args)
#         sample_size = [n_0]*n_systems
        
#         for i in range(n_systems):
#             sample_objectives = np.array(initial_bump[i])
#             sum_samples[i] = np.array([[0]*self.orc.num_obj]).T
#             sum_samples_squared[i] = np.array([[0]*self.orc.num_obj]*self.orc.num_obj)
            
#             for sample in range(sample_objectives.shape[0]):
#                 sum_samples[i] = sum_samples[i] + sample_objectives[sample:sample+1,:].T
#                 sum_samples_squared[i] = sum_samples_squared[i] + sample_objectives[sample:sample+1,:].T @\
#                     sample_objectives[sample:sample+1,:]
#             objectives[i] = sum_samples[i] / sample_size[i]
#             variances[i] = sum_samples_squared[i] / sample_size[i] - objectives[i] @ objectives[i].T
#             objectives[i] = tuple(objectives[i].flatten())
            
#         if metrics == True:
#             pareto_est = list(utils.get_nondom(objectives))
            
#             metrics_out = {'alpha_hats': [alpha_hat], 'alpha_bars': [[size/sum(sample_size) for size in sample_size]], \
#                            'paretos': [pareto_est]}
#             if pareto_true is not None:
#                 MCI_bool = any([pareto_est_system not in pareto_true for pareto_est_system in pareto_est])
#                 MCE_bool = any([pareto_true_system not in pareto_est for pareto_true_system in pareto_true])
#                 MC_bool = MCI_bool or MCE_bool
#                 p = len(pareto_true)
#                 n_correct_select = sum([pareto_est_system in pareto_true for pareto_est_system in pareto_est])
#                 n_false_select = sum([pareto_est_system not in pareto_true for pareto_est_system in pareto_est])
#                 percent_false_exclusion = (p - n_correct_select)/p
#                 percent_false_inclusion = n_false_select / (n_systems - p)
#                 percent_misclassification = ((p - n_correct_select) + n_false_select)/n_systems
                
#                 metrics_out['MCI_bool'] = [MCI_bool]
#                 metrics_out['MCE_bool'] = [MCE_bool]
#                 metrics_out['MC_bool'] = [MC_bool]
#                 metrics_out['percent_false_exclusion'] = [percent_false_exclusion]
#                 metrics_out['percent_false_inclusion'] = [percent_false_inclusion]
#                 metrics_out['percent_misclassification'] = [percent_misclassification]
                
#                 if phantom_rates == True:
#                     if alloc_prob_true is None:
#                         raise ValueError("alloc_prob_true argument is required for calculation of phantom rates")
#                     else:
#                         metrics_out['phantom_rate'] = [calc_phantom_rate([size/sum(sample_size) for size in sample_size],\
#                                    alloc_prob_true)]
            
                         
#         self.num_calls = n_0 * n_systems
#         t_1 = time.time()
#         while self.num_calls <= budget and (t_1-t_0) <= time_budget:
            
#             #get allocation distribution
#             allocation_problem = create_allocation_problem(objectives,variances)
#             alpha_hat = allocate(method, allocation_problem, warm_start = warm_start)[0]
            
#             warm_start = alpha_hat
#             systems_to_sample = self.solver_prn.choices(range(n_systems), weights = alpha_hat, k=delta)
            
#             my_bump = self.orc.bump(systems_to_sample)
            
            
#             for system in set(systems_to_sample):
#                 sample_objectives = np.array(my_bump[system])
                
#                 for sample in range(sample_objectives.shape[0]):
#                     sum_samples[i] = sum_samples[i] + sample_objectives[sample:sample+1,:].T
#                     sum_samples_squared[i] = sum_samples_squared[i] + sample_objectives[sample:sample+1,:].T @\
#                         sample_objectives[sample:sample+1,:]
                        
#                 objectives[i] = sum_samples[i] / sample_size[i]
#                 variances[i] = sum_samples_squared[i] / sample_size[i] - objectives[i] @ objectives[i].T
#                 objectives[i] = tuple(objectives[i].flatten())
#             for system in systems_to_sample:
#                 sample_size[system]+=1
#                 self.num_calls +=1
            
#             if metrics == True:
#                 pareto_est = list(utils.get_nondom(objectives))
                
#                 metrics_out['alpha_hats'].append(alpha_hat)
#                 metrics_out['alpha_bars'].append([size/sum(sample_size) for size in sample_size])
#                 metrics_out['paretos'].append(pareto_est)
                
#                 if pareto_true is not None:
#                     p = len(pareto_true)
#                     n_correct_select = sum([pareto_est_system in pareto_true for pareto_est_system in pareto_est])
#                     n_false_select = sum([pareto_est_system not in pareto_true for pareto_est_system in pareto_est])
#                     percent_false_exclusion = (p - n_correct_select)/p
#                     percent_false_inclusion = n_false_select / (n_systems - p)
#                     percent_misclassification = ((p - n_correct_select) + n_false_select)/n_systems
#                     MCI_bool = any([pareto_est_system not in pareto_true for pareto_est_system in pareto_est])
#                     MCE_bool = any([pareto_true_system not in pareto_est for pareto_true_system in pareto_true])
#                     MC_bool = MCI_bool or MCE_bool
#                     metrics_out['MCI_bool'].append(MCI_bool)
#                     metrics_out['MCE_bool'].append(MCE_bool)
#                     metrics_out['MC_bool'].append(MC_bool)
#                     metrics_out['percent_misclassification'].append(percent_misclassification)
#                     metrics_out['percent_false_inclusion'].append(percent_false_inclusion)
#                     metrics_out['percent_false_exclusion'].append(percent_false_exclusion)
                    
#                     if phantom_rates == True:
#                         if alloc_prob_true is None:
#                             raise ValueError("alloc_prob_true argument is required for calculation of phantom rates")
#                         else:
#                             metrics_out['phantom_rate'] = calc_phantom_rate([size/sum(sample_size) for size in sample_size],\
#                                        alloc_prob_true)
#             t_1 = time.time()
                            
                    
                
        
        
#         outs = {}
#         outs['paretos'] = list(utils.get_nondom(objectives))
#         outs['objectives'] = objectives
#         outs['variances'] = variances
#         outs['alpha_hat'] = alpha_hat
#         outs['sample_sizes'] = sample_size
        
#         if metrics == True:
#             return outs, metrics_out
#         else:
#             return outs
