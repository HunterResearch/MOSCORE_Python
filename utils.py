#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Summary
-------
Provide useful functions for the library.

Listing
-------
nearestSPD : function
is_pareto_efficient : function
create_allocation_problem : function
find_phantoms : function
sweep : function
calc_min_obj_gap : function
calc_min_paired_obj_gap : function
"""

import numpy as np
import itertools as it

# import pymoso.chnutils as moso_utils
# from pymoso.chnutils import do_work
# from pymoso.prng.mrg32k3a import MRG32k3a, get_next_prnstream

# from multiprocessing.pool import Pool
# import multiprocessing.context as context


def nearestSPD(A):
    """Find the nearest (in Frobenius norm) symmetric positive definite
    matrix to a matrix A.

    Notes
    -----
    From Higham: "The nearest symmetric positive semidefinite matrix in the
    Frobenius norm to an arbitrary real matrix A is shown to be (B + H)/2,
    where H is the symmetric polar factor of B=(A + A')/2."\n

    http://www.sciencedirect.com/science/article/pii/0024379588902236

    Parameters
    ----------
    A : matrix
        Square matrix, which will be converted to the nearest symmetric
        positive definite matrix.

    Returns
    -------
    Ahat : matrix
        The matrix chosen as the nearest SPD matrix to A.\n
        From Susan R Hunter's MATLAB implementation.
    """
    Asize = np.size(A)
    if Asize[0] != Asize[1]:
        raise ValueError("A must be a square matrix")
    B = (A + A.transpose()) / 2
    U, Sigma, V = np.linalg.svd(B)
    H = V @ Sigma @ V.transpose()
    Ahat = (B + H) / 2
    Ahat = (Ahat + Ahat.transpose())

    # Check that Ahat is in fact PD. If not, then tweak by machine epsilon.
    p = False
    k = 0
    while not p:
        p = np.all(np.linalg.eigvals(Ahat) >= 0) and np.all(Ahat - Ahat.T == 0)
        k = k + 1
        if not p:
            mineig = min(np.linalg.eig(Ahat)[0])
            Ahat = Ahat + (-1 * mineig * k**2 + np.finfo(mineig).eps) * np.identity(Asize[0])
    return Ahat


def is_pareto_efficient(costs, return_mask=True):
    """Find the pareto-efficient systems.

    Notes
    -----
    PyMOSO's pareto finder works with dictionaries, which is great for the rest
    of the code here but not great for finding phantom paretos. We use this instead:

    from stackexchange thread https://stackoverflow.com/questions/32791911/fast-calculation-of-pareto-front-in-python
    user 851699, Peter
    Thanks Peter.
    As with all code posted to stackexchange since Feb 2016, this function is covered
    under the MIT open source license.

    Parameters
    ----------
    costs: array
        Array of size (n_points, n_costs).
    return_mask: bool, default=True
        True if a mask is to be returned, otherwise False.

    Returns
    -------
    array
        An array of indices of pareto-efficient points.
        If return_mask is True, this will be an (n_points, ) boolean array.
        Otherwise it will be a (n_efficient_points, ) integer array of indices.
    """
    is_efficient = np.arange(costs.shape[0])
    n_points = costs.shape[0]
    next_point_index = 0  # Next index in the is_efficient array to search for.
    while next_point_index < len(costs):
        nondominated_point_mask = np.any(costs < costs[next_point_index], axis=1)
        nondominated_point_mask[next_point_index] = True
        is_efficient = is_efficient[nondominated_point_mask]  # Remove dominated points.
        costs = costs[nondominated_point_mask]
        next_point_index = np.sum(nondominated_point_mask[:next_point_index]) + 1
    if return_mask:
        is_efficient_mask = np.zeros(n_points, dtype=bool)
        is_efficient_mask[is_efficient] = True
        return is_efficient_mask
    else:
        return is_efficient


def find_phantoms(paretos, n_obj):
    """Find the phantom pareto set.

    Parameters
    ----------
    paretos : numpy matrix
        Matrix representing the pareto set, with a row for each pareto
        system and a column for each objective.
    n_obj : int
        Number of objectives.

    Returns
    -------
    phantoms : numpy matrix
        Matrix similar in structure to paretos characterizing the phantom
        paretos. Same number of columns but the number of rows vary.
    """
    max_y = np.inf
    phantoms = np.zeros([1, n_obj])
    v = range(n_obj)

    for b in v:
        # Get all subsets of size b of our objective dimensions.
        T = list(it.combinations(v, b + 1))
        for dims in T:
            temp_paretos = paretos[:, dims]
            temp_paretos = np.unique(temp_paretos, axis=0)

            # Only want points which are paretos in our b-dimensional subspace.
            temp_paretos = temp_paretos[is_pareto_efficient(temp_paretos, return_mask=False), :]

            # Do the sweep.
            phants = sweep(temp_paretos)
            phan = np.ones([len(phants), n_obj]) * max_y
            phan[:, dims] = phants
            phantoms = np.append(phantoms, phan, axis=0)
    phantoms = phantoms[1:, :]
    return phantoms


def sweep(paretos):
    """Find the phantom pareto set in a lower-dimensional space.

    Parameters
    ----------
    paretos : numpy matrix
        Matrix representing the pareto set, with a row for each pareto
        system and a column for each objective.
    n_obj : int
        Number of objectives.

    Returns
    -------
    phantoms : numpy matrix
        Matrix similar in structure to paretos characterizing the phantom
        paretos. Same number of columns but the number of rows vary.
    """
    n_obj = len(paretos[0, :])
    num_par = len(paretos)

    if n_obj == 1:
        return paretos[[paretos[:, 0].argmin()], :]
    else:
        phantoms = np.zeros([1, n_obj])
        # Python indexes from 0, so the index for our last dimension is n_obj - 1.
        d = n_obj - 1
        # Other dimensions.
        T = range(d)

        # Sort paretos by dimension d in descending order.
        paretos = paretos[(-paretos[:, d]).argsort(), :]

        # Sweep dimension d from max to min, projecting into dimensions T
        for i in range(num_par - (n_obj - 1)):  # Need at least n_obj - 1 paretos to get a phantom.
            max_y = paretos[i, d]
            temp_paretos = paretos[i + 1:num_par, :]  # Paretos following the current max pareto.
            temp_paretos = temp_paretos[:, T]  # Remove dimension d or project onto dimension d.
            temp_paretos = temp_paretos[is_pareto_efficient(temp_paretos, return_mask=False), :]  # Find pareto front.
            phants = sweep(temp_paretos)  # Find phantoms from hyperplane passing through pareto i along T.

            # Of phantom candidates, include only those which are dominated by pareto i.
            # TODO: Vectorize?
            for j in T:
                non_dom = phants[:, j] > paretos[i, j]
                phants = phants[non_dom, :]
            phan = np.ones([len(phants), n_obj]) * max_y
            phan[:, T] = phants
            phantoms = np.append(phantoms, phan, axis=0)
        # Remove the row we used to initialize phantoms.
        phantoms = phantoms[1:, :]
        return phantoms


def calc_min_obj_gap(alloc_problem):
    """Calculate the minimum gap between objectives of any pareto
    and any pareto/nonpareto system.

    Parameters
    ----------
    alloc_problem : base.MO_Alloc_Problem
        Details of allocation problem: objectives, variances, inverse variances, indices of Pareto/non-Pareto systems.

    Returns
    -------
    min_obj_gap : float
        Minimum gap between objectives of any pareto system and any pareto/nonpareto system.
    """
    obj_vals = np.array([alloc_problem.obj[idx] for idx in range(alloc_problem.n_systems)])
    paretos_mask = is_pareto_efficient(costs=obj_vals, return_mask=True)
    paretos = [idx for idx in range(alloc_problem.n_systems) if paretos_mask[idx]]
    non_paretos = [idx for idx in range(alloc_problem.n_systems) if idx not in paretos]
    min_pareto_pareto_obj_gap = calc_min_paired_obj_gap(obj_vals=obj_vals, group1=paretos, group2=paretos)
    min_parteo_nonpareto_obj_gap = calc_min_paired_obj_gap(obj_vals=obj_vals, group1=paretos, group2=non_paretos)
    min_obj_gap = min(min_pareto_pareto_obj_gap, min_parteo_nonpareto_obj_gap)
    return min_obj_gap


def calc_min_paired_obj_gap(obj_vals, group1, group2):
    """Calculate the minimum gap between objectives of any pair between
    two groups of systems.

    Parameters
    ----------
    obj_vals : numpy array of size (n_systems, n_obj).
        Objective function values for all systems.
    group1 : list
        List of indices of systems in first group.
    group2: list
        List of indices of systems in second group.

    Returns
    -------
    min_obj_gap : float
        Minimum gap between objectives of any pair between two groups of systems.
    """
    # Initialize objective gap.
    min_obj_gap = np.inf
    for group1_idx in group1:
        for group2_idx in group2:
            if group1_idx != group2_idx:
                # Sum absolute objective gaps over all objectives.
                min_obj_gap_pair = sum(abs(obj_vals[group1_idx] - obj_vals[group2_idx]))
                # Take a minimum over all pairs.
                if min_obj_gap_pair < min_obj_gap:
                    min_obj_gap = min_obj_gap_pair
    return min_obj_gap


# class NoDaemonProcess(context.Process):
#     def _get_daemon(self):
#         return False
#     def _set_daemon(self, value):
#         pass
#     daemon = property(_get_daemon, _set_daemon)

# class MyPool(Pool):
#     def Process(self, *args, **kwds):
#         return NoDaemonProcess(*args, **kwds)

# def _mp_objmethod(instance, name, args=(), kwargs=None):
#     """
#     Wraps an instance method with arguments for use in multiprocessing
#     functions.
#     Parameters
#     ----------
#     instance : Oracle
#     name : str
# 		The name of the 'instance' method to execute in a
# 		multiprocessing routine.
# 	args : tuple, optional
# 		Positional arguments requiredby 'instance.name'
# 	kwargs : dict, optional
# 		Keyword arguments used by 'instance.name'
# 	Returns
# 	-------
# 	instance.name
# 		A callable method.
# 	See also
# 	--------
# 	getattr
#     #from deprecated version of pymoso by Kyle Cooper
#     """

#     if kwargs is None:
#         kwargs = {}
#     return getattr(instance, name)(*args, **kwargs)

# def testsolve(tester, solver_class, n_0, budget, method, time_budget = 604800, delta=1, \
#               seed = (12345, 12345, 12345, 12345, 12345, 12345),\
#               macroreps = 1, proc = 1, simpar = 1, crn = False, phantom_rate = False):
#     """runs multiple macroreplications of the MORS solver, returning solutions, runtime metrics, and empirical
#     misclassification rates

#     Arguments
#     ---------
#     tester: MORS_tester object
#         interested users are welcome to define a class which inherits from MORS_Tester if an alternative
#         metrics aggregator function is desired

#     n_0: initial allocation to each system for each macroreplication, necessary to make an initial estimate
#         of the objectivevalues and covariance matrics. Must be greater than or equal to n_obj plus 1 to guarantee
#         positive definite covariance matrices

#     budget: int
#             simulation allocation budget. After running n_0 simulation replications of each system, the function
#             will take additional replications in increments of delta until the budget is exceeded

#     method: str
#             chosen allocation method. Options are "iSCORE", "SCORE", "Phantom", and "Brute Force"

#     solver_class: MORS_solver class
#             Setting an alternative value for this argument is not recommended unless users wish to implement
#             a different implementation of the sequential solver

#     time_budget: int or float
#             before each new allocation evaluation, if the time budget for a given macroreplication
#             is exceeded the solver macroreplication will terminate.

#     delta: int
#             the number of simulation replications taken before re-evaluating the allocation
#             for a given macroreplication

#     seed: tuple of int
#         MRG32k3a seed. Must be a tuple of six integers.

#     macroreps: int
#         the number of desired solver macroreplications

#     proc: int
#         the number of parallel processes in which to run solver macroreplications.

#     simpar: int
#         the number of parallel processes which each macroreplication will use to take
#         simulation replications. If the number of desired macroreplications is greater than
#         or equal to the number of processors available, it is recommended to leave this as 1.

#     crn: bool
#         if crn is True, the oracle will utilize common random numbers

#     phantom_rate: bool
#         if phantom_rates is True, and alloc_prob_true is provided, solver_metrics will include the phantom ratecalculated at each allocation

#     Returns
#     -------

#     solver_metrics: list of dictionaries returned by each solver macroreplication as metrics_out

#     est_rates: dict
#         ``"MCI_rate"``: list of float
#         empirical MCI rate at a given point across sequential solver macroreplications

#         ``"MCE_rate"``: list of float
#         empirical MCE rate at a given point across sequential solver macroreplications

#         ``"MC_rate"``: list of float
#         empirical MC rate at a given point across sequential solver macroreplications

#     solver_outputs: list of dictionaries returned by each solver macroreplication as outs

#     endseed: tuple of int
#         MRG32k3a seed of independent random number stream subsequent to those used in the solver
#         and simulation replications.
#     """

#     oracle_streams, solver_streams, endseed = get_testsolve_streams(macroreps, seed, crn)

#     joblist = []
#     for i in range(macroreps):

#         this_oracle = tester.ranorc(oracle_streams[i], crnflag = crn, simpar = simpar)
#         positional_args = (solver_class, solver_streams[i], n_0, budget, method, this_oracle)
#         kwargs = {'delta': delta, 'metrics': True, 'pareto_true':tester.solution, 'time_budget': time_budget,\
#                   'alloc_prob_true': tester.problem_struct}
#         if phantom_rate == True:
#             kwargs['alloc_prob_true'] = tester.problem_struct
#             kwargs['phantom_rates'] = True
#         joblist.append((positional_args,kwargs))

#     outputs = par_runs(joblist, proc)

#     #solver outputs contain the default outputs of the solve command for each macroreplication
#     solver_outputs = []
#     #solver metrics contain the data collection putputs of the solve command for each macroreplication
#     solver_metrics = []

#     for out in outputs:
#         solver_outputs.append(out[0])
#         solver_metrics.append(out[1])


#     est_rates = tester.aggregate_metrics(solver_metrics)

#     return solver_metrics, est_rates, solver_outputs, endseed


# def isp_run(solver_class, solver_stream, n_0, budget, method, oracle, **kwargs):
#     """
#     runs a macroreplication of a problem using a single algorithm, used for parallelism

#     Arguments
#     ---------
#     solver_class: a MORS solver class
#     solver_stream: pymoso.prng.MRG32k3a object
#     n_0: int
#     budget: int or float
#     method: str
#     oracle: MORS oracle object
#     kwargs: dict

#     Returns
#     -------
#     outs: tuple
#     output of a MORS_solver.solve call

#     """
#     solver = solver_class(oracle, solver_stream)
#     outs = solver.solve(n_0, budget, method, **kwargs)
#     return outs

# def par_runs(joblst, num_proc=1):
#     """
#     Solve many problems in parallel.
#     Parameters
#     ----------

#     joblist : list of tuple
#         Each tuple is length 2. 'tuple[0]' is tuple of positional\n
#         arguments, 'tuple[1]' is dict of keyword arguments.
#     num_proc : int
#         Number of processes to use in parallel. Default is 1.\

#     Returns
#     -------
#     runtots : dict
#         Contains the results of every chnbase.MOSOSOlver.solve call

#     Note
#     ----
#     note: this par_runs is a slightly edited version of the one that exists in PyMOSO
#     """
#     NUM_PROCESSES = num_proc
#     rundict = []
#     #print(joblst)
#     with MyPool(NUM_PROCESSES) as p:
#         worklist = [(isp_run, (e[0]), (e[1])) for e in joblst]
#         app_rd = [p.apply_async(do_work, job) for job in worklist]
#         for r in app_rd:
#             myitem = r.get()
#             rundict.append(myitem)
#     return rundict


# #keyword args to pass: alpha_epsilon, budget, method, delta=1, metrics = False


# def get_testsolve_streams(macroreps, seed, crn):
#     """Create random number generators for multiple independent macroreplications
#     of a sequential MORS algorithm

#     Parameters
#     ----------
#     macroreps: int
#         number of macroreplications with which to test the algorithm

#     seed: tuple of int
#         starting seed to create the random number generators

#     crn: bool
#         indicates whether common random numbers is being used (determines whether caching
#         will be used in generators)

#     Returns
#     -------
#     oracle_generators: list of pymoso.prng.MRG32k3a objects

#     solver_generators: list of pymoso.prng.MRG32k3a objects

#     seed: tuple of int
#         next independent random number seed
#     """

#     oracle_generators = []
#     solver_generators = []

#     for t in range(macroreps):
#         solver_generator = get_next_prnstream(seed, False)
#         seed = solver_generator.get_seed()
#         solver_generators.append(solver_generator)

#     for t in range(macroreps):
#         oracle_generator = get_next_prnstream(seed, crn)
#         seed = oracle_generator.get_seed()
#         oracle_generators.append(oracle_generator)

#     next_seed_generator = get_next_prnstream(seed, False)
#     seed = next_seed_generator.get_seed()

#     return oracle_generators, solver_generators, seed

"""
Created on Sun Sep 15 18:09:55 2019

@author: nathangeldner
"""
