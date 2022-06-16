#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 18:09:55 2019

@author: nathangeldner
"""

from pymoso.prng.mrg32k3a import MRG32k3a, get_next_prnstream
from pymoso.chnutils import do_work
from multiprocessing.pool import Pool
import numpy as np
import pymoso.chnutils as moso_utils
#from smart_allocations import score_allocation_smart, phantom_allocation_smart, bf_allocation_smart
#from iscore_allocation import iscore_allocation


import multiprocessing.context as context


class NoDaemonProcess(context.Process):
    def _get_daemon(self):
        return False
    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)

class MyPool(Pool):
    def Process(self, *args, **kwds):
        return NoDaemonProcess(*args, **kwds)


def _mp_objmethod(instance, name, args=(), kwargs=None):
    """
    Wraps an instance method with arguments for use in multiprocessing
    functions.
    Parameters
    ----------
    instance : Oracle
    name : str
		The name of the 'instance' method to execute in a
		multiprocessing routine.
	args : tuple, optional
		Positional arguments requiredby 'instance.name'
	kwargs : dict, optional
		Keyword arguments used by 'instance.name'
	Returns
	-------
	instance.name
		A callable method.
	See also
	--------
	getattr
    #from deprecated version of pymoso by Kyle Cooper
    """

    if kwargs is None:
        kwargs = {}
    return getattr(instance, name)(*args, **kwargs)



def testsolve(tester, solver_class, n_0, budget, method, time_budget = 604800, delta=1, \
              seed = (12345, 12345, 12345, 12345, 12345, 12345),\
              macroreps = 1, proc = 1, simpar = 1, crn = False, phantom_rate = False):
    """runs multiple macroreplications of the MORS solver, returning solutions, runtime metrics, and empirical
    misclassification rates

    Arguments
    ---------
    tester: MORS_tester object
        interested users are welcome to define a class which inherits from MORS_Tester if an alternative
        metrics aggregator function is desired

    n_0: initial allocation to each system for each macroreplication, necessary to make an initial estimate
        of the objectivevalues and covariance matrics. Must be greater than or equal to n_obj plus 1 to guarantee
        positive definite covariance matrices

    budget: int
            simulation allocation budget. After running n_0 simulation replications of each system, the function
            will take additional replications in increments of delta until the budget is exceeded

    method: str
            chosen allocation method. Options are "iSCORE", "SCORE", "Phantom", and "Brute Force"

    solver_class: MORS_solver class
            Setting an alternative value for this argument is not recommended unless users wish to implement
            a different implementation of the sequential solver

    time_budget: int or float
            before each new allocation evaluation, if the time budget for a given macroreplication
            is exceeded the solver macroreplication will terminate.

    delta: int
            the number of simulation replications taken before re-evaluating the allocation
            for a given macroreplication

    seed: tuple of int
        MRG32k3a seed. Must be a tuple of six integers.

    macroreps: int
        the number of desired solver macroreplications

    proc: int
        the number of parallel processes in which to run solver macroreplications.

    simpar: int
        the number of parallel processes which each macroreplication will use to take
        simulation replications. If the number of desired macroreplications is greater than
        or equal to the number of processors available, it is recommended to leave this as 1.

    crn: bool
        if crn is True, the oracle will utilize common random numbers

    phantom_rate: bool
        if phantom_rates is True, and alloc_prob_true is provided, solver_metrics will include the phantom rate
            calculated at each allocation

    Returns
    -------

    solver_metrics: list of dictionaries returned by each solver macroreplication as metrics_out
    est_rates: dict
            rates['MCI_rate']: list of float
                empirical MCI rate at a given point across sequential solver macroreplications
            rates['MCE_rate']: list of float
                empirical MCE rate at a given point across sequential solver macroreplications
            rates['MC_rate']: list of float
                empirical MC rate at a given point across sequential solver macroreplications
    solver_outputs: list of dictionaries returned by each solver macroreplication as outs
    endseed: tuple of int
        MRG32k3a seed of independent random number stream subsequent to those used in the solver
        and simulation replications.



    """

    oracle_streams, solver_streams, endseed = get_testsolve_streams(macroreps, seed, crn)



    joblist = []
    for i in range(macroreps):

        this_oracle = tester.ranorc(oracle_streams[i], crnflag = crn, simpar = simpar)
        positional_args = (solver_class, solver_streams[i], n_0, budget, method, this_oracle)
        kwargs = {'delta': delta, 'metrics': True, 'pareto_true':tester.solution, 'time_budget': time_budget,\
                  'alloc_prob_true': tester.problem_struct}
        if phantom_rate == True:
            kwargs['alloc_prob_true'] = tester.problem_struct
            kwargs['phantom_rates'] = True
        joblist.append((positional_args,kwargs))

    outputs = par_runs(joblist, proc)

    #solver outputs contain the default outputs of the solve command for each macroreplication
    solver_outputs = []
    #solver metrics contain the data collection putputs of the solve command for each macroreplication
    solver_metrics = []

    for out in outputs:
        solver_outputs.append(out[0])
        solver_metrics.append(out[1])


    est_rates = tester.aggregate_metrics(solver_metrics)

    return solver_metrics, est_rates, solver_outputs, endseed


def isp_run(solver_class, solver_stream, n_0, budget, method, oracle, **kwargs):
    """
    runs a macroreplication of a problem using a single algorithm, used for parallelism

    Arguments
    ---------
    solver_class: a MORS solver class
    solver_stream: pymoso.prng.MRG32k3a object
    n_0: int
    budget: int or float
    method: str
    oracle: MORS oracle object
    kwargs: dict

    Returns
    -------
    outs: tuple
    output of a MORS_solver.solve call

    """
    solver = solver_class(oracle, solver_stream)
    outs = solver.solve(n_0, budget, method, **kwargs)
    return outs

def par_runs(joblst, num_proc=1):
    """
    Solve many problems in parallel.
    Parameters
    ----------
    joblist : list of tuple
        Each tuple is length 2. 'tuple[0]' is tuple of positional
        arguments, 'tuple[1]' is dict of keyword arguments.
    num_proc : int
        Number of processes to use in parallel. Default is 1.
    Returns
    -------
    runtots : dict
        Contains the results of every chnbase.MOSOSOlver.solve call
    note: this par_runs is a slightly edited version of the one that exists in PyMOSO
    """
    NUM_PROCESSES = num_proc
    rundict = []
    #print(joblst)
    with MyPool(NUM_PROCESSES) as p:
        worklist = [(isp_run, (e[0]), (e[1])) for e in joblst]
        app_rd = [p.apply_async(do_work, job) for job in worklist]
        for r in app_rd:
            myitem = r.get()
            rundict.append(myitem)
    return rundict


#keyword args to pass: alpha_epsilon, budget, method, delta=1, metrics = False


def get_testsolve_streams(macroreps, seed, crn):
    """Create random number generators for multiple independent macroreplications
    of a sequential MORS algorithm

    Parameters
    ----------
    macroreps: int
        number of macroreplications with which to test the algorithm

    seed: tuple of int
        starting seed to create the random number generators

    crn: bool
        indicates whether common random numbers is being used (determines whether caching
        will be used in generators)

    Returns
    -------
    oracle_generators: list of pymoso.prng.MRG32k3a objects

    solver_generators: list of pymoso.prng.MRG32k3a objects

    seed: tuple of int
        next independent random number seed
    """

    oracle_generators = []
    solver_generators = []

    for t in range(macroreps):
        solver_generator = get_next_prnstream(seed, False)
        seed = solver_generator.get_seed()
        solver_generators.append(solver_generator)

    for t in range(macroreps):
        oracle_generator = get_next_prnstream(seed, crn)
        seed = oracle_generator.get_seed()
        oracle_generators.append(oracle_generator)

    next_seed_generator = get_next_prnstream(seed, False)
    seed = next_seed_generator.get_seed()

    return oracle_generators, solver_generators, seed



def calc_phantom_rate(alphas, problem):
    """Calculates the phantom rate of an allocation given a MORS problem

    Arguments
    ---------

    alphas: list of float
        an initial simulation allocation which sets the starting point for determining the optimal allocation.
        Length must be equal to the number of systems.

    problem: dict
        problem must have the following structure:
            alloc_prob_true['obj'] is a dictionary of numpy arrays, indexed by system number,
                each of which corresponds to the objective values of a system
            alloc_prob_true['var'] is a dictionary of 2d numpy arrays, indexed by system number,
                each of which corresponds to the covariance matrix of a system
            alloc_prob_true['inv_var'] is a dictionary of 2d numpy, indexed by system number,
                each of which corresponds to the inverse covariance matrix of a system
            alloc_prob_true['pareto_indices'] is a list of pareto systems ordered by the first objective
            alloc_prob_true['non_pareto_indices'] is a list of non-pareto systems ordered by the first objective

    """
    from phantom_allocation import find_phantoms, MCI_phantom_rates
    from brute_force_allocation import MCE_brute_force_rates


    paretos = problem['pareto_indices']
    n_paretos = len(paretos)
    n_obj = len(problem['obj'][0])
    n_systems = len(problem['obj'])

    pareto_array  = np.zeros([n_paretos,n_obj])

    for i in range(n_paretos):
        pareto_array[i,:] = problem['obj'][problem['pareto_indices'][i]]
    #get the phantoms
    phantom_values = find_phantoms(pareto_array,n_obj,n_paretos)

    for i in range(n_obj):
        phantom_values = phantom_values[(phantom_values[:,n_obj-1-i]).argsort(kind='mergesort')]
    #phantom_values = phantom_values[(phantom_values[:,0]).argsort()]

    n_phantoms = len(phantom_values)

    #TODO: consider using something other than inf as a placeholder.
    #unfortunately, inf is a float in numpy, and arrays must be homogenous
    #and floats don't automatically cast to ints for indexing leading to an error
    #right now we're casting as ints for indexing, but that's a little gross
    #also, inf doesn't cast to intmax if you cast as int, it ends up being very negative
    phantoms = np.ones([n_phantoms,n_obj])*np.inf


    #TODO vectorize?
    for i in range(n_phantoms):
        for j in range(n_obj):
            for k in range(n_paretos):
                if pareto_array[k,j] == phantom_values[i,j]:
                    phantoms[i,j] = problem['pareto_indices'][k]

    #alphas = alphas.append(alphas,0)
    alphas = np.append(alphas,0)

    MCE_rates = MCE_brute_force_rates(alphas, problem, n_paretos,n_systems, n_obj)[0]

    MCI_rates = MCI_phantom_rates(alphas, problem, phantoms, n_paretos, n_systems, n_obj)[0]

    return min(min(-MCE_rates),min(-MCI_rates))



def nearestSPD(A):
    """ nearestSPD - the nearest (in Frobenius norm) Symmetric Positive Definite matrix to A

usage: Ahat = nearestSPD(A)

From Higham: "The nearest symmetric positive semidefinite matrix in the
Frobenius norm to an arbitrary real matrix A is shown to be (B + H)/2,
where H is the symmetric polar factor of B=(A + A')/2."

 http://www.sciencedirect.com/science/article/pii/0024379588902236
 arguments: (input)
  A - square matrix, which will be converted to the nearest Symmetric
    Positive Definite Matrix.

 Arguments: (output)
  Ahat - The matrix chosen as the nearest SPD matrix to A.
  From Susan R Hunter's MATLAB implementation"""
    Asize = np.size(A)

    if Asize[0]!= Asize[1]:
        raise ValueError("A must be a square matrix")

    B = (A + A.transpose())/2

    U, Sigma, V = np.linalg.svd(B)

    H = V @ Sigma @ V.transpose()

    Ahat = (B + H)/2

    Ahat = (Ahat + Ahat.transpose())


    #test that Ahat is in fact PD. if not, then tweak by machine epsilon

    p = False
    k = 0

    while not p:
        p = np.all(np.linalg.eigvals(Ahat)>=0) and np.all(Ahat-Ahat.T==0)
        k = k+1
        if not p:
            mineig = min(np.linalg.eig(Ahat)[0])
            Ahat = Ahat + (-1*mineig*k**2 + np.finfo(mineig).eps)*np.identity(Asize[0])
    return Ahat

def is_pareto_efficient(costs, return_mask = True):
    """
    PyMOSO's pareto finder works with dictionaries, which is great for the rest
    of the code here but not great for finding phantom paretos. We use this instead
    from stackexchange thread https://stackoverflow.com/questions/32791911/fast-calculation-of-pareto-front-in-python
    user 851699, Peter
    Thanks Peter.
    As with all code posted to stackexchange since Feb 2016, this function is covered under the MIT open source license
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




def create_allocation_problem(obj_vals,obj_vars):
    """Takes in obj_vals, a dictionary of tuples of objective values (other functionsa assume equal length)
    keyed by system number and obj_vars, a dictionary of covariance  (numpy 2d arrays) keyed by system number with number
    of rows and columns equal to the number of objectives. returns a dictionary
    with keys "obj" and "var" pointing to obj_vals and obj_var respectively, and with keys
    pareto_indices and non_pareto_indices pointing to the system numbers of the pareto and non_pareto
    systems respectively, sorted by their value on the first objective (the sortedness may not be necessary
    but is useful for debugging while comparing results to the matlab code)"""

    #TODO check for positive semidefinite?

    pareto_indices = list(moso_utils.get_nondom(obj_vals))

    #I'm not sure if this is necessary, but it was a huge help in debugging so as to keep the pareto_indices
    #in line with the indices used in the matlab code
    pareto_indices.sort(key = lambda x: obj_vals[x][0])

    non_pareto_indices = [system for system in range(len(obj_vals)) if system not in pareto_indices]
    #same here
    non_pareto_indices.sort(key = lambda x: obj_vals[x][0])

    inv_vars = {}
    obj_val = {}
    for i in range(len(obj_vals)):
        obj_val[i] = np.array(obj_vals[i])

    for i in range(len(obj_vars)):
        inv_vars[i] = np.linalg.inv(obj_vars[i])

    #for i in range(len(obj_vals)):
    #    obj_vals[i] = obj_vals[i].transpose()
    systems = {"obj": obj_val, "var": obj_vars, "inv_var": inv_vars, "pareto_indices": pareto_indices, "non_pareto_indices": non_pareto_indices}




    return systems

def calc_bf_rate(alphas, problem):
    """Calculates the brute force rate of an allocation given a MORS problem

    Arguments
    ---------

    alphas: list of float
        an initial simulation allocation which sets the starting point for determining the optimal allocation.
        Length must be equal to the number of systems.

    problem: dict
        problem must have the following structure:
            alloc_prob_true['obj'] is a dictionary of numpy arrays, indexed by system number,
                each of which corresponds to the objective values of a system
            alloc_prob_true['var'] is a dictionary of 2d numpy arrays, indexed by system number,
                each of which corresponds to the covariance matrix of a system
            alloc_prob_true['inv_var'] is a dictionary of 2d numpy, indexed by system number,
                each of which corresponds to the inverse covariance matrix of a system
            alloc_prob_true['pareto_indices'] is a list of pareto systems ordered by the first objective
            alloc_prob_true['non_pareto_indices'] is a list of non-pareto systems ordered by the first objective

    """
    from phantom_allocation import find_phantoms
    from brute_force_allocation import MCE_brute_force_rates, MCI_brute_force_rates
    import itertools as it

    paretos = problem['pareto_indices']
    n_paretos = len(paretos)
    n_obj = len(problem['obj'][0])
    n_systems = len(problem['obj'])

    pareto_array  = np.zeros([n_paretos,n_obj])

    for i in range(n_paretos):
        pareto_array[i,:] = problem['obj'][problem['pareto_indices'][i]]

    v = range(n_obj)
    #kappa here is a list of tuples. each tuple is of length num_par with elements
    #corresponding to objective indices.
    #to exclude a pareto, a non-pareto must dominate the pareto with number equal to the kappa index
    #along objective equal to the kappa value, for some kappa.
    kappa = list(it.product(v,repeat=n_paretos))

    #alphas = alphas.append(alphas,0)
    alphas = np.append(alphas,0)

    MCE_rates = MCE_brute_force_rates(alphas, problem, n_paretos, n_systems, n_obj)[0]

    MCI_rates = MCI_brute_force_rates(alphas, problem, kappa, n_paretos, n_systems, n_obj)[0]

    return min(min(-MCE_rates),min(-MCI_rates))
