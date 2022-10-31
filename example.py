#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Summary
-------
Example MORS problems and functions to generate ranodom
allocation problems.
"""

import numpy as np
import scipy as sp
import pymoso.chnutils as moso_utils


from utils import nearestSPD, create_allocation_problem
from base import MORS_Problem


class TestProblem(MORS_Problem):
    """Example implementation of a user-defined MORS problem."""
    def __init__(self):
        self.n_obj = 2
        self.systems = [(5, 0), (4, 1), (3, 2), (2, 3), (1, 4), (0, 5),
                        (6, 3), (5, 2), (4, 3), (3, 4), (2, 5), (1, 6),
                        (7, 1), (6, 2), (5, 3), (4, 4), (3, 5), (2, 6)]
        self.n_systems = len(self.systems)
        self.true_means = [list(self.systems[idx]) for idx in range(self.n_systems)]
        self.true_covs = [[[1, 0], [0, 1]] for _ in range(self.n_systems)]
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
        obj1 = x[0] + self.rng.normalvariate(0, 1)
        obj2 = x[1] + self.rng.normalvariate(0, 1)
        obj = (obj1, obj2)
        return obj


class TestProblem2(MORS_Problem):
    """Example implementation of a user-defined MORS problem."""
    def __init__(self):
        self.n_obj = 2
        self.systems = [(1, 0), (0, 1), (1, 1)]
        self.n_systems = len(self.systems)
        self.true_means = [list(self.systems[idx]) for idx in range(self.n_systems)]
        self.true_covs = [[[1, 0], [0, 1]] for _ in range(self.n_systems)]
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
        obj1 = x[0] + self.rng.normalvariate(0, 1)
        obj2 = x[1] + self.rng.normalvariate(0, 1)
        obj = (obj1, obj2)
        return obj


class TestProblem3(MORS_Problem):
    """Example implementation of a user-defined MORS problem."""
    def __init__(self):
        self.n_obj = 2
        self.systems = [(5.0, 0.0), (4.0, 6.0), (3.0, 2.0), (2.0, 3.0), (6.0, 4.0), (0.0, 5.0), (1.0, 1.0)]
        self.n_systems = len(self.systems)
        self.true_means = [list(self.systems[idx]) for idx in range(self.n_systems)]
        self.true_covs = [[[1.0, 0.5], [0.5, 1.0]] for _ in range(self.n_systems)]
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
        obj1 = x[0] + self.rng.normalvariate(0, 1)
        obj2 = x[1] + self.rng.normalvariate(0, 1)
        obj = (obj1, obj2)
        return obj


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


def create_fixed_pareto_random_problem(n_systems, n_obj, n_paretos, sigma=1, corr=None, center=100, radius=6, minsep=0.0001):
    """Randomly create a MORS problem with a fixed number of pareto systems.

    Notes
    -----
    Uses numpy random number generator.
    For reproducibility, users may set seed prior to running.

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
        minimum separation between a pareto and any other system

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
    paretos = {}
    for i in range(n_paretos):
        ind = 0
        # Loop until new pareto system is added.
        while ind == 0:
            # TODO: Update to use mrg32k3a RNG.
            y = np.random.multivariate_normal([0] * n_obj, np.identity(n_obj))  # Generate standard normal vector.
            u = np.random.uniform()  # Generate a Uniform[0, 1] random variate.
            z = y / np.linalg.norm(y)  # Normalize to put uniformly on the unit sphere.
            z = -1 * np.abs(z)  # Rotate to put in negative orthant. By symmetry, this is equiprobable.
            y = z * radius + np.ones(n_obj) * center  # Recenter.
            if i == 0:  # If this is the first system, add it to the list.
                paretos[i] = y
                ind = 1  # Exit loop.
            else:
                ind2 = 0
                # Compare to other pareto systems.
                for j in range(i):
                    if np.any(np.abs(paretos[j] - y) < minsep):  # Check whether it's too close.
                        ind2 = 1  # If it's too close, don't use it.
                        break
                if ind2 == 0:
                    paretos[i] = y
                    ind = 1  # Exit loop.

    # Generate non-pareto systems.
    n_non_paretos = n_systems - n_paretos
    nonparetos = {}
    for i in range(n_non_paretos):
        ind = 0
        while ind == 0:  # Do until new non-pareto is added.
            # Generate system in ball with radius rad.
            y = np.random.multivariate_normal([0] * n_obj, np.identity(n_obj))  # Generate standard normal vector.
            u = np.random.uniform()  # Generate a Uniform[0, 1] random variate.
            z = (y / np.linalg.norm(y)) * u**(1/n_obj) # Normalize and reweight by radius to put uniformly within the unit ball.
            y = z * radius + np.ones(n_obj) * center  # Recenter.
            ind2 = 0
            ind3 = 0
            for j in range(n_paretos):  # Compare to pareto systems.
                if np.any(np.abs(paretos[j] - y) < minsep):  # Check if it's close to a pareto system.
                    ind2 = 1
                    break
                if np.all(y - paretos[j] > 0):  # Check that it's dominated by a pareto system.
                    ind3 += 1
            if ind2 == 0 and ind3 > 0:
                nonparetos[i + n_paretos] = y
                ind = 1  # Exit loop.
    objectives = {**paretos, **nonparetos}

    if corr is None:
        p = False
        while not p:
            rho = np.random.uniform(low=-1, high=1)
            covar = rho * sigma * sigma
            cp = (np.ones(n_obj) - np.identity(n_obj)) * covar + np.identity(n_obj) * sigma**2
            # Check if it's positive semi-definite.
            p = np.all(np.linalg.eigvals(cp) > 0) and np.all(cp - cp.T == 0)
    else:
        rho = corr
        covar = rho * sigma * sigma
        cp = (np.ones(n_obj) - np.identity(n_obj)) * covar + np.identity(n_obj) * sigma**2
        # Check if it's positive semi-definite.
        p = np.all(np.linalg.eigvals(cp) > 0) and np.all(cp - cp.T == 0)
        if not p:
            print("Input correlation value results in non-SPD covariance matrix. Finding nearest SPD matrix.")
            cp = nearestSPD(cp)
    variances = {}
    for i in range(n_systems):
        variances[i] = cp
        objectives[i] = list(objectives[i])
    return create_allocation_problem(objectives, variances)


def create_variable_pareto_random_problem(n_systems, n_obj, sigma=1, corr=None, center=100, radius=6, minsep=0.0001):
    """Randomly create a MORS problem with a variable number of pareto systems.

    Notes
    -----
    Uses numpy random number generator.
    For reproducibility, users may set seed prior to running.

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
        minimum separation between a pareto and any other system

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
    X = {}
    for i in range(n_systems):
        # Generate system in uniformly in sphere with radius rad.
        # X[i] = np.random.multivariate_normal([0] * n_obj, np.identity(n_obj))
        # s2 = sum(X[i]**2)
        # X[i] = X[i] * radius * (sp.special.gammainc(s2 / 2, n_obj / 2)**(1 / n_obj)) / np.sqrt(s2)
        # X[i] = list(X[i] + center)  # Recenter.
        y = np.random.multivariate_normal([0] * n_obj, np.identity(n_obj))  # Generate standard normal vector.
        u = np.random.uniform()  # Generate a Uniform[0, 1] random variate.
        z = (y / np.linalg.norm(y)) * u**(1/n_obj) # Normalize and reweight by radius to put uniformly within the unit ball.
        X[i] = z * radius + np.ones(n_obj) * center  # Recenter.
    if minsep > 0:
        # Find paretos.
        paretos = list(moso_utils.get_nondom(X))
        non_paretos = [system for system in range(n_systems) if system not in paretos]
        # A system will be designated bad if it's too close to a pareto system.
        bads = []
        for par1_ind in range(len(paretos)):
            for par2_ind in range(par1_ind):
                par1 = paretos[par1_ind]
                par2 = paretos[par2_ind]
            # No need to make a comparison if one of the systems is bad.
                if par2 not in bads and par1 not in bads:
                    # A pareto is bad if it's within minsep of a non-bad pareto along any objective
                    if np.any([np.abs(X[par1][obj] - X[par2][obj]) < minsep for obj in range(n_obj)]):
                        bads.append(par2)
        # Remove duplicates.
        bads = list(set(bads))
        # Remove bads from paretos.
        paretos = [par for par in paretos if par not in bads]

        for par in paretos:
            for nonpar in non_paretos:
                if np.any([np.abs(X[nonpar][obj] - X[par][obj]) < minsep for obj in range(n_obj)]):
                    bads.append(nonpar)

        for system in bads:
            ind = 0
            while ind == 0:
                y = np.random.multivariate_normal([0] * n_obj, np.identity(n_obj))  # Generate standard normal vector.
                u = np.random.uniform()  # Generate a Uniform[0, 1] random variate.
                z = (y / np.linalg.norm(y)) * u**(1/n_obj) # Normalize and reweight by radius to put uniformly within the unit ball.
                X[system] = z * radius + np.ones(n_obj) * center  # Recenter.
                # X[system] = np.random.multivariate_normal([1] * n_obj, np.identity(n_obj))
                # s2 = sum(X[system]**2)
                # X[system] = X[system] * radius * (sp.special.gammainc(s2 / 2, n_obj / 2)**(1 / n_obj)) / np.sqrt(s2)
                # X[system] = list(X[system] + center)  # Recenter.
                ind2 = 0
                ind3 = 0
                for par in paretos:  # Compare to each pareto system.
                    if np.any([abs(X[system][obj] - X[par][obj]) < minsep for obj in range(n_obj)]):  # Check minimum separation.
                        ind2 = ind2 + 1
                    if np.all([X[system][obj] - X[par][obj] > 0 for obj in range(n_obj)]):  # Make sure new system is not pareto.
                        ind3 = ind3 + 1
                if ind2 == 0 and ind3 > 0:
                    ind = 1  # Exit loop if we're outside of minsep of all paretos and dominated by any pareto.

    objectives = X

    if corr is None:
        p = False
        while not p:
            rho = np.random.uniform(low=-1, high=1)
            covar = rho * sigma * sigma
            cp = (np.ones(n_obj) - np.identity(n_obj)) * covar + np.identity(n_obj) * sigma**2
            # Check if it's positive semi-definite.
            p = np.all(np.linalg.eigvals(cp) > 0) and np.all(cp - cp.T == 0)
    else:
        rho = corr
        covar = rho * sigma * sigma
        cp = (np.ones(n_obj) - np.identity(n_obj)) * covar + np.identity(n_obj) * sigma**2
        # Check if it's positive semi-definite.
        p = np.all(np.linalg.eigvals(cp) > 0) and np.all(cp - cp.T == 0)
        if not p:
            print("Input correlation value results in non-SPD covariance matrix. Finding nearest PSD matrix.")
            cp = nearestSPD(cp)
    variances = {}
    for i in range(n_systems):
        variances[i] = cp
        objectives[i] = list(objectives[i])
    return create_allocation_problem(objectives, variances)


def create_mocba_problem(covtype):
    """Generate the MOCBA 25 example problem.

    Parameters
    ----------
    covtype : str
        "ind" sets objectives to be independent
        "pos" sets all objectives to have parwise correlation of 0.4
        "neg" sets all objectives to have pairwise correlation of -0.4

    Returns
    -------
        problem : dict

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
    obj = {0: [8, 36, 60], 1: [12, 32, 52], 2: [14, 38, 54], 3: [16, 46, 48], 4: [4, 42, 56],
           5: [18, 40, 62], 6: [10, 44, 58], 7: [20, 34, 64], 8: [22, 28, 68], 9: [24, 40, 62],
           10: [26, 38, 64], 11: [28, 40, 66], 12: [30, 42, 62], 13: [32, 44, 64], 14: [26, 40, 66],
           15: [28, 42, 64], 16: [32, 38, 66], 17: [30, 40, 62], 18: [34, 42, 64], 19: [26, 44, 60],
           20: [28, 38, 66], 21: [32, 40, 62], 22: [30, 46, 64], 23: [32, 44, 66], 24: [30, 40, 64]}
    covs = {}

    if covtype == "ind":
        cov = np.identity(n_obj) * 8
    elif covtype == "pos":
        cov = np.array([[64, 0.4 * 8 * 8, 0.4 * 8 * 8], [0.4 * 8 * 8, 64, 0.4 * 8 * 8], [0.4 * 8 * 8, 0.4 * 8 * 8, 64]])
    elif covtype == "neg":
        cov = np.array([[64, -0.4 * 8 * 8, -0.4 * 8 * 8], [-0.4 * 8 * 8, 64, -0.4 * 8 * 8], [-0.4 * 8 * 8, -0.4 * 8 * 8, 64]])
    else:
        raise ValueError("Invalid covtype. Valid choices are ind, pos, and neg.")
    for key in obj.keys():
        covs[key] = cov
    return create_allocation_problem(obj, covs)


def create_test_problem_2():
    """Generate Test Problem 2 from **insert citation**.

    Returns
    -------
    problem : dict

            problem['obj'] is a dictionary of numpy arrays, indexed by system number,
                each of which corresponds to the objective values of a system
            problem['var'] is a dictionary of 2d numpy arrays, indexed by system number,
                each of which corresponds to the covariance matrix of a system
            problem['inv_var'] is a dictionary of 2d numpy, indexed by system number,
                each of which corresponds to the inverse covariance matrix of a system
            problem['pareto_indices'] is a list of pareto systems ordered by the first objective
            problem['non_pareto_indices'] is a list of pareto systems ordered by the first objective
    """
    # Read in problem details (objectives) from a file.
    obj_array = np.genfromtxt('TP2_Objs.csv', delimiter=',')
    obj = {}
    for i in range(len(obj_array[:, 0])):
        obj[i] = list(obj_array[i, :])
    covs = {}
    for key in obj.keys():
        covs[key] = np.identity(len(obj[key]))
    return create_allocation_problem(obj, covs)


# TODO: Function below has not been updated.

# def allocation_to_sequential(allocation_problem, rng, crnflag = False, simpar = 1):
#     """Create an oracle object that produces a multivariate normal objective value
#     with the "true" mean and variance structure provided.

#     Parameters
#     ----------
#     allocation_problem : dict

#             allocation_problem['obj'] is a dictionary of numpy arrays, indexed by system number,
#                 each of which corresponds to the objective values of a system
#             allocation_problem['var'] is a dictionary of 2d numpy arrays, indexed by system number,
#                 each of which corresponds to the covariance matrix of a system
#             allocation_problem['inv_var'] is a dictionary of 2d numpy, indexed by system number,
#                 each of which corresponds to the inverse covariance matrix of a system
#             allocation_problem['pareto_indices'] is a list of pareto systems ordered by the first objective
#             allocation_problem['non_pareto_indices'] is a list of pareto systems ordered by the first objective

#     rng : a pymoso.prng.MRG32k3a object

#     crnflag : bool
#         if true, the oracle will utilize common random numbers

#     simpar : int
#         the number of parallel processes used in taking simulation replications

#     Returns
#     -------
#     allocation_problem : dict
#         identical to input

#     mors_problem : Random_MORS_Problem object
#         inherits from oracle
#     """
#     mors_problem = Random_MORS_Problem(allocation_problem["obj"], allocation_problem["var"], rng, crnflag=crnflag, simpar=simpar)
#     return allocation_problem, mors_problem
