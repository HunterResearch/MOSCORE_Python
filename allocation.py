"""
Summary
-------
Provide allocate_wrapper() function and class definitions for
convex-optimization allocation algorithms.

Listing
-------
allocate_wrapper : function
allocate : function
equal_allocation : function
"""

import numpy as np

# Temporary imports
from iscore_allocation import iscore_allocation
from score_allocation import score_allocation
from phantom_allocation import calc_phantom_allocation
from brute_force_allocation import calc_bf_allocation


def allocate_wrapper(method, systems, warm_start=None):
    """Generate a non-sequential simulation allocation for the MORS problem.

    Parameters
    ----------
    method : str
        Chosen allocation method. Options are "iSCORE", "SCORE", "Phantom", "Brute Force", and "Brute Force Ind".
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
    warm_start : list of float
        An initial simulation allocation from which to determine the optimal allocation.\
        Length must be equal to the number of systems.

    Returns
    -------
    alpha : tuple
        The estimated optimal simulation allocation, which is a list of float of length equal to the number of systems.
    z : float
        The estimated rate of convergence.
    """
    if warm_start is not None and len(warm_start) != len(systems['obj']):
        raise ValueError("Length of warm_start must be equal to the number of systems.")
    # Call the proper allocation rule.
    # For certain settings, call a related allocation rule to be more efficient,
    # e.g., warmstart.
    if method == "Equal":
        return equal_allocation(systems)
    elif method == "iSCORE":
        return allocate(method="iSCORE", systems=systems, warm_start=warm_start)
    elif method == "SCORE":
        # If more than 3 objectives, use iSCORE allocation as a warmer-start solution.
        if len(systems['obj'][0]) > 3:
            warm_start, _ = allocate(method="iSCORE", systems=systems, warm_start=warm_start)
        return allocate(method="SCORE", systems=systems, warm_start=warm_start)
    elif method == "Phantom":
        # If more than 3 objectives, use iSCORE allocation as a warmer-start solution.
        if len(systems['obj'][0]) > 3:
            warm_start, _ = allocate(method="iSCORE", systems=systems, warm_start=warm_start)
        return allocate(method="Phantom", systems=systems, warm_start=warm_start)
    elif method == "Brute Force":
        # If 2 or fewer objetives, use phantom allocation instead.
        # It is equivalent to the brute-force allocation, but easier to solve.
        if len(systems['obj'][0]) <= 2:
            return allocate(method="Phantom", systems=systems, warm_start=warm_start)
        # If more than 3 objectives, use iSCORE allocation as a warmer-start solution.
        else:
            warm_start, _ = allocate(method="iSCORE", systems=systems, warm_start=warm_start)
            return allocate(method="Brute Force", systems=systems, warm_start=warm_start)
    elif method == "Brute Force Ind":
        # Extract number of objective and number of systems.
        n_obj = len(systems["obj"][0])
        n_systems = len(systems["obj"])
        # Replace covariance matrices with identity matrices.
        idmat = np.identity(n_obj)
        for s in range(n_systems):
            systems['var'][s] = idmat
        # Perform allocation as in 'normal' brute force.
        # If more than 3 objectives, use iSCORE allocation as a warmer-start solution.
        if len(systems['obj'][0]) > 3:
            warm_start, _ = allocate(method="iSCORE", systems=systems, warm_start=warm_start)
        return allocate(method="Brute Force", systems=systems, warm_start=warm_start)
    else:
        raise ValueError("Invalid method selected. Valid methods are 'Equal', 'iSCORE', 'SCORE', 'Phantom', 'Brute Force', and 'Brute Force Ind'.")


def allocate(method, systems, warm_start=None):
    """Generate a simulation allocation for the MORS problem using
    a specified method.

    Parameters
    ----------
    method : str
        Chosen allocation method. Options are "iSCORE", "SCORE", "Phantom", "Brute Force".
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
    warm_start : list of float
        An initial simulation allocation from which to determine the optimal allocation.\
        Length must be equal to the number of systems.

    Returns
    -------
    alpha : tuple
        The estimated optimal simulation allocation, which is a list of float of length equal to the number of systems.
    z : float
        The estimated rate of convergence.
    """
    if method == "iSCORE":
        cvxoptallocalg = ISCORE(systems=systems)
    elif method == "SCORE":
        cvxoptallocalg = SCORE(systems=systems)
    elif method == "Phantom":
        cvxoptallocalg = Phantom(systems=systems)
    elif method == "Brute Force":
        cvxoptallocalg = BruteForce(systems=systems)
    alpha, z = cvxoptallocalg.solve(warm_start=warm_start)
    return alpha, z


def equal_allocation(systems):
    """Generate a non-sequential simulation allocation for the MORS problem
    using equal allocation.

    Parameters
    ----------
    systems : dict
        ``"obj"``
        A dictionary of numpy arrays, indexed by system number,
            each of which corresponds to the objective values of a system.
        ``"var"``
        A dictionary of 2d numpy arrays, indexed by system number,
            each of which corresponds to the covariance matrix of a system.
        ``"inv_var"``
        A dictionary of 2d numpy, indexed by system number,
            each of which corresponds to the inverse covariance matrix of a system.
        ``"pareto_indices"``
        A list of pareto systems ordered by the first objective.
        ``"non_pareto_indices"``
        A list of non-pareto systems ordered by the first objective.

    Returns
    -------
    alloc : tuple
        The estimated optimal simulation allocation, which is a list of float of length equal to the number of systems.
    zval : float
        The estimated rate of convergence.
    """
    n_systems = len(systems["obj"])
    alloc = [1 / n_systems for _ in range(n_systems)]
    # Associated rate is set as zero.
    zval = 0
    return alloc, zval


class ConvexOptAllocAlg(object):
    """Class for allocation algorithms that solve convex
    optimization problems.

    Attributes
    ----------
    systems : dict
        ``"obj"``
        A dictionary of numpy arrays, indexed by system number,
            each of which corresponds to the objective values of a system.
        ``"var"``
        A dictionary of 2d numpy arrays, indexed by system number,
            each of which corresponds to the covariance matrix of a system.
        ``"inv_var"``
        A dictionary of 2d numpy, indexed by system number,
            each of which corresponds to the inverse covariance matrix of a system.
        ``"pareto_indices"``
        A list of pareto systems ordered by the first objective.
        ``"non_pareto_indices"``
        A list of non-pareto systems ordered by the first objective.
    """
    def __init__(self, systems):
        self.systems = systems

    def solve(self, warm_start=None):
        alpha, z = equal_allocation(systems=self.systems)  # TODO
        return alpha, z


class ISCORE(ConvexOptAllocAlg):
    """Class for iSCORE allocation algorithm.

    Attributes
    ----------
    systems : dict
        ``"obj"``
        A dictionary of numpy arrays, indexed by system number,
            each of which corresponds to the objective values of a system.
        ``"var"``
        A dictionary of 2d numpy arrays, indexed by system number,
            each of which corresponds to the covariance matrix of a system.
        ``"inv_var"``
        A dictionary of 2d numpy, indexed by system number,
            each of which corresponds to the inverse covariance matrix of a system.
        ``"pareto_indices"``
        A list of pareto systems ordered by the first objective.
        ``"non_pareto_indices"``
        A list of non-pareto systems ordered by the first objective.
    """
    def solve(self, warm_start=None):
        alpha, z = iscore_allocation(systems=self.systems, warm_start=warm_start)  # TODO
        return alpha, z


class SCORE(ConvexOptAllocAlg):
    """Class for SCORE allocation algorithm.

    Attributes
    ----------
    systems : dict
        ``"obj"``
        A dictionary of numpy arrays, indexed by system number,
            each of which corresponds to the objective values of a system.
        ``"var"``
        A dictionary of 2d numpy arrays, indexed by system number,
            each of which corresponds to the covariance matrix of a system.
        ``"inv_var"``
        A dictionary of 2d numpy, indexed by system number,
            each of which corresponds to the inverse covariance matrix of a system.
        ``"pareto_indices"``
        A list of pareto systems ordered by the first objective.
        ``"non_pareto_indices"``
        A list of non-pareto systems ordered by the first objective.
    """
    def solve(self, warm_start=None):
        alpha, z = score_allocation(systems=self.systems, warm_start=warm_start)  # TODO
        return alpha, z


class Phantom(ConvexOptAllocAlg):
    """Class for phantom allocation algorithm.

    Attributes
    ----------
    systems : dict
        ``"obj"``
        A dictionary of numpy arrays, indexed by system number,
            each of which corresponds to the objective values of a system.
        ``"var"``
        A dictionary of 2d numpy arrays, indexed by system number,
            each of which corresponds to the covariance matrix of a system.
        ``"inv_var"``
        A dictionary of 2d numpy, indexed by system number,
            each of which corresponds to the inverse covariance matrix of a system.
        ``"pareto_indices"``
        A list of pareto systems ordered by the first objective.
        ``"non_pareto_indices"``
        A list of non-pareto systems ordered by the first objective.
    """
    def solve(self, warm_start=None):
        alpha, z = calc_phantom_allocation(systems=self.systems, warm_start=warm_start)  # TODO
        return alpha, z


class BruteForce(ConvexOptAllocAlg):
    """Class for brute force allocation algorithm.

    Attributes
    ----------
    systems : dict
        ``"obj"``
        A dictionary of numpy arrays, indexed by system number,
            each of which corresponds to the objective values of a system.
        ``"var"``
        A dictionary of 2d numpy arrays, indexed by system number,
            each of which corresponds to the covariance matrix of a system.
        ``"inv_var"``
        A dictionary of 2d numpy, indexed by system number,
            each of which corresponds to the inverse covariance matrix of a system.
        ``"pareto_indices"``
        A list of pareto systems ordered by the first objective.
        ``"non_pareto_indices"``
        A list of non-pareto systems ordered by the first objective.
    """
    def solve(self, warm_start=None):
        alpha, z = calc_bf_allocation(systems=self.systems, warm_start=warm_start)  # TODO
        return alpha, z
