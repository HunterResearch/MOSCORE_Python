#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Summary
-------
Provide functions for determining allocations under different rules.

Listing
-------
allocate : function
equal_allocation : function
score_allocation_smart : function
phantom_allocation_smart : function
bf_allocation_smart : function
bfind_allocation_smart : function
"""

import numpy as np

from iscore_allocation import iscore_allocation
from score_allocation import score_allocation
from phantom_allocation import calc_phantom_allocation
from brute_force_allocation import calc_bf_allocation


def allocate(method, systems, WSFlag=False, warm_start=None):
    """Generate a non-sequential simulation allocation for the MORS problem.

    Parameters
    ----------
    method : str
        Chosen allocation method. Options are "iSCORE", "SCORE", "Phantom", "Brute Force", and "Brute Force Ind".
    systems : dict
            systems['obj'] is a dictionary of numpy arrays, indexed by system number,
                each of which corresponds to the objective values of a system
            systems['var'] is a dictionary of 2d numpy arrays, indexed by system number,
                each of which corresponds to the covariance matrix of a system
            systems['inv_var'] is a dictionary of 2d numpy, indexed by system number,
                each of which corresponds to the inverse covariance matrix of a system
            systems['pareto_indices'] is a list of pareto systems ordered by the first objective
            systems['non_pareto_indices'] is a list of non-pareto systems ordered by the first objective
    WSFlag : bool
        True if warm-start is to be done, otherwise False.
    warm_start : list of float
        An initial simulation allocation from which to determine the optimal allocation.
        Length must be equal to the number of systems.

    Returns
    -------
    outs : tuple
        outs[0] is the estimated optimal simulation allocation.
            A list of float of length equal to the number of systems.
        outs[1] is the estimated rate of convergence.
    """
    if warm_start is not None and len(warm_start) != len(systems['obj']):
        raise ValueError("Length of warm_start must be equal to the number of systems.")
    # Call the proper allocation rule.
    if method == "Equal":
        return equal_allocation(systems)
    elif method == "iSCORE":
        return iscore_allocation(systems, warm_start=warm_start)
    elif method == "SCORE":
        return score_allocation_smart(systems, WSFlag=WSFlag, warm_start=warm_start)
    elif method == "Phantom":
        return phantom_allocation_smart(systems, WSFlag=WSFlag, warm_start=warm_start)
    elif method == "Brute Force":
        return bf_allocation_smart(systems, WSFlag=WSFlag, warm_start=warm_start)
    elif method == "Brute Force Ind":
        return bfind_allocation_smart(systems, WSFlag=WSFlag, warm_start=warm_start)
    else:
        raise ValueError("Invalid method selected. Valid methods are 'Equal', 'iSCORE', 'SCORE', 'Phantom', 'Brute Force', and 'Brute Force Ind'.")


def equal_allocation(systems):
    """Generate a non-sequential simulation allocation for the MORS problem
    using equal allocation.

    Parameters
    ----------
    systems : dict
        systems['obj'] is a dictionary of numpy arrays, indexed by system number,
            each of which corresponds to the objective values of a system
        systems['var'] is a dictionary of 2d numpy arrays, indexed by system number,
            each of which corresponds to the covariance matrix of a system
        systems['inv_var'] is a dictionary of 2d numpy, indexed by system number,
            each of which corresponds to the inverse covariance matrix of a system
        systems['pareto_indices'] is a list of pareto systems ordered by the first objective
        systems['non_pareto_indices'] is a list of non-pareto systems ordered by the first objective

    Returns
    -------
    outs : tuple
        outs[0] is the estimated optimal simulation allocation.
            A list of float of length equal to the number of systems.
        outs[1] is automatically set to zero
    """
    n_systems = len(systems["obj"])
    alloc = [1 / n_systems for _ in range(n_systems)]
    # Associated rate is set as zero.
    zval = 0
    return alloc, zval


def score_allocation_smart(systems, WSFlag=False, warm_start=None):
    """Generate a non-sequential simulation allocation for the MORS problem
    using the SCORE method.

    Parameters
    ----------
    systems : dict
        systems['obj'] is a dictionary of numpy arrays, indexed by system number,
            each of which corresponds to the objective values of a system
        systems['var'] is a dictionary of 2d numpy arrays, indexed by system number,
            each of which corresponds to the covariance matrix of a system
        systems['inv_var'] is a dictionary of 2d numpy, indexed by system number,
            each of which corresponds to the inverse covariance matrix of a system
        systems['pareto_indices'] is a list of pareto systems ordered by the first objective
        systems['non_pareto_indices'] is a list of non-pareto systems ordered by the first objective
    WSFlag : bool
        True if warm-start is to be done, otherwise False.
    warm_start : list of float
        An initial simulation allocation from which to determine the optimal allocation.
        Length must be equal to the number of systems.

    Returns
    -------
    outs : tuple
        outs[0] is the estimated optimal simulation allocation.
            A list of float of length equal to the number of systems.
        outs[1] is the estimated rate of convergence.
    """
    # If more than 3 objectives, use iSCORE allocation as a warm-start solution.
    if len(systems['obj'][0]) > 3 and WSFlag:
        warm_start = iscore_allocation(systems, warm_start=warm_start)[0]
        # [0] corresponds to allocation. [1] would be associated rate.
    return score_allocation(systems, warm_start=warm_start)


def phantom_allocation_smart(systems, WSFlag=False, warm_start=None):
    """Generate a non-sequential simulation allocation for the MORS problem
    using the Phantom method.

    Parameters
    ----------
    systems : dict
        systems['obj'] is a dictionary of numpy arrays, indexed by system number,
            each of which corresponds to the objective values of a system
        systems['var'] is a dictionary of 2d numpy arrays, indexed by system number,
            each of which corresponds to the covariance matrix of a system
        systems['inv_var'] is a dictionary of 2d numpy, indexed by system number,
            each of which corresponds to the inverse covariance matrix of a system
        systems['pareto_indices'] is a list of pareto systems ordered by the first objective
        systems['non_pareto_indices'] is a list of non-pareto systems ordered by the first objective
    WSFlag : bool
        True if warm-start is to be done, otherwise False.
    warm_start : list of float
        An initial simulation allocation from which to determine the optimal allocation.
        Length must be equal to the number of systems.

    Returns
    -------
    outs : tuple
        outs[0] is the estimated optimal simulation allocation.
            A list of float of length equal to the number of systems.
        outs[1] is the estimated rate of convergence.
    """
    # If more than 3 objectives, use iSCORE allocation as a warm-start solution.
    if len(systems['obj'][0]) > 3 and WSFlag:
        warm_start = iscore_allocation(systems, warm_start=warm_start)[0]
        # [0] corresponds to allocation. [1] would be associated rate.
    return calc_phantom_allocation(systems, warm_start=warm_start)


def bf_allocation_smart(systems, WSFlag=False, warm_start=None):
    """Generate a non-sequential simulation allocation for the MORS problem
    using the Brute Force method.

    Parameters
    ----------
    systems : dict
        systems['obj'] is a dictionary of numpy arrays, indexed by system number,
            each of which corresponds to the objective values of a system
        systems['var'] is a dictionary of 2d numpy arrays, indexed by system number,
            each of which corresponds to the covariance matrix of a system
        systems['inv_var'] is a dictionary of 2d numpy, indexed by system number,
            each of which corresponds to the inverse covariance matrix of a system
        systems['pareto_indices'] is a list of pareto systems ordered by the first objective
        systems['non_pareto_indices'] is a list of non-pareto systems ordered by the first objective
    WSFlag : bool
        True if warm-start is to be done, otherwise False.
    warm_start : list of float
        An initial simulation allocation from which to determine the optimal allocation.
        Length must be equal to the number of systems.

    Returns
    -------
    outs : tuple
        outs[0] is the estimated optimal simulation allocation.
            A list of float of length equal to the number of systems.
        outs[1] is the estimated rate of convergence.
    """
    # If more than 3 objectives, use iSCORE allocation as a warm-start solution.
    if len(systems['obj'][0]) > 3 and WSFlag:
        warm_start = iscore_allocation(systems, warm_start=warm_start)[0]
        # [0] corresponds to allocation. [1] would be associated rate.
    return calc_bf_allocation(systems, warm_start=warm_start)


def bfind_allocation_smart(systems, WSFlag=False, warm_start=None):
    """Generate a non-sequential simulation allocation for the MORS problem
    using the Brute Force Independent method.

    Parameters
    ----------
    systems : dict
        systems['obj'] is a dictionary of numpy arrays, indexed by system number,
            each of which corresponds to the objective values of a system
        systems['var'] is a dictionary of 2d numpy arrays, indexed by system number,
            each of which corresponds to the covariance matrix of a system
        systems['inv_var'] is a dictionary of 2d numpy, indexed by system number,
            each of which corresponds to the inverse covariance matrix of a system
        systems['pareto_indices'] is a list of pareto systems ordered by the first objective
        systems['non_pareto_indices'] is a list of non-pareto systems ordered by the first objective
    WSFlag : bool
        True if warm-start is to be done, otherwise False.
    warm_start : list of float
        An initial simulation allocation from which to determine the optimal allocation.
        Length must be equal to the number of systems.

    Returns
    -------
    outs : tuple
        outs[0] is the estimated optimal simulation allocation.
            A list of float of length equal to the number of systems.
        outs[1] is the estimated rate of convergence.
    """
    # Extract number of objective and number of systems.
    n_obj = len(systems["obj"][0])
    n_systems = len(systems["obj"])
    # Replace covariance matrices with identity matrices.
    idmat = np.identity(n_obj)
    for s in range(n_systems):
        systems['var'][s] = idmat
    # Perform allocation as in 'normal' brute force.
    # If more than 3 objectives, use iSCORE allocation as a warm-start solution.
    if len(systems['obj'][0]) > 3 and WSFlag:
        warm_start = iscore_allocation(systems, warm_start=warm_start)[0]
        # [0] corresponds to allocation. [1] would be associated rate.
    return calc_bf_allocation(systems, warm_start=warm_start)

# """
# Created on Tue Aug 13 17:42:23 2019
# @author: nathangeldner
# Updated on 11/06/19 by Eric Applegate
# """
