#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Summary
-------
Provide functions for phantom allocation.

Listing
-------
calc_phantom_allocation : function
phantom_constraints_wrapper : function
phantom_constraints : function
MCI_phantom_rates : function
find_phantoms : function
sweep : function
"""


import numpy as np
import itertools as it
import scipy.optimize as opt

from brute_force_allocation import MCE_brute_force_rates, objective_function, hessian_zero
from MCI_hard_coded import MCI_1d, MCI_2d, MCI_3d, MCI_four_d_plus
from utils import is_pareto_efficient


def calc_phantom_allocation(systems, warm_start=None):
    """Calculates Phantom Allocation given a set of systems and optional warm start

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
            out_tuple[1]: the estimated convergence rate associatedc with the optimal allocation"""
    # Extract number of objectives, number of systems, and number of pareto systems.
    n_obj = len(systems["obj"][0])
    n_systems = len(systems["obj"])
    num_par = len(systems["pareto_indices"])

    # Get array of pareto values for the phantom finder.
    pareto_array = np.zeros([num_par, n_obj])
    for i in range(num_par):
        pareto_array[i, :] = systems['obj'][systems['pareto_indices'][i]]
    # Get the phantom systems.
    phantom_values = find_phantoms(pareto_array, n_obj)

    # Sort the phantom system.
    for i in range(n_obj):
        phantom_values = phantom_values[(phantom_values[:, n_obj - 1 - i]).argsort(kind='mergesort')]
    n_phantoms = len(phantom_values)

    # The commented part doesn't give different results, but this makes the constraint
    # ordering identical to that of the matlab code, which you'll want for debugging
    # phantom_values = phantom_values[(phantom_values[:,0]).argsort()]

    # TODO: consider using something other than inf as a placeholder.
    # Unfortunately, inf is a float in numpy, and arrays must be homogenous
    # and floats don't automatically cast to ints for indexing leading to an error.
    # Right now we're casting as ints for indexing, but that's a little gross.
    # Also, inf doesn't cast to intmax if you cast as int.
    # It ends up being very negative for some reason.
    phantoms = np.ones([n_phantoms, n_obj]) * np.inf

    # TODO: Vectorize?
    for i in range(n_phantoms):
        for j in range(n_obj):
            for k in range(num_par):
                if pareto_array[k, j] == phantom_values[i, j]:
                    phantoms[i, j] = systems['pareto_indices'][k]

    phantom_constraints_wrapper.last_alphas = None

    # We define a callable for the constraint values and another for the constraint jacobian.
    def constraint_values(x):
        return phantom_constraints_wrapper(x, systems, phantoms, num_par, n_obj, n_systems)[0]

    def constraint_jacobian(x):
        return phantom_constraints_wrapper(x, systems, phantoms, num_par, n_obj, n_systems)[1]

    # lambda is for function handles.
    # constraint_values = lambda x: phantom_constraints_wrapper(x, systems, phantoms, num_par,n_obj, n_systems)[0]
    # constraint_jacobian = lambda x: phantom_constraints_wrapper(x, systems, phantoms, num_par,n_obj, n_systems)[1]

    # Define nonlinear constraint object for the optimizer.
    # Will not work if we switch away from trust-constr, but the syntax isn't that different if we do.
    nonlinear_constraint = opt.NonlinearConstraint(constraint_values,
                                                   lb=-np.inf,
                                                   ub=0.0,
                                                   jac=constraint_jacobian,
                                                   keep_feasible=False
                                                   )

    # Define bounds on alpha values and z (the last element of our decision variable array).
    my_bounds = [(10**-12, 1.0) for i in range(n_systems)] + [(0.0, np.inf)]

    if warm_start is None:
        warm_start = np.array([1.0 / n_systems] * n_systems + [0.0])
    else:
        warm_start = np.append(warm_start, 0)

    # The objective is -z and its derivative wrt z is -1.
    def obj_function(x):
        return objective_function(x)
    # obj_function = lambda x: objective_function(x)

    # Set sum of alphas (not z) to 1.
    equality_constraint_array = np.ones(n_systems + 1)
    equality_constraint_array[-1] = 0.0
    equality_constraint_bound = 1.0
    equality_constraint = opt.LinearConstraint(equality_constraint_array,
                                               equality_constraint_bound,
                                               equality_constraint_bound
                                               )

    # Solve optimization problem.
    res = opt.minimize(fun=obj_function,
                       x0=warm_start,
                       method='trust-constr',
                       jac=True,
                       hess=hessian_zero,
                       bounds=my_bounds,
                       constraints=[equality_constraint, nonlinear_constraint]
                       )

    # If first attempt to optimize terminated improperly, warm-start at
    # final solution and try again.
    if res.status == 0:
        print("cycling")
        res = opt.minimize(fun=objective_function,
                           x0=res.x,
                           method='trust-constr',
                           jac=True,
                           hess=hessian_zero,
                           bounds=my_bounds,
                           constraints=[equality_constraint, nonlinear_constraint]
                           )

    # stop_flag = 1
    # if res.status == 0:
    #     stop_flag = 0
    #     print("means = ", systems["obj"])
    #     print("var = ", systems["var"])
    # # If latest attempt to optimize terminated improperly, warm-start at
    # # final solution and try again.
    # while stop_flag == 0:
    #     print("cycling")
    #     print(res.message)
    #     print(res.status)
    #     print("allocation = ", res.x[:-1])
    #     print("rate = ", res.x[-1])
    #     res = opt.minimize(fun=objective_function,
    #                        x0=res.x,
    #                        method='trust-constr',
    #                        jac=True,
    #                        hess=hessian_zero,
    #                        bounds=my_bounds,
    #                        constraints=[equality_constraint, nonlinear_constraint]
    #                        )
    #     if res.status != 0:
    #         stop_flag = 1
    # # print(res.constr_violation)
    # print(res.message)

    # res.x includes alphas and z. To return only alphas, subset with res.x[0:-1].
    return res.x[0:-1], res.x[-1]


def phantom_constraints_wrapper(alphas, systems, phantoms, num_par, n_obj, n_systems):
    """scipy optimization methods don't directly support simultaneous computation
    of constraint values and their gradients. Additionally, it only considers one constraint and its gradient
    at a time, as separate functions. Thus we check whether we're looking at the same alphas
    as the last call, and if so return the same output

    parameters:
            alphas: numpy array of length n_systems + 1 consisting of allocation for each system and estimated convergence rate
            systems: dict, as described under calc_bf_allocation()
            phantoms: numpy matrix with n_obj columns and an arbitrary number of rows, where each element is a pareto system number. Each row corresponds to a phantom pareto system - pareto system number n in column j implies that the phantom pareto has the same value in objective j as pareto system n
            num_par: integer, number of estimated pareto systems
            n_obj: number of systems
            n_systems: integer, number of total systems

    output:
            rates: numpy array, giving the value of z(estimated convergence rate) minus the convergence rate upper bound associated with each constraint
            jacobian: 2d numy array, giving the jacobian of the rates with respect to the vector alpha (including the final element z)

    """
    if all(alphas == phantom_constraints_wrapper.last_alphas):
        return phantom_constraints_wrapper.last_outputs
    else:
        rates, jacobian = phantom_constraints(alphas, systems, phantoms, num_par, n_obj, n_systems)
        phantom_constraints_wrapper.last_alphas = alphas
        phantom_constraints_wrapper.last_outputs = rates, jacobian
        return rates, jacobian


def phantom_constraints(alphas, systems, phantoms, num_par, n_obj, n_systems):
    """calculates MCE constraints and MCI constraints on the convergence rate and appends them together, where the value of each constraint is equal to z, the total rate estimator, minus the rate associated with each possible MCI and MCE event based on the phantom pareto set, each of which serves as an upper bound on the total rate

    parameters:
            alphas: numpy array of length n_systems + 1 consisting of allocation for each system and estimated convergence rate
            systems: dict, as described under calc_bf_allocation()
            phantoms: numpy matrix with n_obj columns and an arbitrary number of rows, where each element is a pareto system number. Each row corresponds to a phantom pareto system - pareto system number n in column j implies that the phantom pareto has the same value in objective j as pareto system n
            num_par: integer, number of estimated pareto systems
            n_obj: number of systems
            n_systems: integer, number of total systems

    output:
            rates: numpy array, giving the value of z(estimated convergence rate) minus the convergence rate upper bound associated with each constraint
            jacobian: 2d numy array, giving the jacobian of the rates with respect to the vector alpha (including the final element z)

    """
    # Compose MCE and MCI constraint values and gradients.
    MCE_rates, MCE_grads = MCE_brute_force_rates(alphas, systems, num_par, n_systems, n_obj)
    MCI_rates, MCI_grads = MCI_phantom_rates(alphas, systems, phantoms, num_par, n_systems, n_obj)
    rates = np.append(MCE_rates, MCI_rates, axis=0)
    grads = np.append(MCE_grads, MCI_grads, axis=0)
    return rates, grads


def MCI_phantom_rates(alphas, systems, phantoms, num_par, n_systems, n_obj):
    """calculates the MCI Phantom rate constraint values and jacobian

    parameters:
            alphas:  numpy array of length n_systems + 1 consisting of allocation for each system and estimated convergence rate
            systems: dict, as described under calc_bf_allocation()
            phantoms: numpy matrix with n_obj columns and an arbitrary number of rows, where each element is a pareto system number. Each row corresponds to a phantom pareto system - pareto system number n in column j implies that the phantom pareto has the same value in objective j as pareto system n
            num_par: integer, number of estimated pareto systems
            n_systems: integer, number of total systems
            n_obj: integer, number of objectives
    output:
            MCE_rates: numpy array, giving the value of z(estimated convergence rate) minus the convergence rate upper bound associated with each MCE constraint
            MCE_grads: 2d numy array, giving the jacobian of the MCE constraint values with respect to the vector alpha (including the final element z)"""

    tol = 10**-12
    n_phantoms = len(phantoms)
    n_nonpar = n_systems - num_par
    n_MCI = n_nonpar * n_phantoms

    MCI_rates = np.zeros(n_MCI)
    MCI_grads = np.zeros([n_MCI, n_systems + 1])

    count = 0
    alphas[0:-1][alphas[0:-1] <= tol] = 0
    for j in systems['non_pareto_indices']:
        for m in range(n_phantoms):
            # Get the pareto indices corresponding to phantom l.
            phantom_indices = phantoms[m, :]
            if alphas[j] <= tol:
                # The rate and gradients are zero. Only have to worry about gradient
                # wrt z since we initialize with zero.
                MCI_grads[count, -1] = 1
            else:
                phantom_obj = np.zeros(n_obj)
                phantom_var = np.zeros(n_obj)
                phantom_alphas = np.zeros(n_obj)
                phantom_objectives = np.array(range(n_obj))
                phantom_objective_count = n_obj
                alpha_zeros = 0

                # Extract objective and variance values for the phantom pareto system.
                for b in range(n_obj):
                    if phantom_indices[b] < np.inf:
                        pareto_system = int(phantom_indices[b])
                        phantom_obj[b] = systems['obj'][pareto_system][b]
                        phantom_var[b] = systems['var'][pareto_system][b, b]
                        if alphas[pareto_system] <= tol:
                            phantom_alphas[b] = 0
                            alpha_zeros = alpha_zeros + 1
                        else:
                            phantom_alphas[b] = alphas[pareto_system]
                    else:
                        phantom_objective_count -= 1

                # Keep track of which objectives are included in phantom set.
                phantom_objectives = phantom_objectives[phantom_indices < np.inf]
                obj_j = systems['obj'][j][phantom_objectives]
                # Only want covariances for the phantom objectives.
                # np.ix_ allows us to subset nicely that way.
                cov_j = systems['var'][j][np.ix_(phantom_objectives, phantom_objectives)]

                # Remove unassigned objective indices for phantom variances and objectives.
                phantom_obj = phantom_obj[phantom_objectives]
                phantom_var = phantom_var[phantom_objectives]
                phantom_alphas = phantom_alphas[phantom_objectives]

                # If all of the alphas corresponding to the phantom objectives are zero:
                if alpha_zeros == phantom_objective_count:
                    rate = 0
                    grad_j = 0
                    phantom_grads = 0.5 * ((obj_j - phantom_obj) ** 2) / phantom_var

                    # Note: floats equal to ints don't get automatically converted when used for indices.
                    # Need to convert.
                    MCI_grads[count, phantom_indices[phantom_indices < np.inf].astype(int)] = -1.0 * phantom_grads
                    MCI_grads[count, -1] = 1
                    MCI_rates[count] = alphas[-1] - rate
                else:
                    length = len(phantom_objectives)
                    if length == 1:
                        rate, grad_j, phantom_grads = MCI_1d(alphas[j], obj_j, cov_j, phantom_alphas, phantom_obj, phantom_var)
                    elif length == 2:
                        rate, grad_j, phantom_grads = MCI_2d(alphas[j], obj_j, cov_j, phantom_alphas, phantom_obj, phantom_var)
                    elif length == 3:
                        rate, grad_j, phantom_grads = MCI_3d(alphas[j], obj_j, cov_j, phantom_alphas, phantom_obj, phantom_var)
                    else:
                        rate, grad_j, phantom_grads = MCI_four_d_plus(alphas[j], obj_j, cov_j, phantom_alphas, phantom_obj, phantom_var)

                    # TODO: Hard-code solutions for 1-3 objectives.
                    MCI_grads[count, phantom_indices[phantom_indices < np.inf].astype(int)] = -1.0 * phantom_grads
                    MCI_grads[count, -1] = 1
                    MCI_grads[count, j] = -1.0 * grad_j
                    MCI_rates[count] = alphas[-1] - rate

            count = count + 1
    return MCI_rates, MCI_grads


def find_phantoms(paretos, n_obj):
    """finds the phantom pareto set

    paramters:
            paretos: a numpy matrix representing the pareto set, with a row for each pareto system and a column for each objective
            n_obj: integer number of objectives
            num_par: integer number of pareto systems

    output:
            phantoms: a numpy matrix similar in structure to paretos characterizing the phantom paretos. same number of columns but a difficult-to-predict number of rows
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
    """TO DO"""

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


"""
Created on Mon May 27 19:25:06 2019

@author: nathangeldner
"""
