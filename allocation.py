#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Summary
-------
Provide smart_allocate() function and class definitions for
convex-optimization allocation algorithms.

Listing
-------
smart_allocate : function
allocate : function
equal_allocation : function
ConvexOptAllocAlg : class
BruteForce : class
Phantom : class
MOSCORE : class
IMOSCORE : class
calc_brute_force_rate : function
calc_phantom_rate : function
calc_score_rate : function
calc_iscore_rate : function
"""

import numpy as np
import itertools as it
import scipy.optimize as opt
import scipy.linalg as linalg
from cvxopt import matrix, solvers

from MCE_hard_coded import MCE_2d, MCE_3d, MCE_four_d_plus
from MCI_hard_coded import MCI_1d, MCI_2d, MCI_3d, MCI_four_d_plus
from MOSCORE_hard_coded import SCORE_1d, SCORE_2d, SCORE_3d, score_four_d_plus
from utils import find_phantoms

def smart_allocate(method, systems, warm_start=None, resolve=False):
    """Generate a non-sequential simulation allocation for the MORS problem.

    Parameters
    ----------
    method : str
        Chosen allocation method. Options are "iMOSCORE", "MOSCORE", "Phantom", "Brute Force", and "Brute Force Ind".
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
    resolve : bool
        Resolve the optimization problem if insufficiently solved if True, otherwise False.

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
    elif method == "iMOSCORE":
        return allocate(method="iMOSCORE", systems=systems, warm_start=warm_start, resolve=resolve)
    elif method == "MOSCORE":
        # If more than 3 objectives, use  iMOSCORE allocation as a warmer-start solution.
        if len(systems['obj'][0]) > 3:
            warm_start, _ = allocate(method="iMOSCORE", systems=systems, warm_start=warm_start, resolve=resolve)
        return allocate(method="MOSCORE", systems=systems, warm_start=warm_start, resolve=resolve)
    elif method == "Phantom":
        # If more than 3 objectives, use iMOSCORE allocation as a warmer-start solution.
        if len(systems['obj'][0]) > 3:
            warm_start, _ = allocate(method="iMOSCORE", systems=systems, warm_start=warm_start, resolve=resolve)
        return allocate(method="Phantom", systems=systems, warm_start=warm_start, resolve=resolve)
    elif method == "Brute Force":
        # If 2 or fewer objetives, use phantom allocation instead.
        # It is equivalent to the brute-force allocation, but easier to solve.
        if len(systems['obj'][0]) <= 2:
            return allocate(method="Phantom", systems=systems, warm_start=warm_start, resolve=resolve)
        # If more than 3 objectives, use iMOSCORE allocation as a warmer-start solution.
        else:
            warm_start, _ = allocate(method="iMOSCORE", systems=systems, warm_start=warm_start, resolve=resolve)
            return allocate(method="Brute Force", systems=systems, warm_start=warm_start, resolve=resolve)
    elif method == "Brute Force Ind":
        # NOTE: Earlier version had independence applied before iMOSCORE warmstart.
        # If more than 3 objectives, use iMOSCORE allocation as a warmer-start solution.
        if len(systems['obj'][0]) > 3:
            warm_start, _ = allocate(method="iMOSCORE", systems=systems, warm_start=warm_start, resolve=resolve)
        return allocate(method="Brute Force Ind", systems=systems, warm_start=warm_start, resolve=resolve)
    else:
        raise ValueError("Invalid method selected. Valid methods are 'Equal', 'iMOSCORE', 'MOSCORE', 'Phantom', 'Brute Force', and 'Brute Force Ind'.")


def allocate(method, systems, warm_start=None, resolve=False):
    """Generate a simulation allocation for the MORS problem using
    a specified method.

    Parameters
    ----------
    method : str
        Chosen allocation method. Options are "Equal", "iMOSCORE", "MOSCORE", "Phantom", "Brute Force", "Brute Force Ind".
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
    resolve : bool
        Resolve the optimization problem if insufficiently solved if True, otherwise False.

    Returns
    -------
    alpha : tuple
        The estimated optimal simulation allocation, which is a list of float of length equal to the number of systems.
    z : float
        The estimated rate of convergence.
    """
    if method == "Equal":
        alpha, z = equal_allocation(systems=systems)
    else:
        if method == "iMOSCORE":
            cvxoptallocalg = IMOSCORE(systems=systems)
        elif method == "MOSCORE":
            cvxoptallocalg = MOSCORE(systems=systems)
        elif method == "Phantom":
            cvxoptallocalg = Phantom(systems=systems)
        elif method == "Brute Force":
            cvxoptallocalg = BruteForce(systems=systems)
        elif method == "Brute Force Ind":
            # Modify the allocation problem to have diagonal covariance matrices.
            # Extract number of systems.
            n_systems = len(systems["obj"])
            # Modify covariance and precision matrices.
            for s in range(n_systems):
                systems['var'][s] = np.diag(np.diag(systems['var'][s]))
                systems['inv_var'][s] = np.linalg.inv(systems['var'][s])
            # Perform allocation as in 'normal' brute force.
            cvxoptallocalg = BruteForce(systems=systems)
        cvxoptallocalg.setup_opt_problem(warm_start=warm_start)
        alpha, z = cvxoptallocalg.solve_opt_problem(resolve=resolve)
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
    alpha : tuple
        The estimated optimal simulation allocation, which is a list of float of length equal to the number of systems.
    z : float
        The estimated rate of convergence.
    """
    n_systems = len(systems["obj"])
    alpha = [1 / n_systems for _ in range(n_systems)]
    # Associated rate is set as zero.
    z = 0
    return alpha, z


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
    n_obj : int
        Number of objectives
    n_systems : int
        Number of systems.
    n_paretos : int
        Number of Pareto systems.
    """
    def __init__(self, systems):
        self.systems = systems
        # Extract number of objectives, number of systems, and number of pareto systems.
        self.n_obj = len(systems["obj"][0])
        self.n_systems = len(systems["obj"])
        self.n_paretos = len(systems["pareto_indices"])
        # The following attribute may be overwritten in the __init__ function
        # of some subclasses.
        self.n_system_decision_variables = self.n_systems

    def objective_function(self, x):
        """Return the objective function value and associated gradient at a given solution.

        Parameters
        ----------
        x : a numpy array
            A feasible solution for the optimization problem.

        Returns
        -------
        out1 : float
            Objective function value.
        out2 : numpy array
            Gradient value.
        """
        # We want to maximize the convergence rate.
        # The objective is -z and its derivative wrt z is -1.
        # The objective function is -1 times the convergence rate.
        # The gradient is zero with respect to alphas and -1 with respect to the convergence rate (the last term).
        gradient = np.zeros(len(x))
        gradient[-1] = -1
        return -1 * x[-1], gradient

    def hessian_zero(self, x):
        """Return the Hessian of the objective function at a given solution.

        Parameters
        ----------
        x : a numpy array
            A feasible solution for the optimization problem.

        Returns
        -------
        out : numpy matrix
        """
        # Hessian of the objective function is just a matrix of zeros.
        # Return a square matrix of zeros with len(alpha) rows and columns.
        return np.zeros([len(x), len(x)])

    def setup_opt_problem(self, warm_start=None):
        """Setup optimization problem."""

        # We don't have a good way to pass in constraint values and gradients simultaneously.
        # This is to help with our workaround for that. Described under brute_force_constraints_wrapper.
        self.constraints_wrapper.__func__.last_x = None

        # We define a callable for the constraint values and another for the constraint jacobian.
        def constraint_values(x):
            return self.constraints_wrapper(x)[0]

        def constraint_jacobian(x):
            return self.constraints_wrapper(x)[1]

        # Define nonlinear constraint object for the optimizer.
        # Will not work if we switch away from trust-constr, but the syntax isn't that different if we do.
        self.nonlinear_constraint = opt.NonlinearConstraint(constraint_values,
                                                            lb=-np.inf,
                                                            ub=0.0,
                                                            jac=constraint_jacobian,
                                                            keep_feasible=False
                                                            )

        # Define bounds on alpha values and z (the last element of our decision variable array).
        self.bounds = [(10**-12, 1.0) for i in range(self.n_system_decision_variables)] + [(0.0, np.inf)]

        # Set sum of alphas (not z) to 1.
        equality_constraint_array = np.ones(self.n_system_decision_variables + 1)
        equality_constraint_array[-1] = 0.0
        equality_constraint_bound = 1.0
        self.equality_constraint = opt.LinearConstraint(equality_constraint_array,
                                                        equality_constraint_bound,
                                                        equality_constraint_bound
                                                        )
        # Set up warmstart solution.
        self.set_warmstart(warm_start)

    def constraints_wrapper(self, x):
        """Wrapper to go around self.constraint().

        Notes
        -----
        scipy optimization methods don't directly support simultaneous computation
        of constraint values and their gradients. Additionally, it only considers one constraint and its gradient
        at a time, as separate functions. Thus we check whether we're looking at the same alphas
        as the last call, and if so return the same output.

        Parameters
        ----------
        x : numpy array of length n_systems + 1
            Consists of an allocation for each system and estimated convergence rate.

        Returns
        -------
        rates : numpy array
            The value of z(estimated convergence rate) minus the convergence rate upper bound associated with each constraint.
        jacobian : 2d numpy array
            The jacobian of the rates with respect to the vector alpha (including the final element z).
        """
        if all(x == self.constraints_wrapper.last_x):
            return self.constraints_wrapper.last_outputs
        else:
            rates, jacobian = self.constraints(x)
            self.constraints_wrapper.__func__.last_x = x
            self.constraints_wrapper.__func__.last_outputs = rates, jacobian
            return rates, jacobian

    def constraints(self, x):
        """Calculates MCE constraints and MCI constraints on the convergence rate and appends them together,
        where the value of each constraint is equal to z, the total rate estimator,
        minus the rate associated with each possible MCI and MCE event, each of which serves as an
        upper bound on the total rate.

        Parameters
        ----------
        x : numpy array of length n_systems + 1
            allocation for each system and estimated convergence rate

        Returns
        -------
        rates : numpy array
            The value of z(estimated convergence rate) minus the convergence rate upper bound associated with each constraint
        jacobian : 2d numy array
            The jacobian of the constraint values with respect to the vector alpha (including the final element z)
        """
        # tol = 10**-12
        # x[0:-1][x[0:-1] < tol] = 0
        # Compose MCE and MCI constraint values and gradients.
        MCE_rates, MCE_grads = self.MCE_rates(x)
        MCI_rates, MCI_grads = self.MCI_rates(x)
        rates = np.append(MCE_rates, MCI_rates, axis=0)
        grads = np.append(MCE_grads, MCI_grads, axis=0)
        return rates, grads

    def MCE_rates(self, x):
        """Calculate the MCE rate constraint values and jacobian.

        Parameters
        ----------
        x : numpy array of length n_systems + 1
            allocation for each system and estimated convergence rate

        Returns
        -------
        MCE_rates : numpy array
            The value of z (estimated convergence rate) minus the convergence rate upper bound associated with each MCE constraint
        MCE_grads : 2d numy array
            The jacobian of the MCE constraint values with respect to the vector alpha (including the final element z)
        """
        raise NotImplementedError

    def MCI_rates(self, x):
        """Calculate the MCE rate constraint values and jacobian.

        Parameters
        ----------
        x : numpy array of length n_systems + 1
            allocation for each system and estimated convergence rate

        Returns
        -------
        MCI_rates : numpy array
            The value of z (estimated convergence rate) minus the convergence rate upper bound associated with each MCI constraint
        MCI_grads: 2d numy array
            The jacobian of the MCI constraint values with respect to the vector alpha (including the final element z)
        """
        raise NotImplementedError

    def set_warmstart(self, warm_start=None):
        """Set warm-start solution.

        Parameters
        ----------
        warm_start : list of float
            An initial simulation allocation from which to determine the optimal allocation.\
            Length must be equal to the number of systems.
        """
        if warm_start is None:
            self.warm_start = np.array([1.0 / self.n_system_decision_variables] * self.n_system_decision_variables + [0])
        else:
            # Add an initial z value of 0.
            self.warm_start = np.append(warm_start, 0)

    def solve_opt_problem(self, resolve):
        """Solve optimization problem from warm-start solution.

        Parameters
        ----------
        resolve : bool
            Resolve the optimization problem if insufficiently solved if True, otherwise False.

        Returns
        -------
        alpha : tuple
            The estimated optimal simulation allocation, which is a list of float of length equal to the number of systems.
        z : float
            The estimated rate of convergence.
        """
        print(self.warm_start)
        # Solve optimization problem.
        res = opt.minimize(fun=self.objective_function,
                           x0=self.warm_start,
                           method='trust-constr',
                           jac=True,
                           hess=self.hessian_zero,
                           bounds=self.bounds,
                           constraints=[self.equality_constraint, self.nonlinear_constraint]
                           )
                           # options = {'disp': False}
                           # options = {'gtol': 10**-12, 'xtol': 10**-12, 'maxiter': 10000})

        # (Optional) If first attempt to optimize terminated improperly, warm-start at
        # final solution and try again.
        if resolve:
            if res.status == 0:
                print("cycling")
                res = opt.minimize(fun=self.objective_function,
                                x0=res.x,
                                method='trust-constr',
                                jac=True,
                                hess=self.hessian_zero,
                                bounds=self.bounds,
                                constraints=[self.equality_constraint, self.nonlinear_constraint]
                                )

        alpha, z = self.post_process(opt_sol=res.x)
        return alpha, z

    def post_process(self, opt_sol):
        """Convert solution to optimization problem into an allocation
        and a convergence rate.

        Parameters
        ----------
        opt_sol : tuple
            Optimal solution to the allocation optimization problem.

        Returns
        -------
        alpha : tuple
            The estimated optimal simulation allocation, which is a list of float of length equal to the number of systems.
        z : float
            The estimated rate of convergence.
        """
        alpha = opt_sol[0:-1]
        z = opt_sol[-1]
        return alpha, z

    def calc_rate(self, alphas):
        """Calculate the rate of an allocation given a MORS problem.
        Use the estimated rate associated with the convex optimization problem.

        Parameters
        ----------
        alphas : list of float
            An initial simulation allocation which sets the starting point for
            determining the optimal allocation. Length must be equal to the
            number of systems.

        Returns
        -------
        z : float
            Convergence rate associated with alphas.
        """
        # Append a zero for the convergence rate.
        x = np.append(alphas, 0)
        MCE_rates, _ = self.MCE_rates(x)
        MCI_rates, _ = self.MCI_rates(x)
        z = min(min(-1 * MCE_rates), min(-1 * MCI_rates))
        return z


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
    kappas : numpy list (length n_obj^n_paretos) of tuples (length n_paretos)
        each tuple indicates that an MCI event may occur if a non-pareto dominates pareto i in objective tuple[i] for all i in range(num_par)
    """
    def __init__(self, systems):
        super().__init__(systems=systems)
        # kappas is a list of tuples. Each tuple is of length n_paretos with elements
        # corresponding to objective indices.
        # To exclude a pareto, a non-pareto must dominate the pareto with number equal to the kappa index
        # along objective equal to the kappa value, for some kappa.
        v = range(self.n_obj)
        self.kappas = list(it.product(v, repeat=self.n_paretos))

    def MCE_rates(self, x):
        """Calculate the MCE rate constraint values and jacobian.

        Parameters
        ----------
        x : numpy array of length n_systems + 1
            allocation for each system and estimated convergence rate

        Returns
        -------
        MCE_rates : numpy array
            The value of z (estimated convergence rate) minus the convergence rate upper bound associated with each MCE constraint
        MCE_grads : 2d numy array
            The jacobian of the MCE constraint values with respect to the vector alpha (including the final element z)
        """
        # Negative alphas break the quadratic optimizer called below.
        # alphas that are too small may give us numerical precision issues.
        tol = 10**-12
        x[0:-1][x[0:-1] <= tol] = 0

        # There is an MCE constraint for every non-diagonal element of a (paretos x paretos) matrix.
        n_MCE = self.n_paretos * (self.n_paretos - 1)
        MCE_rates = np.zeros(n_MCE)
        MCE_grads = np.zeros([n_MCE, self.n_systems + 1])

        # Assign value of 1 to (d constraint_val) / (d z)
        MCE_grads[:, self.n_systems] = 1

        # Construct the rates and gradients.
        count = 0
        for i in self.systems['pareto_indices']:
            for j in self.systems['pareto_indices']:
                if i != j:
                    if x[i] <= tol or x[j] <= tol:
                        # It can be shown that if either alpha is zero,
                        # the rate is zero and the derivatives are zero.
                        # Constraint values are z - rate, so set "rate" here to z.
                        rate = x[-1]
                        d_rate_d_i = 0
                        d_rate_d_j = 0
                    else:
                        if self.n_obj == 2:  # 2-objective case.
                            rate, d_rate_d_i, d_rate_d_j = MCE_2d(aI=x[i],
                                                                  aJ=x[j],
                                                                  Iobj=self.systems["obj"][i],
                                                                  Isig=self.systems["var"][i],
                                                                  Jobj=self.systems["obj"][j],
                                                                  Jsig=self.systems["var"][j],
                                                                  inv_var_i=self.systems["inv_var"][i],
                                                                  inv_var_j=self.systems["inv_var"][j]
                                                                  )
                        elif self.n_obj == 3:  # 3-objective case.
                            rate, d_rate_d_i, d_rate_d_j = MCE_3d(aI=x[i],
                                                                  aJ=x[j],
                                                                  Iobj=self.systems["obj"][i],
                                                                  Isig=self.systems["var"][i],
                                                                  Jobj=self.systems["obj"][j],
                                                                  Jsig=self.systems["var"][j],
                                                                  inv_var_i=self.systems["inv_var"][i],
                                                                  inv_var_j=self.systems["inv_var"][j]
                                                                  )
                        else:
                            rate, d_rate_d_i, d_rate_d_j = MCE_four_d_plus(alpha_i=x[i],
                                                                           alpha_j=x[j],
                                                                           obj_i=self.systems["obj"][i],
                                                                           inv_var_i=self.systems["inv_var"][i],
                                                                           obj_j=self.systems["obj"][j],
                                                                           inv_var_j=self.systems["inv_var"][j],
                                                                           n_obj=self.n_obj
                                                                           )
                        rate = x[-1] - rate
                    MCE_rates[count] = rate
                    MCE_grads[count, i] = -1.0 * d_rate_d_i
                    MCE_grads[count, j] = -1.0 * d_rate_d_j
                    count = count + 1
        return MCE_rates, MCE_grads

    def MCI_rates(self, x):
        """Calculate the MCE rate constraint values and jacobian.

        Parameters
        ----------
        x : numpy array of length n_systems + 1
            allocation for each system and estimated convergence rate

        Returns
        -------
        MCI_rates : numpy array
            The value of z (estimated convergence rate) minus the convergence rate upper bound associated with each MCI constraint
        MCI_grads: 2d numy array
            The jacobian of the MCI constraint values with respect to the vector alpha (including the final element z)
        """
        tol = 10**-12
        n_kap = len(self.kappas)

        # we have one constraint for every non-pareto for every kappa vector
        n_non_par = self.n_systems - self.n_paretos
        n_MCI = n_non_par * n_kap
        MCI_rates = np.zeros(n_MCI)
        MCI_grad = np.zeros([n_MCI, self.n_systems + 1])
        MCI_grad[:, self.n_systems] = 1

        # Set alphas of  systems to zero if they are less than tol.
        # However, zeros break the optimizer, instead use tiny value.
        x[0:-1][x[0:-1] <= tol] = tol

        count = 0
        for j in self.systems['non_pareto_indices']:
            obj_j = self.systems['obj'][j]
            inv_var_j = self.systems['inv_var'][j]
            # TODO See if this actually belongs. We do this in the phantom and it
            # keeps the quadratic optimization from breaking if we don't set zero alphas to tol.
            for kap in self.kappas:
                if False:  # alphas[j] <= tol:
                    # The rate and gradients are zero, so we only have to worry about gradient wrt z since
                    # we initialize with zero.
                    # MCI_grad[count,-1] = 1
                    # ?
                    None
                else:
                    # Initialize objectives and variances.
                    relevant_objectives = np.zeros(self.n_paretos)
                    relevant_variances = np.zeros(self.n_paretos)

                    for p in range(self.n_paretos):
                        # Get the actual index of the pareto system.
                        pareto_system_ind = self.systems['pareto_indices'][p]
                        # Extract variances and objective values.
                        relevant_objectives[p] = self.systems['obj'][pareto_system_ind][kap[p]]
                        relevant_variances[p] = self.systems['var'][pareto_system_ind][kap[p], kap[p]]
                    # Get the alpha of the pareto system.
                    pareto_alphas = x[self.systems['pareto_indices']]

                    # Quadratic optimization step.
                    # Setup.
                    P = linalg.block_diag(x[j] * inv_var_j, np.diag(pareto_alphas * (1 / relevant_variances)))
                    q = matrix(-1 * np.append(x[j] * inv_var_j @ obj_j,
                                              pareto_alphas * 1 / relevant_variances * relevant_objectives))
                    G_left_side = np.zeros([self.n_paretos, self.n_obj])
                    G_left_side[range(self.n_paretos), kap] = 1
                    G = matrix(np.append(G_left_side, -1 * np.identity(self.n_paretos), axis=1))
                    h = matrix(np.zeros(self.n_paretos))
                    P = matrix((P + P.transpose()) / 2)

                    # Solve.
                    solvers.options['show_progress'] = False
                    x_star = np.array(solvers.qp(P, q, G, h)['x']).flatten()

                    # Reformat results of optimization.
                    rate = 0.5 * x[j] * (obj_j - x_star[0:self.n_obj]) @ inv_var_j @ (obj_j - x_star[0:self.n_obj]) +\
                        0.5 * np.sum(pareto_alphas * (x_star[self.n_obj:] - relevant_objectives) * (1 / relevant_variances) * (x_star[self.n_obj:] - relevant_objectives))
                    MCI_grad[count, j] = -1.0 * 0.5 * (obj_j - x_star[0:self.n_obj]) @ inv_var_j @ (obj_j - x_star[0:self.n_obj])
                    MCI_grad[count, self.systems['pareto_indices']] = -1.0 * 0.5 * (x_star[self.n_obj:] - relevant_objectives) * (1 / relevant_variances) * (x_star[self.n_obj:] - relevant_objectives)
                    MCI_rates[count] = x[-1] - rate

                count = count + 1

        return MCI_rates, MCI_grad


class Phantom(BruteForce):
    """Class for phantom allocation algorithm.

    Notes
    -----
    MCE_rates() method is inherited from BruteForce class.

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
        super().__init__(systems=systems)

        # Get array of pareto values for the phantom finder.
        pareto_array = np.zeros([self.n_paretos, self.n_obj])
        for i in range(self.n_paretos):
            pareto_array[i, :] = systems['obj'][systems['pareto_indices'][i]]
        # Get the phantom systems.
        phantom_values = find_phantoms(pareto_array, self.n_obj)

        # Sort the phantom system.
        for i in range(self.n_obj):
            phantom_values = phantom_values[(phantom_values[:, self.n_obj - 1 - i]).argsort(kind='mergesort')]
        self.n_phantoms = len(phantom_values)

        # The commented part doesn't give different results, but this makes the constraint
        # ordering identical to that of the matlab code, which you'll want for debugging
        # phantom_values = phantom_values[(phantom_values[:,0]).argsort()]

        # TODO: consider using something other than inf as a placeholder.
        # Unfortunately, inf is a float in numpy, and arrays must be homogenous
        # and floats don't automatically cast to ints for indexing leading to an error.
        # Right now we're casting as ints for indexing, but that's a little gross.
        # Also, inf doesn't cast to intmax if you cast as int.
        # It ends up being very negative for some reason.
        self.phantoms = np.ones([self.n_phantoms, self.n_obj]) * np.inf

        # TODO: Vectorize?
        for i in range(self.n_phantoms):
            for j in range(self.n_obj):
                for k in range(self.n_paretos):
                    if pareto_array[k, j] == phantom_values[i, j]:
                        self.phantoms[i, j] = systems['pareto_indices'][k]

    def MCI_rates(self, x):
        """Calculate the MCE rate constraint values and jacobian.

        Parameters
        ----------
        x : numpy array of length n_systems + 1
            allocation for each system and estimated convergence rate

        Returns
        -------
        MCI_rates : numpy array
            The value of z (estimated convergence rate) minus the convergence rate upper bound associated with each MCI constraint
        MCI_grads: 2d numy array
            The jacobian of the MCI constraint values with respect to the vector alpha (including the final element z)
        """
        tol = 10**-12
        n_nonparetos = self.n_systems - self.n_paretos
        n_MCI = n_nonparetos * self.n_phantoms

        MCI_rates = np.zeros(n_MCI)
        MCI_grads = np.zeros([n_MCI, self.n_systems + 1])

        count = 0
        x[0:-1][x[0:-1] <= tol] = 0
        for j in self.systems['non_pareto_indices']:
            for m in range(self.n_phantoms):
                # Get the pareto indices corresponding to phantom l.
                phantom_indices = self.phantoms[m, :]
                if x[j] <= tol:
                    # The rate and gradients are zero. Only have to worry about gradient
                    # wrt z since we initialize with zero.
                    MCI_grads[count, -1] = 1
                else:
                    phantom_obj = np.zeros(self.n_obj)
                    phantom_var = np.zeros(self.n_obj)
                    phantom_alphas = np.zeros(self.n_obj)
                    phantom_objectives = np.array(range(self.n_obj))
                    phantom_objective_count = self.n_obj
                    alpha_zeros = 0

                    # Extract objective and variance values for the phantom pareto system.
                    for b in range(self.n_obj):
                        if phantom_indices[b] < np.inf:
                            pareto_system = int(phantom_indices[b])
                            phantom_obj[b] = self.systems['obj'][pareto_system][b]
                            phantom_var[b] = self.systems['var'][pareto_system][b, b]
                            if x[pareto_system] <= tol:
                                phantom_alphas[b] = 0
                                alpha_zeros = alpha_zeros + 1
                            else:
                                phantom_alphas[b] = x[pareto_system]
                        else:
                            phantom_objective_count -= 1

                    # Keep track of which objectives are included in phantom set.
                    phantom_objectives = phantom_objectives[phantom_indices < np.inf]
                    obj_j = self.systems['obj'][j][phantom_objectives]
                    # Only want covariances for the phantom objectives.
                    # np.ix_ allows us to subset nicely that way.
                    cov_j = self.systems['var'][j][np.ix_(phantom_objectives, phantom_objectives)]

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
                        MCI_rates[count] = x[-1] - rate
                    else:
                        length = len(phantom_objectives)
                        if length == 1:
                            rate, grad_j, phantom_grads = MCI_1d(x[j], obj_j, cov_j, phantom_alphas, phantom_obj, phantom_var)
                        elif length == 2:
                            rate, grad_j, phantom_grads = MCI_2d(x[j], obj_j, cov_j, phantom_alphas, phantom_obj, phantom_var)
                        elif length == 3:
                            rate, grad_j, phantom_grads = MCI_3d(x[j], obj_j, cov_j, phantom_alphas, phantom_obj, phantom_var)
                        else:
                            rate, grad_j, phantom_grads = MCI_four_d_plus(x[j], obj_j, cov_j, phantom_alphas, phantom_obj, phantom_var)

                        # TODO: Hard-code solutions for 1-3 objectives.
                        MCI_grads[count, phantom_indices[phantom_indices < np.inf].astype(int)] = -1.0 * phantom_grads
                        MCI_grads[count, -1] = 1
                        MCI_grads[count, j] = -1.0 * grad_j
                        MCI_rates[count] = x[-1] - rate

                count = count + 1
        return MCI_rates, MCI_grads


class MOSCORE(Phantom):
    """Class for MOSCORE allocation algorithm.

    Notes
    -----
    Much of __init__() method is inherited from Phantom class.

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
        super().__init__(systems=systems)

        # Note: This next part is repeated from the __init__ of the Phantom subclass.
        # The only difference is one line in the creation of the phantoms matrix.

        # Get array of pareto values for the phantom finder.
        pareto_array = np.zeros([self.n_paretos, self.n_obj])
        for i in range(self.n_paretos):
            pareto_array[i, :] = systems['obj'][systems['pareto_indices'][i]]
        # Get the phantom systems.
        phantom_values = find_phantoms(pareto_array, self.n_obj)

        # Sort the phantom system.
        for i in range(self.n_obj):
            phantom_values = phantom_values[(phantom_values[:, self.n_obj - 1 - i]).argsort(kind='mergesort')]
        self.n_phantoms = len(phantom_values)

        # The commented part doesn't give different results, but this makes the constraint
        # ordering identical to that of the matlab code, which you'll want for debugging
        # phantom_values = phantom_values[(phantom_values[:,0]).argsort()]

        # TODO: consider using something other than inf as a placeholder.
        # Unfortunately, inf is a float in numpy, and arrays must be homogenous
        # and floats don't automatically cast to ints for indexing leading to an error.
        # Right now we're casting as ints for indexing, but that's a little gross.
        # Also, inf doesn't cast to intmax if you cast as int.
        # It ends up being very negative for some reason.
        self.phantoms = np.ones([self.n_phantoms, self.n_obj]) * np.inf

        # TODO: Vectorize?
        for i in range(self.n_phantoms):
            for j in range(self.n_obj):
                for k in range(self.n_paretos):
                    if pareto_array[k, j] == phantom_values[i, j]:
                        self.phantoms[i, j] = k
                        # The previous line is different from the
                        # phantom subclass.

        # Specify number of allocation decision variables.
        self.n_system_decision_variables = self.n_paretos + 1
        # Calculate j_star, lambda, and M_star.
        self.j_star, self.lambdas = self.calc_SCORE()
        self.M_star = self.calc_SCORE_MCE()

    def set_warmstart(self, warm_start=None):
        """Set warm-start solution.

        Parameters
        ----------
        warm_start : list of float
            An initial simulation allocation from which to determine the optimal allocation.\
            Length must be equal to the number of systems.
        """
        if warm_start is None:
            self.warm_start = np.array([1 / (2 * self.n_paretos)] * self.n_paretos + [0.5] + [0])
        else:
            self.warm_start = np.append(warm_start[0:self.n_paretos], [sum(warm_start[self.n_paretos:]), 0])

    def post_process(self, opt_sol):
        """Convert solution to optimization problem into an allocation
        and a convergence rate.

        Parameters
        ----------
        opt_sol : tuple
            Optimal solution to the allocation optimization problem.

        Returns
        -------
        alpha : tuple
            The estimated optimal simulation allocation, which is a list of float of length equal to the number of systems.
        z : float
            The estimated rate of convergence.
        """
        # Sort output by system number.
        alpha = np.zeros(self.n_systems)
        alpha[self.systems['pareto_indices']] = opt_sol[0:self.n_paretos]
        for j in range(self.n_systems - self.n_paretos):
            alpha[self.systems['non_pareto_indices'][j]] = self.lambdas[j] * opt_sol[-2]
        z = opt_sol[-1]
        return alpha, z

    def calc_SCORE(self):
        """Calculates the SCORE for MCI constraints.

        Returns
        -------
        j_star : numpy matrix
            ???
        lambdas : numpy array
            ???
        """
        scores = np.zeros([self.n_phantoms, self.n_systems - self.n_paretos])
        j_star = np.zeros([self.n_phantoms * self.n_obj, 2])
        # Note: the matlab code pre-computes several vectors for speed here
        # which I instead initialize individually.
        # This is because initializing vector v at this point, setting v_instance = v
        # and then modifying v_instance would modify v, which is undesireable.

        # Loop over phantoms.
        for i in range(self.n_phantoms):

            # phantom_pareto_inds refers to the pareto system number from which each phantom objective
            # gets its value. We drop objectives that are infinity in the phantom
            # and keep track of the rest of the objectives in objectives_playing.
            phantom_pareto_nums = self.phantoms[i, :]
            objectives_playing = np.array(range(self.n_obj))[phantom_pareto_nums < np.inf]
            phantom_pareto_nums = phantom_pareto_nums[phantom_pareto_nums < np.inf]
            # pareto_phantom_inds refers to the actual system indices.

            n_obj_playing = len(objectives_playing)

            phantom_objectives = np.zeros(n_obj_playing)
            # Extract the objectives for the phantoms.
            for obj in range(n_obj_playing):
                phantom_pareto_ind = self.systems['pareto_indices'][int(phantom_pareto_nums[obj])]
                phantom_objectives[obj] = self.systems['obj'][phantom_pareto_ind][objectives_playing[obj]]

            j_comps = np.ones(self.n_obj) * np.inf
            j_indices = np.ones(self.n_obj) * np.inf
            size = len(objectives_playing)
            for j in range(self.n_systems - self.n_paretos):
                j_ind = self.systems['non_pareto_indices'][j]
                obj_j = self.systems['obj'][j_ind][objectives_playing]
                cov_j = self.systems['var'][j_ind][np.ix_(objectives_playing, objectives_playing)]

                # TODO: Hard code solutions for 1, 2, 3 objectives.
                if size == 1:
                    score, binds = SCORE_1d(phantom_objectives, obj_j, cov_j)
                elif size == 2:
                    score, binds = SCORE_2d(phantom_objectives, obj_j, cov_j)
                elif size == 3:
                    score, binds = SCORE_3d(phantom_objectives, obj_j, cov_j)
                else:
                    score, binds = score_four_d_plus(phantom_objectives, obj_j, cov_j)

                j_current = np.ones(self.n_obj) * np.inf
                j_current[objectives_playing] = binds
                scores[i, j] = score

                for m in range(self.n_obj):
                    if j_current[m] < j_comps[m]:
                        j_comps[m] = j_current[m]
                        j_indices[m] = j

            L_indices = np.ones(self.n_obj) * i

            # For every constraint (with n_obj rows in J_star), we want the non-pareto index
            # per objective and the phantom index
            # TODO: consider instead having one row per constraint, one column per objective
            # and separate matrices (arrays) for J indices and L indices. Or actually we wouldn't
            # need L indices I think because the J matrix would then be ordered as such.

            j_star[self.n_obj * i:self.n_obj * (i + 1), :] = np.vstack((j_indices, L_indices)).T

        # inv_scores is the inverse of the minimum of each column in scores, resulting in one
        # value per non-pareto system.
        inv_scores = 1 / np.minimum.reduce(scores)
        lambdas = inv_scores / sum(inv_scores)

        # TODO: Not sure why we're doing this, but we remove the rows of J_star where
        # the first column is infinity.
        j_star = j_star[j_star[:, 0] < np.inf, :]
        j_star = np.unique(j_star, axis=0)

        return j_star, lambdas

    def calc_SCORE_MCE(self):
        """Calculate the SCORE for MCE constraints.

        Returns
        -------
        M_star : numpy matrix
            ???
        """
        scores = np.zeros([self.n_paretos, self.n_paretos])
        all_scores = np.zeros(self.n_paretos * (self.n_paretos - 1))
        M_star = np.zeros([self.n_paretos * self.n_obj, 2])

        count = 0
        for i in range(self.n_paretos):
            i_ind = self.systems['pareto_indices'][i]
            obj_i = self.systems['obj'][i_ind]
            j_comps = np.ones(self.n_obj) * np.inf
            j_inds = np.ones(self.n_obj) * np.inf

            for j in range(self.n_paretos):
                if i != j:
                    j_ind = self.systems['pareto_indices'][j]
                    obj_j = self.systems['obj'][j_ind]
                    cov_j = self.systems['var'][j_ind]

                    # TODO: Hard code solutions for <4 objectives.
                    if self.n_obj == 1:
                        score, binds = SCORE_1d(obj_i, obj_j, cov_j)
                    elif self.n_obj == 2:
                        score, binds = SCORE_2d(obj_i, obj_j, cov_j)
                    elif self.n_obj == 3:
                        score, binds = SCORE_3d(obj_i, obj_j, cov_j)
                    else:
                        score, binds = score, binds = score_four_d_plus(obj_i, obj_j, cov_j)
                    scores[i, j] = score

                    all_scores[count] = score

                    count = count + 1

                    j_current = binds
                    # TODO: Vectorize?
                    for m in range(self.n_obj):
                        if j_current[m] < j_comps[m]:
                            j_comps[m] = j_current[m]
                            j_inds[m] = j

            L_inds = np.ones(self.n_obj) * i
            M_star[self.n_obj * i:self.n_obj * (i + 1), :] = np.vstack((j_inds, L_inds)).T

        # TODO: Not sure why we're doing this, but we remove the rows of M_star where
        # the first column is infinity.
        M_star = M_star[M_star[:, 0] < np.inf, :]
        # Switch columns and append.
        M_star_b = M_star[:, [1, 0]]
        M_star = np.append(M_star, M_star_b, axis=0)

        # Add pairs of systems where SCORE < percentile (all scores).
        score_percentile = np.percentile(all_scores, 25)

        for a in range(self.n_paretos):
            for b in range(self.n_paretos):
                if a != b and scores[a, b] <= score_percentile:
                    M_star = np.append(M_star, [[a, b]], axis=0)
                    M_star = np.append(M_star, [[b, a]], axis=0)

        M_star = np.unique(M_star, axis=0)
        return M_star

    def MCI_rates(self, x):
        """Calculate the MCE rate constraint values and jacobian.

        Parameters
        ----------
        x : numpy array of length n_systems + 1
            allocation for each system and estimated convergence rate

        Returns
        -------
        MCI_rates : numpy array
            The value of z (estimated convergence rate) minus the convergence rate upper bound associated with each MCI constraint
        MCI_grads: 2d numy array
            The jacobian of the MCI constraint values with respect to the vector alpha (including the final element z)
        """
        tol = 10**-12

        n_MCI = len(self.j_star)
        MCI_rates = np.zeros(n_MCI)
        MCI_grads = np.zeros([n_MCI, len(x)])

        for i in range(n_MCI):
            j = int(self.j_star[i, 0])
            j_ind = self.systems['non_pareto_indices'][j]
            lambda_j = self.lambdas[j]
            alpha_j = lambda_j * x[-2]
            obj_j = self.systems['obj'][j_ind]
            cov_j = self.systems['var'][j_ind]

            phantom_ind = int(self.j_star[i, 1])
            phantom_pareto_inds = self.phantoms[phantom_ind, :]

            if alpha_j < tol:
                # Rate is 0. Returned rate is z. Grads wrt anything but z are left as zero.
                MCI_rates[i] = x[-1]
                MCI_grads[i, -1] = 1
            else:
                phantom_objectives = np.zeros(self.n_obj)
                phantom_vars = np.zeros(self.n_obj)
                phantom_alphas = np.zeros(self.n_obj)
                objectives_playing = np.array(range(self.n_obj))
                alpha_zeros = 0
                n_objectives_playing = self.n_obj

                for b in range(self.n_obj):
                    if phantom_pareto_inds[b] < np.inf:
                        pareto_system_num = int(phantom_pareto_inds[b])
                        pareto_system_ind = self.systems['pareto_indices'][pareto_system_num]
                        phantom_objectives[b] = self.systems['obj'][pareto_system_ind][b]
                        phantom_vars[b] = self.systems['var'][pareto_system_ind][b, b]
                        if x[pareto_system_num] < tol:
                            phantom_alphas[b] = 0
                            alpha_zeros += 1
                        else:
                            phantom_alphas[b] = x[pareto_system_num]
                    else:
                        n_objectives_playing -= 1

                objectives_playing = objectives_playing[phantom_pareto_inds < np.inf]

                obj_j = obj_j[objectives_playing]
                cov_j = cov_j[np.ix_(objectives_playing, objectives_playing)]

                phantom_objectives = phantom_objectives[objectives_playing]
                phantom_vars = phantom_vars[objectives_playing]
                phantom_alphas = phantom_alphas[objectives_playing]

                if alpha_zeros == n_objectives_playing:
                    MCI_rates[i] = x[-1]
                    MCI_grads[i, phantom_pareto_inds[phantom_pareto_inds < np.inf].astype(int)] = -0.5 * ((obj_j - phantom_objectives) ** 2) / phantom_vars
                    MCI_grads[i, -1] = 1
                else:
                    # TODO: Hard code solutions for < 4 objectives.
                    length = len(objectives_playing)
                    if length == 1:
                        rate, grad_j, phantom_grads = MCI_1d(alpha_j, obj_j, cov_j, phantom_alphas, phantom_objectives, phantom_vars)
                    elif length == 2:
                        rate, grad_j, phantom_grads = MCI_2d(alpha_j, obj_j, cov_j, phantom_alphas, phantom_objectives, phantom_vars)
                    elif length == 3:
                        rate, grad_j, phantom_grads = MCI_3d(alpha_j, obj_j, cov_j, phantom_alphas, phantom_objectives, phantom_vars)
                    else:
                        rate, grad_j, phantom_grads = MCI_four_d_plus(alpha_j, obj_j, cov_j, phantom_alphas, phantom_objectives, phantom_vars)

                    MCI_rates[i] = x[-1] - rate
                    phantom_grads[phantom_grads < tol] = 0
                    MCI_grads[i, phantom_pareto_inds[phantom_pareto_inds < np.inf].astype(int)] = -1.0 * phantom_grads
                    MCI_grads[i, -2] = -1 * lambda_j * grad_j
                    MCI_grads[i, -1] = 1

        return MCI_rates, MCI_grads

    def MCE_rates(self, x):
        """Calculate the MCE rate constraint values and jacobian.

        Parameters
        ----------
        x : numpy array of length n_systems + 1
            allocation for each system and estimated convergence rate

        Returns
        -------
        MCE_rates : numpy array
            The value of z (estimated convergence rate) minus the convergence rate upper bound associated with each MCE constraint
        MCE_grads : 2d numy array
            The jacobian of the MCE constraint values with respect to the vector alpha (including the final element z)
        """
        tol = 10**-12
        n_MCE = len(self.M_star)
        MCE_rates = np.zeros(n_MCE)
        MCE_grads = np.zeros([n_MCE, len(x)])

        for k in range(n_MCE):
            i = int(self.M_star[k, 0])
            j = int(self.M_star[k, 1])

            if (x[i] < tol or x[j] < tol):
                rate = x[-1]
                grad_i = 0
                grad_j = 0
            else:
                i_ind = self.systems['pareto_indices'][i]
                j_ind = self.systems['pareto_indices'][j]
                obj_i = self.systems['obj'][i_ind]
                inv_cov_i = self.systems['inv_var'][i_ind]
                obj_j = self.systems['obj'][j_ind]
                inv_cov_j = self.systems['inv_var'][j_ind]

                # TODO: Hard code solutions for <4 dimensions.
                if self.n_obj == 2:
                    rate, grad_i, grad_j = MCE_2d(aI=x[i],
                                                  aJ=x[j],
                                                  Iobj=obj_i,
                                                  Isig=self.systems["var"][i_ind],
                                                  Jobj=obj_j,
                                                  Jsig=self.systems["var"][j_ind],
                                                  inv_var_i=inv_cov_i,
                                                  inv_var_j=inv_cov_j
                                                  )
                elif self.n_obj == 3:
                    rate, grad_i, grad_j = MCE_3d(aI=x[i],
                                                  aJ=x[j],
                                                  Iobj=obj_i,
                                                  Isig=self.systems["var"][i_ind],
                                                  Jobj=obj_j,
                                                  Jsig=self.systems["var"][j_ind],
                                                  inv_var_i=inv_cov_i,
                                                  inv_var_j=inv_cov_j
                                                  )
                else:
                    rate, grad_i, grad_j = MCE_four_d_plus(alpha_i=x[i],
                                                           alpha_j=x[j],
                                                           obj_i=obj_i,
                                                           inv_var_i=inv_cov_i,
                                                           obj_j=obj_j,
                                                           inv_var_j=inv_cov_j,
                                                           n_obj=self.n_obj
                                                           )
                rate = x[-1] - rate

            MCE_rates[k] = rate
            MCE_grads[k, i] = -1 * grad_i
            MCE_grads[k, j] = -1 * grad_j
            MCE_grads[k, -1] = 1.0

        return MCE_rates, MCE_grads


class IMOSCORE(MOSCORE):
    """Class for iMOSCORE allocation algorithm.

    Notes
    -----
    __init__() and set_warmstart() are inherited from SCORE.

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
        super().__init__(systems=systems)
        # Specify number of allocation decision variables.
        self.n_system_decision_variables = self.n_paretos + 1
        # Calculate j_star, lambda, and M_star.
        self.j_star, self.lambdas = self.calc_iSCORE()
        self.M_star = self.calc_iSCORE_MCE()

    def calc_iSCORE(self):
        """Calculates the SCORE for MCI constraints.

        Returns
        -------
        j_star : numpy matrix
            ???
        lambdas : numpy array
            ???
        """
        scores = np.zeros([self.n_phantoms, self.n_systems - self.n_paretos])
        j_star = np.zeros([self.n_phantoms * self.n_obj, 2])

        for i in range(self.n_phantoms):
            phantom_indices = self.phantoms[i, :]
            phantom_objs = np.ones(self.n_obj) * np.inf

            for b in range(self.n_obj):
                if phantom_indices[b] < np.inf:
                    pareto_num = int(phantom_indices[b])
                    pareto_system = self.systems['pareto_indices'][pareto_num]
                    phantom_objs[b] = self.systems['obj'][pareto_system][b]

            j_comps = np.ones(self.n_obj) * np.inf
            j_inds = np.ones(self.n_obj) * np.inf

            for j in range(self.n_systems - self.n_paretos):
                j_ind = self.systems['non_pareto_indices'][j]
                obj_j = self.systems['obj'][j_ind]
                cov_j = self.systems['var'][j_ind]

                score = 0
                binds = np.ones(self.n_obj) * np.inf
                for m in range(self.n_obj):
                    if obj_j[m] > phantom_objs[m]:
                        score = score + (phantom_objs[m] - obj_j[m])**2 / (2 * cov_j[m, m])
                        binds[m] = 1

                j_current = binds * score
                scores[i, j] = score

                # Determine if this non-pareto is closer than another
                # TODO: vectorize.
                for m in range(self.n_obj):
                    if j_current[m] < j_comps[m]:
                        j_comps[m] = j_current[m]
                        j_inds[m] = j

            L_indices = np.ones(self.n_obj) * i
            j_star[i * self.n_obj:(i + 1) * self.n_obj, :] = np.vstack((j_inds, L_indices)).T

        inv_scores = 1 / np.minimum.reduce(scores)
        lambdas = inv_scores / sum(inv_scores)

        j_star = j_star[j_star[:, 0] < np.inf, :]
        j_star = np.unique(j_star, axis=0)

        return j_star, lambdas

    def calc_iSCORE_MCE(self):
        """Calculates the iSCORE for MCE constraints.

        Returns
        -------
        M_star : a numpy matrix
            ???
        """
        scores = np.zeros([self.n_paretos, self.n_paretos])
        all_scores = np.zeros(self.n_paretos * (self.n_paretos - 1))
        M_star = np.zeros([self.n_paretos * self.n_obj, 2])

        count = 0

        for i in range(self.n_paretos):
            i_ind = self.systems['pareto_indices'][i]
            obj_i = self.systems['obj'][i_ind]

            j_comps = np.ones(self.n_obj) * np.inf
            j_inds = np.ones(self.n_obj) * np.inf

            for j in range(self.n_paretos):
                if i != j:
                    j_ind = self.systems['pareto_indices'][j]
                    obj_j = self.systems['obj'][j_ind]
                    cov_j = self.systems['var'][j_ind]

                    score = 0
                    binds = np.ones(self.n_obj) * np.inf
                    for m in range(self.n_obj):
                        if obj_j[m] > obj_i[m]:
                            score = score + (obj_i[m] - obj_j[m])**2 / (2 * cov_j[m, m])
                            binds[m] = 1

                    j_current = binds * score
                    scores[i, j] = score
                    all_scores[count] = score

                    count = count + 1
                    # TODO: vectorize?
                    for m in range(self.n_obj):
                        if j_current[m] < j_comps[m]:
                            j_comps[m] = j_current[m]
                            j_inds[m] = j

            L_inds = np.ones(self.n_obj) * i
            M_star[self.n_obj * i:self.n_obj * (i + 1), :] = np.vstack((j_inds, L_inds)).T

        # TODO not sure why we're doing this, but we remove the rows of M_star where
        # the first column is infinity.
        M_star = M_star[M_star[:, 0] < np.inf, :]
        # Switch columns and append.
        M_star_b = M_star[:, [1, 0]]
        M_star = np.append(M_star, M_star_b, axis=0)

        # Add pairs of systems where SCORE < percentile(all scores).
        score_percentile = np.percentile(all_scores, 25)

        for a in range(self.n_paretos):
            for b in range(self.n_paretos):
                if a != b and scores[a, b] <= score_percentile:

                    M_star = np.append(M_star, [[a, b]], axis=0)
                    M_star = np.append(M_star, [[b, a]], axis=0)

        M_star = np.unique(M_star, axis=0)
        return M_star

    def MCI_rates(self, x):
        """Calculate the MCE rate constraint values and jacobian.

        Parameters
        ----------
        x : numpy array of length n_systems + 1
            allocation for each system and estimated convergence rate

        Returns
        -------
        MCI_rates : numpy array
            The value of z (estimated convergence rate) minus the convergence rate upper bound associated with each MCI constraint
        MCI_grads: 2d numy array
            The jacobian of the MCI constraint values with respect to the vector alpha (including the final element z)
        """
        tol = 10**-50
        n_MCI = len(self.j_star)
        MCI_rates = np.zeros(n_MCI)
        MCI_grads = np.zeros([n_MCI, len(x)])

        for i in range(n_MCI):
            j = int(self.j_star[i, 0])
            j_ind = self.systems['non_pareto_indices'][j]
            lambda_j = self.lambdas[j]
            alpha_j = lambda_j * x[-2]

            obj_j = self.systems['obj'][j_ind]
            cov_j = self.systems['var'][j_ind]

            phantom_ind = int(self.j_star[i, 1])
            phantom_pareto_nums = self.phantoms[phantom_ind, :]

            if alpha_j < tol:
                MCI_rates[i] = x[-1]
                MCI_grads[i, -1] = 1
                # All the grads aside from z are zero here, already zero from initialization
            else:
                phantom_objectives = np.ones(self.n_obj) * np.inf
                phantom_vars = np.zeros(self.n_obj)
                phantom_alphas = np.zeros(self.n_obj)

                for b in range(self.n_obj):
                    if phantom_pareto_nums[b] < np.inf:
                        phantom_pareto_num = int(phantom_pareto_nums[b])
                        phantom_pareto_ind = self.systems['pareto_indices'][phantom_pareto_num]
                        phantom_objectives[b] = self.systems['obj'][phantom_pareto_ind][b]
                        phantom_vars[b] = self.systems['var'][phantom_pareto_ind][b, b]
                        phantom_alphas[b] = x[phantom_pareto_num]

                rate = 0
                grad_j = 0
                for m in range(self.n_obj):
                    if obj_j[m] > phantom_objectives[m]:
                        rate = rate + (phantom_alphas[m] * alpha_j * (phantom_objectives[m] - obj_j[m])**2) / (2 * (alpha_j * phantom_vars[m] + phantom_alphas[m] * cov_j[m, m]))
                        grad_j = grad_j + (phantom_alphas[m]**2 * cov_j[m, m] * (phantom_objectives[m] - obj_j[m])**2) / (2 * (alpha_j * phantom_vars[m] + phantom_alphas[m] * cov_j[m, m])**2)
                        grad = (alpha_j**2 * phantom_vars[m] * (phantom_objectives[m] - obj_j[m])**2) / (2 * (alpha_j * phantom_vars[m] + phantom_alphas[m] * cov_j[m, m])**2)
                        MCI_grads[i, int(phantom_pareto_nums[m])] = -1.0 * grad

                MCI_rates[i] = x[-1] - rate
                MCI_grads[i, -1] = 1
                MCI_grads[i, -2] = -1.0 * lambda_j * grad_j

        return MCI_rates, MCI_grads

    def MCE_rates(self, x):
        """Calculate the MCE rate constraint values and jacobian.

        Parameters
        ----------
        x : numpy array of length n_systems + 1
            allocation for each system and estimated convergence rate

        Returns
        -------
        MCE_rates : numpy array
            The value of z (estimated convergence rate) minus the convergence rate upper bound associated with each MCE constraint
        MCE_grads : 2d numy array
            The jacobian of the MCE constraint values with respect to the vector alpha (including the final element z)
        """
        tol = 10**-50
        n_MCE = len(self.M_star)
        MCE_rates = np.zeros(n_MCE)
        MCE_grads = np.zeros([n_MCE, len(x)])

        for k in range(n_MCE):
            i = int(self.M_star[k, 0])
            j = int(self.M_star[k, 1])

            if (x[i] < tol or x[j] < tol):
                rate = x[-1]
                grad_i = 0
                grad_j = 0
            else:
                i_ind = self.systems['pareto_indices'][i]
                j_ind = self.systems['pareto_indices'][j]
                obj_i = self.systems['obj'][i_ind]
                cov_i = self.systems['var'][i_ind]
                obj_j = self.systems['obj'][j_ind]
                cov_j = self.systems['var'][j_ind]

                rate = 0
                grad_i = 0
                grad_j = 0

                # TODO: vectorize.
                for m in range(self.n_obj):
                    if obj_i[m] > obj_j[m]:
                        rate = rate + (x[i] * x[j] * (obj_i[m] - obj_j[m])**2) / (2 * (x[j] * cov_i[m, m] + x[i] * cov_j[m, m]))
                        grad_i = grad_i + (x[j]**2 * cov_i[m, m] * (obj_i[m] - obj_j[m])**2) / (2 * (x[j] * cov_i[m, m] + x[i] * cov_j[m, m])**2)
                        grad_j = grad_j + (x[i]**2 * cov_j[m, m] * (obj_i[m] - obj_j[m])**2) / (2 * (x[j] * cov_i[m, m] + x[i] * cov_j[m, m])**2)

                rate = x[-1] - rate

            MCE_rates[k] = rate
            MCE_grads[k, i] = -1 * grad_i
            MCE_grads[k, j] = -1 * grad_j
            MCE_grads[k, -1] = 1.0
        return MCE_rates, MCE_grads


def calc_brute_force_rate(alphas, systems):
    """Calculate the brute force rate of an allocation given a MORS problem.

    Parameters
    ----------
    alphas : list of float
        An initial simulation allocation which sets the starting point for
        determining the optimal allocation. Length must be equal to the
        number of systems.

    systems : dict
        ``"obj"``:
        is a dictionary of numpy arrays, indexed by system number, each of which corresponds to the objective values of a system

        ``"var"``:
        is a dictionary of 2d numpy arrays, indexed by system number, each of which corresponds to the covariance matrix of a system

        ``"inv_var"``:
        is a dictionary of 2d numpy, indexed by system number, each of which corresponds to the inverse covariance matrix of a system

        ``"pareto_indices"``:
        is a list of pareto systems ordered by the first objective

        ``"non_pareto_indices"``:
        is a list of non-pareto systems ordered by the first objective

    Returns
    -------
    z : float
        Brute force convergence rate associated with alphas.
    """
    bf_problem = BruteForce(systems=systems)
    z = bf_problem.calc_rate(alphas=alphas)
    return z


def calc_phantom_rate(alphas, systems):
    """Calculate the phantom rate of an allocation given a MORS problem.

    Parameters
    ----------
    alphas : list of float
        An initial simulation allocation which sets the starting point for
        determining the optimal allocation. Length must be equal to the
        number of systems.

    systems : dict
        ``"obj"``:
        is a dictionary of numpy arrays, indexed by system number, each of which corresponds to the objective values of a system

        ``"var"``:
        is a dictionary of 2d numpy arrays, indexed by system number, each of which corresponds to the covariance matrix of a system

        ``"inv_var"``:
        is a dictionary of 2d numpy, indexed by system number, each of which corresponds to the inverse covariance matrix of a system

        ``"pareto_indices"``:
        is a list of pareto systems ordered by the first objective

        ``"non_pareto_indices"``:
        is a list of non-pareto systems ordered by the first objective

    Returns
    -------
    z : float
        Phantom convergence rate associated with alphas.
    """
    phantom_problem = Phantom(systems=systems)
    z = phantom_problem.calc_rate(alphas=alphas)
    return z


def calc_score_rate(alphas, systems):
    """Calculate the SCORE rate of an allocation given a MORS problem.

    Parameters
    ----------
    alphas : list of float
        An initial simulation allocation which sets the starting point for
        determining the optimal allocation. Length must be equal to the
        number of systems.

    systems : dict
        ``"obj"``:
        is a dictionary of numpy arrays, indexed by system number, each of which corresponds to the objective values of a system

        ``"var"``:
        is a dictionary of 2d numpy arrays, indexed by system number, each of which corresponds to the covariance matrix of a system

        ``"inv_var"``:
        is a dictionary of 2d numpy, indexed by system number, each of which corresponds to the inverse covariance matrix of a system

        ``"pareto_indices"``:
        is a list of pareto systems ordered by the first objective

        ``"non_pareto_indices"``:
        is a list of non-pareto systems ordered by the first objective

    Returns
    -------
    z : float
        SCORE convergence rate associated with alphas.
    """
    score_problem = MOSCORE(systems=systems)
    z = score_problem.calc_rate(alphas=alphas)
    return z


def calc_iscore_rate(alphas, systems):
    """Calculate the IMOSCORE rate of an allocation given a MORS problem.

    Parameters
    ----------
    alphas : list of float
        An initial simulation allocation which sets the starting point for
        determining the optimal allocation. Length must be equal to the
        number of systems.

    systems : dict
        ``"obj"``:
        is a dictionary of numpy arrays, indexed by system number, each of which corresponds to the objective values of a system

        ``"var"``:
        is a dictionary of 2d numpy arrays, indexed by system number, each of which corresponds to the covariance matrix of a system

        ``"inv_var"``:
        is a dictionary of 2d numpy, indexed by system number, each of which corresponds to the inverse covariance matrix of a system

        ``"pareto_indices"``:
        is a list of pareto systems ordered by the first objective

        ``"non_pareto_indices"``:
        is a list of non-pareto systems ordered by the first objective

    Returns
    -------
    z : float
        IMOSCORE convergence rate associated with alphas.
    """
    iscore_problem = IMOSCORE(systems=systems)
    z = iscore_problem.calc_rate(alphas=alphas)
    return z
