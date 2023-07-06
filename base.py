#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Summary
-------
Provide class definitions for MORS Problems and Solvers and pairings.

Listing
-------
MO_Alloc_Problem : class
MORS_Problem : class
MORS_Solver : class
record_metrics : function
MORS_Tester : class
make_rate_plots : function
make_phantom_rate_plots : function
attach_rng : function
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import time

import pymoso.chnutils as chnutils
from mrg32k3a.mrg32k3a import MRG32k3a

from allocate import smart_allocate, calc_phantom_rate, calc_moscore_rate, calc_imoscore_rate
from utils import is_pareto_efficient   # _mp_objmethod,


class MO_Alloc_Problem(object):
    """Class for multi-objective allocation problems.
    This refers to the sample means/covariances that dictate what the allocation
    should be at a given time.

    Attributes
    ----------
    n_systems : int
        number of systems

    obj : dict of numpy arrays
        indexed by system number, each of which corresponds to the objective values of a system

    var : dict of 2d numpy arrays
        indexed by system number, each of which corresponds to the covariance matrix of a system

    inv_var : dict of 2d numpy arrays
        indexed by system number,each of which corresponds to the inverse covariance matrix of a system

    pareto_indices : list
        a list of pareto systems ordered by the first objective

    non_pareto_indices : list
        a list of non-pareto systems ordered by the first objective

    Parameters
    ----------
    obj_vals : dict
        Dictionary of tuples of objective values keyed by system number.
        Tuples of objective values are assumed to be of equal length.

    obj_vars : dict
        Dictionary of covariance (numpy 2d arrays) keyed by system number.
        Numbers of rows and columns are equal to the number of objectives.
    """
    def __init__(self, obj_vals, obj_vars):
        # TODO: Check for positive semidefinite?
        # Replacing the following line with call to is_pareto_efficient().
        # pareto_indices = list(moso_utils.get_nondom(obj_vals))
        self.n_systems = len(obj_vals)
        self.obj = obj_vals
        self.var = obj_vars
        self.inv_var = {system: np.linalg.inv(obj_vars[system]) for system in range(self.n_systems)}
        # Determine indices of Pareto systems.
        obj_vals_matrix = np.array([list(obj_vals[system]) for system in range(self.n_systems)])
        pareto_indices = list(is_pareto_efficient(costs=obj_vals_matrix, return_mask=False))
        pareto_indices.sort(key=lambda x: obj_vals_matrix[x][0])
        self.pareto_indices = pareto_indices
        # Determine indices of non-Pareto systems.
        non_pareto_indices = [system for system in range(self.n_systems) if system not in pareto_indices]
        non_pareto_indices.sort(key=lambda x: obj_vals_matrix[x][0])
        self.non_pareto_indices = non_pareto_indices


class MORS_Problem(object):
    """Class for multi-objective ranking-and-selection problem.

    Attributes
    ----------
    n_objectives : int
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
        """
        self.rng = rng

    def reset_statistics(self):
        """Reset sample statistics for all systems.
        """
        self.sample_sizes = [0 for _ in range(self.n_systems)]
        self.sums = [[0 for _ in range(self.n_objectives)] for _ in range(self.n_systems)]
        self.sums_of_products = [[[0 for _ in range(self.n_objectives)] for _ in range(self.n_objectives)] for _ in range(self.n_systems)]
        self.sample_means = [[None for _ in range(self.n_objectives)] for _ in range(self.n_systems)]
        self.sample_covs = [[[None for _ in range(self.n_objectives)] for _ in range(self.n_objectives)] for _ in range(self.n_systems)]

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
            for obj_idx in range(self.n_objectives):
                self.sums[system_idx][obj_idx] += objs[idx][obj_idx]
                self.sample_means[system_idx][obj_idx] = self.sums[system_idx][obj_idx] / self.sample_sizes[system_idx]
            # Add outputs to running sums of products and recompute sample variance-covariance matrix.
            for obj_idx1 in range(self.n_objectives):
                for obj_idx2 in range(self.n_objectives):
                    self.sums_of_products[system_idx][obj_idx1][obj_idx2] += objs[idx][obj_idx1] * objs[idx][obj_idx2]
                    if self.sample_sizes[system_idx] > 1:
                        # From https://www.randomservices.org/random/sample/Covariance.html,
                        #   sample cov (x, y) = n / (n-1) * [sample mean (x*y) - sample mean (x) * sample mean (y)]
                        self.sample_covs[system_idx][obj_idx1][obj_idx2] = self.sample_sizes[system_idx] / (self.sample_sizes[system_idx] - 1) * \
                            (self.sums_of_products[system_idx][obj_idx1][obj_idx2] / self.sample_sizes[system_idx] - self.sample_means[system_idx][obj_idx1] * self.sample_means[system_idx][obj_idx2])
        # TODO: Make more efficient by only recomputing stats outside the for loop.

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
        # Assume the true_means and true_vars are specified, as in our test examples.
        # As default behavior, generate observations from a multivariate normal distribution.
        system_idx = self.systems.index(x)
        obj = self.rng.mvnormalvariate(mean_vec=self.true_means[system_idx],
                                       cov=self.true_covs[system_idx],
                                       factorized=False)
        return tuple(obj)

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
        chosen allocation method. Options are "iMOSCORE", "MOSCORE", "Phantom", and "Brute Force"
    alpha_epsilon : float
        lower threshold for enforcing a minimum required sample size
    crn_across_solns : bool
        indicates whether CRN are used when sampling across systems (True) or not (False)
    """
    def __init__(self, budget, n0, delta, allocation_rule, alpha_epsilon, crn_across_solns):
        self.budget = budget
        self.n0 = n0
        self.delta = delta
        self.allocation_rule = allocation_rule
        self.alpha_epsilon = alpha_epsilon
        self.crn_across_solns = crn_across_solns

    def attach_rng(self, rng):
        """Attach random number generator to MORS_Solver object.

        Parameters
        ----------
        rng : MRG32k3a object
            random number generator to use for simulating replications
        """
        self.rng = rng

    def solve(self, problem):
        """Solve a given MORS problem - one macroreplication.

        Notes
        -----
        To skip our suggestions for more efficient allocations, one can
        directly call allocate() instead of smart_allocate(), with the
        same inputs.

        Parameters
        ----------
        problem : base.MORS_Problem object
            multi-objective R&S problem to solve

        Returns
        -------
        outputs : dict

            ``"alpha_hat"``
            list of float, final simulation allocation by system

            ``"paretos"``
            list of indices of estimated pareto systems at termination

            ``"objectives"``
            dictionary, keyed by system index, of lists containing estimated objective values at termination

            ``"variances"``
            dictionary, keyed by system index, of estimated covariance matrices as numpy arrays at termination

            ``"sample_sizes"``
            list of int, final sample size for each system

        metrics(optional): dict

            ``"alpha_hats"``
            list of lists of float, the simulation allocation selected at each step in the solver

            ``"alpha_bars"``
            list of lists of float, the portion of the simulation budget that has been allocatedto each system at each step in the solver

            ``"paretos"``
            list of lists of int, the estimated pareto frontier at each step in the solver

            ``"MCI_bool"``
            list of Bool, indicating whether an misclassification by inclusion occured at each step in the solver

            ``"MCE_bool"``
            list of Bool, indicating whether an misclassification by exclusion occured at each step in the solver

            ``"MC_bool"``
            list of Bool, indicating whether an misclassification occured at each step in the solver

            ``"percent_false_exclusion"``
            list of float, the portion of true pareto systems that are falsely excluded at each step in the solver

            ``"percent_false_inclusion"``
            list of float, the portion of true non-pareto systems that are falsely included at each step in the solver

            ``"percent_misclassification"``
            list of float, the portion of systems that are misclassified at each step in the solver

            ``"timings"``
            list of float, the time spent calculating the allocation distribution at each step in the solver

        """
        if self.n0 < problem.n_objectives + 1:
            raise ValueError("n0 has to be greater than or equal to n_objectives plus 1 to guarantee positive definite\
                  sample variance-covariance matrices.")

        # No warm start solution for first call to smart_allocate().
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
                   'percent_misclassification': [],
                   'timings': []
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

            # Enforce minimum sampling requirement.
            # ["alpha_bars"][-1] refers to current sampling proportions for all systems.
            undersampled_systems = [idx for idx in range(problem.n_systems) if metrics["alpha_bars"][-1][idx] < self.alpha_epsilon]

            cardinality_S_epsilon = len(undersampled_systems)
            if cardinality_S_epsilon > self.delta:  # If too many systems are undersampled...
                systems_to_sample = self.rng.choices(population=undersampled_systems,
                                                     weights=[1 / cardinality_S_epsilon for _ in range(cardinality_S_epsilon)],
                                                     k=self.delta
                                                     )
                # Record timing of zero.
                metrics["timings"].append(0.0)
                # Previous value of alpha_hat will be recorded.
                # alpha_hat = None
            else:
                # Force sample undersampled systems.
                delta_epsilon = self.delta - cardinality_S_epsilon
                if cardinality_S_epsilon > 1:
                    systems_to_sample = undersampled_systems
                else:
                    systems_to_sample = []

                # Use remaining samples (if any) to sample according to allocation distribution.
                # Note: If |S_epsilon| = delta, all sampling is compulsory --> no need to compute allocation.
                if delta_epsilon > 0:
                    # Get distribution for sampling allocation.
                    obj_vals = {idx: problem.sample_means[idx] for idx in range(problem.n_systems)}
                    obj_vars = {idx: np.array(problem.sample_covs[idx]) for idx in range(problem.n_systems)}
                    allocation_problem = MO_Alloc_Problem(obj_vals=obj_vals, obj_vars=obj_vars)
                    tic = time.perf_counter()
                    alpha_hat, _ = smart_allocate(self.allocation_rule, allocation_problem, warm_start=warm_start)
                    toc = time.perf_counter()
                    # In subsequent iterations, use previous alpha_hat as warm start
                    warm_start = alpha_hat

                    # Compare recommended allocation to equal allocation, in case solver struggled.
                    alpha_eq = np.array([1 / problem.n_systems for _ in range(problem.n_systems)])
                    # If 4+ objectives or 100+ systems, use z^imo for comparisons. Otherwise use z^mo.
                    if problem.n_objectives >= 4 or problem.n_systems >= 100:
                        z_alpha_hat = calc_imoscore_rate(alphas=alpha_hat, systems=allocation_problem)
                        z_alpha_eq = calc_imoscore_rate(alphas=alpha_eq, systems=allocation_problem)
                    else:
                        z_alpha_hat = calc_moscore_rate(alphas=alpha_hat, systems=allocation_problem)
                        z_alpha_eq = calc_moscore_rate(alphas=alpha_eq, systems=allocation_problem)
                    # Use equal allocation if it's better than what solver recommends.
                    if z_alpha_hat < z_alpha_eq:
                        alpha_hat = alpha_eq

                    # Record timing.
                    metrics["timings"].append(toc - tic)
                    # Sequentially choose systems to simulate by drawing independently from allocation distribution.
                    # Append to list of systems that require sampling (if any).
                    systems_to_sample += self.rng.choices(population=range(problem.n_systems),
                                                          weights=alpha_hat,
                                                          k=delta_epsilon
                                                          )
                    # Repeated systems are possible.
                else:
                    # Record timing of zero.
                    metrics["timings"].append(0.0)
                    # Previous value of alpha_hat will be recorded.
                    # alpha_hat = None

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

        ``"alpha_hats"``
        list of lists of float, the simulation allocation selected at each step in the solver

        ``"alpha_bars"``
        list of lists of float, the portion of the simulation budget that has been allocated
        to each system at each step in the solver

        ``"paretos"``
        list of lists of int, the estimated pareto frontier at each step in the solver

        ``"MCI_bool"``
        list of Bool, indicating whether an misclassification by inclusion
        occured at each step in the solver

        ``"MCE_bool"``
        list of Bool, indicating whether an misclassification by exclusion
        occured at each step in the solver

        ``"MC_bool"``
        list of Bool, indicating whether an misclassification
        occured at each step in the solver

        ``"percent_false_exclusion"``
        list of float, the portion of true pareto systems
        which are falsely excluded at each step in the solver

        ``"percent_false_inclusion"``
        list of float, the portion of true non-pareto systems
        which are falsely included at each step in the solver

        ``"percent_misclassification"``
        list of float, the portion of systems which are
        misclassified at each step in the solver

    problem : base.MORS_Problem object
        multi-objective R&S problem to solve
    alpha_hat : list
        allocation recommended by allocation rule

    Returns
    -------
    metrics : dict

        ``"alpha_hats"``
        list of lists of float, the simulation allocation selected at each step in the solver

        ``"alpha_bars"``
        list of lists of float, the portion of the simulation budget that has been allocated
        to each system at each step in the solver

        ``"paretos"``
        list of lists of int, the estimated pareto frontier at each step in the solver

        ``"MCI_bool"``
        list of Bool, indicating whether an misclassification by inclusion
        occured at each step in the solver

        ``"MCE_bool"``
        list of Bool, indicating whether an misclassification by exclusion
        occured at each step in the solver

        ``"MC_bool"``
        list of Bool, indicating whether an misclassification
        occured at each step in the solver

        ``"percent_false_exclusion"``
        list of float, the portion of true pareto systems
        which are falsely excluded at each step in the solver

        ``"percent_false_inclusion"``
        list of float, the portion of true non-pareto systems
        which are falsely included at each step in the solver

        ``"percent_misclassification"``
        list of float, the portion of systems which are
        misclassified at each step in the solver

    """
    # Record recommended and empirical allocation proportions.
    metrics['alpha_hats'].append(alpha_hat)
    metrics['alpha_bars'].append(np.array([size / sum(problem.sample_sizes) for size in problem.sample_sizes]))

    # Record systems that look Pareto efficient.
    est_obj_vals = {idx: problem.sample_means[idx] for idx in range(problem.n_systems)}
    est_pareto = list(chnutils.get_nondom(est_obj_vals))
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
    intermediate_budgets : list
        intermediate budgets at which statistics are reported
    all_outputs : list of dict
        list of terminal statistics from each macroreplication
    all_metrics : list of dict
        list of statistics over time from each macroreplication
    rates : dict
        ``"MCI_rate"``
        list of float, empirical MCI rate at a given point across sequential solver macroreplications

        ``"MCE_rate"``
        list of float, empirical MCE rate at a given point across sequential solver macroreplications

        ``"MC_rate"``
        list of float, empirical MC rate at a given point across sequential solver macroreplications

        ``"avg_percent_false_exclusion"``
        list of float, the average proportion of true pareto systems that are falsely excluded at each step in the solver

        ``"avg_percent_false_inclusion"``
        list of float, the average proportion of true non-pareto systems that are falsely included at each step in the solver

        ``"avg_percent_misclassification"``
        list of float, the proportion of systems that are misclassified at each step in the solver

        ``"phantom_rate_25pct"``
        list of float, 25-percentile of difference of phantom rates for phantom allocation and empirical allocation over time

        ``"phantom_rate_50pct"``
        list of float, 50-percentile of difference of phantom rates for phantom allocation and empirical allocation over time

        ``"phantom_rate_75pct"``
        list of float, 75-percentile of difference of phantom rates for phantom allocation and empirical allocation over time

    file_name_path : str
        name of files for saving results
    """
    def __init__(self, solver, problem):
        self.problem = problem
        self.solver = solver
        # Initialize tracking of statistics for each macroreplication.
        self.intermediate_budgets = [solver.n0 * problem.n_systems + i * solver.delta for i in range(int(np.ceil((solver.budget - solver.n0 * problem.n_systems) / solver.delta) + 1))]
        self.all_outputs = []
        self.all_metrics = []

    def setup_rng_states(self):
        """Setup rng states for each system based on whether solver uses CRN.
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
        print(f"Running MORS Solver with allocation rule {self.solver.allocation_rule} with budget={self.solver.budget}, n0={self.solver.n0}, and delta={self.solver.delta}.")
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
        # Record results to .txt file and save MORS_Tester object in .pickle file.
        self.record_tester_results()

    def aggregate_metrics(self):
        """Aggregate run-time statistics over macroreplications, e.g., calculate
        empirical misclassification rates.
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

    def record_tester_results(self):
        """
        Write summary to .txt and save base.MORS_Tester object to .pickle file.
        """
        # Common file name for .txt and .pickle files.
        self.file_name_path = f"outputs/{self.solver.allocation_rule}_with_budget={self.solver.budget}_n0={self.solver.n0}_delta={self.solver.delta}_mreps={self.n_macroreps}"
        # Write summary to .txt file.
        with open(self.file_name_path + ".txt", "w+") as file:
            file.write("The file " + self.file_name_path + ".pickle contains the MORS Tester object for the following experiment: \n")
            file.write("\nMORS Problem:\n")
            file.write(f"\tnumber of objectives = {self.problem.n_objectives}\n")
            file.write(f"\tnumber of systems = {self.problem.n_systems}\n")
            file.write("\n")
            for system_idx in range(self.problem.n_systems):
                file.write(f"\tsystem {system_idx} has true means {self.problem.true_means[system_idx]} and covariance matrix {self.problem.true_covs[system_idx]}\n")
            file.write("\nMORS Solver:\n")
            file.write(f"\tallocation rule = {self.solver.allocation_rule}\n")
            file.write(f"\tbudget = {self.solver.budget}\n")
            file.write(f"\tn0 = {self.solver.n0}\n")
            file.write(f"\tdelta = {self.solver.delta}\n")
            file.write(f"\n{self.n_macroreps} macroreplications were run:\n")
            for mrep in range(self.n_macroreps):
                avg_timing = np.mean(self.all_metrics[mrep]["timings"])
                file.write(f"\tmacroreplication {mrep}: avg time per allocation call = {avg_timing:.6f} s\n")
        # Save MORS_Tester object to .pickle file.
        with open(self.file_name_path + ".pickle", "wb") as file:
            pickle.dump(self, file, pickle.HIGHEST_PROTOCOL)


def make_rate_plots(testers):
    """Make plots of MCI/MCE/MC and average percentage misclassification rates.

    Parameters
    ----------
    testers : `list` [`MORS_Tester`]
        list of testers for comparison
    """
    plot_types = ['MCI_rate',
                  'MCE_rate',
                  'MC_rate',
                  'avg_percent_false_exclusion',
                  'avg_percent_false_inclusion',
                  'avg_percent_misclassification'
                  ]
    y_axis_labels = [r"$\hat{P}$(MCI)",
                     r"$\hat{P}$(MCE)",
                     r"$\hat{P}$(MC)",
                     r"Avg $\%$ False Exclusion",
                     r"Avg $\%$ False Inclusion",
                     r"Avg $\%$ Misclassification"
                     ]
    marker_list = ["o", "v", "s", "*", "P", "X", "D", "V", ">", "<"]
    # Assume the solvers have all been run on the same problem.
    for plot_idx in range(len(plot_types)):
        # Set up a new plot.
        plot_type = plot_types[plot_idx]
        plt.figure()
        # plt.ylim((-0.05, 1.05))
        plt.xlim((0, testers[0].solver.budget))
        plt.xlabel(r"Sample Size $n$", size=14)
        plt.ylabel(y_axis_labels[plot_idx], size=14)
        solver_curve_handles = []
        # Plot rate curve for each solver.
        for tester_idx in range(len(testers)):
            tester = testers[tester_idx]
            solver_curve_handle, = plt.plot(tester.intermediate_budgets,
                                            tester.rates[plot_type],
                                            color="C" + str(tester_idx),
                                            marker=marker_list[tester_idx],
                                            linestyle="-",
                                            linewidth=2
                                            )
            solver_curve_handles.append(solver_curve_handle)
        # Add a legend.
        # Assume solver allocation rules are unique.
        solver_names = [tester.solver.allocation_rule for tester in testers]
        plt.legend(handles=solver_curve_handles, labels=solver_names, loc="upper right")
        # Save the plot.
        path_name = f"plots/{plot_type}_rates.png"
        plt.savefig(path_name, bbox_inches="tight")


def make_phantom_rate_plots(testers):
    """Make plots of MCI/MCE/MC and average percentage misclassification rates.

    Parameters
    ----------
    testers : `list` [`MORS_Tester`]
        list of testers for comparison
    """
    # Assume all testers have the same problem.
    common_problem = testers[0].problem
    # Calculate phantom rate of phantom allocation for true problem.
    true_obj_vals = {idx: common_problem.true_means[idx] for idx in range(common_problem.n_systems)}
    true_obj_vars = {idx: np.array(common_problem.true_covs[idx]) for idx in range(common_problem.n_systems)}
    true_allocation_problem = MO_Alloc_Problem(obj_vals=true_obj_vals, obj_vars=true_obj_vars)
    _, phantom_rate_of_phantom_alloc = smart_allocate(method="Phantom", systems=true_allocation_problem)
    # print(phantom_rate_of_phantom_alloc)

    # Calculate phantom rates for each tester.
    for tester in testers:
        phantom_rate_of_empirical_alloc_curves = []
        for mrep in range(tester.n_macroreps):
            z_phantom_alpha_bar_curve = [calc_phantom_rate(alphas=tester.all_metrics[mrep]["alpha_bars"][budget_idx], systems=true_allocation_problem) for budget_idx in range(len(tester.intermediate_budgets))]
            # print(z_phantom_alpha_bar_curve)
            phantom_rate_of_empirical_alloc_curves.append(z_phantom_alpha_bar_curve)
        tester.rates["phantom_rate_25pct"] = [np.quantile(a=[phantom_rate_of_phantom_alloc - phantom_rate_of_empirical_alloc_curves[mrep][budget_idx] for mrep in range(tester.n_macroreps)], q=0.25) for budget_idx in range(len(tester.intermediate_budgets))]
        tester.rates["phantom_rate_50pct"] = [np.quantile(a=[phantom_rate_of_phantom_alloc - phantom_rate_of_empirical_alloc_curves[mrep][budget_idx] for mrep in range(tester.n_macroreps)], q=0.50) for budget_idx in range(len(tester.intermediate_budgets))]
        tester.rates["phantom_rate_75pct"] = [np.quantile(a=[phantom_rate_of_phantom_alloc - phantom_rate_of_empirical_alloc_curves[mrep][budget_idx] for mrep in range(tester.n_macroreps)], q=0.75) for budget_idx in range(len(tester.intermediate_budgets))]

        # Save updated MORS_Tester object to .pickle file.
        with open(tester.file_name_path + ".pickle", "wb") as file:
            pickle.dump(tester, file, pickle.HIGHEST_PROTOCOL)

    # Set up a new plot.
    plt.figure()
    # plt.ylim((-0.05, 1.05))
    # Assume all solvers have same budget.
    plt.xlim((0, testers[0].solver.budget))
    plt.xlabel(r"Sample Size $n$", size=14)
    plt.ylabel(r"$z^{ph}(\alpha^{ph}) - z^{ph}(\bar{\alpha}_n)$", size=14)
    marker_list = ["o", "v", "s", "*", "P", "X", "D", "V", ">", "<"]
    solver_curve_handles = []
    # Plot rate curve for each solver.
    for tester_idx in range(len(testers)):
        tester = testers[tester_idx]
        # Plot 25%, 50%, and 75% quantiles.
        solver_curve_handle, = plt.plot(tester.intermediate_budgets,
                                        tester.rates["phantom_rate_50pct"],
                                        color="C" + str(tester_idx),
                                        marker=marker_list[tester_idx],
                                        linestyle="-",
                                        linewidth=2
                                        )
        solver_curve_handles.append(solver_curve_handle)
        plt.plot(tester.intermediate_budgets,
                 tester.rates["phantom_rate_25pct"],
                 color="C" + str(tester_idx),
                 marker=marker_list[tester_idx],
                 linestyle="-",
                 linewidth=2
                 )
        plt.plot(tester.intermediate_budgets,
                 tester.rates["phantom_rate_75pct"],
                 color="C" + str(tester_idx),
                 marker=marker_list[tester_idx],
                 linestyle="-",
                 linewidth=2
                 )
    # Add a legend.
    # Assume solver allocation rules are unique.
    solver_names = [tester.solver.allocation_rule for tester in testers]
    plt.legend(handles=solver_curve_handles, labels=solver_names, loc="upper right")
    # Save the plot.
    path_name = "plots/phantom_rates.png"
    plt.savefig(path_name, bbox_inches="tight")
