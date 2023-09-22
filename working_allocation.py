#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Summary
-------
A script for interacting with the MORS Problem, Solver, and Tester classes.
"""

#from pymoso.prng.mrg32k3a import MRG32k3a, get_next_prnstream
from mrg32k3a.mrg32k3a import MRG32k3a
import numpy as np
import time

# from base import MORS_Problem, MORS_Solver, MORS_Tester, make_rate_plots, make_phantom_rate_plots
from example import TestProblem, TestProblem2, TestProblem3
from allocate import allocate, smart_allocate
from base import MO_Alloc_Problem
# from allocate import allocate

# from iscore_allocation import iscore_allocation
# from score_allocation import score_allocation

myproblem = TestProblem2()

myrng = MRG32k3a()
myproblem.attach_rng(myrng)
myproblem.rng_states = [myrng._current_state] * myproblem.n_systems

warm_start = None

# alpha_hat = np.array([1 / myproblem.n_systems for _ in range(myproblem.n_systems)])
# system_indices = list(range(myproblem.n_systems)) * 10
# objs = myproblem.bump(system_indices=system_indices)
# myproblem.update_statistics(system_indices=system_indices, objs=objs)

# obj_vals = {idx: myproblem.sample_means[idx] for idx in range(myproblem.n_systems)}
# obj_vars = {idx: np.array(myproblem.sample_covs[idx]) for idx in range(myproblem.n_systems)}

obj_vals = {idx: myproblem.true_means[idx] for idx in range(myproblem.n_systems)}
obj_vars = {idx: np.array(myproblem.true_covs[idx]) for idx in range(myproblem.n_systems)}

allocation_problem = MO_Alloc_Problem(obj_vals=obj_vals, obj_vars=obj_vars)

# # from utils import calc_brute_force_rate
# from allocation import calc_brute_force_rate, calc_phantom_rate, calc_score_rate, calc_iscore_rate
# z_new = calc_iscore_rate(alphas=alpha_hat, systems=allocation_problem)
# print("znew", z_new)

# from utils import old_calc_brute_force_rate
# z_old = old_calc_brute_force_rate(alphas=alpha_hat, systems=allocation_problem)
# print("zold", z_old)

# # print(allocation_problem.obj)
# # print(allocation_problem.var)
# # print(allocation_problem.pareto_indices)

# res = score_allocation(systems=allocation_problem, warm_start=warm_start)
# alpha_hat = res[0]
# z = res[1]
tic = time.perf_counter()
# alpha_hat, z = allocate("iSCORE", allocation_problem, warm_start=warm_start)
alpha_hat, z = smart_allocate("Brute Force Ind", allocation_problem, warm_start=warm_start)
toc = time.perf_counter()
print("time", toc - tic)
print("alpha_hat", alpha_hat)
print("z", z)
