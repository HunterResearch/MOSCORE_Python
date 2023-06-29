#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Summary
-------
A script for reproducing the experiments in the Applegate et al. (2020, JoS) paper.
"""

import numpy as np
import time
from mrg32k3a.mrg32k3a import MRG32k3a

from utils import create_allocation_problem, is_pareto_efficient, find_phantoms
from allocate import allocate, smart_allocate, calc_brute_force_rate, calc_phantom_rate
from example import create_fixed_pareto_random_problem, create_variable_pareto_random_problem

# 3-objective, 3-system example from Li et al. (2018).
obj_vals = {0: [3.0, 4.0, 2.2], 1: [3.5, 5.0, 3.0], 2: [4.0, 3.5, 2.0]}
obj_vars = {0: np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]]),
            1: np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]]),
            2: np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])}
allocation_problem = create_allocation_problem(obj_vals=obj_vals, obj_vars=obj_vars)

for rule in ["Brute Force", "Phantom", "SCORE", "Brute Force Ind", "iSCORE", "Equal"]:
    tic = time.perf_counter()
    alpha_hat, z = allocate(method=rule, systems=allocation_problem)
    #alpha_hat, z = smart_allocate(method=rule, systems=allocation_problem)
    toc = time.perf_counter()
    print("Allocation Rule:", rule)
    # print("alpha_hat:", alpha_hat)
    # print("z", z)
    print("T:", round(toc - tic, 3), "s")
    z_bf = calc_brute_force_rate(alphas=alpha_hat, systems=allocation_problem)
    print("Z^bf(alpha) x 10^5:", round(z_bf * 10**5, 4))
    z_ph = calc_phantom_rate(alphas=alpha_hat, systems=allocation_problem)
    print("Z^ph(alpha) x 10^5:", round(z_ph * 10**5, 4))
    print("\n")

# # Repeat fixed-Pareto experiments from Figures 3.
# n_problems = 10

# d = 3  # Number of objectives
# r = 500  # Number of systems
# p = 10  # Number of Pareto systems

# #rules = ["Brute Force", "Phantom", "MOSCORE", "Brute Force Ind", "iMOSCORE", "Equal"]
# #rules = ["Phantom", "MOSCORE", "iMOSCORE", "Equal"]
# rules = ["MOSCORE", "iMOSCORE", "Equal"]

# n_paretos_list = []
# n_phantoms_list = []

# solve_times = {rule: [] for rule in rules}
# z_bf_list = {rule: [] for rule in rules}
# z_ph_list = {rule: [] for rule in rules}

# for problem_idx in range(n_problems):
#     print("Generating problem", problem_idx + 1, "of", n_problems)
#     # TODO: Revise to use corr/sigma arguments.
#     random_problem = create_fixed_pareto_random_problem(n_systems=r, n_obj=d, n_paretos=p, sigma=1, corr=None, center=100, radius=6, minsep=0.0001)
#     obj_array = np.array([random_problem["obj"][i] for i in range(r)])
#     n_paretos = len(is_pareto_efficient(costs=obj_array, return_mask=False))

#     # Construct Pareto array
#     pareto_array = np.zeros([n_paretos, d])
#     for i in range(n_paretos):
#         pareto_array[i, :] = random_problem['obj'][random_problem['pareto_indices'][i]]

#     n_phantoms = len(find_phantoms(paretos=pareto_array, n_obj=d))
#     n_paretos_list.append(n_paretos)
#     n_phantoms_list.append(n_phantoms)

#     for rule in rules:
#         print("Solve with", rule, "rule.")
#         tic = time.perf_counter()
#         alpha_hat, z = allocate(method=rule, systems=random_problem)
#         toc = time.perf_counter()
#         solve_time = toc - tic
#         if r <= 10:  # Skip brute force if too expensive.
#             z_bf = calc_brute_force_rate(alphas=alpha_hat, systems=random_problem)
#         z_ph = calc_phantom_rate(alphas=alpha_hat, systems=random_problem)

#         # Record statistics.
#         solve_times[rule].append(solve_time)
#         if r <= 10:  # Skip brute force if too expensive.
#             z_bf_list[rule].append(z_bf)
#         z_ph_list[rule].append(z_ph)

# print("Median # of Pareto systems, p =", np.median(n_paretos_list))
# print("Median # of phantom systems, |P^ph| =", np.median(n_phantoms_list))
# print("\n")

# for rule in rules:
#     print("Allocation Rule:", rule)

#     print("Median wall-clock time, T =", round(np.median(solve_times[rule]), 3), "s")
#     print("75-percentile wall-clock time, T =", round(np.quantile(solve_times[rule], 0.75), 3), "s")
#     if r <= 10:  # Skip brute force if too expensive
#         print("Median brute force convergence rate, Z^bf(alpha) x 10^5 =", round(np.median(z_bf_list[rule]) * 10**5, 4))
#     print("Median phantom convergence rate, Z^ph(alpha) x 10^5 =", round(np.median(z_ph_list[rule]) * 10**5, 4))
#     print("\n")
