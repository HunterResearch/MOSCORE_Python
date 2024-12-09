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
import pymatreader

from utils import is_pareto_efficient, find_phantoms
from allocate import allocate, smart_allocate, calc_brute_force_rate, calc_phantom_rate
from example import create_fixed_pareto_random_problem, create_variable_pareto_random_problem
from base import MO_Alloc_Problem

print("Results for Li et al. (2018) problem experiment.\n--------------------------------\n")
# 3-objective, 3-system example from Li et al. (2018).
obj_vals = {0: np.array([3.0, 4.0, 2.2]), 1: np.array([3.5, 5.0, 3.0]), 2: np.array([4.0, 3.5, 2.0])}
obj_vars = {0: np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]]),
            1: np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]]),
            2: np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])}
allocation_problem = MO_Alloc_Problem(obj_vals=obj_vals, obj_vars=obj_vars)

#for rule in ["Brute Force", "Phantom", "MOSCORE", "Brute Force Ind", "iMOSCORE", "Equal"]:
for rule in ["MOSCORE", "iMOSCORE"]:

    print(f"Solve problem Li et al. (2018) problem with {rule} rule.")


    tic = time.perf_counter()
    alpha_hat, z = allocate(method=rule, alloc_problem=allocation_problem)
    #alpha_hat, z = smart_allocate(method=rule, alloc_problem=allocation_problem)
    toc = time.perf_counter()
    print("Allocation Rule:", rule)
    # print("alpha_hat:", alpha_hat)
    # print("z", z)
    print("T:", round(toc - tic, 3), "s")
    z_bf = calc_brute_force_rate(alpha=alpha_hat, alloc_problem=allocation_problem)
    print("Z^bf(alpha) x 10^5:", round(z_bf * 10**5, 4))
    z_ph = calc_phantom_rate(alpha=alpha_hat, alloc_problem=allocation_problem)
    print("Z^ph(alpha) x 10^5:", round(z_ph * 10**5, 4))
    print("\n")

# Repeat fixed-Pareto experiments from Table 3.
n_problems = 1  # Number of problems within each row of table.   # Normally 10
#d_vector = [3, 3, 3, 4, 4, 5]
#r_vector = [10, 500, 10000, 5000, 10000, 10000]

d_vector = [3]
r_vector = [10]

#d_vector = [4]
#r_vector = [5000]

rules = ["MOSCORE", "iMOSCORE"]
solve_times = {rule: [] for rule in rules}
z_bf_list = {rule: [] for rule in rules}
z_ph_list = {rule: [] for rule in rules}

# Write summary to .txt file.
results_filename = "outputs/Figure3_results.txt"
with open(results_filename, "w+") as file:
    file.write(f"The file {results_filename} contains the results for the following experiment: \n\n")

    for d, r in zip(d_vector, r_vector):

        print(f"Running experiment with d = {d} objectives and r = {r} systems.\n--------------------------------\n")
        file.write(f"Running experiment with d = {d} and r = {r}.\n")

        for prob_idx in range(n_problems):

            # Reuse exact problem instances tested in JoS paper by reading in data from file.

            experiment_filename = f"PyMORS-MatFiles/TimeProbs{d}D/TimeProb_{d}D_{r}_{prob_idx+1}_FIX.mat"
            experiment_dict = pymatreader.read_mat(experiment_filename)
            obj_vals = experiment_dict["systems"]["obj"]
            obj_vars = experiment_dict["systems"]["cov"]
            alloc_problem = MO_Alloc_Problem(obj_vals, obj_vars)

            # OR
            # Create new problem instances.
            # ...
            # alloc_problem = MO_Alloc_Problem(obj_vals, obj_vars)

            file.write(f"\nMORS Problem # {prob_idx+1} of {n_problems}:\n")
            file.write("\n")
            #for system_idx in range(alloc_problem.n_systems):
            #    file.write(f"System {system_idx} has sample means {alloc_problem.obj[system_idx]} and sample covariance matrix\n{alloc_problem.var[system_idx]}\n\n")

            for rule in rules:
                print(f"Solve problem {prob_idx} with {rule} rule.")
                
                tic = time.perf_counter()
                alpha_hat, z = allocate(method=rule, alloc_problem=alloc_problem)
                toc = time.perf_counter()
                solve_time = toc - tic
                if r <= 10:  # Skip brute force if too expensive.
                    z_bf = calc_brute_force_rate(alpha=alpha_hat, alloc_problem=alloc_problem)
                z_ph = calc_phantom_rate(alpha=alpha_hat, alloc_problem=alloc_problem)

                # Record statistics.
                solve_times[rule].append(solve_time)
                if r <= 10:  # Skip brute force if too expensive.
                    z_bf_list[rule].append(z_bf)
                z_ph_list[rule].append(z_ph)

                # Record results to file
                file.write(f"\nResults for {rule} rule:\n----------------------------\n")
                file.write(f"Allocation (alpha) = {alpha_hat}.\n")
                if r <= 10:  # Skip brute force if too expensive
                    file.write(f"Brute force convergence rate, Z^bf(alpha) x 10^5 = {z_bf * 10**5}.\n")
#                    file.write(f"Brute force convergence rate, Z^bf(alpha) x 10^5 = {round(z_bf * 10**5, 4)}.\n")
                file.write(f"Phantom rate (z^ph(alpha)) x 10^5 = {z_ph * 10**5}.\n")
#                file.write(f"Phantom rate (z^ph(alpha)) x 10^5 = {round(z_ph * 10**5, 4)}.\n")
                file.write(f"Solve time = {round(solve_time, 4)} s.\n")
                
        print(f"\nResults for experiment with d = {d} and r = {r}.\n--------------------------------\n")
        for rule in rules:
            print("Allocation Rule:", rule)

            print(f"Median wall-clock time, T = {round(np.median(solve_times[rule]), 3)} s")
            print(f"75-percentile wall-clock time, T = {round(np.quantile(solve_times[rule], 0.75), 3)} s")
            if r <= 10:  # Skip brute force if too expensive
                print(f"Median brute force convergence rate, Z^bf(alpha) x 10^5 = {round(np.median(z_bf_list[rule]) * 10**5, 4)}")
            print(f"Mean phantom convergence rate, Z^ph(alpha) x 10^5 = {round(np.mean(z_ph_list[rule]) * 10**5, 4)}")
            print(f"Median phantom convergence rate, Z^ph(alpha) x 10^5 = {round(np.median(z_ph_list[rule]) * 10**5, 4)}")
            print("\n")


# d = 3  # Number of objectives
# r = 10  # Number of systems
# p = 5  # Number of Pareto systems

# n_paretos_list = []
# n_phantoms_list = []

# for problem_idx in range(n_problems):
#     print("Generating problem", problem_idx + 1, "of", n_problems)
#     # TODO: Revise to use corr/sigma arguments.
#     random_problem = create_fixed_pareto_random_problem(n_systems=r, n_objectives=d, n_paretos=p, sigma=1, corr=None, center=100, radius=6, minsep=0.0001)
#     obj_array = np.array([random_problem.obj[i] for i in range(r)])
#     n_paretos = len(is_pareto_efficient(costs=obj_array, return_mask=False))

#     # Construct Pareto array
#     pareto_array = np.zeros([n_paretos, d])
#     for i in range(n_paretos):
#         pareto_array[i, :] = random_problem.obj[random_problem.pareto_indices[i]]

#     # n_phantoms = len(find_phantoms(paretos=pareto_array, n_objectives=d))
#     # n_paretos_list.append(n_paretos)
#     # n_phantoms_list.append(n_phantoms)

        

    # print("Median # of Pareto systems, p =", np.median(n_paretos_list))
    # print("Median # of phantom systems, |P^ph| =", np.median(n_phantoms_list))
    # print("\n")

    