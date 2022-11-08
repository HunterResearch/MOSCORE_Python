import csv
import numpy as np
import time

from utils import create_allocation_problem
from allocation import allocate, calc_brute_force_rate, calc_phantom_rate

# Repeat fixed-Pareto experiments from Figures 3.

n_problems = 10
d = 3  # Number of objectives
r = 10  # Number of systems
p = 5 # Number of Pareto systems

for prob_idx in range(n_problems):
    obj_vals = {}
    obj_vars = {}
    for system_idx in range(r):
        obj_filename = f"problem_data/TimeProb_3D_10_{prob_idx + 1}_FIX_obj{system_idx + 1}.csv"
        cov_filename = f"problem_data/TimeProb_3D_10_{prob_idx + 1}_FIX_cov{system_idx + 1}.csv"

        obj_vec = np.concatenate(list(csv.reader(open(obj_filename, "rt"), delimiter=","))).astype("float")
        cov_mat = np.array(list(csv.reader(open(cov_filename, "rt"), delimiter=","))).astype("float")

        obj_vals[system_idx] = obj_vec
        obj_vars[system_idx] = cov_mat

    systems = create_allocation_problem(obj_vals, obj_vars)

    #rules = ["Brute Force", "Phantom", "MOSCORE", "Brute Force Ind", "iMOSCORE", "Equal"]
    rules = ["Phantom", "MOSCORE", "iMOSCORE", "Equal"]
    #rules = ["MOSCORE", "iMOSCORE", "Equal"]

    solve_times = {rule: [] for rule in rules}
    z_bf_list = {rule: [] for rule in rules}
    z_ph_list = {rule: [] for rule in rules}

    for rule in rules:
        print("Solve with", rule, "rule.")
        tic = time.perf_counter()
        alpha_hat, z = allocate(method=rule, systems=systems)
        toc = time.perf_counter()
        solve_time = toc - tic
        if r <= 10:  # Skip brute force if too expensive.
            z_bf = calc_brute_force_rate(alphas=alpha_hat, systems=systems)
        z_ph = calc_phantom_rate(alphas=alpha_hat, systems=systems)

        # Record statistics.
        solve_times[rule].append(solve_time)
        if r <= 10:  # Skip brute force if too expensive.
            z_bf_list[rule].append(z_bf)
        z_ph_list[rule].append(z_ph)

for rule in rules:
    print("Allocation Rule:", rule)

    print("Median wall-clock time, T =", round(np.median(solve_times[rule]), 3), "s")
    print("75-percentile wall-clock time, T =", round(np.quantile(solve_times[rule], 0.75), 3), "s")
    if r <= 10:  # Skip brute force if too expensive
        print("Median brute force convergence rate, Z^bf(alpha) x 10^5 =", round(np.median(z_bf_list[rule]) * 10**5, 4))
    print("Median phantom convergence rate, Z^ph(alpha) x 10^5 =", round(np.median(z_ph_list[rule]) * 10**5, 4))
    print("\n")
