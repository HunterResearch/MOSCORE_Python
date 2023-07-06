import csv
import numpy as np
import time

# from utils import create_allocation_problem
from allocate import allocate, calc_brute_force_rate, calc_phantom_rate, calc_moscore_rate, calc_imoscore_rate
from base import MO_Alloc_Problem

# Repeat fixed-Pareto experiments from Table 3.

n_problems = 10
d = 3  # Number of objectives
r = 10  # Number of systems
p = 5  # Number of Pareto systems

# # rules = ["Brute Force", "Phantom", "MOSCORE", "Brute Force Ind", "iMOSCORE", "Equal"]
# rules = ["Phantom", "MOSCORE", "iMOSCORE", "Equal"]
# # rules = ["MOSCORE", "iMOSCORE", "Equal"]

# solve_times = {rule: [] for rule in rules}
# z_bf_list = {rule: [] for rule in rules}
# z_ph_list = {rule: [] for rule in rules}

# for prob_idx in range(n_problems):
#     obj_vals = {}
#     obj_vars = {}
#     for system_idx in range(r):
#         obj_filename = f"problem_data/TimeProb_3D_10_{prob_idx + 1}_FIX_obj{system_idx + 1}.csv"
#         cov_filename = f"problem_data/TimeProb_3D_10_{prob_idx + 1}_FIX_cov{system_idx + 1}.csv"

#         obj_vec = np.concatenate(list(csv.reader(open(obj_filename, "rt"), delimiter=","))).astype("float")
#         cov_mat = np.array(list(csv.reader(open(cov_filename, "rt"), delimiter=","))).astype("float")

#         obj_vals[system_idx] = obj_vec
#         obj_vars[system_idx] = cov_mat

#     alloc_problem = MO_Alloc_Problem(obj_vals, obj_vars)

#     for rule in rules:
#         print("Solve with", rule, "rule.")
#         tic = time.perf_counter()
#         alpha_hat, z = allocate(method=rule, alloc_problem=alloc_problem)
#         toc = time.perf_counter()
#         solve_time = toc - tic
#         solve_times[rule].append(solve_time)
#         if r <= 10:  # Skip brute force if too expensive.
#             z_bf = calc_brute_force_rate(alphas=alpha_hat, alloc_problem=alloc_problem)
#             z_bf_list[rule].append(z_bf)
#         z_ph = calc_phantom_rate(alphas=alpha_hat, alloc_problem=alloc_problem)
#         z_ph_list[rule].append(z_ph)

# for rule in rules:
#     print("Allocation Rule:", rule)

#     print("Median wall-clock time, T =", round(np.median(solve_times[rule]), 3), "s")
#     print("75-percentile wall-clock time, T =", round(np.quantile(solve_times[rule], 0.75), 3), "s")
#     if r <= 10:  # Skip brute force if too expensive
#         print("Median brute force convergence rate, Z^bf(alpha) x 10^5 =", round(np.median(z_bf_list[rule]) * 10**5, 4))
#     print("Median phantom convergence rate, Z^ph(alpha) x 10^5 =", round(np.median(z_ph_list[rule]) * 10**5, 4))

#     # Extra print statements to compare to MATLAB, problem by problem.
#     if r <= 10:  # Skip brute force if too expensive
#         print("All brute force convergence rates, Z^bf(alpha) x 10^5:", [round(z_bf * 10**5, 4) for z_bf in z_bf_list[rule]])
#     print("All phantom convergence rates, Z^ph(alpha) x 10^5:", [round(z_ph * 10**5, 4) for z_ph in z_ph_list[rule]])
#     print("\n")

# BLOCK TO CHECK A FEW SPECIFIC PROBLEMS
rule = "iMOSCORE"
prob_idx = 1

obj_vals = {}
obj_vars = {}
for system_idx in range(r):
    obj_filename = f"problem_data/TimeProb_3D_10_{prob_idx}_FIX_obj{system_idx + 1}.csv"
    cov_filename = f"problem_data/TimeProb_3D_10_{prob_idx}_FIX_cov{system_idx + 1}.csv"

    obj_vec = np.concatenate(list(csv.reader(open(obj_filename, "rt"), delimiter=","))).astype("float")
    cov_mat = np.array(list(csv.reader(open(cov_filename, "rt"), delimiter=","))).astype("float")

    obj_vals[system_idx] = obj_vec
    obj_vars[system_idx] = cov_mat

alloc_problem = MO_Alloc_Problem(obj_vals, obj_vars)

# # Create a 3D scatter plot of the means.
# import matplotlib.pyplot as plt
# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')

# first_objs = [obj_vals[system_idx][0] for system_idx in range(r)]
# second_objs = [obj_vals[system_idx][1] for system_idx in range(r)]
# third_objs = [obj_vals[system_idx][2] for system_idx in range(r)]
# ax.scatter(first_objs, second_objs, third_objs)
# plt.show()

tic = time.perf_counter()
alpha_hat, z = allocate(method=rule, alloc_problem=alloc_problem)
print(f"Alpha hat sum: {sum(alpha_hat)}.")
toc = time.perf_counter()
print(f"Solve time = {round(toc - tic, 4)}s.")

#print("z (from solver):", round(z, 8))
# alpha_hat, z = allocate(method="Equal", alloc_problem=alloc_problem)
# print("Calculating BF rate")
# z_bf = calc_brute_force_rate(alphas=alpha_hat, alloc_problem=alloc_problem)
print("Calculating Phantom rate")
z_ph = calc_phantom_rate(alpha=alpha_hat, alloc_problem=alloc_problem)
if rule == "MOSCORE":
    print("\nCalculating MOSCORE rate")
    print("MOSCORE MCE/MCI rates at termination (per plugging alpha back in):")
    z_mo = calc_moscore_rate(alpha=alpha_hat, alloc_problem=alloc_problem)
if rule == "iMOSCORE":
    print("Calculating iMOSCORE rate")
    z_imo = calc_imoscore_rate(alpha=alpha_hat, alloc_problem=alloc_problem)

# If using Equal, try this.
# print("Calculating MOSCORE rate")
# z_mo = calc_moscore_rate(alphas=alpha_hat, alloc_problem=alloc_problem)

print(f"\nResults for {rule} on Problem {prob_idx}:")
# print("Brute force convergence rate, Z^bf(alpha) x 10^5:", round(z_bf * 10**5, 4))
print("Phantom convergence rate, Z^ph(alpha) x 10^5:", round(z_ph * 10**5, 4))
if rule == "MOSCORE":
    print("MOSCORE convergence rate, Z^mo(alpha) x 10^5:", round(z_mo * 10**5, 4))
if rule == "iMOSCORE":
    print("iMOSCORE convergence rate, Z^imo(alpha) x 10^5:", round(z_imo * 10**5, 4))
# print("Objectives:", np.array([alloc_problem.obj[system_idx] for system_idx in range(r)]))

#### New print statements
# from allocation import MOSCORE
# print("\nCompute MCE/MCI rates.")
# x = np.append(alpha_hat, 0)
# moscore_object = MOSCORE(alloc_problem=alloc_problem)
# MCE_rates, _ = moscore_object.MCE_rates(x)
# MCI_rates, _ = moscore_object.MCI_rates(x)
#print([MCE_rates, MCI_rates])
#print([z - MCE_rates, z - MCI_rates])
# print("NEXT RATES ARE EQUIVALENT TO THOSE WHEN CALCULATING z^mo")
# print("MCE rates at termination (per plugging alpha back in)", [round(-1 * MCE_rate, 4) for MCE_rate in MCE_rates])
# print("MCI rates at termination (per plugging alpha back in)", [round(-1 * MCI_rate, 4) for MCI_rate in MCI_rates])


print("Allocation:", [round(a / sum(alpha_hat), 4) for a in alpha_hat])
print("Sum of Allocation:", round(sum(alpha_hat), 4))
print("Convergence rate, z* x 10^5:", round(z * 10**5, 6))
