#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Summary
-------
A script for interacting with the MORS Problem, Solver, and Tester classes.
"""

# #from pymoso.prng.mrg32k3a import MRG32k3a, get_next_prnstream
# from mrg32k3a import MRG32k3a

# from base import MORS_Problem, MORS_Solver, MORS_Tester, make_rate_plots, make_phantom_rate_plots
# from example import TestProblem, TestProblem2, TestProblem3
# from allocate import allocate

# myproblem = TestProblem2()

# # myrng = MRG32k3a()
# # myproblem.attach_rng(myrng)

# # myproblem.rng_states = [myrng._current_state] * myproblem.n_systems

# # system_indices = [0, 0]
# # objs = myproblem.bump(system_indices = system_indices)

# # myproblem.update_statistics(system_indices = system_indices, objs=objs)

# mysolver = MORS_Solver(budget=200,
#                        n0=10,
#                        delta=10,
#                        allocation_rule="MOSCORE",
#                        alpha_epsilon=1e-8,
#                        crn_across_solns=False
#                        )

# mytester = MORS_Tester(solver=mysolver, problem=myproblem)
# mytester.run(n_macroreps=10)

# # mysolver2 = MORS_Solver(budget=200,
# #                        n0=10,
# #                        delta=10,
# #                        allocation_rule="Equal",
# #                        alpha_epsilon=1e-8,
# #                        crn_across_solns=False
# #                        )
# # mytester2 = MORS_Tester(solver=mysolver2, problem=myproblem)
# # mytester2.run(n_macroreps=5)


# # make_rate_plots(testers=[mytester, mytester2])

# # make_phantom_rate_plots(testers=[mytester, mytester2])

# # START OF OLD CODE.
# # """
# # Created on Tue Jul 23 13:59:12 2019

# # @author: nathangeldner
# # """

# # from base import MORS_Tester, solve, solve2
# # from utils import testsolve
# # from example import MyProblem
# # from test_problems import rand_problem_fixed, RandomSequentialProblem, allocation_to_sequential

# # Here is a minimum reproduceable example of testsolve:
    
    
# # my_tester = MORS_Tester(MyProblem, [0,1,2,3,4,5])

# # outs = testsolve(my_tester, MORS_solver, 20, 500, "iSCORE", delta=50, macroreps = 2, proc=1, simpar = 1)


# # Here is a minimum reproduceable example of solve:

    
# # moop = solve(MyProblem, MORS_Tester, 20, 500, "iSCORE")

# # MyProblem here is an example oracle, like in pymoso.


# # # Here is a minimum reproduceable example of allocate:
    
# # ex_prob = rand_problem_fixed(10, 3, 5)

# # boop = allocate("iSCORE", ex_prob)

# # # if you want to run a random problem sequentially, you use:
    
    
# # seed =  (42, 42, 42, 42, 42, 42)
# # my_rng = MRG32k3a(x= seed)

# # ex_prob, my_prob = allocation_to_sequential(ex_prob, my_rng)


# # shoop = solve2(my_prob, MORS_Tester, 20, 500, "iSCORE")

# # Test random problem generation.

# import matplotlib.pyplot as plt
# from example import create_fixed_pareto_random_problem, create_variable_pareto_random_problem

# # myprob = create_fixed_pareto_random_problem(n_systems=100, n_obj=2, n_paretos=6, sigma=1, corr=None, center=100, radius=6)
# # obj1 = [myprob["obj"][idx][0] for idx in range(100)]
# # obj2 = [myprob["obj"][idx][1] for idx in range(100)]
# # plt.scatter(obj1, obj2)
# # plt.show()

# myprob = create_variable_pareto_random_problem(n_systems=500, n_obj=3, sigma=1, corr=None, center=100, radius=6)
# obj1 = [myprob["obj"][idx][0] for idx in range(500)]
# obj2 = [myprob["obj"][idx][1] for idx in range(500)]
# obj3 = [myprob["obj"][idx][2] for idx in range(500)]
# plt.scatter(obj1, obj2, obj3)
# plt.show()

# Test relationship between problem solve time and objective gap.

import matplotlib.pyplot as plt
from example import create_fixed_pareto_random_problem, create_variable_pareto_random_problem
from utils import calc_min_obj_gap
import time
from allocation import allocate

n_problems = 20
n_obj = 3
n_systems = 500
n_paretos = 10
method = "iMOSCORE"

min_obj_gaps = []
single_solve_times = []
double_solve_times = []
single_solve_rates = []
double_solve_rates = []
for prob_idx in range(n_problems):
    print(f"Problem {prob_idx + 1} of {n_problems}.")
    random_problem = create_fixed_pareto_random_problem(n_systems=n_systems, n_obj=n_obj, n_paretos=n_paretos, sigma=1, corr=None, center=100, radius=6)
    min_obj_gaps.append(calc_min_obj_gap(systems=random_problem))
    
    # Solve the optimization problem once.
    tic = time.perf_counter()
    alpha_hat, z = allocate(method=method, systems=random_problem, resolve=False)  # Solve opt problem only once.
    toc = time.perf_counter()
    single_solve_times.append(toc - tic)
    single_solve_rates.append(z)

    # Solve the same optimization problem. If insufficiently solved, resolve.
    tic = time.perf_counter()
    alpha_hat, z = allocate(method=method, systems=random_problem, resolve=True)  # Solve opt problem only once.
    toc = time.perf_counter()
    double_solve_times.append(toc - tic)
    double_solve_rates.append(z)

fig, (ax1, ax2) = plt.subplots(2)

fig.suptitle(f"{method} with {n_obj} objs and {n_systems} systems.")

ax1.scatter(min_obj_gaps, double_solve_times, c='blue')
ax1.scatter(min_obj_gaps, single_solve_times, c='red')

ax2.scatter(min_obj_gaps, double_solve_rates, c='blue')
ax2.scatter(min_obj_gaps, single_solve_rates, c='red')

ax1.set(ylabel="Solve Time (s)")
ax2.set(xlabel="Minimum Objective Gap", ylabel="Large Deviations Rates (s)")

plt.show()

# ax1.xlabel("Minimum Objective Gap", size=14)
# ax1.ylabel("Solve Time (s)", size=14)

# ax2.xlabel("Minimum Objective Gap", size=14)
# ax2.ylabel("Large Deviations Rates (s)", size=14)

