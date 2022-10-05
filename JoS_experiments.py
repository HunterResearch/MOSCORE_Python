#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Summary
-------
A script for reproducing the experiments in the Applegate et al. (2020, JoS) paper.
"""

import numpy as np
import time

from mrg32k3a import MRG32k3a
from utils import create_allocation_problem
from allocation import allocate, smart_allocate, calc_brute_force_rate, calc_phantom_rate
# from base import MORS_Problem, MORS_Solver, MORS_Tester, make_rate_plots, make_phantom_rate_plots
# from example import TestProblem, TestProblem2, TestProblem3, LiEtAl2018Problem

# 3-objective, 3-system example from Li et al. (2018).
obj_vals = {0: [3.0, 4.0, 2.2], 1: [3.5, 5.0, 3.0], 2: [4.0, 3.5, 2.0]}
obj_vars = {0: np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]]),
            1: np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]]),
            2: np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])}
allocation_problem = create_allocation_problem(obj_vals=obj_vals, obj_vars=obj_vars)

for rule in ["Brute Force", "Phantom", "SCORE", "Brute Force Ind", "iSCORE"]:
    tic = time.perf_counter()
    # alpha_hat, z = allocate("iSCORE", allocation_problem, warm_start=warm_start)
    alpha_hat, z = smart_allocate(method=rule, systems=allocation_problem)
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
