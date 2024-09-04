#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Summary
-------
A script for reproducing the experiments in the Applegate et al. (2020, JoS) paper.
"""

from example import MOCBA_25_Problem
from base import MORS_Solver, MORS_Tester, make_phantom_rate_plots


test_problem = MOCBA_25_Problem(cov_type="ind")
common_budget = 150  # n=75000 in the paper

equal_solver = MORS_Solver(budget=common_budget,
                           n0=5,
                           delta=10,
                           allocation_rule="Equal",
                           alpha_epsilon=1e-8,
                           crn_across_solns=False
                           )
MOSCORE_solver = MORS_Solver(budget=common_budget,
                             n0=5,
                             delta=10,
                             allocation_rule="MOSCORE",
                             alpha_epsilon=1e-8,
                             crn_across_solns=False
                             )
iMOSCORE_solver = MORS_Solver(budget=common_budget,
                              n0=5,
                              delta=10,
                              allocation_rule="iMOSCORE",
                              alpha_epsilon=1e-8,
                              crn_across_solns=False
                              )
phantom_solver = MORS_Solver(budget=common_budget,
                             n0=5,
                             delta=10,
                             allocation_rule="Phantom",
                             alpha_epsilon=1e-8,
                             crn_across_solns=False
                             )

equal_tester = MORS_Tester(solver=equal_solver, problem=test_problem)
equal_tester.run(n_macroreps=10)  # 5000 mreps in the paper

MOSCORE_tester = MORS_Tester(solver=MOSCORE_solver, problem=test_problem)
MOSCORE_tester.run(n_macroreps=10)  # 5000 mreps in the paper

iMOSCORE_tester = MORS_Tester(solver=iMOSCORE_solver, problem=test_problem)
iMOSCORE_tester.run(n_macroreps=10)  # 5000 mreps in the paper

phantom_tester = MORS_Tester(solver=phantom_solver, problem=test_problem)
phantom_tester.run(n_macroreps=10)  # 5000 mreps in the paper

make_phantom_rate_plots(testers=[equal_tester, MOSCORE_tester, iMOSCORE_tester, phantom_tester])
