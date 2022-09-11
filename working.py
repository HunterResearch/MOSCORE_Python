#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Summary
-------
A script for interacting with the MORS Problem, Solver, and Tester classes.
"""

#from pymoso.prng.mrg32k3a import MRG32k3a, get_next_prnstream
from mrg32k3a import MRG32k3a

from base import MORS_Problem, MORS_Solver, MORS_Tester, make_rate_plots, make_phantom_rate_plots
from example import TestProblem, TestProblem2
from allocate import allocate

myproblem = TestProblem2()

# myrng = MRG32k3a()
# myproblem.attach_rng(myrng)

# myproblem.rng_states = [myrng._current_state] * myproblem.n_systems

# system_indices = [0, 0]
# objs = myproblem.bump(system_indices = system_indices)

# myproblem.update_statistics(system_indices = system_indices, objs=objs)

mysolver = MORS_Solver(budget=200,
                       n0=10,
                       delta=10,
                       allocation_rule="Phantom",
                       alpha_epsilon=1e-8,
                       crn_across_solns=False
                       )

mytester = MORS_Tester(solver=mysolver, problem=myproblem)
mytester.run(n_macroreps=10)

# mysolver2 = MORS_Solver(budget=200,
#                        n0=10,
#                        delta=10,
#                        allocation_rule="Equal",
#                        alpha_epsilon=1e-8,
#                        crn_across_solns=False
#                        )
# mytester2 = MORS_Tester(solver=mysolver2, problem=myproblem)
# mytester2.run(n_macroreps=5)


# make_rate_plots(testers=[mytester, mytester2])

# make_phantom_rate_plots(testers=[mytester, mytester2])

# START OF OLD CODE.
# """
# Created on Tue Jul 23 13:59:12 2019

# @author: nathangeldner
# """

# from base import MORS_Tester, solve, solve2
# from utils import testsolve
# from example import MyProblem
# from test_problems import rand_problem_fixed, RandomSequentialProblem, allocation_to_sequential

# Here is a minimum reproduceable example of testsolve:
    
    
# my_tester = MORS_Tester(MyProblem, [0,1,2,3,4,5])

# outs = testsolve(my_tester, MORS_solver, 20, 500, "iSCORE", delta=50, macroreps = 2, proc=1, simpar = 1)


# Here is a minimum reproduceable example of solve:

    
# moop = solve(MyProblem, MORS_Tester, 20, 500, "iSCORE")

# MyProblem here is an example oracle, like in pymoso.


# # Here is a minimum reproduceable example of allocate:
    
# ex_prob = rand_problem_fixed(10, 3, 5)

# boop = allocate("iSCORE", ex_prob)

# # if you want to run a random problem sequentially, you use:
    
    
# seed =  (42, 42, 42, 42, 42, 42)
# my_rng = MRG32k3a(x= seed)

# ex_prob, my_prob = allocation_to_sequential(ex_prob, my_rng)


# shoop = solve2(my_prob, MORS_Tester, 20, 500, "iSCORE")
