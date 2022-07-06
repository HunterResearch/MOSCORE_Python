#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 13:59:12 2019

@author: nathangeldner
"""
from example import MyProblem, TestProblem

from base import MORS_Problem, MORS_Solver, MORS_Tester

#from pymoso.prng.mrg32k3a import MRG32k3a, get_next_prnstream
from mrg32k3a import MRG32k3a

from test_problems import rand_problem_fixed, RandomSequentialProblem

from allocate import allocate


# from base import MORS_Tester, solve, solve2
# from utils import testsolve

myproblem = TestProblem()

# myrng = MRG32k3a()
# myproblem.attach_rng(myrng)

# myproblem.rng_states = [myrng._current_state] * myproblem.n_systems

# system_indices = [0, 0]
# objs = myproblem.bump(system_indices = system_indices)

# myproblem.update_statistics(system_indices = system_indices, objs=objs)

mysolver = MORS_Solver(budget = 200,
                       n0=10,
                       delta=10,
                       allocation_rule="Equal",
                       crn_across_solns=False
                       )

mytester = MORS_Tester(solver=mysolver, problem=myproblem)
mytester.run(n_macroreps=1)

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
