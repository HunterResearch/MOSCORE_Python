#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Summary
-------
An example MORS problem.
"""

from base import MORS_Problem

class TestProblem(MORS_Problem):
    """Example implementation of a user-defined MORS problem."""
    def __init__(self):
        self.n_obj = 2
        self.systems = [(5, 0), (4, 1), (3, 2), (2, 3), (1, 4), (0, 5), \
                        (6, 3), (5, 2), (4, 3), (3, 4), (2, 5), (1, 6), \
                        (7, 1), (6, 2), (5, 3), (4, 4), (3, 5), (2, 6)]
        self.n_systems = len(self.systems)
        self.true_means = [list(self.systems[idx]) for idx in range(self.n_systems)]
        self.true_covs = [[[1, 0], [0, 1]] for _ in range(self.n_systems)]
        super().__init__()

    def g(self, x):
        """Perform a single replication at a given system.
        Obtain a noisy estimate of its objectives.
        
        Parameters
        ----------
        x : tuple
            tuple of values (possibly non-numerical) of inputs
            characterizing the simulatable system
        
        Returns
        -------
        obj : tuple
            tuple of estimates of the objectives
        """
        obj1 = x[0] + self.rng.normalvariate(0, 1)
        obj2 = x[1] + self.rng.normalvariate(0, 1)
        obj = (obj1, obj2)
        return obj


class TestProblem2(MORS_Problem):
    """Example implementation of a user-defined MORS problem."""
    def __init__(self):
        self.n_obj = 2
        self.systems = [(0.5, 0), (0, 0.5), (0.5, 0.5)]
        self.n_systems = len(self.systems)
        self.true_means = [list(self.systems[idx]) for idx in range(self.n_systems)]
        self.true_covs = [[[1, 0], [0, 1]] for _ in range(self.n_systems)]
        super().__init__()

    def g(self, x):
        """Perform a single replication at a given system.
        Obtain a noisy estimate of its objectives.
        
        Parameters
        ----------
        x : tuple
            tuple of values (possibly non-numerical) of inputs
            characterizing the simulatable system
        
        Returns
        -------
        obj : tuple
            tuple of estimates of the objectives
        """
        obj1 = x[0] + self.rng.normalvariate(0, 1)
        obj2 = x[1] + self.rng.normalvariate(0, 1)
        obj = (obj1, obj2)
        return obj


# START OF OLD CODE

# """
# Created on Sun Sep 15 19:15:42 2019

# @author: nathangeldner
# """

# from base import Oracle

# class MyProblem(Oracle):
#     """Example implementation of a user-defined MOSO problem."""
#     def __init__(self, rng, crnflag=False, simpar = 1):
#         """Specify the number of objectives and dimensionality of points."""
#         self.num_obj = 2
#         self.n_systems = 18
#         super().__init__(self.n_systems, rng, crnflag=crnflag, simpar = simpar)

#     def g(self, x, rng):
#         """simulation oracle function goes here"""
#         # initialize obj to empty and is_feas to False
#         systems = {0: (5,0), 1: (4, 1), 2: (3, 2), 3: (2,3), 4: (1,4), 5: (0,5),\
#                    6:(6, 3), 7:(5, 2), 8:(4, 3), 9:(3, 4), 10:(2, 5), 11:(1, 6),\
#                    12:(7, 1), 13:(6, 2), 14:(5, 3), 15:(4,4), 16:(3, 5), 17:(2,6)}
#         obj = []
        
#         #use rng to generate random numbers
#         z0 = rng.normalvariate(0, 1)
#         z1 = rng.normalvariate(0, 1)
#         obj1 = systems[x][0]+ z0
#         obj2 = systems[x][1] + z1
#         obj = (obj1, obj2)
        
#         return obj
