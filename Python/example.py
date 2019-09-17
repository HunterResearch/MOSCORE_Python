#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 19:15:42 2019

@author: nathangeldner
"""

from base import Oracle

class MyProblem(Oracle):
    """Example implementation of a user-defined MOSO problem."""
    def __init__(self, rng, crnflag=False, simpar = 1):
        """Specify the number of objectives and dimensionality of points."""
        self.num_obj = 2
        self.n_systems = 18
        super().__init__(self.n_systems, rng, crnflag=crnflag, simpar = simpar)

    def g(self, x, rng):
        """simulation oracle function goes here"""
        # initialize obj to empty and is_feas to False
        systems = {0: (5,0), 1: (4, 1), 2: (3, 2), 3: (2,3), 4: (1,4), 5: (0,5),\
                   6:(6, 3), 7:(5, 2), 8:(4, 3), 9:(3, 4), 10:(2, 5), 11:(1, 6),\
                   12:(7, 1), 13:(6, 2), 14:(5, 3), 15:(4,4), 16:(3, 5), 17:(2,6)}
        obj = []
        
        #use rng to generate random numbers
        z0 = rng.normalvariate(0, 1)
        z1 = rng.normalvariate(0, 1)
        obj1 = systems[x][0]+ z0
        obj2 = systems[x][1] + z1
        obj = (obj1, obj2)
        
        return obj

        