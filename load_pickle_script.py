#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Summary
-------
A script to load in Tester objects from Figure 3 experiment to
produce Figure 4.
"""

from base import MORS_Tester, make_rate_plots
import pickle

with open('outputs/Equal_with_budget=1000_n0=5_delta=10_mreps=10.pickle','rb') as f:
    equal_tester = pickle.load(f)

with open('outputs/MOSCORE_with_budget=1000_n0=5_delta=10_mreps=10.pickle','rb') as f:
    MOSCORE_tester = pickle.load(f)

with open('outputs/iMOSCORE_with_budget=1000_n0=5_delta=10_mreps=10.pickle','rb') as f:
    iMOSCORE_tester = pickle.load(f)

make_rate_plots(testers=[equal_tester, MOSCORE_tester, iMOSCORE_tester])
