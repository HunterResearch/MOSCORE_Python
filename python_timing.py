#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 11/06/19

@author: applegae
"""
from scipy import io
import numpy as np
from allocate import allocate
from utils import create_allocation_problem, calc_phantom_rate, calc_bf_rate
from pathlib import Path
import time
import pickle


def RunTimingRow(numobj, ssize, fix, WhichProbs, WSFlag):

    if WhichProbs == "OLD":
        foldname = "TimeProbs"+str(numobj)+"D/"
    elif WhichProbs == "NEW":
        foldname = "NewTimeProbs"+str(numobj)+"D/"
    else:
        ValueError("WhichProbs is not set to OLD or NEW appropriately!")

    path = Path.cwd()
    foldname = "TimeProbs"+str(numobj)+"D/"
    foldername = path.parent / 'Timing' / foldname

    if WSFlag == 1:
        WSstring = "_WS"
    else:
        WSstring = "_noWS"

    if fix == 1:
        #fname = "TimeProb_"+str(numobj)+"D_"+str(ssize)+"_FIX_SUMMARY"+WSstring+".mat"
        sumfname = "TimeProb_"+str(numobj)+"D_"+str(ssize)+"_FIX_SUMMARY"+WSstring+".pickle"
    else:
        #fname = "TimeProb_"+str(numobj)+"D_"+str(ssize)+"_RAN_SUMMARY"+WSstring+".mat"
        sumfname = "TimeProb_"+str(numobj)+"D_"+str(ssize)+"_RAN_SUMMARY"+WSstring+".pickle"

    fn = foldername / sumfname
    if Path(fn).exists():
        print("This row has already been completed!")
        RowSummary(numobj, ssize, fix, WhichProbs, WSFlag, foldername, fn)
        return None

    SumMatrix = np.zeros([10, 18])

    if fix == 1:
        #fname = "TimeProb_"+str(numobj)+"D_"+str(ssize)+"_FIX_SUMMARY"+WSstring+".mat"
        ppfname = "TimeProb_"+str(numobj)+"D_"+str(ssize)+"_FIX_ParPhant.pickle"
    else:
        #fname = "TimeProb_"+str(numobj)+"D_"+str(ssize)+"_RAN_SUMMARY"+WSstring+".mat"
        ppfname = "TimeProb_"+str(numobj)+"D_"+str(ssize)+"_RAN_ParPhant.pickle"

    fn2 = foldername / ppfname
    if not Path(fn2).exists():
        RunParPhantSummary(numobj, ssize, fix, foldername, fn2)

    with open(fn2, 'rb') as parphantf:
        medPars, medPhants = pickle.load(parphantf)
    print("Median Number of Pareto Systems: ", round(medPars))
    print("Median Number of Phantom Pareto Systems: ", round(medPhants))

    # Run Brute Force for one row
    if ssize == 10:
        print("Starting Brute Force allocations")
        [Times, Rates, BFRates] = RunTimingCell(numobj, ssize, foldername, "Brute Force", 1, WSFlag)
        SumMatrix[:, 0] = Times
        SumMatrix[:, 1] = Rates
        SumMatrix[:, 2] = BFRates
        time50 = secs2hms(np.percentile(Times, 50))
        time75 = secs2hms(np.percentile(Times, 75))
        rate50 = np.percentile(Rates, 50)*10**5
        BFrate50 = np.percentile(BFRates, 50)*10**5
        print("Median Time: ", time50)
        print("75th Percentile Time: ", time75)
        print("Median Brute Force Rate (x10^5): ", round(BFrate50, 10))
        print("Median Phantom Rate (x10^5): ", round(rate50, 10))
    else:
        print("Brute Force not run for this row")

    print("=================================================== ")

    # Run Phantom for certain rows
    if (fix == 1 and numobj == 3 and (ssize == 10 or ssize == 500)) or (fix == 0 and (ssize == 50 or ssize == 250)):
        print("Starting Phantom allocations ")
        [Times, Rates, BFRates] = RunTimingCell(numobj, ssize, foldername, "Phantom", fix, WSFlag)
        SumMatrix[:, 3] = Times
        SumMatrix[:, 4] = Rates
        SumMatrix[:, 5] = BFRates
        time50 = secs2hms(np.percentile(Times, 50))
        time75 = secs2hms(np.percentile(Times, 75))
        rate50 = np.percentile(Rates, 50)*10**5
        BFrate50 = np.percentile(BFRates, 50)*10**5
        print("Median Time: ", time50)
        print("75th Percentile Time: ", time75)
        if ssize == 10:
            print("Median Brute Force Rate (x10^5): ", round(BFrate50, 10))
        print("Median Phantom Rate (x10^5): ", round(rate50, 10))
    else:
        print("Phantom not run for this row ")

    print("=================================================== ")

    # Run MO-SCORE for certain rows
    if fix == 1 or (fix == 0 and (numobj == 3 or ssize == 50)):
        print("Starting MO-SCORE allocations ")
        [Times, Rates, BFRates] = RunTimingCell(numobj, ssize, foldername, "SCORE", fix, WSFlag)
        SumMatrix[:, 6] = Times
        SumMatrix[:, 7] = Rates
        SumMatrix[:, 8] = BFRates
        time50 = secs2hms(np.percentile(Times, 50))
        time75 = secs2hms(np.percentile(Times, 75))
        rate50 = np.percentile(Rates, 50)*10**5
        BFrate50 = np.percentile(BFRates, 50)*10**5
        print("Median Time: ", time50)
        print("75th Percentile Time: ", time75)
        if ssize == 10:
            print("Median Brute Force Rate (x10^5): ", round(BFrate50, 10))
        print("Median Phantom Rate (x10^5): ", round(rate50, 10))
    else:
        print("MO-SCORE not run for this row ")

    print("=================================================== ")

    # Run BruteForce Independent for one row
    if ssize == 10:
        print("Starting Brute Force Independent allocations ")
        [Times, Rates, BFRates] = RunTimingCell(
            numobj, ssize, foldername, "Brute Force Ind", 1, WSFlag)
        SumMatrix[:, 9] = Times
        SumMatrix[:, 10] = Rates
        SumMatrix[:, 11] = BFRates
        time50 = secs2hms(np.percentile(Times, 50))
        time75 = secs2hms(np.percentile(Times, 75))
        rate50 = np.percentile(Rates, 50)*10**5
        BFrate50 = np.percentile(BFRates, 50)*10**5
        print("Median Time: ", time50)
        print("75th Percentile Time: ", time75)
        print("Median Brute Force Rate (x10^5): ", round(BFrate50, 10))
        print("Median Phantom Rate (x10^5): ", round(rate50, 10))
    else:
        print("Brute Force Independent not run for this row ")

    print("=================================================== ")

    # Run iMO-SCORE for all rows
    print("Starting iMO-SCORE allocations ")
    [Times, Rates, BFRates] = RunTimingCell(numobj, ssize, foldername, "iSCORE", fix, WSFlag)
    SumMatrix[:, 12] = Times
    SumMatrix[:, 13] = Rates
    SumMatrix[:, 14] = BFRates
    time50 = secs2hms(np.percentile(Times, 50))
    time75 = secs2hms(np.percentile(Times, 75))
    rate50 = np.percentile(Rates, 50)*10**5
    BFrate50 = np.percentile(BFRates, 50)*10**5
    print("Median Time: ", time50)
    print("75th Percentile Time: ", time75)
    if ssize == 10:
        print("Median Brute Force Rate (x10^5): ", round(BFrate50, 10))
    print("Median Phantom Rate (x10^5): ", round(rate50, 10))

    print("=================================================== ")

    # Run Equal for all rows
    print("Starting Equal allocations ")
    [Times, Rates, BFRates] = RunTimingCell(numobj, ssize, foldername, "Equal", fix, WSFlag)
    SumMatrix[:, 15] = Times
    SumMatrix[:, 16] = Rates
    SumMatrix[:, 17] = BFRates
    time50 = secs2hms(np.percentile(Times, 50))
    time75 = secs2hms(np.percentile(Times, 75))
    rate50 = np.percentile(Rates, 50)*10**5
    BFrate50 = np.percentile(BFRates, 50)*10**5
    print("Median Time: ", time50)
    print("75th Percentile Time: ", time75)
    if ssize == 10:
        print("Median Brute Force Rate (x10^5): ", round(BFrate50, 10))
    print("Median Phantom Rate (x10^5): ", round(rate50, 10))

    print("=================================================== ")

    with open(fn, 'wb') as ppfname:
        pickle.dump(SumMatrix, ppfname)
    print(numobj, "Objective, System Size ", ssize, " Row Complete!")

    return None


def RunParPhantSummary(numobj, ssize, fix, foldername, fn2):
    from phantom_allocation import find_phantoms

    ParMatrix = np.zeros(10)
    PhantMatrix = np.zeros(10)

    for prob in range(10):
        # generate filename
        probno = prob+1
        if fix == 1:
            filnam = "TimeProb_"+str(numobj)+"D_"+str(ssize)+"_"+str(probno)+"_FIX.mat"
            fn = foldername / filnam
        else:
            filnam = "TimeProb_"+str(numobj)+"D_"+str(ssize)+"_"+str(probno)+"_RAN.mat"
            fn = foldername / filnam

        data = io.loadmat(fn, squeeze_me=True)
        # get data in desired structure
        systems = data['systems']
        num = systems['num']
        obj = systems['obj']
        cov = systems['cov']
        objs = {}
        covs = {}
        for i in range(len(num)):
            #print("System: ",num[i],"has objective values",obj[i],"and covariance matrix",cov[i])
            objs[i] = list(obj[i])
            covs[i] = cov[i]
        systems = create_allocation_problem(objs, covs)
        # create copy of true_problem to use in allocation
        EstObj = systems

        n_obj = len(systems['obj'][0])
        n_systems = len(systems["obj"])
        num_par = len(systems["pareto_indices"])
        pareto_array = np.zeros([num_par, n_obj])
        for i in range(num_par):
            pareto_array[i, :] = systems['obj'][systems['pareto_indices'][i]]
        phantom_values = find_phantoms(pareto_array, n_obj, num_par)
        n_phantoms = len(phantom_values)

        ParMatrix[prob] = num_par
        PhantMatrix[prob] = n_phantoms

    medPars = np.percentile(ParMatrix, 50)
    medPhants = np.percentile(PhantMatrix, 50)

    with open(fn2, 'wb') as ppfname:
        pickle.dump([medPars, medPhants], ppfname)

    print("Paretos/Phants determined for row", numobj, "-", ssize)
    return None


def RunTimingCell(numobj, ssize, foldername, algorithm, fix, WSFlag):
    Times = np.zeros(10)
    Rates = np.zeros(10)
    BFRates = np.zeros(10)
    for prob in range(10):
        probno = prob+1
        # generate filename
        if fix == 1:
            filnam = "TimeProb_"+str(numobj)+"D_"+str(ssize)+"_"+str(probno)+"_FIX.mat"
            fn = foldername / filnam
        else:
            filnam = "TimeProb_"+str(numobj)+"D_"+str(ssize)+"_"+str(probno)+"_RAN.mat"
            fn = foldername / filnam

        print("Problem ", probno, "...")
        [time, rate, BFrate, allo, z] = RunTimingProblem(fn, algorithm, ssize, WSFlag)

        Times[prob] = time
        Rates[prob] = rate
        BFRates[prob] = BFrate
    return Times, Rates, BFRates


def RunTimingProblem(probfile, algorithm, ssize, WSFlag):
    data = io.loadmat(probfile, squeeze_me=True)
    # get data in desired structure
    systems = data['systems']
    num = systems['num']
    obj = systems['obj']
    cov = systems['cov']
    objs = {}
    covs = {}
    for i in range(len(num)):
        #print("System: ",num[i],"has objective values",obj[i],"and covariance matrix",cov[i])
        objs[i] = list(obj[i])
        covs[i] = cov[i]
    systems = create_allocation_problem(objs, covs)
    # create copy of true_problem to use in allocation
    EstObj = systems

    # start timing and run allocation
    t_0 = time.time()
    [allo, z] = allocate(algorithm, EstObj, WSFlag)
    t_1 = time.time()
    allotime = t_1-t_0

    print("Calculating Phantom Rate...")
    rate = calc_phantom_rate(allo, systems)

    if ssize == 10:
        print("Calculating BruteForce Rate...")
        BFrate = calc_bf_rate(allo, systems)
    else:
        BFrate = -1

    print("Problem Complete!")

    return allotime, rate, BFrate, allo, z


def secs2hms(time_in_secs):
    # this sub-function converts a time (real number representing seconds) into
    # a string that displays the time in hours/minutes/seconds
    from math import floor
    time_string = str()
    nhours = 0
    nmins = 0
    if time_in_secs >= 3600:
        nhours = floor(time_in_secs/3600)
        if nhours > 1:
            hour_string = " hours, "
        else:
            hour_string = " hour, "
        time_string = str(nhours)+hour_string
    if time_in_secs >= 60:
        nmins = floor((time_in_secs - 3600*nhours)/60)
        if nmins > 1:
            minute_string = " mins, "
        else:
            minute_string = " min, "
        time_string = time_string+str(nmins)+minute_string
    nsecs = time_in_secs - 3600*nhours - 60*nmins
    if nmins > 1:
        secstring = str(round(nsecs))
        time_string = time_string+secstring+" secs"
    else:
        secstring = str(round(nsecs, 4))
        time_string = time_string+secstring+" secs"

    return time_string


def RowSummary(numobj, ssize, fix, WhichProbs, WSFlag, foldername, fn):

    with open(fn, 'rb') as summf:
        SumMatrix = pickle.load(summf)

    if fix == 1:
        #fname = "TimeProb_"+str(numobj)+"D_"+str(ssize)+"_FIX_SUMMARY"+WSstring+".mat"
        ppfname = "TimeProb_"+str(numobj)+"D_"+str(ssize)+"_FIX_ParPhant.pickle"
    else:
        #fname = "TimeProb_"+str(numobj)+"D_"+str(ssize)+"_RAN_SUMMARY"+WSstring+".mat"
        ppfname = "TimeProb_"+str(numobj)+"D_"+str(ssize)+"_RAN_ParPhant.pickle"

    fn2 = foldername / ppfname
    with open(fn2, 'rb') as parphantf:
        medPars, medPhants = pickle.load(parphantf)
    print("Median Number of Pareto Systems: ", round(medPars))
    print("Median Number of Phantom Pareto Systems: ", round(medPhants))

    # Run Brute Force for one row
    if ssize == 10:
        print("Brute Force")
        Times = SumMatrix[:, 0]
        Rates = SumMatrix[:, 1]
        BFRates = SumMatrix[:, 2]
        time50 = secs2hms(np.percentile(Times, 50))
        time75 = secs2hms(np.percentile(Times, 75))
        rate50 = np.percentile(Rates, 50)*10**5
        BFrate50 = np.percentile(BFRates, 50)*10**5
        print("Median Time: ", time50)
        print("75th Percentile Time: ", time75)
        print("Median Brute Force Rate (x10^5): ", round(BFrate50, 10))
        print("Median Phantom Rate (x10^5): ", round(rate50, 10))
    else:
        print("Brute Force not run for this row")

    print("=================================================== ")

    # Run Phantom for certain rows
    if (fix == 1 and numobj == 3 and (ssize == 10 or ssize == 500)) or (fix == 0 and (ssize == 50 or ssize == 250)):
        print("Phantom ")
        Times = SumMatrix[:, 3]
        Rates = SumMatrix[:, 4]
        BFRates = SumMatrix[:, 5]
        time50 = secs2hms(np.percentile(Times, 50))
        time75 = secs2hms(np.percentile(Times, 75))
        rate50 = np.percentile(Rates, 50)*10**5
        BFrate50 = np.percentile(BFRates, 50)*10**5
        print("Median Time: ", time50)
        print("75th Percentile Time: ", time75)
        if ssize == 10:
            print("Median Brute Force Rate (x10^5): ", round(BFrate50, 10))
        print("Median Phantom Rate (x10^5): ", round(rate50, 10))
    else:
        print("Phantom not run for this row ")

    print("=================================================== ")

    # Run MO-SCORE for certain rows
    if fix == 1 or (fix == 0 and (numobj == 3 or ssize == 50)):
        print("MO-SCORE ")
        Times = SumMatrix[:, 6]
        Rates = SumMatrix[:, 7]
        BFRates = SumMatrix[:, 8]
        time50 = secs2hms(np.percentile(Times, 50))
        time75 = secs2hms(np.percentile(Times, 75))
        rate50 = np.percentile(Rates, 50)*10**5
        BFrate50 = np.percentile(BFRates, 50)*10**5
        print("Median Time: ", time50)
        print("75th Percentile Time: ", time75)
        if ssize == 10:
            print("Median Brute Force Rate (x10^5): ", round(BFrate50, 10))
        print("Median Phantom Rate (x10^5): ", round(rate50, 10))
    else:
        print("MO-SCORE not run for this row ")

    print("=================================================== ")

    # Run BruteForce Independent for one row
    if ssize == 10:
        print("Brute Force Independent ")
        Times = SumMatrix[:, 9]
        Rates = SumMatrix[:, 10]
        BFRates = SumMatrix[:, 11]
        time50 = secs2hms(np.percentile(Times, 50))
        time75 = secs2hms(np.percentile(Times, 75))
        rate50 = np.percentile(Rates, 50)*10**5
        BFrate50 = np.percentile(BFRates, 50)*10**5
        print("Median Time: ", time50)
        print("75th Percentile Time: ", time75)
        print("Median Brute Force Rate (x10^5): ", round(BFrate50, 10))
        print("Median Phantom Rate (x10^5): ", round(rate50, 10))
    else:
        print("Brute Force Independent not run for this row ")

    print("=================================================== ")

    # Run iMO-SCORE for all rows
    print("iMO-SCORE ")
    Times = SumMatrix[:, 12]
    Rates = SumMatrix[:, 13]
    BFRates = SumMatrix[:, 14]
    time50 = secs2hms(np.percentile(Times, 50))
    time75 = secs2hms(np.percentile(Times, 75))
    rate50 = np.percentile(Rates, 50)*10**5
    BFrate50 = np.percentile(BFRates, 50)*10**5
    print("Median Time: ", time50)
    print("75th Percentile Time: ", time75)
    if ssize == 10:
        print("Median Brute Force Rate (x10^5): ", round(BFrate50, 10))
    print("Median Phantom Rate (x10^5): ", round(rate50, 10))

    print("=================================================== ")

    # Run Equal for all rows
    print("Equal ")
    Times = SumMatrix[:, 15]
    Rates = SumMatrix[:, 16]
    BFRates = SumMatrix[:, 17]
    time50 = secs2hms(np.percentile(Times, 50))
    time75 = secs2hms(np.percentile(Times, 75))
    rate50 = np.percentile(Rates, 50)*10**5
    BFrate50 = np.percentile(BFRates, 50)*10**5
    print("Median Time: ", time50)
    print("75th Percentile Time: ", time75)
    if ssize == 10:
        print("Median Brute Force Rate (x10^5): ", round(BFrate50, 10))
    print("Median Phantom Rate (x10^5): ", round(rate50, 10))

    print("=================================================== ")

    return None


# ==========================================================================
# Run the 'large' 4D and 5D RANDOM problems? (We recommend to run on a
# high-performance computing cluster - 4D, 5000 systems and 5D, 2000 systems)
BigProbs = 'YES'  # enter 'YES' or 'NO'
# ==========================================================================
# Run the existing problems 'OLD' or randomly generated 'NEW' problems?
WhichProbs = 'OLD'  # enter 'OLD' or 'NEW'
# ==========================================================================
WSFlag = 0  # a 0/1 value that determines whether to use
#           warm_starts for certain algorithms when d>3
# ==========================================================================

# Run FIXED Pareto problems
print(" ******** FIXED Pareto Timing Rows ********")
for numobj in range(3, 6):

    if numobj == 3:
        sys_sizes = np.array([10, 500, 10000])
    elif numobj == 4:
        sys_sizes = np.array([5000, 10000])
    else:
        sys_sizes = np.array([10000])

    syslen = len(sys_sizes)
    for sysind in range(syslen):
        ssize = sys_sizes[sysind]
        print("** FIXED Timing Table, ", numobj, "- Objectives, ", ssize, " Systems ** ")
        RunTimingRow(numobj, ssize, 1, WhichProbs, WSFlag)

# Run RANDOM Pareto problems
print(" ******** RANDOM Pareto Timing Rows ******** ")
for numobj in range(3, 6):

    if numobj == 3:
        sys_sizes = np.array([250, 5000, 10000])
    elif numobj == 4:
        if BigProbs == 'YES':
            sys_sizes = np.array([50, 1000, 2000, 5000])
        else:
            sys_sizes = np.array([50, 1000, 2000])
    else:
        if BigProbs == 'YES':
            sys_sizes = np.array([50, 2000])
        else:
            sys_sizes = np.array([50])

    syslen = len(sys_sizes)
    for sysind in range(syslen):
        ssize = sys_sizes[sysind]
        print("** RANDOM Timing Table, ", numobj, "- Objectives, ", ssize, " Systems ** ")
        RunTimingRow(numobj, ssize, 0, WhichProbs, WSFlag)

print("Complete!")
