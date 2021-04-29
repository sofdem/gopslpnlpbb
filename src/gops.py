#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 11:07:46 2021

@author: Sophie Demassey

Run the B&B on a subset of the easiest instances
bounds are read from a file (.hdf)

"""

from instance import Instance
from datetime import date
import convexrelaxation as rel
import lpnlpbb as bb
import sys
import graphic
from hydraulics import HydraulicNetwork
from pathlib import Path
from stats import Stat

EPSILON = 1e-2
MIPGAP = 1e-6

BENCH = {
    'FSD': {'ntk': 'Simple_Network', 'D0': 1,  'H0': '/01/2013 00:00'},
    'RIC': {'ntk': 'Richmond',       'D0': 21, 'H0': '/05/2013 07:00'},
    'ANY': {'ntk': 'Anytown',        'D0': 1,  'H0': '/01/2013 00:00'},
    }
PROFILE = {'s': 'Profile_5d_30m_smooth', 'n': 'Profile_5d_30m_smooth'}
STEPLENGTH = {'12': 4, '24': 2, '48': 1}

# ex of instance id: "FSD s 24 3"
def makeinstance(instid: str):
    a = instid.split()
    assert len(a) == 4, f"wrong instance key {instid}"
    d = BENCH[a[0]]
    dstart = f"{(d['D0'] + int(a[3]) - 1):02d}" + d['H0']
    dend   = f"{(d['D0'] + int(a[3])):02d}" + d['H0']
    return Instance(d['ntk'], PROFILE[a[1]], dstart, dend, STEPLENGTH[a[2]])


FASTBENCH = [
    'FSD s 12 1',
    'FSD s 24 1',
    'FSD s 24 2',
    'FSD s 24 3',
    'FSD s 24 4',
    'FSD s 24 5',
    'FSD s 48 1',
    'RIC s 12 3',
    'RIC s 12 4',
    'RIC s 24 3',
    'RIC s 48 3',
]

OUTDIR = Path("../output/")

# possible modes are: None, 'CVX' (solve MIP relaxation), 'SOLVE' (run adjustemnt heur), 'CUT' (cut with adjustment heur),
def solve(instance, epsilon, mipgap, mode, drawsolution, stat=None):

    print('***********************************************')
    print(instance.tostr_basic())
    print(instance.tostr_network())


    print("obbt: parse bounds")
    try:
        instance.parse_bounds()
    except UnicodeDecodeError as err:
        print(f'obbt bounds not read: {err}')


    print("create model")
    cvxmodel = rel.build_model(instance, epsilon)
    # cvxmodel.write('convrel.lp')
    cvxmodel.params.MIPGap = mipgap
    cvxmodel.params.timeLimit = 3600
    # cvxmodel.params.OutputFlag = 0
    # cvxmodel.params.Threads = 1
    #cvxmodel.params.FeasibilityTol = 1e-5


    print("solve model")
    costreal = bb.solveconvex(cvxmodel, instance, drawsolution) if mode=='CVX' \
            else bb.lpnlpbb(cvxmodel, instance, drawsolution, adjust_mode=mode)

    if not stat:
        stat = Stat(mode)
    stat.fill(cvxmodel, costreal)
    print('***********************************************')
    print(f"solution for {instance.tostr_basic()}")
    print(stat.tostr_basic())

    cvxmodel.terminate()
    return stat


def solvebench(bench=FASTBENCH, epsilon=EPSILON, mipgap=MIPGAP, mode='CUT', drawsolution=False):
    stat = Stat(mode)
    now = date.today().strftime("%y%m%d")
    resfilename = Path(OUTDIR, f'res{now}-{mode}.csv')
    f = open(resfilename, 'w')
    f.write(f"gops, {now}, epsilon={epsilon}, mipgap={mipgap}, mode={mode}, non-valid lbs if nogood cuts at feas nodes\n")
    f.write(f'ntk T day, {stat.tocsv_title()}\n')
    f.close()

    for i in bench:
        instance = makeinstance(i)

        stat = solve(instance, epsilon, mipgap, mode, drawsolution, stat)

        f = open(resfilename, 'a')
        f.write(f"{i}, {stat.tocsv_basic()}\n")
        f.close()

def solveinstance(instid, epsilon=EPSILON, mipgap=MIPGAP, mode='CUT', drawsolution=True):
    instance = makeinstance(instid)
    solve(instance, epsilon, mipgap, mode, drawsolution)


def testsolution(instid, solfilename, epsilon=EPSILON, mipgap=MIPGAP, mode='CVX', drawsolution=True):
    instance = makeinstance(instid)
    inactive = instance.parsesolution(solfilename)
    network = HydraulicNetwork(instance, feastol=1e-6)
    flow, hreal, volume, violations = network.extended_period_analysis(inactive, stopatviolation=False)
    cost = sum(instance.eleccost(t) * sum(pump.power[0] + pump.power[1] * flow[t][a]
                                      for a, pump in instance.pumps.items() if not a in inactive[t])
               for t in instance.horizon())

    print(f'real plan cost (without draw cost) = {cost}')
    graphic.pumps(instance, flow)
    graphic.tanks(instance, flow, volume)

    solve(instance, epsilon, mipgap, mode, drawsolution)

#solveinstance('FSD s 24 1', mode=None)
#solveinstance('RIC s 12 3', mode='CUT')
#testsolution('RIC s 12 1', "sol.csv")
solvebench(bench=FASTBENCH[:7], mode='CUT')