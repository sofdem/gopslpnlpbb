#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 11:07:46 2021

@author: Sophie Demassey

Run the B&B on a subset of the easiest instances
bounds are read from a file (.hdf)

"""

from instance import Instance
from datetime import datetime
import convexrelaxation as rel
# import convexrelicae as rel
import lpnlpbb as bb
import csv
import graphic
from hydraulics import HydraulicNetwork
from networkanalysis import NetworkAnalysis
from pathlib import Path
from stats import Stat
import os

OA_GAP = 1e-2
MIP_GAP = 1e-6
TIME_LIMIT = 3600

BENCH = {
    'FSD': {'ntk': 'Simple_Network', 'D0': 1, 'H0': '/01/2013 00:00'},
    'RIC': {'ntk': 'Richmond', 'D0': 21, 'H0': '/05/2013 07:00'},
    'ANY': {'ntk': 'Anytown', 'D0': 1, 'H0': '/01/2013 00:00'},
}
PROFILE = {'s': 'Profile_5d_30m_smooth', 'n': 'Profile_5d_30m_smooth'}
STEPLENGTH = {'12': 4, '24': 2, '48': 1}


# ex of instance id: "FSD s 24 3"
def makeinstance(instid: str):
    a = instid.split()
    assert len(a) == 4, f"wrong instance key {instid}"
    d = BENCH[a[0]]
    dbeg = f"{(d['D0'] + int(a[3]) - 1):02d}" + d['H0']
    dend = f"{(d['D0'] + int(a[3])):02d}" + d['H0']
    return Instance(d['ntk'], PROFILE[a[1]], dbeg, dend, STEPLENGTH[a[2]])


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
defaultfilename = Path(OUTDIR, 'resall.csv')
SOLFILE = Path(OUTDIR, 'solutions.csv')

# !todo remove RECORD and ADJUST
# RECORD (default: gurobi manages incumbent), FATHOM (cut feas int nodes) or CVX (MIP relaxation only)
# NOADJUST (default: no adjustment heuristic), ADJUST (run heur) or ADJUSTNCUT (cut with heur solutions)
MODES = {"solve": ['RECORD', 'FATHOM', 'CVX'],
         "adjust": ['NOADJUST', 'ADJUST', 'ADJUSTNCUT'],
         "obbt": ['C1', 'C0', 'C1icae']}


def parsemode(modes):
    pm = {k: mk[0] for k, mk in MODES.items()}
    if modes is None:
        return pm
    elif type(modes) is str:
        modes = modes.split(" ")
    for k, mk in MODES.items():
        for mode in mk:
            if mode in modes:
                pm[k] = mode
                break
    return pm


def solve(instance, oagap, mipgap, drawsolution, stat, arcvals=None, varvals=None):
    print('***********************************************')
    print(instance.tostr_basic())
    print(instance.tostr_network())

    print("create model")
    cvxmodel = rel.build_model(instance, oagap, arcvals=arcvals)
    if varvals:
        print(f"!!! fixed values !!! {varvals}")
        rel.postvalues(cvxmodel, varvals)

    cvxmodel.write('convrel.lp')
    cvxmodel.params.MIPGap = mipgap
    cvxmodel.params.timeLimit = TIME_LIMIT
    # cvxmodel.params.OutputFlag = 0
    cvxmodel.params.Threads = 1
    # cvxmodel.params.FeasibilityTol = 1e-5

    print("solve model")
    costreal, plan = bb.solveconvex(cvxmodel, instance, drawsolution=drawsolution) if stat.solveconvex() \
        else bb.lpnlpbb(cvxmodel, instance, stat.modes, drawsolution=drawsolution)

    stat.fill(cvxmodel, costreal)
    trace = cvxmodel._trace if not stat.solveconvex() else None
    if costreal:
        cvxmodel.printQuality()
    cvxmodel.printStats()
    cvxmodel.terminate()
    return costreal, plan, trace


def solveinstance(instid, oagap=OA_GAP, mipgap=MIP_GAP, modes=None, drawsolution=True, stat=None, file=defaultfilename):
    stat = Stat(parsemode(modes)) if stat is None else stat
    now = datetime.now().strftime("%y%m%d-%H%M")
    print(now)

    instance = makeinstance(instid)
    if stat.useobbtbounds():
        instance.parse_bounds_obbt(obbtlevel=stat.getobbtmode())

    cost, plan, trace = solve(instance, oagap, mipgap, drawsolution, stat)
    print('***********************************************')
    print(f"solution for {instance.tostr_basic()}")
    print(stat.tostr_basic())
    if trace:
        figname = f"{instid}-{now}"
        plt = graphic.progress(trace, figname)
        if drawsolution:
            plt.show()
        else:
            plt.savefig(Path(OUTDIR, f"trace-{figname.replace(' ', '')}.png"))

    if cost:
        writeplan(instance, plan, f"{now}, {instid}, {cost},")
    fileexists = os.path.exists(file)
    f = open(file, 'a')
    if not fileexists:
        f.write(f"date, oagap, mipgap, ntk T day, {stat.tocsv_title()}\n")
    f.write(f"{now}, {oagap}, {mipgap}, {instid}, {stat.tocsv_basic()}\n")
    f.close()


def writeplan(instance, activity, preamb, solfile=SOLFILE):
    assert len(activity) == instance.nperiods() and len(activity[0]) == len(instance.arcs)
    plan = {a: [activity[t][a] for t in instance.horizon()] for a in instance.varcs}
    f = open(solfile, 'a')
    f.write(f"{preamb} {plan}\n")
    f.close()


def solvebench(bench, oagap=OA_GAP, mipgap=MIP_GAP, modes=None, drawsolution=False):
    stat = Stat(parsemode(modes))
    now = datetime.now().strftime("%y%m%d-%H%M")
    resfilename = Path(OUTDIR, f'res{now}.csv')
    for i in bench:
        solveinstance(i, oagap=oagap, mipgap=mipgap, drawsolution=drawsolution, stat=stat, file=resfilename)


def testplan(instid, solfilename, oagap=OA_GAP, mipgap=MIP_GAP, modes='CVX', drawsolution=True):
    instance = makeinstance(instid)
    inactive = instance.parsesolution(solfilename)
    network = NetworkAnalysis(instance, mipgap)
    # network = HydraulicNetwork(instance, feastol=mipgap)
    flow, volume, nbviolations = network.extended_period_analysis(inactive, stopatviolation=False)
    cost = sum(instance.eleccost(t) * sum(pump.power[0] + pump.power[1] * flow[t][a]
                                          for a, pump in instance.pumps.items() if a not in inactive[t])
               for t in instance.horizon())
    print(f'real plan cost (without draw cost) = {cost} with {nbviolations} violations')
    if drawsolution:
        graphic.pumps(instance, flow)
        graphic.tanks(instance, flow, volume)

    stat = Stat(parsemode(modes))
    if stat.useobbtbounds():
        instance.parse_bounds_obbt(obbtlevel=stat.getobbtmode())
    arcvals = {(a, t): 0 if a in inactive[t] else 1 for a in instance.varcs for t in instance.horizon()}
    solve(instance, oagap, mipgap, drawsolution, stat, arcvals=arcvals)

def testsolution(instid, solfilename, oagap=OA_GAP, mipgap=MIP_GAP, modes='CVX', drawsolution=True):
    instance = makeinstance(instid)
    inactive = instance.parsesolution(solfilename)
    network = NetworkAnalysis(instance, mipgap)
    flow, volume, nbviolations = network.extended_period_analysis(inactive, stopatviolation=False)
    cost = sum(instance.eleccost(t) * sum(pump.power[0] + pump.power[1] * flow[t][a]
                                          for a, pump in instance.pumps.items() if a not in inactive[t])
               for t in instance.horizon())
    print(f'real plan cost (without draw cost) = {cost} with {nbviolations} violations')
    if drawsolution:
        graphic.pumps(instance, flow)
        graphic.tanks(instance, flow, volume)

    varvals = {f"q({arc.id},{t})": flow[t][a] for a, arc in instance.arcs.items() for t in instance.horizon()}
    varvals.update({f"x({instance.arcs[a].id},{t})": 0 if a in inactive[t] else 1 for a in instance.varcs for t in instance.horizon()})
    stat = Stat(parsemode(modes))
    if stat.useobbtbounds():
        instance.parse_bounds_obbt(obbtlevel=stat.getobbtmode())

    solve(instance, oagap, mipgap, drawsolution, stat, varvals=varvals)


def testsolutions(instid, solfilename, oagap=OA_GAP, mipgap=MIP_GAP, modes='CVX', drawsolution=True):
    csvfile = open(solfilename)
    rows = csv.reader(csvfile, delimiter=',')
    data = [[float(x.strip()) for x in row] for row in rows]
    csvfile.close()

    print('************ TEST SOLUTIONS ***********************************')
    instance = makeinstance(instid)
    print(instance.tostr_basic())
    print(instance.tostr_network())

    stat = Stat(parsemode(modes))
    print("create model")
    for i, d in enumerate(data):
        print(f"create model {i}")
        cvxmodel = rel.build_model(instance, oagap)
        rel.postsolution(cvxmodel, d)
        cvxmodel.params.MIPGap = mipgap
        cvxmodel.params.timeLimit = 1200
        # cvxmodel.params.FeasibilityTol = mipgap
        # network = HydraulicNetwork(instance, feastol=feastol)
        # cvxmodel.write("sd.lp")

        print("solve model")
        costreal, plan = bb.solveconvex(cvxmodel, instance, drawsolution=drawsolution) if stat.solveconvex() \
            else bb.lpnlpbb(cvxmodel, instance, stat.modes, drawsolution=drawsolution)

        stat.fill(cvxmodel, costreal)
        print('***********************************************')
        print(f"solution for {instance.tostr_basic()}")
        print(stat.tostr_basic())

        cvxmodel.terminate()


# solveinstance('FSD s 48 1', modes='', drawsolution=False)
# solveinstance('RIC s 12 4', modes='C1', drawsolution=False)
# testplan('RIC s 12 4', Path(OUTDIR, "solric124.csv"), modes="RECORD C1", drawsolution=False)
# testfullsolutions('FSD s 48 4', "solerror.csv", modes="CVX")

solvebench(FASTBENCH[:7], modes='C1', drawsolution=False)
