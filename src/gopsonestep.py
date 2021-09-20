#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 11:07:46 2021

@author: Sophie Demassey

Run one-step model for testing cuts
bounds are read from a file (.hdf)

"""

import gops
from instance import Instance
from datetime import date
import convexonestep as rel
import outerapproximation as oa
import gurobipy as gp
from gurobipy import GRB
from hydraulics import HydraulicNetwork
import sys
from pathlib import Path

EPSILON = 1e-2
MIPGAP = 1e-4
OUTDIR = Path("../output/")

def drawpoints(inst, model, realflow, realhead, title):
    for (i,j), pipe in inst.pipes.items():
        points = []
        points.append({'x': realflow[(i,j)], 'y': realhead[i] - realhead[j], 'fmt': 'go'})
        points.append({'x': model._qvar[(i,j)].x, 'y': model._hvar[i].x - model._hvar[j].x, 'fmt': 'ro'})
        coeff = [pipe.hloss[2], pipe.hloss[1]]
        oa._draw_oa(pipe.qmin, pipe.qmax, coeff, model._oa[(i, j)][0], model._oa[(i, j)][1], f'({i}, {j}) {title}', 'PIPE', points)
    for (i,j), pump in inst.pumps.items():
        if (i != 'R3'):
            continue
        points = []
        points.append({'x': realflow[(i,j)], 'y': realhead[j] - realhead[i], 'fmt': 'go'})
        points.append({'x': model._qvar[(i,j)].x, 'y': model._hvar[j].x - model._hvar[i].x, 'fmt': 'ro'})
        oa._draw_oa(pump.qmin, pump.qmax, pump.hgain, model._oa[(i, j)][0], model._oa[(i, j)][1], f'({i}, {j}) {title}', 'PUMP', points)

def solveonestep(instance, t, binstring, activity, volinit, headinit, network, epsilon, mipgap):

    stat = None

    #############################################################################
    # solve the one-step convex relaxation with fixed pump status and tank heads

    cvxmodel = rel.build_model(instance, t, activity, headinit, epsilon)
    #cvxmodel.write(f'convrel{t}-{binstring}.lp')
    #cvxmodel.params.timeLimit = 1000 #3600
    cvxmodel.params.MIPGap = mipgap
    cvxmodel.params.OutputFlag = 0
    # cvxmodel.params.Threads = 1
    #cvxmodel.params.FeasibilityTol = 1e-5

    cvxmodel.optimize()

    objval = cvxmodel.objVal if cvxmodel.status == GRB.OPTIMAL else float('inf')

    if cvxmodel.status != GRB.OPTIMAL:
        print('Optimization was stopped with status %d' % cvxmodel.status)

    #############################################################################
    # solve the one-step nonconvex problem with fixed pump status and tank heads

    inactive = {k for k,v in activity.items() if v == 0}
    flow, head = network._flow_analysis(inactive, t, volinit)

    volume = {}
    violation = 0
    for j, tank in instance.tanks.items():
        volume[j] = volinit[j] + 3.6 * instance.tsinhours() \
            * (sum(flow[a] for a in instance.inarcs(j))
               - sum(flow[a] for a in instance.outarcs(j)))

        if volume[j] < tank.vmin or volume[j] > tank.vmax:
            print(f'violation tk={j}: {tank.vmin:.2f} < {volume[j]:2f} < {tank.vmax:.2f}')
            violation += 1
        elif volume[j] > tank.vmax:
            print(f'violation tk={j}: {volume[j]-tank.vmax:.2f}')
            violation += 1

    for a, pump in instance.pumps.items():
        if activity[a] == 1 and (flow[a] < pump.qmin or flow[a] > pump.qmax) :
            print(f'invalid flow in pump {a}: {pump.qmin:.2f} < {flow[a]:.2f} < {pump.qmax:.2f}')
            violation += 1

    #############################################################################
    # check consistency of the convex relaxation

    if cvxmodel.status != GRB.OPTIMAL:
        assert violation > 0, f"a feasible solution exists but not in the convex relaxation"

    else:
        costreal = sum(activity[a] * cvxmodel._svar[a].obj
                   + flow[a] * cvxmodel._qvar[a].obj for a in cvxmodel._svar)

        stat =  f"cost: {objval:.2f}, realcost: {costreal:.2f}, gap: {cvxmodel.MIPGap:.2f}, cpu: {cvxmodel.Runtime:.2f}s, violations: {violation}"
        drawpoints(instance, cvxmodel, flow, head, f't={t}, {binstring}')

    cvxmodel.terminate()
    return stat


def solve(instance, instname, resfilename, epsilon, mipgap):

    print('***********************************************')
    print(instance.tostr_basic())
    print(instance.tostr_network())

    print("obbt: parse bounds")
    try:
        instance.parse_bounds()
    except UnicodeDecodeError as err:
        print(f'obbt bounds not read: {err}')

    activeelem = list(instance.pumps.keys())
    activeelem.extend(instance.valves.keys())
    npermutations = 2 ** len(activeelem)
    print(f"{npermutations} permutations to evaluate")

#    volinit =  {j: tank.vinit for j, tank in instance.tanks.items()}
    volinit =  {j: (tank.vmin + tank.vmax)/2 for j, tank in instance.tanks.items()}
    headinit = {j: tank.head(volinit[j]) for j, tank in instance.tanks.items()}

    network = HydraulicNetwork(instance)

    for t in instance.horizon():
        print(f"demand for time {t}:")
        [print(junc.demand(t)) for j, junc in instance.junctions.items()]

        for n in {0, 1, 3, 7}:
#        for n in range(npermutations):

            binstring = bin(n)[2:].zfill(len(activeelem))
            assert len(binstring) <= len(activeelem), f"{len(binstring)} <= {len(activeelem)}"
            activity = {activeelem[i]: (1 if d=='1' else 0) for i, d in enumerate(binstring)}

            print(f"solve {t} {binstring} ************************************")
            stat = solveonestep(instance, t, binstring, activity, volinit, headinit, network, epsilon, mipgap)
            if stat:
                print(stat)
                f = open(resfilename, 'a')
                f.write(f"{instname}, {t}, {binstring}, {stat}\n")
                f.close()


def solvebench(bench=gops.FASTBENCH, epsilon=EPSILON, mipgap=MIPGAP):
    now = date.today().strftime("%y%m%d")
    resfilename = Path(OUTDIR, f'res1step{now}.csv')

    for k, i in bench.items():
        instance = gops.makeinstance(i)
        solve(instance, k, resfilename, epsilon, mipgap)

def solveinstance(instid, epsilon=EPSILON, mipgap=MIPGAP):
        now = date.today().strftime("%y%m%d")
        resfilename = Path(OUTDIR, f'res1step{now}.csv')
        instance = gops.makeinstance(instid)
        solve(instance, instid, resfilename, epsilon, mipgap)

solveinstance('FSD s 12 1')
#solvebench()



