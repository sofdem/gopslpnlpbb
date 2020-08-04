#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 15:10:16 2020

@author: sofdem
"""

import gurobipy as gp
from gurobipy import GRB
import time


def adjust_steplength(instance, network, activity, inactive, timelimit=60):
    """
    Given an unfeasible pumping plan X that violates a tank capacity, adjust the duration 
    for running each configuration X[t] to solve the violations and to minimize the power cost.
    Each configuration X[t] is allowed to start up to one time step earlier and to end up to one 
    time step later.

    Durations are computed by iteratively solving a BIP where hydraulics are approximated and 
    refined at each iteration until a feasible solution (with the nonconvex constraint) is found
    or the timelimit is reached.
    Args:
        instance (instance): the instance.
        network (network): the network associated to the instance.
        activity (dict): the pumping plan as a list of boolean activity[t][a] iff a in X[t].
        inactive (Set): the pumping plan as a list of sets inactive[t] = A \ X[t].

    Returns:
        costlinear (float): the solution linear cost if exists, 0 otherwise.
    """

    starttime = time.time()
    remainingtime = timelimit
    niter = 0
    feasible = False

    model, dvar, vctr = build_model(instance, network, activity, inactive)
    model.params.OutputFlag = 0

    #!!! only small changes at each iteration... stop if the violation is too high
    # find a lower bound, eg. see Richmond: violation at t=12 bc the vinit cannot be reached
    while not feasible and remainingtime >= 0:
        #print(f'primal heuristic: iteration {niter}, remainingtime {remainingtime} s.')
        model.params.timeLimit = remainingtime
        model.optimize()
        #model.write(f'primal{niter}.lp')
        if model.status == GRB.INFEASIBLE:
            print('primal heuristic: unfeasible model')
            break

        feasible, costlinear = update_model(instance, network, inactive, model, dvar, vctr)
        niter += 1
        remainingtime += starttime - time.time()

    return costlinear if feasible else 0


def update_model(inst, network, inactive, model, dvar, vctr):
    """
    Run the extended period flow analysis with the new durations. Check feasibility 
    at the tank capacities, compute the linear cost, and update the coeffs of dvar
    in the objective and in the volume conservation constraints vctr given the new volume estimates
    computed at each subperiod (t,i).

    Args:
        inst (instance): the instance.
        network (network): the network associated to the instance.
        inactive (Set): the pumping plan as a list of sets inactive[t] = A \ X[t].
        model (model): the Gurobi BIP model (should be optimized first !).
        dvar (dict) : gurobi variable dvar[t][i] of the duration of subperiod (t,i)
        vctr (dict): gurobi constraint vctr[t][j] of the volume conservation at tank j on period t

    Returns:
        feasible (Bool): whether a feasible solution has been found.
        cost (float): the solution linear cost if exists, 0 otherwise.

    """

    feasible = True
    vol = {j: tank.vinit for j, tank in inst.tanks.items()}
    cost = 0
    nperiods = inst.nperiods()
    for t in range(nperiods):
        for i, d in dvar[t].items():
            if d:
                duration = d.x
                q, h = network._flow_analysis(inactive[t+i], t, vol)

                # pumping cost on subperiod (t, i)
                subcost = sum(pump.powerval(q[a]) for a, pump in inst.pumps.items()) * inst.tariff[t] / 1000
                dvar[t][i].obj = subcost
                cost += subcost * duration

                # volumes at tanks at the end of subperiod (t, i)
                for j, tank in inst.tanks.items():
                    qtank = sum(q[a] for a in inst.inarcs(j)) - sum(q[a] for a in inst.outarcs(j))
                    model.chgCoeff(vctr[t][j], dvar[t][i], -qtank)
                    vol[j] += qtank * duration
                    vlb = tank.vinit if t == nperiods-1 else tank.vmin
                    if vol[j] < vlb - 1e-6 or vol[j] > tank.vmax + 1e-6:
                        #print(f'primal violation at {t+1} at tank {j}: {vlb:.2f} <= {vol[j]:.2f} <= {tank.vmax:.2f}')
                        feasible = False
    return feasible, cost



def build_model(inst, network, activity, inactive):
    """
    Given an unfeasible pumping plan X that violates a tank capacity, adjust the duration 
    for running each configuration X[t] to solve the violations and to minimize the power cost.

    Each configuration X[t] is allowed to start up to one time step earlier and to end up to one 
    time step later.
    Solve a binary linear program where each period t is partitioned in 3 consecutive subperiods
    (t,i) of variable lengths, each running configuration X[t+i] for i= -1, 0, 1.
    Flows are estimated on each subperiod (t, i) by running the flow analysis for config X[t+i]
    with demands D[t] and fixed volumes V[t,i] where, at the first iteration:
    V[0,0] = Vinit, V[t-1,1] = V[t,-1] = V[t,0] = V[t-1,0] + Q[t-1,0] * DeltaT 
    and at the next iterations (with dur = subperiod duration found at the previous iteration): 
    V[0,0] = Vinit, V[t,-1] = V[t-1,1]+Q[t-1,1]*dur(t-1,1), V[t,i] = V[t,i-1]+Q[t,i-1]*dur(t,i-1)
    and fixed volume (initially vol[t,i] = volume[t] if i=-1 or 0, and vol[t,1] = volume[t+1].
    The tank volume capacities are also slighty more constrained.

    Args:
        inst (instance): the instance.
        network (network): the network associated to the instance.
        activity (dict): the pumping plan as a list of boolean activity[t][a] iff a in X[t].
        inactive (Set): the pumping plan as a list of sets inactive[t] = A \ X[t].

    Returns:
        model (model): the Gurobi BIP model.
        dvar (dict) : gurobi variable dvar[t][i] of the duration of subperiod (t,i)
        vctr (dict): gurobi constraint vctr[t][j] of the volume conservation at tank j on period t
    """

    m = gp.Model('primal')

    dvar = {} # duration dvar[t][i] of subperiod (t,i) for i=-1, 0, 1
    uvar = {} # boolean uvar[t][j] iff dvar[t][i] > 0
    vvar = {0: {j: tank.vinit for j, tank in inst.tanks.items()}} # volume vvar[t][j] of tank j when period t starts
    vctr = {} # constraint vctr[t][j] of volume conservation at tank j during period t

    #!!! in Gratien's code only the changes in the pump config (not valve) is considered
    sameasbefore = True  # no need to define subperiods (t-1, 1) and (t,-1) if inactive[t-1]==inactive[t]

    nperiods = inst.nperiods()
    vol = {j: tank.vinit for j, tank in inst.tanks.items()}
    for t in range(nperiods):
        dvar[t] = {}
        uvar[t] = {}
        vvar[t+1]= {}
        vctr[t] = {}
        fillin = {j: gp.LinExpr() for j in inst.tanks}

        sameasafter = (t == nperiods-1) or (inactive[t] == inactive[t+1])
        for i in range(-1, 2):
            if (i == -1 and sameasbefore) or (i == 1 and sameasafter):
                dvar[t][i] = 0
                uvar[t][i] = 0
            else:
                q, h = network._flow_analysis(inactive[t+i], t, vol)

                # pumping cost on subperiod (t, i)
                power = sum(pump.powerval(q[a]) for a, pump in inst.pumps.items())
                dvar[t][i] = m.addVar(lb=0, ub=inst.tsinhours(), obj=power * inst.tariff[t] / 1000, name=f'd({t},{i})')

                for j in inst.tanks:
                    qtank = sum(q[a] for a in inst.inarcs(j)) - sum(q[a] for a in inst.outarcs(j))
                    fillin[j].addTerms(qtank, dvar[t][i])
                    if i == 0:
                        vol[j] += qtank * inst.tsinhours()

                if i != 0:
                    uvar[t][i] = m.addVar(vtype=GRB.BINARY, name=f'u({t},{i})')
                    m.addConstr(dvar[t][i] <= uvar[t][i] * inst.tsinhours())
                    if i == -1:
                        # either X[t-1] ends later or X[t] starts earlier
                        m.addConstr(uvar[t-1][1] + uvar[t][-1] <= 1)

        sameasbefore = sameasafter

        # period length decomposition
        m.addConstr(gp.quicksum(d for d in dvar[t].values()) == inst.tsinhours())

        # tank volume conservation
        for j, tank in inst.tanks.items():
            vlb = tank.vinit if t == nperiods-1 else tank.vmin
            vvar[t+1][j] = m.addVar(lb=vlb + tank.vmax * 0.001, ub=tank.vmax * 0.999, name=f'v({t+1},{j})')
            vctr[t][j] = m.addConstr(vvar[t+1][j] == vvar[t][j] + fillin[j])

    # minimum final volume
    for j, tank in inst.tanks.items():
        vvar[nperiods][j].lb = tank.vinit + tank.vmax * 0.001

    # PUMP OFF >= 1/2h, ON >= 1h
    dvar[-1] = {1: 0}
    dvar[nperiods] = {-1: 0}
    for k, pump in inst.pumps.items():
        start = 0
        for t in range(1, nperiods+1):
            if t==nperiods or activity[t][k] != activity[t-1][k]:
                end = t
                mind = 1 if activity[t-1][k] else 0.5
                if (end - start - 2) * inst.tsinhours() < mind:
                    m.addConstr((end - start) * inst.tsinhours() + dvar[start-1][1] - dvar[start][-1]
                                + dvar[end][-1] - dvar[end-1][1] >= mind)
                start = end

    m.ModelSense = GRB.MINIMIZE

    return m, dvar, vctr
