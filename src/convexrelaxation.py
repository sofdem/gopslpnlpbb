#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 11:07:46 2020

@author: Sophie Demassey, Gratien Bonvin
"""

import gurobipy as gp
from gurobipy import GRB
import outerapproximation as oa
from instance import Instance
import datetime as dt


# !!! check round values
# noinspection PyArgumentList
def build_model(inst: Instance, oagap: float, arcvals=None):
    """Build the convex relaxation gurobi model."""

    milp = gp.Model('Pumping_Scheduling')

    qvar = {}  # arc flow
    dhvar = {}  # arc hloss
    svar = {}  # arc status
    ivar = {}  # pump ignition status
    hvar = {}  # node head
    qexpr = {}  # node inflow

    nperiods = inst.nperiods()
    horizon = inst.horizon()

    for t in horizon:
        for (i, j), pump in inst.pumps.items():
            ivar[(i, j), t] = milp.addVar(vtype=GRB.BINARY, name=f'ik({i},{j},{t})')

        for j in inst.junctions:
            hvar[j, t] = milp.addVar(name=f'hj({j},{t})')

        for j, res in inst.reservoirs.items():
            hvar[j, t] = milp.addVar(lb=res.head(t), ub=res.head(t), name=f'hr({j},{t})')

        for j, tank in inst.tanks.items():
            lbt = tank.head(tank.vinit) if t == 0 else tank.head(tank.vmin)
            ubt = tank.head(tank.vinit) if t == 0 else tank.head(tank.vmax)
            hvar[j, t] = milp.addVar(lb=lbt, ub=ubt, name=f'ht({j},{t})')

        milp.update()

        for (i, j), a in inst.arcs.items():

            if a.control:
                qvar[(i, j), t] = milp.addVar(lb=-GRB.INFINITY, name=f'q({i},{j},{t})')
                dhvar[(i, j), t] = milp.addVar(lb=-GRB.INFINITY, name=f'H({i},{j},{t})')
                svar[(i, j), t] = milp.addVar(vtype=GRB.BINARY, name=f'x({i},{j},{t})')
                # q_a=0 if x_a=0 otherwise in [qmin,qmax]
                milp.addConstr(qvar[(i, j), t] <= a.qmax * svar[(i, j), t])
                milp.addConstr(qvar[(i, j), t] >= a.qmin * svar[(i, j), t])
                # dh_a = (h_i - h_j) * x_a
                dhmin = max(a.hlossval(a.qmin), hvar[i, t].lb - hvar[j, t].ub)
                dhmax = min(a.hlossval(a.qmax), hvar[i, t].ub - hvar[j, t].lb)
                milp.addConstr(dhvar[(i, j), t] <= dhmax * svar[(i, j), t])
                milp.addConstr(dhvar[(i, j), t] >= dhmin * svar[(i, j), t])
                milp.addConstr(dhvar[(i, j), t] <= hvar[i, t] - hvar[j, t] - a.dhmin * (1-svar[(i, j), t]))
                milp.addConstr(dhvar[(i, j), t] >= hvar[i, t] - hvar[j, t] - a.dhmax * (1-svar[(i, j), t]))

            else:
                qvar[(i, j), t] = milp.addVar(lb=a.qmin, ub=a.qmax, name=f'q({i},{j},{t})')
                dhvar[(i, j), t] = milp.addVar(lb=a.hlossval(a.qmin), ub=a.hlossval(a.qmax), name=f'H({i},{j},{t})')
                svar[(i, j), t] = milp.addVar(vtype=GRB.BINARY, lb=1, name=f'x({i},{j},{t})')
                milp.addConstr(dhvar[(i, j), t] == hvar[i, t] - hvar[j, t])

    for j, tank in inst.tanks.items():
        hvar[j, nperiods] = milp.addVar(lb=tank.head(tank.vinit), ub=tank.head(tank.vmax), name=f'ht({j},T)')

    milp.update()

    # FLOW CONSERVATION
    for t in horizon:
        for j in inst.nodes:
            qexpr[j, t] = gp.quicksum(qvar[a, t] for a in inst.inarcs(j)) - gp.quicksum(qvar[a, t] for a in inst.outarcs(j))

        for j, junc in inst.junctions.items():
            milp.addConstr(gp.quicksum(qvar[a, t] for a in inst.inarcs(j))
                           - gp.quicksum(qvar[a, t] for a in inst.outarcs(j)) == junc.demand(t), name=f'fc({j},{t})')

        for j, tank in inst.tanks.items():
            milp.addConstr(hvar[j, t+1] - hvar[j, t] == (3.6 * inst.tsinhours() / tank.surface) * qexpr[j, t], name=f'fc({j},{t})')

    # MAX WITHDRAWAL AT RESERVOIRS
    for j, res in inst.reservoirs.items():
        if res.drawmax:
            milp.addConstr(res.drawmax + 3.6 * inst.tsinhours() * gp.quicksum(qexpr[j, t] for t in horizon), name=f'w({j})')

    # CONVEXIFICATION OF HEAD-FLOW
    for (i, j), arc in inst.arcs.items():
        cutbelow, cutabove = oa.hlossoa(arc.qmin, arc.qmax, arc.hloss, (i, j), oagap, drawgraph=False)
        print(f'{arc}: {len(cutbelow)} cutbelow, {len(cutabove)} cutabove')
        for t in horizon:
            x = svar[(i, j), t] if arc.control else 1
            for n, c in enumerate(cutbelow):
                milp.addConstr(dhvar[(i, j), t] >= c[1] * qvar[(i, j), t] + c[0] * x, name=f'hpl{n}({i},{j},{t})')
            for n, c in enumerate(cutabove):
                milp.addConstr(dhvar[(i, j), t] <= c[1] * qvar[(i, j), t] + c[0] * x, name=f'hpu{n}({i},{j},{t})')

    # PUMP SWITCHING
    sympumps = inst.symmetries
    uniquepumps = inst.pumps_without_sym()
    print('symmetries:', uniquepumps)

    def getv(vdict, pump, t):
        return gp.quicksum(vdict[a, t] for a in sympumps) if pump == 'sym' else vdict[pump, t]

    # !!! check the max ignition constraint for the symmetric group
    # !!! make ivar[a,0] = svar[a,0]
    for a in uniquepumps:
        rhs = 6 * len(sympumps) if a == 'sym' else 6 - svar[a, 0]
        milp.addConstr(gp.quicksum(getv(ivar, a, t) for t in range(1, nperiods)) <= rhs)
        for t in range(1, nperiods):
            milp.addConstr(getv(ivar, a, t) >= getv(svar, a, t) - getv(svar, a, t - 1))
            if inst.tsduration == dt.timedelta(minutes=30) and t < inst.nperiods() - 1:
                # minimum 1 hour activity
                milp.addConstr(getv(svar, a, t + 1) + getv(svar, a, t - 1) >= getv(svar, a, t))

    # PUMP DEPENDENCIES
    if sympumps:
        for t in horizon:
            for i, pump in enumerate(sympumps[:-1]):
                milp.addConstr(ivar[pump, t] >= ivar[sympumps[i + 1], t])
                milp.addConstr(svar[pump, t] >= svar[sympumps[i + 1], t])

    if inst.dependencies:
        for t in horizon:
            for s in inst.dependencies['p1 => p0']:
                milp.addConstr(svar[s[0], t] >= svar[s[1], t])
            for s in inst.dependencies['p0 xor p1']:
                milp.addConstr(svar[s[0], t] + svar[s[1], t] >= 1)
            for s in inst.dependencies['p0 = p1 xor p2']:
                milp.addConstr(svar[s[0], t] == svar[s[1], t] + svar[s[2], t])
            # for s in inst.dependencies['p1 => not p0']:
            #    milp.addConstr(svar[s[0], t] + svar[s[1], t] <= 1)

    if arcvals:
        for a in inst.varcs:
            for t in horizon:
                v = arcvals.get((a, t))
                if v == 1:
                    svar[a, t].lb = 1
                elif v == 0:
                    svar[a, t].ub = 0

    obj = gp.quicksum(inst.eleccost(t) * (pump.power[0] * svar[k, t] + pump.power[1] * qvar[k, t])
                      for k, pump in inst.pumps.items() for t in horizon)

    milp.setObjective(obj, GRB.MINIMIZE)
    milp.update()

    milp._svar = svar
    milp._ivar = ivar
    milp._qvar = qvar
    milp._hvar = hvar
    milp._obj = obj

    return milp


def postsolution(model, vals, precision=1e-6):
    i = 0
    for var in model._svar:
        var.lb = vals[i]
        var.ub = vals[i]
        i += 1
    for var in model._qvar:
        var.lb = vals[i] - precision
        var.ub = vals[i] + precision
        i += 1
#    for var in model._hvar:
#        var.lb = vals[i] - precision
#        var.ub = vals[i] + precision
#        i += 1
