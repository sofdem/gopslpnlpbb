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
def build_model(inst: Instance, epsilon: float, pumpvals=None):
    """Build the convex relaxation gurobi model."""

    milp = gp.Model('Pumping_Scheduling')

    qvar = {}  # arc flow
    svar = {}  # pump/valve activity status
    ivar = {}  # pump ignition status
    hvar = {}  # node head

    nperiods = inst.nperiods()
    horizon = inst.horizon()

    for t in horizon:
        for (i, j), pipe in inst.pipes.items():
            qvar[(i, j), t] = milp.addVar(lb=pipe.qmin, ub=pipe.qmax, name=f'qp({i},{j},{t})')

        for (i, j), valve in inst.valves.items():
            qvar[(i, j), t] = milp.addVar(lb=valve.qmin, ub=valve.qmax, name=f'qv({i},{j},{t})')
            svar[(i, j), t] = milp.addVar(vtype=GRB.BINARY, name=f'xv({i},{j},{t})')

        for (i, j), pump in inst.pumps.items():
            ivar[(i, j), t] = milp.addVar(vtype=GRB.BINARY, name=f'ik({i},{j},{t})')
            drawcost = 3.6 * inst.tsinhours() * inst.reservoirs[i].drawcost if i in inst.reservoirs else 0
            qvar[(i, j), t] = milp.addVar(lb=0, ub=pump.qmax, obj=drawcost + inst.eleccost(t) * pump.power[1],
                                          name=f'qk({i},{j},{t})')
            svar[(i, j), t] = milp.addVar(obj=inst.eleccost(t) * pump.power[0], vtype=GRB.BINARY,
                                          name=f'xk({i},{j},{t})')

        for j in inst.junctions:
            hvar[j, t] = milp.addVar(name=f'hj({j},{t})')

        for j, res in inst.reservoirs.items():
            hvar[j, t] = milp.addVar(lb=res.head(t), ub=res.head(t), name=f'hr({j},{t})')

        for j, tank in inst.tanks.items():
            hvar[j, t] = milp.addVar(lb=tank.head(tank.vmin), ub=tank.head(tank.vmax), name=f'ht({j},{t})')

    for j, tank in inst.tanks.items():
        hvar[j, nperiods] = milp.addVar(lb=tank.head(tank.vinit), ub=tank.head(tank.vmax), name=f'ht({j},T)')
        hvar[j, 0].lb = tank.head(tank.vinit)
        hvar[j, 0].ub = tank.head(tank.vinit)
    milp.update()

    # FLOW CONSERVATION (JUNCTIONS)
    for j, junc in inst.junctions.items():
        for t in horizon:
            # !!! original code: round(demand,2)
            milp.addConstr(gp.quicksum(qvar[a, t] for a in inst.inarcs(j))
                           - gp.quicksum(qvar[a, t] for a in inst.outarcs(j)) == junc.demand(t), name=f'fc({j},{t})')

    # FLOW CONSERVATION (TANKS)
    for j, tank in inst.tanks.items():
        for t in horizon:
            milp.addConstr(hvar[j, t + 1] - hvar[j, t] == 3.6 * inst.tsinhours() / tank.surface *
                           (gp.quicksum(qvar[a, t] for a in inst.inarcs(j))
                            - gp.quicksum(qvar[a, t] for a in inst.outarcs(j))), name=f'fc({j},{t})')

    # MAX WITHDRAWAL AT RESERVOIRS
    for j, res in inst.reservoirs.items():
        if res.drawmax:
            milp.addConstr(3.6 * inst.tsinhours()
                           * gp.quicksum(qvar[a, t] for a in inst.outarcs(j) for t in horizon)
                           <= res.drawmax, name=f'w({j})')

    # VALVES
    for (i, j), valve in inst.valves.items():
        if valve.type == 'GV' or valve.type == 'PRV':
            for t in horizon:
                x = svar[(i, j), t] if (valve.type == 'GV') else (1 - svar[(i, j), t])
                milp.addConstr(qvar[(i, j), t] <= valve.qmax * svar[(i, j), t], name=f'vu({i},{j},{t})')
                milp.addConstr(qvar[(i, j), t] >= valve.qmin * x, name=f'vl({i},{j},{t})')
                milp.addConstr(hvar[i, t] - hvar[j, t] >= valve.hlossmin * (1 - svar[(i, j), t]),
                               name=f'hvl({i},{j},{t})')
                milp.addConstr(hvar[i, t] - hvar[j, t] <= valve.hlossmax * (1 - x), name=f'hvu({i},{j},{t})')

    # CONVEXIFICATION OF HEAD-FLOW (PIPES)
    for (i, j), pipe in inst.pipes.items():
        coeff = [pipe.hloss[2], pipe.hloss[1]]
        cutbelow, cutabove = oa.pipecuts(pipe.qmin, pipe.qmax, coeff, (i, j), epsilon, drawgraph=False)
        # print(f'{pipe}: {len(cutbelow)} cutbelow, {len(cutabove)} cutabove')
        for t in horizon:
            for n, c in enumerate(cutbelow):
                milp.addConstr(hvar[i, t] - hvar[j, t] >= c[1] * qvar[(i, j), t] + c[0], name=f'hpl{n}({i},{j},{t})')
            for n, c in enumerate(cutabove):
                milp.addConstr(hvar[i, t] - hvar[j, t] <= c[1] * qvar[(i, j), t] + c[0], name=f'hpu{n}({i},{j},{t})')

    # ACTIVITY (PUMPS)
    for a, pump in inst.pumps.items():
        for t in horizon:
            # !!! original code: integer round ???
            milp.addConstr(qvar[a, t] >= svar[a, t] * pump.qmin)
            milp.addConstr(qvar[a, t] <= svar[a, t] * pump.qmax)

    # CONVEXIFICATION OF HEAD-FLOW (PUMPS)
    for (i, j), pump in inst.pumps.items():
        cutbelow, cutabove = oa.pumpcuts(pump.qmin, pump.qmax, pump.hgain, '(i, j)', epsilon)
        for t in horizon:
            for n, c in enumerate(cutbelow):
                milp.addConstr(hvar[j, t] - hvar[i, t] >= c[1] * qvar[(i, j), t]
                               + (c[0] - pump.offdhmin) * svar[(i, j), t] + pump.offdhmin, name=f'hkl{n}({i},{j},{t})')
            for n, c in enumerate(cutabove):
                # !!! original code: gapabove = 0 if pump.offdhmax == 1000 else pump.offdhmax - c[1] ???
                milp.addConstr(hvar[j, t] - hvar[i, t] <= c[1] * qvar[(i, j), t]
                               + (c[0] - pump.offdhmax) * svar[(i, j), t] + pump.offdhmax, name=f'hku{n}({i},{j},{t})')

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
            for s in inst.dependencies['p0 or p1']:
                milp.addConstr(svar[s[0], t] + svar[s[1], t] >= 1)
            for s in inst.dependencies['p0 <=> p1 xor p2']:
                milp.addConstr(svar[s[0], t] == svar[s[1], t] + svar[s[2], t])
            for s in inst.dependencies['p1 => not p0']:
                milp.addConstr(svar[s[0], t] + svar[s[1], t] <= 1)

    if pumpvals:
        for pump in inst.pumps:
            for t in horizon:
                v = pumpvals.get((pump, t))
                if v == 1:
                    svar[pump, t].lb = 1
                elif v == 0:
                    svar[pump, t].ub = 0

    milp.ModelSense = GRB.MINIMIZE
    milp.update()

    milp._svar = svar
    milp._ivar = ivar
    milp._qvar = qvar
    milp._hvar = hvar
    milp._obj = milp.getObjective()

    return milp
