#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  22 11:07:46 2021

@author: Sophie Demassey
"""

import gurobipy as gp
from gurobipy import GRB
import outerapproximation as oa
from instance import Instance


# !!! check round values
def build_model(inst: Instance, step: int, pumpval, headval, epsilon):
    """Build the convex relaxation gurobi model."""

    LP = gp.Model('OneStepPump')

    qvar = {}  # arc flow
    svar = {}  # pump/valve activity status
    hvar = {}  # node head

    for (i, j), pipe in inst.pipes.items():
        qvar[(i, j)] = LP.addVar(lb=pipe.qmin, ub=pipe.qmax, name=f'qp({i},{j})')

    for (i, j), valve in inst.valves.items():
        qvar[(i, j)] = LP.addVar(lb=valve.qmin, ub=valve.qmax, name=f'qv({i},{j})')
        svar[(i, j)] = LP.addVar(vtype=GRB.BINARY, name=f'xv({i},{j})', lb=pumpval.get(valve, 0), ub=pumpval.get(valve, 1))

    for (i, j), pump in inst.pumps.items():
        addQcost = 3.6 * inst.tsinhours() * inst.reservoirs[i].drawcost if i in inst.reservoirs else 0
        qvar[(i, j)] = LP.addVar(lb=0, ub=pump.qmax, obj=addQcost + inst.eleccost(step) * pump.power[1], name=f'qk({i},{j})')
        svar[(i, j)] = LP.addVar(obj=inst.eleccost(step) * pump.power[0], vtype=GRB.BINARY, name=f'xk({i},{j})',
                                 lb=pumpval.get((i,j), 0), ub=pumpval.get((i,j), 1))
        
#        print(f'xk({i},{j}: lb = {pumpval.get((i,j), 0)}, ub = {pumpval.get((i,j), 1)}, pumpval = {pumpval.get((i,j))}')

    for j in inst.junctions:
        hvar[j] = LP.addVar(name=f'hj({j})')

    for j, res in inst.reservoirs.items():
        hvar[j] = LP.addVar(lb=res.head(step), ub=res.head(step), name=f'hr({j})')

    for j, tank in inst.tanks.items():
        hvar[j] = LP.addVar(lb=tank.head(tank.vmin), ub=tank.head(tank.vmax), name=f'ht({j})')

    LP.update()

    # FLOW CONSERVATION (JUNCTIONS)
    for j, junc in inst.junctions.items():
        LP.addConstr(gp.quicksum(qvar[a] for a in inst.inarcs(j))
                     - gp.quicksum(qvar[a] for a in inst.outarcs(j)) == junc.demand(step), name=f'fc({j})')

    # FLOW CONSERVATION (TANKS)
    for j, tank in inst.tanks.items():
        LP.addConstr(hvar[j] - headval[j] == 3.6 * inst.tsinhours() / tank.surface *
                     (gp.quicksum(qvar[a] for a in inst.inarcs(j))
                      - gp.quicksum(qvar[a] for a in inst.outarcs(j))), name=f'fc({j})')

    # VALVES
    for (i, j), valve in inst.valves.items():
        if valve.type == 'GV' or valve.type == 'PRV':
            x = svar[(i, j)] if (valve.type == 'GV') else (1 - svar[(i, j)])
            LP.addConstr(qvar[(i, j)] <= valve.qmax * svar[(i, j)], name=f'vu({i},{j})')
            LP.addConstr(qvar[(i, j)] >= valve.qmin * x, name=f'vl({i},{j})')
            LP.addConstr(hvar[i] - hvar[j] >= valve.hlossmin * (1 - svar[(i, j)]), name=f'hvl({i},{j})')
            LP.addConstr(hvar[i] - hvar[j] <= valve.hlossmax * (1 - x), name=f'hvu({i},{j})')

    # CONVEXIFICATION OF HEAD-FLOW (PIPES)
    LP._oa = {}
    for (i, j), pipe in inst.pipes.items():
        coeff = [pipe.hloss[2], pipe.hloss[1]]
        cutbelow, cutabove = oa.pipecuts(pipe.qmin, pipe.qmax, coeff, f'({i}, {j})', epsilon)
        # print(f'{pipe}: {len(cutbelow)} cutbelow, {len(cutabove)} cutabove')
        for n, c in enumerate(cutbelow):
            LP.addConstr(hvar[i] - hvar[j] >= c[1]*qvar[(i, j)] + c[0], name=f'hpl{n}({i},{j})')
        for n, c in enumerate(cutabove):
            LP.addConstr(hvar[i] - hvar[j] <= c[1]*qvar[(i, j)] + c[0], name=f'hpu{n}({i},{j})')
        LP._oa[(i,j)] = (cutbelow, cutabove)


    # ACTIVITY (PUMPS)
    for a, pump in inst.pumps.items():
        LP.addConstr(qvar[a] >= svar[a] * pump.qmin)
        LP.addConstr(qvar[a] <= svar[a] * pump.qmax)

    # CONVEXIFICATION OF HEAD-FLOW (PUMPS)
    for (i, j), pump in inst.pumps.items():
        cutbelow, cutabove = oa.pumpcuts(pump.qmin, pump.qmax, pump.hgain, f'({i}, {j})', epsilon) #, drawgraph=True)
        #cutbelow, cutabove = oa.pumpcutsgratien(pump.qmin, pump.qmax, pump.hgain, (i, j)) #, drawgraph=True)
        for n, c in enumerate(cutbelow):
            LP.addConstr(hvar[j] - hvar[i] >= c[1] * qvar[(i, j)]
                         + (c[0] - pump.offdhmin) * svar[(i, j)] + pump.offdhmin, name=f'hkl{n}({i},{j})')
        for n, c in enumerate(cutabove):
            LP.addConstr(hvar[j] - hvar[i] <= c[1] * qvar[(i, j)]
                         + (c[0] - pump.offdhmax) * svar[(i, j)] + pump.offdhmax, name=f'hku{n}({i},{j})')
        LP._oa[(i,j)] = (cutbelow, cutabove)

    LP.ModelSense = GRB.MINIMIZE
    LP.update()

    LP._svar = svar
    LP._qvar = qvar
    LP._hvar = hvar
    LP._obj = LP.getObjective()

    return LP

