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
def build_model(inst: Instance, pumpvals={}):
    """Build the convex relaxation gurobi model."""

    LP = gp.Model('Pumping_Schedulding')

    qvar = {}  # arc flow
    svar = {}  # pump/valve activity status
    ivar = {}  # pump ignition status
    hvar = {}  # node head

    for t in inst.horizon():
        for (i, j), pipe in inst.pipes.items():
            qvar[(i, j), t] = LP.addVar(lb=pipe.qmin, ub=pipe.qmax, name=f'qp({i},{j},{t})')

        for (i, j), valve in inst.valves.items():
            qvar[(i, j), t] = LP.addVar(lb=valve.qmin, ub=valve.qmax, name=f'qv({i},{j},{t})')
            svar[(i, j), t] = LP.addVar(vtype=GRB.BINARY, name=f'xv({i},{j},{t})')

        for (i, j), pump in inst.pumps.items():
            ivar[(i, j), t] = LP.addVar(vtype=GRB.BINARY, name=f'ik({i},{j},{t})')
            addQcost = 3.6 * inst.tsinhours() * inst.reservoirs[i].drawcost if i in inst.reservoirs else 0
            qvar[(i, j), t] = LP.addVar(lb=0, ub=pump.qmax, obj=addQcost + inst.eleccost(t) * pump.power[1], name=f'qk({i},{j},{t})')
            svar[(i, j), t] = LP.addVar(obj=inst.eleccost(t) * pump.power[0], vtype=GRB.BINARY, name=f'xk({i},{j},{t})')

        for j in inst.junctions:
            hvar[j, t] = LP.addVar(name=f'hj({j},{t})')

        for j, res in inst.reservoirs.items():
            hvar[j, t] = LP.addVar(lb=res.head(t), ub=res.head(t), name=f'hr({j},{t})')

        for j, tank in inst.tanks.items():
            hvar[j, t] = LP.addVar(lb=tank.head(tank.vmin), ub=tank.head(tank.vmax), name=f'ht({j},{t})')

    for j, tank in inst.tanks.items():
        hvar[j, inst.nperiods()] = LP.addVar(lb=tank.head(tank.vinit), ub=tank.head(tank.vmax), name=f'ht({j},T)')
        hvar[j,0].lb = tank.head(tank.vinit)
        hvar[j,0].ub = tank.head(tank.vinit)
    LP.update()

    # FLOW CONSERVATION (JUNCTIONS)
    for j, junc in inst.junctions.items():
        for t in inst.horizon():
            # !!! original code: round(demand,2)
            LP.addConstr(gp.quicksum(qvar[a, t] for a in inst.inarcs(j))
                         - gp.quicksum(qvar[a, t] for a in inst.outarcs(j)) == junc.demand(t), name=f'fc({j},{t})')

    # FLOW CONSERVATION (TANKS)
    for j, tank in inst.tanks.items():
        for t in inst.horizon():
            LP.addConstr(hvar[j, t+1] - hvar[j, t] == 3.6 * inst.tsinhours() / tank.surface *
                         (gp.quicksum(qvar[a, t] for a in inst.inarcs(j))
                          - gp.quicksum(qvar[a, t] for a in inst.outarcs(j))), name=f'fc({j},{t})')

    # MAX WITHDRAWAL AT RESERVOIRS
    for j, res in inst.reservoirs.items():
        if res.drawmax:
            LP.addConstr(3.6 * inst.tsinhours()
                         * gp.quicksum(qvar(a, t) for a in inst.outarc[j] for t in inst.horizon())
                         <= res.drawmax, name=f'w({j})')

    # VALVES
    for (i, j), valve in inst.valves.items():
        if valve.type == 'GV' or valve.type == 'PRV':
            for t in inst.horizon():
                x = svar[(i, j), t] if (valve.type == 'GV') else (1 - svar[(i, j), t])
                LP.addConstr(qvar[(i, j), t] <= valve.qmax * svar[(i, j), t], name=f'vu({i},{j},{t})')
                LP.addConstr(qvar[(i, j), t] >= valve.qmin * x, name=f'vl({i},{j},{t})')
                LP.addConstr(hvar[i, t] - hvar[j, t] >= valve.hlossmin * (1 - svar[(i, j), t]), name=f'hvl({i},{j},{t})')
                LP.addConstr(hvar[i, t] - hvar[j, t] <= valve.hlossmax * (1 - x), name=f'hvu({i},{j},{t})')

    # CONVEXIFICATION OF HEAD-FLOW (PIPES)
    for (i, j), pipe in inst.pipes.items():
        coeff = [pipe.hloss[2], pipe.hloss[1]]
        cutbelow, cutabove = oa.pipecuts(pipe.qmin, pipe.qmax, coeff, (i, j))
        # print(f'{pipe}: {len(cutbelow)} cutbelow, {len(cutabove)} cutabove')
        for t in inst.horizon():
            for n, c in enumerate(cutbelow):
                LP.addConstr(hvar[i, t] - hvar[j, t] >= c[0]*qvar[(i, j), t] + c[1], name=f'hpl{n}({i},{j},{t})')
            for n, c in enumerate(cutabove):
                LP.addConstr(hvar[i, t] - hvar[j, t] <= c[0]*qvar[(i, j), t] + c[1], name=f'hpu{n}({i},{j},{t})')

    # ACTIVITY (PUMPS)
    for a, pump in inst.pumps.items():
        for t in inst.horizon():
            # !!! original code: integer round ???
            LP.addConstr(qvar[a, t] >= svar[a, t] * pump.qmin)
            LP.addConstr(qvar[a, t] <= svar[a, t] * pump.qmax)

    # CONVEXIFICATION OF HEAD-FLOW (PUMPS)
    for (i, j), pump in inst.pumps.items():
        cutbelow, cutabove = oa.pumpcuts(pump.qmin, pump.qmax, pump.hgain, (i, j)) #, drawgraph=True)
        #cutbelow, cutabove = oa.pumpcutsgratien(pump.qmin, pump.qmax, pump.hgain, (i, j)) #, drawgraph=True)
        for t in inst.horizon():
            for n, c in enumerate(cutbelow):
                LP.addConstr(hvar[j, t] - hvar[i, t] >= c[0] * qvar[(i, j), t]
                             + (c[1] - pump.offdhmin) * svar[(i, j), t] + pump.offdhmin, name=f'hkl{n}({i},{j},{t})')
            for n, c in enumerate(cutabove):
                # !!! original code: gapabove = 0 if pump.offdhmax == 1000 else pump.offdhmax - c[1] ???
                LP.addConstr(hvar[j, t] - hvar[i, t] <= c[0] * qvar[(i, j), t]
                             + (c[1] - pump.offdhmax) * svar[(i, j), t] + pump.offdhmax, name=f'hku{n}({i},{j},{t})')

    ### PUMP SWITCHING
    sympumps = inst.symmetries
    uniquepumps = inst.pumps_without_sym()
    print('symmetries:', uniquepumps)

    def getv(vdict, pump, t):
        return gp.quicksum(vdict[a,t] for a in sympumps) if pump == 'sym' else vdict[pump, t]

    # !!! check the max ignition constraint for the symmetric group
    # !!! make ivar[a,0] = svar[a,0]
    for a in uniquepumps:
        rhs = 6 * len(sympumps) if a=='sym' else 6 - svar[a,0]
        LP.addConstr(gp.quicksum(getv(ivar,a,t) for t in range(1,inst.nperiods())) <= rhs)
        for t in range(1,inst.nperiods()):
            LP.addConstr(getv(ivar,a,t) >= getv(svar,a,t) - getv(svar,a,t-1))
            if inst.tsduration == dt.timedelta(minutes=30) and t < inst.nperiods()-1:
                # minimum 1 hour activity
                LP.addConstr(getv(svar,a,t+1) + getv(svar,a,t-1) >= getv(svar,a,t))

    ### PUMP DEPENDENCIES
    if sympumps:
        for t in inst.horizon():
            for i, pump in enumerate(sympumps[:-1]):
                LP.addConstr(ivar[pump,t] >= ivar[sympumps[i+1],t])
                LP.addConstr(svar[pump,t] >= svar[sympumps[i+1],t])

    if inst.dependencies:
        for t in inst.horizon():
            for s in inst.dependencies['p1 => p0']:
                LP.addConstr(svar[s[0],t] >= svar[s[1],t])
            for s in inst.dependencies['p0 or p1']:
                LP.addConstr(svar[s[0],t] + svar[s[1],t] >= 1)
            for s in inst.dependencies['p0 <=> p1 xor p2']:
                LP.addConstr(svar[s[0],t] == svar[s[1],t] + svar[s[2],t])
            for s in inst.dependencies['p1 => not p0']:
                LP.addConstr(svar[s[0],t] + svar[s[1],t] <= 1)

    if pumpvals:
        for pump in inst.pumps:
            for t in inst.horizon():
                LP.addConstr(svar[pump, t] == pumpvals[pump, t])

    LP.ModelSense = GRB.MINIMIZE
    LP.update()

    LP._svar = svar
    LP._ivar = ivar
    LP._qvar = qvar
    LP._hvar = hvar
    LP._obj = LP.getObjective()

    LP.write('modelnew.lp')
    return LP
