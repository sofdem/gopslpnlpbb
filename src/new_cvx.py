# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 18:53:11 2022

@author: amirhossein.tavakoli
"""

import gurobipy as gp
from gurobipy import GRB
import outerapproximation as oa
from instance import Instance
import datetime as dt
import math


# !!! check round values
# noinspection PyArgumentList
def build_model(inst: Instance, Z, C, D, P0, P1, oagap: float, arcvals=None):
    """Build the convex relaxation gurobi model."""

    milp = gp.Model('Pumping_Scheduling')
    milp.params.TimeLimit= 80

    qvar = {}  # arc flow
    dhvar = {}  # arc hloss
    svar = {}  # arc status
    ivar = {}  # pump ignition status
    hvar = {}  # node head
    qexpr = {}  # node inflow
    
#    svar_new = {} #lifted variable

    
    
    
    
###    s3_var= {}


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
            lbt = tank.head(tank.vinit) if t == 0 else D[j, t][0]
            ubt = tank.head(tank.vinit) if t == 0 else D[j, t][1]
            hvar[j, t] = milp.addVar(lb=lbt, ub=ubt, name=f'ht({j},{t})')
            
        for j, tank in inst.tanks.items():
            lbt = tank.head(tank.vinit) if t == 0 else D[j, t][0]
            ubt = tank.head(tank.vinit) if t == 0 else D[j, t][1]


        milp.update()


        for (i, j), a in inst.arcs.items():

            if a.control:
                qvar[(i, j), t] = milp.addVar(lb=-GRB.INFINITY, name=f'q({i},{j},{t})')
                dhvar[(i, j), t] = milp.addVar(lb=-GRB.INFINITY, name=f'H({i},{j},{t})')
                if ((i, j), t) in P0:
                    svar[(i, j), t] = milp.addVar(lb=0, ub=0.001, name=f'x({i},{j},{t})')
                elif ((i, j), t) in P1:
                    svar[(i, j), t] = milp.addVar(lb=0.999, ub=1, name=f'x({i},{j},{t})')
                else:
                    svar[(i, j), t] = milp.addVar(vtype=GRB.BINARY, name=f'x({i},{j},{t})')
                # q_a=0 if x_a=0 otherwise in [qmin,qmax]
                milp.addConstr(qvar[(i, j), t] <= Z[(i, j), t][1] * svar[(i, j), t], name=f'qxup({i},{j},{t})')
                milp.addConstr(qvar[(i, j), t] >= Z[(i, j), t][0] * svar[(i, j), t], name=f'qxlo({i},{j},{t})')
                
###                milp.addConstr(qvar[('R1', 'J2'), t] <= Z[('R1', 'J2'), t][1] * (svar[('R1', 'J2'), t]-svar[('R2', 'J2'), t])+Z[('R2', 'J2'), t][1] * (svar[('R2', 'J2'), t]), name=f'qxupn({i},{j},{t})')
###                milp.addConstr(qvar[('R1', 'J2'), t] <= Z[('R1', 'J2'), t][1] * (svar[('R1', 'J2'), t]-svar[('R3', 'J2'), t])+Z[('R3', 'J2'), t][1] * (svar[('R3', 'J2'), t]), name=f'qxupn({i},{j},{t})')
###                milp.addConstr(qvar[('R2', 'J2'), t] <= Z[('R2', 'J2'), t][1] * (svar[('R2', 'J2'), t]-svar[('R3', 'J2'), t])+Z[('R3', 'J2'), t][1] * (svar[('R3', 'J2'), t]), name=f'qxupn({i},{j},{t})')
#                milp.addConstr(qvar[(i, j), t] >= 0, name=f'qxlo({i},{j},{t})')
                # dh_a = (h_i - h_j) * x_a
                dhmin = max(a.hlossval(a.qmin), hvar[i, t].lb - hvar[j, t].ub)
                dhmax = min(a.hlossval(Z[(i, j), t][1]), hvar[i, t].ub - hvar[j, t].lb)
                milp.addConstr(dhvar[(i, j), t] <= dhmax * svar[(i, j), t], name=f'dhxup({i},{j},{t})')
                milp.addConstr(dhvar[(i, j), t] >= dhmin * svar[(i, j), t], name=f'dhxlo({i},{j},{t})')
#                milp.addConstr(dhvar[(i, j), t] <= hvar[i, t] - hvar[j, t] - a.dhmin * (1-svar[(i, j), t]), name=f'dhhub({i},{j},{t})')
                milp.addConstr(dhvar[(i, j), t] <= hvar[i, t] - hvar[j, t] - C[(i, j), t][0] * (1-svar[(i, j), t]), name=f'dhhub({i},{j},{t})')
#                milp.addConstr(dhvar[(i, j), t] >= hvar[i, t] - hvar[j, t] - a.dhmax * (1-svar[(i, j), t]), name=f'dhhlo({i},{j},{t})')
                milp.addConstr(dhvar[(i, j), t] >= hvar[i, t] - hvar[j, t] - C[(i, j), t][1] * (1-svar[(i, j), t]), name=f'dhhlo({i},{j},{t})')

            else:
                qvar[(i, j), t] = milp.addVar(lb=Z[(i, j), t][0], ub=Z[(i, j), t][1], name=f'q({i},{j},{t})')
                dhvar[(i, j), t] = milp.addVar(lb=a.hlossval(a.qmin), ub=a.hlossval(Z[(i, j), t][1]), name=f'H({i},{j},{t})')
                svar[(i, j), t] = milp.addVar(vtype=GRB.BINARY, lb=1, name=f'x({i},{j},{t})')
                milp.addConstr(dhvar[(i, j), t] == hvar[i, t] - hvar[j, t], name=f'dhh({i},{j},{t})')
                


    for j, tank in inst.tanks.items():
        hvar[j, nperiods] = milp.addVar(lb=tank.head(tank.vinit), ub=tank.head(tank.vmax), name=f'ht({j},T)')
        

            

    milp.update()

    # FLOW CONSERVATION
    for t in horizon:
        for j in inst.nodes:
            qexpr[j, t] = gp.quicksum(qvar[a, t] for a in inst.inarcs(j)) \
                          - gp.quicksum(qvar[a, t] for a in inst.outarcs(j))

        for j, junc in inst.junctions.items():
            milp.addConstr(gp.quicksum(qvar[a, t] for a in inst.inarcs(j))
                           - gp.quicksum(qvar[a, t] for a in inst.outarcs(j)) == junc.demand(t), name=f'fc({j},{t})')

        for j, tank in inst.tanks.items():
            milp.addConstr(hvar[j, t+1] - hvar[j, t] == inst.flowtoheight(tank) * qexpr[j, t], name=f'fc({j},{t})')


                           

    # MAX WITHDRAWAL AT RESERVOIRS
    for j, res in inst.reservoirs.items():
        if res.drawmax:
            milp.addConstr(res.drawmax >=
                           inst.flowtovolume() * gp.quicksum(qexpr[j, t] for t in horizon), name=f'w({j})')
            
###    for t in horizon:
###        for j, tank in inst.tanks.items():
###            milp.addConstr(qexpr[j, t] <= (Q3[j, t][1]- Q2[j, t][1])*svar[('R3','J2'), t]+ (Q2[j, t][1]- Q1[j, t][1])*svar[('R2','J2'), t] + (Q1[j, t][1]- Q0[j, t][1])*svar[('R1','J2'), t]+ Q0[j, t][1]+0.00001)
###            milp.addConstr(qexpr[j, t] >= (Q3[j, t][0]- Q2[j, t][0])*svar[('R3','J2'), t]+ (Q2[j, t][0]- Q1[j, t][0])*svar[('R2','J2'), t] + (Q1[j, t][0]- Q0[j, t][0])*svar[('R1','J2'), t]+ Q0[j, t][0]-0.00001)


    # CONVEXIFICATION OF HEAD-FLOW
    for (i, j), arc in inst.arcs.items():
        for t in horizon:
            cutbelow, cutabove = oa.hlossoa(Z[(i, j), t][0], Z[(i, j), t][1]+0.00001, arc.hloss, (i, j), oagap, drawgraph=False)
            print(f'{arc}: {len(cutbelow)} cutbelow, {len(cutabove)} cutabove')
            x = svar[(i, j), t] if arc.control else 1
            for n, c in enumerate(cutbelow):
                milp.addConstr(dhvar[(i, j), t] >= c[1] * qvar[(i, j), t] + c[0] * x, name=f'hpl{n}({i},{j},{t})')
            for n, c in enumerate(cutabove):
                milp.addConstr(dhvar[(i, j), t] <= c[1] * qvar[(i, j), t] + c[0] * x, name=f'hpu{n}({i},{j},{t})')


    strongdualityconstraints(inst, Z, C, D, milp, hvar, qvar, svar, dhvar, qexpr, horizon, True)

    binarydependencies(inst, milp, ivar, svar, nperiods, horizon)

    if arcvals:
        postbinarysolution(inst, arcvals, horizon, svar)

    obj = gp.quicksum(inst.eleccost(t) * (pump.power[0] * svar[k, t] + pump.power[1] * qvar[k, t])
                      for k, pump in inst.pumps.items() for t in horizon)

    milp.setObjective(obj, GRB.MINIMIZE)
    milp.update()

    milp._svar = svar
    milp._ivar = ivar
    milp._qvar = qvar
    milp._hvar = hvar
    milp._Z= Z
    milp._obj = obj

    return milp


def strongdualityconstraints(inst, Z, C, D, milp, hvar, qvar, svar, dhvar, qexpr, horizon, withoutz):
    print("#################  STRONG DUALITY: 5 gvar(pipe) + 10 gvar (pump)")
    # strong duality constraint: sum_a gvar[a,t] + sdexpr[t] <= 0
    gvar = {}    # arc component:    x_a * (\Phi_a(q_a) - \Phi_a(\phi^{-1}(h_a)) + h_a\phi^{-1}(h_a))
    sdexpr = {}  # node component:   sum_n (q_n * h_n)
    hqvar = {}   # tank component:   q_r * h_r
    for t in horizon:

        # McCormick's envelope of hq_rt = h_rt * q_rt = h_rt * (h_{r,t+1}-h_rt)/c
        for j, tank in inst.tanks.items():
            c = inst.flowtoheight(tank)
            (h0, h1) = (hvar[j, t], hvar[j, t + 1])
            if t==0:
            
                (l0, l1, u0, u1) = (h0.lb, D[j, t+1][0], h0.ub, D[j, t+1][1])
            elif t== inst.nperiods()-1:
                (l0, l1, u0, u1) = (D[j, t][0], h1.lb, D[j, t][1], h1.ub)
            else:
                (l0, l1, u0, u1) = (D[j, t][0], D[j, t+1][0], D[j, t][1], D[j, t+1][1])
            if l0 == u0:
                hqvar[j, t] = (h1 - l0) * l0 / c
            else:
                hqvar[j, t] = milp.addVar(lb=-GRB.INFINITY, name=f'hqt({j},{t})')
                inflow = {a: [inst.arcs[a].qmin, inst.arcs[a].qmax] for a in inst.inarcs(j)}
                outflow = {a: [inst.arcs[a].qmin, inst.arcs[a].qmax] for a in inst.outarcs(j)}
                print(f"inflow: {inflow}")
                print(f"outflow: {outflow}")
                lq = max(c * inst.inflowmin(j), c * Z[j, t][0], l1 - u0)
                uq = min(c * inst.inflowmax(j), c * Z[j, t][1], u1 - l0)
                # refining with a direction indicator variable
                if withoutz:
                    milp.addConstr(c * hqvar[j, t] >= l0 * (h1 - h0) + lq * (h0 - l0), name=f'hqlo({j},{t})')
                    milp.addConstr(c * hqvar[j, t] >= u0 * (h1 - h0) + uq * (h0 - u0), name=f'hqup({j},{t})')
                else:
                    zvar = milp.addVar(vtype=GRB.BINARY, name=f'z({j},{t})')
                    hzvar = milp.addVar(lb=0, ub=u0, name=f'hz({j},{t})')
                    milp.addConstr(hzvar <= u0 * zvar, name=f'hz1up({j},{t})')
                    milp.addConstr(hzvar >= l0 * zvar, name=f'hz1lo({j},{t})')
                    milp.addConstr(hzvar <= h0 - l0 * (1 - zvar), name=f'hz0up({j},{t})')
                    milp.addConstr(hzvar >= h0 - u0 * (1 - zvar), name=f'hz0lo({j},{t})')
                    milp.addConstr(c * hqvar[j, t] >= l0 * (h1 - h0) + lq * (hzvar - l0 * zvar), name=f'hqlo({j},{t})')
                    milp.addConstr(c * hqvar[j, t] >= u0 * (h1 - h0) + uq * (h0 - hzvar - u0 * (1 - zvar)), name=f'hqup({j},{t})')

        # sdexpr[t] = milp.addVar(lb=-GRB.INFINITY, name=f'sd({t})')
        sdexpr[t] = gp.quicksum(hqvar[j, t] for j in inst.tanks) \
            + gp.quicksum(junc.demand(t) * hvar[j, t] for j, junc in inst.junctions.items()) \
            + gp.quicksum(res.head(t) * qexpr[j, t] for j, res in inst.reservoirs.items())

        # OA for the convex function g_a >= Phi_a(q_a) - Phi_a(phi^{-1}(h_a)) + h_a * phi^{-1}(h_a)
        for (i,j), arc in inst.arcs.items():
            a = (i, j)
            gvar[a, t] = milp.addVar(lb=-GRB.INFINITY, name=f'g({i},{j},{t})')
            noacut = 10 if a in inst.pumps else 5
            for n in range(noacut):
                qstar = (arc.qmin + Z[(i,j), t][1]) * n / (noacut - 1)
                milp.addConstr(gvar[a, t] >= arc.hlossval(qstar) *
                               (qvar[a, t] - qstar * svar[a, t]) + qstar * dhvar[a, t], name=f'goa{n}({i},{j},{t})')

        milp.addConstr(gp.quicksum(gvar[a, t] for a in inst.arcs) + sdexpr[t] <= milp.Params.MIPGapAbs, name=f'sd({t})')


def binarydependencies(inst, milp, ivar, svar, nperiods, horizon):
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
        milp.addConstr(gp.quicksum(getv(ivar, a, t) for t in range(1, nperiods)) <= rhs, name='ig')
        for t in range(1, nperiods):
            milp.addConstr(getv(ivar, a, t) >= getv(svar, a, t) - getv(svar, a, t - 1), name=f'ig({t})')
            if inst.tsduration == dt.timedelta(minutes=30) and t < inst.nperiods() - 1:
                # minimum 1 hour activity
                milp.addConstr(getv(svar, a, t + 1) + getv(svar, a, t - 1) >= getv(svar, a, t), name=f'ig1h({t})')

    # PUMP DEPENDENCIES
    if sympumps:
        for t in horizon:
            for i, pump in enumerate(sympumps[:-1]):
                milp.addConstr(ivar[pump, t] >= ivar[sympumps[i + 1], t], name=f'symi({t})')
                milp.addConstr(svar[pump, t] >= svar[sympumps[i + 1], t], name=f'symx({t})')

    if inst.dependencies:
        for t in horizon:
            for s in inst.dependencies['p1 => p0']:
                milp.addConstr(svar[s[0], t] >= svar[s[1], t], name=f'dep1({t})')
            for s in inst.dependencies['p0 xor p1']:
                milp.addConstr(svar[s[0], t] + svar[s[1], t] >= 1, name=f'dep2({t})')
            for s in inst.dependencies['p0 = p1 xor p2']:
                milp.addConstr(svar[s[0], t] == svar[s[1], t] + svar[s[2], t], name=f'dep3({t})')
            # for s in inst.dependencies['p1 => not p0']:
            #    milp.addConstr(svar[s[0], t] + svar[s[1], t] <= 1, name=f'dep4({t})')


def postbinarysolution(inst, arcvals, horizon, svar):
    assert arcvals
    for a in inst.varcs:
        for t in horizon:
            v = arcvals.get((a, t))
            if v == 1:
                svar[a, t].lb = 1
            elif v == 0:
                svar[a, t].ub = 0


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
