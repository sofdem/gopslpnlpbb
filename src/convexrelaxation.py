#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 11:07:46 2020

@author: Sophie Demassey, Gratien Bonvin
"""

import gurobipy as gp
from gurobipy import GRB
from instance import Instance
import datetime as dt


# !!! check round values
# noinspection PyArgumentList
def build_model(inst: Instance, oagap: float):
    """Build the convex relaxation gurobi model."""

    milp = gp.Model('Pumping_Scheduling')

    qvar = {}   # arc flow
    dhvar = {}  # arc hloss: dh_a = h_i - h_j = phi_a(q_a) si x_a = 1 // dh_a = \phi_a(0) si x_a = 0
    svar = {}   # arc status
    ivar = {}   # pump ignition status
    hvar = {}   # node head
    qexpr = {}  # node inflow
    qjvar = {}  # tank inflow

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
            hvar[j, t] = milp.addVar(lb=tank.hmin(t), ub=tank.hmax(t), name=f'ht({j},{t})')

        milp.update()

        for (i, j), a in inst.arcs.items():
            dh0 = a.hloss.value(0)
            dhmin = a.hloss.value(a.qmin(t))
            dhmax = a.hloss.value(a.qmax(t))
            vid = f'({a.id},{t})' # f'({i},{j},{t})'
            if a.control and a.isfixedoff(t):
                svar[(i, j), t] = milp.addVar(vtype=GRB.BINARY, lb=0, ub=0, name=f'x{vid}')
                qvar[(i, j), t] = milp.addVar(lb=0, ub=0, name=f'q{vid}')
                dhvar[(i, j), t] = milp.addVar(lb=dh0, ub=dh0, name=f'H{vid}')
            elif a.control and not a.isfixedon(t):
                svar[(i, j), t] = milp.addVar(vtype=GRB.BINARY, name=f'x{vid}')
                x = svar[(i, j), t]
                # q_a=0 if x_a=0 otherwise in [qmin,qmax]
                qvar[(i, j), t] = milp.addVar(lb=a.qmin(t), ub=a.qmax(t), name=f'q{vid}')
                milp.addConstr(qvar[(i, j), t] <= a.qmaxifon(t) * x, name=f'qxup{vid}')
                milp.addConstr(qvar[(i, j), t] >= a.qminifon(t) * x, name=f'qxlo{vid}')
                # dh_a = h_i - h_j  if x_a =1 otherwise phi_a(0)
                dhvar[(i, j), t] = milp.addVar(lb=min(dhmin, dh0), ub=max(dhmax, dh0), name=f'H{vid}')
                milp.addConstr(dhvar[(i, j), t] <= a.hloss.value(a.qmaxifon(t)) * x + dh0 * (1 - x), name=f'dhxup{vid}')
                milp.addConstr(dhvar[(i, j), t] >= a.hloss.value(a.qminifon(t)) * x + dh0 * (1 - x), name=f'dhxlo{vid}')
                milp.addConstr(dhvar[(i, j), t] <= hvar[i, t] - hvar[j, t] + (dh0 - a.dhminifoff(t)) * (1 - x),
                               name=f'dhhub{vid}')
                milp.addConstr(dhvar[(i, j), t] >= hvar[i, t] - hvar[j, t] + (dh0 - a.dhmaxifoff(t)) * (1 - x),
                               name=f'dhhlo{vid}')
            else:
                qvar[(i, j), t] = milp.addVar(lb=a.qmin(t), ub=a.qmax(t), name=f'q{vid}')
                dhvar[(i, j), t] = milp.addVar(lb=dhmin, ub=dhmax, name=f'H{vid}')
                svar[(i, j), t] = milp.addVar(vtype=GRB.BINARY, lb=1, name=f'x{vid}')
                milp.addConstr(dhvar[(i, j), t] == hvar[i, t] - hvar[j, t], name=f'dhh{vid}')

    for j, tank in inst.tanks.items():
        hvar[j, nperiods] = milp.addVar(lb=tank.head(tank.vinit), ub=tank.hmax(nperiods), name=f'ht({j},T)')

    milp.update()

    # FLOW CONSERVATION
    for t in horizon:
        for j in inst.nodes:
            qexpr[j, t] = gp.quicksum(qvar[a, t] for a in inst.inarcs(j)) \
                          - gp.quicksum(qvar[a, t] for a in inst.outarcs(j))

        for j, junc in inst.junctions.items():
            milp.addConstr(qexpr[j, t] == junc.demand(t), name=f'fc({j},{t})')

        for j, tank in inst.tanks.items():
            qjvar[j, t] = milp.addVar(lb=tank.qinmin(t), ub=tank.qinmax(t), name=f'qt({j},{t})')
            milp.addConstr(qjvar[j, t] == qexpr[j, t], name=f'fcq({j},{t})')

            milp.addConstr(hvar[j, t+1] - hvar[j, t] == inst.flowtoheight(tank) * qjvar[j, t],
                           name=f'fch({j},{t})')

    # MAX WITHDRAWAL AT RESERVOIRS
    for j, res in inst.reservoirs.items():
        if res.drawmax:
            milp.addConstr(res.drawmax >= inst.flowtovolume() * gp.quicksum(qexpr[j, t] for t in horizon),
                           name=f'w({j})')

    # CONVEXIFICATION OF HEAD-FLOW
    for (i, j), a in inst.arcs.items():
        dh0 = a.hloss.value(0)
        for t in horizon:
            vid = f'({i},{j},{t})'
            if not a.control or not a.isfixedoff(t):
                qmin = a.qminifon(t) if a.control else a.qmin(t)
                qmax = a.qmaxifon(t) if a.control else a.qmax(t)
                assert qmin <= qmax + inst.feastol, \
                    f"qmin>=qmax {vid} : {qmin} >= {qmax}"
                if qmin < qmax - inst.feastol:
                    x = svar[(i, j), t] if a.control else 1
                    c = a.hloss.isquasilinear(inst.feastol * 1e-2, qmin, qmax)
                    # TODO h = aq + (b-dh0)x + dh0 gerer les cas où les coeffs sont < feastol ?
                    if c:
                        if not a.control or abs(c[0] - dh0) < inst.feastol:
                            milp.addConstr(dhvar[(i, j), t] == c[1] * qvar[(i, j), t] + c[0],
                                           name=f'hp{vid}')
                        else:
                            milp.addConstr(dhvar[(i, j), t] == c[1] * qvar[(i, j), t] + (c[0]-dh0) * x + dh0,
                                           name=f'hp{vid}')
                    else:
                        cutbelow = a.hloss.underestimatepwlfunc(oagap, qmin, qmax)
                        for n, c in enumerate(cutbelow):
                            milp.addConstr(dhvar[(i, j), t] >= c[1] * qvar[(i, j), t] + (c[0]-dh0) * x + dh0,
                                           name=f'hpl{n}{vid}')
                        cutabove = a.hloss.overestimatepwlfunc(oagap, qmin, qmax)
                        for n, c in enumerate(cutabove):
                            milp.addConstr(dhvar[(i, j), t] <= c[1] * qvar[(i, j), t] + (c[0]-dh0) * x + dh0,
                                           name=f'hpu{n}{vid}')
                        # if vid == "(745,753,0)":
                            #a.hloss.drawoa(qmin, qmax, cutbelow, cutabove, vid, points=None)
                            # a.hloss.testoa(oagap, qmin, qmax, vid)
                            # a.hloss.drawoa(-100, 100, {}, {}, vid, points=None)

    strongdualityconstraints(inst, milp, hvar, qvar, svar, dhvar, qexpr, horizon, True)

    binarydependencies(inst, milp, ivar, svar, nperiods, horizon)

    obj = gp.quicksum(inst.eleccost(t) * (pump.power[0] * svar[k, t] + pump.power[1] * qvar[k, t])
                      for k, pump in inst.pumps.items() for t in horizon)

    milp.setObjective(obj, GRB.MINIMIZE)
    milp.update()

    milp._svar = svar
    milp._ivar = ivar
    milp._qvar = qvar
    milp._hvar = hvar
    milp._obj = obj
    milp._dhvar = dhvar

    return milp


def strongdualityconstraints(inst, milp, hvar, qvar, svar, dhvar, qexpr, horizon, withoutz):
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
            (l0, l1, u0, u1) = (h0.lb, h1.lb, h0.ub, h1.ub)
            if l0 == u0:
                hqvar[j, t] = (h1 - l0) * l0 / c
            else:
                hqvar[j, t] = milp.addVar(lb=-GRB.INFINITY, name=f'hqt({j},{t})')
                # inflow = {a: [inst.arcs[a].qmin(t), inst.arcs[a].qmax(t)] for a in inst.inarcs(j)}
                # outflow = {a: [inst.arcs[a].qmin(t), inst.arcs[a].qmax(t)] for a in inst.outarcs(j)}
                # print(f"inflow: {inflow}")
                # print(f"outflow: {outflow}")
                lq = max(c * tank.qinmin(t), l1 - u0)
                uq = min(c * tank.qinmax(t), u1 - l0)
                # refining with a direction indicator variable
                if withoutz:
                    milp.addConstr(c * hqvar[j, t] >= l0 * (h1 - h0) + lq * (h0 - l0), name=f'hqlo({j},{t})')
                    milp.addConstr(c * hqvar[j, t] >= u0 * (h1 - h0) + uq * (h0 - u0), name=f'hqup({j},{t})')
                else:
                    zvar = milp.addVar(vtype=GRB.BINARY, name=f'z({j},{t})')
                    hzvar = milp.addVar(lb=0, ub=u0, name=f'hz({j},{t})')
                    milp.addConstr(h1 - h0 <= (u1 - l0) * zvar, name=f'z0up({j},{t})')
                    milp.addConstr(h1 - h0 >= (l1 - u0) * (1 - zvar), name=f'z0lo({j},{t})')
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

        # OA for g_a(q, h) = Phi_a(q) + Psi_a(h) convex  and g_a(q, h) = q * h on {(q, phi_a(q)), q_a\in R}
        # let h'=phi_a(q') then g_a >= h' * q_a  + q' * dh_a  - h' * q' * x_a
        for (i, j), arc in inst.arcs.items():
            a = (i, j)
            gvar[a, t] = milp.addVar(lb=-GRB.INFINITY, name=f'g({i},{j},{t})')
            noacut = 10 if a in inst.pumps else 5
            for n in range(noacut):
                qstar = (arc.qmin(t) + arc.qmax(t)) * n / (noacut - 1)
                milp.addConstr(gvar[a, t] >= arc.hloss.value(qstar) *
                               (qvar[a, t] - qstar * svar[a, t]) + qstar * dhvar[a, t], name=f'goa{n}({i},{j},{t})')

        milp.addConstr(gp.quicksum(gvar[a, t] for a in inst.arcs) + sdexpr[t] <= milp.Params.MIPGapAbs, name=f'sd({t})')

def sdconstraints2(inst, milp, hvar, qvar, svar, dhvar, qexpr, horizon, withoutz):
    print("#################  STRONG DUALITY 2: 5 gvar(pipe) + 10 gvar (pump)")
    ##########################
    # convexity of q_a * h_a:
    # let define g[a,t] = q[a,t] * h[a,t] (given that h[a,t] = phi_a(q[a,t])
    # if s[a,t]=1: g[a,t] coincides with the convex function (q,h) -> Phi_a(q) + Psi_a(h) on {(q,phi_a(q)) | q\in R}
    # then g[a,t] >= q' * dh[a,t] + h' * q[a,t] - q' * h' * x[a,t] for any (q',h'=phi(q'))
    # and McCormick:
    # g[a,t] <= hmin * q[a,t] + qmax * dh[a,t] - hmin * qmax * s[a,t]
    # g[a,t] <= qmin * dh[a,t] + hmax * q[a,t] - qmin * hmax * s[a,t]
    ##########################
    # flow conservation sum_r h_r * q_r = - sum_a h_a * q_a - sum_s h_s * Q_s:
    # hq[r,t] =


    gvar = {}    # arc component:    x_a * (\Phi_a(q_a) - \Phi_a(\phi^{-1}(h_a)) + h_a\phi^{-1}(h_a))
    sdexpr = {}  # node component:   sum_n (q_n * h_n)
    hqvar = {}   # tank component:   q_r * h_r
    for t in horizon:

        # McCormick's envelope of hq_rt = h_rt * q_rt = h_rt * (h_{r,t+1}-h_rt)/c
        for j, tank in inst.tanks.items():
            c = inst.flowtoheight(tank)
            (h0, h1) = (hvar[j, t], hvar[j, t + 1])
            (l0, l1, u0, u1) = (h0.lb, h1.lb, h0.ub, h1.ub)
            if l0 == u0:
                hqvar[j, t] = (h1 - l0) * l0 / c
            else:
                hqvar[j, t] = milp.addVar(lb=-GRB.INFINITY, name=f'hqt({j},{t})')
                # inflow = {a: [inst.arcs[a].qmin(t), inst.arcs[a].qmax(t)] for a in inst.inarcs(j)}
                # outflow = {a: [inst.arcs[a].qmin(t), inst.arcs[a].qmax(t)] for a in inst.outarcs(j)}
                # print(f"inflow: {inflow}")
                # print(f"outflow: {outflow}")
                lq = max(c * tank.qinmin(t), l1 - u0)
                uq = min(c * tank.qinmax(t), u1 - l0)
                # refining with a direction indicator variable
                if withoutz:
                    milp.addConstr(c * hqvar[j, t] >= l0 * (h1 - h0) + lq * (h0 - l0), name=f'hqlo({j},{t})')
                    milp.addConstr(c * hqvar[j, t] >= u0 * (h1 - h0) + uq * (h0 - u0), name=f'hqup({j},{t})')
                else:
                    zvar = milp.addVar(vtype=GRB.BINARY, name=f'z({j},{t})')
                    hzvar = milp.addVar(lb=0, ub=u0, name=f'hz({j},{t})')
                    milp.addConstr(h1 - h0 <= (u1 - l0) * zvar, name=f'z0up({j},{t})')
                    milp.addConstr(h1 - h0 >= (l1 - u0) * (1 - zvar), name=f'z0lo({j},{t})')
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

        # OA for g_a(q, h) = Phi_a(q) + Psi_a(h) convex  and g_a(q, h) = q * h on {(q, phi_a(q)), q_a\in R}
        # let h'=phi_a(q') then g_a >= h' * q_a  + q' * dh_a  - h' * q' * x_a
        for (i, j), arc in inst.arcs.items():
            a = (i, j)
            gvar[a, t] = milp.addVar(lb=-GRB.INFINITY, name=f'g({i},{j},{t})')
            noacut = 10 if a in inst.pumps else 5
            for n in range(noacut):
                qstar = (arc.qmin(t) + arc.qmax(t)) * n / (noacut - 1)
                milp.addConstr(gvar[a, t] >= arc.hloss.value(qstar) *
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
                milp.addConstr(svar[s[0], t] + svar[s[1], t] == 1, name=f'dep2({t})')
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

def postvalues(model, varvals):
    for varname, val in varvals.items():
        var = model.getVarByName(varname)
        var.lb = val
        var.ub = val
