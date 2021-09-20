#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 11:07:46 2020

@author: Sophie Demassey, Gratien Bonvin
"""

import gurobipy as gp
from gurobipy import GRB
import primalheuristic as ph
import time
from hydraulics import HydraulicNetwork
import graphic


def _attach_callback_data(model, instance, adjust_mode):
    model._incumbent = GRB.INFINITY
    model._solutions = []
    model._callbacktime = 0
    model._gaptol = model.Params.MIPGap  # 1e-2
    model._nperiods = instance.nperiods()
    model._network = HydraulicNetwork(instance, model.Params.FeasibilityTol)
    model._instance = instance
    model._adjust = adjust_mode
    model._adjusttime = time.time()
    model._intnodes = {'unfeas': 0, 'feas': 0, 'adjust': 0}
    model._adjust_solutions = []


def mycallback(m, where):
    # STOP if UB-LB < tol
    if where == GRB.Callback.MIP:
        if m._incumbent - m.cbGet(GRB.Callback.MIP_OBJBND) < m._gaptol * m._incumbent:
            print('Stop early - ', m._gaptol * 100, '% gap achieved')
            m.terminate()

    # elif where == GRB.Callback.MIPNODE:
    #    if not m._rootlb:
    #        m._rootlb = m.cbGet(GRB.Callback.MIPNODE_OBJBND)

    # at an integer solution
    elif where == GRB.Callback.MIPSOL:
        fstring = f"integer leaf: {m.cbGet(GRB.Callback.MIPSOL_OBJ):.2f} " \
                  f"[{m.cbGet(GRB.Callback.MIPSOL_OBJBND):.2f}, " \
                  f"{'inf' if m._incumbent == GRB.INFINITY else round(m._incumbent, 2)}]"
        m._starttime = time.time()

        inactive = {t: set() for t in range(m._nperiods)}
        activity = {t: {} for t in range(m._nperiods)}
        for (a, t), svar in m._svar.items():
            if m.cbGetSolution(svar) < 0.5:
                inactive[t].add(a)
                activity[t][a] = 0
            else:
                activity[t][a] = 1

        qreal, hreal, vreal, violation = m._network.extended_period_analysis(inactive)

        nogood_lastperiod = m._nperiods

        if violation:
            v = violation[0]
            m._intnodes['unfeas'] += 1
            nogood_lastperiod = v[0]
            costreal = GRB.INFINITY
            print(fstring + f' violation t={v[0]} tk={v[1]}: {v[2]:.2f}')
            m.cbLazy(_nogoodcut(m._svar, nogood_lastperiod, activity) >= 0)

            if m._adjust and m.cbGet(GRB.Callback.MIPSOL_OBJ) < m._incumbent and time.time() > m._adjusttime + 5:
                m._intnodes['adjust'] += 1
                m._adjusttime = time.time()
                # !!! build the model once when bb starts then just update according to activity: see primalheuristic
                adjustcost = ph.adjust_steplength(m._instance, m._network, activity, inactive, timelimit=60)
                print(f'primal heuristic: {adjustcost}')
                adjustbest = m._adjust_solutions[-1]["cost"] if m._adjust_solutions else m._incumbent
                if 0 < adjustcost < min(adjustbest, m._incumbent):
                    m._adjust_solutions.append(
                        {'plan': activity, 'cost': adjustcost, 'cpu': m.cbGet(GRB.Callback.RUNTIME)})
                    # !!! diff from original code: generate smallest/deepest (stop to the violated period) nogood cut
                    if m._adjust == "CUT":
                        costreal = adjustcost
                        qreal = None
                        vreal = None

        else:
            # !!! set activity[t][a] = 0 when qreal[a][t] == 0 ? remove svar[a,t] from nogood ?
            costreal = solutioncost(m, activity, qreal)
            m._intnodes['feas'] += 1
            assert costreal >= m.cbGet(GRB.Callback.MIPSOL_OBJ), 'relaxed cost > real cost !! '
            print(fstring + f" feasible: {costreal:2f}")

        if costreal < m._incumbent:
            m._incumbent = costreal
            m._solutions.append({'plan': activity, 'cost': costreal, 'flows': qreal, 'volumes': vreal,
                                 'cpu': m.cbGet(GRB.Callback.RUNTIME), 'adjusted': (qreal is None)})
            gap = (m._incumbent - m.cbGet(GRB.Callback.MIPSOL_OBJBND)) / m._incumbent
            print(
                f'UPDATE INCUMBENT gap={gap * 100:.4f}%')
            # ": {m._incumbent} -> {costreal} (lb={m.cbGet(GRB.Callback.MIPSOL_OBJBND):2f})')
            if gap < m._gaptol:
                print('Stop early - ', m._gaptol * 100, '% gap achieved')
                m.terminate()
            else:
                m.cbLazy(m._obj <= (1 - m._gaptol) * m._incumbent)

        m._callbacktime += time.time() - m._starttime

        # discarding feasible nodes with nogoods cuts make the final ObjBound a non-valid lower bound
        # and prevent Gurobi to keep the integer solutions
        # !!! TODO (not supported in Gurobi 9.1): cbcSetSolution at feas nodes and set costreal as the LB of the node
        # m.cbLazy(_nogoodcut(m._svar, nogood_lastperiod, activity) >= 0)


def _parse_activity(horizon, svars):
    inactive = {t: set() for t in horizon}
    activity = {t: {} for t in horizon}
    for (a, t), svar in svars.items():
        if svar.x < 0.5:
            inactive[t].add(a)
            activity[t][a] = 0
        else:
            activity[t][a] = 1
    return inactive, activity


def _nogoodcut(svars, last_period, activity):
    linexpr = gp.LinExpr()
    nbact = 0
    for t in range(last_period):
        for a, active in activity[t].items():
            if active:
                linexpr.addTerms(-1.0, svars[a, t])
                nbact += 1
            else:
                linexpr.addTerms(1.0, svars[a, t])
    linexpr.addConstant(nbact - 1)
    #    print('nogood:', str(linexpr))
    return linexpr


def solutioncost(cvxmodel, status, flows):
    return sum(status[t][a] * cvxmodel._svar[a, t].obj
               + flows[t][a] * cvxmodel._qvar[a, t].obj for (a, t) in cvxmodel._svar)


def lpnlpbb(cvxmodel, instance, drawsolution=True, adjust_mode="SOLVE"):
    """Apply the LP-NLP Branch-and-bound using the convex relaxation model cvxmodel."""

    _attach_callback_data(cvxmodel, instance, adjust_mode)
    cvxmodel.params.LazyConstraints = 1

    cvxmodel.optimize(mycallback)

    if cvxmodel.status != GRB.OPTIMAL:
        print('Optimization was stopped with status %d' % cvxmodel.status)

    cost = 0
    if cvxmodel._solutions:
        bestsol = cvxmodel._solutions[-1]
        plan = bestsol['plan']
        flow = bestsol['flows']
        volume = bestsol['volumes']
        cost = cvxmodel._incumbent
        if not flow:
            print('best solution found by the time-adjustment heuristic')
            inactive = {t: set(a for a, act in activity_t.items() if not act) for t, activity_t in plan.items()}
            flow, hreal, volume, violation = cvxmodel._network.extended_period_analysis(inactive, stopatviolation=False)
            assert violation, 'solution was time-adjusted and should be slightly unfeasible'
            for v in violation:
                print(f'violation t={v[0]} tk={v[1]}: {v[2]:.2f}')
            cost = solutioncost(cvxmodel, plan, flow)
            print(f'real plan cost = {cost} / time adjustment cost = {cvxmodel._incumbent}')
        if drawsolution:
            graphic.pumps(instance, flow)
            graphic.tanks(instance, flow, volume)
    else:
        print('no solution found')

    return cost


def solveconvex(cvxmodel, instance, drawsolution=True):
    """Solve the convex relaxation model cvxmodel."""
    # cvxmodel.params.SolutionLimit = 1

    cvxmodel.optimize()

    if cvxmodel.status != GRB.OPTIMAL:
        print('Optimization was stopped with status %d' % cvxmodel.status)

    costreal = 0
    if cvxmodel.SolCount:
        inactive, activity = _parse_activity(instance.horizon(), cvxmodel._svar)
        net = HydraulicNetwork(instance, 1e-4)
        qreal, hreal, vreal, violations = net.extended_period_analysis(inactive, stopatviolation=False)
        for v in violations:
            print(f'violation t={v[0]} tk={v[1]}: {v[2]:.2f}')
        actreal = {t: {a: (0 if abs(q) < 1e-6 else 1) for a, q in qreal[t].items()} for t in qreal}
        costreal = solutioncost(cvxmodel, actreal, qreal)

        if drawsolution:
            graphic.pumps(instance, qreal)
            graphic.tanks(instance, qreal, vreal)

        for a, s in cvxmodel._svar.items():
            print(f'{a}: {round(s.x)} {round(cvxmodel._qvar[a].x, 4)}')
    else:
        iisfile = "cvxmodel.ilp"
        print("f write IIS in {iisfile}")
        cvxmodel.computeIIS()
        cvxmodel.write(iisfile)

    return costreal
