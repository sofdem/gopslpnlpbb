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


def _attach_callback_data(model, instance):
    model._bestplan  = {}
    model._nbleaves  = {'feas': 0, 'heur': 0, 'unfeas': 0}
    model._incumbent = GRB.INFINITY
    model._heurtime = time.time()
    model._callbacktime = 0
    model._gaptol    = model.Params.OptimalityTol # 1e-2
    model._nperiods  = instance.nperiods()
    model._network   = HydraulicNetwork(instance)
    model._instance  = instance

def mycallback(m, where):

    # STOP if UB-LB < tol
    if where == GRB.Callback.MIP:
        if abs(m._incumbent - m.cbGet(GRB.Callback.MIP_OBJBND)) < m._gaptol * m._incumbent:
            print('Stop early - ', m._gaptol * 100, '% gap achieved')
            m.terminate()

    # at an integer solution
    elif where == GRB.Callback.MIPSOL:
        print(f'integer convex solution: {m.cbGet(GRB.Callback.MIPSOL_OBJ)}')
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

        if violation:
            m._nbleaves['unfeas'] += 1
            nogood_lastperiod = violation
            costreal = GRB.INFINITY
            # print('constraint violation at t =', violation, relaxcost)

            if m.cbGet(GRB.Callback.MIPSOL_OBJ) < m._incumbent and time.time() > m._heurtime + 30:
                m._heurtime = time.time()
                # !!! build the model once when bb starts then just update according to activity: see primalheuristic
                linearcost = ph.adjust_steplength(m._instance, m._network, activity, inactive, timelimit=60)
                print(f'primal heuristic: {linearcost}')
                if linearcost > 0:
                    nogood_lastperiod = m._nperiods
                    costreal = linearcost
                    m._nbleaves['heur'] += 1
        else:
            nogood_lastperiod = m._nperiods
            # !!! 1) diff from the original code which did count the reservoirs withdrawal cost
            # !!! 2) set activity[t][a] = 0 when qreal[a][t] == 0 ? remove svar[a,t] from nogood ?
            costreal = sum(activity[t][a] * m._svar[a, t].obj
                           + qreal[t][a] * m._qvar[a, t].obj for (a, t) in m._svar)
            m._nbleaves['feas'] += 1
            assert costreal >= m.cbGet(GRB.Callback.MIPSOL_OBJ), 'relaxed cost > real cost !! '
            print('solution is feasible:', costreal)

        if costreal < m._incumbent:
            print(f'UPDATE INCUMBENT: {m._incumbent} -> {costreal}')
            m._incumbent = costreal
            m._bestplan = activity
            m.cbLazy(m._obj <= (1-m._gaptol) * m._incumbent)

        m.cbLazy(_nogoodcut(m._svar, nogood_lastperiod, activity) >= 0)
        m._callbacktime += time.time() - m._starttime


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
    linexpr.addConstant(nbact-1)
#    print('nogood:', str(linexpr))
    return linexpr


def lpnlpbb(cvxmodel, instance, ub=1e6, drawsolution = True):
    """Apply the LP-NLP Branch-and-bound using the convex relaxation model cvxmodel."""
    _attach_callback_data(cvxmodel, instance)

    cvxmodel.params.timeLimit = 3600
    #cvxmodel.params.MIPGap = 0.01
    # cvxmodel.params.OutputFlag = 0
    # cvxmodel.params.Threads = 1
    cvxmodel.params.LazyConstraints = 1
    cvxmodel.params.FeasibilityTol = 1e-5

    cvxmodel.optimize(mycallback)

    if cvxmodel.status != GRB.OPTIMAL:
        print('Optimization was stopped with status %d' % cvxmodel.status)
        if not cvxmodel._bestplan:
            return cvxmodel._incumbent, cvxmodel.ObjBound, cvxmodel.Runtime, GRB.INFINITY

    print('Nb of leaf nodes =', cvxmodel._nbleaves)
    print('Total time in callback =', cvxmodel._callbacktime)

    gap = (cvxmodel._incumbent - cvxmodel.ObjBound) / cvxmodel._incumbent

    if drawsolution:
        inactive = {t: set(a for a, act in activity_t.items() if not act)
                    for t, activity_t in cvxmodel._bestplan.items()}
        qreal, hreal, vreal, violation = cvxmodel._network.extended_period_analysis(inactive, stopatviolation=False)

        if violation:
            print('last solution found by heuristic... times have been adjusted')
        costreal = sum(cvxmodel._bestplan[t][a] * cvxmodel._svar[a, t].obj
                       + qreal[t][a] * cvxmodel._qvar[a, t].obj for (a, t) in cvxmodel._svar)
        assert violation or abs(costreal - cvxmodel._incumbent) < 1e-6, 'recomputed cost {costreal} differs from incumbent {cvxmodel._incumbent}'

        #for (a, t) in cvxmodel._svar:
        #    pump = instance.pumps.get(a)
        #    if pump and cvxmodel._bestplan[t][a]:
        #        print(f'q_({a},{t}) = {qreal[t][a]} in [{pump.qmin}, {pump.qmax}]')

        graphic.pumps(instance, qreal)
        graphic.tanks(instance, qreal, vreal)

    return cvxmodel._incumbent, cvxmodel.ObjBound, cvxmodel.Runtime, gap


def solveconvex(cvxmodel, instance):
    """Solve the convex relaxation model cvxmodel."""
    cvxmodel.params.timeLimit = 200
    cvxmodel.params.MIPGap = 0.01
    # cvxmodel.params.OutputFlag = 0
    # cvxmodel.params.Threads = 1
    cvxmodel.params.SolutionLimit = 1

    cvxmodel.optimize()

    if cvxmodel.status != GRB.OPTIMAL:
        print('Optimization was stopped with status %d' % cvxmodel.status)

    if not cvxmodel.SolCount:
        return GRB.INFINITY, GRB.INFINITY, cvxmodel.Runtime, cvxmodel.MIPGap

    inactive, activity = _parse_activity(instance.horizon(), cvxmodel._svar)
    net = HydraulicNetwork(instance)
    qreal, hreal, vreal, violations = net.extended_period_analysis(inactive, stopatviolation=False)

    graphic.pumps(instance, qreal)
    graphic.tanks(instance, qreal, vreal)

    actreal = {t: {a: (0 if abs(q) < 1e-6 else 1) for a, q in qreal[t].items()} for t in qreal}
    costreal = sum(actreal[t][a] * cvxmodel._svar[a, t].obj
                   + qreal[t][a] * cvxmodel._qvar[a, t].obj for (a, t) in cvxmodel._svar)

    return costreal, cvxmodel.objVal, cvxmodel.Runtime, cvxmodel.MIPGap

