#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 11:07:46 2020

@author: Sophie Demassey, Gratien Bonvin
"""

import gurobipy as gp
from gurobipy import GRB
from datetime import date
import primalheuristic as ph
import time
from hydraulics import HydraulicNetwork
import graphic

class Stat:
    """Statistics for solving one instance."""

    def __init__(self, cvxmodel, instance, costreal):
        self.instance = instance
        self.date = date.today()
        self.status = cvxmodel.status
        self.cpu = cvxmodel.Runtime
        self.realub = costreal
        self.lb = cvxmodel.objBound
        self.nodes = cvxmodel.NodeCount
        self.iter = cvxmodel.IterCount
        if not instance:
            self.instance = cvxmodel._instance
            self.cpu_cb = cvxmodel._callbacktime
            self.ub = cvxmodel._incumbent if cvxmodel._solutions else float('inf')
            self.intnodes = cvxmodel._intnodes
            self.gap = (self.ub - self.lb) / self.ub
        else:
            self.cpu_cb = 0
            self.ub = cvxmodel.objVal
            self.intnodes = 0
            self.gap = cvxmodel.MIPGap

    @staticmethod
    def tocsv_title():
        return 'ub, real_ub, lb, gap, cpu, cpu_cb, nodes, int_nodes'

    def tocsv_basic(self):
        return f"{self.ub:.2f}, {self.realub:.2f}, {self.lb:.2f}, {self.gap:.4f}, {self.cpu:.1f}, {self.cpu_cb:.1f}, {self.nodes}, {self.intnodes}"
    def tostr_basic(self):
        return f"cost: {self.ub:.2f}, gap: {self.gap:.4f}%, cpu: {self.cpu:.1f}s, cpu_cb: {self.cpu_cb:.1f}s, nodes: {self.nodes}, {self.intnodes}"
    def tostr_full(self):
        return f"cost: {self.ub:.2f}, realcost: {self.realub:.2f}, lb: {self.lb:.2f}, cpu: {self.cpu:.1f}s, cpu_cb: {self.cpu_cb:.2f}s, nodes: {self.nodes:.0f}, {self.intnodes}"




def _attach_callback_data(model, instance, adjust_mode):
    model._incumbent  = GRB.INFINITY
    model._solutions  = []
    model._callbacktime = 0
    model._gaptol    = 1e-4 #model.Params.OptimalityTol # 1e-2
    model._nperiods  = instance.nperiods()
    model._network   = HydraulicNetwork(instance)
    model._instance  = instance
    model._adjust    = adjust_mode
    model._adjusttime = time.time()
    model._intnodes  = {'unfeas': 0, 'feas': 0, 'adjust': 0}
    model._adjust_solutions  = []

def mycallback(m, where):

    # STOP if UB-LB < tol
    if where == GRB.Callback.MIP:
        if m._incumbent - m.cbGet(GRB.Callback.MIP_OBJBND) < m._gaptol * m._incumbent:
            print('Stop early - ', m._gaptol * 100, '% gap achieved')
            m.terminate()

    #elif where == GRB.Callback.MIPNODE:
    #    if not m._rootlb:
    #        m._rootlb = m.cbGet(GRB.Callback.MIPNODE_OBJBND)

    # at an integer solution
    elif where == GRB.Callback.MIPSOL:
        print(f'integer convex solution: {m.cbGet(GRB.Callback.MIPSOL_OBJ):.2f} (lb={m.cbGet(GRB.Callback.MIPSOL_OBJBND):.2f})')
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
            m._intnodes['unfeas'] += 1
            nogood_lastperiod = violation
            costreal = GRB.INFINITY
            # print('constraint violation at t =', violation, relaxcost)

            if m._adjust and m.cbGet(GRB.Callback.MIPSOL_OBJ) < m._incumbent and time.time() > m._adjusttime + 5:
                m._intnodes['adjust'] += 1
                m._adjusttime = time.time()
                # !!! build the model once when bb starts then just update according to activity: see primalheuristic
                adjustcost = ph.adjust_steplength(m._instance, m._network, activity, inactive, timelimit=60)
                print(f'primal heuristic: {adjustcost}')
                adjustbest = m._adjust_solutions[-1]["cost"] if m._adjust_solutions else m._incumbent
                if adjustcost > 0 and adjustcost < min(adjustbest, m._incumbent):
                    m._adjust_solutions.append({'plan': activity, 'cost': adjustcost})
                    if m._adjust == "CUT":
                        costreal = adjustcost
                        qreal = None
                        vreal = None
                    # !!! diff from previously: generate the smallest/deepest (stop to the violated period) nogood cut

        else:
            # !!! 1) diff from the original code which did count the reservoirs withdrawal cost
            # !!! 2) set activity[t][a] = 0 when qreal[a][t] == 0 ? remove svar[a,t] from nogood ?
            costreal = solutioncost(m, activity, qreal)
            m._intnodes['feas'] += 1
            assert costreal >= m.cbGet(GRB.Callback.MIPSOL_OBJ), 'relaxed cost > real cost !! '
            print('solution is feasible:', costreal)

        if costreal < m._incumbent:
            print(f'UPDATE INCUMBENT: {m._incumbent} -> {costreal} (lb={m.cbGet(GRB.Callback.MIPSOL_OBJBND):2f})')
            m._incumbent = costreal
            m._solutions.append({'plan': activity, 'cost': costreal, 'flows': qreal, 'volumes': vreal})

            if m._incumbent - m.cbGet(GRB.Callback.MIPSOL_OBJBND) < m._gaptol * m._incumbent:
                print('Stop early - ', m._gaptol * 100, '% gap achieved')
                m.terminate()
            else:
                m.cbLazy(m._obj <= (1-m._gaptol) * m._incumbent)

        m.cbLazy(_nogoodcut(m._svar, nogood_lastperiod, activity) >= 0)
        m._callbacktime += time.time() - m._starttime

        #!!! TODO: remove lazycut when feasible but add the nonoconvex solution to the solver:
        # requires to feed values for all the variables of the convex model

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

def solutioncost(cvxmodel, status, flows):
    return sum(status[t][a] * cvxmodel._svar[a, t].obj
             + flows[t][a] * cvxmodel._qvar[a, t].obj for (a, t) in cvxmodel._svar)


def lpnlpbb(cvxmodel, instance, ub=1e6, drawsolution = True, adjust_mode = "SOLVE"):
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
            cost = solutioncost(cvxmodel, plan, flow)
            print(f'real plan cost = {cost} / time adjustment cost = {cvxmodel._incumbent}')
        if drawsolution:
            graphic.pumps(instance, flow)
            graphic.tanks(instance, flow, volume)
    else:
            print('no solution found')

    return Stat(cvxmodel, None, cost)


def solveconvex(cvxmodel, instance, drawsolution = True):
    """Solve the convex relaxation model cvxmodel."""
    #cvxmodel.params.SolutionLimit = 1

    cvxmodel.optimize()

    if cvxmodel.status != GRB.OPTIMAL:
         print('Optimization was stopped with status %d' % cvxmodel.status)

    costreal = 0
    if cvxmodel.SolCount:
        inactive, activity = _parse_activity(instance.horizon(), cvxmodel._svar)
        net = HydraulicNetwork(instance)
        qreal, hreal, vreal, violations = net.extended_period_analysis(inactive, stopatviolation=False)
        actreal = {t: {a: (0 if abs(q) < 1e-6 else 1) for a, q in qreal[t].items()} for t in qreal}
        costreal = solutioncost(cvxmodel, actreal, qreal)

        if drawsolution:
            graphic.pumps(instance, qreal)
            graphic.tanks(instance, qreal, vreal)

    return Stat(cvxmodel, instance, costreal)

