#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 11:07:46 2020

@author: Sophie Demassey, Gratien Bonvin
"""
import sys

import gurobipy as gp
from gurobipy import GRB
# import primalheuristic as ph
import time
from hydraulics import HydraulicNetwork
from networkanalysis import NetworkAnalysis

import graphic
import csv
from pathlib import Path

# every unfeasible integer node X is discarded with a nogood cut: |x-X|>=1
# feasible integer nodes are updated with bound cut: obj >= (realcost(X)-eps) * |1-X|
# which should invalidate the current MILP solution (when its cost is strictly lower than the real feasible solution)
# the corresponding MINLP feasible solution is provided as a heuristic solution to Gurobi
# Gurobi 10.0 now allows to set the solution at MIPSOL but it seems to accept it (useSolution)
# and update the cutoff value only in a 2nd step (MIPSOL cb called twice at the same node)

OUTDIR = Path("../output/")
IISFILE = Path(OUTDIR, f'modeliis.ilp').as_posix()
TESTNETANAL = True


def _attach_callback_data(model, instance):
    model._instance = instance
    model._nperiods = instance.nperiods()

    model._network = NetworkAnalysis(instance, model.Params.FeasibilityTol) if TESTNETANAL \
        else HydraulicNetwork(instance, feastol=model.Params.FeasibilityTol)

    model._solvars = [*model._svar.values(), *model._qvar.values()]
    model._clonemodel = None  # model.copy()

    model._incumbent = GRB.INFINITY
    model._callbacktime = 0
    model._solutions = []
    model._intnodes = {'unfeas': 0, 'feas': 0, 'adjust': 0}
    model._trace = []  # None
    model._rootlb = [0, 0]

def mycallback(m, where):

    # STOP if UB-LB < tol
    if where == GRB.Callback.MIP:
        if m._incumbent - m.cbGet(GRB.Callback.MIP_OBJBND) < m.Params.MIPGap * m._incumbent:
            print('Stop early - ', m.Params.MIPGap * 100, '% gap achieved')
            m.terminate()

    # store the lower bound at root node
    elif where == GRB.Callback.MIPNODE:
        currentnode = int(m.cbGet(GRB.Callback.MIPNODE_NODCNT))
        if currentnode == 0:
            currentlb = m.cbGet(GRB.Callback.MIPNODE_OBJBND)
            if m._rootlb[0] == 0:
                m._rootlb[0] = currentlb
            m._rootlb[1] = currentlb
            trace_progress(m._trace, m.cbGet(GRB.Callback.RUNTIME), 0, currentlb, None, None, None)

    elif where == GRB.Callback.MIPSOL:

        costmip = m.cbGet(GRB.Callback.MIPSOL_OBJ)
        currentlb = m.cbGet(GRB.Callback.MIPSOL_OBJBND)
        currentnode = int(m.cbGet(GRB.Callback.MIPSOL_NODCNT))
        currentphase = int(m.cbGet(GRB.Callback.MIPSOL_PHASE))
        fstring = f"MIPSOL #{currentnode} P{currentphase}: {costmip} [{currentlb:.2f}, " \
                  f"{'inf' if m._incumbent == GRB.INFINITY else round(m._incumbent, 2)}]"

        # at same node AND  same integer solution than the last best known MINLP solution: pass callback
        if m._solutions and currentnode == m._solutions[-1]['node'] and sameplan(m, m._solutions[-1]['plan']):
            print(f"pass CB: {fstring} same plan as stored {m._incumbent}")
            assert abs(costmip - m._incumbent) < m.Params.MIPGap
            return
            #if abs(costmip - m._incumbent) > m.Params.MIPGap:
            #    print(f"{mystr} but different costs mipsol={costmip} feas={m._incumbent}", file=sys.stderr)
            #    assert costmip < m._incumbent - m.Params.MIPGap
            #else:
            #    print(f"{mystr} same cost (inc={m._incumbent})")
            #    return

        # check MINLP feasibility and compute the actual plan cost
        m._starttime = time.time()
        costreal = GRB.INFINITY
        inactive, activity = getplan(m)
        qreal, vreal, violperiod = m._network.extended_period_analysis(inactive, stopatviolation=True)

        # plan X is not feasible for MINLP: forbid the node with a combinatorial nogood cut: |x-X|>=1
        if violperiod:
            m._intnodes['unfeas'] += 1
            print(fstring + f" t={violperiod}")
            addnogoodcut(m, _linearnorm(m._svar, activity, violperiod), currentnode)

        # plan X is feasible for MINLP: enforce the bound with: obj >= (realcost(X)-tol) * (1-|x-X|)
        else:
            m._intnodes['feas'] += 1
            costreal = solutioncost(m, activity, qreal)
            assert costreal >= costmip - m.Params.MIPGapAbs
            print(fstring + f" feasible: {costreal}")

            # cost is improved: update the incumbent and inject solution (X,Q) to the solver
            if costreal < m._incumbent:
                m._incumbent = costreal
                m._solutions.append({'plan': activity, 'cost': costreal, 'flows': qreal, 'volumes': vreal,
                                     'cpu': m.cbGet(GRB.Callback.RUNTIME), 'node': currentnode, 'adjusted': (qreal is None)})
                gap = (m._incumbent - currentlb) / m._incumbent
                print(f'UPDATE INCUMBENT gap={gap * 100:.4f}%')
                setsolution(m, costreal, activity, qreal)

            addboundcut(m, _linearnorm(m._svar, activity), costreal, currentnode)

        m._callbacktime += time.time() - m._starttime
        trace_progress(m._trace, m.cbGet(GRB.Callback.RUNTIME), currentnode,
                       currentlb, m._incumbent, costmip, costreal)


def setsolution(m, costreal, actreal, qreal):
    m.cbSetSolution(m._solvars, getminlpsol(m, actreal, qreal))
    grbcost = m.cbUseSolution()
    print(f"try set solution: {costreal} -> {grbcost}")
    # print([f"{vars[k].varname}={vals[k]}; " for k in range(len(m._svar), len(vars))])
    if grbcost < GRB.INFINITY:
        print("solution accepted !!")
        assert abs(costreal - grbcost) < m.Params.MIPGap
        return True
    # cloneandchecksolution(m, vals)
    return False


def trace_progress(trace, cpu, node, lb, ub, costmip, costreal):
    if trace:
        trace.append([cpu, node, lb, ub if ub < GRB.INFINITY else None,
                      costmip, costreal if costreal < GRB.INFINITY else None])


def getminlpsol(m, activity, qreal):
    solx = [activity[t][a] for (a, t) in m._svar]
    solq = [qreal[t][a] for (a, t) in m._qvar]
    # solh = [hreal[t][j] for (j, t) in m._hvar]
    return [*solx, *solq]


def getplan(m, cb=True):
    inactive = {t: set() for t in range(m._nperiods)}
    activity = {t: {} for t in range(m._nperiods)}
    for (a, t), svar in m._svar.items():
        if getval(m, svar, cb) < 0.5:
            inactive[t].add(a)
            activity[t][a] = 0
        else:
            activity[t][a] = 1
    return inactive, activity


def getval(m, svar, cb):
    return m.cbGetSolution(svar) if cb else svar.x


def sameplan(m, activity):
    for (a, t), svar in m._svar.items():
        sval = getval(m, svar, cb=True)
        if activity[t][a]:
            if sval < 0.5:
                print(f"not same plan ! x[{t}][{a}]: {sval} != {activity[t][a]}")
                return False
        elif sval > 0.5:
            print(f"not same plan ! x[{t}][{a}]: {sval} != {activity[t][a]}")
            return False
    return True


def addnogoodcut(m, linnorm, n):
    m.cbLazy(linnorm >= 1)
    if m._clonemodel:
        c = clonelinexpr(m, linnorm)
        m._clonemodel.addConstr(c >= 1, name=f'nogood{n}')


def addboundcut(m, linnorm, costrealsol, n):
    cost = costrealsol - m.Params.MIPGapAbs
#    print(f"post bound cut obj >= {cost} (1-|x-X|)")  # nogood: {str(linexpr)}")
    m.cbLazy(m._obj >= cost * (1 - linnorm))
    if m._clonemodel:
        c = clonelinexpr(m, linnorm)
        cobj = m._clonemodel.getObjective()
        m._clonemodel.addConstr(cobj >= cost * (1 - c), name=f'bound{n}')


def addcutoff(m, cutoff, n):
    m.cbLazy(m._obj <= cutoff)
    if m._clonemodel:
        cobj = m._clonemodel.getObjective()
        m._clonemodel.addConstr(cobj <= cutoff, name=f'cutoff{n}')


def clonelinexpr(m, linexpr):
    assert m._clonemodel
    sz = linexpr.size()
    coeffs = [linexpr.getCoeff(i) for i in range(sz)]
    vars = [m._clonemodel.getVarByName(linexpr.getVar(i).varname) for i in range(sz)]
    c = gp.LinExpr(linexpr.getConstant())
    c.addTerms(coeffs, vars)
    assert c.size() == sz and c.getConstant() == linexpr.getConstant()
    return c

def _linearnorm(svars, activity, last_period=-1):
    linexpr = gp.LinExpr()
    nbact = 0
    if last_period < 0:
        last_period = len(activity)
    for t in range(last_period):
        for a, active in activity[t].items():
            # _filterwithsymmetry(activity[t], a):
            if active:
                linexpr.addTerms(-1.0, svars[a, t])
                nbact += 1
            else:
                linexpr.addTerms(1.0, svars[a, t])
    linexpr.addConstant(nbact)
    return linexpr


def solutioncost(cvxmodel, status, flows):
    return sum(status[t][a] * cvxmodel._svar[a, t].obj
               + flows[t][a] * cvxmodel._qvar[a, t].obj for (a, t) in cvxmodel._svar)


def lpnlpbb(cvxmodel, instance, modes, drawsolution=True):
    """Apply the LP-NLP Branch-and-bound using the convex relaxation model cvxmodel."""

    _attach_callback_data(cvxmodel, instance)
    cvxmodel.params.LazyConstraints = 1

    cvxmodel.optimize(mycallback)

    if cvxmodel.status != GRB.OPTIMAL:
        print('Optimization was stopped with status %d' % cvxmodel.status)
    if cvxmodel.status == GRB.INFEASIBLE and drawsolution:
        print(f'no solution found write IIS file {IISFILE}')
        cvxmodel.computeIIS()
        cvxmodel.write(IISFILE)
    if cvxmodel.solcount == 0:
        return 0, {}

    print("check gurobi best solution")
    solx = [v.x for v in cvxmodel._svar.values()]
    print(solx)
    # cvxmodel._clonemodel.write("check.lp")
    solq = [v.x for v in cvxmodel._qvar.values()]
    cloneandchecksolution(cvxmodel, [*solx, *solq])

    cost = 0
    plan = {}
    if cvxmodel._solutions:
        bestsol = cvxmodel._solutions[-1]
        plan = bestsol['plan']
        flow = bestsol['flows']
        volume = bestsol['volumes']
        cost = cvxmodel._incumbent
        if not flow:
            print('best solution found by the time-adjustment heuristic')
            inactive = {t: set(a for a, act in activity_t.items() if not act) for t, activity_t in plan.items()}
            flow, volume, nbviolations = cvxmodel._network.extended_period_analysis(inactive, stopatviolation=False)
            assert nbviolations, 'solution was time-adjusted and should be slightly unfeasible'
            cost = solutioncost(cvxmodel, plan, flow)
            print(f'real plan cost = {cost} / time adjustment cost = {cvxmodel._incumbent}')
        if drawsolution:
            graphic.pumps(instance, flow)
            graphic.tanks(instance, flow, volume)
    return cost, plan


def solveconvex(cvxmodel, instance, drawsolution=True):
    """Solve the convex relaxation model cvxmodel."""
    # cvxmodel.params.SolutionLimit = 1
    cvxmodel.optimize()

    if cvxmodel.status != GRB.OPTIMAL:
        print('Optimization was stopped with status %d' % cvxmodel.status)
    if cvxmodel.status == GRB.INFEASIBLE and drawsolution:
        print(f"f write IIS in {IISFILE}")
        cvxmodel.computeIIS()
        cvxmodel.write(IISFILE)

    costreal = 0
    plan = {}
    if cvxmodel.SolCount:
        cvxmodel._nperiods = instance.horizon()
        inactive, activity = getplan(cvxmodel, cb=False)
        net = HydraulicNetwork(instance, cvxmodel.Params.FeasibilityTol)
        qreal, vreal, nbviolations = net.extended_period_analysis(inactive, stopatviolation=False)
        print(f"real plan with {nbviolations} violations")
        plan = {t: {a: (0 if abs(q) < 1e-6 else 1) for a, q in qreal[t].items()} for t in qreal}
        costreal = solutioncost(cvxmodel, plan, qreal)
        if drawsolution:
            graphic.pumps(instance, qreal)
            graphic.tanks(instance, qreal, vreal)
        #for a, s in cvxmodel._svar.items():
        #    print(f'{a}: {round(s.x)} {round(cvxmodel._qvar[a].x, 4)}')
    return costreal, plan


def recordandwritesolution(m, activity, qreal, filename):
    sol = getminlpsol(m, activity, qreal)
    f = open(filename, 'a')
    write = csv.writer(f)
    write.writerow(sol)
    return sol


def cloneandchecksolution(m, vals):
    assert len(vals) == len(m._solvars)
    model = m._clonemodel if m._clonemodel else m.copy()
    for i, var in enumerate(m._solvars):
        clonevar = model.getVarByName(var.varname)
        clonevar.lb = vals[i]
        clonevar.ub = vals[i]
    print("test real solution / cvx model (without internal cuts nor processing)")
    model.optimize()
    if model.status != GRB.OPTIMAL:
        print('Optimization was stopped with status %d' % model.status)
        model.computeIIS()
        model.write(IISFILE)
    model.terminate()
