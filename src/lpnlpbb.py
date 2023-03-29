#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 11:07:46 2020

@author: Sophie Demassey, Gratien Bonvin
"""

import gurobipy as gp
from gurobipy import GRB
# import primalheuristic as ph
import time
from hydraulics import HydraulicNetwork
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


def _attach_callback_data(model, instance, modes):
    model._instance = instance
    model._nperiods = instance.nperiods()
    model._network = HydraulicNetwork(instance, model.Params.FeasibilityTol)

    model._incumbent = GRB.INFINITY
    model._rootlb = [0,0]
    model._callbacktime = 0
    model._solutions = []
    model._intnodes = {'unfeas': 0, 'feas': 0, 'adjust': 0}
    # model._recordsol = (modes["solve"] == "RECORD")

    vs = [*model._svar.values(), *model._qvar.values()]
    model._lastsol = {'node': -1, 'cost': GRB.INFINITY, 'vars': vs}
    model._clonemodel = model.copy() # None
    model._trace = [] # None

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
            trace_progress(m, m.cbGet(GRB.Callback.RUNTIME), 0, currentlb, None, None, None)

    elif where == GRB.Callback.MIPSOL:
        costmipsol = m.cbGet(GRB.Callback.MIPSOL_OBJ)
        bestmipsol = m.cbGet(GRB.Callback.MIPSOL_OBJBST)
        currentlb = m.cbGet(GRB.Callback.MIPSOL_OBJBND)
        currentnode = int(m.cbGet(GRB.Callback.MIPSOL_NODCNT))
        currentphase = int(m.cbGet(GRB.Callback.MIPSOL_PHASE))
        fstring = f"MIPSOL #{currentnode} P{currentphase}: {costmipsol} [{currentlb:.2f}, " \
                  f"{'inf' if m._incumbent == GRB.INFINITY else round(m._incumbent, 2)}]"

        # the solver comes with the MINLP solution previously posted (but rejected): pass cb and accept it now
        if currentnode == m._lastsol['node']:
            print(f"still in node #{currentnode} with rejected solution {m._lastsol['obj']}: obj={costmipsol}")
            if costmipsol - m.Params.MIPGap <= m._lastsol['obj'] <= costmipsol + m.Params.MIPGap:
                print(f"{fstring}\n It is the rejected solution: pass callback and just accept it !!")
                inactive, activity = getplan(m)
                assert activity == m._lastsol['plan'], "same cost, but not the same plan"
                return

        m._starttime = time.time()
        nogood_lastperiod = m._nperiods
        costrealsol = GRB.INFINITY

        inactive, activity = getplan(m)
        qreal, hreal, vreal, violperiod = m._network.extended_period_analysis(inactive, stopatviolation=True)

        # plan X is not feasible for MINLP: cut with a combinatorial nogood |x-X|>=1
        if violperiod:
            m._intnodes['unfeas'] += 1
            nogood_lastperiod = violperiod
            print(fstring + f" t={violperiod}")
            addnogoodcut(m, _linearnorm(m._svar, nogood_lastperiod, activity), currentnode)

        # plan X is feasible for MINLP: cut it with obj >= (realcost(X)-eps) * |1-X|
        else:
            m._intnodes['feas'] += 1
            costrealsol = solutioncost(m, activity, qreal)
            assert costrealsol >= costmipsol - m.Params.MIPGapAbs
            print(fstring + f" feasible: {costrealsol}")

            # if better, update the incumbent and inject solution to the solver
            if costrealsol < m._incumbent:
                m._incumbent = min(m._incumbent, costrealsol)
                m._solutions.append({'plan': activity, 'cost': costrealsol, 'flows': qreal, 'volumes': vreal,
                                     'cpu': m.cbGet(GRB.Callback.RUNTIME), 'adjusted': (qreal is None)})
                gap = (m._incumbent - currentlb) / m._incumbent
                print(f'UPDATE INCUMBENT gap={gap * 100:.4f}%')

                m.cbSetSolution(m._lastsol['vars'], getrealsol(m, activity, qreal))
                m._lastsol['node'] = currentnode
                m._lastsol['obj'] = costrealsol
                m._lastsol['plan'] = activity
                m._lastsol['cost'] = m.cbUseSolution()
                print(f"MIPSOL #{currentnode} oldbest = {bestmipsol} "
                      f"try set solution: {costrealsol} -> {m._lastsol['cost']}")
                # gurobi 10.0: solution is always? rejected but the solver recalls it in a second step
                if m._lastsol['cost'] >= GRB.INFINITY:
                    print(f"!!!!!!!!!! SOLUTION NOT SET {m._incumbent} !!!!!!!!!!!")
                else:
                    print(f"solution accepted: update the cutoff value manually {m._incumbent}")
                    m._lastsol['node'] = -1
                    m._lastsol['obj'] = -1
                    addcutoff(m, (1 - m.Params.MIPGap) * m._incumbent, currentnode)

            linexpr = _linearnorm(m._svar, nogood_lastperiod, activity)
            # print(f"bound cut {costrealsol}")  # nogood: {str(linexpr)}")
            addboundcut(m, linexpr, costrealsol, currentnode)
            # addnogoodcut(m, linexpr, currentnode)

        m._callbacktime += time.time() - m._starttime
        bestval = m._incumbent if m._incumbent < GRB.INFINITY else None
        realval = costrealsol if costrealsol < GRB.INFINITY else None
        trace_progress(m, m.cbGet(GRB.Callback.RUNTIME), currentnode, currentlb, bestval, costmipsol, realval)

def mycallbackrecord(m, where):

    # STOP if UB-LB < tol
    if where == GRB.Callback.MIP:
        if m._incumbent - m.cbGet(GRB.Callback.MIP_OBJBND) < m.Params.MIPGap * m._incumbent:
            print('Stop early - ', m.Params.MIPGap * 100, '% gap achieved')
            m.terminate()

    elif m._recordsol and where == GRB.Callback.MIPNODE:
        if m._lastsol['cost'] < GRB.INFINITY:
            lastcost = m._lastsol['cost']
            m._lastsol['cost'] = GRB.INFINITY
            assert m._lastsol['node'] == m.cbGet(GRB.Callback.MIPNODE_NODCNT)
            assert len(m._lastsol['vars']) == len(m._lastsol['vals'])
            oldbest = m.cbGet(GRB.Callback.MIPNODE_OBJBST)
            m.cbSetSolution(m._lastsol['vars'], m._lastsol['vals'])
            objval = m.cbUseSolution()
            print(f"MIPNODE #{int(m.cbGet(GRB.Callback.MIPNODE_NODCNT))} "
                  f"oldbest = {oldbest} "
                  f"set solution #{m.cbGet(GRB.Callback.MIPNODE_SOLCNT)}: {lastcost} -> {objval}")
            if objval >= GRB.INFINITY and abs(oldbest-lastcost) > m.Params.MIPGapAbs:
                cloneandchecksolution(m, m._lastsol['vals'])
                print("if MILP feasible then the solution must violate a lazy cut")
        if int(m.cbGet(GRB.Callback.MIPNODE_NODCNT)) == 0:
            currentlb = m.cbGet(GRB.Callback.MIPNODE_OBJBND)
            if m._rootlb[0] == 0:
                m._rootlb[0] = currentlb
            m._rootlb[1] = currentlb
            trace_progress(m, m.cbGet(GRB.Callback.RUNTIME), 0, currentlb, None, None, None)


    # at an integer solution
    elif where == GRB.Callback.MIPSOL:

        # note that this integer solution may not be LP-optimal (e.g. if computed by a heuristic)
        costmipsol = m.cbGet(GRB.Callback.MIPSOL_OBJ)
        bestmipsol = m.cbGet(GRB.Callback.MIPSOL_OBJBST)
        currentlb = m.cbGet(GRB.Callback.MIPSOL_OBJBND)
        currentnode = int(m.cbGet(GRB.Callback.MIPSOL_NODCNT))
        fstring = f"MIPSOL #{currentnode}: {costmipsol} [{currentlb:.2f}, " \
                  f"{'inf' if m._incumbent == GRB.INFINITY else round(m._incumbent, 2)}]" #, best = {bestmipsol}"

        # prevent to evaluate again the same plan... (warning: no cut so the MILP solution is considered as feasible)
        # 1/ set by useSolution at the next MIPNODE (then MIPSOL is called again)
        # 2/ recomputed after adding the bound cut (sometimes (why not always ??) MIPSOL is called again) (=> the 10*)
        # 3/ due to multithreading delays (the incumbent update seems to be communicated but not the cutoff value)
        # if costmipsol > m._incumbent - 10*m.Params.MIPGapAbs:
        #    print(fstring + f" obj >= incumbent {m._incumbent}... SKIP (MILP solution accepted) ")
        #    if m._lastsol['cost'] < GRB.INFINITY:
        #        print("last solution has not yet been posted to the solver... let gurobi record this one instead")
        #        m._lastsol['cost'] = GRB.INFINITY
        #        assert costmipsol
        #        sol = [0 if m.cbGetSolution(svar) < 0.5 else 1 for svar in m._svar.values()]
        #        assert sol == m._lastsol['vals'][:len(sol)], "plan has changed between 2 consecutive MIPSOL ???"
        #    return
        if m._lastsol['node'] == currentnode:
            print(fstring + " same node: no way to cut this MILP solution, let gurobi save it and skip NLP computation")
            if m._lastsol['cost'] < GRB.INFINITY:
                assert m._lastsol['cost'] == m._incumbent
                print(f"last feasible solution {m._incumbent} has not yet been posted")
                # if abs(costmipsol - m._incumbent) < 10 * m.Params.MIPGapAbs:
                # print("obj = last ... SKIP (probably ?! same plan / almost same solution)")
                # m._lastsol = GRB.INFINITY
                # else:
                # print("obj > last ... SKIP (not same plan but should be fathomed later)")
                # sol = [0 if m.cbGetSolution(svar) < 0.5 else 1 for svar in m._svar.values()]
                # print(f"same plan ? {sol == m._lastsol['vals'][:len(sol)]}")
            return

        m._starttime = time.time()
        nogood_lastperiod = m._nperiods
        costrealsol = GRB.INFINITY

        inactive, activity = getplan(m)
        qreal, hreal, vreal, violperiod = m._network.extended_period_analysis(inactive, stopatviolation=True)

        if violperiod:
            m._intnodes['unfeas'] += 1
            nogood_lastperiod = violperiod
            print(fstring + f" t={violperiod}")
            addnogoodcut(m, _linearnorm(m._svar, nogood_lastperiod, activity), currentnode)

        else:
            m._intnodes['feas'] += 1
            costrealsol = solutioncost(m, activity, qreal)
            print(fstring + f" feasible: {costrealsol}")

            if costrealsol < costmipsol - m.Params.MIPGapAbs:
                print(f"mip solution cost {costmipsol} (heuristic, non-lp optimal?) > real cost {costrealsol}")
                # solvecvxmodelwithsolution(m, getrealsol(m, activity, qreal))

            linexpr = _linearnorm(m._svar, nogood_lastperiod, activity)
            if m._recordsol:
                print(f"bound cut {costrealsol}")
                print('nogood:', str(linexpr))
                addboundcut(m, linexpr, costrealsol, currentnode)
            else:
                addnogoodcut(m, linexpr, currentnode)

        if costrealsol < m._incumbent:
            m._incumbent = costrealsol
            m._solutions.append({'plan': activity, 'cost': costrealsol, 'flows': qreal, 'volumes': vreal,
                                 'cpu': m.cbGet(GRB.Callback.RUNTIME), 'adjusted': (qreal is None)})
            gap = (m._incumbent - currentlb) / m._incumbent
            print(f'UPDATE INCUMBENT gap={gap * 100:.4f}%')

            if m._recordsol:
                m._lastsol['cost'] = costrealsol
                m._lastsol['vals'] = getrealsol(m, activity, qreal)
                m._lastsol['node'] = m.cbGet(GRB.Callback.MIPSOL_NODCNT)
            else:
                addcutoff(m, (1 - m.Params.MIPGap) * m._incumbent, currentnode)

        m._callbacktime += time.time() - m._starttime
        bestval = m._incumbent if m._incumbent < GRB.INFINITY else None
        realval = costrealsol if costrealsol < GRB.INFINITY else None
        trace_progress(m, m.cbGet(GRB.Callback.RUNTIME), currentnode, currentlb, bestval, costmipsol, realval)


def trace_progress(m, cpu, node, lb, ub, mipval, minlpval):
    if m._trace != None:
        m._trace.append([cpu, node, lb, ub, mipval, minlpval])


def getrealsol(m, activity, qreal):
    solx = [activity[t][a] for (a, t) in m._svar]
    solq = [qreal[t][a] for (a, t) in m._qvar]
    # for (j,t) in m._hvar:
    #     sol.append(hreal[t][j])
    return [*solx, *solq]

def getplan(m):
    inactive = {t: set() for t in range(m._nperiods)}
    activity = {t: {} for t in range(m._nperiods)}
    for (a, t), svar in m._svar.items():
        if m.cbGetSolution(svar) < 0.5:
            inactive[t].add(a)
            activity[t][a] = 0
        else:
            activity[t][a] = 1
    return inactive, activity

def addnogoodcut(m, linnorm, n):
    m.cbLazy(linnorm >= 1)
    if m._clonemodel:
        c = clonelinexpr(m, linnorm)
        m._clonemodel.addConstr(c >= 1, name=f'nogood{n}')

def addboundcut(m, linnorm, costrealsol, n):
    cost = costrealsol - m.Params.MIPGapAbs
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


def _linearnorm(svars, last_period, activity):
    linexpr = gp.LinExpr()
    nbact = 0

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

    _attach_callback_data(cvxmodel, instance, modes)
    cvxmodel.params.LazyConstraints = 1

    cvxmodel.optimize(mycallback)

    if cvxmodel.status != GRB.OPTIMAL:
        print('Optimization was stopped with status %d' % cvxmodel.status)

    if cvxmodel.status == GRB.INFEASIBLE:
        print(f'no solution found write IIS file {IISFILE}')
        cvxmodel.computeIIS()
        cvxmodel.write(IISFILE)

    if cvxmodel.solcount == 0:
        return 0, {}

    print("check gurobi best solution")
    solx =  [v.x for v in cvxmodel._svar.values()]
    print(solx)
    # cvxmodel._clonemodel.write("check.lp")
    solq =  [v.x for v in cvxmodel._qvar.values()]
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
            flow, hreal, volume, nbviolations = cvxmodel._network.extended_period_analysis(inactive, stopatviolation=False)
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

    if cvxmodel.status == GRB.INFEASIBLE:
        print(f"f write IIS in {IISFILE}")
        cvxmodel.computeIIS()
        cvxmodel.write(IISFILE)

    costreal = 0
    plan = {}
    if cvxmodel.SolCount:
        inactive, activity = _parse_activity(instance.horizon(), cvxmodel._svar)
        net = HydraulicNetwork(instance, cvxmodel.Params.FeasibilityTol)
        qreal, hreal, vreal, nbviolations = net.extended_period_analysis(inactive, stopatviolation=False)
        print(f"real plan with {nbviolations} violations")
        plan = {t: {a: (0 if abs(q) < 1e-6 else 1) for a, q in qreal[t].items()} for t in qreal}
        costreal = solutioncost(cvxmodel, plan, qreal)

        if drawsolution:
            graphic.pumps(instance, qreal)
            graphic.tanks(instance, qreal, vreal)

        for a, s in cvxmodel._svar.items():
            print(f'{a}: {round(s.x)} {round(cvxmodel._qvar[a].x, 4)}')

    return costreal, plan


def recordandwritesolution(m, activity, qreal, filename):
    sol = getrealsol(m, activity, qreal)
    f = open(filename, 'a')
    write = csv.writer(f)
    write.writerow(sol)
    return sol


def cloneandchecksolution(m, vals):
    vars = m._lastsol['vars']
    assert len(vals) == len(vars)
    model = m._clonemodel if m._clonemodel else m.copy()
    for i, var in enumerate(vars):
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
