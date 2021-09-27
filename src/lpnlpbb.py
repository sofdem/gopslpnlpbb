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
# RECORDSOL = False:
# feasible integer nodes are discarded with nogoodcuts too and incumbent/cutoff are updated/checked outside of Gurobi
# => ObjBound is not a valid lower bound
# RECORDSOL = True:
# feasible integer nodes are updated with bound cut: obj >= (realcost-eps) * (1-X)
# which should invalidate the current MILP solution (when its cost is strictly lower than the real feasible solution)
# the real feasible solution is provided as a heuristic solution to Gurobi
# Problem 1: Gurobi 9.1 allows to set solutions at MIPNODE but not at MIPSOL
# so the solution found at MIPSOL must be recorded to be set at the next MIPNODE event
# Problem 2: even if Gurobi accepts the provided solution, it does not seem to directly update the cutoff value
# in multithreading: it visits nodes where incumbent has been updated but not OBJ_BST
# it vists nodes with higher costs

OUTDIR = Path("../output/")
IISFILE = Path(OUTDIR, f'modeliis.ilp')


def _attach_callback_data(model, instance, modes):
    model._instance = instance
    model._nperiods = instance.nperiods()
    model._network = HydraulicNetwork(instance, model.Params.FeasibilityTol)

    model._incumbent = GRB.INFINITY
    # model._gaptol = model.Params.MIPGapAbs
    model._callbacktime = 0
    model._solutions = []
    model._intnodes = {'unfeas': 0, 'feas': 0, 'adjust': 0}

    model._recordsol = (modes["solve"] == "RECORD")

    if modes["adjust"] != "NOADJUST":
        print("the primal heuristic based on time period adjustment is currently deactivated !")
    model._adjustmode = None
    model._adjusttime = time.time()
    model._adjust_solutions = []

    model._lastsolutioncost = GRB.INFINITY
    vs = list(model._svar.values())
    vs.extend(model._qvar.values())
    model._lastsolution = {'vars': vs}

    model._clonemodel = model.copy()

def mycallback(m, where):

    # STOP if UB-LB < tol
    if where == GRB.Callback.MIP:
        if m._incumbent - m.cbGet(GRB.Callback.MIP_OBJBND) < m.Params.MIPGap * m._incumbent:
            print('Stop early - ', m.Params.MIPGap * 100, '% gap achieved')
            m.terminate()

    elif m._recordsol and where == GRB.Callback.MIPNODE:
        if m._lastsolutioncost < GRB.INFINITY:
            oldbest = m.cbGet(GRB.Callback.MIPNODE_OBJBST)
            oldbest = 'inf' if oldbest == GRB.INFINITY else f"{oldbest:.6f}"
            assert len(m._lastsolution['vars']) == len(m._lastsolution['vals'])
            m.cbSetSolution(m._lastsolution['vars'], m._lastsolution['vals'])
            objval = m.cbUseSolution()
            print(f"MIPNODE #{int(m.cbGet(GRB.Callback.MIPNODE_NODCNT))} oldbest = {oldbest} "
                  f"set solution #{m.cbGet(GRB.Callback.MIPNODE_SOLCNT)}: {m._lastsolutioncost} -> {objval}")
            if objval >= GRB.INFINITY:
                solvecvxmodelwithsolution(m, m._lastsolution['vars'], m._lastsolution['vals'])
                print("if MILP feasible then either the solution violates a lazy cut or a similar solution has already been recorded")

            m._lastsolutioncost = GRB.INFINITY
        # if not m._rootlb:
        #    m._rootlb = m.cbGet(GRB.Callback.MIPNODE_OBJBND)

    # at an integer solution
    elif where == GRB.Callback.MIPSOL:

        costmipsol = m.cbGet(GRB.Callback.MIPSOL_OBJ)
        bestmipsol = m.cbGet(GRB.Callback.MIPSOL_OBJBST)
        currentlb = m.cbGet(GRB.Callback.MIPSOL_OBJBND)
        fstring = f"MIPSOL #{int(m.cbGet(GRB.Callback.MIPSOL_NODCNT))}: {costmipsol} [{currentlb:.2f}, " \
                  f"{'inf' if m._incumbent == GRB.INFINITY else round(m._incumbent, 2)}]" #, best = {bestmipsol}"

        # prevent to evaluate again the same plan... warning: Gurobi will keep this MILP solution as feasible !
        # 1/ set by useSolution at the next MIPNODE (then MIPSOL is called again)
        # 2/ recomputed after adding the bound cut (sometimes (why not always ??) MIPSOL is called again) (=> the 10*)
        # 3/ due to multithreading delays (the incumbent update seems to be communicated but not the cutoff value)
        if costmipsol > m._incumbent - 10*m.Params.MIPGapAbs:
            print(fstring + f" obj >= incumbent {m._incumbent}... SKIP (MILP solution accepted) ")
            if m._lastsolutioncost < GRB.INFINITY:
                print("last solution has not yet been posted to the solver... let gurobi record this one instead")
                m._lastsolutioncost = GRB.INFINITY
                assert costmipsol
                sol = [0 if m.cbGetSolution(svar) < 0.5 else 1 for svar in m._svar.values()]
                assert sol == m._lastsolution['vals'][:len(sol)], "plan has changed between 2 consecutive MIPSOL ???"
            return

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
        costrealsol = GRB.INFINITY

        if violation:
            v = violation[0]
            m._intnodes['unfeas'] += 1
            nogood_lastperiod = v[0]
            print(fstring + f' violation t={v[0]} tk={v[1]}: {v[2]:.2f}')
            addnogoodcut(m, _linearnorm(m._svar, nogood_lastperiod, activity))

        else:
            # TODO set activity[t][a] = 0 when qreal[a][t] == 0 ? remove svar[a,t] from nogood ?
            m._intnodes['feas'] += 1
            costrealsol = solutioncost(m, activity, qreal)
            print(fstring + f" feasible: {costrealsol}")

            # assert costrealsol >= costmipsol, 'relaxed cost > real cost !! '
            if costrealsol < costmipsol - m.Params.MIPGapAbs:
                print(f"relaxed cost {costmipsol} > real cost {costrealsol} !!!!!!!!!!!!!! best={bestmipsol}")
                print("######################## test MILP solution")
                solrelax = [m.cbGetSolution(var) for var in m._lastsolution['vars']]
                solvecvxmodelwithsolution(m, m._lastsolution['vars'], solrelax)
                print("######################## test real solution")
                solreal = [activity[t][a] for (a, t) in m._svar]
                for (a, t) in m._qvar:
                    solreal.append(qreal[t][a])
                solvecvxmodelwithsolution(m, m._lastsolution['vars'], solreal)

            if m._recordsol:
                addboundcut(m, _linearnorm(m._svar, nogood_lastperiod, activity), costrealsol)
                # print(f"new bound cut at {costrealsol - m.Params.MIPGapAbs}")
            else:
                addnogoodcut(m, _linearnorm(m._svar, nogood_lastperiod, activity))

        if costrealsol < m._incumbent:
            m._incumbent = costrealsol
            m._solutions.append({'plan': activity, 'cost': costrealsol, 'flows': qreal, 'volumes': vreal,
                                 'cpu': m.cbGet(GRB.Callback.RUNTIME), 'adjusted': (qreal is None)})
            gap = (m._incumbent - currentlb) / m._incumbent
            print(f'UPDATE INCUMBENT gap={gap * 100:.4f}%')
            # ": {m._incumbent} -> {costreal} (lb={m.cbGet(GRB.Callback.MIPSOL_OBJBND):2f})')

            if m._recordsol:
                # print(f"new solution real={costrealsol} relax={costmipsol} old={bestmipsol}")
                m._lastsolutioncost = costrealsol
                sol = [activity[t][a] for (a, t) in m._svar]
                for (a, t) in m._qvar:
                    sol.append(qreal[t][a])
                # for (j,t) in m._hvar:
                #     sol.append(hreal[t][j])
                m._lastsolution['vals'] = sol

            # if gap < m.Params.MIPGap:
            #    print(f"Stop early - {100*gap:2f}% gap achieved")
            #    m.terminate()
            # !!! cbcSetSolution seems to not change the cutoff value even if MISOL_OBJBND is correctly updated
            elif not m._recordsol:
                addcutoff(m, (1 - m.Params.MIPGap) * m._incumbent)

        m._callbacktime += time.time() - m._starttime


def addnogoodcut(m, linnorm):
    m.cbLazy(linnorm >= 1)
    if m._clonemodel:
        c = clonelinexpr(m, linnorm)
        m._clonemodel.addConstr(c >= 1)


def addboundcut(m, linnorm, costrealsol):
    m.cbLazy(m._obj >= (costrealsol - m.Params.MIPGapAbs) * (1 - linnorm))
    if m._clonemodel:
        c = clonelinexpr(m, linnorm)
        cobj = m._clonemodel.getObjective()
        m._clonemodel.addConstr(cobj >= (costrealsol - m.Params.MIPGapAbs) * (1 - c))

def addcutoff(m, cutoff):
    m.cbLazy(m._obj <= cutoff)
    if m._clonemodel:
        cobj = m._clonemodel.getObjective()
        m._clonemodel.addConstr(cobj <= cutoff)


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
    # print('nogood:', str(linexpr))
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
        print(f'no solution found write IIS file {IISFILE}')
        cvxmodel.computeIIS()
        cvxmodel.write(IISFILE)

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
        net = HydraulicNetwork(instance, cvxmodel.Params.FeasibilityTol)
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
        print(f"f write IIS in {IISFILE}")
        cvxmodel.computeIIS()
        cvxmodel.write(IISFILE)

    return costreal


def recordandwritesolution(m, activity, qreal, filename):
    sol = [activity[t][a] for (a, t) in m._svar]
    for (a, t) in m._qvar:
        sol.append(qreal[t][a])
    f = open(filename, 'a')
    write = csv.writer(f)
    write.writerow(sol)
    return sol


def solvecvxmodelwithsolution(m, vars, vals):
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
