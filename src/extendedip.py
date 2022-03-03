#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 15:03:46 2022

@author: Sophie Demassey

Build and solve the extended IP from the set of columns (t,(S,V),(V',E)):

min sum_t sum_(S,V) C_t * E_(t,S,V) * y_(t,S,V)
st.:
sum_(S,V) y_(t,S,V) == 1 forall t
sum_(S,V) V'_(t-1,S,V) * y_(t-1,S,V) == sum_(S,V) V_(t,S,V) * y_(t,S,V) forall t>0

"""

import gurobipy as gp
from gurobipy import GRB


def build_model(inst, confgen):
    milp = gp.Model('exps')

    yvar = [{ck: milp.addVar(vtype=GRB.BINARY, obj=inst.eleccost(t) * confgen.power(cv), name=f"y({t},{ck})".replace(' ', '')) for ck, cv in cols.items()}
            for t, cols in enumerate(confgen.columns)]  # configuration choice

    previouscols = confgen.columns[0]
    for t, cols in enumerate(confgen.columns):
        milp.addConstr(gp.quicksum(yvar[t].values()) == 1, name=f'ctp{t}')
        if t > 0:
            for k, (j, tank) in enumerate(inst.tanks.items()):
                levelpre = gp.quicksum((confgen.volpre(ck, k)+1) * yvar[t][ck] for ck, cv in cols.items())
                levelpost = gp.quicksum((confgen.volpost(cv, k)+1) * yvar[t-1][ck] for ck, cv in previouscols.items())
                milp.addConstr(levelpre == levelpost, name=f'ctl{t},{j}')
        previouscols = cols

    milp._confgen = confgen
    milp._yvars = yvar
    milp.write("exip.lp")
    return milp


def parse_solution(confgen, yvars):
    """ Retrieves the active column in solution yvars for each time and computes the volume configuration profile """
    inactive = {}
    profile = {}
    for t, yt in enumerate(yvars):
        ck = get_active_config(yt)
        assert ck, "no active configuration found"
        inactive[t] = confgen.getinactiveset(ck)
        if t == 0:
            profile[0] = confgen.volpreall(0, ck)
        profile[t+1] = confgen.volpostall(t, ck)
    return inactive, profile


def get_active_config(yvars):
    """ Retrieves the first variable set to 1. """
    for ck, yvar in yvars.items():
        if yvar.x > 0.5:
            return ck


def solve(model):
    """ solve the model then simulates the resulting command plan to check violations or get the real cost/flow. """
    model.optimize()

    if model.status != GRB.OPTIMAL:
        print('Optimization was stopped with plan %d' % model.status)

    if model.SolCount:
        inactive, volprofile = parse_solution(model._confgen, model._yvars)
        return inactive
    else:
        iisfile = "exmodel.ilp"
        print(f"write IIS in {iisfile}")
        model.computeIIS()
        model.write(iisfile)
        return None
