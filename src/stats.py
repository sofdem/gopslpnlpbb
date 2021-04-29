#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 12:45:57 2021

@author: sofdem
"""

from gurobipy import GRB

class Stat:
    """Statistics for solving one instance."""

    basicfmt = {'ub': 2, 'realub': 2, 'lb': 2, 'gap': 1, 'cpu': 1, 'nodes': 0}
    bbfmt = {'unfeas': 0, 'feas': 0, 'adjust': 0, 'cpu_cb': 1, 'ub_best': 2, 'cpu_best': 1, 'ub_1st': 2, 'cpu_1st': 1}
    adjfmt = {'nb_adj': 0, 'ub_adj': 2, 'cpu_adj': 1, 'ub_1st_adj': 2, 'cpu_1st_adj': 1}

    def __init__(self, mode):
        self.mode = mode
        self.fmt = Stat.basicfmt if mode == "CVX" else dict(Stat.basicfmt, **(Stat.bbfmt))
        if mode == "SOLVE" or mode == "CUT":
            self.fmt.update(Stat.adjfmt)

    def fill(self, model, costreal):
        self.all = {
            'status': model.status,
            'cpu': model.Runtime,
            'ub': model.objVal if model.status != GRB.INFEASIBLE else float('inf'),
            'realub': costreal,
            'lb': model.objBound,
            'gap': model.MIPGap*100,
            'nodes': int(model.NodeCount),
            'iter': model.IterCount }

        if self.mode != 'CVX':
            self.all['ub'] = model._incumbent if model._solutions else float('inf')
            self.all['gap'] = 100 * (self.all['ub'] - self.all['lb']) / self.all['ub']
            self.all.update(model._intnodes)
            self.all['cpu_cb'] = model._callbacktime
            feassol = [s for s in model._solutions if not s['adjusted']]
            self.all['ub_best'] = feassol[-1]['cost'] if  feassol else float('inf')
            self.all['cpu_best'] = feassol[-1]['cpu'] if  feassol else 0
            self.all['ub_1st'] = feassol[0]['cost'] if  feassol else float('inf')
            self.all['cpu_1st'] = feassol[0]['cpu'] if  feassol else 0

        if self.mode == "SOLVE" or self.mode == "CUT":
            self.all['nb_adj'] = len(model._adjust_solutions)
            self.all['ub_adj'] = model._adjust_solutions[-1]['cost'] if  model._adjust_solutions else float('inf')
            self.all['cpu_adj'] = model._adjust_solutions[-1]['cpu'] if  model._adjust_solutions else 0
            self.all['ub_1st_adj'] = model._adjust_solutions[0]['cost'] if  model._adjust_solutions else float('inf')
            self.all['cpu_1st_adj'] = model._adjust_solutions[0]['cpu'] if  model._adjust_solutions else 0

    def tocsv_title(self):
        return ", ".join(self.fmt.keys())

    def tocsv_basic(self):
        fmtlst = [str(round(self.all[k], f)) for k,f in self.fmt.items()]
        return ", ".join(fmtlst)

    def tostr_basic(self):
        fmtlst = [f"{k}: {round(self.all[k], f)}" for k,f in self.fmt.items()]
        return ", ".join(fmtlst)


class OldStat:
    """Statistics for solving one instance."""

    def __init__(self, cvxmodel, instance, costreal):
        self.status = cvxmodel.status
        self.cpu = cvxmodel.Runtime
        self.realub = costreal
        self.lb = cvxmodel.objBound
        self.nodes = cvxmodel.NodeCount
        self.iter = cvxmodel.IterCount
        self.cpu_1st = cvxmodel._solution[0]['cpu'] if cvxmodel._solutions else 0

        if not instance:
            self.instance = cvxmodel._instance
            self.cpu_cb = cvxmodel._callbacktime
            self.ub = cvxmodel._incumbent if cvxmodel._solutions else float('inf')
            self.intnodes = cvxmodel._intnodes
            self.gap = 100 * (self.ub - self.lb) / self.ub
        else:
            self.cpu_cb = 0
            self.ub = cvxmodel.objVal
            self.intnodes = 0
            self.gap = cvxmodel.MIPGap

    @staticmethod
    def tocsv_title():
        return 'ub, real_ub, lb, gap, cpu, cpu_cb, nodes, int_nodes, ub_1st, cpu_1st, ub_1adj, cpu_1adj'

    def tocsv_basic(self):
        return f"{self.ub:.2f}, {self.realub:.2f}, {self.lb:.2f}, {self.gap:.1f}%, {self.cpu:.1f}, {self.cpu_cb:.1f}, {self.nodes:.0f}, {self.intnodes}, {self.cpu_1st:.1f}"
    def tostr_basic(self):
        return f"cost: {self.ub:.2f}, gap: {self.gap:.1f}%, cpu: {self.cpu:.1f}s, cpu_cb: {self.cpu_cb:.1f}s, nodes: {self.nodes:.0f}, {self.intnodes}"
    def tostr_full(self):
        return f"cost: {self.ub:.2f}, realcost: {self.realub:.2f}, lb: {self.lb:.2f}, cpu: {self.cpu:.1f}s, cpu_cb: {self.cpu_cb:.2f}s, nodes: {self.nodes:.0f}, {self.intnodes}"




