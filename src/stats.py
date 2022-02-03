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

    def __init__(self, modes):
        self.modes = modes
        self.fmt = Stat.basicfmt if self._basicformat() else dict(Stat.basicfmt, **(Stat.bbfmt))
        self.all = None
        if self.withadjust():
            self.fmt.update(Stat.adjfmt)

    def _basicformat(self):
        return self.modes["solve"] in set(['CVX', 'EXIP', 'EXLP'])

    def solvelprelaxation(self):
        return self.modes["solve"] == 'EXLP'

    def solveconvex(self):
        return self.modes["solve"] == 'CVX'

    def withadjust(self):
        return self.modes["adjust"] != "NOADJUST"

    def getsolvemode(self):
        return self.modes["solve"]

    def fill(self, model, costreal):
        self.all = {
            'status': model.status,
            'cpu': model.Runtime,
            'ub': model.objVal if model.status != GRB.INFEASIBLE else float('inf'),
            'realub': costreal,
            'lb': model.objBound,
            'gap': model.MIPGap * 100,
            'nodes': int(model.NodeCount),
            'iter': model.IterCount}

        if not self._basicformat():
            self.all['ub'] = model._incumbent if model._solutions else float('inf')
            self.all['gap'] = 100 * (self.all['ub'] - self.all['lb']) / self.all['ub']
            self.all.update(model._intnodes)
            self.all['cpu_cb'] = model._callbacktime
            feassol = [s for s in model._solutions if not s['adjusted']]
            self.all['ub_best'] = feassol[-1]['cost'] if feassol else float('inf')
            self.all['cpu_best'] = feassol[-1]['cpu'] if feassol else 0
            self.all['ub_1st'] = feassol[0]['cost'] if feassol else float('inf')
            self.all['cpu_1st'] = feassol[0]['cpu'] if feassol else 0

        if self.withadjust():
            self.all['nb_adj'] = len(model._adjust_solutions)
            self.all['ub_adj'] = model._adjust_solutions[-1]['cost'] if model._adjust_solutions else float('inf')
            self.all['cpu_adj'] = model._adjust_solutions[-1]['cpu'] if model._adjust_solutions else 0
            self.all['ub_1st_adj'] = model._adjust_solutions[0]['cost'] if model._adjust_solutions else float('inf')
            self.all['cpu_1st_adj'] = model._adjust_solutions[0]['cpu'] if model._adjust_solutions else 0

    def tocsv_title(self):
        return ", ".join(self.fmt.keys())

    def tocsv_basic(self):
        fmtlst = [str(round(self.all[k], f)) for k, f in self.fmt.items()]
        return ", ".join(fmtlst)

    def tostr_basic(self):
        fmtlst = [f"{k}: {round(self.all[k], f)}" for k, f in self.fmt.items()]
        return ", ".join(fmtlst)
