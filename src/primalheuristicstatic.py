#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 15:10:16 2020

@author: sofdem
"""

import gurobipy as gp
from gurobipy import GRB
import time


class AdjustStepLengthHeuristic:

    def __init__(self, instance, network):
        self.instance = instance
        self.network = network
        self.vinit = {j: tank.vinit for j, tank in instance.tanks.items()}
        self.tsinit = instance.tsinhours()
        self.model, self.dvar, self.uvar, self.vctr, self.skip = self._build_global_model()

    def _build_global_model(self):
        """
        Given an unfeasible pumping plan X that violates a tank capacity, adjust the duration 
        for running each configuration X[t] to solve the violations and to minimize the power cost.
    
        Each configuration X[t] is allowed to start up to one time step earlier and to end up to one 
        time step later.
        Solve a binary linear program where each period t is partitioned in 3 consecutive subperiods
        (t,i) of variable lengths, each running configuration X[t+i] for i= -1, 0, 1.
        Flows are estimated on each subperiod (t, i) by running the flow analysis for config X[t+i]
        with demands D[t] and fixed volumes V[t,i] where, at the first iteration:
        V[0,0] = Vinit, V[t-1,1] = V[t,-1] = V[t,0] = V[t-1,0] + Q[t-1,0] * DeltaT 
        and at the next iterations (with dur = subperiod duration found at the previous iteration): 
        V[0,0] = Vinit, V[t,-1] = V[t-1,1]+Q[t-1,1]*dur(t-1,1), V[t,i] = V[t,i-1]+Q[t,i-1]*dur(t,i-1)
        and fixed volume (initially vol[t,i] = volume[t] if i=-1 or 0, and vol[t,1] = volume[t+1].
        The tank volume capacities are also slighty more constrained.
    
        Args:

        Returns:
            model (model): the Gurobi BIP model.
            dvar (dict) : gurobi variable dvar[t][i] of the duration of subperiod (t,i)
            vctr (dict): gurobi constraint vctr[t][j] of the volume conservation at tank j on period t
        """
    
        model = gp.Model('primal')
    
        dvar = {} # duration dvar[t][i] of subperiod (t,i) for i=-1, 0, 1
        uvar = {} # boolean uvar[t][j] iff dvar[t][i] > 0
        vvar = {0: self.vinit} # volume vvar[t][j] of tank j when period t starts
        vctr = {} # constraint vctr[t][j] of volume conservation at tank j during period t

        nperiods = self.instance.nperiods()
        for t in range(nperiods):
            dvar[t] = {}
            uvar[t] = {}
            vctr[t] = {}

            for i in range(-1, 2):
                if (t == 0 and i == -1) or (t == nperiods-1 and i == 1):
                    dvar[t][i] = 0
                    uvar[t][i] = 0
                else:
                    dvar[t][i] = model.addVar(lb=0, ub=self.tsinit)
                    if i != 0:
                        uvar[t][i] = model.addVar(vtype=GRB.BINARY)
                        model.addConstr(dvar[t][i] <= uvar[t][i] * self.tsinit)
                        if i == -1:
                            # either X[t-1] ends later or X[t] starts earlier
                            model.addConstr(uvar[t-1][1] + uvar[t][-1] <= 1)
    
            # period length decomposition
            model.addConstr(gp.quicksum(d for d in dvar[t].values()) == self.tsinit)
    
            # tank volume conservation
            vvar[t+1] = {}
            vctr = {}
            for j, tank in self.instance.tanks.items():
                vlb = tank.vinit if t == nperiods-1 else tank.vmin
                vvar[t+1][j] = model.addVar(lb=vlb + tank.vmax * 0.001, ub=tank.vmax * 0.999)
                vctr[t][j] = model.addConstr(vvar[t+1][j] == vvar[t][j])


        model.ModelSense = GRB.MINIMIZE
    
        return model, dvar, uvar, vctr


    def _update_pumpingplan(self, activity, inactive, volume):
    
        #!!! in Gratien's code only the changes in the pump config (not valve) is considered
        sameasbefore = True  # no need to define subperiods (t-1, 1) and (t,-1) if inactive[t-1]==inactive[t]
    
        vol = self.vinit.copy()
        nperiods = self.instance.nperiods()
        for t in range(nperiods):
            sameasafter = (t == nperiods-1) or (inactive[t] == inactive[t+1])
            for i in range(-1, 2):
                if (i == -1 and sameasbefore) or (i == 1 and sameasafter):
                    if self.uvar[t].get(i):
                        self.uvar[t][i].ub = 0
                else:
                    if self.uvar[t].get(i):
                        self.uvar[t][i].ub = 1

                    q, h = self.network._flow_analysis(inactive[t+i], t, vol)
    
                    # pumping cost on subperiod (t, i)
                    power = sum(pump.powerval(q[a]) for a, pump in self.instance.pumps.items())
                    self.dvar[t][i].obj = power * self.instance.tariff[t] / 1000
    
                    for j in self.instance.tanks:
                        qtank = sum(q[a] for a in self.instance.inarcs(j)) - sum(q[a] for a in self.instance.outarcs(j))
                        self.model.chgCoeff(self.vctr[t][j], self.dvar[t][i], -qtank)
                        if i == 0:
                            vol[j] + qtank * self.instance.tsinhours()
            sameasbefore = sameasafter

        #!!! TO SEE LATER (many redundancy and should be first removed from the preceding call)
        ## PUMP OFF >= 1/2h, ON >= 1h
        #for k, pump in self.instance.pumps.items():
        #    start = 0
        #    for t in range(1, nperiods+1):
        #        if t==nperiods or activity[t][k] != activity[t-1][k]:
        #            end = t
        #            mind = 1 if activity[t-1][k] else 0.5
        #            if (end - start - 2) * self.tsinit < mind:
        #                self.model.addConstr((end - start) * self.tsinit
        #                                     + self.dvar[start-1][1] - self.dvar[start][-1]
        #                                     + self.dvar[end][-1] - self.dvar[end-1][1] >= mind)
        #            start = end

    def _update_nonconvexmodel(self, inactive):
        """
        Run the extended period flow analysis with the new durations. Check feasibility 
        at the tank capacities, compute the linear cost, and update the coeffs of dvar
        in the objective and in the volume conservation constraints vctr given the new volume estimates
        computed at each subperiod (t,i).
    
        Args:
            inactive (Set): the pumping plan as a list of sets inactive[t] = A \ X[t].
    
        Returns:
            feasible (Bool): whether a feasible solution has been found.
            cost (float): the solution linear cost if exists, 0 otherwise.    
        """
    
        feasible = True
        vol = self.vinit.copy()
        cost = 0
        for t in self.dvar:
            for i, d in self.dvar[t].items():
                if d and (i== 0 or self.uvar[t][i].ub == 1):
                    duration = d.x
                    q, h = self.network._flow_analysis(inactive[t+i], t, vol)
    
                    # pumping cost on subperiod (t, i)
                    subcost = sum(pump.powerval(q[a]) for a, pump in self.instance.pumps.items()) \
                        * self.instance.tariff[t] / 1000
                    self.dvar[t][i].obj = subcost
                    cost += subcost * duration
    
                    # volumes at tanks at the end of subperiod (t, i)
                    for j, tank in self.instance.tanks.items():
                        qtank = sum(q[a] for a in self.instance.inarcs(j)) - sum(q[a] for a in self.instance.outarcs(j))
                        self.model.chgCoeff(self.vctr[t][j], self.dvar[t][i], -qtank)
                        vol[j] += qtank * duration
                        if vol[j] < tank.vmin - 1e-6 or vol[j] > tank.vmax + 1e-6:
                            feasible = False
        if vol[j] < tank.vinit:
            feasible = False
        return feasible, cost


    def adjust_steplength(self, activity, inactive, timelimit=60):
        """
        Given an unfeasible pumping plan X that violates a tank capacity, adjust the duration 
        for running each configuration X[t] to solve the violations and to minimize the power cost.
        Each configuration X[t] is allowed to start up to one time step earlier and to end up to one 
        time step later.
    
        Durations are computed by iteratively solving a BIP where hydraulics are approximated and 
        refined at each iteration until a feasible solution (with the nonconvex constraint) is found
        or the timelimit is reached.
        Args:
            activity (dict): the pumping plan as a list of boolean activity[t][a] iff a in X[t].
            inactive (Set): the pumping plan as a list of sets inactive[t] = A \ X[t].
    
        Returns:
            costlinear (float): the solution linear cost if exists, 0 otherwise.
        """
    
        starttime = time.time()
        remainingtime = timelimit
        niter = 0
        feasible = False
    
        self._update_pumpingplan(activity, inactive)
        self.model.params.OutputFlag = 0
    
        while not feasible and remainingtime >= 0:
            self.model.params.timeLimit = remainingtime
            self.model.optimize()
            if self.model.status == GRB.INFEASIBLE():
                print('primal heuristic: unfeasible model')
                break
    
            feasible, costlinear = self.update_nonconvexmodel(inactive)
            niter += 1
            remainingtime += starttime - time.time()
    
        return costlinear if feasible else 0




