#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 11:07:46 2020

@author: Sophie Demassey, Gratien Bonvin
"""

import gurobipy as gp
from gurobipy import GRB
import outerapproximation as oa
from instance import Instance
import datetime as dt


# !!! check round values
# noinspection PyArgumentList
##def build_model(inst: Instance, q_inf, penalt, oagap: float, arcvals=None):
def build_model(inst: Instance, q_inf, penalt, arcvals=None):
    """Build the convex relaxation gurobi model."""

    milp = gp.Model('Pumping_Scheduling')


    hvar = {}  # node head
    qexpr = {}  # node inflow
    
    epsi = {}

    nperiods = inst.nperiods()
    horizon = inst.horizon()

    for t in horizon:

#        for j in inst.junctions:
#            hvar[j, t] = milp.addVar(name=f'hj({j},{t})')

#        for j, res in inst.reservoirs.items():
#            hvar[j, t] = milp.addVar(lb=res.head(t), ub=res.head(t), name=f'hr({j},{t})')
        
        for j, tank in inst.tanks.items():
            lbt = tank.head(tank.vinit) if t == 0 else tank.head(tank.vmin)
            ubt = tank.head(tank.vinit) if t == 0 else tank.head(tank.vmax)
            hvar[j, t] = milp.addVar(lb=lbt, ub=ubt, name=f'ht({j},{t})')

        milp.update()



    for j, tank in inst.tanks.items():
        hvar[j, nperiods] = milp.addVar(lb=tank.head(tank.vinit), ub=tank.head(tank.vmax), name=f'ht({j},{nperiods})')

    milp.update()

    # FLOW CONSERVATION in tanks
    for t in horizon:
        for j, tank in inst.tanks.items():
            qexpr[j, t] = milp.addVar(lb=-1000,ub=1000, name=f'qr({j},{t})')

        for j, tank in inst.tanks.items():
            milp.addConstr(hvar[j, t+1] - hvar[j, t] == inst.flowtoheight(tank) * qexpr[j, t], name=f'fc({j},{t})')
            
    #the absolute value as the two linear constraints
    for t in horizon:
        for j, tank in inst.tanks.items():
            
            epsi[j, t]=milp.addVar(lb=0)
            
            milp.addConstr(epsi[j, t]>= qexpr[j, t]-q_inf[j, t])
            milp.addConstr(epsi[j, t]>= -qexpr[j, t]+q_inf[j, t])
            
    milp.update()


    obj= gp.quicksum(penalt[k, t]*(epsi[k, t])
                      for k, tank in inst.tanks.items() for t in horizon)

    milp.setObjective(obj, GRB.MINIMIZE)
    milp.update()


    milp._hvar = hvar
    milp._obj = obj

    return milp

