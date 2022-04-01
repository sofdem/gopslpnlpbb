# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 15:34:48 2022

@author: amirhossein.tavakoli
"""

import gurobipy as gp
from gurobipy import GRB
import outerapproximation as oa
from instance import Instance
import datetime as dt
import numpy as np
import math
from math import sqrt


def beta(q, coeff):
    return coeff[2] * q * abs(q) + coeff[1] * q + coeff[0]


def beta1(q, coeff):
    return -coeff[2] * q * abs(q) + (-coeff[1]) * q - coeff[0]


# !!! check round values
# noinspection PyArgumentList
def build_common_model(inst: Instance, Z, C, D, P0, P1, m_, n_, k, t, mode_BT, oa_types, accuracy: float, envelop: float, oagap: float, arcvals=None):
    """Build the convex relaxation gurobi model."""

    milp = gp.Model('Pumping_Scheduling')
#    milp.params.NonConvex = 2
    milp.update()

    qvar = {}  # arc flow
    dhvar = {}  # arc hloss
    svar = {}  # arc status
    hvar = {}  # node head
    qexpr = {}  # node inflow
    



    nperiods = inst.nperiods()
    horizon = inst.horizon()


    for j in inst.junctions:
        hvar[j, t] = milp.addVar(name=f'hj({j},{t})')

    for j, res in inst.reservoirs.items():
        hvar[j, t] = milp.addVar(lb=res.head(t), ub=res.head(t), name=f'hr({j},{t})')

    for j, tank in inst.tanks.items():
        lbt = tank.head(tank.vinit) if t == 0 else D[j, t][0]
        ubt = tank.head(tank.vinit) if t == 0 else D[j, t][1]
        hvar[j, t] = milp.addVar(lb=lbt, ub=ubt, name=f'ht({j},{t})')
        hvar[j, t+1] = milp.addVar(lb=tank.head(tank.vmin), ub=tank.head(tank.vmax), name=f'ht({j},{t})')
        if t != nperiods-1:
            hvar[j, t+1] = milp.addVar(lb=D[j, t+1][0], ub=D[j, t+1][1], name=f'ht({j},{t})')
        

    milp.update()

    for (i, j), a in inst.arcs.items():

                    
        if a.control:

            if ((i, j), t) in P0:
                svar[(i, j), t] = milp.addVar(ub=0, vtype=GRB.BINARY, name=f'x({i},{j},{t})')
            elif ((i, j), t) in P1:
                svar[(i, j), t] = milp.addVar(lb=1, vtype=GRB.BINARY, name=f'x({i},{j},{t})')
            else:
                if (i, j) != (m_, n_):
                    svar[(i, j), t] = milp.addVar(vtype=GRB.BINARY, name=f'x({i},{j},{t})')

                else:
                    if  (mode_BT == 'PUMP_OFF_MIN' or mode_BT == 'PUMP_OFF_MAX'):
                        
                        svar[(i, j), t] = milp.addVar(ub=0, vtype=GRB.BINARY, name=f'x({i},{j},{t})')
                    else:                        
                        if (mode_BT == 'ARC_MIN' or mode_BT == 'ARC_MAX'):
                            svar[(i, j), t] = milp.addVar(lb=1, vtype=GRB.BINARY, name=f'x({i},{j},{t})')
                        elif ((mode_BT == 'TANK_MIN' or mode_BT == 'TANK_MAX')):
                            svar[(i, j), t] = milp.addVar(vtype=GRB.BINARY, name=f'x({i},{j},{t})')
                        else:
                            pass
            
            qvar[(i, j), t] = milp.addVar(lb=-GRB.INFINITY, name=f'q({i},{j},{t})')
            dhvar[(i, j), t] = milp.addVar(lb=-GRB.INFINITY, name=f'H({i},{j},{t})')
            milp.addConstr(qvar[(i, j), t] <= Z[(i, j), t][1] * svar[(i, j), t], name=f'qxup({i},{j},{t})')
            milp.addConstr(qvar[(i, j), t] >= Z[(i, j), t][0] * svar[(i, j), t], name=f'qxlo({i},{j},{t})')
            dhmin = max(a.hlossval(a.qmin), hvar[i, t].lb - hvar[j, t].ub)
            dhmax = min(a.hlossval(a.qmax), hvar[i, t].ub - hvar[j, t].lb)
            
            milp.addConstr(dhvar[(i, j), t] <= dhmax * svar[(i, j), t], name=f'dhxup({i},{j},{t})')
            milp.addConstr(dhvar[(i, j), t] >= dhmin * svar[(i, j), t], name=f'dhxlo({i},{j},{t})')
            if (oa_types == 'oa_cuts' or 'partial_SOS'):
                milp.addConstr(dhvar[(i, j), t] <= hvar[i, t] - hvar[j, t] - C[(i, j), t][0] * (1-svar[(i, j), t]), name=f'dhhub({i},{j},{t})')
                milp.addConstr(dhvar[(i, j), t] >= hvar[i, t] - hvar[j, t] - C[(i, j), t][1] * (1-svar[(i, j), t]), name=f'dhhlo({i},{j},{t})')
            else:
                pass
        else:
            qvar[(i, j), t] = milp.addVar(lb=Z[(i, j), t][0], ub=Z[(i, j), t][1], name=f'q({i},{j},{t})')
            dhvar[(i, j), t] = milp.addVar(lb=a.hlossval(a.qmin), ub=a.hlossval(a.qmax), name=f'H({i},{j},{t})')
            svar[(i, j), t] = milp.addVar(vtype=GRB.BINARY, lb=1, name=f'x({i},{j},{t})')
            milp.addConstr(dhvar[(i, j), t] == hvar[i, t] - hvar[j, t], name=f'dhh({i},{j},{t})')            
            
            
 
    for j, tank in inst.tanks.items():
        hvar[j, nperiods] = milp.addVar(lb=tank.head(tank.vinit), ub=tank.head(tank.vmax), name=f'ht({j},T)')

    milp.update()

    # FLOW CONSERVATION

    for j in inst.nodes:
        qexpr[j, t] = gp.quicksum(qvar[a, t] for a in inst.inarcs(j)) \
                          - gp.quicksum(qvar[a, t] for a in inst.outarcs(j))

    for j, junc in inst.junctions.items():
        milp.addConstr(gp.quicksum(qvar[a, t] for a in inst.inarcs(j))
                           - gp.quicksum(qvar[a, t] for a in inst.outarcs(j)) == junc.demand(t), name=f'fc({j},{t})')

    for j, tank in inst.tanks.items():
        milp.addConstr(hvar[j, t+1] - hvar[j, t] == inst.flowtoheight(tank) * qexpr[j, t], name=f'fc({j},{t})')



    # CONVEXIFICATION OF HEAD-FLOW
    for (i, j), arc in inst.arcs.items():


        x = svar[(i, j), t] if arc.control else 1
        if oa_types == 'full_SOS':
            f_SOS(milp, arc, i, j, t, x, dhvar, hvar, qvar, Z, oagap, accuracy, envelop, True)
        elif oa_types == 'partial_SOS':
            f_SOS(milp, arc, i, j, t, x, dhvar, hvar, qvar, Z, oagap, accuracy, envelop, False)
        else:
            cutbelow, cutabove = oa.hlossoa(Z[(i, j), t][0], Z[(i, j), t][1]+0.000001, arc.hloss, (i, j), oagap, drawgraph=False)
#%#        print(f'{arc}: {len(cutbelow)} cutbelow, {len(cutabove)} cutabove')
            for n, c in enumerate(cutbelow):
                milp.addConstr(dhvar[(i, j), t] >= c[1] * qvar[(i, j), t] + c[0] * x, name=f'hpl{n}({i},{j},{t})')
            for n, c in enumerate(cutabove):
                milp.addConstr(dhvar[(i, j), t] <= c[1] * qvar[(i, j), t] + c[0] * x, name=f'hpu{n}({i},{j},{t})')





    strongdualityconstraints(inst, t, Z, C, D, milp, hvar, qvar, svar, dhvar, qexpr, horizon, True)
    

    binarydependencies(inst, t, milp, svar, nperiods, horizon)
        
        

    if arcvals:
        postbinarysolution(inst, t, arcvals, horizon, svar)

    if mode_BT =='ARC_MIN':
        obj = qvar[(m_, n_), t]
        milp.setObjective(obj, GRB.MINIMIZE)
    elif mode_BT == 'ARC_MAX':
        obj = qvar[(m_, n_), t]
        milp.setObjective(obj, GRB.MAXIMIZE)
    elif mode_BT == 'TANK_MIN':
        obj = qexpr[m_, t]
        milp.setObjective(obj, GRB.MINIMIZE)
    elif mode_BT == 'TANK_MAX':
        obj = qexpr[m_, t]
        milp.setObjective(obj, GRB.MAXIMIZE)
    elif mode_BT == 'PUMP_OFF_MIN':
        obj = hvar[m_, t]- hvar[n_, t]
        milp.setObjective(obj, GRB.MINIMIZE)
    elif mode_BT == 'PUMP_OFF_MAX':
        obj = hvar[m_, t]- hvar[n_, t]
        milp.setObjective(obj, GRB.MAXIMIZE)

    else:
        pass
    


    milp.update()

    milp._svar = svar
    milp._qvar = qvar
    milp._hvar = hvar
    milp._obj = obj


    return milp




    


def strongdualityconstraints(inst, t, Z, C, D, milp, hvar, qvar, svar, dhvar, qexpr, horizon, withoutz):
    print("#################  STRONG DUALITY: 5 gvar(pipe) + 10 gvar (pump)")
    # strong duality constraint: sum_a gvar[a,t] + sdexpr[t] <= 0
    gvar = {}    # arc component:    x_a * (\Phi_a(q_a) - \Phi_a(\phi^{-1}(h_a)) + h_a\phi^{-1}(h_a))
    sdexpr = {}  # node component:   sum_n (q_n * h_n)
    hqvar = {}   # tank component:   q_r * h_r
#    for t in horizon:

        # McCormick's envelope of hq_rt = h_rt * q_rt = h_rt * (h_{r,t+1}-h_rt)/c
    for j, tank in inst.tanks.items():
        c = inst.flowtoheight(tank)
        (h0, h1) = (hvar[j, t], hvar[j, t + 1])
        if t == 0:

            (l0, l1, u0, u1) = (h0.lb, D[j, t + 1][0], h0.ub, D[j, t + 1][1])
        elif t == inst.nperiods() - 1:
            (l0, l1, u0, u1) = (D[j, t][0], h1.lb, D[j, t][1], h1.ub)
        else:
            (l0, l1, u0, u1) = (D[j, t][0], D[j, t + 1][0], D[j, t][1], D[j, t + 1][1])
        if l0 == u0:
            hqvar[j, t] = (h1 - l0) * l0 / c
        else:
            hqvar[j, t] = milp.addVar(lb=-GRB.INFINITY, name=f'hqt({j},{t})')
            inflow = {a: [inst.arcs[a].qmin, inst.arcs[a].qmax] for a in inst.inarcs(j)}
            outflow = {a: [inst.arcs[a].qmin, inst.arcs[a].qmax] for a in inst.outarcs(j)}
            print(f"inflow: {inflow}")
            print(f"outflow: {outflow}")
            lq = max(c * inst.inflowmin(j), c * Z[j, t][0], l1 - u0)
            uq = min(c * inst.inflowmax(j), c * Z[j, t][1], u1 - l0)
                # refining with a direction indicator variable
            if withoutz:
                milp.addConstr(c * hqvar[j, t] >= l0 * (h1 - h0) + lq * (h0 - l0), name=f'hqlo({j},{t})')
                milp.addConstr(c * hqvar[j, t] >= u0 * (h1 - h0) + uq * (h0 - u0), name=f'hqup({j},{t})')
            else:
                zvar = milp.addVar(vtype=GRB.BINARY, name=f'z({j},{t})')
                hzvar = milp.addVar(lb=0, ub=u0, name=f'hz({j},{t})')
                milp.addConstr(hzvar <= u0 * zvar, name=f'hz1up({j},{t})')
                milp.addConstr(hzvar >= l0 * zvar, name=f'hz1lo({j},{t})')
                milp.addConstr(hzvar <= h0 - l0 * (1 - zvar), name=f'hz0up({j},{t})')
                milp.addConstr(hzvar >= h0 - u0 * (1 - zvar), name=f'hz0lo({j},{t})')
                milp.addConstr(c * hqvar[j, t] >= l0 * (h1 - h0) + lq * (hzvar - l0 * zvar), name=f'hqlo({j},{t})')
                milp.addConstr(c * hqvar[j, t] >= u0 * (h1 - h0) + uq * (h0 - hzvar - u0 * (1 - zvar)), name=f'hqup({j},{t})')

        # sdexpr[t] = milp.addVar(lb=-GRB.INFINITY, name=f'sd({t})')
    sdexpr[t] = gp.quicksum(hqvar[j, t] for j in inst.tanks) \
        + gp.quicksum(junc.demand(t) * hvar[j, t] for j, junc in inst.junctions.items()) \
        + gp.quicksum(res.head(t) * qexpr[j, t] for j, res in inst.reservoirs.items())

        # OA for the convex function g_a >= Phi_a(q_a) - Phi_a(phi^{-1}(h_a)) + h_a * phi^{-1}(h_a)
    for (i,j), arc in inst.arcs.items():
        a = (i, j)
        gvar[a, t] = milp.addVar(lb=-GRB.INFINITY, name=f'g({i},{j},{t})')
        noacut = 10 if a in inst.pumps else 5
        for n in range(noacut):
            qstar = (arc.qmin + Z[(i, j), t][1]) * n / (noacut - 1)
            milp.addConstr(gvar[a, t] >= arc.hlossval(qstar) *
                               (qvar[a, t] - qstar * svar[a, t]) + qstar * dhvar[a, t], name=f'goa{n}({i},{j},{t})')

    milp.addConstr(gp.quicksum(gvar[a, t] for a in inst.arcs) + sdexpr[t] <= milp.Params.MIPGapAbs, name=f'sd({t})')


def binarydependencies(inst, t, milp, svar, nperiods, horizon):
    # PUMP SWITCHING
    sympumps = inst.symmetries
    uniquepumps = inst.pumps_without_sym()
    print('symmetries:', uniquepumps)

    def getv(vdict, pump, t):
        return gp.quicksum(vdict[a, t] for a in sympumps) if pump == 'sym' else vdict[pump, t]


    # PUMP DEPENDENCIES
    if sympumps:

        for i, pump in enumerate(sympumps[:-1]):
#                milp.addConstr(ivar[pump, t] >= ivar[sympumps[i + 1], t], name=f'symi({t})')
            milp.addConstr(svar[pump, t] >= svar[sympumps[i + 1], t], name=f'symx({t})')

    if inst.dependencies:
        for s in inst.dependencies['p1 => p0']:
            milp.addConstr(svar[s[0], t] >= svar[s[1], t], name=f'dep1({t})')
        for s in inst.dependencies['p0 xor p1']:
#            milp.addConstr(svar[s[0], t] + svar[s[1], t] >= 1, name=f'dep2({t})')
            milp.addConstr(svar[s[0], t] + svar[s[1], t] <= 1, name=f'dep2({t})')
        for s in inst.dependencies['p0 = p1 xor p2']:
            milp.addConstr(svar[s[0], t] == svar[s[1], t] + svar[s[2], t], name=f'dep3({t})')
            # for s in inst.dependencies['p1 => not p0']:
            #    milp.addConstr(svar[s[0], t] + svar[s[1], t] <= 1, name=f'dep4({t})')


def postbinarysolution(inst, t, arcvals, horizon, svar):
    assert arcvals
    for a in inst.varcs:
        v = arcvals.get((a, t))
        if v == 1:
            svar[a, t].lb = 1
        elif v == 0:
            svar[a, t].ub = 0


def postsolution(model, vals, precision=1e-6):
    i = 0
    for var in model._svar:
        var.lb = vals[i]
        var.ub = vals[i]
        i += 1
    for var in model._qvar:
        var.lb = vals[i] - precision
        var.ub = vals[i] + precision
        i += 1


def f_SOS(milp, arc, i, j, t, x, dhvar, hvar, qvar, Z, oagap, accuracy, envelop, full_SOS):

#%#        print(f'{arc}: {len(cutbelow)} cutbelow, {len(cutabove)} cutabove')
#        for t in horizon:


    if arc.control:
        if full_SOS == True:              
            step= sqrt(accuracy /(arc.hloss[2]+abs(arc.hloss[1])))
                    
            numbe= math.floor((Z[(i, j), t][1]-arc.qmin)/step)+1
            if numbe <=5:
                numbe=numbe+3
            else:
                pass
                
            x_samples = np.linspace(arc.qmin, Z[(i, j), t][1], numbe)
            y_samples1 = beta1(x_samples, arc.hloss)+envelop
            y_samples2 = beta1(x_samples, arc.hloss)-envelop


# 1) Instantiate a new model
            dhmax=arc.dhmax
            dhmin=arc.dhmin

            x_ = milp.addVar(lb=arc.qmin, ub=Z[(i, j), t][1], vtype=GRB.CONTINUOUS)
            y = milp.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS)
            weights = milp.addVars(len(x_samples), lb=0, ub=1, vtype=GRB.CONTINUOUS)


            milp.addSOS(GRB.SOS_TYPE2, weights)

            milp.addConstr(gp.quicksum(weights) == 1)
            milp.addConstr(gp.quicksum([weights[i]*x_samples[i] for i in range(len(x_samples))]) == x_)
            milp.addConstr(gp.quicksum([weights[i]*y_samples1[i] for i in range(len(y_samples1))]) >= y)
            milp.addConstr(gp.quicksum([weights[i]*y_samples2[i] for i in range(len(y_samples2))]) <= y)
                
            milp.addConstr(x_ == qvar[(i, j), t])
            milp.addConstr( (hvar[j, t] - hvar[i, t])-y <= (-dhmin+arc.hloss[0])*(1-x))
            milp.addConstr( (hvar[j, t] - hvar[i, t])-y >= (-dhmax+arc.hloss[0])*(1-x))
        elif full_SOS == False:
            cutbelow, cutabove = oa.hlossoa(Z[(i, j), t][0], Z[(i, j), t][1]+0.00001, arc.hloss, (i, j), oagap, drawgraph=False)
#%#        print(f'{arc}: {len(cutbelow)} cutbelow, {len(cutabove)} cutabove')
            for n, c in enumerate(cutbelow):
                milp.addConstr(dhvar[(i, j), t] >= c[1] * qvar[(i, j), t] + c[0] * x, name=f'hpl{n}({i},{j},{t})')
            for n, c in enumerate(cutabove):
                milp.addConstr(dhvar[(i, j), t] <= c[1] * qvar[(i, j), t] + c[0] * x, name=f'hpu{n}({i},{j},{t})')
                            



    else:
            
#            step= 2* sqrt(oagap/arc.hloss[2])
        step= sqrt(accuracy /arc.hloss[2])
        numbe= math.floor((Z[(i, j), t][1]-Z[(i, j), t][0])/step)+1
        if numbe <=3:
            numbe=numbe+2
        else:
            pass
        x_samples = np.linspace(Z[(i, j), t][0], Z[(i, j), t][1], numbe)
        y_samples1 = beta(x_samples, arc.hloss)+envelop
        y_samples2 = beta(x_samples, arc.hloss)-envelop



        x_ = milp.addVar(lb=Z[(i, j), t][0], ub=Z[(i, j), t][1], vtype=GRB.CONTINUOUS)
        y = milp.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS)
        weights = milp.addVars(len(x_samples), lb=0, ub=1, vtype=GRB.CONTINUOUS)


        milp.addSOS(GRB.SOS_TYPE2, weights)

        milp.addConstr(gp.quicksum(weights) == 1)
        milp.addConstr(gp.quicksum([weights[i]*x_samples[i] for i in range(len(x_samples))]) == x_)
        milp.addConstr(gp.quicksum([weights[i]*y_samples1[i] for i in range(len(y_samples1))]) >= y)
        milp.addConstr(gp.quicksum([weights[i]*y_samples2[i] for i in range(len(y_samples2))]) <= y)
                
        milp.addConstr(x_ == qvar[(i, j), t])
        milp.addConstr(y == (hvar[i, t] - hvar[j, t]))

