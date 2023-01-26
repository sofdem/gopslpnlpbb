# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 16:33:34 2022

@author: amirhossein.tavakoli
"""

from instance import Instance
from datetime import datetime
import convexrelaxation as rel
import lpnlpbb as bb
import csv
import graphic
from hydraulics import HydraulicNetwork
from pathlib import Path
from stats import Stat
import os
import numpy as np
import copy 
import BT_one_st
import BT_h
import new_cvx
import bt_h_

OA_GAP = 1e-2
MIP_GAP = 1e-6

BENCH = {
    'FSD': {'ntk': 'Simple_Network', 'D0': 1, 'H0': '/01/2013 00:00'},
    'RIC': {'ntk': 'Richmond', 'D0': 21, 'H0': '/05/2013 07:00'},
    'ANY': {'ntk': 'Anytown', 'D0': 1, 'H0': '/01/2013 00:00'},
}
PROFILE = {'s': 'Profile_5d_30m_smooth', 'n': 'Profile_5d_30m_smooth'}
STEPLENGTH = {'12': 4, '24': 2, '48': 1}


# ex of instance id: "FSD s 24 3"
def makeinstance(instid: str):
    a = instid.split()
    assert len(a) == 4, f"wrong instance key {instid}"
    d = BENCH[a[0]]
    dbeg = f"{(d['D0'] + int(a[3]) - 1):02d}" + d['H0']
    dend = f"{(d['D0'] + int(a[3])):02d}" + d['H0']
    return Instance(d['ntk'], PROFILE[a[1]], dbeg, dend, STEPLENGTH[a[2]])


FASTBENCH = [
    'FSD s 12 1',
    'FSD s 24 1',
    'FSD s 24 2',
    'FSD s 24 3',
    'FSD s 24 4',
    'FSD s 24 5',
    'FSD s 48 1',
    'RIC s 12 3',
    'RIC s 12 4',
    'RIC s 24 3',
    'RIC s 48 3',
]

OUTDIR = Path("../output/")
defaultfilename = Path(OUTDIR, f'resall.csv')
SOLFILE = Path(OUTDIR, f'solutions.csv')


# RECORD (default: gurobi manages incumbent), FATHOM (cut feas int nodes) or CVX (MIP relaxation only)
# NOADJUST (default: no adjustment heuristic), ADJUST (run heur) or ADJUSTNCUT (cut with heur solutions)
MODES = {"solve": ['RECORD', 'FATHOM', 'CVX'],
         "adjust": ['NOADJUST', 'ADJUST', 'ADJUSTNCUT']}


def parsemode(modes):
    pm = {k: mk[0] for k, mk in MODES.items()}
    if modes is None:
        return pm
    elif type(modes) is str:
        modes = [modes]
    for k, mk in MODES.items():
        for mode in mk:
            if mode in modes:
                pm[k] = mode
                break
    return pm


def solve(instance, oagap, mipgap, drawsolution, stat, arcvals=None):
    print('***********************************************')
    print(instance.tostr_basic())
    print(instance.tostr_network())

    print("obbt: parse bounds")
    try:
        instance.parse_bounds()
    except UnicodeDecodeError as err:
        print(f'obbt bounds not read: {err}')
    # instance.print_all()

    print("create model")
    cvxmodel = rel.build_model(instance, oagap, arcvals=arcvals)
    # cvxmodel.write('convrel.lp')
    cvxmodel.params.MIPGap = mipgap
    cvxmodel.params.timeLimit = 1000
    # cvxmodel.params.OutputFlag = 0
    cvxmodel.params.Threads = 1
    # cvxmodel.params.FeasibilityTol = 1e-5

    print("solve model")
    costreal, plan = bb.solveconvex(cvxmodel, instance, drawsolution=drawsolution) if stat.solveconvex() \
        else bb.lpnlpbb(cvxmodel, instance, stat.modes, drawsolution=drawsolution)

    stat.fill(cvxmodel, costreal)
    print('***********************************************')
    print(f"solution for {instance.tostr_basic()}")
    print(stat.tostr_basic())

    cvxmodel.terminate()
    return costreal, plan


def solveinstance(instid, oagap=OA_GAP, mipgap=MIP_GAP, modes=None, drawsolution=True, stat=None, file=defaultfilename):
    instance = makeinstance(instid)
    stat = Stat(parsemode(modes)) if stat is None else stat
    now = datetime.now().strftime("%y%m%d-%H%M")
    print(now)
    cost, plan = solve(instance, oagap, mipgap, drawsolution, stat)
    if cost:
        writeplan(instance, plan, f"{now}, {instid}, {cost},")
    fileexists = os.path.exists(file)
    f = open(file, 'a')
    if not fileexists:
        f.write(f"date, oagap, mipgap, mode, ntk T day, {stat.tocsv_title()}\n")
    f.write(f"{now}, {oagap}, {mipgap}, {stat.getsolvemode()}, {instid}, {stat.tocsv_basic()}\n")
    f.close()


def writeplan(instance, activity, preamb, solfile=SOLFILE):
    assert len(activity) == instance.nperiods() and len(activity[0]) == len(instance.arcs)
    plan = {a: [activity[t][a] for t in instance.horizon()] for a in instance.varcs}
    f = open(solfile, 'a')
    f.write(f"{preamb} {plan}\n")
    f.close()


def solvebench(bench, oagap=OA_GAP, mipgap=MIP_GAP, modes=None, drawsolution=False):
    stat = Stat(parsemode(modes))
    now = datetime.now().strftime("%y%m%d-%H%M")
    resfilename = Path(OUTDIR, f'res{now}.csv')
    for i in bench:
        solveinstance(i, oagap=oagap, mipgap=mipgap, drawsolution=drawsolution, stat=stat, file=resfilename)


def testsolution(instid, solfilename, oagap=OA_GAP, mipgap=MIP_GAP, modes='CVX', drawsolution=True):
    instance = makeinstance(instid)
    inactive = instance.parsesolution(solfilename)
    network = HydraulicNetwork(instance, feastol=mipgap)
    flow, hreal, volume, nbviolations = network.extended_period_analysis(inactive, stopatviolation=False)
    cost = sum(instance.eleccost(t) * sum(pump.power[0] + pump.power[1] * flow[t][a]
                                          for a, pump in instance.pumps.items() if a not in inactive[t])
               for t in instance.horizon())
    print(f'real plan cost (without draw cost) = {cost} with {nbviolations} violations')
    graphic.pumps(instance, flow)
    graphic.tanks(instance, flow, volume)

    stat = Stat(parsemode(modes))
    arcvals = {(a, t): 0 if a in inactive[t] else 1 for a in instance.varcs for t in instance.horizon()}
    solve(instance, oagap, mipgap, drawsolution, stat, arcvals=arcvals)


def testfullsolutions(instid, solfilename, oagap=OA_GAP, mipgap=MIP_GAP, modes='CVX', drawsolution=True):
    csvfile = open(solfilename)
    rows = csv.reader(csvfile, delimiter=',')
    data = [[float(x.strip()) for x in row] for row in rows]
    csvfile.close()

    print('************ TEST SOLUTIONS ***********************************')
    instance = makeinstance(instid)
    print(instance.tostr_basic())
    print(instance.tostr_network())

    print("obbt: parse bounds")
    try:
        instance.parse_bounds()
    except UnicodeDecodeError as err:
        print(f'obbt bounds not read: {err}')

    stat = Stat(parsemode(modes))
    print("create model")
    for i, d in enumerate(data):
        print(f"create model {i}")
        cvxmodel = rel.build_model(instance, oagap)
        rel.postsolution(cvxmodel, d)
        cvxmodel.params.MIPGap = mipgap
        cvxmodel.params.timeLimit = 1200
        # cvxmodel.params.FeasibilityTol = mipgap
        # network = HydraulicNetwork(instance, feastol=feastol)
        # cvxmodel.write("sd.lp")

        print("solve model")
        costreal, plan = bb.solveconvex(cvxmodel, instance, drawsolution=drawsolution) if stat.solveconvex() \
            else bb.lpnlpbb(cvxmodel, instance, stat.modes, drawsolution=drawsolution)

        stat.fill(cvxmodel, costreal)
        print('***********************************************')
        print(f"solution for {instance.tostr_basic()}")
        print(stat.tostr_basic())

        cvxmodel.terminate()
        
def bound_tight(instance, oagap, arcvals= None):
    
    if instance.name == 'Simple_Network':
        z_flow= np.load('..//data//Simple_Network//Bound0_flow_arcs_fsd.npy',allow_pickle=True)
        zz=z_flow.tolist()
        c_head=np.load('..//data//Simple_Network//Bound0_head_arcs_fsd.npy',allow_pickle=True)
        cc=c_head.tolist()
    
    
    if instance.name == 'Richmond':
        z_flow= np.load('..//data//Richmond//Bound0_flow_arcs_ric.npy',allow_pickle=True)
        zz=z_flow.tolist()
        c_head=np.load('..//data//Richmond//Bound0_head_arcs_ric.npy',allow_pickle=True)
        cc=c_head.tolist()
    
    
    
    

        
    


    if instance.name == 'Richmond':
        length=len(instance.horizon())
###    z_flow= np.load('..//data//Richmond//Bound0_flow_arcs_ric.npy',allow_pickle=True)
###    zz=z_flow.tolist()
###    c_head=np.load('..//data//Richmond//Bound0_head_arcs_ric.npy',allow_pickle=True)
###    cc=c_head.tolist()
    
    
        z_flow= np.load(f'..//data//Richmond//C1_{length}//Bound_flow_arcs.npy',allow_pickle=True)
        Z=z_flow.tolist()
        c_head=np.load(f'..//data//Richmond//C1_{length}//Bound_flow_tanks.npy',allow_pickle=True)
        C=c_head.tolist()
        d_head=np.load(f'..//data//Richmond//C1_{length}//Bound_h_tanks.npy',allow_pickle=True)
        D=d_head.tolist()
        p1_h=np.load(f'..//data//Richmond//C1_{length}//Probed1.npy',allow_pickle=True)
        P1=p1_h.tolist()
        p0_h=np.load(f'..//data//Richmond//C1_{length}//probed0.npy',allow_pickle=True)
        P0=p0_h.tolist()
    
    for tau in range(0, 0):
    
        Y=0
        N=0
        prob_zero1=[]
        warning=[]
        bt_arc_dic=[]
        for (i, j), k in instance.arcs.items():

                
            K=(i, j)
            for t in range(0, len(list(instance.horizon()))):

        
                    if (K, t) in P0:
                        prob_zero1.append([K, t, 0.01])
                        arc_min_it= 0
                        arc_max_it= 0
                        x_n= (K, t)
                        prob_zero= dict([(x_n, prob_zero1[0][2]) ])
                        P0={**P0, **prob_zero}
                
                    else:
                        if tau == 0:
                            arc_min_iter= BT_one_st.build_common_model(instance, Z, C, D, P0, P1, i, j, k, t,  'ARC_MIN', 'oa_cuts', accuracy=0.01, envelop=0.05, oagap=OA_GAP, arcvals=None)
                        else:
                            arc_min_iter= BT_one_st.build_common_model(instance, Z, C, D, P0, P1, i, j, k, t,  'ARC_MIN', 'full_SOS', accuracy=0.01, envelop=0.05, oagap=OA_GAP, arcvals=None)
                        arc_min_iter.optimize()
                        arc_min_iter1= arc_min_iter.getObjective()
                        

                        
        
                        if arc_min_iter.status == 2:
                            arc_min_it= arc_min_iter1.getValue()
                            if tau == 0:
                                arc_max_iter= BT_one_st.build_common_model(instance, Z, C, D, P0, P1, i, j, k, t,  'ARC_MAX', 'oa_cuts', accuracy=0.01, envelop=0.05, oagap=OA_GAP, arcvals=None)
                            else:
                                arc_max_iter= BT_one_st.build_common_model(instance, Z, C, D, P0, P1, i, j, k, t,  'ARC_MAX', 'full_SOS', accuracy=0.01, envelop=0.05, oagap=OA_GAP, arcvals=None)
                            arc_max_iter.optimize()
                            arc_max_iter1= arc_max_iter.getObjective()
                            arc_max_it= arc_max_iter1.getValue()
                            if abs(arc_min_it) <= 0.0001:
                                arc_min_it= 0
                                
                            
                            else:
                                pass
                            
                            bt_arc_dic= dict([((K, t), [max(((arc_min_it)), k.qmin) , min(((arc_max_it)), k.qmax)]) ])
                            
                        else:
            
                            if k.control:
                                arc_min_it= k.qmin
                                arc_max_it= k.qmin
                                prob_zero1.append([K, t, 0.01])
                                x_n= (K, t)
                                prob_zero_=dict([(x_n, 0.01) ])
                                P0={**P0, **prob_zero_}
                                bt_arc_dic= dict([((K, t), [k.qmin , k.qmin+0.0001 ])])
                            else:
                                assert "BT is wrong"
                                arc_min_it= k.qmin
                                arc_max_it= k.qmax
            
                    x_n=[]
                    Z= {**Z, **bt_arc_dic}
        
        
        for K, k in instance.tanks.items():
            for t in range(0, len(list(instance.horizon()))):

                t_infl_min= BT_one_st.build_common_model(instance, Z, C, D, P0, P1, K, K, k, t,  'TANK_MIN', 'partial_SOS', accuracy=0.05, envelop=0.5, oagap=OA_GAP, arcvals=None)
                t_infl_max= BT_one_st.build_common_model(instance, Z, C, D, P0, P1, K, K, k, t,  'TANK_MAX', 'partial_SOS', accuracy=0.05, envelop=0.5, oagap=OA_GAP, arcvals=None)                    
        
                t_infl_min.optimize()
                t_infl_max.optimize()
        
                t_min_= t_infl_min.getObjective()
                t_max_= t_infl_max.getObjective()
        
                t_min= t_min_.getValue()
                t_max= t_max_.getValue()
        
                bt_t_infl= dict([((K, t), [t_min-0.01, t_max+0.01]) ])
        
                Z={**Z, **bt_t_infl}
                
                
        wow=0
        prob_one=[]
        bt_P_off={}
        prob_one_=[]
        for (mm, nn), k in instance.arcs.items():
            for t in range(0, len(list(instance.horizon()))):
##            for t in range(1, 1):
        
                if k.control:
            
                    if ((mm, nn), t) in P1:
                        pass
                    else:
                
            
                        dh_off_mi= BT_one_st.build_common_model(instance, Z, C, D, P0, P1, mm, nn, k, t, 'PUMP_OFF_MIN', 'partial_SOS', accuracy=0.05, envelop=0.5, oagap=OA_GAP, arcvals=None)
                        dh_off_ma= BT_one_st.build_common_model(instance, Z, C, D, P0, P1, mm, nn, k, t, 'PUMP_OFF_MAX', 'partial_SOS', accuracy=0.05, envelop=0.5, oagap=OA_GAP, arcvals=None)

        
                        dh_off_mi.optimize()
                        dh_off_ma.optimize()
                
                        dh_p_mi_off= dh_off_mi.getObjective()
                        dh_p_ma_off= dh_off_ma.getObjective()
                
                
                        if dh_off_mi.status == 2:
                            d0_min_= dh_p_mi_off.getValue()
                            d0_max_= dh_p_ma_off.getValue()
                            bt_P_off= dict([(((mm, nn), t), [max(d0_min_-0.01, k.dhmin), min(d0_max_+0.01, k.dhmax)]) ])
                    
                        else:
                            x_n= ((mm, nn), t)
                            prob_one_.append([(mm, nn), t, 0.99])
                            prob_one= dict([(x_n, 0.99) ])
                            P1={**P1, **prob_one}
                    
                
                        C={**C, **bt_P_off}
                    
                    
            
            

                else:
                    pass
                
        for K, k in instance.tanks.items():
            for t_ in range(1, len(list(instance.horizon()))):
                if tau>=2:
                    pass
                else:
                    if instance.name == 'Richmond':
        
                        h_min_milp= bt_h_.build_model_BT_h(instance, Z, C, D, P0, P1, K, t_, 'MILP', 'oa_cuts', accuracy=0.1, envelop=0.5, Minim= True, two_h= False, oagap=OA_GAP, arcvals=None)
                        h_max_milp= bt_h_.build_model_BT_h(instance, Z, C, D, P0, P1, K, t_, 'MILP', 'oa_cuts', accuracy=0.1, envelop=0.5, Minim= False, two_h= False, oagap=OA_GAP, arcvals=None)
                    else:
                        h_min_milp= BT_h.build_model_BT_h(instance, Z, C, D, P0, P1, K, t_, 'MILP', 'oa_cuts', accuracy=0.1, envelop=0.5, Minim= True, two_h= False, oagap=OA_GAP, arcvals=None)
                        h_max_milp= BT_h.build_model_BT_h(instance, Z, C, D, P0, P1, K, t_, 'MILP', 'oa_cuts', accuracy=0.1, envelop=0.5, Minim= False, two_h= False, oagap=OA_GAP, arcvals=None)        
                    h_min_milp.optimize()
                    h_max_milp.optimize()
        
                    h_min_milp_= h_min_milp.getObjective()
                    h_max_milp_= h_max_milp.getObjective()
                
                    h_min= h_min_milp.ObjBound
                    h_max= h_max_milp.ObjBound
        
#                h_min= h_min_milp_.getValue()
#                h_max= h_max_milp_.getValue()

        
                    bt_h_milp= dict([((K, t_), [max(h_min-0.0001, D[K, t_][0]), min(h_max+0.0001, D[K, t_][1])]) ])
        
                    D={**D, **bt_h_milp}
                
    return Z, C, D, P0, P1



#solveinstance('FSD s 24 2', modes='', drawsolution=False)
#solveinstance('RIC s 12 3', modes='')
# testsolution('RIC s 12 1', "sol.csv")
# testfullsolutions('FSD s 48 4', "solerror.csv", modes="CVX")

###solvebench(FASTBENCH[:7], modes=None)
#solveinstance('FSD s 24 2', modes='', drawsolution=False)


def solve_BT(instance, Z, C, D, P0, P1, oagap, mipgap, drawsolution, stat, arcvals=None):
    print('***********************************************')
    print(instance.tostr_basic())
    print(instance.tostr_network())

    print("obbt: parse bounds")
#    try:
#        instance.parse_bounds()
#    except UnicodeDecodeError as err:
#        print(f'obbt bounds not read: {err}')
    # instance.print_all()

    print("create model")
    cvxmodel = new_cvx.build_model(instance, Z, C, D, P0, P1, oagap, arcvals=arcvals)
####    cvxmodel = rel.build_model(instance, oagap, arcvals=arcvals)
    # cvxmodel.write('convrel.lp')
    cvxmodel.params.MIPGap = mipgap
    cvxmodel.params.timeLimit = 3600
    # cvxmodel.params.OutputFlag = 0
    cvxmodel.params.Threads = 1
    # cvxmodel.params.FeasibilityTol = 1e-5

    print("solve model")
    costreal, plan = bb.solveconvex(cvxmodel, instance, drawsolution=drawsolution) if stat.solveconvex() \
        else bb.lpnlpbb(cvxmodel, instance, stat.modes, drawsolution=drawsolution)

    stat.fill(cvxmodel, costreal)
    print('***********************************************')
    print(f"solution for {instance.tostr_basic()}")
    print(stat.tostr_basic())

    cvxmodel.terminate()
    return costreal, plan

def solveinstance_BT(instid, oagap=OA_GAP, mipgap=MIP_GAP, modes=None, drawsolution=True, stat=None, file=defaultfilename):
    instance = makeinstance(instid)
    stat = Stat(parsemode(modes)) if stat is None else stat
    now = datetime.now().strftime("%y%m%d-%H%M")
    print(now)
    Z, C, D, P0, P1= bound_tight(instance, oagap, arcvals=None)
    cost, plan = solve_BT(instance, Z, C, D, P0, P1, oagap, mipgap, drawsolution, stat)
    if cost:
        writeplan(instance, plan, f"{now}, {instid}, {cost},")
    fileexists = os.path.exists(file)
    f = open(file, 'a')
    if not fileexists:
        f.write(f"date, oagap, mipgap, mode, ntk T day, {stat.tocsv_title()}\n")
    f.write(f"{now}, {oagap}, {mipgap}, {stat.getsolvemode()}, {instid}, {stat.tocsv_basic()}\n")
    f.close()
    
solveinstance_BT('RIC s 12 1', modes='', drawsolution=False)
