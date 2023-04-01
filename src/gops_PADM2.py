#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 11:07:46 2021

@author: Sophie Demassey

Run the B&B on a subset of the easiest instances
bounds are read from a file (.hdf)

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

import datetime as dt

from ast import literal_eval

from networkanalysis import NetworkAnalysis
from new_partition import NetworkPartition

import time
import configgenerator_me
import configgenerator_me_coupling
import second_subproblem
import numpy as np
import random
import math
import pickle

import json

import matplotlib.pyplot as plt

OA_GAP = 1e-2
MIP_GAP = 1e-6



TESTNETANAL = True

BENCH = {
    'FSD': {'ntk': 'Simple_Network', 'H0': '01/01/2013 00:00'},
    'RIC': {'ntk': 'Richmond', 'H0': '21/05/2013 07:00'},
    'ANY': {'ntk': 'Anytown', 'H0': '01/01/2013 00:00'},
    'RIY': {'ntk': 'Richmond', 'H0': '01/01/2012 00:00'},
    'VAN': {'ntk': 'Vanzyl', 'H0': '21/05/2013 07:00'},
}
PROFILE = {'s': 'Profile_5d_30m_smooth', 'n': 'Profile_5d_30m_smooth', 'y': 'Profile_365d_30m_smooth'}
STEPLENGTH = {'12': 4, '24': 2, '48': 1}
PARAMS = {
    'FSD s 24': {'mipgap': 1e-6, 'vdisc': 400, 'safety': 2},
    'FSD s 48': {'mipgap': 1e-6, 'vdisc': 100, 'safety': 25},
    'RIC s 24': {'mipgap': 1e-6, 'vdisc': 3, 'safety': 0},
    'RIC s 48': {'mipgap': 1e-6, 'vdisc': 2, 'safety': 0},
    'ANY s 24': {'mipgap': 1e-6, 'vdisc': 40, 'safety': 30},
    'ANY s 48': {'mipgap': 1e-6, 'vdisc': 20, 'safety': 10},
    'RIY y 24': {'mipgap': 1e-6, 'vdisc': 3, 'safety': 0},
    'VAN s 24': {'mipgap': 1e-6, 'vdisc': 3, 'safety': 0},
    'default' : {'mipgap': 1e-6, 'vdisc': 10, 'safety': 2}}
FASTBENCH = [
    'FSD s 12 1',
    'FSD s 24 1',
    'FSD s 24 2',
    'FSD s 24 3',
    'FSD s 24 4',
    'FSD s 24 5',
    'FSD s 48 1',
    'FSD s 48 2',
    'FSD s 48 3',
    'FSD s 48 4',
    'FSD s 48 5',
    'RIC s 12 3',
    'RIC s 12 4',
    'RIC s 24 3',
    'RIC s 48 3',
]
OUTDIR = Path("../output/")
OUTFILE = Path(OUTDIR, f'resallex.csv')
HEIGHTFILE = Path("../data/Richmond/hauteurs220222.csv")
""" solution mode: 'EXIP' (default: IP extended model), 'EXLP' (LP extended relaxation)
    time adjustment heuristic: NOADJUST (default: no heuristic) """
MODES = {"solve": ['EXIP', 'EXLP'],
         "adjust": ['NOADJUST']}





OUTDIR = Path("../output/")
defaultfilename = Path(OUTDIR, f'resall.csv')
SOLFILE = Path(OUTDIR, f'solutions.csv')




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


def makeinstance(instid: str) -> Instance:
    """ create the instance object named instid, e.g.: "FSD s 24 3". """
    datefmt = "%d/%m/%Y %H:%M"
    a = instid.split()
    assert len(a) == 4, f"wrong instance key {instid}"
    d = BENCH[a[0]]
    day = int(a[3]) - 1
    assert day in range(0, 366)
    dateday = dt.datetime.strptime(d['H0'], datefmt) + dt.timedelta(days=day)
    dbeg = dateday.strftime(datefmt)
    dateday += dt.timedelta(days=1)
    dend = dateday.strftime(datefmt)
    return Instance(d['ntk'], PROFILE[a[1]], dbeg, dend, STEPLENGTH[a[2]])




            
            


def defaultparam(instid: str) -> dict:
    """ return the default parameter values for the given instance. """
    params = PARAMS.get(instid[:8])
    return params if params else PARAMS['default']


def parsemode(modes: str) -> dict:
    """ read the exec mode (see MODES with space separator, e.g.: 'EXIP NOADJUST' ). """
    pm = {k: mk[0] for k, mk in MODES.items()}
    if modes is None:
        return pm
    ms = modes.split()
    for k, mk in MODES.items():
        for mode in mk:
            if mode in ms:
                pm[k] = mode
            break
    return pm


def makeinstance(instid: str) -> Instance:
    """ create the instance object named instid, e.g.: "FSD s 24 3". """
    datefmt = "%d/%m/%Y %H:%M"
    a = instid.split()
    assert len(a) == 4, f"wrong instance key {instid}"
    d = BENCH[a[0]]
    day = int(a[3]) - 1
    assert day in range(0, 366)
    dateday = dt.datetime.strptime(d['H0'], datefmt) + dt.timedelta(days=day)
    dbeg = dateday.strftime(datefmt)
    dateday += dt.timedelta(days=1)
    dend = dateday.strftime(datefmt)
    return Instance(d['ntk'], PROFILE[a[1]], dbeg, dend, STEPLENGTH[a[2]])


def parseheigthfile(instid: str, solfilepath: Path) -> dict:
    """
    parse the solution file including height profiles and return the solution for just one instance is specified;
    return {instid: {'inactiveplan': {t: set(inactive arcs at t), forall time t},
            'dhprofiles': {tk: [dh_tk[t] forall time t] forall tank tk}}}.
    """
    csvfile = open(solfilepath)
    # instid = instid.replace(" y ", " s ") if instid and instid.startswith("RIY y 24") else None
    rows = csv.reader(csvfile, delimiter=';')
    solutions = {}
    for row in rows:
        if (not instid) or (row[0].strip() == instid):
            strdict = literal_eval(row[2].strip())
            solutionplan = {(k[1], k[2]): v for k, v in strdict.items() if len(k) == 3 and k[0] == 'X'}
            dhprofiles = {k[1]: v for k, v in strdict.items() if len(k) == 2 and k[0] == 'DH'}
            inactiveplan = {t: set(k for k, v in solutionplan.items() if v[t] == 0) for t in range(24)}
            solutions[row[0].strip()] = {'inactiveplan': inactiveplan, 'dhprofiles': dhprofiles}
    return solutions






def second_subprob(instance: Instance, q_inf, penalt, params: dict, stat: Stat, drawsolution: bool, meanvolprofiles: list = None):
    if meanvolprofiles is None:
        meanvolprofiles = []
    print(f'********** SOLVE LP MODEL SECOND SUBPROBLEM ************************************')
    print(instance.tostr_basic())
    print(instance.tostr_network())

    feastol = params["mipgap"]
    gentime = time.time()
    cpugen = time.time() - gentime
    model = second_subproblem.build_model(instance, q_inf, penalt)
    model.params.MIPGap = params["mipgap"]
    model.params.timeLimit = 3600
    model.optimize()
    model_lp= model.getObjective()
    obj_lp=model_lp.getValue()
    k_h_=[]
    
    
    for i in range(0,len(model.getVars())):
            k_h_.append([model.VarName[i], model.X[i]])
            k_h_arr=np.array(k_h_)
            keydicts= k_h_arr[:, 0]
            bt_h_p= dict([
                (key, [float(k_h_arr[i][1])]) for key, i in zip(keydicts, range(len(k_h_arr)))])
    level={}
    inf_l={}
    
    #in-outflow of the tanks from second subproblem (LP solution)
    for ts in range(0, len(instance.horizon())):
        for k, tank in instance.tanks.items():
            inf_l[k, ts]= bt_h_p[f'qr({k},{ts})'][0]
            
    # levels from second subproblem (LP solution)       
    for ts in range(0, len(instance.horizon())+1):
        for k, tank in instance.tanks.items():
            level[k, ts]= bt_h_p[f'ht({k},{ts})'][0]
            
    print("solve model")
    model.terminate()
    
    #returning the levels, in-outflows, and objective value of the second subporblem
    return level, inf_l, obj_lp
    


def first_subproblem(instance: Instance, levels, penalt, params: dict, stat: Stat, drawsolution: bool, meanvolprofiles: list = None):
    """ generate and solve the extended pump scheduling model (either LP or ILP according to the mode)"""

    feastol = params["mipgap"]
    network = NetworkAnalysis(instance, feastol) if TESTNETANAL \
        else HydraulicNetwork(instance, feastol=feastol)
    netpart= NetworkPartition(instance)

    print("generate configurations")
    gentime = time.time()
    col={}
    columns = configgenerator_me_coupling.ConfigGen(instance, network, netpart, levels, penalt, feastol, params["vdisc"], params["safety"],
                                        meanvolprofiles)

    for t in instance.horizon():

        col[t]= columns.generate_all_columns_me(t)[t]
           
    new_dict = {}
    for t, inner_dict in col.items():
        new_dict[t] = {}
        for (c, k), value in inner_dict.items():
            if k in new_dict[t]:
                new_dict[t][k].append([c, value])
            else:
                new_dict[t][k] = [[c, value]]
                
    min_value_dic={}
    min_value_nest= {}

    for t in instance.horizon():
        min_value_nest[t]={}
        for compo in new_dict[t].keys():
            min_tempp_value=10000000
            for kj, kn in new_dict[t][compo]: 
                if kn['power']<= min_tempp_value:
                    min_tempp_value= kn['power']
                    min_value_dic[t,compo]= [kj, kn]
                    min_value_nest[t][compo]=[kj,kn]
                else:
                    pass
        
    cpugen = time.time() - gentime
    #returning the time requires to solve first subproblem at each inner iteration, the dictionary of the optimal solution of the first subproblem
    return cpugen, min_value_nest


def solveinstance_PADM_coup(instid: str, lev_initial: dict, perturb, initi_pen, params: dict = None, modes: str = "", stat: Stat = None, drawsolution: bool = True,
                  outfile: Path = OUTFILE):
    """ solve the extended model for a given instance: report the result in 'outfile' """
    if params is None:
        params = defaultparam(instid)
    instance = makeinstance(instid)
    stat = Stat(parsemode(modes)) if stat is None else stat
    now = dt.datetime.now().strftime("%y%m%d-%H%M")
    print(now)
    solution = parseheigthfile(instid, HEIGHTFILE).get(instid)
    dhprofiles = solution.get('dhprofiles') if solution else None
    meanvolprofiles = instance.getvolumeprofiles(dhprofiles)
    
    levels_second, penalt, inflow_first, inflo, big_flag, diff_dict, diff_h1, small_flag, counttt, Livius, tank_in_out, configggg, numb_ite, diff_hh, mattt, powerr = {}, {}, {}, 0, {}, {}, {}, {}, 0, {}, {}, {}, {}, {}, {}, {}
    
    #number of different initialization      
    for it in range(0, 1):   
        for k, m in lev_initial.items():
            levels_second[k] = lev_initial[k] + random.uniform(-perturb, perturb) if k[1] != 0 else lev_initial[k]
        for t in instance.horizon():
            for k, tank in instance.tanks.items():
                inflow_first[k, t]= (levels_second[k, t+1]-levels_second[k, t])/instance.flowtoheight(tank)
                
        Livius[it], tank_in_out[it], small_flag[it], numb_ite[it], mattt[it], diff_h1[it], diff_hh[it], diff_dict[it], powerr[it], penalt[it] = {}, {}, {}, {}, {}, {}, {}, {}, {}, {}
        iteration= 0
        big_flag[it]={}
        #initialization of the penalty
        penalt[it][iteration]={}
        penalt[it][iteration] = {(k, t): initi_pen for t in range(len(instance.horizon())) for k, tank in instance.tanks.items()}               
        q_inff={}
        # the number of outer iteration
        for iteration in range (0, 5):
            Livius[it][iteration], tank_in_out[it][iteration], small_flag[it][iteration], numb_ite[it][iteration], mattt[it][iteration], diff_h1[it][iteration], diff_hh[it][iteration], diff_dict[it][iteration], powerr[it][iteration] = {}, {}, {}, {}, {}, {}, {}, {}, {}
            ite, stopping_cri, out_stop_cri=0, 10000, 1000
            for i in range (0, 85):
                inflow_first={}
                q_inff[i]={}
                stopping_cri_dic={}                
                diff_h1[it][iteration][i], diff_hh[it][iteration][i], diff_dict[it][iteration][i], Livius[it][iteration][i], tank_in_out[it][iteration][i] = {}, {}, {}, {}, {}
                if stopping_cri<0.001 or out_stop_cri <0.001:
                    break
                #first subproblem
                timetoget, min_value_nest = first_subproblem(instance, levels_second, penalt[it][iteration], params, stat, drawsolution, meanvolprofiles)
                inflow_first = {(k, t): min_value_nest[t][cmp][1]['mein_tank'][k] for t in instance.horizon() for cmp, rmd in min_value_nest[t].items() for k, tank in instance.tanks.items()}
                #second subproblem
                levels_second, inflo, objec = second_subprob(instance, inflow_first, penalt[it][iteration], params, stat, drawsolution, meanvolprofiles)
                
                Livius[it][iteration][i]= levels_second
                tank_in_out[it][iteration][i]= inflo
                mattt[it][iteration][i]= min_value_nest
                for t in instance.horizon():
                        for k, tank in instance.tanks.items():                        
                            q_inff[ite][k, t]= inflow_first[k, t]
                            if ite>=1:
                                # computing the stopping criteria: the variations of the in-outflow of the tanks at two consequtive inner loop
                                stopping_cri_dic[k, t]= abs(q_inff[ite][k, t]-q_inff[ite-1][k, t])
                if ite>=1:
                        # h_{\inf} norm
                        stopping_cri=max((stopping_cri_dic.values()))
    
                ite+=1
                
                #nested dictionary shows the trend for convergence
                numb_ite[it][iteration][i]=[i, timetoget, stopping_cri]
                
                for k, val in min_value_nest.items():                   
                    diff_dict[it][iteration][i][k] = [[v[1]['mein_tank'][ln] - inflo[ln, k] for ln, lm in v[1]['mein_tank'].items()] for kk, v in val.items()]
                    diff_h1[it][iteration][i][k] = [[v[1]['tank_level'][ln] - levels_second[ln, k+1] for ln, lm in v[1]['tank_level'].items()] for kk, v in val.items()]

                for k, val in min_value_nest.items():
                    for kk, v in val.items():
                        for ln, lm in v[1]['tank_level'].items():
                            diff_hh[it][iteration][i][ln, k]=v[1]['tank_level'][ln] - levels_second[ln, k+1]
                        
                out_stop_cri= sum(sum(abs(x) for x in klmmn) for ii, jj in diff_h1[it][iteration][i].items() for klmmn in diff_h1[it][iteration][i][ii])
                powerr[it][iteration][i] = sum(min_value_nest[k][mg][1]['power'] for k, m in min_value_nest.items() for mg, mk in m.items())
                

            temppo=0
            for ii, jj in diff_dict[it][iteration][ite-1].items():
                for klmmn in diff_dict[it][iteration][ite-1][ii]:
                    temppo=temppo+sum(klmmn)
            big_flag[it][iteration]=temppo
                
            configggg[it] = min_value_nest
            if abs(big_flag[it][iteration]) <=0.02:
                    counttt+=1
                    break
            penalt[it][iteration+1]={}
            #updating the penalty term for the next outer loop according to violations
            bbb=list(diff_hh[it][iteration].keys())[-1]-1
            for ii, jj in diff_hh[it][iteration][bbb].items():
                    if abs(diff_hh[it][iteration][bbb][ii])>=0.05:
                        penalt[it][iteration+1][ii]= 5*random.uniform(0.75,1)*math.exp(-iteration/10)* penalt[it][iteration][ii]+1                  
                    else:
                        penalt[it][iteration+1][ii]= 2*random.uniform(0.75,1)*math.exp(-iteration/10)* penalt[it][iteration][ii]+1
                    
    return mattt, tank_in_out, stopping_cri, numb_ite, big_flag, Livius, penalt, counttt, powerr, configggg, objec, diff_hh


##instid choices:'VAN s 24 1','VAN s 24 2','VAN s 24 3', 'VAN s 24 4', 'VAN s 48 1', 'VAN s 48 2' 'VAN s 48 3'
instid= 'VAN s 48 1'
instance= makeinstance(instid)

import os

file_dir = os.path.join('..', '..', 'gopslpnlpbb', 'data', 'Vanzyl', 'results')
file_path = os.path.join(file_dir, f'vol_{instid}.json')

if os.path.exists(file_path):
    with open(file_path) as f:
        cc = json.load(f)
else:
    print(f"File '{file_path}' not found in '{file_dir}'.")
#f = open(f'vol_{instid}.json')
#cc = json.load(f)

h_lev={}
                
for k in range(0,len(cc)):
    for kk, v in cc[k].items():
        for tt, tank in instance.tanks.items():  
            if tt== kk:
                h_lev[kk, k]= tank.head(v)

#deviation from optimal tank profile                
perturbation=0.02
#initial penalty weight uniformly for all tanks and time step
init_penalt= 10

sol_first_sp, tank_inflow_second_sp, stopping_cri, numb, big_flag, levels_second_sp, penalization, number_feas_solutions, power, confg, cost_second_sp, levels_viol_dict= solveinstance_PADM_coup(instid, h_lev, perturbation, init_penalt, modes='EXIP')
