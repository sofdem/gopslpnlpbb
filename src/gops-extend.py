#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 15:03:46 2022

@author: Sophie Demassey

Build and solve the extended IP

"""

from instance import Instance
import datetime as dt
import extendedip as exip
import configgenerator
from pathlib import Path
import csv
from ast import literal_eval

from hydraulics import HydraulicNetwork
from networkanalysis import NetworkAnalysis
import graphic
from stats import Stat
import os
import time


TESTNETANAL = True

BENCH = {
    'FSD': {'ntk': 'Simple_Network', 'H0': '01/01/2013 00:00'},
    'RIC': {'ntk': 'Richmond', 'H0': '21/05/2013 07:00'},
    'ANY': {'ntk': 'Anytown', 'H0': '01/01/2013 00:00'},
    'RIY': {'ntk': 'Richmond', 'H0': '01/01/2012 00:00'},
}
PROFILE = {'s': 'Profile_5d_30m_smooth', 'n': 'Profile_5d_30m_smooth', 'y': 'Profile_365d_30m_smooth'}
STEPLENGTH = {'12': 4, '24': 2, '48': 1}

PARAMS = {
    'FSD s 24': {'mipgap': 1e-6, 'vdisc': 400, 'safety': 2},
    'FSD s 48': {'mipgap': 1e-6, 'vdisc': 100, 'safety': 25},
    'RIC s 48': {'mipgap': 1e-6, 'vdisc': 2, 'safety': 0},
    'ANY s 24': {'mipgap': 1e-6, 'vdisc': 40, 'safety': 30},
    'ANY s 48': {'mipgap': 1e-6, 'vdisc': 20, 'safety': 10},
    'RIY y 48': {'mipgap': 1e-6, 'vdisc': 3, 'safety': 0},
    'default' : {'mipgap': 1e-6, 'vdisc': 10, 'safety': 2}}

HEIGHTFILE = "hauteurs220222.csv"

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
defaultexfilename = Path(OUTDIR, f'resallex.csv')

MODES = {"solve": ['EXIP', 'EXLP'],
         "adjust": ['NOADJUST']}

def defaultparam(instid: str):
    params = PARAMS.get(instid[:8])
    return params if params else PARAMS['default']

# possible modes are: 'EXIP' (solve IP extended model), 'EXLP' (solve LP extended relaxation)
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


# ex of instance id: "FSD s 24 3"
def makeinstance(instid: str):
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

def getheightprofile(instid: str, filename = HEIGHTFILE):
    csvfile = open(Path("../data/", BENCH[instid.split()[0]]['ntk'], filename))
    instid = instid.replace(" y ", " s ")
    rows = csv.reader(csvfile, delimiter=';')
    for row in rows:
        if row[0].strip() == instid:
            strdict = literal_eval(row[2].strip())
            heightprofile = {k[1]: v for k, v in strdict.items() if len(k) == 2 and k[0] == 'DH'}
            print(heightprofile)
            return heightprofile

def simulate(instance, network, inactive):
    network.feastol = 1e-4
    network.removehistory()
    #net = HydraulicNetwork(instance, 1e-4)
    flow, volume, nbviolations = network.extended_period_analysis(inactive, stopatviolation=False)
    plan = {t: {a: (0 if abs(q) < 1e-6 else 1) for a, q in flow[t].items()} for t in flow}
    cost = instance.solutioncost(plan, flow)
    return cost, plan, flow, volume, nbviolations


def solve(instance, params, stat, drawsolution):
    print('***********************************************')
    print(instance.tostr_basic())
    print(instance.tostr_network())

    feastol = params["mipgap"]

    network = NetworkAnalysis(instance, feastol=feastol) if TESTNETANAL \
        else HydraulicNetwork(instance, feastol=feastol)

    print("generate configurations")
    gentime = time.time()
    columns = configgenerator.ConfigGen(instance, network, feastol, params["vdisc"], params["safety"])
    cpugen = time.time() - gentime
    nbgen = columns.nbcols()
    print(f"{nbgen} generated in {cpugen} seconds.")

    print("create extended model")
    model = exip.build_model(instance, columns)
    # model.write('extendedmodel.lp')
    model.params.MIPGap = params["mipgap"]
    model.params.timeLimit = 3600
    # model.params.OutputFlag = 0
    # model.params.Threads = 1
    # model.params.FeasibilityTol = 1e-5

    print("solve model")
    inactiveplan = exip.solveLP(model) if stat.solvelprelaxation() else exip.solve(model)
    stat.fill(model)

    if inactiveplan:
        print("simulate plan")
        cost, plan, flow, volume, nbviolations = simulate(instance, network, inactiveplan)
        stat.fill_realcost(cost)
        print(f"costs: MIP = {model.objVal}, simu = {cost} with {nbviolations} violations")
        if drawsolution:
            graphic.pumps(instance, flow)
            graphic.tanks(instance, flow, volume)

    print('***********************************************')
    print(f"solution for {instance.tostr_basic()}")
    print(stat.tostr_basic())
    print('***********************************************')

    model.terminate()
    return cpugen, nbgen


def solvebench(bench, params=None, modes=None, drawsolution=False):
    stat = Stat(parsemode(modes))
    now = dt.datetime.now().strftime("%y%m%d-%H%M")
    resfilename = Path(OUTDIR, f'res{now}-ex.csv')
    for i in bench:
        solveinstance(i, params=params, stat=stat, drawsolution=drawsolution, file=resfilename)


def solveinstance(instid, params=None, modes=None, stat=None, drawsolution=True, file=defaultexfilename):
    if params is None:
        params = defaultparam(instid)
    instance = makeinstance(instid)
    stat = Stat(parsemode(modes)) if stat is None else stat
    now = dt.datetime.now().strftime("%y%m%d-%H%M")
    print(now)
    cpugen, nbgen = solve(instance, params, stat, drawsolution)

    fileexists = os.path.exists(file)
    f = open(file, 'a')
    if not fileexists:
        paramstr = ", ".join(list(params.keys()))
        f.write(f"date, ntk T day, mode, {paramstr}, cpucols, nbcols, {stat.tocsv_title()}\n")
    paramstr = ", ".join([str(v) for v in params.values()])
    f.write(f"{now}, {instid}, {stat.getsolvemode()}, {paramstr}, {int(cpugen)}, "
            f"{nbgen}, {stat.tocsv_basic()}\n")
    f.close()



# solveinstance('FSD s 24 4', modes='EXIP')
solveinstance('FSD s 48 1', modes='EXIP')
# solveinstance('RIC s 48 1', modes='EXIP')
# solveinstance('ANY s 24 2', modes='EXIP')

#solveinstance('RIY y 48 3', modes='EXIP')

#getheightprofile('RIY y 24 322')

# solvebench(FASTBENCH[:6], modes=None)
