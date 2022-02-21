#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 15:03:46 2022

@author: Sophie Demassey

Build and solve the extended IP

"""

from instance import Instance
import datetime
import extendedip as exip
import configgenerator
from pathlib import Path
from stats import Stat
from datetime import datetime
import os
import time




BENCH = {
    'FSD': {'ntk': 'Simple_Network', 'D0': 1, 'H0': '/01/2013 00:00'},
    'RIC': {'ntk': 'Richmond', 'D0': 21, 'H0': '/05/2013 07:00'},
    'ANY': {'ntk': 'Anytown', 'D0': 1, 'H0': '/01/2013 00:00'},
}
PROFILE = {'s': 'Profile_5d_30m_smooth', 'n': 'Profile_5d_30m_smooth'}
STEPLENGTH = {'12': 4, '24': 2, '48': 1}

PARAMS = {
    'FSD s 24': {'mipgap': 1e-6, 'vdisc': 400, 'safety': 2},
    'FSD s 48': {'mipgap': 1e-6, 'vdisc': 100, 'safety': 25},
    'RIC s 48': {'mipgap': 1e-6, 'vdisc': 1, 'safety': 0},
    'default' : {'mipgap': 1e-6, 'vdisc': 100, 'safety': 2}}

def defaultparam(instid: str):
    params = PARAMS.get(instid[:8])
    return params if params else PARAMS['default']

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
defaultexfilename = Path(OUTDIR, f'resallex.csv')

MODES = {"solve": ['EXIP', 'EXLP'],
         "adjust": ['NOADJUST']}


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


def solve(instance, params, stat, drawsolution):
    print('***********************************************')
    print(instance.tostr_basic())
    print(instance.tostr_network())
    instance.cc_partition_reservoirs()
    return

    print("generate configurations")
    gentime = time.time()
    columns = configgenerator.ConfigGen(instance, params["mipgap"], params["vdisc"], params["safety"])
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

    feasreal, costreal, planreal = exip.solveLP(model, instance, drawsolution=drawsolution) if stat.solvelprelaxation() \
        else exip.solve(model, instance, drawsolution=drawsolution)

    stat.fill(model, costreal)
    print('***********************************************')
    print(f"solution for {instance.tostr_basic()}")
    print(stat.tostr_basic())

    model.terminate()
    return feasreal, costreal, planreal, cpugen, nbgen


def solvebench(bench, params=None, modes=None, drawsolution=False):
    stat = Stat(parsemode(modes))
    now = datetime.now().strftime("%y%m%d-%H%M")
    resfilename = Path(OUTDIR, f'res{now}-ex.csv')
    for i in bench:
        solveinstance(i, params=params, stat=stat, drawsolution=drawsolution, file=resfilename)


def solveinstance(instid, params=None, modes=None, stat=None, drawsolution=True, file=defaultexfilename):
    if params is None:
        params = defaultparam(instid)
    instance = makeinstance(instid)
    stat = Stat(parsemode(modes)) if stat is None else stat
    now = datetime.now().strftime("%y%m%d-%H%M")
    print(now)
    feasreal, costreal, planreal, cpugen, nbgen = solve(instance, params, stat, drawsolution)
    # if cost:
    #    writeplan(instance, plan, f"{now}, {instid}, {cost},")

    fileexists = os.path.exists(file)
    f = open(file, 'a')
    if not fileexists:
        paramstr = ", ".join(list(params.keys()))
        f.write(f"date, ntk T day, mode, {paramstr}, cpucols, nbcols, feasible, {stat.tocsv_title()}\n")
    paramstr = ", ".join([str(v) for v in params.values()])
    f.write(f"{now}, {instid}, {stat.getsolvemode()}, {paramstr}, {int(cpugen)}, "
            f"{nbgen}, {1 if feasreal else 0}, {stat.tocsv_basic()}\n")
    f.close()


#solveinstance('FSD s 24 4', modes='EXIP')
#solveinstance('FSD s 48 1', params=paramsFSD48, modes='EXIP')
#solveinstance('RIC s 48 1', modes='EXIP')

solvebench(FASTBENCH[:6], modes=None)
