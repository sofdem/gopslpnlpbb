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

""" networkanalysis vs hydraulics """
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
    'RIC s 24': {'mipgap': 1e-6, 'vdisc': 3, 'safety': 0},
    'RIC s 48': {'mipgap': 1e-6, 'vdisc': 2, 'safety': 0},
    'ANY s 24': {'mipgap': 1e-6, 'vdisc': 40, 'safety': 30},
    'ANY s 48': {'mipgap': 1e-6, 'vdisc': 20, 'safety': 10},
    'RIY y 24': {'mipgap': 1e-6, 'vdisc': 3, 'safety': 0},
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


def simulateinstance(instid: str, solfilepath: Path = HEIGHTFILE, feastol: float = 1e-4, drawsolution: bool = True):
    """ Run extended network analysis for one instance on the activation plan given in the solution file. """
    print(f'********** SIMULATE PLAN (tol={feastol}) ************************************')
    solution = parseheigthfile(instid, solfilepath).get(instid)
    if solution:
        instance = makeinstance(instid)
        inactiveplan = solution.get('inactiveplan')
        simulateplan(instance, inactiveplan, feastol, drawsolution)


def simulatebench(solfilepath=HEIGHTFILE, feastol=1e-4, drawsolution=False):
    """ Run extended network analysis for bench instances on the activation plans given in the solution file. """
    print(f'********** SIMULATE BENCH (tol={feastol}) ************************************')
    cputime = time.time()
    solution = parseheigthfile("", solfilepath)
    ninfeasible = 0
    nviolation = 0
    meancost = 0
    for k, v in solution.items():
        instance = makeinstance(k)
        inactiveplan = v.get('inactiveplan')
        cost, violations = simulateplan(instance, inactiveplan, feastol, drawsolution)
        meancost += cost
        if violations:
            nviolation += violations
            ninfeasible += 1
    print(f'********** END SIMULATION (tol={feastol}) ************************************')
    cputime = time.time() - cputime
    print(f"solution time: {cputime} seconds.")
    ninstance = len(solution)
    nfeasible = ninstance - ninfeasible
    avgviolation = nviolation // ninfeasible
    print(f"{ninstance} plans: avg cost={meancost/ninstance}; {nfeasible} feasible;"
          f"{avgviolation} violations in average for {ninfeasible} infeasible plans")


def simulateplan(instance: Instance, inactiveplan: dict, feastol: float, drawsolution=True) -> tuple:
    """ run extended network analysis for an instance on the specified activation plan;
        return the plan cost and the list of violations (exceeding the tolerance feastol) """
    print(instance.tostr_basic())
    network = NetworkAnalysis(instance, feastol) if TESTNETANAL \
        else HydraulicNetwork(instance, feastol=feastol)
    cost, plan, flow, volume, violations = _simulate(instance, network, inactiveplan)
    maxviolation = max(violations.values()) if isinstance(violations, dict) else 0
    nviolation = len(violations) if isinstance(violations, dict) else violations
    print(f"cost = {cost} with {nviolation} violations max={maxviolation:.2f}: {violations}")
    if drawsolution:
        graphic.pumps(instance, flow)
        graphic.tanks(instance, flow, volume)
    return cost, nviolation


def _simulate(instance: Instance, network, inactiveplan: dict) -> tuple:
    """ launch extended network analysis for an instance on the specified activation plan;
        return the solution: cost, plan, flow, volume, violations. """
    flow, volume, violations = network.extended_period_analysis(inactiveplan, stopatviolation=False)
    plan = {t: {a: (0 if abs(q) < 1e-6 else 1) for a, q in flow[t].items()} for t in flow}
    cost = instance.solutioncost(plan, flow)
    return cost, plan, flow, volume, violations


def solve(instance: Instance, params: dict, stat: Stat, drawsolution: bool, meanvolprofiles: list = None):
    """ generate and solve the extended pump scheduling model (either LP or ILP according to the mode)"""
    if meanvolprofiles is None:
        meanvolprofiles = []
    print(f'********** SOLVE EXTENDED MODEL ************************************')
    print(instance.tostr_basic())
    print(instance.tostr_network())

    feastol = params["mipgap"]
    network = NetworkAnalysis(instance, feastol) if TESTNETANAL \
        else HydraulicNetwork(instance, feastol=feastol)

    print("generate configurations")
    gentime = time.time()
    columns = configgenerator.ConfigGen(instance, network, feastol, params["vdisc"], params["safety"],
                                        meanvolprofiles)

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
        cost, plan, flow, volume, violations = _simulate(instance, network, inactiveplan)
        stat.fill_realcost(cost)
        print(f"costs: MIP = {model.objVal}, simu = {cost} with {len(violations)} violations\n {violations}")
        if drawsolution:
            graphic.pumps(instance, flow)
            graphic.tanks(instance, flow, volume)

    print('***********************************************')
    print(f"solution for {instance.tostr_basic()}")
    print(stat.tostr_basic())
    print('***********************************************')

    model.terminate()
    return cpugen, nbgen


def solvebench(bench: dict, params: dict = None, modes: str = "", drawsolution: bool = False):
    """ solve the extended model for a given bench of instances. """
    stat = Stat(parsemode(modes))
    now = dt.datetime.now().strftime("%y%m%d-%H%M")
    outfile = Path(OUTDIR, f'res{now}-ex.csv')
    for i in bench:
        solveinstance(i, params=params, stat=stat, drawsolution=drawsolution, outfile=outfile)


def solveinstance(instid: str, params: dict = None, modes: str = "", stat: Stat = None, drawsolution: bool = True,
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
    cpugen, nbgen = solve(instance, params, stat, drawsolution, meanvolprofiles)

    fileexists = os.path.exists(outfile)
    f = open(outfile, 'a')
    if not fileexists:
        paramstr = ", ".join(list(params.keys()))
        f.write(f"date, ntk T day, mode, {paramstr}, cpucols, nbcols, {stat.tocsv_title()}\n")
    paramstr = ", ".join([str(v) for v in params.values()])
    f.write(f"{now}, {instid}, {stat.getsolvemode()}, {paramstr}, {int(cpugen)}, "
            f"{nbgen}, {stat.tocsv_basic()}\n")
    f.close()

solveinstance('FSD s 24 4', modes='EXIP')
# solveinstance('FSD s 48 1', modes='EXIP')
# solveinstance('RIC s 48 1', modes='EXIP')
# solveinstance('ANY s 24 2', modes='EXIP')

# solveinstance('RIY y 48 3', modes='EXIP')

# simulateinstance('RIY y 24 322', feastol=1, drawsolution=True)
# simulatebench(feastol=1e-4, drawsolution=False)
# solveinstance('RIY y 24 322', modes='EXIP')

# solveinstance('RIC s 24 1', modes='EXIP')
# solveinstance('FSD s 24 4', modes='EXIP')


# solvebench(FASTBENCH[:6], modes=None)
