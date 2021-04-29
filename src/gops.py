#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 11:07:46 2021

@author: Sophie Demassey

Run the B&B on a subset of the easiest instances
bounds are read from a file (.hdf)

"""

from instance import Instance
from datetime import date
import convexrelaxation as rel
import lpnlpbb as bb
import sys

fastbench = [
    ['FSDs 1 12', 'Simple_Network', 'Profile_5d_30m_smooth', '01/01/2013 00:00', '02/01/2013 00:00', 4],
    ['FSDs 1 24', 'Simple_Network', 'Profile_5d_30m_smooth', '01/01/2013 00:00', '02/01/2013 00:00', 2],
    ['FSDs 2 24', 'Simple_Network', 'Profile_5d_30m_smooth', '02/01/2013 00:00', '03/01/2013 00:00', 2],
    ['FSDs 3 24', 'Simple_Network', 'Profile_5d_30m_smooth', '03/01/2013 00:00', '04/01/2013 00:00', 2],
    ['FSDs 4 24', 'Simple_Network', 'Profile_5d_30m_smooth', '04/01/2013 00:00', '05/01/2013 00:00', 2],
    ['FSDs 5 24', 'Simple_Network', 'Profile_5d_30m_smooth', '05/01/2013 00:00', '06/01/2013 00:00', 2],
    ['FSDs 1 48', 'Simple_Network', 'Profile_5d_30m_smooth', '01/01/2013 00:00', '02/01/2013 00:00', 1],
    ['RICs 3 12', 'Richmond',       'Profile_5d_30m_smooth', '23/05/2013 07:00', '24/05/2013 07:00', 4],
    ['RICs 4 12', 'Richmond',       'Profile_5d_30m_smooth', '24/05/2013 07:00', '25/05/2013 07:00', 4],
]

now = date.today().strftime("%y%m%d")
resfilename = f'res{now}.csv'
f = open(resfilename, 'w')
f.write('ntk day T, ' + bb.Stat.tocsv_title() + '\n')
f.close()

for i in fastbench:

    print('***********************************************')
    instance = Instance(i[1],i[2],i[3],i[4],i[5])
    print(instance.tostr_basic())
    print(instance.tostr_network())


    print("obbt: parse bounds")
    try:
        instance.parse_bounds()
    except UnicodeDecodeError as err:
        print(f'obbt bounds not read: {err}')


    print("create model")
    cvxmodel = rel.build_model(instance)
    # cvxmodel.write('convrel.lp')
    cvxmodel.params.timeLimit = 1000 #3600
    #cvxmodel.params.MIPGap = 0.01
    # cvxmodel.params.OutputFlag = 0
    # cvxmodel.params.Threads = 1
    #cvxmodel.params.FeasibilityTol = 1e-5


    print("solve model")
    # stats = bb.solveconvex(cvxmodel, instance, drawsolution = False)
    stats = bb.lpnlpbb(cvxmodel, instance, drawsolution = False, adjust_mode="CUT")

    print('***********************************************')
    print(f"solution for {instance.tostr_basic()}")
    print(stats.tostr_basic())
    print(stats.tostr_full())

    f = open(resfilename, 'a')
    f.write(f"{i[0]}, {stats.tocsv_basic()}\n")
    f.close()
    cvxmodel.terminate()

