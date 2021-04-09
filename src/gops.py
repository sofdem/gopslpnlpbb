#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 11:07:46 2020

@author: Sophie Demassey, Gratien Bonvin

Solve the LPNLP B&B with all the variable bounds previously tighten:
bounds are read from a file (.hdf)

"""

from instance import Instance
import convexrelaxation as rel
import lpnlpbb as bb
import sys

instance = Instance('Simple_Network', 'Profile_5d_30m_smooth', '01/01/2013 00:00', '02/01/2013 00:00', 1)
#instance = Instance('Richmond', 'Profile_5d_30m_smooth', '21/05/2013 07:00', '22/05/2013 07:00', 4)
#instance = Instance('Anytown', 'Profile_5d_30m_smooth', '01/01/2013 00:00', '02/01/2013 00:00', 2)
print('parse:', instance.name, 'horizon start:', instance.periods[0],
      'horizon length:', instance.horizon(), 'timestep:', instance.tsinhours())
print(len(instance.pumps), 'pumps', len(instance.valves), 'valves', len(instance.tanks), 'tanks')

#instance.print_all()

print("obbt: parse bounds")
try:
    #instance.transcript_bounds(f'{instance.name}_bounds.csv')
    instance.parse_bounds()
except UnicodeDecodeError as err:
  print(f'obbt bounds not read: {err}')
#instance.print_all()


print("create model")git
cvxmodel = rel.build_model(instance)
cvxmodel.write('convrel.lp')
cvxmodel.params.timeLimit = 3600
# cvxmodel.params.MIPGap = 0.01
# cvxmodel.params.OutputFlag = 0
# cvxmodel.params.Threads = 1
# cvxmodel.params.FeasibilityTol = 1e-5


print("solve model")
#stats = bb.solveconvex(cvxmodel, instance)
stats = bb.lpnlpbb(cvxmodel, instance, adjust_mode="")

print(f"solution for {instance.tostr_basic()}")
print(stats.tostr_basic())
print(stats.tostr_full())

