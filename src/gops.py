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

instance = Instance('Simple_Network', 'Profile_5d_30m_smooth', '01/01/2013 00:00', '02/01/2013 00:00', 2)
#instance = Instance('Richmond', 'Profile_5d_30m_smooth', '21/05/2013 07:00', '22/05/2013 07:00', 4)
#instance = Instance('Anytown', 'Profile_5d_30m_smooth', '01/01/2013 00:00', '02/01/2013 00:00', 2)
print('parse:', instance.name, 'horizon start:', instance.periods[0],
      'horizon length:', instance.horizon(), 'timestep:', instance.tsinhours())
print(len(instance.pumps), 'pumps', len(instance.valves), 'valves', len(instance.tanks), 'tanks')


print("obbt: parse bounds")
try:
    instance.parse_bounds()
except UnicodeDecodeError as err:
  print(f'obbt bounds not read: {err}')


print("create model")
cvxmodel = rel.build_model(instance)
cvxmodel.write('convrel.lp')

print("solve model")
#cost, costrelax, time, gap = bb.solveconvex(cvxmodel, instance)
cost, costrelax, time, gap = bb.lpnlpbb(cvxmodel, instance)


print("SOLUTION FOR", instance.name, 'start:', instance.periods[0],
      ' horizon:', instance.horizon(), 'timestep:', instance.tsinhours())
print('costReal:', cost, ' costMIP:', costrelax, ' MIPgap:', gap, ' time:', time)

