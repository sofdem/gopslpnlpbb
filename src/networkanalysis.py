#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 10:48:46 2022

With network.py should replace hydraulics.py:
NetworkAnalysis partitions the network along the tanks nodes then allows to:
- run network analysis independently on each component for any fixed values of demand or head on each node
 then check the capacity of the shared tanks
- run extended period analysis on the whole planning horizon starting from vinit at t=0

@author: Sophie Demassey
"""

from network import PotentialNetwork


class NetworkAnalysis:

    class FlowError(Exception):
        pass

    def __init__(self, instance, feastol=1e-8):
        self.instance = instance
        self.feastol = feastol

        d0junctions = [nid for nid, node in instance.junctions.items() if node.dmean == 0]
        djunctions = [nid for nid, node in instance.junctions.items() if node.dmean != 0]
        netwk = {'d': [*d0junctions, *djunctions],
                 'h': [*instance.tanks, *instance.reservoirs],
                 'a': [*instance.varcs, *instance.farcs]}
        self.varccc = {arc: 0 for arc in instance.varcs}
        self.partition = NetworkAnalysis._build_partition(netwk, self.varccc)
        self.component = NetworkAnalysis._build_components(self.partition, instance)
        self.violations = None

    def removehistory(self):
        for cc in self.component.values():
            cc.removehistory()

    def violation(self, msg):
        if self.violations:
            self.violations.add(msg)
        else:
            raise NetworkAnalysis.FlowError(msg)

    def extended_period_analysis(self, inactive: dict, stopatviolation=True):
        v0 = {i: tank.vinit for i, tank in self.instance.tanks.items()}
        return self.extended_period_analysis_from(inactive, v0, stopatviolation)

    def extended_period_analysis_from(self, inactive: dict, volinit: dict, stopatviolation: bool):
        """Run flow analysis progressively on each time period."""
        self.violations = None if stopatviolation else []
        nperiods = len(inactive)
        volumes = {0: volinit}
        flow = {}
        for t in range(nperiods):
            try:
                flow[t] = self._flow_analysis(inactive[t], t, volumes[t])
                volumes[t+1] = self.next_volumes(flow[t], volumes[t], t, t == nperiods - 1)
            except NetworkAnalysis.FlowError as err:
                print(f'violation at {t + 1}: {err}')
                return flow, volumes, t+1
        nbviolations = 0 if self.violations is None else len(self.violations)
        return flow, volumes, nbviolations

    def next_volumes(self, flow, prevol, t, lastperiod: bool):
        postvol = {}
        for i, tank in self.instance.tanks.items():
            postvol[i] = prevol[i] + self.instance.flowtovolume() \
                        * (sum(flow[a] for a in self.instance.inarcs(i))
                            - sum(flow[a] for a in self.instance.outarcs(i)))
            if postvol[i] < (tank.vinit if lastperiod else tank.vmin) - self.feastol:
                self.violation(f'capacity tk={i}: {postvol[i] - (tank.vinit if lastperiod else tank.vmin):.2f}')
            if postvol[i] > tank.vmax + self.feastol:
                self.violation(f'capacity tk={i}: {postvol[i] - tank.vmax:.2f}')
        return postvol

    def _flow_analysis(self, inactivearcs: set, period: int, volumes: dict):
        flow = {}
        for k, cck in self.partition.items():
            inactivepart = tuple(aid for aid in cck["a"] if aid in inactivearcs)
            demand = [self.instance.junctions[nid].demand(period) for nid in cck["d"]]
            head = [self.instance.tanks[nid].head(volumes[nid]) if nid in volumes
                    else self.instance.reservoirs[nid].head(period) for nid in cck["h"]]
            subflow = self.component[k].flow_analysis(inactivepart, demand, head, self.feastol)
            self.check_bounds(subflow)
            flow.update(subflow)
        return flow

    def flow_analysis(self, inactive: set, period: int, volumes: dict, stopatviolation):
        """ Computes columns for fixed (period, command/inactive set) for all possible volume configurations. """
        self.violations = None if stopatviolation else []
        try:
            flow = self._flow_analysis(inactive, period, volumes)
        except NetworkAnalysis.FlowError as err:
            if period == 20:
                print(err)
            return 0
        return flow


    def check_bounds(self, flows):
        for arc, q in flows.items():
            if q < self.instance.arcs[arc].qmin - self.feastol:
                self.violation(f"lbound {arc}: q={q} < qmin={self.instance.arcs[arc].qmin}")
            if q > self.instance.arcs[arc].qmax + self.feastol:
                self.violation(f"ubound {arc}: q={q} > qmax={self.instance.arcs[arc].qmax}")

    @staticmethod
    def cc_dfs(nid, ccs, ccnum):
        """
        recursive dfs of the connected component (cc) of node nid
        Args:
            nid: the current visited node in the cc
            ccs: the in-progress assignment of ccs to nodes and arcs
            ccnum: the number (>=1) of the current cc
        """
        cch = ccs['h'].get(nid, -1)
        if cch != -1:
            ccs['h'][nid].add(ccnum)
        elif ccs['d'][nid] == 0:
            ccs['d'][nid] = ccnum
            for aid, acc in ccs['a'].items():
                if acc == 0:
                    for inout in (0, 1):
                        if aid[inout] == nid:
                            ccs['a'][aid] = ccnum
                            NetworkAnalysis.cc_dfs(aid[1-inout], ccs, ccnum)

    @staticmethod
    def cc_partition_reservoirs(ccs):
        """
        get the connected components (cc) of the graph after duplicating all the fixed head nodes.

        Args:
            ccs: the assignment of cc numbers (init=0) to the fixed demand ('d') or head ('h') nodes anad arcs ('a')

        Returns:
            ccnum: the number of ccs (numbered from 1 to ccnum+1)
        """
        ccnum = 0
        for nid, ccn in ccs['d'].items():
            if not ccn:
                ccnum += 1
                NetworkAnalysis.cc_dfs(nid, ccs, ccnum)
        return ccnum

    @staticmethod
    def _build_partition(network, varccc):
        ccs = {s: {eid: set() if s == 'h' else 0 for eid in idset} for s, idset in network.items()}
        ccnum = NetworkAnalysis.cc_partition_reservoirs(ccs)
        assert 0 not in ccs['a'].values(), "an arc has no cc"
        assert 0 not in ccs['d'].values(), "a junction node has no cc"
        assert set() not in ccs['h'].values(), "a reservoir node has no cc"

        for aid in varccc:
            varccc[aid] = ccs['a'][aid]

        partition = {cc: {s: [] for s in ccs} for cc in range(1, ccnum+1)}
        for s in ccs:
            for n, cc in ccs[s].items():
                if s == 'h':
                    for ccn in cc:
                        partition[ccn][s].append(n)
                else:
                    partition[cc][s].append(n)
        return partition

    @staticmethod
    def _build_components(partition, instance):
        components = {}
        for k, cck in partition.items():
            hloss = [instance.arcs[aid].hloss for aid in cck['a']]
            nd0nodes = len([nid for nid in cck['d'] if instance.junctions[nid].dmean == 0])
            nvarcs = len([aid for aid in cck['a'] if aid in instance.varcs])
            components[k] = PotentialNetwork(nd0nodes, nvarcs, hloss, history=True, **cck)
        return components
