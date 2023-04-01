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
from instance import Instance
from typing import Sequence, Tuple, Union


class NetworkAnalysis:
    """ network partition and simulation of a configuration/plan:
    Attributes:
        instance: pump scheduling problem instance
        feastol: feasibility tolerance for tank capacities
        varccc[a]: number of the connected component the controllable arc a belongs to
        partition[cc][s]: list of fixed demand nodes (s='d'), fixed head nodes ('h'), controllable arcs ('a') in cc
        component[cc]: the PotentialNetwork object for component cc
        violations: the list of violations computed by the simulation: None if stop at first violation
    """

    class FlowError(Exception):
        pass

    # @todo uniformize feastol (in volume or in flow)
    def __init__(self, instance: Instance, feastol: float):
        self.instance = instance
        self.feastol = feastol

        d0junctions = [nid for nid, node in instance.junctions.items() if node.dmean == 0]
        djunctions = [nid for nid, node in instance.junctions.items() if node.dmean != 0]
        netwk = {'d': [*d0junctions, *djunctions],
                 'h': [*instance.tanks, *instance.reservoirs],
                 'a': [*instance.varcs, *instance.farcs]}
        self.partition, self.varccc = NetworkAnalysis._build_partition(netwk, instance.varcs)
        self.component = NetworkAnalysis._build_components(self.partition, instance)
        self.violations = None

    def _erase_history(self):
        """ erase the history of simulations. """
        for cc in self.component.values():
            cc.erasehistory()

    def violation(self, errid: str, period: int, element: Union[str, Sequence[str]], gap: float):
        """ store or raise a violation of an absolute gap value for a given element on a given period. """
        if self.violations is None:
            raise NetworkAnalysis.FlowError(f"{errid} {element}: t={period}, err={gap}")
        else:
            self.violations[(period, element)] = gap

    def extended_period_analysis(self, inactiveplan: dict,
                                 stopatviolation: bool = True, erasehistory: bool = True) -> Tuple:
        """ simulate the plan, given as the inactiveplan[t] the set of inactive arcs at time t, by running flow analysis
        progressively on each time period; stop at first violation if specified by 'stopatviolation'."""
        if erasehistory:
            self._erase_history()
        self.violations = None if stopatviolation else {}
        volinit = {i: tank.vinit for i, tank in self.instance.tanks.items()}
        nperiods = len(inactiveplan)
        volumes = {0: volinit}
        flow = {}
        for t in range(nperiods):
            try:
                flow[t] = self._flow_analysis(inactiveplan[t], t, volumes[t])
                volumes[t+1] = self.next_volumes(flow[t], volumes[t], t, t == nperiods - 1)
            except NetworkAnalysis.FlowError as err:
                print(f'violation at {t + 1}: {err}')
                return flow, volumes, t+1
        violations = 0 if stopatviolation else self.violations
        return flow, volumes, violations

    def next_volumes(self, flow: dict, prevol: dict, t: int, islastperiod: bool) -> dict:
        """ given arc flows flow[a] and previous tank volumes prevol[tk], return new tank volumes postvol[tk]
        generate violations stamped at time t for each capacity exceeded vs feastol. """
        postvol = {}
        for i, tank in self.instance.tanks.items():
            postvol[i] = prevol[i] + self.instance.inflow(i, flow)
            hgap = tank.checkcapacity(postvol[i], islastperiod, self.feastol)
            if hgap != 0:
                self.violation('H', t, i, hgap)
        return postvol

    def _flow_analysis(self, inactivearcs: set, period: int, volumes: dict):
        """ call flow analysis on each component and generate violations if bounds exceeded. """
        flow = {}
        for k, cck in self.partition.items():
            print("aybaba")
            print(cck)
            print(k)
            inactivepart = tuple(aid for aid in cck["a"] if aid in inactivearcs)
            demand = [self.instance.junctions[nid].demand(period) for nid in cck["d"]]
            head = [self.instance.tanks[nid].head(volumes[nid]) if nid in volumes
                    else self.instance.reservoirs[nid].head(period) for nid in cck["h"]]
            subflow = self.component[k].flow_analysis(inactivepart, demand, head)
            # @todo temp test
            self.check_bounds(subflow, period)
            flow.update(subflow)
        return flow
    
    def _flow_analysis_me(self, inactivearcs: set, period: int, volumes: dict):
        """ call flow analysis on each component and generate violations if bounds exceeded. """
        flow = {}
        for k, cck in self.partition.items():
            print("aybaba")
            print(cck)
#            print(volumes)
#            print(volumes.keys())
            inactivepart = tuple(aid for aid in cck["a"] if aid in inactivearcs)
            demand = [self.instance.junctions[nid].demand(period) for nid in cck["d"]]
            head = [volumes[(nid, period)] if (nid, period) in volumes.keys()
                    else self.instance.reservoirs[nid].head(period) for nid in cck["h"]]
            subflow = self.component[k].flow_analysis(inactivepart, demand, head)
            # @todo temp test
#            self.check_bounds(subflow, period)
            flow.update(subflow)
        return flow
    

    def flow_analysis_compon(self, keyy, inactivearcs: set, period: int, volumes: dict):
        """ call flow analysis on each component and generate violations if bounds exceeded. """
        flow = {}
        for k, cck in self.partition.items():
            if keyy==k:
                inactivepart = tuple(aid for aid in cck["a"] if aid in inactivearcs)
                demand = [self.instance.junctions[nid].demand(period) for nid in cck["d"]]
                head = [self.instance.tanks[nid].head(volumes[(nid, period)]) if (nid, period) in volumes
                    else self.instance.reservoirs[nid].head(period) for nid in cck["h"]]
                subflow = self.component[k].flow_analysis(inactivepart, demand, head)
            # @todo temp test
                self.check_bounds(subflow, period)
                flow.update(subflow)
        return flow

    def flow_analysis(self, inactivearcs: set, period: int, volumes: dict, stopatviolation: bool) -> dict:
        """ simulate the command, given as inactivearcs the set of inactive arcs, for a given
         tank volumes configuration, by running flow analysis, and return the arc flows.
         store the violations or stop at the first one as specified by 'stopatviolation'. """
        self.violations = None if stopatviolation else {}
        try:
            flow = self._flow_analysis(inactivearcs, period, volumes)
        except NetworkAnalysis.FlowError as err:
            print(err)
            return {}
        return flow
    
    def flow_analysis_me(self, inactivearcs: set, period: int, volumes: dict) -> dict:
        """ simulate the command, given as inactivearcs the set of inactive arcs, for a given
         tank volumes configuration, by running flow analysis, and return the arc flows.
         store the violations or stop at the first one as specified by 'stopatviolation'. """

        flow = self._flow_analysis_me(inactivearcs, period, volumes)
        return flow

    def check_bounds(self, flows: dict, period: int):
        """ store or raise violation when flow bounds are exceeded. """
        for arc, q in flows.items():
            gap = self.instance.arcs[arc].check_qbounds(q, self.feastol)
            if gap != 0:
                self.violation('Q', period, arc, gap)

    @staticmethod
    def cc_dfs(nid: str, ccs: dict, ccnum: int):
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
    def cc_partition_reservoirs(ccs: dict) -> int:
        """
        get the connected components (cc) of the graph after duplicating all the fixed head nodes.

        Args:
            ccs: the assignment of cc numbers (init=0) to the fixed demand ('d') or head ('h') nodes and the arcs ('a')

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
    def _build_partition(network: dict, varcs: dict) -> Tuple[dict, dict]:
        """
        partition a graph described as lists network[s] (of fixed demand nodes (s='d'), fixed head nodes ('h'),
        and arcs ('a') ) along the fixed head nodes,
        return the partition and the mapping controllable arcs-component number.

        Args:
            network: graph as lists of elements network[s]: demand nodes (s='d'), head nodes ('h'), arcs ('a')
            varcs: list of controllable arcs

        Returns:
            partition: dict of subgraphs (as network) indexed from the component numbers from 1 to ccnum
            varccc: mapping from controllable arcs to their component number
        """
        ccs = {s: {eid: set() if s == 'h' else 0 for eid in idset} for s, idset in network.items()}
        ccnum = NetworkAnalysis.cc_partition_reservoirs(ccs)
        assert 0 not in ccs['a'].values(), "an arc has no cc"
        assert 0 not in ccs['d'].values(), "a junction node has no cc"
        assert set() not in ccs['h'].values(), "a reservoir node has no cc"

        varccc = {aid: ccs['a'][aid] for aid in varcs}

        partition = {cc: {s: [] for s in ccs} for cc in range(1, ccnum+1)}
        for s in ccs:
            for n, cc in ccs[s].items():
                if s == 'h':
                    for ccn in cc:
                        partition[ccn][s].append(n)
                else:
                    partition[cc][s].append(n)
        return partition, varccc

    @staticmethod
    def _build_components(partition: dict, instance: Instance) -> dict:
        """
        create the PotentialNetwork object for each component of the graph partition.
        Args:
            partition: dict of subgraphs (as network) indexed from the component numbers from 1 to ccnum
            instance: pump scheduling instance

        Returns:
            components: dict of PotentialNetwork objects for each component
        """
        components = {}
        for k, cck in partition.items():
            hloss = [instance.arcs[aid].hloss for aid in cck['a']]
            nd0nodes = len([nid for nid in cck['d'] if instance.junctions[nid].dmean == 0])
            nvarcs = len([aid for aid in cck['a'] if aid in instance.varcs])
            components[k] = PotentialNetwork(nd0nodes, nvarcs, hloss, history=True, **cck)
        return components
