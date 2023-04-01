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


class NetworkPartition:
    """ network partition and simulation of a configuration/plan:
    Attributes:
        instance: pump scheduling problem instance
        feastol: feasibility tolerance for tank capacities
        varccc[a]: number of the connected component the controllable arc a belongs to
        partition[cc][s]: list of fixed demand nodes (s='d'), fixed head nodes ('h'), controllable arcs ('a') in cc
        component[cc]: the PotentialNetwork object for component cc
        violations: the list of violations computed by the simulation: None if stop at first violation
    """

#    class FlowError(Exception):
#        pass

    # @todo uniformize feastol (in volume or in flow)
    def __init__(self, instance: Instance):
        self.instance = instance

        d0junctions = [nid for nid, node in instance.junctions.items() if node.dmean == 0]
        djunctions = [nid for nid, node in instance.junctions.items() if node.dmean != 0]
        netwk = {'d': [*d0junctions, *djunctions],
                 'h': [*instance.tanks, *instance.reservoirs],
                 'a': [*instance.varcs, *instance.farcs]}
        self.partition, self.varccc = NetworkPartition._build_partition(netwk, instance.varcs)
        self.var_inv = NetworkPartition._build_partition_me_lanati(netwk, instance.varcs)
        self.component = NetworkPartition._build_components(self.partition, instance)
#        self.violations = None

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
                            NetworkPartition.cc_dfs(aid[1-inout], ccs, ccnum)

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
                NetworkPartition.cc_dfs(nid, ccs, ccnum)
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
        ccnum = NetworkPartition.cc_partition_reservoirs(ccs)
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
    def _build_partition_me(network: dict, varcs: dict) -> Tuple[dict, dict]:
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
            var_inv: mapping from component number to their controllable arcs
        """
        ccs = {s: {eid: set() if s == 'h' else 0 for eid in idset} for s, idset in network.items()}
        ccnum = NetworkPartition.cc_partition_reservoirs(ccs)
        assert 0 not in ccs['a'].values(), "an arc has no cc"
        assert 0 not in ccs['d'].values(), "a junction node has no cc"
        assert set() not in ccs['h'].values(), "a reservoir node has no cc"

        varccc = {aid: ccs['a'][aid] for aid in varcs}
        var_inv= {ccs['a'][aid]: aid for aid in varcs}

        partition = {cc: {s: [] for s in ccs} for cc in range(1, ccnum+1)}
        for s in ccs:
            for n, cc in ccs[s].items():
                if s == 'h':
                    for ccn in cc:
                        partition[ccn][s].append(n)
                else:
                    partition[cc][s].append(n)
        return varccc
    
    @staticmethod
    def _build_partition_me_lanati(network: dict, varcs: dict) -> Tuple[dict, dict]:
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
            var_inv: mapping from component number to their controllable arcs
        """
        ccs = {s: {eid: set() if s == 'h' else 0 for eid in idset} for s, idset in network.items()}
        ccnum = NetworkPartition.cc_partition_reservoirs(ccs)
        assert 0 not in ccs['a'].values(), "an arc has no cc"
        assert 0 not in ccs['d'].values(), "a junction node has no cc"
        assert set() not in ccs['h'].values(), "a reservoir node has no cc"

        varccc = {aid: ccs['a'][aid] for aid in varcs}
#        aidd=[]
        
#        check_li=[]
        
        var_inv= dict()
        for key in varccc.keys():
            
            if varccc[key] in var_inv:
                var_inv[varccc[key]].append(key)
            else:
                var_inv[varccc[key]] = [key]
        
        
 #       for ii in range(0, len(ccs['a'])):
            
            
            
            
#        for aid in varcs:
            
#            if ccs['a'][aid] in check_li:
                
#            aidd.append(aid)
#            var_inv={ccs['a'][aid]:aidd}
#            check_li.append(ccs['a'][aid])
##        var_inv= {ccs['a'][aid]: aid for aid in varcs}


        partition = {cc: {s: [] for s in ccs} for cc in range(1, ccnum+1)}
        for s in ccs:
            for n, cc in ccs[s].items():
                if s == 'h':
                    for ccn in cc:
                        partition[ccn][s].append(n)
                else:
                    partition[cc][s].append(n)
        return var_inv

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
