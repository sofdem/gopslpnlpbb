#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 11:07:46 2020

@author: Sophie Demassey, Gratien Bonvin
"""

import numpy as np
import instance as inst
from copy import deepcopy


class HydraulicNetwork:
    """An alternative view of the water network with aggregated valves."""

    def __init__(self, instance: inst.Instance, feastol: float):
        """Create an HydraulicNetwork from an Instance."""
        self.instance = instance
        self.feastol = feastol
        self.arcs, self.incidence, self.aggregate = self._aggregate(instance)
        assert self._check_network(self.incidence)

    def _aggregate(self, instance):
        """Get the network with valves aggregated to the preceding pipes."""
        arcs = {}
        incidence = {n: set() for n in instance.nodes}
        aggregate = {}

        for (i, j), pump in instance.pumps.items():
            arcs[(i, j)] = [-round(h, 8) for h in pump.hgain]
            incidence[i].add((i, j))
            incidence[j].add((i, j))

        # !!!  aggregate pipes to valves rather than the opposite
        for (i, j), pipe in instance.pipes.items():
            outvalves = [v for v in instance.outarcs(j) if instance.valves.get(v)]
            assert len(outvalves) < 2, f'several valves after pipe {(i,j)}: {outvalves}'
            newj = j
            if len(outvalves) == 1:
                valve = outvalves[0]
                newj = valve[1]
                aggregate[(i, newj)] = [(i, j), valve]
                aggregate[valve] = (i, newj)
                assert (not incidence[j]), f'impossible to aggregate {(i,j)} + {valve}'
                del incidence[j]
                # print("aggregate ", (i, j), ' + ', valve)
            arcs[(i, newj)] = [round(h, 8) for h in pipe.hloss]
            incidence[i].add((i, newj))
            incidence[newj].add((i, newj))

        return arcs, incidence, aggregate

    def _check_network(self, incidence):
        check = True
        for nodeid, arcs in incidence.items():
            if len(arcs) == 0:
                check = False
                print("Error: isolated node", nodeid)
            node = self.instance.junctions.get(nodeid)
            if node and node.dmean == 0 and len(arcs) == 1:
                check = False
                print("Error: leaf no-demand junction", nodeid)
        return check

    def _remove_arc_from_junction(self, j, arc, incidence):
        removejunction = False
        assert arc in incidence[j], f'arc {arc} not in incidence[{j}]= {incidence[j]}'
        incidence[j].remove(arc)
        if len(incidence[j]) == 0:
            del incidence[j]
        elif len(incidence[j]) == 1:
            junction = self.instance.junctions.get(j)
            removejunction = junction and junction.dmean == 0
        return removejunction

    # !!! merge active_network and build_TP_matrices, i.e. directly work on the incidence matrices
    # !!! regenerate from the period before if no new active element; identical OR new inactives
    def active_network(self, inactive):
        """Remove the inactive pumps and valves then the non-demand leaf nodes, recursively."""
        arcs = deepcopy(self.arcs)
        incidence = deepcopy(self.incidence)

        leavestoremove = set()
        for (i, j) in inactive:
            aggreg = self.aggregate.get((i, j))
            if aggreg:
                assert aggreg[1] == j, f'valve = {(i,j)}, aggregate = {aggreg}'
                i = aggreg[0]
            del arcs[(i, j)]
            if self._remove_arc_from_junction(i, (i, j), incidence):
                leavestoremove.add(i)
            if self._remove_arc_from_junction(j, (i, j), incidence):
                leavestoremove.add(j)

        while leavestoremove:
            leaf = leavestoremove.pop()
            # !!! it is possible that leaf has already been droped: replace assert by condition
            assert incidence.get(leaf), f'node {leaf} has already been removed'
            assert len(incidence[leaf]) == 1, f'{leaf} not a leaf: {incidence[leaf]}'
            branch = incidence[leaf].pop()
            del incidence[leaf]
            del arcs[branch]
            altj = branch[1] if (branch[0] == leaf) else branch[0]
            if self._remove_arc_from_junction(altj, branch, incidence):
                leavestoremove.add(altj)

        assert arcs, "the active network is empty"
        return arcs, incidence


    def build_TP_matrices(self, arcs, incidence, period, volumes):
        """Build the matrices for the Todini-Pilati algorithm."""
        nodeindex = {}

        # create q and H0: the column vectors of fixed demand and fixed head nodes
        demand = []
        head = []
        ndemand = 0
        nhead = 0
        for nodeid in incidence:
            node = self.instance.nodes[nodeid]
            if isinstance(node, inst._Junction):
                demand.append(round(node.demand(period), 2))
                nodeindex[nodeid] = (True, ndemand)
                ndemand += 1
            else:
                nodehead = node.head(period if isinstance(node, inst._Reservoir) else volumes[nodeid])
                head.append(round(nodehead, 2))
                nodeindex[nodeid] = (False, nhead)
                nhead += 1

        q = np.array(demand).reshape(ndemand, 1)
        H0 = np.array(head).reshape(nhead, 1)

        # create the incidence matrices over fixed demand and fixed head nodes
        # print('SIZE = ', ndemand, len(arcs), nhead)
        A21 = np.zeros((ndemand, len(arcs)))
        A12 = np.zeros((len(arcs), ndemand))
        A01 = np.zeros((nhead, len(arcs)))
        A10 = np.zeros((len(arcs), nhead))

        for (inout, val) in {(1, 1), (0, -1)}:
            for j, arc in enumerate(arcs):
                (isDemand, i) = nodeindex.get(arc[inout])
                if isDemand:
                    A21[i][j] = val
                    A12[j][i] = val
                else:
                    A01[i][j] = val
                    A10[j][i] = val
        return nodeindex, q, H0, A21, A12, A10


    def _flow_analysis(self, inactive: set, period: int, volumes: dict):

        arcs, incidence = self.active_network(inactive)
        nodeindex, q, H0, A21, A12, A10 = self.build_TP_matrices(arcs, incidence, period, volumes)
        Q, H = TodiniPilati(arcs, q, H0, A21, A12, A10)

        flow = {a: 0 for a in self.instance.arcs}
        head = {n: 0 for n in self.instance.nodes}

        for j, arc in enumerate(arcs):
            aggreg = self.aggregate.get(arc)
            if aggreg:
                flow[aggreg[0]] = Q[j][0]
                flow[aggreg[1]] = Q[j][0]
            else:
                flow[arc] = Q[j][0]
                if abs(Q[j][0]) < 1e-6:
                    print(f'null flow {Q[j][0]} for active element {arc}')

        for node, (isDemand, i) in nodeindex.items():
            head[node] = H[i][0] if isDemand else H0[i][0]

        return flow, head


    def extended_period_analysis(self, inactive: dict, stopatviolation=True):
        """Run flow analysis progressively on each time period."""
        violations = []
        nperiods = len(inactive)
        volumes = [{} for t in range(nperiods+1)]
        volumes[0] = {i: tank.vinit for i, tank in self.instance.tanks.items()}
        flow = {}
        head = {}

        for t in range(nperiods):

            flow[t], head[t] = self._flow_analysis(inactive[t], t, volumes[t])

            for i, tank in self.instance.tanks.items():
                volumes[t+1][i] = volumes[t][i] + 3.6 * self.instance.tsinhours() \
                    * (sum(flow[t][a] for a in self.instance.inarcs(i))
                       - sum(flow[t][a] for a in self.instance.outarcs(i)))

                if volumes[t+1][i] < tank.vmin - self.feastol or volumes[t+1][i] > tank.vmax + self.feastol:
                    violations.append((t+1, i,
                                       volumes[t+1][i]-tank.vmin if volumes[t+1][i] < tank.vmin - self.feastol
                                       else volumes[t+1][i]-tank.vmax))

                    if stopatviolation:
                        return flow, head, volumes, violations

        for i, tank in self.instance.tanks.items():
            if volumes[nperiods][i] < tank.vinit - self.feastol:
                violations.append((nperiods, i, volumes[nperiods][i]))
                if stopatviolation:
                    return flow, head, volumes, violations

        return flow, head, volumes, violations


def TodiniPilati(arcs, q, H0, A21, A12, A10):
    """Apply the Todini Pilati algorithm of flow analysis, return flow and head."""
    Id = np.identity(len(arcs))
    Q = np.full((len(arcs), 1), 10)
    H = np.zeros((1, len(q)))

    A11 = np.zeros((len(arcs), len(arcs)))
    D = np.zeros((len(arcs), len(arcs)))

    gap = 1
    while gap > 1e-8:
        Qold = np.copy(Q)
        for i, (a, arc) in enumerate(arcs.items()):
            A11[i][i] =    arc[2] * abs(Q[i][0]) + arc[1] + arc[0] / Q[i][0]
            D[i][i]   = (2*arc[2] * abs(Q[i][0]) + arc[1])**(-1)

        H = - np.linalg.inv(A21 @ D @A12) @ (A21 @ D @ (A11 @ Q + A10 @ H0) + q - A21 @ Q)
        Q = (Id - D @ A11) @ Q - D @ (A12 @ H + A10 @ H0)

        #!!! assert Q!=0 and Q[pump]>0
        gap = sum(abs(Q[i][0]-Qold[i][0]) for i, arc in enumerate(arcs)) \
            / sum(abs(Q[i][0]) for i, arc in enumerate(arcs))

    return Q, H
