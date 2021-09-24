#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 11:07:46 2020

@author: Sophie Demassey, Gratien Bonvin
"""

import numpy as np
import instance as inst
from copy import deepcopy

NEWTON_TOL = 1e-8

# TODO merge with instance
class HydraulicNetwork:
    """An alternative view of the water network."""

    def __init__(self, instance: inst.Instance, feastol: float):
        """Create an HydraulicNetwork from an Instance."""
        self.instance = instance
        self.feastol = feastol
        self.arcs, self.incidence = self._build_incidence(instance)
        self.removed_nodes = {}
        assert self._check_network(self.incidence)

    @staticmethod
    def _build_incidence(instance):
        """Get the incidence matrix."""
        arcs = {}
        incidence = {n: set() for n in instance.nodes}

        for (i, j), arc in instance.arcs.items():
            arcs[(i, j)] = [h for h in arc.hloss]
            incidence[i].add((i, j))
            incidence[j].add((i, j))

        return arcs, incidence

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
        self.removed_nodes = {}

        leavestoremove = set()
        for (i, j) in inactive:
            del arcs[(i, j)]
            if self._remove_arc_from_junction(i, (i, j), incidence):
                leavestoremove.add(i)
            if self._remove_arc_from_junction(j, (i, j), incidence):
                leavestoremove.add(j)

        while leavestoremove:
            leaf = leavestoremove.pop()
            assert incidence.get(leaf), f'node {leaf} has already been removed'
            assert len(incidence[leaf]) == 1, f'{leaf} not a leaf: {incidence[leaf]}'
            branch = incidence[leaf].pop()
            del incidence[leaf]
            del arcs[branch]
            altj = branch[1] if (branch[0] == leaf) else branch[0]
            self.removed_nodes[leaf] = (branch, altj)
            if self._remove_arc_from_junction(altj, branch, incidence):
                leavestoremove.add(altj)

        assert arcs, "the active network is empty"
        return arcs, incidence

    def build_matrices(self, arcs, incidence, period, volumes):
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
                demand.append(node.demand(period))
                nodeindex[nodeid] = (True, ndemand)
                ndemand += 1
            else:
                nodehead = node.head(period if isinstance(node, inst._Reservoir) else volumes[nodeid])
                head.append(nodehead)
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
        nodeindex, q, H0, A21, A12, A10 = self.build_matrices(arcs, incidence, period, volumes)
        Q, H = self.todini_pilati(arcs, q, H0, A21, A12, A10)

        flow = {a: 0 for a in self.instance.arcs}
        head = {n: 0 for n in self.instance.nodes}

        for node, (isDemand, i) in nodeindex.items():
            head[node] = H[i][0] if isDemand else H0[i][0]

        for k, ((i, j), arc) in enumerate(arcs.items()):
            flow[(i, j)] = Q[k][0]
            assert self.check_hloss(arc, flow[(i, j)], head[i], head[j]), f'hloss a=({i},{j}) t={period}'
            assert self.check_bounds((i, j), flow[(i, j)]), f'qbnds a=({i},{j}) t={period}'

        # recover the head at removed nodes
        for leaf, (branch, altj) in self.removed_nodes.items():
            assert head[leaf] == 0 and flow[branch] == 0 and (branch not in self.instance.pumps)
            head[leaf] = head[altj]

        return flow, head

    def check_hloss(self, arc, q, hi, hj):
        ok = True
        hlossval = (arc[2]*abs(q) + arc[1])*q + arc[0]
        if abs(hlossval-hi+hj) > self.feastol:
            ok = False
            print(f"q={q}: {hlossval} != {hi} - {hj} = {hi-hj}")
        return ok

    def check_bounds(self, arc, q):
        ok = True
        qmin = self.instance.arcs[arc].qmin
        if q < self.instance.arcs[arc].qmin - self.feastol:
            ok = False
            print(f"violated flow arc bound ! q={q} < qmin={qmin}")
        qmax = self.instance.arcs[arc].qmax
        if q > self.instance.arcs[arc].qmax + self.feastol:
            ok = False
            print(f"violated flow arc bound ! q={q} > qmax={qmax}")
        return ok

    def extended_period_analysis(self, inactive: dict, stopatviolation=True):
        """Run flow analysis progressively on each time period."""
        violations = []
        nperiods = len(inactive)
        volumes = [{} for _ in range(nperiods + 1)]
        volumes[0] = {i: tank.vinit for i, tank in self.instance.tanks.items()}
        flow = {}
        head = {}

        for t in range(nperiods):

            flow[t], head[t] = self._flow_analysis(inactive[t], t, volumes[t])

            for i, tank in self.instance.tanks.items():
                volumes[t + 1][i] = volumes[t][i] + 3.6 * self.instance.tsinhours() \
                                    * (sum(flow[t][a] for a in self.instance.inarcs(i))
                                       - sum(flow[t][a] for a in self.instance.outarcs(i)))

                if volumes[t + 1][i] < tank.vmin - self.feastol or volumes[t + 1][i] > tank.vmax + self.feastol:
                    violations.append((t + 1, i,
                                       volumes[t + 1][i] - tank.vmin if volumes[t + 1][i] < tank.vmin - self.feastol
                                       else volumes[t + 1][i] - tank.vmax))

                    if stopatviolation:
                        return flow, head, volumes, violations

        head[nperiods] = {}
        for i, tank in self.instance.tanks.items():
            if volumes[nperiods][i] < tank.vinit - self.feastol:
                violations.append((nperiods, i, volumes[nperiods][i]))
                if stopatviolation:
                    return flow, head, volumes, violations
            head[nperiods][i] = tank.head(volumes[nperiods][i])

        return flow, head, volumes, violations


    def todini_pilati(self, arcs, q, H0, A21, A12, A10):
        """Apply the Todini Pilati algorithm of flow analysis, return flow and head."""
        Id = np.identity(len(arcs))
        Q = np.full((len(arcs), 1), 10)
        H = np.zeros((1, len(q)))

        A11 = np.zeros((len(arcs), len(arcs)))
        D = np.zeros((len(arcs), len(arcs)))

        gap = 1
        while gap > NEWTON_TOL:
            Qold = np.copy(Q)
            for i, (a, arc) in enumerate(arcs.items()):
                A11[i][i] = arc[2] * abs(Q[i][0]) + arc[1] + arc[0] / Q[i][0]
                D[i][i] = (2 * arc[2] * abs(Q[i][0]) + arc[1]) ** (-1)

            H = - np.linalg.inv(A21 @ D @ A12) @ (A21 @ D @ (A11 @ Q + A10 @ H0) + q - A21 @ Q)
            Q = (Id - D @ A11) @ Q - D @ (A12 @ H + A10 @ H0)

            # !!! assert Q!=0 and Q[pump]>0
            gap = sum(abs(Q[i][0] - Qold[i][0]) for i, arc in enumerate(arcs)) \
                / sum(abs(Q[i][0]) for i, arc in enumerate(arcs))

        return Q, H
