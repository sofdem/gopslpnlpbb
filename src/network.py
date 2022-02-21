#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 10:48:46 2022

@author: Sophie Demassey
"""

import numpy as np
from copy import deepcopy
from instance import _Junction
from instance import _Node
from instance import _Arc

NEWTON_TOL = 1e-8

class PotentialNetwork:
    """Weakly connected potential network."""

    def __init__(self, feastol: float, arcs: list, nbfixarcs: int, nodes: list, nbdemnodes: int, resistances: list):
        """Create Potential network."""
        self.feastol = feastol

        self.arcs = arcs
        self.invarcs = {val: idx for idx, val in enumerate(self.arcs)}
        self.nbfixarcs = nbfixarcs

        self.nodes = nodes
        self.invnodes = {val: idx for idx, val in enumerate(self.nodes)}
        self.nbdemnodes = nbdemnodes

        self.incidence = self._build_incidence()
        self.maskconfigs = self._build_maskconfigs()

        assert self._check_network(self.incidence)

    def _build_incidence(self):
        """Get the incidence matrix."""
        incidence = np.zeros(len(self.nodes), len(self.arcs))
        for ida, arc in enumerate(self.arcs):
            i = self.invnodes[arc.nodes[0]]
            j = self.invnodes[arc.nodes[1]]
            incidence[i, ida] = -1
            incidence[j, ida] = 1
        return incidence

    # @todo store for each possible command in 2^(len(arcs)-nbfixedarcs) :
    # the subsets of active nodes/rows and active arcs/columns indices in the incidence matrix according to the command
    # the set of arcs removed by filtering leaf junctions with zero demand
    # def _build_maskconfigs(self, command: int)

    def build_matrices(self, arcs, command: int, demand: dict, volumes: list):
        """Build the matrices for the Todini-Pilati algorithm.
        demand: the demand value for all junctiom nodes
        """
        actarcs  = self.maskconfig[command][0]
        actnodes = self.maskconfig[command][1]
        remarcs  = self.maskconfig[command][2]
        incidence = self.incidence[np.ix(actnodes, actarcs)]

        q0 = [demand[self.arcs[actnidx]] for actnidx in actnodes if actnidx < self.nbdemnodes]
        h0 = [head[self.arcs[actnidx]] for actnidx in actnodes if actnidx < self.nbdemnodes]
        nodeindex = {}

            # create q and h0: the column vectors of fixed demand and fixed head nodes
            demand = []
            head = []
            ndemand = 0
            nhead = 0
                if isinstance(node, _Junction):
            for nodeid in incidence:
                node = self.instance.nodes[nodeid]
                    demand.append(node.demand(period))
                    nodeindex[nodeid] = (True, ndemand)
                    ndemand += 1
                else:
                    nodehead = node.head(period if isinstance(node, inst._Reservoir) else volumes[nodeid])
                    head.append(nodehead)
                    nodeindex[nodeid] = (False, nhead)
                    nhead += 1

            q0 = np.array(demand).reshape(ndemand, 1)
            h0 = np.array(head).reshape(nhead, 1)

            # create the incidence matrices over fixed demand and fixed head nodes
            # print('SIZE = ', ndemand, len(arcs), nhead)
            a21 = np.zeros((ndemand, len(arcs)))
            a12 = np.zeros((len(arcs), ndemand))
            a01 = np.zeros((nhead, len(arcs)))
            a10 = np.zeros((len(arcs), nhead))

            for (inout, val) in {(1, 1), (0, -1)}:
                for j, arc in enumerate(arcs):
                    (isDemand, i) = nodeindex.get(arc[inout])
                    if isDemand:
                        a21[i][j] = val
                        a12[j][i] = val
                    else:
                        a01[i][j] = val
                        a10[j][i] = val
            return nodeindex, q0, h0, a21, a12, a10

    def _check_network(self, incidence):
        check = True
        for nodeid, arcs in incidence.items():
            if len(arcs) == 0:
                check = False
                print("Error: isolated node", nodeid)
            node = self.instance.nodes[nodeid]
            if isinstance(node, inst._Junction) and node.dmean == 0 and len(arcs) == 1:
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
            node = self.instance.nodes[j]
            removejunction = isinstance(node, inst._Junction) and node.dmean == 0
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

    def _flow_analysis(self, inactive: set, period: int, volumes: dict, stopatviolation):
        violation = False
        arcs, incidence = self.active_network(inactive)
        nodeindex, q0, h0, a21, a12, a10 = self.build_matrices(arcs, incidence, period, volumes)
        q, h = self.todini_pilati(arcs, q0, h0, a21, a12, a10)

        flow = {a: 0 for a in self.instance.arcs}
        head = {n: 0 for n in self.instance.nodes}

        for node, (isDemand, i) in nodeindex.items():
            head[node] = h[i][0] if isDemand else h0[i][0]

        for k, ((i, j), arc) in enumerate(arcs.items()):
            flow[(i, j)] = q[k][0]
            errormsg = self.check_bounds((i, j), flow[(i, j)])
            if errormsg:
                if stopatviolation:
                    return flow, head, errormsg
                violation = True
            assert self.check_hloss(arc, flow[(i, j)], head[i], head[j]), f'hloss a=({i},{j}) t={period}'
            assert self.check_nonnullflow((i, j), flow[(i, j)]),  f'nullflow a=({i},{j}) t={period}'

        # recover the head at removed nodes
        for leaf, (branch, altj) in self.removed_nodes.items():
            assert head[leaf] == 0 and flow[branch] == 0 and (branch not in self.instance.pumps)
            head[leaf] = head[altj]

        return flow, head, violation

    def check_hloss(self, arc, q, hi, hj):
        hlossval = (arc[2]*abs(q) + arc[1])*q + arc[0]
        if abs(hlossval-hi+hj) > self.feastol:
            print(f"hloss q={q}: {hlossval} != {hi} - {hj} = {hi-hj}")
            return False
        return True

    def check_bounds(self, arc, q):
        if q < self.instance.arcs[arc].qmin - self.feastol:
           return f"lbound q={q} < qmin={self.instance.arcs[arc].qmin}"
        if q > self.instance.arcs[arc].qmax + self.feastol:
            return f"ubound q={q} > qmax={self.instance.arcs[arc].qmax}"
        return

    def check_nonnullflow(self, arc, q):
        if self.instance.arcs[arc].nonnull_flow_when_on() and -self.feastol < q < self.feastol:
            print(f"null flow q={q} on active pump")
            return False
        return True

    def extended_period_analysis(self, inactive: dict, stopatviolation=True):
        """Run flow analysis progressively on each time period."""
        nbviolations = 0
        nperiods = len(inactive)
        volumes = [{} for _ in range(nperiods + 1)]
        volumes[0] = {i: tank.vinit for i, tank in self.instance.tanks.items()}
        flow = {}
        head = {}

        for t in range(nperiods):
            flow[t], head[t], errormsg = self._flow_analysis(inactive[t], t, volumes[t], stopatviolation)
            if errormsg:
                print(f'violation at {t + 1}: {errormsg}')
                if stopatviolation:
                    return flow, head, volumes, t + 1
                nbviolations += 1

            for i, tank in self.instance.tanks.items():
                volumes[t + 1][i] = volumes[t][i] + self.instance.flowtovolume() \
                                    * (sum(flow[t][a] for a in self.instance.inarcs(i))
                                       - sum(flow[t][a] for a in self.instance.outarcs(i)))
                if volumes[t + 1][i] < tank.vmin - self.feastol:
                    print(f'violation at {t + 1}: capacity tk={i}: {volumes[t + 1][i] - tank.vmin:.2f}')
                    nbviolations += 1
                    if stopatviolation:
                        return flow, head, volumes, t+1

                elif volumes[t + 1][i] > tank.vmax + self.feastol:
                    print(f'violation at {t + 1}: capacity tk={i}: {volumes[t + 1][i] - tank.vmax:.2f}')
                    nbviolations += 1
                    if stopatviolation:
                        return flow, head, volumes, t+1

        head[nperiods] = {}
        for i, tank in self.instance.tanks.items():
            if volumes[nperiods][i] < tank.vinit - self.feastol:
                print(f'violation at {nperiods}: capacity tk={i}: {volumes[nperiods][i] - tank.vinit:.2f}')
                nbviolations += 1
                if stopatviolation:
                    return flow, head, volumes, nperiods
            head[nperiods][i] = tank.head(volumes[nperiods][i])

        return flow, head, volumes, nbviolations

    @staticmethod
    def todini_pilati(arcs, q0, h0, a21, a12, a10):
        """Apply the Todini Pilati algorithm of flow analysis, return flow and head."""
        ident = np.identity(len(arcs))
        q = np.full((len(arcs), 1), 10)
        h = np.zeros((1, len(q0)))

        a11 = np.zeros((len(arcs), len(arcs)))
        d = np.zeros((len(arcs), len(arcs)))

        gap = 1
        while gap > NEWTON_TOL:
            qold = np.copy(q)
            for i, (a, arc) in enumerate(arcs.items()):
                a11[i][i] = arc[2] * abs(q[i][0]) + arc[1] + arc[0] / q[i][0]
                d[i][i] = (2 * arc[2] * abs(q[i][0]) + arc[1]) ** (-1)

            h = - np.linalg.inv(a21 @ d @ a12) @ (a21 @ d @ (a11 @ q + a10 @ h0) + q0 - a21 @ q)
            q = (ident - d @ a11) @ q - d @ (a12 @ h + a10 @ h0)

            # !!! assert q!=0 and q[pump]>0
            gap = sum(abs(q[i][0] - qold[i][0]) for i, arc in enumerate(arcs)) \
                / sum(abs(q[i][0]) for i, arc in enumerate(arcs))

        return q, h
