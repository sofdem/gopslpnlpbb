#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 10:48:46 2022

With networkanalysis.py should replace hydraulics.py:
PotentialNetwork represents the network (or a component) directly by its incidence matrix
after reordering and reindexing nodes and arcs (make sure to observe the order !).
Run network analysis (Todini-Pilati) for any fixed values of demand or head on each node
after masking inactive arcs and zero-demand leaves.
(head nodes and non-zero demand nodes should not become disconnected: no verification there)
It is also possible to keep an history of the computations.

@author: Sophie Demassey
"""

import numpy as np


class PotentialNetwork:
    """a weakly connected potential network with controllable arcs and two types of nodes: demand or reservoir."""

    def __init__(self, nd0nodes: int, nvarcs: int, hloss: list, history: bool, a: list, d: list, h: list):
        """Create Potential network."""
        self.nd0nodes = nd0nodes
        # assert self.nd0nodes < len(d), 'a component has no demand'

        dnodes = {nid: idx for idx, nid in enumerate(d)}
        hnodes = {nid: len(d) + idx for idx, nid in enumerate(h)}
        self.nodes = {**dnodes, **hnodes}
        assert len(d) + len(h) == len(self.nodes)
        print(f"nodes = {self.nodes}")

        assert len(hloss) == len(a)
        self.arcs = {aid: idx for idx, aid in enumerate(a)}
        self._arcs = [(self.nodes[aid[0]], self.nodes[aid[1]]) for aid in a]
        print(f"arcs = {self.arcs}")
        self._hloss = np.array(hloss)

        incidence_transpose = PotentialNetwork._build_incidence_transpose(len(self.nodes), self._arcs)
        self._dincidence_transpose = incidence_transpose[:, :len(d)]
        self._hincidence_transpose = incidence_transpose[:, len(d):]
        self._prev_inactivearcs = set()
        self._mask_a = np.ones(len(self._arcs), dtype=bool)
        self._mask_d = np.ones(len(d), dtype=bool) if self.nd0nodes else None
        self._masked_a = self._masked_d = False

        self.history = {} if history else None
        self.hasvarcs = nvarcs > 0

    def removehistory(self):
        self.history = None

    @staticmethod
    def _build_incidence_transpose(nnodes, arcs):
        """Build the incidence matrix."""
        incidence_transpose = np.zeros((len(arcs), nnodes))
        for idx, arc in enumerate(arcs):
            incidence_transpose[idx, arc[0]] = -1
            incidence_transpose[idx, arc[1]] = 1
        return incidence_transpose

    def _mask_inactivearcs(self, inactivearcs: tuple):
        # same configuration as previous execution: do not change masks
        if self._prev_inactivearcs == inactivearcs:
            return
        self._prev_inactivearcs = inactivearcs
        # no arc to disactivate: no mask
        if len(inactivearcs) == 0:
            self._masked_a = self._masked_d = False
            return
        # recompute masks
        self._mask_a.fill(True)
        d0_tocheck = set()
        # mask inactive arcs
        for aid in inactivearcs:
            aidx = self.arcs[aid]
            self._mask_a[aidx] = False
            if self._mask_d is not None:
                for nidx in self._arcs[aidx]:
                    if nidx < self.nd0nodes:
                        d0_tocheck.add(nidx)
        self._masked_a = True
        # recursively mask d0 leaves
        if d0_tocheck:
            self._masked_d = True
            self._mask_d[:self.nd0nodes] = True
            nmasknodes = self._mask_d0leaf(d0_tocheck, 0)
            self._masked_d = nmasknodes > 0
        else:
            self._masked_d = False

    def _mask_d0leaf(self, tocheck, nmasked):
        # no d0 node to check
        if not tocheck:
            return nmasked
        nidx = tocheck.pop()
        # d0 node is already masked
        if not self._mask_d[nidx]:
            return nmasked
        arcs = np.nonzero((self._dincidence_transpose[:, nidx] != 0) & (self._mask_a == True))[0]
        assert len(arcs) > 0
        # d0 node is a leaf: mask node & arc and check adjacent node
        if len(arcs) == 1:
            aidx = arcs[0]
            self._mask_d[nidx] = False
            self._mask_a[aidx] = False
            nmasked += 1
            arc = self._arcs[aidx]
            adjn = arc[1] if arc[0] == nidx else arc[0]
            if adjn < self.nd0nodes and self._mask_d[adjn]:
                tocheck.add(adjn)
        return self._mask_d0leaf(tocheck, nmasked)

    def flow_analysis(self, inactivearcs: tuple, fixeddemand: list, fixedhead: list, feastol: float):
        configid, flow = self.check_history(inactivearcs, fixeddemand, fixedhead)
        if flow:
            # print(f"skip calculation for {configid}: flow = {flow}")
            return flow

        self._mask_inactivearcs(inactivearcs)
        aix = self._mask_a if self._masked_a else None
        dix = self._mask_d if self._masked_d else None

        # subgraph is the entire graph => no mask
        if aix is None:
            assert dix is None
            hloss = self._hloss
            a10 = self._hincidence_transpose
            a12 = self._dincidence_transpose
        # subgraph is empty => flow is 0
        elif not aix.any():
            flow = {aid: 0 for aid, aidx in self.arcs.items()}
            self.record_history_nullflow(configid, flow)
            return flow
        else:
            # get the matrices restricted to the subgraph
            hloss = self._hloss[aix]
            a10 = self._hincidence_transpose[aix]
            a12 = self._dincidence_transpose[aix] if dix is None else self._dincidence_transpose[np.ix_(aix, dix)]

        fd = np.array(fixeddemand)
        q0 = fd if dix is None else fd[dix]
        h0 = np.array(fixedhead)

        q = self.todini_pilati(hloss, q0[:, np.newaxis], h0[:, np.newaxis], a12, a10, feastol)
        assert q.shape == (hloss.shape[0], 1), f"{q.shape} != ({hloss.shape[0]}, 1)"
        if aix is None:
            return {aid: q[aidx, 0] for aid, aidx in self.arcs.items()}
        # @todo use np.maskedarray instead
        offset = 0
        flow = {}
        for aid, aidx in self.arcs.items():
            if self._mask_a[aidx]:
                flow[aid] = q[aidx-offset, 0]
            else:
                flow[aid] = 0
                offset += 1
        assert len(q) + offset == len(self.arcs)

        self.record_history(configid, flow)
        return flow

    @staticmethod
    def todini_pilati(hloss, q0, h0, a12, a10, feastol):
        """Apply the Todini Pilati algorithm of flow analysis, return flow and head."""
        narcs = hloss.shape[0]
        ndnodes = q0.shape[0]
        nhnodes = h0.shape[0]
        assert a12.shape[0] == narcs and a10.shape[0] == narcs
        assert a12.shape[1] == ndnodes and a10.shape[1] == nhnodes
        # print(f"q0 =: {q0}, h0= {h0}, hloss={hloss}")

        a21 = a12.transpose()
        ident = np.identity(narcs)
        q = np.full((narcs, 1), 10)

        gap = 1
        while gap > feastol:
            qold = np.copy(q)
            a11 = np.diag([hloss[i, 2] * abs(q[i, 0]) + hloss[i, 1] + hloss[i, 0] / q[i, 0] for i in range(narcs)])
            d = np.diag([(2 * hloss[i, 2] * abs(q[i, 0]) + hloss[i, 1]) ** (-1) for i in range(narcs)])
            h = - np.linalg.inv(a21 @ d @ a12) @ (a21 @ d @ (a11 @ q + a10 @ h0) + q0 - a21 @ q)
            q = (ident - d @ a11) @ q - d @ (a12 @ h + a10 @ h0)
            gap = np.absolute(q - qold).sum() / np.absolute(q).sum()
        assert PotentialNetwork.check_hloss(q, h, h0, hloss, a10, a12, feastol)
        return q

    @staticmethod
    def check_hloss(q, h, h0, hloss, a10, a12, feastol):
        narcs = hloss.shape[0]
        a11 = np.diag([hloss[i, 2] * abs(q[i, 0]) + hloss[i, 1] + hloss[i, 0] / q[i, 0] for i in range(narcs)])
        hlh = a12 @ h + a10 @ h0
        hlq = a11 @ q
        hlossdiff = np.absolute(hlh + hlq) > feastol
        if hlossdiff.any():
            print(f"hloss diff: {hlh[hlossdiff]} != {hlq[hlossdiff]}")
            return False
        return True

    def check_history(self, inactivearcs: tuple, demand: list, head: list):
        if self.history is None:
            return False, False
        configid = ('N', *inactivearcs)
        flow = self.history.get(configid, False)
        if flow:
            return configid, flow
        configid = (*demand, *head, *inactivearcs)
        return configid, self.history.get(configid, False)

    def record_history(self, configid: tuple, flow: dict):
        if self.history is not None:
            self.history[configid] = flow

    def record_history_nullflow(self, inactivearcs: tuple, flow: dict):
        if self.history is not None:
            configid = ('N', *inactivearcs)
            self.history[configid] = flow
