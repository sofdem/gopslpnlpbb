#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 11:07:46 2020

@author: Sophie Demassey, Gratien Bonvin
"""

import csv
import json
import math
from typing import Tuple, Dict, List
import sys
from functions import FuncPol
import decimal

import numpy as np
import datetime as dt
from pathlib import Path

TARIFF_COLNAME: str = 'elix'
TRUNCATION: int = 12
FEASTOL: float = 10 ** (3-TRUNCATION)


# !todo manage rounding values and bounds correctly and uniformly throughout the code
#def myround(value: float, decimals: int = TRUNCATION) -> float:
#    return value
#    return round(value, decimals)
#    with decimal.localcontext() as ctx:
#        d = decimal.Decimal(value)
#        ctx.rounding = decimal.ROUND_DOWN
#        return float(round(d, decimals))


def roundlb(value: float) -> float:
    d: int = 10 ** TRUNCATION
    return math.floor(value * d) / d


def roundub(value: float) -> float:
    d: int = 10 ** TRUNCATION
    return math.ceil(value * d) / d


def mylowerthan(v1: float, v2: float, feastol: float = FEASTOL) -> bool:
    return v1 <= v2 + feastol


def assertmylowerthan(v1: float, v2: float, feastol: float = FEASTOL):
    assert mylowerthan(v1, v2, feastol), f"{v1} > {v2}"


def myequalto(v1: float, v2: float, feastol: float = FEASTOL) -> bool:
    return mylowerthan(v1, v2, feastol) and mylowerthan(v2, v1, feastol)


def assertmyequalto(v1: float, v2: float, feastol: float = FEASTOL):
    assert mylowerthan(v1, v2, feastol) and mylowerthan(v2, v1, feastol), f"{v1} != {v2}"


class _Node:
    """Generic network node. coordinates X, Y, Z (in m)."""

    def __init__(self, id_, x, y, z):
        self.id = id_
        self.coord = {'x': x, 'y': y, 'z': z}

    def altitude(self):
        return self.coord['z']


class _Tank(_Node):
    """Network node of type cylindrical water tank.

    vmin    : minimum volume value (in m3)
    vmax    : maximum volume value (in m3)
    vinit   : initial volume       (in m3)
    surface : surface              (in m2)
    """

    def __init__(self, id_, x, y, z, vmin, vmax, vinit, surface):
        _Node.__init__(self, id_, x, y, z)
        self.vmin = vmin
        self.vmax = vmax
        self.vinit = vinit
        self.surface = surface
        self._hbounds = []
        self._qinbounds = []

    def head(self, volume):
        return self.altitude() + volume / self.surface

    #! todo initialize hmin/hmax(t) even if hbounds are not provided (pb at t=T)
    def hmin(self, t: int) -> float:
        return self._hbounds[t][0] if self._hbounds else self.head(self.vinit) if t == 0 else self.head(self.vmin)

    def hmax(self, t: int) -> float:
        return self._hbounds[t][1] if self._hbounds else self.head(self.vinit) if t == 0 else self.head(self.vmax)

    def sethbounds(self, hbounds: list):
        for ht in hbounds:
            assertmylowerthan(self.head(self.vmin), ht[0])
            assertmylowerthan(ht[1], self.head(self.vmax))
        assertmyequalto(self.head(self.vinit), hbounds[0][0])
        assertmyequalto(self.head(self.vinit), hbounds[0][1])
        assertmyequalto(self.head(self.vinit), hbounds[-1][0])
        assertmyequalto(self.head(self.vmax), hbounds[-1][1])
        self._hbounds = [(roundlb(ht[0]), roundub(ht[1])) for ht in hbounds]

    def qinmin(self, t: int) -> float:
        return self._qinbounds[t][0] if self._qinbounds else -1e8

    def qinmax(self, t: int) -> float:
        return self._qinbounds[t][1] if self._qinbounds else 1e8

    #! todo initialize to vmax-vmin / duration when no dynamic bounds are provided
    def setqinbounds(self, qinbounds: list):
        self._qinbounds = [(roundlb(qt[0]), roundub(qt[1])) for qt in qinbounds]

    def checkcapacity(self, vol: float, vminisvinit: bool, tol: float):
        hgap = (vol - (self.vinit if vminisvinit else self.vmin)) / self.surface
        if hgap < -tol:
            return hgap
        hgap = (vol - self.vmax) / self.surface
        if hgap > tol:
            return hgap
        return 0

class _Junction(_Node):
    """Network node of type junction.

    dmean    : mean demand (in L/s)
    dprofile : demand pattern profile id
    """

    def __init__(self, id_, x, y, z, dmean, profileid):
        _Node.__init__(self, id_, x, y, z)
        self.dmean = dmean
        self.profileid = profileid
        self.dprofile = None
        self.demands = None

    def setprofile(self, profile):
        self.dprofile = profile[self.profileid]
        self.demands = [self.dmean * p for p in self.dprofile]

    def demand(self, t):
        return self.demands[t]


class _Reservoir(_Node):
    """Network node of type infinite reservoirs (sources).

    hprofile : head profile id
    drawmax  : maximal daily withdrawal (in m3)
    drawcost : withdrawal cost  (in euro/m3)
    """

    def __init__(self, id_, x, y, z, profileid, drawmax, drawcost):
        _Node.__init__(self, id_, x, y, z)
        assert drawmax == 'NO', "max source withdrawal not completetly supported"
        self.drawmax = None if (drawmax == 'NO') else drawmax
        self.drawcost = drawcost
        if drawcost != 0:
            raise "drawcost not yet supported"
        self.profileid = profileid
        self.hprofile = None
        self.heads = None

    def setprofile(self, profile):
        self.hprofile = profile[self.profileid]
        self.heads = [self.altitude() * p for p in self.hprofile]

    def head(self, t):
        return self.heads[t]


class _Arc:
    """Generic network arc.

    id      : identifier
    nodes   : (i,j) with i the start node id, and j the end node id
    _qmin    : minimum flow value <= q(i,j) (in L/s) when active
    _qmax    : maximum flow value >= q(i,j) (in L/s) when active
    _qbounds : specific bounds on flow when active indexed on times (optional)
    hloss   : head loss polynomial function: h(i) - h(j) = sum_n hloss[n] q(i,j)^n (L/s -> m)
    control : is the arc controllable or not ? (valved pipe or pump)
    """

    def __init__(self, id_, nodes: Tuple[str, str], qmin: float, qmax: float, hloss: tuple):
        self.id = id_
        self.nodes = nodes
        self._qmin = roundlb(qmin)
        self._qmax = roundub(qmax)
        self._qbounds = []
        self.hloss = FuncPol(hloss) if hloss else None
        self.control = False

    def qmin(self, t: int = -1) -> float:
        """ return the minimum flow value, at time t if specified. """
        return self._qmin if t < 0 or not self._qbounds else self._qbounds[t][0]

    def qmax(self, t: int = -1) -> float:
        """ return the maximum flow value, at time t if specified. """
        return self._qmax if t < 0 or not self._qbounds else self._qbounds[t][1]

    def setqbounds(self, qbounds: list, qminmax: Tuple[float, float]):
        """ import specific time indexed active bounds when arc is active;
        if specified, overwrite the original general bounds with qminmax. """
        if qminmax:
            self._qmin = roundlb(qminmax[0])
            self._qmax = roundub(qminmax[1])
        self._qbounds = [(roundlb(qt[0]), roundub(qt[1])) for qt in qbounds]

    def check_qbounds(self, q: float, tol: float):
        gap = q - self.qmin()
        if gap < -tol:
            return gap
        gap = q - self.qmax()
        if gap > tol:
            return gap
        return 0

    def nonnull_flow_when_on(self):
        return False

    def __str__(self):
        return f'{self.id}: {self.nodes} [{self._qmin}, {self._qmax}] {self.hloss}'


class _ControllableArc(_Arc):
    """Controllable network arc: valved pipe or pump
    _dhmin   : minimum head loss value when arc is off (valve open or pump off)
    _dhmax   : maximum head loss value when arc is off (valve open or pump off)
    _dhbounds : specific bounds on head loss indexed on times (optional)
    _fixed[t] : know status (inactive: 0, active: 1, unknown: -1) at time t (optional)
    """

    def __init__(self, id_, nodes, qmin, qmax, hloss, dhmin, dhmax):
        _Arc.__init__(self, id_, nodes, qmin, qmax, hloss)
        self._dhmin = roundlb(dhmin)
        self._dhmax = roundub(dhmax)
        self._dhbounds = []
        self._fixed = []
        self.control = True

    def qminifon(self, t: int = -1) -> float:
        """ return the best known minimum flow value when active, at time t if specified. """
        return self._qmin if t < 0 or not self._qbounds else self._qbounds[t][0]

    def qmaxifon(self, t: int = -1) -> float:
        """ return the best known maximum flow value when active, at time t if specified. """
        return self._qmax if t < 0 or not self._qbounds else self._qbounds[t][1]

    def qmin(self, t: int = -1) -> float:
        """ return the minimum flow value, at time t if specified. """
        return self.qminifon(t) if self.isfixedon(t) else min(0.0, self.qminifon(t))

    def qmax(self, t: int = -1) -> float:
        """ return the  maximum flow value, at time t if specified. """
        return self.qmaxifon(t) if self.isfixedon(t) else max(0.0, self.qmaxifon(t))

    def dhminifoff(self, t: int = -1) -> float:
        """ return the best known minimum head loss value, at time t if specified. """
        return self._dhmin if t < 0 or not self._dhbounds else self._dhbounds[t][0]

    def dhmaxifoff(self, t: int = -1) -> float:
        """ return the best known maximum head loss value, at time t if specified. """
        return self._dhmax if t < 0 or not self._dhbounds else self._dhbounds[t][1]

    def setdhbounds(self, dhbounds: list, dhminmax: Tuple[float, float]):
        """ import specific time indexed head loss bounds;
        if specified, overwrite the original general bounds with dhminmax. """
        if dhminmax:
            self._dhmin = roundlb(dhminmax[0])
            self._dhmax = roundub(dhminmax[1])
        self._dhbounds = [(roundlb(dht[0]), roundub(dht[1])) for dht in dhbounds]

    def setfixed(self, fixedtimes: list, nperiods: int):
        """ import known fixed status: build the time-indexed table with 0 (inactive), 1 (active) or -1. """
        assert len(fixedtimes) == 2
        if fixedtimes[0] or fixedtimes[1]:
            self._fixed = [-1 for _ in range(nperiods)]
            for t in fixedtimes[0]:
                self._fixed[t] = 0
            for t in fixedtimes[1]:
                assert self._fixed[t] < 0
                self._fixed[t] = 1

    def isfixedon(self, t: int) -> bool:
        """ return True if arc is known to be active at time t. """
        return self._fixed and self._fixed[t] == 1

    def isfixedoff(self, t: int) -> bool:
        """ return True if arc is known to be inactive at time t. """
        return self._fixed and self._fixed[t] == 0

    def __str__(self):
        return f'{self.id} [{self._qmin}, {self._qmax}] {self.hloss} [{self._dhmin}, {self._dhmax}]'


class _ValvedPipe(_ControllableArc):
    """Network arc of type pipe + valve.

    valve type     : 'GV' or 'PRV' or 'CV'
    """
    def __init__(self, id_, nodes, type_, dhmin, dhmax, qmin, qmax):
        _ControllableArc.__init__(self, id_, nodes, qmin, qmax, [], dhmin, dhmax)
        self.type = type_
        if type_ != 'GV':
            raise NotImplementedError('pressure reducing valves are not yet supported')
        self.valve = nodes
        self.pipe = None

    def __str__(self):
        return f'V{self.id} {self.type} [{self._qmin}, {self._qmax}] {self.hloss} [{self._dhmin}, {self._dhmax}]'

    def merge_pipe(self, pipe):
        # print(f'merge valve {self.nodes} + pipe {pipe.nodes}')
        self.pipe = pipe.nodes
        assert self.nodes[0] == pipe.nodes[1], f'valve {self.nodes} + pipe {pipe.nodes}'
        auxnode = self.nodes[0]
        self.nodes = (pipe.nodes[0], self.nodes[1])
        self.hloss = pipe.hloss
        # print(f'valve bounds = [{self._qmin}, {self._qmax}]')
        # print(f'pipe bounds = [{pipe.qmin()}, {pipe.qmax()}]')
        self._qmin = max(self._qmin, pipe.qmin())
        self._qmax = min(self._qmax, pipe.qmax())

        return auxnode


class _Pump(_ControllableArc):
    """Network arc of type pump.

    type    : 'FSD' or 'VSD'
    power   : power polynomial function: p = sum_n power[n]q(i,j)^n (L/s -> W)
    """

    def __init__(self, id_, nodes, hloss, power, qmin, qmax, dhmin, dhmax, type_):
        _ControllableArc.__init__(self, id_, nodes, qmin, qmax, hloss, dhmin, dhmax)
        self.type = type_
        self.power = power
        if type_ == 'VSD':
            raise NotImplementedError('variable speed pumps are not yet supported')

    def powerval(self, q):
        assert len(self.power) == 2
        return self.power[0] + self.power[1] * q

    def nonnull_flow_when_on(self):
        return True

    def __str__(self):
        return f'K{self.id} [{self._qmin}, {self._qmax}] {self.hloss} [{self._dhmin}, {self._dhmax}] ' \
               f'{self.power} {self.type} '


def isjunction(node):
    return isinstance(node, _Junction)


def isreservoir(node):
    return isinstance(node, _Reservoir)


class Instance:
    """Instance of the Pump Scheduling Problem."""

    DATADIR = Path("../data/")
    BNDSDIR = Path("../bounds/")

    def __init__(self, name, profilename, starttime, endtime, aggregatesteps):
        self.feastol = FEASTOL
        self.name = name
        self.tanks = self._parse_tanks('Reservoir.csv', self._parse_initvolumes('History_V_0.csv'))
        self.junctions = self._parse_junctions('Junction.csv')
        self.reservoirs = self._parse_reservoirs('Source.csv')
        self.pumps = self._parse_pumps('Pump.csv')
        self.fpipes = self._parse_pipes('Pipe.csv')
        self._valves = self._parse_valves('Valve_Set.csv')
        # @todo merge pipes and valves directly in the instance files
        self.vpipes = self._merge_pipes_and_valves()

        self.farcs = self.fpipes
        self.varcs = {**self.pumps, **self.vpipes}
        self.arcs = {**self.varcs, **self.farcs}
        self.nodes = {**self.junctions, **self.tanks, **self.reservoirs}
        self.incidence = self._getincidence()

        periods, profiles = self._parse_profiles(f'{profilename}.csv', starttime, endtime, aggregatesteps)
        self.periods = periods
        self.profiles = profiles
        self.tariff = profiles[TARIFF_COLNAME]
        self.tsduration = self._get_timestepduration(periods)

        self.dependencies = self._dependencies()
        self.symmetries = self._pump_symmetric()

        for r in self.reservoirs.values():
            r.setprofile(profiles)
        for j in self.junctions.values():
            j.setprofile(profiles)

    def nperiods(self):
        return len(self.periods) - 1

    def horizon(self) -> range:
        return range(self.nperiods())

    def tsinhours(self):
        return self.tsduration.total_seconds() / 3600  # in hour

    def eleccost(self, t):
        return self.tsinhours() * self.tariff[t] / 1000  # in euro/W

    def inarcs(self, node):
        return self.incidence[node, 'in']

    def outarcs(self, node):
        return self.incidence[node, 'out']

    # def inflowmin(self, node):
    #    return (sum(self.arcs[a].abs_qmin() for a in self.inarcs(node))
    #            - sum(self.arcs[a].abs_qmax() for a in self.outarcs(node)))

    # def inflowmax(self, node):
    #    return (sum(self.arcs[a].abs_qmax() for a in self.inarcs(node))
    #            - sum(self.arcs[a].abs_qmin() for a in self.outarcs(node)))

    def inflow(self, node, flow):
        return self.flowtovolume() * (sum(flow[a] for a in self.inarcs(node))
                                      - sum(flow[a] for a in self.outarcs(node)))

    def flowtoheight(self, tank):
        return self.tsduration.total_seconds() / tank.surface / 1000  # in m / (L / s)

    def flowtovolume(self):
        return self.tsduration.total_seconds() / 1000  # in m3` / (L / s)

    #  PARSERS

    def _parsecsv(self, filename):
        csvfile = open(Path(self.DATADIR, self.name, filename))
        rows = csv.reader(csvfile, delimiter=';')
        data = [[x.strip() for x in row] for row in rows]
        return data

    def _parse_initvolumes(self, filename):
        data = self._parsecsv(filename)
        return {A[0]: float(A[1]) for A in data[1:]}

    def _parse_pumps(self, filename):
        data = self._parsecsv(filename)
        return dict({(A[1], A[2]): _Pump(A[0], (A[1], A[2]),
                                         (-float(A[5]), -float(A[4]), -float(A[3])), (float(A[7]), float(A[6])),
                                         float(A[8]), float(A[9]), -float(A[10]), -float(A[11]), A[12])
                     for A in data[1:]})

    def _parse_valves(self, filename):
        data = self._parsecsv(filename)
        return dict({(A[1], A[2]): _ValvedPipe(A[0], (A[1], A[2]), A[3], float(A[4]), float(A[5]),
                                               float(A[6]), float(A[7])) for A in data[1:]})

    def _parse_pipes(self, filename):
        data = self._parsecsv(filename)
        return dict({(A[1], A[2]): _Arc(A[0], (A[1], A[2]), float(A[5]), float(A[6]),
                                        (0, float(A[4]), float(A[3]))) for A in data[1:]})

    def _parse_junctions(self, filename):
        data = self._parsecsv(filename)
        return dict({A[0]: _Junction(A[0], float(A[1]), float(A[2]), float(A[3]), float(A[4]), A[5]) for A in data[1:]})

    def _parse_reservoirs(self, filename):
        data = self._parsecsv(filename)
        return dict({A[0]: _Reservoir(A[0], float(A[1]), float(A[2]), float(A[3]),
                                      A[4], A[5], float(A[6])) for A in data[1:]})

    def _parse_tanks(self, filename, initvolumes):
        data = self._parsecsv(filename)
        return dict({A[0]: _Tank(A[0], float(A[1]), float(A[2]), float(A[3]), float(A[4]), float(A[5]),
                                 initvolumes[A[0]], float(A[6])) for A in data[1:]})

    def _parse_profiles(self, filename, starttime, endtime, aggsteps):
        data = self._parsecsv(filename)
        i = 1
        while i < len(data) and data[i][0] != starttime:
            # print(f'{data[i][0]} == {starttime}')
            i += 1
        assert i < len(data), f'starting time {starttime} not found in {filename}'

        assert data[0][1] == TARIFF_COLNAME, f'2nd column of {filename} is electricity tariff'
        profilename = data[0][1:]
        profiles = {n: [] for n in profilename}
        periods = []
        while i < len(data) and data[i][0] != endtime:
            for j, n in enumerate(profilename):
                profiles[n].append(float(data[i][j + 1]))
            periods.append(dt.datetime.strptime(data[i][0], '%d/%m/%Y %H:%M'))
            i += aggsteps
        assert i < len(data), f'end time {endtime} not found in {filename}'
        periods.append(dt.datetime.strptime(endtime, '%d/%m/%Y %H:%M'))
        return periods, profiles

    def _parse_profiles_aggregate(self, filename, starttime, endtime, aggsteps):
        data = self._parsecsv(filename)
        i = 1
        while i < len(data) and data[i][0] != starttime:
            i += 1
        assert i < len(data), f'starting time {starttime} not found in {filename}'

        assert data[0][1] == TARIFF_COLNAME, f'2nd column of {filename} is electricity tariff'
        profilename = data[0][1:]
        profiles = {n: [] for n in profilename}
        periods = []
        sumagg = [0 for _ in profilename]
        cntagg = 0
        while i < len(data) and data[i][0] != endtime:
            i += 1
            if cntagg == aggsteps:
                cntagg = 0
                for j, s in enumerate(sumagg):
                    profiles[profilename[j]].append(s / aggsteps)
                    sumagg[j] = 0
                    periods.append(dt.datetime.strptime(data[i][0], '%d/%m/%Y %H:%M'))
            cntagg += 1
            for j, s in enumerate(sumagg):
                sumagg[j] += float(data[i][j + 1])
        assert i < len(data), f'{filename}: not found end {endtime}'
        return periods, profiles

    @staticmethod
    def _get_timestepduration(periods):
        duration = periods[1] - periods[0]
        for i in range(len(periods) - 1):
            assert duration == periods[i + 1] - periods[i]
        return duration

    def _merge_pipes_and_valves(self):
        vpipes = {}
        for (iv, j), valve in self._valves.items():
            inpipe = [(i, jv) for (i, jv) in self.fpipes if jv == iv]
            assert len(inpipe) == 1, f'valve {(iv,j)} is not attached to exactly one pipe: {inpipe}'
            i = inpipe[0][0]
            pipe = self.fpipes.pop((i, iv))
            auxnode = valve.merge_pipe(pipe)
            print(f'valved pipe {(i, j)} = {valve.pipe} + {valve.valve}')
            assert self.junctions[auxnode].dmean == 0
            self.junctions.pop(auxnode)
            vpipes[(i, j)] = valve
        return vpipes

    def _getincidence(self):
        incidence = {}
        for node in self.nodes:
            incidence[node, 'in'] = set()
            incidence[node, 'out'] = set()
            for arc in self.arcs:
                if arc[1] == node:
                    incidence[node, 'in'].add(arc)
                elif arc[0] == node:
                    incidence[node, 'out'].add(arc)
        return incidence

    def pumps_without_sym(self):
        """Aggregate symmetric pumps as a fictional 'sym' pump."""
        uniquepumps = set(self.pumps.keys())
        symgroup = self.symmetries
        if symgroup:
            uniquepumps -= set(symgroup)
            uniquepumps.add('sym')
        return sorted(uniquepumps, key=str)

    def _pump_symmetric(self):
        """Return a list of symmetric pumps."""
        if self.name == 'Simple_Network':
            return [('R1', 'J2'), ('R2', 'J2'), ('R3', 'J2')]
        elif self.name == 'Anytown':
            return [('R1', 'J20'), ('R2', 'J20'), ('R3', 'J20')]
        elif self.name == 'Richmond':
            return [('196', '768'), ('209', '766')]
        elif self.name == 'SAUR':
            return [('Arguenon_IN_1', 'Arguenon_OUT'), ('Arguenon_IN_2', 'Arguenon_OUT'),
                    ('Arguenon_IN_3', 'Arguenon_OUT'), ('Arguenon_IN_4', 'Arguenon_OUT')]
        return []

    def _dependencies(self):
        """Return 4 types of control dependencies as a dict of lists of dependent pumps/valves."""
        dep = {'p1 => p0': set(), 'p0 xor p1': set(),
               'p0 = p1 xor p2': set(), 'p1 => not p0': set()}

        if self.name == 'Richmond':
            # dep['p1 => p0'].add((('196', '768'), ('209', '766'))) already in symmetry
            dep['p1 => p0'].add((('196', '768'), ('175', '186')))
            dep['p1 => p0'].add((('312', 'TD'), ('264', '112')))
            dep['p1 => p0'].add((('264', '112'), ('312', 'TD')))

            dep['p0 xor p1'].add((('201', '770'), ('196', '768')))
            dep['p0 xor p1'].add((('321', '312'), ('264', '112')))

            dep['p0 = p1 xor p2'].add((('196', '768'), ('164', '197'), ('175', '186')))
        else:
            dep = None

        return dep

    def tostr_basic(self):
        return f'{self.name} {self.periods[0]} {self.horizon()}'

    def tostr_network(self):
        return f'{len(self.pumps)} pumps, {len(self.vpipes)} valved pipes, {len(self.fpipes)} fixed pipes, ' \
               f'{len(self.tanks)} tanks'

    def print_all(self):
        print(f'{self.tostr_basic()} {self.tostr_network()}')
        print(f'{len(self.arcs)} arcs:')
        for i, a in self.arcs.items():
            print(i)
            print(str(a))

    # def transcript_bounds(self, csvfile):
    #    """Parse bounds in the hdf file."""
    #     file = Path(Instance.BNDSDIR, self.name)
    #     pd.read_hdf(file.with_suffix('.hdf')).to_csv(csvfile)

    def parsesolutionplan(self, filename):
        csvfile = open(filename)
        rows = csv.reader(csvfile, delimiter=',')
        data = [[x.strip() for x in row] for row in rows]
        csvfile.close()
        assert float(data[0][1]) == self.nperiods(), f"different horizons in {data[0]} and {self.tostr_basic()}"
        inactive = {t: set((A[0], A[1]) for A in data[1:] if A[t + 2] == '0') for t in self.horizon()}
        return inactive

    def parse_bounds_obbt(self, obbtlevel: str, obbtoverwrite: bool = True):
        """Parse bounds from the json file and assign them to the network elements.
        if 'obbtoverwrite', then restrict the original bounds to those, otherwise do the reverse."""

        bndfilename = f"{obbtlevel}_{self.nperiods()}.json"
        print(f"parse the new OBBT bound file {bndfilename}: overwrite original bounds ? {obbtoverwrite}")
        try:
            bndfile = open(Path(Instance.BNDSDIR, self.name, bndfilename), "r")
        except FileNotFoundError:
            print(f"No OBBT file {bndfilename} found", file=sys.stderr)
            return

        bnds: Dict[str, Dict[str, List[List[float]]]] = json.load(bndfile)
        bndfile.close()

        for a, arc in self.arcs.items():
            aid = arc.id

            assert len(bnds[aid]["q"]) == self.nperiods() and len(bnds[aid]["q"][-1]) == 2
            qbnds = (min([b[0] for b in bnds[aid]["q"]]), max([b[1] for b in bnds[aid]["q"]]))
            assert qbnds[0] <= qbnds[1], f"arc {a}: qmin {qbnds[0]} > qmax {qbnds[1]}"
            if not obbtoverwrite and (qbnds[0] < arc.qmin() or qbnds[1] > arc.qmax()):
                for t in self.horizon():
                    if not(arc.control and t in bnds[aid]["f"][0]):
                        if bnds[aid]["q"][t][0] < arc.qmin():
                            print(f"adjust qmin({aid}, {t}): {bnds[aid]['q'][t][0]} -> {arc.qmin()} ", file=sys.stderr)
                            bnds[aid]['q'][t][0] = arc.qmin()
                        if bnds[aid]["q"][t][1] > arc.qmax():
                            print(f"adjust qmax({aid}, {t}): {bnds[aid]['q'][t][1]} -> {arc.qmax()} ", file=sys.stderr)
                            bnds[aid]["q"][t][1] = arc.qmax()
                qbnds = None
            arc.setqbounds(bnds[aid]["q"], qbnds)

            if arc.control:
                arc.setfixed(bnds[aid]["f"], self.nperiods())

                assert len(bnds[aid]["dh"]) == self.nperiods() and len(bnds[aid]["dh"][0]) == 2
                dhbnds = (min([b[0] for b in bnds[aid]["dh"]]), max([b[1] for b in bnds[aid]["dh"]]))
                if not obbtoverwrite and (dhbnds[0] < arc.dhminifoff() or dhbnds[1] > arc.dhmaxifoff()):
                    for t in self.horizon():
                        if not t in bnds[aid]["f"][1]:
                            if bnds[aid]["dh"][t][0] < arc.dhminifoff():
                                print(f"adjust dhmin({aid}, {t}): {bnds[aid]['dh'][t][0]} -> {arc.dhminifoff()} ",
                                      file=sys.stderr)
                                bnds[aid]['dh'][t][0] = arc.dhminifoff()
                            if bnds[aid]["dh"][t][1] > arc.dhmaxifoff():
                                print(f"adjust dhmax({aid}, {t}): {bnds[aid]['dh'][t][1]} -> {arc.dhmaxifoff()} ",
                                      file=sys.stderr)
                                bnds[aid]["dh"][t][1] = arc.dhmaxifoff()
                    dhbnds = None
                arc.setdhbounds(bnds[aid]["dh"], dhbnds)

        for j, tk in self.tanks.items():
            assert len(bnds[j]["h"]) == self.nperiods() and len(bnds[j]["h"][0]) == 2
            bnds[j]["h"].append([tk.head(tk.vinit), tk.head(tk.vmax)])
            tk.sethbounds(bnds[j]["h"])

            qinbnds = [(max(bnds[j]["qin"][t][0],
                            sum(self.arcs[a].qmin(t) for a in self.inarcs(j)) - sum(self.arcs[a].qmax(t) for a in self.outarcs(j)),
                            (tk.hmin(t+1) - tk.hmax(t)) / self.flowtoheight(tk)),
                        min(bnds[j]["qin"][t][1],
                            sum(self.arcs[a].qmax(t) for a in self.inarcs(j)) - sum(self.arcs[a].qmin(t) for a in self.outarcs(j)),
                            (tk.hmax(t+1) - tk.hmin(t)) / self.flowtoheight(tk)))
                       for t in self.horizon()]

            tk.setqinbounds(qinbnds)

    # @todo directly generate this json file
    def format_bounds_obbt(self, obbtlevel: str = "C1"):
        """Reformat Amir's OBBT bound files to one json file."""
        print(f"{Instance.BNDSDIR}/{self.name}/{obbtlevel}_{self.nperiods()}")
        rep = Path(Instance.BNDSDIR, self.name, f"{obbtlevel}_{self.nperiods()}")
        bndfile = Path(Instance.BNDSDIR, self.name, f"{obbtlevel}_{self.nperiods()}.json")

        zfile = "Bound_flow_arcs.npy"  # flow/active arc + inflow/tank
        cfile = "Bound_flow_tanks.npy"  # head loss/inactive arc
        dfile = "Bound_h_tanks.npy"  # head/tank
        p1file = "Probed1.npy"  # inactive arcs
        p0file = "Probed0.npy"  # active arcs

        qbounds: Dict = np.load(Path(rep, zfile).as_posix(), allow_pickle=True).tolist()
        hbounds: Dict = np.load(Path(rep, dfile).as_posix(), allow_pickle=True).tolist()
        dhbounds: Dict = np.load(Path(rep, cfile).as_posix(), allow_pickle=True).tolist()
        p1arcs: Dict = np.load(Path(rep, p1file).as_posix(), allow_pickle=True).tolist()
        p0arcs: Dict = np.load(Path(rep, p0file).as_posix(), allow_pickle=True).tolist()
        print(qbounds)

        bounds = {}
        for a, arc in self.arcs.items():
            aid = arc.id
            bounds[aid] = {"q": [[qb for qb in qbounds[(a, t)]] for t in self.horizon()]}
            if arc.control:
                bounds[aid]["dh"] = [[dhb for dhb in dhbounds[(a, t)]] for t in self.horizon()]
                bounds[aid]["f"] = [[t for t in self.horizon() if p0arcs.get((a, t))],
                                    [t for t in self.horizon() if p1arcs.get((a, t))]]
                # set extreme bounds for infeasible cases: for computing min lb and max ub over t
                for t in bounds[aid]["f"][0]:
                    bounds[aid]["q"][t] = (1e8, -1e8)
                for t in bounds[aid]["f"][1]:
                    bounds[aid]["dh"][t] = (1e8, -1e8)

        for j, tank in self.tanks.items():
            bounds[j] = {"qin": [[qb for qb in qbounds[(j, t)]] for t in self.horizon()],
                        "h": [[hb for hb in hbounds[(j, t)]] for t in self.horizon()]}

        assert len(qbounds) == (len(self.arcs) + len(self.tanks)) * self.nperiods()
        assert len(hbounds) == len(self.tanks) * self.nperiods()
        assert len(dhbounds) == len(self.varcs) * self.nperiods()
        assert len(p0arcs) == sum(len(bounds[arc.id]["f"][0]) for a, arc in self.varcs.items())
        assert len(p1arcs) == sum(len(bounds[arc.id]["f"][1]) for a, arc in self.varcs.items())
        print(f"{len(p0arcs)} variables fixed to 0, {len(p1arcs)} variables fixed to 1. ")
        print(bounds)
        bfile = open(bndfile, "w")
        json.dump(bounds, bfile)
        bfile.close()
