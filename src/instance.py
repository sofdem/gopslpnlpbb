#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 11:07:46 2020

@author: Sophie Demassey, Gratien Bonvin
"""

import csv
import pandas as pd
import datetime as dt
from pathlib import Path


TARIFF_COLNAME = 'elix'

def update_min(oldlb: float, newlb: float) -> float:
    """Update lower bound only if better: returns max(oldlb, newlb)."""
    if newlb <= oldlb:
        print(f'do not update min {oldlb:.3f} to {newlb:.3f}')
        return oldlb
    return newlb

def update_max(oldub: float, newub: float) -> float:
    """Update upper bound only if better: returns min(oldub, newub)."""
    if newub >= oldub:
        print(f'do not update max {oldub:.3f} to {newub:.3f}')
        return oldub
    return newub


class _Node:
    """Generic network node. coordinates X, Y, Z."""

    def __init__(self, id_, x, y, z):
        self.id = id_
        self.coord = {'x': x, 'y': y, 'z': z}

    def altitude(self):
        return round(self.coord['z'], 2)


class _Tank(_Node):
    """Network node of type cylindrical water tank.

    vmin    : minimum volume value (in m3)
    vmax    : maximum volume value (in m3)
    vinit   : initial volume       (in m3)
    surface : surface              (in m2)
    """

    def __init__(self, id_, x, y, z, vmin, vmax, vinit, surface):
        _Node.__init__(self, id_, x, y, z)
        self.vmin = round(vmin, 2)
        self.vmax = round(vmax, 2)
        self.vinit = round(vinit, 2)
        self.surface = round(surface, 2)

    def head(self, volume):
        return self.altitude() + volume / self.surface


class _Junction(_Node):
    """Network node of type junction.

    dmean    : mean demand (in m3/h)
    dprofile : demand pattern profile id
    """

    def __init__(self, id_, x, y, z, dmean, profileid):
        _Node.__init__(self, id_, x, y, z)
        self.dmean = dmean
        self.profileid = profileid
        self.dprofile = None

    def setprofile(self, profile):
        self.dprofile = profile[self.profileid]

    def demand(self, t):
        return self.dmean * self.dprofile[t]


class _Reservoir(_Node):
    """Network node of type infinite reservoirs (sources).

    hprofile : head profile id
    drawmax  : maximal withdrawal (in m3/h)
    drawcost : withdrawal cost    (in euro/(m3/h))
    """

    def __init__(self, id_, x, y, z, profileid, drawmax, drawcost):
        _Node.__init__(self, id_, x, y, z)
        self.drawmax = None if (drawmax == 'NO') else drawmax
        self.drawcost = drawcost
        self.profileid = profileid
        self.hprofile = None

    def setprofile(self, profile):
        self.hprofile = profile[self.profileid]

    def head(self, t):
        return self.altitude() * self.hprofile[t]


class _Arc:
    """Generic network arc.

    id      : identifier '(i,j)' with i the start node id, and j the end node id
    qmin    : minimum flow value <= q(i,j) (in m3/h)
    qmax    : maximum flow value >= q(i,j) (in m3/h)
    """

    def __init__(self, id_, nodes, qmin, qmax):
        self.id = id_
        self.nodes = nodes
        self.qmin = qmin
        self.qmax = qmax

    def update_qbounds(self, qmin, qmax):
        self.qmin = update_min(self.qmin, qmin)
        self.qmax = update_max(self.qmax, qmax)

    def __str__(self):
        return f'{self.id} [{self.qmin}, {self.qmax}]'


class _Pipe(_Arc):
    """Network arc of type pipe.

    hloss   : head loss polynomial function: h(i) - h(j) = sum_n hloss[n] q(i,j)^n (in meter)
    """

    def __init__(self, id_, nodes, hloss, qmin, qmax):
        _Arc.__init__(self, id_, nodes, qmin, qmax)
        self.hloss = hloss

    def __str__(self):
        return f'P{self.id} [{self.qmin}, {self.qmax}] {self.hloss}'


class _Valve(_Arc):
    """Network arc of type valve.

    type     : 'GV' or 'PRV' or ???
    hlossmin : minimal head loss <= h(i) - h(j) (in m)
    hlossmax : maximal head loss >= h(i) - h(j) (in m)
    """

    def __init__(self, id_, nodes, type_, hlossmin, hlossmax, qmin, qmax):
        _Arc.__init__(self, id_, nodes, qmin, qmax)
        self.type = type_
        self.hlossmin = hlossmin
        self.hlossmax = hlossmax
        if type_ == 'PRV':
            raise NotImplementedError('pressure reducing valves are not yet supported')

    def update_hbounds(self, dhmin, dhmax):
        self.hlossmin = update_min(self.hlossmin, dhmin)
        self.hlossmax = update_max(self.hlossmax, dhmax)

    def __str__(self):
        return f'V{self.id} [{self.qmin}, {self.qmax}] {self.type} [{self.hlossmin}, {self.hlossmax}]'

class _Pump(_Arc):
    """Network arc of type pump.

    type    : 'FSD' or 'VSD'
    hgain   : head gain polynomial function: h(j) - h(i) = sum_n headgain[n]q(i,j)^n (in meter)
    power   : power polynomial function: p = sum_n power[n]q(i,j)^n (in ???)
    """

    def __init__(self, id_, nodes, hgain, power, qmin, qmax, offdhmin, offdhmax, type_):
        _Arc.__init__(self, id_, nodes, qmin, qmax)
        self.type = type_
        self.hgain = hgain
        self.power = power #[round(p, 6) for p in power]
        self.offdhmin = offdhmin
        self.offdhmax = offdhmax
        if type_ == 'VSD':
            raise NotImplementedError('variable speed pumps are not yet supported')

    def update_hbounds(self, dhmin, dhmax):
        self.offdhmin = update_min(self.offdhmin, dhmin)
        self.offdhmax = update_max(self.offdhmax, dhmax)

    def powerval(self, q):
        assert len(self.power) == 2
        return self.power[0] + self.power[1] * q if q else 0

    def __str__(self):
        return f'K{self.id} [{self.qmin}, {self.qmax}] {self.hgain} {self.power} {self.type} [{self.offdhmin}, {self.offdhmax}]'


class Instance:
    """Instance of the Pump Scheduling Problem."""

    DATADIR = Path("../data/")
    BNDSDIR = Path("../bounds/")

    def __init__(self, name, profilename, starttime, endtime, aggregatesteps):
        self.name        = name
        self.tanks       = self._parse_tanks('Reservoir.csv', self._parse_initvolumes('History_V_0.csv'))
        self.junctions   = self._parse_junctions('Junction.csv')
        self.reservoirs     = self._parse_reservoirs('Source.csv')
        self.pumps       = self._parse_pumps('Pump.csv')
        self.pipes       = self._parse_pipes('Pipe.csv')
        self.valves      = self._parse_valves('Valve_Set.csv')

        self.arcs        = self._getarcs()
        self.nodes       = self._getnodes()
        self.incidence   = self._getincidence()

        periods, profiles = self._parse_profiles(f'{profilename}.csv', starttime, endtime, aggregatesteps)
        self.periods     = periods
        self.profiles    = profiles
        self.tariff      = profiles[TARIFF_COLNAME]
        self.tsduration  = self._get_timestepduration(periods)

        self.dependencies = self._dependencies()
        self.symmetries   = self._pump_symmetric()

        for r in self.reservoirs.values():
            r.setprofile(profiles)
        for j in self.junctions.values():
            j.setprofile(profiles)

    def nperiods(self): return len(self.periods) - 1
    def horizon(self): return range(self.nperiods())
    def tsinhours(self): return self.tsduration.total_seconds() / 3600
    def eleccost(self, t): return self.tsinhours() * self.tariff[t] / 1000
    def inarcs(self, node): return self.incidence[node, 'in']
    def outarcs(self, node): return self.incidence[node, 'out']

    #  PARSERS

    def _parsecsv(self, filename):
        csvfile = open(self.DATADIR / self.name / filename)
        rows = csv.reader(csvfile, delimiter=';')
        data = [[x.strip() for x in row] for row in rows]
        return data


    def _parse_initvolumes(self, filename):
        data = self._parsecsv(filename)
        return {A[0]: float(A[1]) for A in data[1:]}

    def _parse_pumps(self, filename):
        data = self._parsecsv(filename)
        return dict({(A[1], A[2]): _Pump(A[0], (A[1], A[2]),
                                         [round(float(A[c]), 6) for c in [5, 4, 3]],
                                         [round(float(A[c]), 6) for c in [7, 6]],
                                         float(A[8]), float(A[9]),
                                         float(A[11]), float(A[10]), # !!! inverse Min-Max !
                                         A[12]) for A in data[1:]})

    def _parse_valves(self, filename):
        data = self._parsecsv(filename)
        return dict({(A[1], A[2]): _Valve(A[0], (A[1], A[2]), A[3], float(A[4]), float(A[5]),
                                          float(A[6]), float(A[7])) for A in data[1:]})

    def _parse_pipes(self, filename):
        data = self._parsecsv(filename)
        return dict({(A[1], A[2]): _Pipe(A[0], (A[1], A[2]), [0, float(A[4]), float(A[3])],
                                         float(A[5]), float(A[6])) for A in data[1:]})

    def _parse_junctions(self, filename):
        data = self._parsecsv(filename)
        return dict({A[0]: _Junction(A[0], float(A[1]), float(A[2]),
                                     float(A[3]), float(A[4]), A[5]) for A in data[1:]})

    def _parse_reservoirs(self, filename):
        data = self._parsecsv(filename)
        return dict({A[0]: _Reservoir(A[0], float(A[1]), float(A[2]), float(A[3]),
                                   A[4], A[5], float(A[6])) for A in data[1:]})

    def _parse_tanks(self, filename, initvolumes):
        data = self._parsecsv(filename)
        return dict({A[0]: _Tank(A[0], float(A[1]), float(A[2]), float(A[3]),
                                      float(A[4]), float(A[5]), initvolumes[A[0]],
                                      float(A[6])) for A in data[1:]})

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
                profiles[n].append(float(data[i][j+1]))
            periods.append(dt.datetime.strptime(data[i][0], '%d/%m/%Y %H:%M'))
            i += aggsteps
        assert i < len(data), f'end time {starttime} not found in {filename}'
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
        sumagg = [0 for n in profilename]
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
                sumagg[j] += float(data[i][j+1])
        assert i < len(data), f'{filename}: not found end {starttime}'
        return periods, profiles

    def _get_timestepduration(self, periods):
        duration = periods[1] - periods[0]
        for i in range(len(periods)-1):
            assert duration == periods[i+1] - periods[i]
        return duration

    # !!! inverse the dh bounds for pumps in the hdf file
    # !!! do not substract the error margin (1e-2) when lb = 0 (for pumps especially !)
    def parse_bounds(self, filename=None):
        """Parse bounds in the hdf file."""
        file = Path(Instance.BNDSDIR, filename if filename else self.name)
        bounds = pd.read_hdf(file.with_suffix('.hdf')).to_dict()
        for i, b in bounds.items():
            if i[1] == 'flow':
                if i[0] in self.arcs:
                    self.arcs[i[0]].update_qbounds(b[0]-1e-2, b[1]+1e-2)
            elif i[1] == 'head':
                if i[0] in self.pumps:
                    self.arcs[i[0]].update_hbounds(b[1]-1e-2, b[0]+1e-2)
                elif i[0] in self.valves:
                    self.arcs[i[0]].update_hbounds(b[0]-1e-2, b[1]+1e-2)


    def _getarcs(self):
        arcs = dict(self.pumps)
        arcs.update(self.valves)
        arcs.update(self.pipes)
        return arcs


    def _getnodes(self):
        nodes = dict(self.junctions)
        nodes.update(self.tanks)
        nodes.update(self.reservoirs)
        return nodes


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
        dep = {'p1 => p0': set(), 'p0 or p1': set(),
               'p0 <=> p1 xor p2': set(), 'p1 => not p0': set()}

        if self.name == 'Richmond':
            dep['p1 => p0'].add((('196', '768'), ('209', '766')))
            dep['p1 => p0'].add((('196', '768'), ('175', '186')))
            dep['p1 => p0'].add((('312b', 'TD'), ('264', '112')))
            dep['p1 => p0'].add((('264', '112'), ('312b', 'TD')))

            dep['p0 or p1'].add((('201b', '770'), ('196', '768')))
            dep['p0 or p1'].add((('321b', '312'), ('264', '112')))

            dep['p0 <=> p1 xor p2'].add((('196', '768'), ('164b', '197'), ('175', '186')))
        else:
            dep = None

        return dep


    def tostr_basic(self):
        return f'{self.name} {self.periods[0]} {self.horizon()}'


    def tostr_network(self):
        return f'{len(self.pumps)} pumps, {len(self.valves)} valves, {len(self.tanks)} tanks'


    def print_all(self):
        print(f'{self.tostr_basic()} {self.tostr_network()}')
        print(f'{len(self.arcs)} arcs:')
        for i, a in self.arcs.items():
            print(i)
            print(str(a))


    def transcript_bounds(self, csvfile):
        """Parse bounds in the hdf file."""
        file = Path(Instance.BNDSDIR, self.name)
        pd.read_hdf(file.with_suffix('.hdf')).to_csv(csvfile)

