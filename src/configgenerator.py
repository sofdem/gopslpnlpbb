#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 15:03:46 2022

@author: Sophie Demassey

Create columns for the extended IP.
Ranges of volumes for the tanks are discretized in N steps giving a discrete set of volume configurations.
Columns are given as tuples (t, S, V, V', E):
For each time t, each command S and each volume configuration V, we run the network flow analysis
to compute flow/head satisfying demand at time t  (Q, H) s.t. Q_A=Q_A.S_A, Q_J = D_Jt, H_R = V_R/s_R, H_A = F_A(Q_A)
we derive the volume configuration at t+1: V'_J ~ V_J + Dt*Q_J and the pumping energy  E=E0_K.S_K + E1_K.Q_K.
Columns are filtered either before computation: according to pump/valve or demand symmetries, fixed volumes at t=0
or after computation: if flow is unfeasible flow or a tank capacity is violated.
Commands S and volume configurations V are represented as integers in base 2 and base N respectively,
 or as their base10 integer counterpart.
"""

from hydraulics import HydraulicNetwork
import instance


class ConfigGen:
    """ Column generator.
    Attributes:
        instance: (pump scheduling problem instance
        feastol: feasibility tolerance for flow
        binmatch: the ordered list of controllable arcs for the base 2 representation of S
        binlen: length of a command S in base 2
        commands: commands[S] = 0 if S in range(2^binlen) is unfeasible/redundant, or =  set of inactive arcs in S
        safety: safety margin (absolute value) in volumes to decide if a column is feasible or not
        nvsteps: nb of steps for the discretization of the ranges of volumes of the tanks
        steplen: steplen[j] = length of a step for tank j
        network: network flow analysis instance
        columns: the generated set of columns: columns[t][S,V]=[V',E]
    """

    def __init__(self, instance: instance.Instance, feastol: float, nvsteps: int, safety: float):
        self.instance = instance
        self.feastol = feastol
        self.binmatch = list(self.instance.varcs.keys())
        self.binlen = len(self.binmatch)
        self.commands = self.filter_symmetry()
        self.safety = safety
        self.nvsteps = nvsteps
        self.steplen = {j: (tank.vmax - tank.vmin) / self.nvsteps for j, tank in self.instance.tanks.items()} # lengths of discretized volume step for tanks
        print(f"volumes discretization {self.nvsteps} step lengths: {self.steplen} "
              f"initconf: {self.getvolconf({j: tank.vinit for j, tank in self.instance.tanks.items()})} "
              f"safety: {self.safety}")
        self.network = HydraulicNetwork(self.instance, feastol=self.feastol)
        self.columns = self.generate_all_columns()

    # accessors for columns
    def nbcols(self):
        """ returns the total number of columns. """
        return sum(len(cols) for cols in self.columns)

    @staticmethod
    def command(colkey: tuple) -> int:
        """ Returns the command S (base10 int) for column with colkey=(S,V). """
        return colkey[0]

    def getinactiveset(self, colkey: tuple) -> set:
        """ Returns the set of inactive arcs in command S for column with colkey=(S,V)."""
        return self.commands[self.command(colkey)]

    @staticmethod
    def power(colval: tuple) -> float:
        """ Returns the pump energy consumption E for column with colval=(V',E)."""
        return colval[1]

    def volpre(self, colkey: tuple, tknum: int) -> int:
        """ Returns V[tknum] the volume step number of the tknum-th tank in column with colkey=(S,V)."""
        return self.volstep(colkey[1], tknum)

    def volpost(self, colval: tuple, tknum: int) -> int:
        """ Returns V'[tknum] the volume step number of the tknum-th tank in column with colval=(V',E)."""
        return self.volstep(colval[0], tknum)

    def volpreall(self, period: int, colkey: tuple) -> list:
        """ Returns V the list of volume step numbers for all tanks in column with colkey=(S,V)."""
        return [self.volpre(colkey, tknum) for tknum in range(len(self.instance.tanks))]

    def volpostall(self, period: int, colkey: tuple) -> list:
        """ Returns V' the list of volume step numbers for all tanks in column with colval=(V',E)."""
        return [self.volpost(self.columns[period][colkey], tknum) for tknum in range(len(self.instance.tanks))]

    # volumes configurations
    def volstep(self, volconf: int, tknum: int) -> int:
        """ Returns the step number of the tknum-th tank in the volume configuration volconf. """
        assert 0 <= volconf < pow(self.nvsteps, len(self.instance.tanks)), f"0 <= {volconf} <= {self.nvsteps}^{len(self.instance.tanks)}"
        vol = (volconf // pow(self.nvsteps, tknum)) % self.nvsteps
        return vol

    def volmaxstep(self, tkid: str, step: int) -> int:
        """ Returns the maximum volume value at step for tank tkid. """
        return self.instance.tanks[tkid].vmin + (step + 1) * self.steplen[tkid]

    def volmidstep(self, tkid: str, step: int) -> int:
        """ Returns the median volume value at step for tank tkid. """
        return self.instance.tanks[tkid].vmin + (step + 1/2) * self.steplen[tkid]

    def firstmidvolume(self) -> dict:
        """ Returns the minimum configuration of volumes as a dict of volumes. """
        return {j: tank.vmin + self.steplen[j]/2 for j, tank in self.instance.tanks.items()}

    def nextmidvolume(self, intvol: int, volumes: dict):
        """ Updates volumes with the next configuration represented as base10 (intvol+1). """
        intvol += 1
        for j, tank in self.instance.tanks.items():
            if intvol % self.nvsteps:
                volumes[j] += self.steplen[j]
                return
            volumes[j] = tank.vmin + self.steplen[j]
            intvol //= self.nvsteps

    def getvolconf(self, volumes: dict, withsafety=False) -> int:
        """ Returns the base10 configuration figuring volumes or -1 if out of range. """
        assert len(volumes) == len(self.instance.tanks)
        volconf = 0
        factor = 1
        for j, tank in self.instance.tanks.items():
            step = self.getstep(j, tank, volumes[j], withsafety)
            if step == -1:
                return -1
            volconf += step * factor
            factor *= self.nvsteps
        return volconf

    def getstep(self, tkid: str, tank: instance._Tank, volume: float, withsafety: bool) -> int:
        """ Returns the step number corresponding to the volume value of tank tkid or -1 if out of range. """
        safety = self.safety if withsafety else 0
        if volume < tank.vmin - self.feastol + safety or volume > tank.vmax + self.feastol - safety:
            return -1
        maxvolstep = tank.vmin + self.steplen[tkid]
        for k in range(self.nvsteps-1):
            if volume < maxvolstep:
                return k
            maxvolstep += self.steplen[tkid]
        assert volume >= tank.vmax - self.steplen[tkid] - self.feastol
        return self.nvsteps - 1

    def generate_columns(self, command: int, inactive: set, period: int, columns: dict):
        """ Computes columns for fixed (period, command/inactive set) for all possible volume configurations. """
        if period == 0:
            nbcols = 1
            volumes = {j: tank.vinit for j, tank in self.instance.tanks.items()}
            intvolpre = self.getvolconf(volumes, withsafety=True)
            assert intvolpre >= 0
        else:
            intvolpre = 0
            nbcols = pow(self.nvsteps, len(self.steplen))
            volumes = self.firstmidvolume()

        postvol = {j: 0 for j in self.instance.tanks}
        for c in range(nbcols):
            flow, head, errormsg = self.network._flow_analysis(inactive, period, volumes, stopatviolation=True)
            if not errormsg:
                accept = True
                for j, tank in self.instance.tanks.items():
                    postvol[j] = volumes[j] + self.instance.flowtovolume() * \
                                 (sum(flow[a] for a in self.instance.inarcs(j))
                                  - sum(flow[a] for a in self.instance.outarcs(j)))
                    if postvol[j] < tank.vinit - self.feastol + self.safety and period == self.instance.nperiods()-1:
                        #print(f"reject {command} at tnak {j}: {postvol[j]} < {tank.vinit}")
                        accept = False
                        break
                if accept:
                    intvolpost = self.getvolconf(postvol, withsafety=True)
                    if intvolpost >= 0:
                        power = sum(pump.power[0] + pump.power[1] * flow[a]
                                    for a, pump in self.instance.pumps.items() if flow[a] > self.feastol)
                        columns[command, intvolpre] = [intvolpost, power]
            self.nextmidvolume(intvolpre, volumes)
            intvolpre += 1

    # commands: pump/valve configurations

    def inttobin(self, command: int) -> str:
        """ Returns the base2 string of length 'binlen' corresponding to the base10 command.  """
        return f"{command:b}".zfill(self.binlen)

    def inactiveset(self, command: int) -> set:
        """ Returns the set of inactive arcs in configuration 'command' (base10). """
        inactive = set()
        bincommand = self.inttobin(command)
        for i, c in enumerate(bincommand):
            if c == '0':
                # if isinstance(self.binmatch[i], int):
                inactive.add(self.binmatch[i])
                # else:
                #    inactive.update(self.binmatch[i])
        return inactive

    def generate_all_columns(self) -> list:
        """ generate columns for all feasible/non redundant (t=period, S=command, V=volume configuration). """
        columns = [{} for _ in self.instance.horizon()]
        timesymmetry = [0 for _ in self.instance.horizon()]
        for t in self.instance.horizon():
            found = False
            if t != 0 and t != self.instance.nperiods()-1:
                prof = [p[t] for p in self.instance.profiles.values()]
                for ts in range(1, t):
                    if timesymmetry[ts] and timesymmetry[ts] == prof:
                        columns[t] = columns[ts]
                        print(f"step {t}: copy {len(columns[t])} columns from period {ts}")
                        found = True
                        break
                if not found:
                    timesymmetry[t] = prof
                    ts = 0
            if not found:
                for command, inactiveset in enumerate(self.commands):
                    if inactiveset or command == pow(2, self.binlen)-1:
                        self.generate_columns(command, self.inactiveset(command), t, columns[t])
                print(f"step {t}: generate {len(columns[t])} columns")
        return columns

    def filter_identity(self):
        """ identify strictly equivalent pump/valves in commands"""
        identity = [] #self.instance.identity
        if not identity:
            return list(self.instance.varcs.keys())
        assert len(identity) > 1
        firstsym = identity[0]
        return [k if k not in identity else identity
                for k in self.instance.varcs if k == firstsym or k not in identity]

    def filter_symmetry(self):
        """ remove redundant commands given pump/valves symmetries"""
        sym = self.instance.symmetries
        if not sym:
            return [self.inactiveset(intconfig) for intconfig in range(pow(2, self.binlen))]

        assert len(sym) > 1
        symidx = [idx for idx, k in enumerate(self.binmatch) if k in sym]
        assert len(symidx) == len(sym)
        configs = [set() for _ in range(pow(2, self.binlen))]
        nbconfigs = 0
        for intconfig in range(len(configs)):
            binconfig = self.inttobin(intconfig)
            accept = True
            for i in range(1, len(symidx)):
                if binconfig[symidx[i-1]] == '0' and binconfig[symidx[i]] == '1':
                    accept = False
                    break
            if accept:
                configs[intconfig] = self.inactiveset(intconfig)
                nbconfigs += 1
        print(f"nb configurations before/after filtering= {pow(2, self.binlen)} -> {nbconfigs}")
        return configs

