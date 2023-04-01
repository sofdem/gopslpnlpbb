#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 15:03:46 2022

@author: Sophie Demassey

Create columns for the extended IP.
Ranges of volumes for the tanks are discretized in N steps giving a discrete set of volume configurations.
Columns are given as tuples (time t, command S, init volumes V, final volumes V', energy E):
For each time t, each command S and each volume configuration V, we run the network flow analysis
to compute the unique stationary flow/head equilibrium satisfying demand at time t, i.e. (Q, H) s.t.
Q_A=Q_A.S_A, Q_J = D_Jt, H_R = V_R/s_R, H_A = F_A(Q_A), then derive
the volume configuration at t+1: V'_J ~ V_J + Dt*Q_J and the pumping energy  E=E0_K.S_K + E1_K.Q_K.
Columns are filtered either before computation: according to pump/valve or demand symmetries, fixed volumes at t=0
or after computation: if equilibrium is not feasible flow or a tank capacity is violated.
Commands S and configurations V are represented either by their id (their generation number)
or by their id given as an integer in base 2 or N respectively
"""

from instance import Instance

from networkanalysis import NetworkAnalysis

from new_partition import NetworkPartition

from collections import defaultdict

#d = defaultdict(dict)



class ConfigGen:
    """ Column generator.
    Attributes:
        instance: pump scheduling problem instance
        feastol: feasibility tolerance for flow
        binmatch: the list of controllable arcs ordered as in command ids
        binlen: the number of controllable arcs => max 2^bin commands
        commands[2^binlen]: commands[S] = 0 is unfeasible/redundant, or =  set of inactive arcs in S
        safety: safety margin (absolute value) in volumes to decide if a column is feasible or not
        nvsteps: nb of steps for the discretization of the ranges of volumes of the tanks
        vstep[time][tank]: length of a step for a given tank at a given time
        vmin[time][tank]: min volume of a given tank at a given time
        vmax[time][tank]: max volume of a given tank at a given time
        network: network flow analysis instance to compute the equilibria (either NetworkAnalysis or HydraulicNetwork)
        columns[time][command,volume]: the generated set of columns columns[t][S,V]=[V',E]
    """

    # @todo uniformize feastol (in volume or in flow)
    def __init__(self, instance: Instance, network, netpart, tanks_vol, penalt, feastol: float, nvsteps: int, safety: float,
                 meanvolprofiles: list = None, margin: float = 0.2):
        """ create a column generator:
        when 'menavolprofiles' is specified, only plans satisfying these profiles with a given 'margin' are accepted.
        """
        self.instance = instance
        self.feastol = feastol
        self.binmatch = list(self.instance.varcs.keys())
        self.binlen = len(self.binmatch)
        self.compon = NetworkPartition(self.instance)
        self.components = self.compon.partition
        
        self.compts = self.compon.component

        self.varcccc = self.compon.varccc
        self.varinv = self.compon.var_inv
        
        print(type(self.compts))
        
        print(type(self.varinv))
        print(type(self.components))
        self.commands = self.filter_symmetry_compo()
        self.nvsteps = nvsteps
        self.safety = safety
#        self.netpart = netpart
#        self.varcccc = netpart.varccc
#        self.varinv = netpart.var_inv
#        self.components = netpart.component


        
        self.penalt= penalt
        
        self.tanks_vol = tanks_vol
        
        self.vmin, self.vmax, self.vstep = self.makevprofile(meanvolprofiles, margin)
        print(f"volumes discretization {self.nvsteps} steps, safety: {self.safety}")
        self.network = network
        
#        self.netpart = netpart

##        self.components = NetworkAnalysis._build
#        self.columns = self.generate_all_columns()
##        self.columns_me = self.generate_all_columns_me()

    # @todo integrate the satefy margin to the vmin/vmax profiles
    def makevprofile(self, meanvolprofiles: list = None, margin: float = 0.2):
        """ returns the tank volume profiles: accepted bounds (min/max) and discretization step lengths;
        either absolute (if 'meanvolprofiles' is null) or around 'meanvolprofiles' within the given 'margin' delta """
        nperiods = self.instance.nperiods()
        vinit = {j: tank.vinit for j, tank in self.instance.tanks.items()}
        vmind = {0: vinit}
        vmaxd = {0: vinit}
        if not meanvolprofiles:
            vmin = {j: tank.vmin for j, tank in self.instance.tanks.items()}
            vmax = {j: tank.vmax for j, tank in self.instance.tanks.items()}
            for t in range(1, nperiods):
                vmind[t] = vmin
                vmaxd[t] = vmax
            vmind[nperiods] = vinit
            vmaxd[nperiods] = vmax
        else:
            assert len(meanvolprofiles) == nperiods+1 and len(meanvolprofiles[0]) == len(self.instance.tanks)
            for t in range(1, nperiods+1):
                outofbounds = {j for j, tank in self.instance.tanks.items()
                               if meanvolprofiles[t][j] < tank.vmin - 1e-6
                               or meanvolprofiles[t][j] > tank.vmax + 1e-6}
                assert not outofbounds, f"{outofbounds}"
                vmind[t] = {j: max(tank.vmin, meanvolprofiles[t][j]*(1-margin))
                            for j, tank in self.instance.tanks.items()}
                vmaxd[t] = {j: min(tank.vmax, meanvolprofiles[t][j]*(1+margin))
                            for j, tank in self.instance.tanks.items()}
            vmind[nperiods] = {j: max(tank.vinit, meanvolprofiles[nperiods][j]*(1-margin))
                               for j, tank in self.instance.tanks.items()}
            outofbounds = {(j, meanvolprofiles[nperiods][j]) for j, tank in self.instance.tanks.items()
                           if meanvolprofiles[nperiods][j] < tank.vinit - 1e-6}
            assert not outofbounds, f"outofbounds: {outofbounds}"
        vstepd = {t: {j: (vmaxd[t][j] - vmind[t][j])/self.nvsteps for j in self.instance.tanks}
                  for t in range(nperiods+1)}
        return vmind, vmaxd, vstepd

    def nbcols(self):
        """ returns the total number of columns. """
        return sum(len(cols) for cols in self.columns)

    @staticmethod
    def command(colkey: tuple) -> int:
        """ returns the command number S for column with colkey=(S,V). """
        return colkey[0]

    def getinactiveset(self, colkey: tuple) -> set:
        """ returns the set of inactive arcs in command S for column with colkey=(S,V)."""
        return self.commands[self.command(colkey)]

    @staticmethod
    def power(colval: tuple) -> float:
        """ returns the pump energy consumption E for column with colval=(V',E)."""
        return colval[1]

    def volpre(self, colkey: tuple, tknum: int) -> int:
        """ returns V[tknum] the volume step number of the tknum-th tank in column with colkey=(S,V)."""
        return self.volstep(colkey[1], tknum)

    def volpost(self, colval: tuple, tknum: int) -> int:
        """ returns V'[tknum] the volume step number of the tknum-th tank in column with colval=(V',E)."""
        return self.volstep(colval[0], tknum)

    def volpreall(self, colkey: tuple) -> list:
        """ returns V the list of volume step numbers for all tanks in column with colkey=(S,V)."""
        return [self.volpre(colkey, tknum) for tknum in range(len(self.instance.tanks))]

    def volpostall(self, period: int, colkey: tuple) -> list:
        """ returns V' the list of volume step numbers for all tanks in column with colval=(V',E)."""
        return [self.volpost(self.columns[period][colkey], tknum) for tknum in range(len(self.instance.tanks))]

    # volumes configurations
    def volstep(self, volconf: int, tknum: int) -> int:
        """ returns the step number of the tknum-th tank in the volume configuration volconf. """
        assert 0 <= volconf < pow(self.nvsteps, len(self.instance.tanks)), \
            f"0 <= {volconf} <= {self.nvsteps}^{len(self.instance.tanks)}"
        vol = (volconf // pow(self.nvsteps, tknum)) % self.nvsteps
        return vol

    def volmidstep(self, tkid: str, step: int) -> int:
        """ Returns the median volume value at step for tank tkid. """
        return self.instance.tanks[tkid].vmin + (step + 1 / 2) * self.steplen[tkid]

    def firstmidvolume(self, period: int) -> dict:
        """ Returns the minimum configuration of volumes as a dict of volumes. """
        return {j: self.vmin[period][j] + self.vstep[period][j] / 2 for j, tank in self.instance.tanks.items()}

    def nextmidvolume(self, intvol: int, volumes: dict, period: int):
        """ Updates volumes with the next configuration id (intvol+1). """
        intvol += 1
        for j, tank in self.instance.tanks.items():
            if intvol % self.nvsteps:
                volumes[j] += self.vstep[period][j]
                return
            volumes[j] = self.vmin[period][j] + self.vstep[period][j]
            intvol //= self.nvsteps

    def getnextvolconf(self, prevol: dict, flow: dict, period: int, withsafety=False) -> int:
        """ Returns the configuration id figuring next volumes prevol+inflow or -1 if infeasible or out of range. """
        assert len(prevol) == len(self.instance.tanks)
        if not flow:
            return -1
        assert len(flow) == len(self.instance.arcs)
        volconf = 0
        factor = 1
        for j in self.instance.tanks:
            step = self.getstep(j, period, prevol[j] + self.instance.inflow(j, flow), withsafety)
            if step == -1:
                return -1
            volconf += step * factor
            factor *= self.nvsteps
        return volconf
    
    def getnextvolconf_modified(self, prevol: dict, flow: dict, period: int, withsafety=False) -> int:
        """ Returns the configuration id figuring next volumes prevol+inflow or -1 if infeasible or out of range. """
        assert len(prevol) == len(self.instance.tanks)
        if not flow:
            return -1
        assert len(flow) == len(self.instance.arcs)
        volconf = 0
        factor = 1
        for j in self.instance.tanks:
            step = self.getstep(j, period, prevol[j] + self.instance.inflow(j, flow), withsafety)
            if step == -1:
                return -1
            volconf += step * factor
            factor *= self.nvsteps
        return volconf

    def getstep(self, tkid: str, period: int, volume: float, withsafety: bool) -> int:
        """ Returns the step number corresponding to the volume value of tank tkid or -1 if out of range. """
        safety = self.safety if withsafety else 0
        if volume < self.vmin[period][tkid] - self.feastol + safety \
                or volume > self.vmax[period][tkid] + self.feastol - safety:
            return -1
        maxvolstep = self.vmin[period][tkid] + self.vstep[period][tkid]
        for k in range(self.nvsteps - 1):
            if volume < maxvolstep:
                return k
            maxvolstep += self.vstep[period][tkid]
        assert volume >= self.vmax[period][tkid] - self.vstep[period][tkid] - self.feastol
        return self.nvsteps - 1

    # @todo compute only for volume configurations reachable from the previous period
#    def generate_columns(self, command: int, inactive: set, period: int, columns: dict):
#        """ Computes columns for fixed (period, command/inactive set) for all possible volume configurations. """
#        if period == 0:
#            nbcols = 1
#            volumes = self.vmin[0].copy()
#        else:
#            nbcols = pow(self.nvsteps, len(self.instance.tanks))
#            volumes = self.firstmidvolume(period)

#        intvolpre = 0
#        for c in range(nbcols):
#            flow = self.network.flow_analysis(inactive, period, volumes, stopatviolation=True)
#            intvolpost = self.getnextvolconf(volumes, flow, period+1, withsafety=True)
#            if intvolpost >= 0:
#                power = sum(pump.power[0] + pump.power[1] * flow[a]
#                            for a, pump in self.instance.pumps.items() if flow[a] > self.feastol)
#                columns[command, intvolpre] = [intvolpost, power]
#            self.nextmidvolume(intvolpre, volumes, period+1)
#            intvolpre += 1
            
##    def generate_columns_me(self, command: int, inactive: set, period: int, volumes: float, columns: dict):
##        """ Computes columns for fixed (period, command/inactive set) for all possible volume configurations. """
##        """computing the power for each subcinguration corresponding to the components"""
##        """we have exactly one nbcols because the level of the given from second subproblem"""
##        """take the volume from second subproblem"""
##        if period == 0:
##            nbcols = 1
##            volumes = self.vmin[0].copy()
##        else:
##            nbcols = 1
            
#            nbcols = pow(self.nvsteps, len(self.instance.tanks))
#            volumes = self.firstmidvolume(period)

##        intvolpre = 0
##        power= {}
##        sum_qb=0
##        sum_tank={}
##        error_q={}
##        for c in range(nbcols):
##            compo = self.NetworkAnalysis.component
##            for k, cck in compo.items():
##                flow = self.network.flow_analysis_compon(inactive, k, period, volumes, stopatviolation=True)
##                for tt, tank in self.instance.tanks.items():
##                    for a, arc in self.instance.arcs.items():
##                        if a in k and ( a in self.instance.inarc(tt)):
##                            sum_qb= sum_qb+flow[a]
#                            error_q= volumes-flow[a]
##                        elif a in k and (a in self.instance.outarc(tt)):
##                            sum_qb= sum_qb-flow[a]
                            
##                    sum_tank={tt:sum_qb}
##                error_q[command]= 
                    
##                    error_q[command]= volumes-sum_qb-flow[a]
        
##                power[(k, command, period)] = sum(pump.power[0] + pump.power[1] * flow[a]
##                            for a, pump in self.instance.pumps.items() if a in k)
                
                
##                columns[command, k] = power
                
                
#            flow = self.network.flow_analysis(inactive, period, volumes, stopatviolation=True)
#            intvolpost = self.getnextvolconf(volumes, flow, period+1, withsafety=True)
#            if intvolpost >= 0:
#                """power for all possible subconfigurations instead of all configurations"""
#                power = sum(pump.power[0] + pump.power[1] * flow[a]
#                            for a, pump in self.instance.pumps.items() if flow[a] > self.feastol)
#                columns[command, intvolpre] = [intvolpost, power]
#            self.nextmidvolume(intvolpre, volumes, period+1)
#            intvolpre += 1


#    def generate_columns_me_version2(self, commands: dict, inactive: set, period: int, volumes: float, columns: dict):
    def generate_columns_me_version2(self, commands: dict, period: int, volumes: float, columns: dict):
        """ Computes columns for fixed (period, command/inactive set) for all possible volume configurations. """
        """computing the power for each subcinguration corresponding to the components"""
        """we have exactly one nbcols because the level of the given from second subproblem"""
        """take the volume from second subproblem"""
        #the problem of the last version was the fact that command was coming from outside while it is related to component inside the function, it's like removing generate_column
#        if period == 0:
#            nbcols = 1
#            volumes = self.vmin[0].copy()
#        else:
#            nbcols = 1
            
#            nbcols = pow(self.nvsteps, len(self.instance.tanks))
#            volumes = self.firstmidvolume(period)

#        intvolpre = 0
        power= {}
        
        
        sum_inf_tank={}
        sum_inf_tank_com={}
        
        error_q_tot={}
        
#        for c in range(nbcols):
#        compo = self.NetworkAnalysis.component
#        for k, cck in compo.items():
        for k, cck in self.components.items():
#                for command in commands[k]:
                for command, inactiveset in self.commands[k].items():
                    inactive=  self.inactiveset_compo(command, k)
#                    flow = self.network.flow_analysis_compon(inactive, k, period, volumes)
                    flow = self.network.flow_analysis_me(inactive, period, volumes)
                    print("yo ho")
                    print(flow)
                    sum_qb=0
                    error_q=0
                    for tt, tank in self.instance.tanks.items():
                        for a, arc in self.instance.arcs.items():
                            print("Ajab")
                            print(self.instance.inarcs(tt))
                            print(a)
                            print(flow[a])
                            if a in cck['a'] and ( a in self.instance.inarcs(tt)):

                                sum_qb= sum_qb+flow[a]
                                
                                print("so why")
#                            error_q= volumes-flow[a]
                            elif a in cck['a'] and (a in self.instance.outarcs(tt)):
                                sum_qb= sum_qb-flow[a]
                                
                                
                                
                                print("perche")
                                
                        sum_inf_tank={tt:sum_qb}
                        
                    
                        
                        
                        print('Hereeeee')
                        print(cck)
                        print(sum_inf_tank)
                        
                        error_q= error_q + self.penalt[tt, period]* abs((volumes[tt, period+1]-volumes[tt, period])/(self.instance.flowtoheight(tank))-(sum_inf_tank[tt]-flow[('T1','J1')]))
                        
                    error_q_tot[command]=error_q
                    
                    
                    sum_inf_tank_com[command]= sum_qb
                    

                
                    power[(k, command, period)]= sum(self.instance.eleccost(period)*((pump.power[0] if abs(flow[a]) > 0.01 else 0) + (pump.power[1]) * (flow[a]))
                            for a, pump in self.instance.pumps.items() if a in cck['a']) + error_q_tot[command]
                
                    columns[(command, k)] = power[(k, command, period)]
                    
                print("Mozakhrafat")
                print(error_q_tot)
                print(sum_inf_tank_com)
                print("lo lo l o lo")

#    def inttobin(self, command_id: int) -> str:
#        """ Returns the base2 string of length 'binlen' corresponding to the base10 command id.  """
#        return f"{command_id:b}".zfill(self.binlen)

#    def inactiveset(self, command_id: int) -> set:
#        """ Returns the set of inactive arcs corresponding to the base10 command id. """
#        inactive = set()
#        bincommand = self.inttobin(command_id)
#        for i, c in enumerate(bincommand):
#            if c == '0':
#                # if isinstance(self.binmatch[i], int):
#                inactive.add(self.binmatch[i])
#                # else:
#                #    inactive.update(self.binmatch[i])
#        return inactive
    

    def inttobin_compo(self, command_id: int, K) -> str:
        """ Returns the base2 string of length 'binlen' corresponding to the base10 command id.  """
#        compo = self.NetworkAnalysis.component
        for k, cck in self.components.items():
            if k==K:
                binlen_=len(self.varinv[k])
            
        return f"{command_id:b}".zfill(binlen_)
    
    def inactiveset_compo(self, command_id: int, k) -> set:
        """ Returns the set of inactive arcs corresponding to the base10 command id. """
        inactive = set()
        bincommand = self.inttobin_compo(command_id, k)
        for i, c in enumerate(bincommand):
            if c == '0':
                # if isinstance(self.binmatch[i], int):
#                list(varinv.keys())
                inactive.add(self.varinv[k][i])
                # else:
                #    inactive.update(self.binmatch[i])
        return inactive

 #   def generate_all_columns(self) -> list:
 #       """ generate columns for all feasible/non redundant (t=period, S=command, V=volume configuration). """
 #       columns = [{} for _ in self.instance.horizon()]
 #       timesymmetry = [0 for _ in self.instance.horizon()]
 #       for t in self.instance.horizon():
 #           found = False
 #           if t != 0 and t != self.instance.nperiods() - 1:
 #               prof = [p[t] for p in self.instance.profiles.values()]
 #               for ts in range(1, t):
 #                   if timesymmetry[ts] and timesymmetry[ts] == prof:
 #                       columns[t] = columns[ts]
 #                       print(f"step {t}: copy {len(columns[t])} columns from period {ts}")
 #                       found = True
 #                       break
 #               if not found:
 #                   timesymmetry[t] = prof
 #           if not found:
 #               for command, inactiveset in self.commands.items():
 #                   self.generate_columns(command, self.inactiveset(command), t, columns[t])
 #               print(f"step {t}: generate {len(columns[t])} columns")
 #               assert len(columns[t]), f"no feasible configuration at period {t}"
 #       print(columns[-1].keys())
 #       return columns
    
    def generate_all_columns_me(self, period) -> list:
        """ generate columns for all feasible/non redundant (t=period, S=command, V=volume configuration). """
        columns = [{} for _ in self.instance.horizon()]
##        columns = {ti:{} for ti in self.instance.horizon()}
        coll={}
#        timesymmetry = [0 for _ in self.instance.horizon()]
        for t in self.instance.horizon():
            if t == period:
#            found = False
#            if t != 0 and t != self.instance.nperiods() - 1:
#                prof = [p[t] for p in self.instance.profiles.values()]
#                for ts in range(1, t):
#                    if timesymmetry[ts] and timesymmetry[ts] == prof:
#                        columns[t] = columns[ts]
#                        print(f"step {t}: copy {len(columns[t])} columns from period {ts}")
#                        found = True
#                        break
#                if not found:
#                    timesymmetry[t] = prof
#            if not found:
                for command, inactiveset in self.commands.copy().items():
                    
##                    columns[t]= self.generate_columns_me_version2(command, t, self.tanks_vol, columns[t])
####                    print("really")
####                    print(self.commands.items())
#                    self.generate_columns_me_version2(command, self.inactiveset(command), t, self.tanks_vol, columns[t])
                    self.generate_columns_me_version2(command, t, self.tanks_vol, columns[t])
                print(f"step {t}: generate {len(columns[t])} columns")
                assert len(columns[t]), f"no feasible configuration at period {t}"
        print(columns[-1].keys())
        return columns
#        return coll

    def filter_identity(self):
        """ identify strictly equivalent pump/valves in commands"""
        identity = []  # self.instance.identity
        if not identity:
            return list(self.instance.varcs.keys())
        assert len(identity) > 1
        firstsym = identity[0]
        return [k if k not in identity else identity
                for k in self.instance.varcs if k == firstsym or k not in identity]

#    def filter_symmetry(self):
#        """ remove redundant commands given pump/valves symmetries"""
#        sym = self.instance.symmetries
#        symidx = [idx for idx, k in enumerate(self.binmatch) if k in sym] if sym else []
#        assert len(symidx) == len(sym)

#        configs = {}
#        maxconfig = pow(2, self.binlen)
#        for intconfig in range(maxconfig):
#            accept = True
#            if sym:
#                binconfig = self.inttobin(intconfig)
#                for i in range(1, len(symidx)):
#                    if binconfig[symidx[i - 1]] == '0' and binconfig[symidx[i]] == '1':
#                        accept = False
#                        break
#            if accept:
#                inactives = self.inactiveset(intconfig)
#                if self.filter_dependencies(inactives):
#                    configs[intconfig] = inactives
#        print(f"nb configurations before/after filtering= {pow(2, self.binlen)} -> {len(configs)}")
#        return configs
    
    def filter_symmetry_compo(self):
        """ remove redundant commands given pump/valves symmetries"""
        sym = self.instance.symmetries
        configs= defaultdict(dict)
        #it was component in the argument of the function but I change it to self.components
#        for komp, comp in self.components:
        for komp, comp in self.components.items():
            if komp in self.varinv.keys(): 
            
        
#            symidx = [idx for idx, k in enumerate(list(self.varinv[komp].keys())) if k in sym] if sym else []
###                print(self.varinv)
###                print(self.components.items())
###                print("gott")
###                print(comp)
###                print("so far")
###                print(komp)
###                print("until now")
###                print(type(self.varinv))
###                print("so so")
###                print(self.varinv[komp])
###                print("wow")
###                print(self.varinv)
###                print((list(self.varinv[komp])))
                symidx = [idx for idx, k in enumerate(list(self.varinv[komp])) if k in sym] if sym else []
                assert len(symidx) == len(sym)

#            configs = {}
            
#            maxconfig = pow(2, self.len(list(self.varinv[komp].keys())))
                maxconfig = pow(2, len(list(self.varinv[komp])))
            else:
                maxconfig= 0
            for intconfig in range(maxconfig):
                accept = True
                if sym:
                    binconfig = self.inttobin_compo(intconfig, komp)
                    for i in range(1, len(symidx)):
                        if binconfig[symidx[i - 1]] == '0' and binconfig[symidx[i]] == '1':
                            accept = False
                            break
                if accept:
                    inactives = self.inactiveset_compo(intconfig, komp)
                    if self.filter_dependencies(inactives):
                        
                        configs[komp][intconfig] = inactives
#                        configs= {komp:{intconfig:inactives}}
            print(f"nb configurations before/after filtering= {pow(2, self.binlen)} -> {len(configs)}")
        return configs

    def filter_dependencies(self, inactiveset: set) -> bool:
        """ remove unfeasible command given pump/valves dependencies"""
        dep = self.instance.dependencies
        if not dep:
            return True
        # if ({self.instance.varcs[aid].id for aid in inactiveset} & {'v1', '1A', '2A', 'v2', '3A'}) == {'v3', 'v4'}:
        #    print({self.instance.varcs[aid].id for aid in inactiveset})

        for (p0, p1) in dep['p1 => p0']:
            if p0 in inactiveset and p1 not in inactiveset:
                return False
        for (p0, p1) in dep['p0 xor p1']:
            if (p0 in inactiveset) == (p1 in inactiveset):
                return False
        for (p0, p1, p2) in dep['p0 = p1 xor p2']:
            if p1 in inactiveset:
                if (p0 in inactiveset) != (p2 in inactiveset):
                    return False
            elif p2 not in inactiveset:
                return False
            elif p0 in inactiveset:
                return False
        for (p0, p1) in dep['p1 => not p0']:
            if p1 not in inactiveset and p0 not in inactiveset:
                return False
        # print({self.instance.varcs[aid].id for aid in inactiveset} & {'v1', '1A', '2A', '3A', 'v2'})
        # print({self.instance.varcs[aid].id for aid in inactiveset & {'v3', 'v4', '6D'}})
        return True
