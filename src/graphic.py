#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 12:58:58 2020

@author: sofdem
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np


def pumps(instance, qnlp, qconvex=None):
    fig, axs = plt.subplots(nrows=len(instance.pumps) + 1, ncols=1, sharex=True, figsize=(8,(len(instance.pumps)+1)*1.5))
#    fig.suptitle(instance.name, fontsize=14)

    x = instance.periods[:-1]
    bar_width = x[1]-x[0]
    qmax = max(p.qmax for p in instance.pumps.values()) * 1.1


    cm = plt.cm.get_cmap('OrRd')
    eleccost = [instance.eleccost(t) for t in instance.horizon()]
    ecmin = 0 #min(eleccost)/2
    ecspan = max(eleccost)-ecmin
    ecm = [cm((cost-ecmin)/ecspan) for cost in eleccost]

    for n, (a, p) in enumerate(instance.pumps.items()):
        qnlpa = [qt[a] for t, qt in qnlp.items()]
        axs[n].bar(x, qnlpa, bar_width, align='edge', color=ecm, edgecolor='white', label='real flow in $m^3/h$')
        axs[n].set_ylabel(f'({a[0]},{a[1]})', bbox=dict(fc='#557f2d', alpha=0.2, pad=3), fontsize=10)
        axs[n].set_ylim(0, qmax)
        axs[n].fill_between(x, p.qmin, p.qmax, color='#7f6d5f', alpha=0.1)

    axs[-1].step(x, eleccost, where='post', color='#7f6d5f', label='elec in 'u"\u20AC"'/MWh')
    #axs[-1].bar(x, eleccost, bar_width, align='edge', alpha=0.7, color=ecm, label='elec in 'u"\u20AC"'/MWh')
    axs[-1].set_ylim(min(0, min(eleccost)), max(eleccost) * 1.1)

    locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
    formatter = mdates.ConciseDateFormatter(locator)
    axs[-1].xaxis.set_major_locator(locator)
    axs[-1].xaxis.set_major_formatter(formatter)
    axs[-1].set_xlim(x[0], x[-1])

    plt.tight_layout()

    axs[0].set_title('pump flow in $m^3/h$', fontsize=9)
    axs[-1].set_title('electricity cost in 'u"\u20AC"'/MWh', fontsize=9)

    plt.subplots_adjust(bottom=0.1, right=0.95, top=0.9)
    cax = plt.axes([0.975, 0.1, 0.025, 0.8])

    sm = plt.cm.ScalarMappable(cmap=cm, norm=plt.Normalize(ecmin, ecmin + ecspan))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label('electricity cost in 'u"\u20AC"'/MWh', rotation=270, labelpad=25)

    plt.show()





def pumpsStep(instance, qnlp, qconvex=None):
    fig, axs = plt.subplots(nrows=len(instance.pumps) + 1, ncols=1, sharex=True, figsize=(8,(len(instance.pumps)+1)*1.5))
#    fig.suptitle(instance.name, fontsize=14)

    x = instance.periods
    qmax = max(p.qmax for p in instance.pumps.values()) * 1.1

    for n, (a, p) in enumerate(instance.pumps.items()):
        qnlpa = [qt[a] for t, qt in qnlp.items()]
        qnlpa.append(qnlpa[-1])
        axs[n].step(x, qnlpa, where='post', color='#557f2d', label='real flow in $m^3/h$')
        axs[n].set_ylabel(f'({a[0]},{a[1]})', bbox=dict(fc='#557f2d', alpha=0.2, pad=3), fontsize=10)
        axs[n].set_ylim(0, qmax)
        axs[n].fill_between(x, p.qmin, p.qmax, color='#557f2d', alpha=0.2)

    eleccost = [instance.eleccost(t) for t in instance.horizon()]
    eleccost.append(eleccost[-1])
    axs[-1].step(x, eleccost, where='post', color='#7f6d5f', label='elec in 'u"\u20AC"'/MWh')
    axs[-1].set_ylim(min(0, min(eleccost)), max(eleccost) * 1.1)

    locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
    formatter = mdates.ConciseDateFormatter(locator)
    axs[-1].xaxis.set_major_locator(locator)
    axs[-1].xaxis.set_major_formatter(formatter)
    axs[-1].set_xlim(x[0], x[-1])

    plt.tight_layout()

    axs[0].set_title('pump flow in $m^3/h$', fontsize=9)
    axs[-1].set_title('electricity cost in 'u"\u20AC"'/MWh', fontsize=9)
    plt.show()


def tanks(instance, qnlp, volnlp, volconv=None):
    fig, axs = plt.subplots(nrows=len(instance.tanks), ncols=1, sharex=True, figsize=(8, len(instance.tanks)*1.5));
    if len(instance.tanks) == 1 :
        axs = [axs]
    ax2 = []
    x = instance.periods[:-1]
    bar_width = x[1]-x[0]

    for n, (j, tk) in enumerate(instance.tanks.items()):
        inflow = [sum(max(0, qt[a]) for a in instance.inarcs(j)) - sum(min(0, qt[a]) for a in instance.outarcs(j)) for t, qt in qnlp.items()]
        ouflow = [sum(min(0, qt[a]) for a in instance.inarcs(j)) - sum(max(0, qt[a]) for a in instance.outarcs(j)) for t, qt in qnlp.items()]
        axs[n].bar(x, inflow, bar_width, alpha=0.3, align='edge', color='#557f2d', label='inflow in $m^3/h$')
        axs[n].bar(x, ouflow, bar_width, alpha=0.3, align='edge', color='#7f6d5f', label='outflow in $m^3/h$')

        #plt.tick_params(axis='x', which='both', labelbottom='off')
        axs[n].set_ylabel(j, bbox=dict(fc='DarkOrange', pad=3, alpha=0.2), fontsize=12)
        #plt.xlim([instance.periods[0], instance.periods[-1]])

        mini = min(ouflow)
        maxi = max(inflow)
        axs[n].set_ylim(mini * 1.2, maxi * 1.2)
        axs[n].set_yticks([round(mini), 0, round(maxi)])

        ax2.append(axs[n].twinx())
        hwaternlp = [vol[j] / tk.surface for vol in volnlp]
        ax2[n].plot(instance.periods, hwaternlp, 'DarkOrange', linestyle='-', linewidth=3, label='real water height in $m$')

        hwatermin = tk.vmin / tk.surface
        hwatermax = tk.vmax / tk.surface
        ax2[n].fill_between(instance.periods, hwatermin, hwatermax, color='DarkOrange', alpha=0.2)
        #plt.plot(instance.periods, hwatermin, 'black', linestyle='-', linewidth=2)
        #plt.plot(instance.periods, hwatermax, 'black', linestyle='-', linewidth=2)

        ax2[n].tick_params(axis='y', colors='DarkOrange')
        ax2[n].yaxis.label.set_color('DarkOrange')
        ax2[n].set_ylim(hwatermin - 2, hwatermax + 2)
        ax2[n].set_yticks([round(hwatermin, 1), round(hwatermax, 1)])

        # mini = np.floor(hwatermin * 2) / 2.
        # maxi = np.floor(hwatermax * 2) / 2. + 0.5
        # if maxi - mini > 2:
        #     ec = int(maxi - mini)
        #     gap = 2
        # else:
        #     ec = int((maxi - mini) / 0.5)
        #     gap = 0.5
        # ax2[n].set_yticks([mini + gap * r for r in range(ec + 1)])
        # ax2[n].set_ylim(mini - gap, maxi + gap)

        if n == len(instance.tanks) - 1:
            axs[n].legend(ncol=2, bbox_to_anchor=(0.67, 6.5))
            ax2[n].legend(ncol=1, bbox_to_anchor=(1.1, 6.5))

    locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
    formatter = mdates.ConciseDateFormatter(locator)
    axs[-1].xaxis.set_major_locator(locator)
    axs[-1].xaxis.set_major_formatter(formatter)
    axs[-1].set_xlim([instance.periods[0], instance.periods[-1]])

    #plt.tight_layout()
    plt.show()

def tanksGratien(instance, qnlp, volnlp, volconv=None):
    opacity = 0.6
    bar_width = 0.0375 * instance.tsinhours()
    box = dict(facecolor='red', pad=5, alpha=0.3)
    plt.figure('Tanks', figsize=(8, len(instance.tanks) * 1.5));
    ax = []

    x = instance.periods[:-1]

    for n, j in enumerate(instance.tanks):
        ax.append(plt.subplot(len(instance.tanks), 1, n + 1))

        inflow = [sum(max(0, qt[a]) for a in instance.inarcs(j))
                  - sum(min(0, qt[a]) for a in instance.outarcs(j))
                  for t, qt in qnlp.items()]
        ouflow = [sum(min(0, qt[a]) for a in instance.inarcs(j))
                   - sum(max(0, qt[a]) for a in instance.outarcs(j))
                   for t, qt in qnlp.items()]
        plt.bar(x, inflow, bar_width, alpha=opacity, color='green', label='inflow in $m^3/h$')
        plt.bar(x, ouflow, bar_width, alpha=opacity, color='blue', label='outflow in $m^3/h$')

        plt.tick_params(axis='x', which='both', labelbottom='off')
        plt.ylabel(j, bbox=box, fontsize=12)
        plt.xlim([instance.periods[0], instance.periods[-1]])
        mini = round(min(ouflow) * 1.2 / 5) * 5
        maxi = round(max(inflow) * 1.2 / 5) * 5
        plt.ylim(mini, maxi)
        ec = 5 * round((maxi - mini) / 20)
        plt.yticks([mini + ec * u for u in range(5)])

        if n == len(instance.tanks) - 1:
            plt.legend(ncol=2, bbox_to_anchor=(0.67, 6.5))

        ax.append(ax[2 * n].twinx())

        hwaternlp = [vol[j] / instance.tanks[j].surface for vol in volnlp]
        plt.plot(instance.periods, hwaternlp, 'DarkOrange', linestyle='-', linewidth=3, label='real water height in $m$')
        if volconv:
            hwatercon = [vol[j] / instance.tanks[j].surface for vol in volconv]
            plt.plot(instance.periods, hwatercon, 'black', linestyle='-', linewidth=3, label='water height in $m$')

        hwatermin = [instance.tanks[j].vmin / instance.tanks[j].surface for t in instance.periods]
        hwatermax = [instance.tanks[j].vmax / instance.tanks[j].surface for t in instance.periods]
        plt.plot(instance.periods, hwatermin, 'black', linestyle='-', linewidth=2)
        plt.plot(instance.periods, hwatermax, 'black', linestyle='-', linewidth=2)

        if n == len(instance.tanks) - 1:
            plt.legend(ncol=1, bbox_to_anchor=(1.1, 6.5))
        plt.tick_params(color='DarkOrange')
        mini = np.floor(instance.tanks[j].vmin / instance.tanks[j].surface * 2) / 2.
        maxi = np.floor(instance.tanks[j].vmax / instance.tanks[j].surface * 2) / 2. + 0.5
        if maxi - mini > 2:
            ec = int(maxi - mini)
            gap = 2
        else:
            ec = int((maxi - mini) / 0.5)
            gap = 0.5
        plt.yticks([mini + gap * r for r in range(ec + 1)])

        for t in ax[2 * n + 1].get_yticklabels():
            t.set_color('DarkOrange')
        plt.ylim(mini - gap, maxi + gap)
        plt.xticks(rotation=20)
        plt.xlim([instance.periods[0], instance.periods[-1]])


    locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
    formatter = mdates.ConciseDateFormatter(locator)
    ax[-1].xaxis.set_major_locator(locator)
    ax[-1].xaxis.set_major_formatter(formatter)
    #ax[-1].set_xlim(x[0], x[-1])

    plt.show()

def pumpsGratien(instance, qnlp, qconvex=None):
    opacity = 0.6
    barwidth = 0.0375 * instance.tsinhours()
    box = dict(facecolor='blue', pad=3, alpha=0.3)
    fig, axs = plt.subplots(len(instance.pumps) + 1, 1, sharex=True, figsize=(8,(len(instance.pumps)+1)*1.5))
#    fig.suptitle(instance.name, fontsize=14)

    x = instance.periods[:-1]

    for n, a in enumerate(instance.pumps):
        qnlpa = [qt[a] for t, qt in qnlp.items()]
        axs[n].bar(x, qnlpa, barwidth, alpha=opacity, color='blue', edgecolor='black', label='real flow in $m^3/h$')

        if qconvex:
            qcona = [qt[a] for t, qt in qconvex.items()]
            axs[n].bar(x, qcona, barwidth, alpha=opacity, color='red', label='flow in $m^3/h$')

        axs[n].set_ylabel(f'({a[0]},{a[1]})', bbox=box, fontsize=10)


    opacity = 0.6
    eleccost = [instance.eleccost(t) for t in instance.horizon()]
    axs[-1].bar(x, eleccost, barwidth, alpha=opacity, color='red', label='elec in 'u"\u20AC"'/MWh')
    axs[-1].set_ylim([min(min(eleccost) * 1.1, -max(eleccost) * 0.1), max(eleccost) * 1.1])

    locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
    formatter = mdates.ConciseDateFormatter(locator)
    axs[-1].xaxis.set_major_locator(locator)
    axs[-1].xaxis.set_major_formatter(formatter)

    plt.tight_layout()

    axs[0].set_title('pump flow in $m^3/h$', fontsize=9)
    axs[-1].set_title('electricity cost in 'u"\u20AC"'/MWh', fontsize=9)
    plt.show()


