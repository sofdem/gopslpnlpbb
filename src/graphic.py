#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 12:58:58 2020

@author: sofdem
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# import numpy as np

def pumps(instance, qnlp):
    pumps_step(instance, qnlp)
    pumps_bar(instance, qnlp)


# pb with this implementation mixing bar['post']+step subplots: the bar for the last time does not show up
def pumps_bar(instance, qnlp):
    fig, axs = plt.subplots(nrows=len(instance.pumps), ncols=1, sharex='all', figsize=(8, (len(instance.pumps)) * 1.5))
    #    fig.suptitle(instance.name, fontsize=14)

    x = instance.periods  # [:-1]
    bar_width = x[1] - x[0]
    qmax = max(p.qmax() for p in instance.pumps.values()) * 1.1

    cm = plt.cm.get_cmap('OrRd')
    eleccost = [instance.eleccost(t) for t in instance.horizon()]
    eleccost.append(eleccost[-1])
    ecmin = 0  # min(eleccost)/2
    ecspan = max(eleccost) - ecmin
    ecm = [cm((cost - ecmin) / ecspan) for cost in eleccost]

    for n, (a, p) in enumerate(instance.pumps.items()):
        qnlpa = [qt[a] for t, qt in qnlp.items()]
        qnlpa.append(qnlpa[-1])
        # pb when mixing bar['post']+step subplots: last step will not show up
        axs[n].bar(x, qnlpa, bar_width, align='edge', color=ecm, edgecolor='white', label='real flow in $m^3/h$')
        # axs[n].step(x, qnlpa, where='post', color=cm(0.8), label='real flow in $m^3/h$')
        axs[n].set_ylabel(f'({a[0]},{a[1]})', bbox=dict(fc='#557f2d', alpha=0.2, pad=3), fontsize=10)
        axs[n].set_ylim(0, qmax)
        axs[n].fill_between(x, p.qmin(), p.qmax(), color='#7f6d5f', alpha=0.1)

    # axs[-1].step(x, eleccost, where='post', color='#7f6d5f', label='elec in 'u"\u20AC"'/MWh')
    # axs[-1].set_ylim(min(0, min(eleccost)), max(eleccost) * 1.1)

    locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
    formatter = mdates.ConciseDateFormatter(locator)
    axs[-1].xaxis.set_major_locator(locator)
    axs[-1].xaxis.set_major_formatter(formatter)
    axs[-1].set_xlim(x[0], instance.periods[-1])

    plt.tight_layout()

    axs[0].set_title('pump flow in $m^3/h$', fontsize=9)
    #    axs[-1].set_title('electricity cost in 'u"\u20AC"'/MWh', fontsize=9)

    plt.subplots_adjust(bottom=0.1, right=0.95, top=0.9)
    cax = plt.axes([0.975, 0.1, 0.025, 0.8])

    sm = plt.cm.ScalarMappable(cmap=cm, norm=plt.Normalize(ecmin, ecmin + ecspan))
    sm.set_array(None)
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label('electricity cost in 'u"\u20AC"'/MWh', rotation=270, labelpad=25)

    plt.show()


def pumps_step(instance, qnlp):
    fig, axs = plt.subplots(nrows=len(instance.pumps) + 1, ncols=1, sharex='all',
                            figsize=(8, (len(instance.pumps) + 1) * 1.5))
    #    fig.suptitle(instance.name, fontsize=14)

    x = instance.periods
    qmax = max(p.qmax() for p in instance.pumps.values()) * 1.1

    for n, (a, p) in enumerate(instance.pumps.items()):
        qnlpa = [qt[a] for t, qt in qnlp.items()]
        qnlpa.append(qnlpa[-1])
        axs[n].step(x, qnlpa, where='post', color='#557f2d', label='real flow in $m^3/h$')
        axs[n].set_ylabel(f'({a[0]},{a[1]})', bbox=dict(fc='#557f2d', alpha=0.2, pad=3), fontsize=10)
        axs[n].set_ylim(0, qmax)
        axs[n].fill_between(x, p.qmin(), p.qmax(), color='#557f2d', alpha=0.2)

    eleccost = [instance.eleccost(t) for t in instance.horizon()]
    eleccost.append(eleccost[-1])
    axs[-1].step(x, eleccost, where='post', color='#7f6d5f', label='elec in 'u"\u20AC"'/MWh')
    axs[-1].set_ylim(min(0, min(eleccost)), max(eleccost) * 1.1)

    locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
    formatter = mdates.ConciseDateFormatter(locator)
    axs[-1].xaxis.set_major_locator(locator)
    axs[-1].xaxis.set_major_formatter(formatter)
    axs[-1].set_xlim([instance.periods[0], instance.periods[-1]])

    plt.tight_layout()

    axs[0].set_title('pump flow in $m^3/h$', fontsize=9)
    axs[-1].set_title('electricity cost in 'u"\u20AC"'/MWh', fontsize=9)
    plt.show()


def tanks(instance, qnlp, volnlp):
    fig, axs = plt.subplots(nrows=len(instance.tanks), ncols=1, sharex='all', figsize=(8, len(instance.tanks) * 1.5))
    if len(instance.tanks) == 1:
        axs = [axs]
    ax2 = []
    x = instance.periods[:-1]
    bar_width = x[1] - x[0]

    for n, (j, tk) in enumerate(instance.tanks.items()):
        inflow = [sum(max(0, qt[a]) for a in instance.inarcs(j)) - sum(min(0, qt[a]) for a in instance.outarcs(j)) for
                  t, qt in qnlp.items()]
        ouflow = [sum(min(0, qt[a]) for a in instance.inarcs(j)) - sum(max(0, qt[a]) for a in instance.outarcs(j)) for
                  t, qt in qnlp.items()]
        axs[n].bar(x, inflow, bar_width, alpha=0.3, align='edge', color='#557f2d', label='inflow in $m^3/h$')
        axs[n].bar(x, ouflow, bar_width, alpha=0.3, align='edge', color='#7f6d5f', label='outflow in $m^3/h$')

        # plt.tick_params(axis='x', which='both', labelbottom='off')
        axs[n].set_ylabel(j, bbox=dict(fc='DarkOrange', pad=3, alpha=0.2), fontsize=12)
        # plt.xlim([instance.periods[0], instance.periods[-1]])

        mini = min(ouflow)
        maxi = max(inflow)
        axs[n].set_ylim(mini * 1.2, maxi * 1.2)
        axs[n].set_yticks([round(mini), 0, round(maxi)])

        ax2.append(axs[n].twinx())
        hwaternlp = [vol[j] / tk.surface for vol in volnlp]
        ax2[n].plot(instance.periods, hwaternlp, 'DarkOrange', linestyle='-', linewidth=3,
                    label='real water height in $m$')

        hwatermin = tk.vmin / tk.surface
        hwatermax = tk.vmax / tk.surface
        ax2[n].fill_between(instance.periods, hwatermin, hwatermax, color='DarkOrange', alpha=0.2)
        # plt.plot(instance.periods, hwatermin, 'black', linestyle='-', linewidth=2)
        # plt.plot(instance.periods, hwatermax, 'black', linestyle='-', linewidth=2)

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

    # plt.tight_layout()
    plt.show()


def progress(trace, title=""):
    if not trace:
        return
    labels = ['time', 'node', 'lb', 'ub', 'cvx', 'sol']
    vals = {labels[k]: [tr[k] for tr in trace] for k in range(len(labels))}
    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.plot(vals['time'], vals['cvx'], '.')
    ax.plot(vals['time'], vals['sol'], 'o')
    ax.plot(vals['time'], vals['lb'], '-')
    ax.plot(vals['time'], vals['ub'], '-')
    fig.tight_layout()
    return plt
