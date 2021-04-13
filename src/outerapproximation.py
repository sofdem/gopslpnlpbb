#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 11:07:46 2020

@author: Sophie Demassey, Gratien Bonvin
"""

from math import sqrt
import numpy as np
import pylab as plt


EPSILON = 0.001

def quadval(q, coeff):
    """Value of a quadratic function coeff at q."""
    return coeff[0] + coeff[1] * q + coeff[2] * q**2

def quadtangent(q, coeff):
    """Tangent line of a quadratic function coeff at q: f(q) + f'(q)(x-q)."""
    g = coeff[1] + 2 * coeff[2] * q
    return [quadval(q, coeff) - g * q, g]


def hloss(q, coeff):
    """Value of the quadratic head loss function coeff at q."""
    return coeff[0] * q * abs(q) + coeff[1] * q


def hlosstangent(q, coeff):
    """Tangent line of the head loss function coeff at q: f(q) + f'(q)(x-q)."""
    g = 2 * coeff[0] * abs(q) + coeff[1]
    b = -coeff[0] * abs(q) * q
    return [b, g]

def hlosschord(q1, q2, coeff):
    """Line intersecting the head loss function coeff at q1 and q2."""
    a = (hloss(q1, coeff) - hloss(q2, coeff)) / (q1 - q2)
    b = hloss(q1, coeff) - a * q1
    return [b, a]


def _pipescutabovefrommin(epsilon, qmin, qmax, coeff):
    """Compute tangents to the pipe head loss function phi/coeff progressively from qmin on the concave part (q<=0).

    Tangents (f_i)_[1,n] for (q_i)_[1,n] are defined progressively as:
    q_1 = qmin, q_n = min (qmax, qmax(1-sqrt(2))),
    and q_i = q_{i-1} + 2*sqrt(epsilon/coeff[0]) for all 1<i<n.
    Remark: f_i(q) - phi(q) = epsilon <=> q = qi +/- sqrt(epsilon/coeff[0])
    Note that this may compute one more (wrt epsilon) benficial constraint near 0 (see in comparison with dichotomy)

    Args:
        epsilon (float): maximum distance between the tangent set and the curve.
        qmin (float): minimum flow.
        qmax (float): maximum flow.
        coeff (list): coefficients of the head loss polynomial function.

    Returns:
        cutabove (list): list of the coefficients of the tangent linear function.

    """
    cutabove = []
    qsup = qmax*(1-sqrt(2))
    if qmin >= qsup:
        cutabove.append(hlosschord(qmin, qmax, coeff))
    else:
        q = qmin
        qsup = min(qmax, qsup)
        while q < qsup:
            cutabove.append(hlosstangent(q, coeff))
            print(f"cut at {q}")
            q += 2*sqrt(epsilon / coeff[0])
        cutabove.append(hlosstangent(qsup, coeff))
    return cutabove

def _pipescutbelowfrommax(epsilon, qmin, qmax, coeff):
    """Symmetric of _pipescutabovefrommin on the convex part (q>=0).."""

    cutbelow = []
    qinf = qmin*(1-sqrt(2))
    if qinf >= qmax:
        cutbelow.append(hlosschord(qmin, qmax, coeff))
    else:
        q = qmax
        qinf = max(qmin, qinf)
        while q > qinf:
            cutbelow.append(hlosstangent(q, coeff))
            print(f"cut at {q}")
            q -= 2*sqrt(epsilon / coeff[0])
        cutbelow.append(hlosstangent(qinf, coeff))
    return cutbelow


def _pipescutabovedichotomy(epsilon, qmin, qmax, coeff):
    """Compute tangents to the pipe head loss function phi/coeff by dichotomy.

    Tangents (f_i)_[1,n] for (q_i)_[1,n] are defined by dichotomy over [q_a=q_1, q_b=q_n] with
    q_1 = qmin, q_n = min (qmax, qmax(1-sqrt(2))),
    while f_{i-1}((q_a+q_b)/2) - phi((q_a+q_b)/2) > epsilon.
    Remark: q in argmax_{q_a < q < q_b} (f_i(q) - phi(q))  <=> q = (q_a+q_b)/2 and f_a(q)=f_b(q)

    Args:
        epsilon (float): maximum distance between the tangent set and the curve.
        qmin (float): minimum flow.
        qmax (float): maximum flow.
        coeff (list): coefficients of the head loss polynomial function.

    Returns:
        cutabove (list): list of the coefficients of the tangent linear function.

    """
    cutabove = []
    qsup = qmax*(1-sqrt(2))
    if qmin >= qsup:
        cutabove.append(hlosschord(qmin, qmax, coeff))
    else:
        qsup = min(qmax, qsup)
        linmin = hlosstangent(qmin, coeff)
        cutabove.append(linmin)
        _cutdichotomy(cutabove, epsilon, qmin, linmin, qsup, coeff)
        linsup = hlosstangent(qsup, coeff)
        cutabove.append(linsup)
    return cutabove


def _pipescutbelowdichotomy(epsilon, qmin, qmax, coeff):
    cutbelow = []
    qinf = qmin*(1-sqrt(2))
    if qmax <= qinf:
        cutbelow.append(hlosschord(qmin, qmax, coeff))
    else:
        qinf = max(qmin, qinf)
        lininf = hlosstangent(qinf, coeff)
        cutbelow.append(lininf)
        _cutdichotomy(cutbelow, epsilon, qinf, lininf, qmax, coeff)
        linmax = hlosstangent(qmax, coeff)
        cutbelow.append(linmax)
    return cutbelow

def _cutdichotomy(cuts, epsilon, qmin, linmin, qmax, coeff):
    q = (qmin + qmax)/2
    if (abs(linmin[0] + linmin[1]*q - hloss(q, coeff)) > epsilon):
        lin = hlosstangent(q, coeff)
        _cutdichotomy(cuts, epsilon, qmin, linmin, q, coeff)
        cuts.append(lin)
        print(f"cut at {q}")
        _cutdichotomy(cuts, epsilon, q, lin, qmax, coeff)

def pipecuts(qmin, qmax, coeff, pipe, epsilon=EPSILON, drawgraph=True):
    print(f"OA of pipe {pipe} ******************************************************")
    cutabove = _pipescutabovefrommin(epsilon, qmin, qmax, coeff)
    print(f"{len(cutabove)} cuts above")
    cutbelow = _pipescutbelowfrommax(epsilon, qmin, qmax, coeff)
    print(f"{len(cutbelow)} cuts below")
    if drawgraph:
        _drawOA(qmin, qmax, coeff, cutbelow, cutabove, pipe, 'PIPE')
        plt.show()
    return cutbelow, cutabove


def pipecutsdichotomy(qmin, qmax, coeff, pipe, epsilon=EPSILON, drawgraph=True):
    print(f"OA of pipe {pipe} ******************************************************")
    cutabove = _pipescutabovedichotomy(epsilon, qmin, qmax, coeff)
    print(f"{len(cutabove)} cuts above")
    cutbelow = _pipescutbelowdichotomy(epsilon, qmin, qmax, coeff)
    print(f"{len(cutbelow)} cuts below")
    if drawgraph:
        _drawOA(qmin, qmax, coeff, cutbelow, cutabove, pipe, 'PIPE')
        plt.show()
    return cutbelow, cutabove


def pipecutsgratien(qmin, qmax, coeff, pipe, epsilon=EPSILON, drawgraph=True):
    """Reproduce Gratien's code without optimization for conformity test purpose."""
    print(f"OA of pipe {pipe} ******************************************************")
    cutabove = []
    qinflex = qmax * (1-sqrt(2))
    if qmin >= qinflex:
        cutabove.append(hlosschord(qmin, qmax, coeff))
    else:
        if qmax <= 0:
            cutabove.append(hlosstangent(qmax, coeff))
        else:
            cutabove.append(hlosschord(qmax, qinflex, coeff))

        cutabove.append(hlosstangent(qmin, coeff))
        q = qmin
        qsup = min(qmax, qinflex)  # -qmax/(1+np.sqrt(2)) == qmax * (1-np.sqrt(2))
        while True:
            qprx = q + sqrt(epsilon / coeff[0])
            qpry = 2 * coeff[0] * q * qprx - coeff[0] * q**2
            if qprx >= qsup - sqrt(epsilon / coeff[0]):
                break
            else:
                q = (2 * coeff[0] * qprx + sqrt(4 * coeff[0]**2 * qprx**2 - 4 * coeff[0] * qpry)) \
                    / (2 * coeff[0])
                cutabove.append(hlosstangent(q, coeff))
                print(f"cut at {q}")
    print(f"{len(cutabove)} cuts above")


    cutbelow = []
    qinflex = qmin * (1-sqrt(2))
    if qmax <= qinflex:
        cutbelow.append(hlosschord(qmin, qmax, coeff))
    else:
        if qmin >= 0:
            cutbelow.append(hlosstangent(qmin, coeff))
        else:
            cutbelow.append(hlosschord(qmin, qinflex, coeff))

        cutbelow.append(hlosstangent(qmax, coeff))

        q = qmax
        qinf = max(abs(qmin) / (1 + np.sqrt(2)), qmin)
        while True:
            qprx = q - sqrt(epsilon / coeff[0])
            qpry = -2 * coeff[0] * q * qprx + coeff[0] * q**2
            if qprx <= qinf + sqrt(epsilon / coeff[0]):
                break
            else:
                q = (2 * coeff[0] * qprx - sqrt(4 * coeff[0]**2 * qprx**2 + 4 * coeff[0] * qpry)) \
                    / (2 * coeff[0])
                cutbelow.append(hlosstangent(q, coeff))
                print(f"cut at {q}")
    print(f"{len(cutbelow)} cuts below")


    if drawgraph:
        _drawOA(qmin, qmax, coeff, cutbelow, cutabove, pipe, 'PIPE')
        plt.show()
    return cutbelow, cutabove


def pumpcuts(qmin, qmax, coeff, pump, epsilon=EPSILON, drawgraph=False):
    """Approximation of the quadratic pump function hgain = a + bq + cq^2 by its tangents.

    Compute tangents (f_i)_[1,n] progressively from q_1 = qmax such that the intersection q' of
    f_i and f_{i-1} is at distance epsilon from the curve, i.e:
    f_i(q') = f_{i-1}(q') <=> q' = (q_{i-1}+q_i)/2
    f_i(q') - hgain(q') = eps <=> q' = q_i - sqrt(eps/-c) <=> q_{i-1} = q_i - 2*sqrt(eps/-c).

    Args:
        epsilon (float): maximum distance between the tangent set and the curve.
        qmin (float): minimum flow.
        qmax (float): maximum flow.
        coeff (list): coefficients of the head gain polynomial function.

    Returns:
        cutabove (list): list of the coefficients of the tangent linear function.
    """
    cutabove = [[coeff[0], coeff[1]]]  # tangent at q = 0
    #!!! pump 7F in Richmond has a linear hgain
    if coeff[2] != 0: 
        assert coeff[2] < 0, f'{coeff}'
        q = qmax
        while q >= qmin:
            cutabove.append(quadtangent(q, coeff))
            q -= 2 * sqrt(epsilon / -coeff[2])

    # cutbelow = [[(quadval(qmax, coeff) - coeff[0]) / qmax, coeff[0]]]
    #cutbelow = [[coeff[1] + coeff[2] * qmax, coeff[0]]]
    c0 = quadval(qmin, coeff)
    c1 = (c0 - quadval(qmax, coeff)) / (qmin - qmax)
    c0 -= c1*qmin
    cutbelow = [[c0, c1]]


    if drawgraph:
        _drawOA(qmin, qmax, coeff, cutbelow, cutabove, pump, 'PUMP')
        plt.show()
    return cutbelow, cutabove


def pumpcutsgratien(qmin, qmax, coeff, pump, epsilon=EPSILON, drawgraph=False):
    """Same as pumpcuts() above without the useless computations. """
    cutabove = [[coeff[0], coeff[1]]]  # at qmin
    cutabove.append(quadtangent(qmax, coeff))
    if coeff[2] != 0:
        q = qmax
        while True:
            qprx = q - sqrt(epsilon / abs(coeff[2]))
            qpry = -2 * abs(coeff[2]) * q * qprx + abs(coeff[2]) * q**2
            if qprx <= qmin + sqrt(epsilon / abs(coeff[2])):
                break
            else:
                # !!! to optimize code
                q = (2 * abs(coeff[2]) * qprx
                     - sqrt(4 * abs(coeff[2])**2 * qprx**2 + 4 * abs(coeff[2]) * qpry)) \
                    / (2 * abs(coeff[2]))
                cutabove.append(quadtangent(q, coeff))

    # cutbelow = [[(quadval(qmax, coeff) - coeff[0]) / qmax, coeff[0]]]
    cutbelow = [[coeff[0], (coeff[1] * qmax + coeff[2] * qmax**2) / qmax]]

    if drawgraph:
        _drawOA(qmin, qmax, coeff, cutbelow, cutabove, pump, 'PUMP')
        plt.show()
    return cutbelow, cutabove


def _drawOA(qmin, qmax, coeff, cutbelow, cutabove, title, arctype):
    dq = qmax - qmin
    qs = np.arange(qmin-dq/10.0, qmax+dq/10.0, 0.01)
    if arctype == 'PIPE':
        phi = [hloss(q, coeff) for q in qs]
    if arctype == 'PUMP':
        phi = [coeff[2] * q * q + coeff[1] * q + coeff[0] for q in qs]
    plt.title(f'{arctype} ({title[0]}, {title[1]})')
    plt.plot(qs, phi, linewidth=2, color='r')
    plt.axvline(x=qmin, color='k', linestyle='-', linewidth=2)
    plt.axvline(x=qmax, color='k', linestyle='-', linewidth=2)

    for c in cutbelow:
        cuts = [c[1] * q + c[0] for q in qs]
        plt.plot(qs, cuts, color='b', linestyle='-')
    for c in cutabove:
        cuts = [c[1] * q + c[0] for q in qs]
        plt.plot(qs, cuts, color='DarkOrange', linestyle='-')
    plt.axhline(y=0.0, color='k', linestyle='--')
    plt.axvline(x=0.0, color='k', linestyle='--')
    plt.xlim([qmin-dq/10.0, qmax+dq/10.0])
    return None



def test():
    """Test OA."""
    cbelow, cabove = pumpcuts(0, 179, [101.055222, 0.0, -0.000835], 'TestPump', epsilon=1e-1, drawgraph=True)
    cbelow, cabove = pipecuts(0, 179, [0.000835, 0], 'TestPipe', epsilon=1e-1, drawgraph=True)

