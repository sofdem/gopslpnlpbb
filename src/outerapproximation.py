#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 11:07:46 2020

@author: Sophie Demassey, Gratien Bonvin
"""

from math import sqrt
import numpy as np
import pylab as plt


# analytic functions

def hlossval(q, coeff):
    """Value of the quadratic head loss function coeff at q."""
    return coeff[2] * q * abs(q) + coeff[1] * q + coeff[0]


def hlosstangent(q, coeff):
    """Tangent line of the head loss function coeff at q: f(q) + f'(q)(x-q)."""
    return [coeff[0] - coeff[2] * abs(q) * q, 2 * coeff[2] * abs(q) + coeff[1]]


def hlosschord(q1, q2, coeff):
    """Line intersecting a quadratic function coeff at q1 and q2."""
    c0 = hlossval(q1, coeff)
    c1 = (c0 - hlossval(q2, coeff)) / (q1 - q2)
    return [c0 - c1 * q1, c1]


def hlossoa(qmin, qmax, coeff, arcname, oagap, drawgraph=False):
    if coeff[2] == 0:
        print(f'linear hloss for arc {arcname}: {coeff}')
        cut = [[coeff[0], coeff[1]]]
        return cut, cut
    assert coeff[2] > 0
    cutabove = _cutabovefrommin(oagap, qmin, qmax, coeff)
    cutbelow = _cutbelowfrommax(oagap, qmin, qmax, coeff)
    if drawgraph:
        _draw_oa(qmin, qmax, coeff, cutbelow, cutabove, arcname)
        plt.show()
    return cutbelow, cutabove


def _cutabovefrommin(oagap, qmin, qmax, coeff):
    """Compute tangents to the pipe head loss function phi/coeff progressively from qmin on the concave part (q<=0).

    Tangents (f_i)_[1,n] for (q_i)_[1,n] are defined progressively as:
    q_1 = qmin, q_n = min (qmax, qmax(1-sqrt(2))),
    and q_i = q_{i-1} + 2*sqrt(oagap/c2) for all 1<i<n.
    Remark: f_i(q) - phi(q) = oagap <=> q = qi +/- sqrt(oagap/c2)
    Note that this may compute one more (wrt oagap) beneficial constraint near 0 (see in comparison with dichotomy)

    Args:
        oagap (float): maximum distance between the tangent set and the curve.
        qmin (float): minimum flow.
        qmax (float): maximum flow.
        coeff (list): coefficients of the head loss polynomial function.

    Returns:
        cutabove (list): list of the coefficients of the tangent linear function.

    """
    cutabove = []
    qsup = qmax * (1 - sqrt(2))
    if qmin >= qsup:
        cutabove.append(hlosschord(qmin, qmax, coeff))
    else:
        q = qmin
        qsup = min(qmax, qsup)
        while q < qsup:
            cutabove.append(hlosstangent(q, coeff))
            # print(f"cut at {q}")
            q += 2 * sqrt(oagap / coeff[2])
        cutabove.append(hlosstangent(qsup, coeff))
    return cutabove


def _cutbelowfrommax(oagap, qmin, qmax, coeff):
    """Symmetric of _pipescutabovefrommin on the convex part (q>=0).."""

    cutbelow = []
    qinf = qmin * (1 - sqrt(2))
    if qinf >= qmax:
        cutbelow.append(hlosschord(qmin, qmax, coeff))
    else:
        q = qmax
        qinf = max(qmin, qinf)
        while q > qinf:
            cutbelow.append(hlosstangent(q, coeff))
            # print(f"cut at {q}")
            q -= 2 * sqrt(oagap / coeff[2])
        cutbelow.append(hlosstangent(qinf, coeff))
    return cutbelow


def _draw_oa(qmin, qmax, coeff, cutbelow, cutabove, title, points=None):
    dq = qmax - qmin
    qs = np.arange(qmin - dq / 10.0, qmax + dq / 10.0, 0.01)
    phi = [hlossval(q, coeff) for q in qs]

    plt.figure()
    plt.title(f'{title}')
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

    for p in (points or []):
        plt.plot(p['x'], p['y'], p.get('fmt', 'r+'))
        if p['x'] < qmin:
            qmin = p['x']
            print("point is out of range: {p['x']} < {qmin}")
        if p['x'] > qmax:
            qmax = p['x']
            print("point is out of range: {p['x']} > {qmax}")

    plt.xlim([qmin - dq / 10.0, qmax + dq / 10.0])
    plt.ylim([min(phi), max(phi)])

    plt.show()


def test():
    """Test OA."""
    hlossoa(0, 179, [101.055222, 0.0, -0.000835], 'TestPump', oagap=0.01, drawgraph=True)
    hlossoa(0, 179, [0.000835, 0], 'TestPipe', oagap=0.01, drawgraph=True)
