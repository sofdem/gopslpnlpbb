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

def quadval(q, coeff):
    """Value of a quadratic function coeff at q."""
    return coeff[0] + coeff[1] * q + coeff[2] * q ** 2


def quadtangent(q, coeff):
    """Tangent line of a quadratic function coeff at q: f(q) + f'(q)(x-q)."""
    c1 = coeff[1] + 2 * coeff[2] * q
    c0 = quadval(q, coeff) - c1 * q
    return [c0, c1]


def quadchord(q1, q2, coeff):
    """Line intersecting a quadratic function coeff at q1 and q2."""
    c1 = coeff[2] * (q1 + q2) + coeff[1]
    c0 = coeff[0] - coeff[2] * q1 * q2
    return [c0, c1]


def hlossval(q, coeff):
    """Value of the quadratic head loss function coeff at q."""
    return coeff[0] * q * abs(q) + coeff[1] * q


def hlosstangent(q, coeff):
    """Tangent line of the head loss function coeff at q: f(q) + f'(q)(x-q)."""
    c1 = 2 * coeff[0] * abs(q) + coeff[1]
    c0 = -coeff[0] * abs(q) * q
    return [c0, c1]


def hlosschord(q1, q2, coeff):
    """Line intersecting a quadratic function coeff at q1 and q2."""
    assert q1 < q2
    #    return chord(q1, q2, hlossval, coeff)
    c1 = coeff[0] * (q1 + q2) + coeff[1]
    c0 = -coeff[0] * abs(q1) * q2 if (q1 >= 0 or q2 <= 0) else coeff[0] * q1 * q2 * (q1 + q2) / (q1 - q2)
    return [c0, c1]


def chord(q1, q2, fval, coeff):
    """Line intersecting a function fval/coeff at q1 and q2."""
    c0 = fval(q1, coeff)
    c1 = (c0 - fval(q2, coeff)) / (q1 - q2)
    c0 -= c1 * q1
    return [c0, c1]


# outer approximation of the pipe/pump head-flow relations

def pipecuts(qmin, qmax, coeff, pipename, epsilon, drawgraph=False):
    # print(f"OA of pipe {pipe} ******************************************************")
    cutabove = _pipescutabovefrommin(epsilon, qmin, qmax, coeff)
    # print(f"{len(cutabove)} cuts above")
    cutbelow = _pipescutbelowfrommax(epsilon, qmin, qmax, coeff)
    # print(f"{len(cutbelow)} cuts below")
    if drawgraph:
        _draw_oa(qmin, qmax, coeff, cutbelow, cutabove, pipename, 'PIPE')
        plt.show()
    return cutbelow, cutabove


def _pipescutabovefrommin(epsilon, qmin, qmax, coeff):
    """Compute tangents to the pipe head loss function phi/coeff progressively from qmin on the concave part (q<=0).

    Tangents (f_i)_[1,n] for (q_i)_[1,n] are defined progressively as:
    q_1 = qmin, q_n = min (qmax, qmax(1-sqrt(2))),
    and q_i = q_{i-1} + 2*sqrt(epsilon/coeff[0]) for all 1<i<n.
    Remark: f_i(q) - phi(q) = epsilon <=> q = qi +/- sqrt(epsilon/coeff[0])
    Note that this may compute one more (wrt epsilon) beneficial constraint near 0 (see in comparison with dichotomy)

    Args:
        epsilon (float): maximum distance between the tangent set and the curve.
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
            q += 2 * sqrt(epsilon / coeff[0])
        cutabove.append(hlosstangent(qsup, coeff))
    return cutabove


def _pipescutbelowfrommax(epsilon, qmin, qmax, coeff):
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
            q -= 2 * sqrt(epsilon / coeff[0])
        cutbelow.append(hlosstangent(qinf, coeff))
    return cutbelow


def pumpcuts(qmin, qmax, coeff, pumpname, epsilon, drawgraph=False):
    """Approximation of the quadratic pump function hgain = aq^2 + bq + c by its tangents.

    Compute tangents (f_i)_[1,n] progressively from q_1 = qmax such that the intersection q' of
    f_i and f_{i-1} is at distance epsilon from the curve, i.e:
    f_i(q') = f_{i-1}(q') <=> q' = (q_{i-1}+q_i)/2
    f_i(q') - hgain(q') = eps <=> q' = q_i - sqrt(eps/-a) <=> q_{i-1} = q_i - 2*sqrt(eps/-a).
    Note that this may compute one more (wrt epsilon) constraint near qmin.


    Args:
        epsilon (float): maximum distance between the tangent set and the curve.
        qmin (float): minimum flow.
        qmax (float): maximum flow.
        coeff (list): coefficients of the head gain polynomial function.
        pumpname (string): pump name to draw.
        drawgraph (bool): draw the graph of the cuts or not.

    Returns:
        cutabove (list): list of the coefficients of the tangent linear function.
    """
    if coeff[2] == 0:
        print(f"linear gain for pump {pumpname}")
        return [coeff], [coeff]

    assert coeff[2] < 0 <= qmin < qmax, f'{coeff}'
    q = qmax
    cutabove = [quadtangent(qmin, coeff)]
    while q > qmin:
        cutabove.append(quadtangent(q, coeff))
        q -= 2 * sqrt(epsilon / -coeff[2])

    cutbelow = [quadchord(qmin, qmax, coeff)]

    if drawgraph:
        _draw_oa(qmin, qmax, coeff, cutbelow, cutabove, pumpname, 'PUMP')
        plt.show()
    return cutbelow, cutabove


# test and see


def _draw_oa(qmin, qmax, coeff, cutbelow, cutabove, title, arctype, points=None):
    dq = qmax - qmin
    qs = np.arange(qmin - dq / 10.0, qmax + dq / 10.0, 0.01)
    fval = hlossval if arctype == 'PIPE' else quadval
    phi = [fval(q, coeff) for q in qs]
    plt.figure()
    plt.title(f'{arctype} {title}')
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
    pumpcuts(0, 179, [101.055222, 0.0, -0.000835], 'TestPump', epsilon=1e-1, drawgraph=True)
    pipecuts(0, 179, [0.000835, 0], 'TestPipe', epsilon=1e-1, drawgraph=True)
