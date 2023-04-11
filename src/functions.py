#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 11:07:46 2020

@author: Sophie Demassey
"""
import math
from math import sqrt
import numpy as np
import pylab as plt


class FuncPol:
    """
    c  : antisymmetric quadratic function  f(q) = c2.q|q| + c1.q + c0
    """
    def __init__(self, c_):
        self.c = (c_[0], c_[1], c_[2])

    def __str__(self):
        return f'[{self.c[0]}, {self.c[1]}, {self.c[2]}]'

    def coeff(self):
        return self.c

    def value(self, q: float):
        """Value at q: f(q) = c0 + c1.q + c2.q|q|."""
        return self.c[0] + self.c[1] * q + self.c[2] * q * abs(q)

    def tangentfunc(self, q: float) -> tuple:
        """Tangent line at q: g(x) = f(q) + f'(q)(x-q) = (c0 - c2.q|q|) + (2c2|q| + c1)x """
        return self.c[0] - self.c[2] * abs(q) * q, 2 * self.c[2] * abs(q) + self.c[1]

    def chordfunc(self, q1, q2):
        """Line intersecting the function at q1 and q2: g(x) = f(q1) + s.(x-q1) with s=(f(q1)-f(q2))/(q1-q2) """
        f1 = self.value(q1)
        s = (f1 - self.value(q2)) / (q1 - q2)
        return f1 - s * q1, s

    def primitivevalue(self, q):
        """Value of the primitive at q: F(q) = c0.q + c1.q^2/2 + c2.|q|.q^2/3 """
        return self.c[0] * q + self.c[1] * q * q / 2 + self.c[2] * q * q * abs(q) / 3

    def inversevalue(self, x):
        """Value of the inverse function at x: f^{-1}(x) = ."""
        sgn = -1 if self.c[0] > x else 1
        return sgn * (math.sqrt(self.c[1]*self.c[1] + 4 * self.c[2] * (x - self.c[0]))
                      - self.c[1]) / (2 * self.c[2])

    def gval(self, q, x):
        """Value of the duality function g at (q, x)."""
        q2 = self.inversevalue(x)
        return self.primitivevalue(q) - self.primitivevalue(q2) + x * q2

    def islinear(self):
        return self.c[2] == 0

    def overestimatepwlfunc(self, oagap: float, qmin: float, qmax: float):
        """Compute an overestimate PWL function over [qmin, qmax] at a distance less than oagap.
        Generate all tangent lines on the concave part at q1,...,qN
        from q1=qmin to qN=min(qmax, qmax(1-sqrt(2))) included i.e. the last one passes through qmax
        with a constant step of 2 * sqrt(oagap / c2), as: f_i(q) - phi(q) = oagap <=> q = qi +/- step/2,
        (it may compute one more constraint at qN i.e. near 0)
        Args:
            oagap (float): maximum distance of the estimate.
            qmin (float): minimum bound.
            qmax (float): maximum bound.

        Returns:
            cutabove (list): list of the tangent lines.

        """
        qsup = qmax * (1 - sqrt(2))
        if qmin >= qsup:
            return [self.chordfunc(qmin, qmax)]

        qsup = min(qmax, qsup)
        step = 2 * sqrt(oagap / self.c[2])
        nb = math.ceil((qsup-qmin)/step)
        cutabove = [self.tangentfunc(qmin + k*step) for k in range(nb)]
        cutabove.append(self.tangentfunc(qsup))
        return cutabove

    def underestimatepwlfunc(self, oagap: float, qmin: float, qmax: float):
        """Compute an underestimate PWL function over [qmin, qmax] at a distance less than oagap.
        Generate all tangent lines on the convex part at q1,...,qN
        from q1=qmax to qN=max(qmin, qmin(1-sqrt(2))) included i.e. the last one passes through qmin
        with a constant step of -2 * sqrt(oagap / c2), as: phi(q) - f_i(q) = oagap <=> q = qi +/- step/2,
        (it computes one more constraint at qN i.e.near 0)
        Args:
            oagap (float): maximum distance of the estimate.
            qmin (float): minimum bound.
            qmax (float): maximum bound.

        Returns:
            cutbelow (list): list of the tangent lines.

        """
        qinf = qmin * (1 - sqrt(2))
        if qmax <= qinf:
            return [self.chordfunc(qmin, qmax)]

        qinf = max(qmin, qinf)
        step = 2 * sqrt(oagap / self.c[2])
        nb = math.ceil((qmax - qinf) / step)
        cutbelow = [self.tangentfunc(qmax - k * step) for k in range(nb)]
        cutbelow.append(self.tangentfunc(qinf))
        return cutbelow

    def drawoa(self, qmin: float, qmax: float, cutbelow: list, cutabove: list, title: str, points=None):
        dq = qmax - qmin
        qs = np.arange(qmin - dq / 10.0, qmax + dq / 10.0, 0.01)
        phi = [self.value(q) for q in qs]

        plt.figure()
        plt.title(f'{title}')
        plt.plot(qs, phi, linewidth=2, color='r')
        plt.axvline(x=qmin, color='k', linestyle='-', linewidth=2)
        plt.axvline(x=qmax, color='k', linestyle='-', linewidth=2)

        for c in cutbelow:
            plt.plot(qs, [c[1] * q + c[0] for q in qs], color='b', linestyle='-')
        for c in cutabove:
            plt.plot(qs, [c[1] * q + c[0] for q in qs], color='DarkOrange', linestyle='-')
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

    def testoa(self, oagap, qmin, qmax, title):
        cutbelow = self.underestimatepwlfunc(oagap, qmin, qmax)
        cutabove = self.underestimatepwlfunc(oagap, qmin, qmax)
        self.drawoa(oagap, qmin, qmax, cutbelow, cutabove, title)


def test():
    """Test OA."""
    oagap = 0.01
    qmin = 0
    qmax = 179
    FuncPol((101.055222, 0.0, -0.000835)).testoa(oagap, qmin, qmax, 'TestPump')
    FuncPol((0.000835, 0)).testoa(oagap, qmin, qmax, 'TestPipe')


