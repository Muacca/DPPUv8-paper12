"""Two-band Nil transport, direct propagators and endpoint comparisons."""
from argparse import ArgumentParser
from itertools import product
from math import pi, sqrt, cosh, sinh, exp
from pathlib import Path
import json
import numpy as np
from scipy.integrate import solve_ivp

INTERVALS = [(-.5, 0.), (0., 1.5)]


def background(s, charge, holst):
    h = sqrt((1 + holst**2) * charge**2 + 1 / 12)
    d = sqrt(h*h + .25)
    E = exp(2*h*s)
    Y = cosh(2*d*s) - h/d*sinh(2*d*s)
    v = (h*cosh(2*d*s) - d*sinh(2*d*s))/Y
    a2 = E*E*Y
    return E, a2, 1/Y, a2/sqrt(Y), 2*(v-h)


def propagator(field, interval, count):
    # SU(2) evolution of column one; determinant and orthogonality fix column two.
    def rhs(s, flat):
        x, z = field(s)
        p, q = flat.reshape(2, count)
        return -1j*np.r_[z*p+x*q, x*p-z*q]
    y = np.r_[np.ones(count), np.zeros(count)].astype(complex)
    sol = solve_ivp(rhs, interval, y, method="DOP853", rtol=1.5e-12, atol=8e-15)
    assert sol.success
    p, q = sol.y[:, -1].reshape(2, count)
    drift = float(np.max(abs(abs(p)**2+abs(q)**2-1)))
    if drift > 3e-9:
        raise ArithmeticError(drift)
    return p, q, drift


def transitions(p, q, first, last):
    # Project initially negative spinors onto final positive spinors.
    def plus(axis):
        x, y, z = axis
        ang = np.arctan2(np.hypot(x,y), z)
        phase = np.exp(1j*np.arctan2(y, x))
        return np.cos(ang/2), phase*np.sin(ang/2)
    a, b = plus(first)
    c, d = plus(last)
    i, j = -b.conj(), a
    return abs(c*(p*i-q.conj()*j) + d.conj()*(q*i+p.conj()*j))**2


def paired(charge, holst, interval, dense):
    labels = sorted(set(product(range(1, 13), repeat=2)) |
                    {(k, k) for k in (16,32,64,128)} if dense else
                    {(1,1),(1,3),(2,2),(4,4),(12,12)})
    nu = np.array([k/(2*pi) for k, n in labels])
    landau = np.array([sqrt(2*n*k/(2*pi)) for k, n in labels])
    def field(s):
        E,a2,_,_,_ = background(s,charge,holst)
        return landau*E,nu*a2
    def axes(s):
        x,z = field(s)
        omega = np.hypot(x,z)
        shear = background(s,charge,holst)[-1]
        angle_dot = shear*x*z/omega**2
        return (x, np.zeros_like(x), z), (x, -angle_dot/2, z)
    p,q,drift = propagator(field, interval, len(labels))
    ini, end = [axes(s) for s in interval]
    b0,b1 = [transitions(p,q,ini[j],end[j]) for j in (0,1)]
    rows=[dict(k=k,n=n,P0=x,P1=y) for (k,n),x,y in zip(labels,b0,b1)]
    return dict(u=charge,h=holst,I=interval,unitarity=drift,rows=rows,
                square_sums=[dict(N=N,S0=sum(2*r["k"]*r["P0"] for r in rows if max(r["k"],r["n"])<=N),
                                  S1=sum(2*r["k"]*r["P1"] for r in rows if max(r["k"],r["n"])<=N))
                             for N in (3,6,9,12)] if dense else [])


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results_new")/"nil_paired.json")
    target = parser.parse_args().output
    cases = [paired(charge, holst, interval, charge == .375 and holst == 0)
             for charge, holst, interval in product((.125,.375), (0,1), INTERVALS)]
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(cases, indent=2)+"\n", encoding="utf-8")
    print(sum(len(case["rows"]) for case in cases), "two-band mode samples")