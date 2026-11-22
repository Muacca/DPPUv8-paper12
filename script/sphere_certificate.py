"""Validated finite sums, all-mode bounds, and direct two-level ODE ray checks."""
import json
import math
from pathlib import Path

import numpy as np
from flint import acb, arb, arb_series, ctx
from scipy.integrate import solve_ivp


ctx.prec = 180
ctx.cap = 40
Y = (3 / (4 * (arb(3)/100).exp() - 1)).sqrt()
Z = Y * (arb(3)/200).exp()
D = arb("1.2146378733763", "1e-12")
def gap(d):
    return Z/(Z*d).cosh() - Y/(Y*d).cosh()
assert gap(D.lower()) > 0 and gap(D.upper()) < 0
# Bisection avoids dependency inflation when multiplying many propagators.
lo, hi = D.lower(), D.upper()
for _ in range(95):
    m = (lo+hi)/2
    if gap(m) > 0:
        lo = m
    else:
        assert gap(m) < 0
        hi = m
D = lo.union(hi)
R = Z/(Z*D).cosh()
assert 3 < arb(1)/2+3/(2*R) < 4 and D < arb('1.22') and Z < 1

def bounds(a):
    # Outward conversion for the JSON witness.
    return [math.nextafter(float(a.lower()), -math.inf),
            math.nextafter(float(a.upper()), math.inf)]

def cosh_series(x):
    e = x.exp()
    return (e+1/e)/2

def fields(s, mu, eta):
    # Jets in real time. Factor mu*sqrt(1-mu^2) before differentiation.
    # Stable evaluation of z sech(2zs)/(y sech(2ys)) for close y,z.
    ez = (2*(Z-Y)*s).exp()
    ey = (2*Y*s).exp()
    cy, sy = (ey+1/ey)/2, (ey-1/ey)/2
    e = (Z/Y)/((ez+1/ez)/2 + (sy/cy)*(ez-1/ez)/2)
    v = Y*e/cy
    B = 1 + mu**2*(e**2-1)
    lam = 2*v*B.sqrt()
    F = -e.derivative()/(2*B*lam)
    L2 = mu**2*(1-mu**2)
    omega1 = lam*(1+eta**2*L2*F**2).sqrt()
    h1 = F.derivative()/(2*(1+eta**2*L2*F**2))
    V = h1/omega1
    h2 = -V.derivative()/(2*(1+eta**4*L2*V**2))
    omega2 = (omega1**2+eta**4*L2*h1**2).sqrt()
    return F, V, h2, h2/omega2

def endpoint(k, mu, side):
    ctx.cap = 5
    s = arb_series([side*D/2,1])
    F,V,_,_ = fields(s,mu,1/k)
    L = (1-mu*mu).sqrt()
    a1 = (mu*L*F[0]/k).atan()
    a2 = (-mu*L*V[0]/k**2).atan()
    # R_y(theta), with cos(theta)=mu at the endpoint.
    c, d = ((1+mu)/2).sqrt(), ((1-mu)/2).sqrt()
    i = acb(0,1)
    sn,cs = (a1/2).sin_cos()
    U = [[c*cs+d*i*sn,d*(-cs)-c*i*sn],
         [d*cs-c*i*sn,c*cs-d*i*sn]]
    sn2,cs2 = (a2/2).sin_cos()
    V = [[row[0]*cs2+row[1]*sn2,-row[0]*sn2+row[1]*cs2] for row in U]
    return U,V

def ray(k,mu):
    initial,final = endpoint(arb(k),arb(mu),-1),endpoint(arb(k),arb(mu),1)
    y,z,d = map(float,(Y,Z,D))
    def rhs(s,vw):
        v = z/math.cosh(2*z*s)
        x,zcomp = 2*k*math.sqrt(1-mu**2)*v,2*k*mu*v*v*math.cosh(2*y*s)/y
        a,b=vw
        return np.array([-1j*(zcomp*a+x*b),-1j*(x*a-zcomp*b)])
    probs=[]
    for tol in (2e-11,3e-13):
        sol=solve_ivp(rhs,(-d/2,d/2),[1+0j,0j],method='DOP853',rtol=tol,atol=tol/100)
        a,b = sol.y[:,-1]
        assert sol.success and abs(abs(a)**2+abs(b)**2-1)<2e-8
        U=np.array([[a,-b.conjugate()],[b,a.conjugate()]])
        rows=[]
        for left,right in zip(initial,final):
            vi=[complex(row[1]) for row in left]
            vf=[complex(row[0]) for row in right]
            rows.append(float(abs(np.vdot(vf,U@vi))))
        probs.append(rows)
    return dict(k=k,M=k*mu,amplitudes=probs[-1],
                weighted=[k*x*x for x in probs[-1]],
                refinement=max(abs(a-b) for a,b in zip(*probs)),
                scaled_order2=probs[-1][1]*k**3/(1-mu**2)**.5)

def local_fields(cells, degree):
    ctx.cap=degree
    step=D/cells
    local=[]
    for cell in range(cells):
        s=arb_series([-D/2+cell*step,step])
        v=Z/cosh_series(2*Z*s)
        vu=v**2*cosh_series(2*Y*s)/Y
        local.append(([step*2*v[n] for n in range(degree)],
                      [step*2*vu[n] for n in range(degree)]))
    return local

def finite_sum(K, cells=64, degree=31):
    local=local_fields(cells,degree)
    # On complex disks of radius 1/5 around each sampled real start:
    # |cosh(2*z*s)| >= cos(2*z/5); |cosh(2*y*s)| <= cosh(2*y*(D/2+1/5)).
    rho=arb(1)/5
    vbound=Z/(2*Z*rho).cos()
    hzbound=vbound**2*(2*Y*(D/2+rho)).cosh()/Y
    hmax=2*(vbound+hzbound)
    x=D/(cells*rho)
    totals=[arb(0),arb(0)]
    energies=[arb(0),arb(0)]
    records=[]
    for j2 in range(4,2*K+1):
        k=arb(j2)/2
        # Cauchy in operator norm plus the error of a product of almost unitaries.
        err=(rho*k*hmax).exp()*x**(degree+1)/(1-x)
        err_total=((1+err)**cells-1).abs_upper()
        for m2 in range(2 if j2%2==0 else 1,j2,2):
            mu=arb(m2)/j2
            L=(1-mu**2).sqrt()
            a,b=acb(1),acb(0)
            I=acb(0,1)
            for ax,az in local:
                hx=[k*L*t for t in ax]
                hz=[k*mu*t for t in az]
                aa,bb=[a],[b]
                for n in range(1,degree+1):
                    a=-I*sum(hz[j]*aa[-1-j]+hx[j]*bb[-1-j] for j in range(n))/n
                    b=-I*sum(hx[j]*aa[-1-j]-hz[j]*bb[-1-j] for j in range(n))/n
                    aa.append(a); bb.append(b)
                a,b=sum(aa),sum(bb)
            left,right=endpoint(k,mu,-1),endpoint(k,mu,1)
            pr=[]
            for order,(v,w) in enumerate(zip(left,right)):
                l0,l1=v[0][1],v[1][1]
                f0,f1=w[0][0],w[1][0]
                projection=f0.conjugate()*(a*l0-b.conjugate()*l1)+f1.conjugate()*(b*l0+a.conjugate()*l1)
                # Both columns have the SU(2) structure even for the Taylor polynomial.
                amp=abs(projection)+arb(0,err_total)
                assert amp>0
                p=amp**2
                totals[order]+=8*k*p
                energies[order]+=4*k*8*k*p
                pr.append(bounds(p))
            records.append(dict(k=float(k),M=m2/2,probability=pr,
                                propagator_truncation_upper=bounds(err_total)[1]))
    return dict(K=K,cells=cells,degree=degree,counts=[bounds(t) for t in totals],
                energy_qe=[bounds(t) for t in energies],modes=records)

def tail(K, ns=96, nm=20):
    # Use time parity and mu -> -mu. Bounds are for normalized coefficients.
    ctx.cap=6
    eta=arb(0).union(arb(1)/K)
    end=arb_series([D/2,1])
    endpoint_V=arb(0); edge2=arb(0); integral=arb(0)
    for j in range(nm):
        mu=(arb(j)/nm).union(arb(j+1)/nm)
        _,V,_,hf=fields(end,mu,eta)
        endpoint_V=endpoint_V.max(abs(V[0]))
        edge2=edge2.max(abs(hf[0]))
    for t in range(ns):
        s=arb_series([(D*t/(2*ns)).union(D*(t+1)/(2*ns)),1])
        sup=arb(0)
        for j in range(nm):
            mu=(arb(j)/nm).union(arb(j+1)/nm)
            _,_,h,hov=fields(s,mu,eta)
            term=abs(hov[1])+abs(hov[0]*h[0])/K**2
            sup=sup.max(term)
        integral+=D*sup/ns
    C2=(edge2+integral/2).upper()
    B=endpoint_V.upper()
    # alpha2 endpoint rotations give |beta1| <= |beta2|+|alpha2_endpoint|.
    # Sum 4k*sum_M mu^2*(1-mu^2) <= 16*k^2/15 over a half-integer grid; integral upper for k>K.
    scale=arb(32)/15
    s1=scale*(B**2/K+B*C2/K**2+C2**2/(3*K**3))
    s2=scale*C2**2/(3*K**3)
    e2=4*scale*C2**2/(2*K**2)
    return dict(time_cells=ns,mu_cells=nm,C2_upper=bounds(C2)[1],B_upper=bounds(B)[1],
                S1_tail_upper=bounds(s1)[1],S2_tail_upper=bounds(s2)[1],
                qeE2_tail_upper=bounds(e2)[1])

if __name__=='__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser(description='Recompute the 551-pair sphere certificate')
    parser.add_argument('--output', type=Path, default=Path('results_new')/'sphere.json')
    output = parser.parse_args().output
    K=24
    rows={name:[ray(2*n, .5 if name=='interior' else 1-1/(2*n))
                for n in (4,8,16,32,64)] for name in ('interior','edge')}
    print('direct rays:',rows,flush=True)
    tails=tail(K)
    print('uniform tail certificate:',tails,flush=True)
    lows=finite_sum(K)
    print('certified finite sum:',lows['counts'],lows['energy_qe'],flush=True)
    data=dict(D=bounds(D),r=bounds(R),rays=rows,finite=lows,tail=tails)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(data,indent=2)+'\n',encoding='ascii')

