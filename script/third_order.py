"""Certified superadiabatic shell sums on the fixed equal-radius history of Paper 10 (python-flint)."""
import json
import math
from pathlib import Path
from flint import acb, acb_mat, arb, arb_series, ctx


ctx.prec = 200
Y = (3 / (4*(arb(3)/100).exp()-1)).sqrt()
Z = Y*(arb(3)/200).exp()
D = arb("1.2146378733763", "1e-12")
def gap(d):
    return Z/(Z*d).cosh() - Y/(Y*d).cosh()

lo, hi = D.lower(), D.upper()
assert gap(lo)>0 and gap(hi)<0
for _ in range(102):
    mid = (lo+hi)/2
    if gap(mid)>0:
        lo = mid
    else:
        assert gap(mid)<0
        hi = mid
D = lo.union(hi)
R = Z/(Z*D).cosh()
assert 3<arb(1)/2+3/(2*R)<4
I = acb(0,1)
REP = {(2,1),(3,1),(3,2),(4,2),(8,4),(8,7)}

def outward(x):
    assert x.is_finite()
    return [math.nextafter(float(x.lower()),-math.inf),math.nextafter(float(x.upper()),math.inf)]

def ch(x):
    e=x.exp()
    return (e+1/e)/2

def ratios(s,mu,eta,depth):
    # Avoid subtracting two order-one jets to get the small anisotropy field.
    cy = ch(2*Y*s)
    tangent = 1-2/(1+(4*Y*s).exp())
    delta = (2*(Z-Y)*s).exp()
    e = (Z/Y)/(ch(2*(Z-Y)*s)+tangent*(delta-1/delta)/2)
    B = 1+mu**2*(e**2-1)
    omega = 2*Y*e*B.sqrt()/cy
    T = -e.derivative()/(2*B*omega)
    p2 = mu**2*(1-mu**2)
    terms = []
    for n in range(1,depth+1):
        den = 1+p2*eta**(2*n)*T**2
        omega = omega*den.sqrt()
        J = T.derivative()/(2*den)
        terms.append((T,J,J/omega))
        T = (-1)**n*J/omega
    return terms

def frames(mu,k,depth):
    ctx.cap=depth+4
    p=mu*(1-mu**2).sqrt()
    c,d=((1+mu)/2).sqrt(),((1-mu)/2).sqrt()
    result=[]
    for sign in [-1,1]:
        W=acb_mat([[c,-d],[d,c]])
        units=[W]
        for n,(T,_,_) in enumerate(ratios(arb_series([sign*D/2,1]),mu,1/k,depth),1):
            si,co=((p*T[0]/k**n).atan()/2).sin_cos()
            W=W*(acb_mat([[co,-I*si],[-I*si,co]]) if n%2 else acb_mat([[co,-si],[si,co]]))
            units.append(W)
        result.append(units)
    return result

def fields_half(cells,degree):
    ctx.cap=degree
    step=D/(2*cells)
    local=[]
    for n in range(cells):
        t=arb_series([n*step,step])
        v=Z/ch(2*Z*t)
        w=v**2*ch(2*Y*t)/Y
        local.append(([2*step*v[j] for j in range(degree)],
                      [2*step*w[j] for j in range(degree)]))
    return local

def finite(K=32, cells=40, degree=37):
    local=fields_half(cells,degree)
    rho=arb(1)/5
    x=D/(2*cells*rho)
    vb=Z/(2*Z*rho).cos()
    Hmax=2*(vb+vb**2*(2*Y*(D/2+rho)).cosh()/Y)
    totals=[[arb(0) for m in range(3)] for weight in range(4)]
    selected=[]
    moments=[]
    for n in range(4,2*K+1):
        k=arb(n)/2
        cell_err=(rho*k*Hmax).exp()*x**(degree+1)/(1-x)
        err=((1+cell_err)**(2*cells)-1).abs_upper()
        for nM in range(n%2 or 2,n,2):
            mu=arb(nM)/n
            L=(1-mu**2).sqrt()
            a,b=acb(1),acb(0)
            for ax,az in local:
                hx=[k*L*t for t in ax]
                hz=[k*mu*t for t in az]
                aa,bb=[a],[b]
                for order in range(1,degree+1):
                    aa_new=sum(hz[j]*aa[order-1-j]+hx[j]*bb[order-1-j] for j in range(order))
                    bb_new=sum(hx[j]*aa[order-1-j]-hz[j]*bb[order-1-j] for j in range(order))
                    aa.append(-I*aa_new/order)
                    bb.append(-I*bb_new/order)
                a,b=sum(aa),sum(bb)
            half=acb_mat([[a,-b.conjugate()],[b,a.conjugate()]])
            prop=half*half.transpose()
            pair=(n/2,nM/2)
            depth=5 if pair in REP else 3
            initial,final=frames(mu,k,depth)
            probs=[]
            for m in range(depth+1):
                amp=abs((final[m].conjugate().transpose()*prop*initial[m])[0,1])+arb(0,err)
                prob=amp**2
                assert prob>0
                probs.append(prob)
                if 1<=m<=3:
                    S=8*k*prob
                    E=4*k*S
                    for slot,value in enumerate([S,E,S if n<=8 else 0,E if n<=8 else 0]):
                        totals[slot][m-1]+=value
            if pair in REP:
                opt=min(range(6),key=lambda j:float(probs[j]))
                assert all(probs[opt]<q for j,q in enumerate(probs) if j!=opt)
                signs=[('up' if a<b else 'down' if a>b else 'overlap') for a,b in zip(probs,probs[1:])]
                assert 'overlap' not in signs
                selected.append(dict(k=pair[0],M=pair[1],opt=opt,adjacent=signs,prob=list(map(outward,probs))))
            moments.append(dict(k=pair[0],M=pair[1],p123=[outward(q) for q in probs[1:4]],err=outward(err)[1]))
        if n%8==0:
            print("Completed shell",n/2,flush=True)
    return totals,dict(K=K,half_cells=cells,degree=degree,representatives=selected,modes=moments)

def tail(K,ns=384,nm=32):
    ctx.cap=8
    eta=arb(0).union(arb(1)/K)
    b2,b3=arb(0),arb(0)
    edge=arb(0)
    integral=arb(0)
    for j in range(nm):
        mu=(arb(j)/nm).union(arb(j+1)/nm)
        T=ratios(arb_series([D/2,1]),mu,eta,3)
        b2=b2.max(abs(T[1][0][0]))
        b3=b3.max(abs(T[2][0][0]))
        edge=edge.max(abs(T[2][2][0]))
    for t in range(ns):
        s=arb_series([(D*t/(2*ns)).union(D*(t+1)/(2*ns)),1])
        upper=arb(0)
        for j in range(nm):
            mu=(arb(j)/nm).union(arb(j+1)/nm)
            _,h,hover=ratios(s,mu,eta,3)[-1]
            upper=upper.max(abs(hover[1])+abs(hover[0]*h[0])/K**3)
        integral+=upper*D/ns
    C=(edge+integral/2).upper()
    B2,B3=b2.upper(),b3.upper()
    # Successive changes of the two endpoint frames cost at most one full angle.
    polynomials=[{2:B2,3:B3,4:C},{3:B3,4:C},{4:C}]
    def sum_bound(terms,w):
        powers=[(a+b-2-w,ca*cb) for a,ca in terms.items() for b,cb in terms.items()]
        return arb(32)/15*4**w*sum(cc/(K**(n-1)*(n-1)) for n,cc in powers)
    tails=[[sum_bound(v,w) if not(w==1 and m==0) else None
            for m,v in enumerate(polynomials)] for w in [0,1]]
    # Signed leading endpoint coefficient for the third moment, with no finite-k rotations.
    interval_mu=(arb(2)/5).union(arb(1)/2)
    ends=ratios(arb_series([D/2,1]),interval_mu,arb(0),3)
    band1=ends[1][0][0]
    assert not band1.contains(0)
    band=ends[2][0][0]
    assert not band.contains(0)
    return tails,dict(time_cells=ns,mu_cells=nm,B2=outward(B2),B3=outward(B3),C3=outward(C),
                      T3_f_limit_I=outward(band),T2_f_limit_I=outward(band1))

def serialize(totals,tails):
    ans={}
    for w,key in enumerate(['count','energy_qe']):
        entries=[]
        for m in range(3):
            finite_part=totals[w][m]
            low_part=totals[w+2][m]
            t=tails[w][m]
            full=finite_part+arb(0).union(t) if t is not None else None
            entries.append(dict(order=m+1,finite=outward(finite_part),low=outward(low_part),
                                tail_max=outward(t)[1] if t is not None else 'infinite',
                                all=outward(full) if full is not None else 'infinite',
                                low_fraction=outward(low_part/full) if full is not None else [0,0]))
        full2=totals[w][1]+arb(0).union(tails[w][1])
        full3=totals[w][2]+arb(0).union(tails[w][2])
        ans[key]=dict(rows=entries,ratio_32=outward(full3/full2))
    return ans

if __name__=='__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser(description='Recompute the order-three comparison')
    parser.add_argument('--output', type=Path, default=Path('results_new')/'third_order.json')
    output = parser.parse_args().output
    K=32
    tails,certificate=tail(K)
    print("Uniform tail",certificate,flush=True)
    totals,data=finite(K)
    summary=serialize(totals,tails)
    print("Summary",summary,flush=True)
    print("Representatives",data['representatives'],flush=True)
    data.update(summary=summary,tail_certificate=certificate,D=outward(D),r=outward(R),precision_bits=ctx.prec)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(data,indent=2)+'\n',encoding='ascii')

