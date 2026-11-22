"""Check the archived enclosures from individual modes and plot the sphere history."""
import argparse
import json
from collections import defaultdict
from math import cosh, exp, sqrt
from pathlib import Path

from flint import arb, ctx

ROOT=Path(__file__).resolve().parents[1]


def audit():
    ctx.prec=170
    data=json.loads((ROOT/"data/sphere.json").read_text())
    finite,tail=data["finite"],data["tail"]
    expected={(twok,m2) for twok in range(4,49)
              for m2 in range(1 if twok%2 else 2,twok,2)}
    sums=[[arb(0),arb(0)] for _ in range(2)]
    found=set()
    for row in finite["modes"]:
        twok,m2=round(2*row["k"]),round(2*row["M"])
        label=(twok,m2)
        assert label not in found
        found.add(label)
        for order,(lower,upper) in enumerate(row["probability"]):
            assert 0<lower<=upper<=1
            probability=arb(lower).union(arb(upper))
            sums[order][0]+=4*twok*probability
            sums[order][1]+=8*twok*twok*probability
    assert expected==found and len(found)==551
    for order,(count,energy) in enumerate(sums):
        for value,pair in [(count,finite["counts"][order]),(energy,finite["energy_qe"][order])]:
            assert value.overlaps(arb(pair[0]).union(arb(pair[1])))
    reported=[(6.11898e-5,8.06069e-5),(4.05596e-6,4.42568e-6),(5.68374e-5,11.00760e-5)]
    for pair,tail_key,outer in zip([*finite["counts"],finite["energy_qe"][1]],
                                   ["S1_tail_upper","S2_tail_upper","qeE2_tail_upper"],reported):
        assert pair[0]>=outer[0] and arb(pair[1])+arb(tail[tail_key])<arb(outer[1])
    third=json.loads((ROOT/"data/third_order.json").read_text())
    for key,bounds in [("count",(2.02282,2.02964)),("energy_qe",(1.25775,1.31281))]:
        actual=third["summary"][key]["ratio_32"]
        assert bounds[0]<=actual[0]<=actual[1]<=bounds[1]
    nil=json.loads((ROOT/"data/nil_paired.json").read_text())
    assert sum(len(i["rows"]) for i in nil)==326 and len(nil)==8
    print("551 spherical magnetic pairs, three all-mode enclosures, order-three ratios and 326 Nil samples verified.")
    return data


def render(data,target):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    y=sqrt(3/(4*exp(.03)-1))
    z=y*exp(.015)
    D=sum(data["D"])/2
    t=np.linspace(-.5,.5,401)
    v=z/np.cosh(2*z*t*D)
    u=y/np.cosh(2*y*t*D)
    shape=np.log(v/u)/3
    radius=np.sqrt(v*np.exp(shape))
    panels=plt.subplots(1,2,figsize=(8.4,3.2),layout="constrained")
    fig,(left,right)=panels
    left.plot(t,radius,color="#23597C",label="Radius q")
    left.set(xlabel="Harmonic time / duration",ylabel="Radius in charge units",title="Exact equal-radius return")
    ax=left.twinx()
    ax.plot(t,shape*1000,color="#A34A32",ls="--",label="Shape")
    ax.set_ylabel("Shape × 1000",color="#A34A32")
    weighted=[defaultdict(float),defaultdict(float)]
    for mode in data["finite"]["modes"]:
        for order,(lo,hi) in enumerate(mode["probability"]):
            k=mode["k"]
            weighted[order][k]+=32*k*k*(lo+hi)/2
    for order,(values,color) in enumerate(zip(weighted,["#23597C","#A34A32"]),1):
        ks=sorted(values)
        right.semilogy(ks,np.cumsum([values[k] for k in ks]),color=color,label=f"Order {order}")
    right.set(xlabel="Upper shell K",ylabel="Cumulative energy × endpoint radius",title="Finite-shell occupation energies")
    right.legend(frameon=False)
    target.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(target)
    plt.close(fig)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure",type=Path)
    args=parser.parse_args()
    data=audit()
    if args.figure: render(data,args.figure)
