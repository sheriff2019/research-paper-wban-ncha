"""Run the reframed NCHA experiment: multi-seed, honest statistics.
Focus metrics = critical-traffic QoS (P1 delivery, P1 latency), energy, and honest
aggregate PLR. Outputs per-trial CSV + a statistics summary."""
import numpy as np, csv, json
from collections import defaultdict
from scipy import stats
from sim import run_trial

PROTOCOLS = ["NCHA", "NCHA-NP", "M-ATTEMPT", "CICADA", "PRPLC"]
METRICS = ["PLR", "pdr_p1", "pdr_p2", "pdr_p3", "pdr_p4", "latency_p1_ms", "energy_per_pkt_mJ", "throughput_kbps"]
rng = np.random.default_rng(7)

def collect(nodes_list, seeds, T=20.0):
    rows = []
    for N in nodes_list:
        for s in seeds:
            for p in PROTOCOLS:
                r = run_trial(p, seed=s, T_sim=T, n_nodes=N)
                rows.append({k: r[k] for k in ["protocol","seed","n_nodes",*METRICS]})
    return rows

def desc(a):
    a = np.asarray(a, float); a = a[~np.isnan(a)]; n=len(a)
    if n == 0: return dict(mean=float("nan"))
    m=a.mean(); sd=a.std(ddof=1) if n>1 else 0.0
    ci=stats.t.ppf(.975,n-1)*sd/np.sqrt(n) if n>1 else 0.0
    q1,med,q3=np.percentile(a,[25,50,75])
    return dict(mean=m,sd=sd,ci_lo=m-ci,ci_hi=m+ci,median=med,iqr_lo=q1,iqr_hi=q3,n=n)

def rankbis(x,y):
    x=np.asarray(x,float);y=np.asarray(y,float)
    x=x[~np.isnan(x)];y=y[~np.isnan(y)]
    U,pp=stats.mannwhitneyu(x,y,alternative='two-sided')
    return 2*U/(len(x)*len(y))-1, U, pp   # rank-biserial; positive => x>y

def boot(x,y,fn,B=1500):
    x=np.asarray(x,float);y=np.asarray(y,float);x=x[~np.isnan(x)];y=y[~np.isnan(y)]
    v=[fn(rng.choice(x,len(x)),rng.choice(y,len(y))) for _ in range(B)]
    return np.percentile(v,[2.5,97.5])

if __name__ == "__main__":
    import sys, time
    t0=time.time()
    # headline: one representative congested load, many seeds
    HEAD_N=16; seeds=list(range(1,31))
    head=collect([HEAD_N], seeds, T=20.0)
    with open("results_headline.csv","w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(head[0].keys())); w.writeheader(); w.writerows(head)
    by=defaultdict(lambda:defaultdict(list))
    for r in head:
        for m in METRICS: by[r["protocol"]][m].append(r[m])
    print(f"=== HEADLINE (N={HEAD_N}, {len(seeds)} seeds) ===")
    for m in METRICS:
        print(f"\n## {m}")
        for p in PROTOCOLS:
            d=desc(by[p][m]); print(f"  {p:11s} mean={d['mean']:8.2f} sd={d['sd']:7.2f} 95%CI[{d['ci_lo']:.2f},{d['ci_hi']:.2f}] median={d['median']:.2f} IQR[{d['iqr_lo']:.2f},{d['iqr_hi']:.2f}]")
    print("\n=== NCHA vs baselines (MWU, Holm across 4 comparisons per metric, rank-biserial+CI) ===")
    for m in ["pdr_p1","latency_p1_ms","energy_per_pkt_mJ","PLR"]:
        print(f"\n## {m}")
        comps={}
        for b in ["NCHA-NP","M-ATTEMPT","CICADA","PRPLC"]:
            rb,U,pp=rankbis(by["NCHA"][m],by[b][m]); comps[b]=(rb,U,pp)
        order=sorted(comps,key=lambda b:comps[b][2]); mt=len(order)
        holm={b:min(1.0,comps[b][2]*(mt-i)) for i,b in enumerate(order)}
        for b in ["NCHA-NP","M-ATTEMPT","CICADA","PRPLC"]:
            rb,U,pp=comps[b]; lo,hi=boot(by["NCHA"][m],by[b][m],lambda a,c:rankbis(a,c)[0])
            ps="<1e-12" if pp<1e-12 else f"{pp:.2e}"; hs="<1e-12" if holm[b]<1e-12 else f"{holm[b]:.2e}"
            print(f"  NCHA vs {b:10s} U={U:8.0f} p={ps} p_Holm={hs} rank-biserial={rb:+.3f} [{lo:+.3f},{hi:+.3f}]")
    # scaling sweep (fewer seeds)
    sweepN=[6,8,10,12,14,16,18,20]
    sw=collect(sweepN, list(range(1,9)), T=20.0)
    with open("results_scaling.csv","w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(sw[0].keys())); w.writeheader(); w.writerows(sw)
    print(f"\n=== SCALING (mean P1 delivery %, per protocol) ===")
    sb=defaultdict(lambda:defaultdict(list))
    for r in sw: sb[(r["protocol"],r["n_nodes"])]["pdr_p1"].append(r["pdr_p1"])
    hdr="N     "+"".join(f"{p:>11}" for p in PROTOCOLS); print(hdr)
    for N in sweepN:
        line=f"{N:<6}"+"".join(f"{np.mean(sb[(p,N)]['pdr_p1']):>11.1f}" for p in PROTOCOLS); print(line)
    print(f"\n(done in {time.time()-t0:.0f}s; wrote results_headline.csv, results_scaling.csv)")
