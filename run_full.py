"""Reviewer-requested sweeps: CSI-weight sensitivity (R1.1), kappa sweep (R1.2),
traffic-model comparison ON-OFF/MMPP/Poisson (R1.5). Writes CSVs."""
import csv, numpy as np
from collections import defaultdict
from sim import run_trial, Config

def mean_ci(a):
    a=np.asarray(a,float); m=a.mean(); sd=a.std(ddof=1) if len(a)>1 else 0
    from scipy import stats
    ci=stats.t.ppf(.975,len(a)-1)*sd/np.sqrt(len(a)) if len(a)>1 else 0
    return m,ci

SEEDS=list(range(1,13)); N=16; T=20.0

# --- CSI weight sensitivity (NCHA only) ---
WEIGHTS={
 "balanced (0.35/0.30/0.20/0.15)":(0.35,0.30,0.20,0.15),
 "queue-heavy (0.60/0.20/0.10/0.10)":(0.60,0.20,0.10,0.10),
 "drop-heavy (0.20/0.60/0.10/0.10)":(0.20,0.60,0.10,0.10),
 "equal (0.25 each)":(0.25,0.25,0.25,0.25),
 "util-heavy (0.20/0.20/0.50/0.10)":(0.20,0.20,0.50,0.10),
}
with open("sweep_weights.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["weights","seed","pdr_p1","PLR","latency_p1_ms"])
    for name,(wq,wd,wu,wp) in WEIGHTS.items():
        for s in SEEDS:
            r=run_trial("NCHA",seed=s,T_sim=T,n_nodes=N,w_Q=wq,w_D=wd,w_U=wu,w_P=wp)
            w.writerow([name,s,r["pdr_p1"],r["PLR"],r["latency_p1_ms"]])
print("weights sweep:")
d=defaultdict(lambda:defaultdict(list))
for row in csv.DictReader(open("sweep_weights.csv")):
    for m in ("pdr_p1","PLR","latency_p1_ms"): d[row["weights"]][m].append(float(row[m]))
for name in WEIGHTS:
    m1,c1=mean_ci(d[name]["pdr_p1"]); print(f"  {name:38s} P1={m1:.1f}±{c1:.1f}%")

# --- kappa sweep ---
KAPPAS=[0.0,0.5,1.0,1.5,2.0,3.0]
with open("sweep_kappa.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["kappa","seed","pdr_p1","PLR","latency_p1_ms"])
    for kap in KAPPAS:
        for s in SEEDS:
            r=run_trial("NCHA",seed=s,T_sim=T,n_nodes=N,kappa=kap)
            w.writerow([kap,s,r["pdr_p1"],r["PLR"],r["latency_p1_ms"]])
print("kappa sweep:")
dk=defaultdict(lambda:defaultdict(list))
for row in csv.DictReader(open("sweep_kappa.csv")):
    for m in ("pdr_p1","PLR","latency_p1_ms"): dk[row["kappa"]][m].append(float(row[m]))
for kap in KAPPAS:
    m1,_=mean_ci(dk[str(kap)]["pdr_p1"]); ml,_=mean_ci(dk[str(kap)]["latency_p1_ms"])
    print(f"  kappa={kap:<4} P1={m1:.1f}%  latP1={ml:.0f}ms")

# --- traffic model comparison (all protocols) ---
MODELS=["onoff","mmpp","poisson"]
PROTS=["NCHA","M-ATTEMPT","CICADA","PRPLC"]
with open("sweep_traffic.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["model","protocol","seed","pdr_p1","PLR"])
    for tm in MODELS:
        for p in PROTS:
            for s in SEEDS:
                r=run_trial(p,seed=s,T_sim=T,n_nodes=N,traffic_model=tm)
                w.writerow([tm,p,s,r["pdr_p1"],r["PLR"]])
print("traffic-model P1 delivery (NCHA):")
dt=defaultdict(list)
for row in csv.DictReader(open("sweep_traffic.csv")):
    if row["protocol"]=="NCHA": dt[row["model"]].append(float(row["pdr_p1"]))
for tm in MODELS:
    m,_=mean_ci(dt[tm]); print(f"  {tm:8s} P1={m:.1f}%")
print("done")
