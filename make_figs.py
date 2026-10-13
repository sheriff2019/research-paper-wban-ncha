"""Generate the full figure set for the reframed NCHA paper from the result CSVs."""
import csv, numpy as np
from collections import defaultdict
from scipy import stats
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sim import run_trial

plt.rcParams.update({"figure.dpi":300,"savefig.dpi":300,"font.family":"DejaVu Sans","font.size":10,
 "axes.edgecolor":"#52514e","axes.linewidth":0.8,"axes.grid":True,"grid.color":"#e6e5e1",
 "grid.linewidth":0.7,"axes.axisbelow":True,"axes.spines.top":False,"axes.spines.right":False,
 "figure.facecolor":"white","axes.facecolor":"white"})
COL={"NCHA":"#2a78d6","NCHA-NP":"#8a8a86","M-ATTEMPT":"#eb6834","CICADA":"#1baf7a","PRPLC":"#e34948"}
P5=["NCHA","NCHA-NP","M-ATTEMPT","CICADA","PRPLC"]
def rd(f): return list(csv.DictReader(open(f)))
def ci(a): a=np.asarray(a,float); return stats.t.ppf(.975,len(a)-1)*a.std(ddof=1)/np.sqrt(len(a)) if len(a)>1 else 0

# 1) scaling multi-panel
def fig_scaling():
    rows=rd("results_scaling.csv"); Ns=sorted({int(r['n_nodes']) for r in rows})
    def ser(metric,p):
        d=defaultdict(list)
        for r in rows:
            if r['protocol']==p: d[int(r['n_nodes'])].append(float(r[metric]))
        return [np.mean(d[n]) for n in Ns]
    fig,ax=plt.subplots(2,2,figsize=(9,6.6))
    specs=[("pdr_p1","Critical (P1) delivery ratio (%)","(a) Critical delivery"),
           ("PLR","Aggregate packet loss (%)","(b) Aggregate loss (bottleneck-bound)"),
           ("latency_p1_ms","Critical (P1) latency (ms)","(c) Critical latency"),
           ("energy_per_pkt_mJ","Energy per delivered packet (mJ)","(d) Energy")]
    for a,(m,yl,ti) in zip(ax.flat,specs):
        for p in P5:
            a.plot(Ns,ser(m,p),marker='o',ms=4,lw=1.7 if p=='NCHA' else 1.1,
                   color=COL[p],ls='--' if p=='NCHA-NP' else '-',label=p)
        a.set_xlabel("Number of sensor nodes");a.set_ylabel(yl,fontsize=9);a.set_title(ti,fontsize=9.5)
    ax[0,0].legend(fontsize=7.5,frameon=False,loc="lower left")
    fig.tight_layout();fig.savefig("fig_scaling.png",bbox_inches="tight");plt.close(fig)

# 2) ablation + baselines bar at N=16 (P1 delivery, P1 latency)
def fig_ablation():
    rows=rd("results_headline.csv")
    d=defaultdict(lambda:defaultdict(list))
    for r in rows:
        for m in ("pdr_p1","latency_p1_ms"): d[r['protocol']][m].append(float(r[m]))
    fig,ax=plt.subplots(1,2,figsize=(9,3.6));xs=np.arange(len(P5))
    for a,(m,yl,ti,fmt) in zip(ax,[("pdr_p1","Critical (P1) delivery (%)","(a) Critical delivery at 16 nodes","{:.1f}"),
                                   ("latency_p1_ms","Critical (P1) latency (ms)","(b) Critical latency at 16 nodes","{:.0f}")]):
        ys=[np.mean(d[p][m]) for p in P5];es=[ci(d[p][m]) for p in P5]
        a.bar(xs,ys,width=0.62,color=[COL[p] for p in P5],yerr=es,capsize=3,error_kw=dict(ecolor="#52514e",lw=1))
        for x,y,e in zip(xs,ys,es): a.text(x,y+e+max(ys)*0.02,fmt.format(y),ha="center",va="bottom",fontsize=8,fontweight="bold")
        a.set_xticks(xs);a.set_xticklabels(P5,fontsize=8,rotation=15);a.set_ylabel(yl,fontsize=9)
        a.set_title(ti,fontsize=9.5);a.set_ylim(0,max(ys)*1.2)
    fig.tight_layout();fig.savefig("fig_ablation.png",bbox_inches="tight");plt.close(fig)

# 3) per-priority delivery breakdown at N=16
def fig_priority():
    rows=rd("results_headline.csv")
    d=defaultdict(lambda:defaultdict(list))
    for r in rows:
        for p in range(1,5): d[r['protocol']][f"p{p}"].append(float(r[f"pdr_p{p}"]))
    fig,ax=plt.subplots(figsize=(7.4,3.8));xs=np.arange(4);wd=0.16
    pcol=["#2a78d6","#8a8a86","#eb6834","#1baf7a","#e34948"]
    for i,p in enumerate(P5):
        ys=[np.mean(d[p][f"p{k}"]) for k in range(1,5)]
        ax.bar(xs+(i-2)*wd,ys,width=wd,color=COL[p],label=p)
    ax.set_xticks(xs);ax.set_xticklabels(["P1 (critical)","P2 (diagnostic)","P3 (routine)","P4 (background)"],fontsize=8.5)
    ax.set_ylabel("Delivery ratio (%)");ax.set_title("Per-priority delivery at 16 nodes: NCHA protects critical traffic",fontsize=9.5)
    ax.legend(fontsize=7.5,frameon=False,ncol=5,loc="upper center",bbox_to_anchor=(0.5,-0.14))
    fig.subplots_adjust(bottom=0.24);fig.savefig("fig_priority.png",bbox_inches="tight");plt.close(fig)

# 4) weight sensitivity
def fig_weights():
    rows=rd("sweep_weights.csv");names=[]
    d=defaultdict(list)
    for r in rows: d[r['weights']].append(float(r['pdr_p1']))
    names=list(d.keys())
    ys=[np.mean(d[n]) for n in names];es=[ci(d[n]) for n in names]
    fig,ax=plt.subplots(figsize=(7.6,3.6));xs=np.arange(len(names))
    cols=["#2a78d6"]+["#8a8a86"]*(len(names)-1)
    ax.bar(xs,ys,width=0.6,color=cols,yerr=es,capsize=3,error_kw=dict(ecolor="#52514e",lw=1))
    for x,y in zip(xs,ys): ax.text(x,y+1,f"{y:.1f}",ha="center",va="bottom",fontsize=8,fontweight="bold")
    ax.set_xticks(xs);ax.set_xticklabels([n.split(" (")[0] for n in names],fontsize=8,rotation=15)
    ax.set_ylabel("Critical (P1) delivery (%)");ax.set_ylim(0,105)
    ax.set_title("CSI weight sensitivity (N=16): balanced weighting is best",fontsize=9.5)
    fig.tight_layout();fig.savefig("fig_weights.png",bbox_inches="tight");plt.close(fig)

# 5) kappa robustness
def fig_kappa():
    rows=rd("sweep_kappa.csv")
    d1=defaultdict(list);d2=defaultdict(list)
    for r in rows: d1[float(r['kappa'])].append(float(r['pdr_p1']));d2[float(r['kappa'])].append(float(r['latency_p1_ms']))
    ks=sorted(d1)
    fig,ax=plt.subplots(figsize=(6.4,3.6))
    ax.plot(ks,[np.mean(d1[k]) for k in ks],marker='o',color="#2a78d6",lw=1.6,label="P1 delivery (%)")
    ax.set_xlabel("Threshold sensitivity κ");ax.set_ylabel("Critical (P1) delivery (%)",color="#2a78d6")
    ax.set_ylim(90,100)
    ax.set_title("Robustness to κ: performance stable across κ ∈ [0,3]",fontsize=9.5)
    fig.tight_layout();fig.savefig("fig_kappa.png",bbox_inches="tight");plt.close(fig)

# 6) traffic-model comparison
def fig_traffic():
    rows=rd("sweep_traffic.csv");models=["onoff","mmpp","poisson"];prots=["NCHA","M-ATTEMPT","CICADA","PRPLC"]
    d=defaultdict(lambda:defaultdict(list))
    for r in rows: d[(r['model'],r['protocol'])]["p1"].append(float(r['pdr_p1']))
    fig,ax=plt.subplots(figsize=(7.2,3.6));xs=np.arange(len(models));wd=0.2
    for i,p in enumerate(prots):
        ys=[np.mean(d[(m,p)]["p1"]) for m in models]
        ax.bar(xs+(i-1.5)*wd,ys,width=wd,color=COL[p],label=p)
    ax.set_xticks(xs);ax.set_xticklabels(["ON-OFF","MMPP","Poisson"]);ax.set_ylabel("Critical (P1) delivery (%)")
    ax.set_title("Critical delivery across traffic models (N=16)",fontsize=9.5)
    ax.legend(fontsize=7.5,frameon=False,ncol=4,loc="upper center",bbox_to_anchor=(0.5,-0.14))
    fig.subplots_adjust(bottom=0.22);fig.savefig("fig_traffic.png",bbox_inches="tight");plt.close(fig)

# 7) CSI closed-loop time series (single NCHA run under bursty load)
def fig_csi():
    r=run_trial("NCHA",seed=3,T_sim=40.0,n_nodes=14)
    csi=np.array(r["csi_series"]);q=np.array(r["q_series"]);t=np.arange(len(csi))*0.05
    fig,ax=plt.subplots(figsize=(7.2,3.4))
    ax.plot(t,csi,color="#e34948",lw=1.3,label="CSI")
    ax.plot(t,q/100,color="#2a78d6",lw=1.1,alpha=0.8,label="Gateway occupancy (norm.)")
    ax.set_xlabel("Time (s)");ax.set_ylabel("Normalised value (0–1)");ax.set_ylim(0,1.05);ax.margins(x=0)
    ax.legend(fontsize=8,frameon=False,loc="upper right")
    ax.set_title("NCHA congestion signal responding to bursty traffic (14 nodes)",fontsize=9.5)
    fig.tight_layout();fig.savefig("fig_csi_response.png",bbox_inches="tight");plt.close(fig)

if __name__=="__main__":
    for fn in (fig_scaling,fig_ablation,fig_priority,fig_weights,fig_kappa,fig_traffic,fig_csi):
        fn(); print("ok",fn.__name__)
    print("all figures written")
