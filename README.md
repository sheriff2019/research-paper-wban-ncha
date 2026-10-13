# WBAN-Sim — reproducible simulator for the NCHA congestion-control study

Discrete-event simulator and analysis code for the paper **"NCHA: A Priority-Aware
Congestion-Handling Algorithm that Protects Critical Traffic in Wireless Body Area
Networks for the Internet of Medical Things"** (submitted to *Array*, Elsevier).

It models a star-topology WBAN (N sensor nodes → shared CSMA channel → gateway buffer →
sink) with four clinical priority classes, bursty ON-OFF/MMPP traffic, a contention/collision
model, and a state-based radio energy model. Five schemes share the same MAC, channel,
traffic and buffer for a fair comparison: **NCHA**, an ablation **NCHA-NP** (priority
removed), and baselines **M-ATTEMPT**, **CICADA** and **PRPLC**.

## Contents
```
sim.py              core simulator (all parameters in the Config dataclass)
run_experiment.py   headline (16-node) + scaling (6–20) runs with full statistics
run_full.py         CSI-weight, kappa, and traffic-model sweeps
make_figs.py        regenerates all result figures
requirements.txt    numpy, scipy, matplotlib
REPRODUCE.md        exact commands to reproduce every number and figure
LICENSE             MIT
results/            result CSVs + archived statistics output
figures/            generated figures
```

## Quick start
```
pip install -r requirements.txt
python run_experiment.py     # headline + scaling results and statistics
python run_full.py           # reviewer-requested sweeps
python make_figs.py          # figures
```
See **REPRODUCE.md** for the full reproduction guide and the results-to-paper mapping.

## Key result
Past the shared bottleneck, aggregate loss is similar across schemes. NCHA's decisive,
statistically significant advantage is **critical-traffic quality of service**: at 16 nodes
it delivers 96.6% of critical (emergency) packets versus 62.9–76.7% for the baselines, at
~0.57 s critical latency versus 4.6–7.0 s, verified by the NCHA-vs-NCHA-NP ablation (all
comparisons Mann–Whitney U, Holm-corrected p < 0.001).

## Scope
This is a simulation study. No human participants or hardware sensors were involved; all
physiological traffic is generated from documented stochastic models. The MAC/channel is an
abstraction, not a full IEEE 802.15.6 PHY, so absolute throughput is simulation-scale and
only cross-protocol comparisons are meaningful.

## Citation
If you use this code, please cite the paper (details to be added on acceptance) and this
repository.
