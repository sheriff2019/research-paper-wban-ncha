# Reproducing the results in the NCHA paper

This document lists the exact commands that regenerate every number and figure in
"NCHA: A Priority-Aware Congestion-Handling Algorithm that Protects Critical Traffic
in Wireless Body Area Networks for the Internet of Medical Things."

Every run is deterministic in its (protocol, seed, node-count) triple, so the numbers
below reproduce exactly on any machine.

## 1. Environment
```
python -m venv venv && source venv/bin/activate     # optional
pip install -r requirements.txt                      # numpy, scipy, matplotlib
```
Tested with Python 3.11.

## 2. Smoke test (a few seconds)
```
python sim.py
```
Runs one 30 s trial of each protocol and prints PLR, critical delivery, latency and energy.

## 3. Headline and scaling results + statistics (~3 min)
```
python run_experiment.py
```
Writes `results/results_headline.csv` (16 nodes, 30 seeds) and `results/results_scaling.csv`
(6–20 nodes), and prints the full statistics: means, SD, 95% CIs, medians, IQRs, and the
Mann–Whitney U tests with Holm correction and rank-biserial effect sizes with bootstrap CIs.
These are the numbers in Table 4 and Figs 4–6.

## 4. Reviewer-requested sweeps (~3 min)
```
python run_full.py
```
Writes `results/sweep_weights.csv` (CSI weight sensitivity, Fig 7),
`results/sweep_kappa.csv` (threshold robustness, Fig 8), and
`results/sweep_traffic.csv` (ON-OFF / MMPP / Poisson, Fig 9).

## 5. Figures
```
python make_figs.py
```
Regenerates all result figures in `figures/`.

## 6. Mapping results to the paper
| Paper item | Produced by | File |
|---|---|---|
| Table 4, Figs 4–6 | run_experiment.py | results/results_headline.csv, results_scaling.csv |
| Fig 7 (weights) | run_full.py | results/sweep_weights.csv |
| Fig 8 (kappa) | run_full.py | results/sweep_kappa.csv |
| Fig 9 (traffic) | run_full.py | results/sweep_traffic.csv |
| All statistics | run_experiment.py | results/stats_final.txt (archived copy) |

## Notes on scope (as stated in the paper, §5)
- This is a simulation study. No human participants or hardware sensors were involved;
  all traffic is generated from the stochastic models in `sim.py`.
- The MAC/channel is an abstraction, not a full IEEE 802.15.6 PHY; absolute throughput is
  simulation-scale and only the cross-protocol comparison is meaningful.
- All tunable parameters live in the `Config` dataclass at the top of `sim.py`.
