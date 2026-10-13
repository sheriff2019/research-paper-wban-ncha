"""
wbansim.sim — a documented, reproducible discrete-time simulator for congestion
control in Wireless Body Area Networks (star topology: N sensor nodes -> gateway -> sink).

Design goals: every result is produced by an explicit mechanism, every parameter is
exposed in Config, runs are reproducible from a seed, and the four protocols share a
common MAC/channel/traffic so comparisons are fair.

Units: time in seconds, rates in packets/s, power in watts, energy in joules.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field, asdict

# ----------------------------- Configuration --------------------------------
@dataclass
class Config:
    # topology / run
    n_nodes: int = 8
    T_sim: float = 60.0            # seconds
    dt: float = 0.005              # step (s) -> 5 ms
    seed: int = 0
    protocol: str = "NCHA"         # NCHA | M-ATTEMPT | CICADA | PRPLC

    # buffers / channel
    node_buffer: int = 60          # packets per node
    gw_buffer_K: int = 50          # gateway buffer capacity (packets)
    mac_capacity: float = 360.0    # shared CSMA MAC capacity (contention bottleneck)
    gw_service: float = 340.0      # gateway->sink service rate
    collision_gain: float = 0.7    # collision rise past channel saturation

    # radio / energy model (typical low-power body-sensor radio)
    data_rate_bps: float = 250_000.0   # 250 kbps
    packet_bits: int = 1024            # 128-byte packets
    P_tx: float = 0.031            # W (transmit)
    P_rx: float = 0.028            # W (receive)
    P_idle: float = 0.0012         # W (idle listen)
    ctrl_bits: int = 96            # CSI broadcast control packet size (bits)

    # traffic (per node, packets/s) by priority 1..4
    lam_p1_burst: float = 12.0     # emergency ON burst rate
    p1_on_prob: float = 0.01       # per-step prob of entering a burst
    p1_on_len: float = 2.0         # mean burst length (s)
    lam_p2: float = 14.0           # diagnostic (ECG) steady
    lam_p3: float = 6.0            # routine vitals
    lam_p4: float = 16.0           # background
    traffic_model: str = "onoff"   # onoff | mmpp | poisson

    # NCHA — CDM
    w_Q: float = 0.35; w_D: float = 0.30; w_U: float = 0.20; w_P: float = 0.15
    alpha: float = 0.30            # WMA smoothing
    kappa: float = 1.0             # adaptive threshold sensitivity
    csi_window: int = 40           # steps for CSI running mean/std
    control_interval: int = 4      # steps between CDM updates / broadcasts
    piat_ref: float = 0.02         # reference inter-arrival (s) for normalisation

    # NCHA — ARCM (AIMD, priority-weighted)
    R_min: float = 8.0             # packets/s
    R_max: float = 60.0            # packets/s
    R_init: float = 40.0
    gamma0: float = 0.7            # priority-weighted low-priority throttle strength
    delta0: float = 6.0            # additive-increase term (reported; controller is proportional)
    csi_setpoint: float = 0.45     # ARCM drives CSI toward this target
    kp: float = 55.0               # proportional gain (packets/s per unit CSI error)

    def priorities(self):
        return [1, 2, 3, 4]


# ----------------------------- Simulator ------------------------------------
class WBANSim:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.steps = int(round(cfg.T_sim / cfg.dt))
        N = cfg.n_nodes
        # node priority queues: counts per node per priority (index 0->P1 .. 3->P4)
        self.Q = np.zeros((N, 4))
        # per-node transmission rate (packets/s) — used by NCHA/M-ATTEMPT
        self.R = np.full(N, cfg.R_init)
        # ON-OFF emergency state per node
        self.p1_on = np.zeros(N, dtype=bool)
        # energy accumulators (J) per node, by component
        self.E_tx = np.zeros(N); self.E_rx = np.zeros(N); self.E_idle = np.zeros(N)
        # gateway buffer, per priority
        self.gwq = np.zeros(4)
        # CDM smoothed metrics + CSI history
        self.mQ = self.mD = self.mU = self.mP = 0.0
        self.csi_hist = []
        self.CSI = 0.0
        self.tau = 0.5
        # counters
        self.gen = np.zeros(4); self.recv = np.zeros(4)
        self.gw_drop = 0.0; self.node_drop = 0.0
        self.tx_success_total = 0.0
        # time series
        self.ts_qocc = []; self.ts_csi = []; self.ts_cu = []; self.ts_sys = []; self.ts_p1 = []
        self.last_arr_t = 0.0; self.piat_ewma = cfg.piat_ref
        self._pg = 0.0; self._pr = 0.0   # prev cumulative gen/recv for loss estimate

    # ---- traffic generation for this step: returns arrivals[N,4] ----
    def _arrivals(self):
        c = self.cfg; N = c.n_nodes; dt = c.dt
        a = np.zeros((N, 4))
        # P1 emergency
        if c.traffic_model in ("onoff", "mmpp"):
            # toggle ON-OFF per node
            off = ~self.p1_on
            start = off & (self.rng.random(N) < c.p1_on_prob)
            self.p1_on |= start
            stop = self.p1_on & (self.rng.random(N) < dt / c.p1_on_len)
            self.p1_on &= ~stop
            lam1 = np.where(self.p1_on, c.lam_p1_burst, 0.3)  # small baseline off-state
        else:
            lam1 = np.full(N, c.lam_p1_burst * 0.15)
        a[:, 0] = self.rng.poisson(lam1 * dt)
        a[:, 1] = self.rng.poisson(c.lam_p2 * dt, N)
        a[:, 2] = self.rng.poisson(c.lam_p3 * dt, N)
        # background: MMPP gives it bursts too
        lam4 = c.lam_p4 * (1.6 if c.traffic_model == "mmpp" and self.rng.random() < 0.2 else 1.0)
        a[:, 3] = self.rng.poisson(lam4 * dt, N)
        return a

    # ---- per-node airtime budget (packets this step) by protocol ----
    def _budget(self):
        c = self.cfg; N = c.n_nodes; dt = c.dt
        if c.protocol in ("NCHA", "NCHA-NP"):
            return self.R * dt                       # adaptive, proportional-controlled
        if c.protocol == "M-ATTEMPT":
            return np.full(N, c.R_max * dt)          # greedy fixed rate, no feedback
        if c.protocol == "CICADA":
            return np.full(N, (c.mac_capacity / N) * dt)  # TDMA fair share
        if c.protocol == "PRPLC":                    # probabilistic: transmit-or-wait
            gate = (self.rng.random(N) < 0.6).astype(float)
            return c.R_max * dt * gate
        return np.full(N, c.R_max * dt)

    # ---- attempts[N,4]: allocate each node's budget across priorities, P1 first ----
    def _attempts(self):
        c = self.cfg; N = c.n_nodes; dt = c.dt
        budget = self._budget()
        att = np.zeros((N, 4))
        # throttle profile by priority: P1 exempt, P2 (diagnostic) lightly protected,
        # P3 (routine) and P4 (background) absorb the reduction.
        gamma = np.array([0.0, 0.15, 0.6, 1.0]) * c.gamma0
        if c.protocol == "NCHA":
            # priority order within the ARCM-controlled budget: P1 first (protected),
            # low-priority additionally throttled by CSI.
            rem = budget.copy()
            for p in range(4):
                if p == 0:
                    take = np.minimum(self.Q[:, 0], rem)
                else:
                    # throttle low-priority only once congestion is DETECTED (CSI>tau);
                    # kappa (via tau) thus sets how readily NCHA sheds low-priority load.
                    detected = self.CSI > self.tau
                    scale = max(0.05, 1.0 - gamma[p] * self.CSI) if detected else 1.0
                    take = np.minimum(np.minimum(self.Q[:, p], rem), budget * scale)
                att[:, p] = take
                rem = rem - take
        else:
            # NCHA-NP (ablation) and the baselines are all priority-AGNOSTIC: none
            # implements a 4-level clinical priority queue (that is NCHA's contribution),
            # so each serves its buffer FIFO, i.e. budget shared by queue proportion.
            # They differ only in their budget policy (see _budget).
            tot = self.Q.sum(axis=1, keepdims=True)
            share = np.divide(self.Q, tot, out=np.zeros_like(self.Q), where=tot > 0)
            att = np.minimum(self.Q, share * budget[:, None])
        return att

    def step(self, k):
        c = self.cfg; N = c.n_nodes; dt = c.dt
        # 1) arrivals
        a = self._arrivals()
        self.gen += a.sum(axis=0)
        self.Q += a
        # node buffer overflow: drop lowest priority first
        over = self.Q.sum(axis=1) - c.node_buffer
        if np.any(over > 0):
            for p in (3, 2, 1, 0):
                d = np.minimum(np.maximum(over, 0), self.Q[:, p])
                self.Q[:, p] -= d; over -= d
                self.node_drop += d.sum()
        # 2) attempts: each node's airtime budget allocated P1..P4 (priority first)
        attempts = self._attempts()
        attempts = np.floor(attempts + self.rng.random(attempts.shape))  # stochastic rounding
        attempts = np.minimum(attempts, self.Q)
        att_per_node = attempts.sum(axis=1)
        A = att_per_node.sum()
        cap_step = c.mac_capacity * dt
        # 3) channel contention + collisions
        if A <= cap_step or A == 0:
            served_frac = 1.0
        else:
            served_frac = cap_step / A     # only capacity transmitted; rest stays queued
        rho_ch = A / cap_step if cap_step > 0 else 0.0
        coll = 0.0 if rho_ch <= 1 else (1 - np.exp(-c.collision_gain * (rho_ch - 1)))
        # transmitted (attempted on channel) and successfully delivered
        transmitted = attempts * served_frac
        success = transmitted * (1 - coll)
        # remove successfully delivered from node queues; collided stay (retransmit)
        self.Q -= success
        self.Q = np.maximum(self.Q, 0)
        # energy: every transmitted packet (incl. collided) costs tx time
        tx_pkts_node = transmitted.sum(axis=1)
        t_tx = tx_pkts_node * (c.packet_bits / c.data_rate_bps)
        self.E_tx += t_tx * c.P_tx
        self.E_idle += np.maximum(dt - t_tx, 0) * c.P_idle
        CU = min(rho_ch, 1.0)
        # 4) gateway buffer intake (fill by priority P1..P4, capacity K shared) + overflow
        deliver = success.sum(axis=0)              # per-priority delivered to gw
        for p in range(4):
            free = c.gw_buffer_K - self.gwq.sum()
            take = min(deliver[p], max(free, 0.0))
            self.gwq[p] += take
            self.gw_drop += deliver[p] - take
        # 5) gateway serves to sink, highest priority first
        budget = c.gw_service * dt
        for p in range(4):
            s = min(self.gwq[p], budget)
            self.gwq[p] -= s; budget -= s
            self.recv[p] += s
        self.tx_success_total += success.sum()
        # PIAT (inter-arrival at gateway): approx from arrivals this step
        arr_rate = success.sum() / dt if dt > 0 else 0.0
        piat = 1.0 / arr_rate if arr_rate > 1e-9 else c.piat_ref * 5
        self.piat_ewma = 0.2 * piat + 0.8 * self.piat_ewma
        # 6) CDM + CSI + ARCM (NCHA only), every control_interval
        if c.protocol in ("NCHA", "NCHA-NP") and (k % c.control_interval == 0):
            qn = self.gwq.sum() / c.gw_buffer_K
            # drop term = end-to-end delivery deficit the gateway infers over the interval
            g = self.gen.sum(); rv = self.recv.sum()
            dg = g - self._pg; dr = rv - self._pr
            dn = float(np.clip(1 - dr / dg, 0, 1)) if dg > 0 else 0.0
            self._pg = g; self._pr = rv
            un = CU
            pn = np.clip(1 - self.piat_ewma / (c.piat_ref * 5), 0, 1)  # smaller PIAT -> higher
            self.mQ = c.alpha * qn + (1 - c.alpha) * self.mQ
            self.mD = c.alpha * dn + (1 - c.alpha) * self.mD
            self.mU = c.alpha * un + (1 - c.alpha) * self.mU
            self.mP = c.alpha * pn + (1 - c.alpha) * self.mP
            csi = c.w_Q*self.mQ + c.w_D*self.mD + c.w_U*self.mU + c.w_P*self.mP
            self.CSI = float(np.clip(csi, 0, 1))
            self.csi_hist.append(self.CSI)
            if len(self.csi_hist) > c.csi_window:
                self.csi_hist.pop(0)
            mu = np.mean(self.csi_hist); sd = np.std(self.csi_hist)
            self.tau = mu + c.kappa * sd
            # ARCM: proportional rate control driving the fused CSI toward a target.
            # CSI fuses queue, drop, channel-utilisation and PIAT, so this responds to
            # whichever resource (gateway buffer or shared channel) is congesting.
            # (Low-priority flows are additionally throttled by CSI in _attempts.)
            err = self.CSI - c.csi_setpoint
            self.R = np.clip(self.R - c.kp * err, c.R_min, c.R_max)
            # control broadcast reception energy
            self.E_rx += (c.ctrl_bits / c.data_rate_bps) * c.P_rx
        # time series (sample every 10 steps)
        if k % 10 == 0:
            self.ts_qocc.append(self.gwq.sum() / c.gw_buffer_K * 100)
            self.ts_csi.append(self.CSI)
            self.ts_cu.append(CU)
            self.ts_sys.append(self.Q.sum() + self.gwq.sum())
            self.ts_p1.append(self.Q[:, 0].sum() + self.gwq[0])

    def gw_drop_rate(self):
        tot = self.gw_drop + self.tx_success_total
        return self.gw_drop / tot if tot > 0 else 0.0

    def run(self):
        for k in range(self.steps):
            self.step(k)
        return self.results()

    def results(self):
        c = self.cfg
        gen = self.gen.sum(); recv = self.recv.sum()
        pdr = recv / gen if gen > 0 else 0.0
        E = (self.E_tx + self.E_rx + self.E_idle).sum()
        thr_bps = recv * c.packet_bits / c.T_sim
        # mean packets in system (nodes + gateway) -> Little's law mean latency
        Lbar = np.mean(self.ts_sys) if self.ts_sys else 0.0
        thr_pps = recv / c.T_sim
        latency = (Lbar / thr_pps) if thr_pps > 1e-9 else float("nan")
        # critical-traffic (P1) latency via per-class Little's law
        L_p1 = np.mean(self.ts_p1) if self.ts_p1 else 0.0
        thr_p1 = self.recv[0] / c.T_sim
        lat_p1 = (L_p1 / thr_p1) if thr_p1 > 1e-9 else float("nan")
        per_pri_pdr = [ (self.recv[p]/self.gen[p]) if self.gen[p]>0 else float("nan") for p in range(4)]
        return dict(
            protocol=c.protocol, seed=c.seed, n_nodes=c.n_nodes,
            PDR=pdr*100, PLR=(1-pdr)*100,
            latency_ms=latency*1000, latency_p1_ms=lat_p1*1000,
            energy_per_pkt_mJ=(E/recv*1000) if recv>0 else float("nan"),
            throughput_kbps=thr_bps/1000,
            gw_drop=self.gw_drop, node_drop=self.node_drop,
            E_tx=self.E_tx.sum(), E_rx=self.E_rx.sum(), E_idle=self.E_idle.sum(),
            pdr_p1=per_pri_pdr[0]*100, pdr_p2=per_pri_pdr[1]*100,
            pdr_p3=per_pri_pdr[2]*100, pdr_p4=per_pri_pdr[3]*100,
            csi_series=self.ts_csi, q_series=self.ts_qocc,
        )


def run_trial(protocol, seed, **overrides):
    cfg = Config(protocol=protocol, seed=seed, **overrides)
    return WBANSim(cfg).run()


if __name__ == "__main__":
    import time
    for proto in ["NCHA", "M-ATTEMPT", "CICADA", "PRPLC"]:
        t0 = time.time()
        r = run_trial(proto, seed=1, T_sim=30.0, n_nodes=8)
        print(f"{proto:10s} PDR={r['PDR']:.2f}% PLR={r['PLR']:.2f}% "
              f"lat={r['latency_ms']:.1f}ms E/pkt={r['energy_per_pkt_mJ']:.2f}mJ "
              f"thr={r['throughput_kbps']:.1f}kbps P1={r['pdr_p1']:.1f}% P4={r['pdr_p4']:.1f}% "
              f"({time.time()-t0:.1f}s)")
