"""
n1_contingency_study.py  (v2 — adapter-integrated)
===================================================
N-1 Contingency Analysis for Transmission Interconnection Planning
Part of the GOALS framework.

Coupling loop (per contingency)
--------------------------------
  1. Apply outage
  2. Quick NR solve (15 iters) — gets V_pcc and max_line_load
  3. Feed V_pcc + freq into DatacenterPhysics.step()
  4. Update pandapower load element with physics P_mw, Q_mvar
  5. Final NR solve (30 iters) — severity assessment

The physics correctly computes:
  - Frequency-watt droop from COI frequency deviation
  - Volt-VAR droop from V_pcc deviation
  - LVRT/FRT ride-through flag
  - Induction motor speed transient at depressed voltage
  - Price-elastic DR (uses pre-contingency LMP)
"""

from __future__ import annotations
import copy
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import os

import pandapower as pp
import pandapower.networks as pn

from datacenter_registry import register, get_datacenter, deregister
from datacenter_core import DatacenterPhysics, CanonicalInput, CanonicalOutput
from adapters import OPFAdapter

warnings.filterwarnings("ignore")

# =============================================================================
#  CONFIGURATION
# =============================================================================

DC_SIZES: Dict[str, float] = {
    "S0 Baseline (no DC)":   0.0,
    "S1  100 MW DC":       100.0,
    "S2  250 MW DC":       250.0,
    "S3  500 MW DC":       500.0,
}

DC_BUS_IDX   = 15     # 0-indexed in pandapower (= bus 16 in 1-based IEEE 39)
DC_BUS_LABEL = 16

# Physics scaling (same constants as feasibility study)
S_BASE_DC   = 2.0    # MVA — physics model base
LF_NOMINAL  = 0.323  # mean load factor from GPU trace
WARMUP_STEPS= 500    # warm-up steps at dt=0.02 s

GEN_COSTS = {
    0:("Nuclear-b30",18.0,0.003), 1:("Coal-b32",22.0,0.008),
    2:("Coal-b33",21.0,0.008),   3:("CCGT-b34",28.0,0.012),
    4:("Nuclear-b35",16.0,0.003),5:("Hydro-b36",12.0,0.001),
    6:("CCGT-b37",30.0,0.012),   7:("OilGT-b38",55.0,0.040),
    8:("Nuclear-b39",17.0,0.003),"ext":("GasCC-b31",35.0,0.015),
}

LINE_LIMIT_PCT = 100.0
V_MIN_PU       = 0.95
V_MAX_PU       = 1.05
PRE_LMP        = 65.0    # representative LMP for pre-contingency OPF bid

OUT_DIR   = '../Outputs'
OUT_PLOT  = os.path.join(OUT_DIR, 'n1_contingency_results.png') #OUT_DIR / "distribution_dynamics_results.png"
OUT_CSV   = os.path.join(OUT_DIR, 'n1_contingency_data.csv')
OUT_RPT   = os.path.join(OUT_DIR, 'n1_contingency_report.txt')


# =============================================================================
#  DATA CLASS
# =============================================================================

@dataclass
class CResult:
    dc_label:   str
    dc_mw:      float
    c_type:     str
    c_idx:      int
    c_label:    str
    pre_conv:   bool  = False
    pre_cost:   float = 0.0
    pre_V_dc:   float = 1.0
    pre_maxll:  float = 0.0
    post_conv:  bool  = False
    post_V_dc:  float = 1.0
    post_V_min: float = 1.0
    post_Vb_min:int   = 0
    post_maxll: float = 0.0
    n_ol:       int   = 0
    n_vv:       int   = 0
    # Physics-derived DR fields (new in v2)
    dc_P_pre:      float = 0.0
    dc_P_post:     float = 0.0
    dc_Q_inj:      float = 0.0
    dc_dr_active:  float = 0.0
    dc_dP_fw:      float = 0.0   # freq-watt droop contribution [MW]
    dc_dQ_vv:      float = 0.0   # volt-VAR droop contribution [MVAR]
    dc_riding_thru:bool  = False  # LVRT/FRT ride-through flag
    dc_omega_r:    float = 0.97   # motor rotor speed post-contingency [pu]
    severity:   str   = "OK"

    @property
    def is_violation(self): return self.severity in ("VIOLATION","SEVERE")


# =============================================================================
#  NETWORK SETUP
# =============================================================================

def build_network() -> pp.pandapowerNet:
    net = pn.case39()
    net.poly_cost = net.poly_cost.iloc[0:0]
    for i in range(len(net.gen)):
        _, cp1, cp2 = GEN_COSTS[i]
        pp.create_poly_cost(net, i, "gen", cp0_eur=50.0,
                            cp1_eur_per_mw=cp1, cp2_eur_per_mw2=cp2)
    _, cp1e, cp2e = GEN_COSTS["ext"]
    pp.create_poly_cost(net, 0, "ext_grid", cp0_eur=50.0,
                        cp1_eur_per_mw=cp1e, cp2_eur_per_mw2=cp2e)
    net.bus["max_vm_pu"] = V_MAX_PU + 0.01
    net.bus["min_vm_pu"] = V_MIN_PU - 0.01
    return net


def add_dc_load(net: pp.pandapowerNet,
                P_mw: float, Q_mvar: float) -> int:
    """
    Add datacenter as controllable pandapower load.
    P and Q come from OPFAdapter.get_bid() — physics-derived.
    """
    if P_mw <= 0:
        return -1
    idx = pp.create_load(
        net, DC_BUS_IDX, P_mw, Q_mvar,
        name=f"DC_{P_mw:.0f}MW",
        controllable=True,
        max_p_mw=P_mw,
        min_p_mw=P_mw * 0.75,       # 25% min curtailment floor
        max_q_mvar=abs(Q_mvar)*1.5,
        min_q_mvar=0.0,
    )
    pp.create_poly_cost(net, idx, "load", cp0_eur=0.0,
                        cp1_eur_per_mw=-60.0, cp2_eur_per_mw2=0.0)
    return idx


# =============================================================================
#  ADAPTER LIFECYCLE
# =============================================================================

def make_physics(dc_label: str, dc_mw: float) -> Tuple[DatacenterPhysics, float, str]:
    """
    Register datacenter, warm up physics, return (physics, scale, name).
    Scale maps 2 MVA physics output to facility MW rating.
    """
    name = f"N1_{dc_label.replace(' ','_').replace('/','')}"
    cfg  = dict(seed=42, n_cooling_units=3, dt_micro=0.02, dt=0.02,
                price_threshold=55.0, price_max=300.0, max_curtail_pu=0.25)
    register(name, cfg)
    phys: DatacenterPhysics = get_datacenter(name, "opf", interval_min=5)._p
    for w in range(WARMUP_STEPS):
        phys.step(CanonicalInput(V_pu=1.0, freq_hz=60.0,
                                 t_sim=float(w)*0.02, dt=0.02))
    scale = dc_mw / (S_BASE_DC * LF_NOMINAL)
    return phys, scale, name


def physics_dr_response(phys: DatacenterPhysics,
                        scale: float,
                        V_pcc: float,
                        freq_hz: float,
                        t: float) -> CanonicalOutput:
    """
    Call DatacenterPhysics.step() with post-contingency V and f.
    Returns CanonicalOutput whose P_mw, Q_mvar, droop signals and
    LVRT flag are all physics-consistent.
    """
    out = phys.step(CanonicalInput(
        V_pu          = max(V_pcc, 0.01),
        freq_hz       = freq_hz,
        V_d_pu        = max(V_pcc, 0.01),
        V_q_pu        = 0.0,
        price_per_mwh = PRE_LMP,          # same price as pre-contingency bid
        t_sim         = t,
        dt            = 0.1,
    ))
    # Scale all power quantities from 2 MVA base to facility rating
    return CanonicalOutput(
        P_mw            = out.P_mw           * scale,
        Q_mvar          = out.Q_mvar         * scale,
        P_flex_mw       = out.P_flex_mw      * scale,
        P_committed_mw  = out.P_committed_mw * scale,
        dP_droop_mw     = out.dP_droop_mw    * scale,
        dQ_droop_mvar   = out.dQ_droop_mvar  * scale,
        riding_through  = out.riding_through,
        omega_r_pu      = out.omega_r_pu,
        slip            = out.slip,
        V_pcc_pu        = V_pcc,
        freq_hz         = freq_hz,
        P_server_kw     = out.P_server_kw    * scale,
        P_cool_kw       = out.P_cool_kw      * scale,
        vsc_id          = out.vsc_id,
        vsc_iq          = out.vsc_iq,
    )


# =============================================================================
#  CONTINGENCY ENGINE
# =============================================================================

def pre_opf(net_base: pp.pandapowerNet,
            phys: DatacenterPhysics,
            scale: float,
            dc_mw: float) -> Tuple[pp.pandapowerNet, int, bool, float]:
    """
    Pre-contingency OPF:
      1. Get physics bid (P_mw, Q_mvar) from DatacenterPhysics
      2. Scale to facility rating and add as pandapower load
      3. Run pp.runopp() — native AC-OPF handles all dispatch
    """
    net = copy.deepcopy(net_base)

    if dc_mw > 0:
        # Physics-derived bid
        raw_bid = phys.step(CanonicalInput(
            V_pu=1.0, freq_hz=60.0, t_sim=0.0,
            dt=0.1, price_per_mwh=PRE_LMP))
        P_bid = raw_bid.P_mw   * scale
        Q_bid = raw_bid.Q_mvar * scale
        load_idx = add_dc_load(net, P_bid, Q_bid)
    else:
        load_idx = -1
        P_bid    = 0.0

    try:
        pp.runopp(net, numba=False, verbose=False)
        conv = bool(net.OPF_converged)
        cost = float(net.res_cost) if conv else 0.0
    except Exception:
        try:
            pp.runpp(net, numba=False)
            conv = bool(net.converged)
        except Exception:
            conv = False
        cost = 0.0

    return net, load_idx, conv, cost, P_bid


def run_n1(net_pre:   pp.pandapowerNet,
           load_idx:  int,
           phys:      DatacenterPhysics,
           scale:     float,
           dc_mw:     float,
           dc_label:  str,
           c_type:    str,
           c_idx:     int,
           c_label:   str,
           pre_conv:  bool,
           pre_cost:  float,
           dc_P_pre:  float,
           t_cont:    float) -> CResult:
    """
    One N-1 contingency with physics-derived DR response.

    Workflow:
      1. Apply outage on deep copy of pre-contingency network
      2. Quick NR (15 iters) — get V_pcc and max_line_load for DR input
      3. Feed V_pcc + freq into DatacenterPhysics.step() via physics_dr_response()
      4. Update pandapower load element with physics P_mw and Q_mvar
      5. Final NR (30 iters) — severity assessment
    """
    r = CResult(dc_label=dc_label, dc_mw=dc_mw,
                c_type=c_type, c_idx=c_idx, c_label=c_label,
                pre_conv=pre_conv, pre_cost=pre_cost, dc_P_pre=dc_P_pre)

    if pre_conv and len(net_pre.res_bus) > 0:
        r.pre_V_dc  = float(net_pre.res_bus.vm_pu.iloc[DC_BUS_IDX])
        r.pre_maxll = float(net_pre.res_line.loading_percent.max())

    # 1. Apply outage
    net = copy.deepcopy(net_pre)
    if c_type == "line":
        net.line.at[c_idx, "in_service"] = False
    elif c_type == "generator":
        if c_idx < len(net.gen):
            net.gen.at[c_idx, "in_service"] = False
            net.gen.at[c_idx, "p_mw"]       = 0.0
        else:
            net.ext_grid.at[0, "in_service"] = False

    # 2. Quick NR — get post-contingency V_pcc and system frequency
    try:
        pp.runpp(net, numba=False, max_iteration=15)
        init_conv = bool(net.converged)
    except Exception:
        init_conv = False

    if init_conv and load_idx >= 0 and dc_mw > 0:
        V_post  = float(net.res_bus.vm_pu.iloc[DC_BUS_IDX])
        # Estimate post-contingency frequency from generator output change
        gen_loss = (r.pre_maxll - float(net.res_line.loading_percent.max())) / 100.0
        freq_est = 60.0 - 0.3 * max(0.0, gen_loss)   # simplified ROCOF model

        # 3. Physics DR response — V and f from pandapower result
        out: CanonicalOutput = physics_dr_response(
            phys, scale, V_post, freq_est, t_cont)

        # 4. Update pandapower load with physics output
        net.load.at[load_idx, "p_mw"]   = out.P_mw
        net.load.at[load_idx, "q_mvar"] = out.Q_mvar

        r.dc_P_post     = out.P_mw
        r.dc_Q_inj      = out.dQ_droop_mvar
        r.dc_dr_active  = dc_P_pre - out.P_mw
        r.dc_dP_fw      = out.dP_droop_mw
        r.dc_dQ_vv      = out.dQ_droop_mvar
        r.dc_riding_thru= out.riding_through
        r.dc_omega_r    = out.omega_r_pu
    else:
        r.dc_P_post = dc_P_pre

    # 5. Final NR — post-DR power flow
    try:
        pp.runpp(net, numba=False, max_iteration=30)
        r.post_conv = bool(net.converged)
    except Exception:
        r.post_conv = False

    if r.post_conv:
        vm = net.res_bus.vm_pu
        ll = net.res_line.loading_percent
        r.post_V_dc    = float(vm.iloc[DC_BUS_IDX])
        r.post_V_min   = float(vm.min())
        r.post_Vb_min  = int(vm.idxmin())
        r.post_maxll   = float(ll.max())
        r.n_ol         = int((ll > LINE_LIMIT_PCT).sum())
        r.n_vv         = int(((vm < V_MIN_PU) | (vm > V_MAX_PU)).sum())
        if r.n_ol == 0 and r.n_vv == 0:          r.severity = "OK"
        elif r.n_ol <= 1 and r.n_vv <= 2:        r.severity = "WARNING"
        elif r.n_ol <= 3 or  r.n_vv <= 5:        r.severity = "VIOLATION"
        else:                                      r.severity = "SEVERE"
    else:
        r.severity = "SEVERE"

    return r


# =============================================================================
#  STUDY CLASS
# =============================================================================

class N1Study:

    def __init__(self):
        print("\n" + "="*70)
        print("  GOALS — N-1 Contingency Study  (v2 — adapter-integrated)")
        print("  IEEE 39-Bus  ·  pandapower AC-OPF + NR  ·  Bus 16 datacenter")
        print("  DR: DatacenterPhysics → CanonicalOutput → pandapower load")
        print("="*70)

        print("\n[1/4] Building network …")
        self.net_base = build_network()
        pp.runpp(self.net_base, numba=False)
        print(f"      Converged: {self.net_base.converged}  "
              f"Max line load: {self.net_base.res_line.loading_percent.max():.1f}%")

        print("[2/4] Building contingency list …")
        self.contingencies: List[Tuple[str, int, str]] = []
        for i in range(len(self.net_base.line)):
            fb = int(self.net_base.line.from_bus.iloc[i]) + 1
            tb = int(self.net_base.line.to_bus.iloc[i]) + 1
            self.contingencies.append(("line", i, f"L{fb}-{tb}"))
        for i in range(len(self.net_base.gen)):
            b = int(self.net_base.gen.bus.iloc[i]) + 1
            self.contingencies.append(("generator", i, f"G{GEN_COSTS[i][0]}"))
        self.contingencies.append(("generator", len(self.net_base.gen), "G-ExtGrid"))

        n_l = sum(1 for t,_,_ in self.contingencies if t=="line")
        print(f"      {len(self.contingencies)} contingencies: "
              f"{n_l} lines + {len(self.contingencies)-n_l} generators")

        self.results: List[CResult] = []

    # ─────────────────────────────────────────────────────────────────────────
    def run(self):
        n_total = len(DC_SIZES) * len(self.contingencies)
        print(f"\n[3/4] Running {n_total} contingency cases …")
        print(f"      P/Q and DR from DatacenterPhysics via CanonicalInput/Output\n")

        done  = 0
        t_all = time.time()

        for dc_label, dc_mw in DC_SIZES.items():
            print(f"  ── {dc_label} ──")

            # Build physics instance for this DC size
            if dc_mw > 0:
                phys, scale, phys_name = make_physics(dc_label, dc_mw)
            else:
                phys = scale = phys_name = None

            # Pre-contingency OPF
            if dc_mw > 0:
                net_pre, load_idx, pre_conv, pre_cost, P_bid = \
                    pre_opf(self.net_base, phys, scale, dc_mw)
            else:
                net_pre = copy.deepcopy(self.net_base)
                try:
                    pp.runopp(net_pre, numba=False, verbose=False)
                    pre_conv = bool(net_pre.OPF_converged)
                    pre_cost = float(net_pre.res_cost) if pre_conv else 0.0
                except Exception:
                    pp.runpp(net_pre, numba=False)
                    pre_conv = bool(net_pre.converged); pre_cost = 0.0
                load_idx = -1; P_bid = 0.0

            if pre_conv:
                V_dc = (float(net_pre.res_bus.vm_pu.iloc[DC_BUS_IDX])
                        if len(net_pre.res_bus) > 0 else 1.0)
                gen_mw = (float(net_pre.res_gen.p_mw.sum()
                           + net_pre.res_ext_grid.p_mw.sum()))
                print(f"    OPF: cost={pre_cost:,.0f} $/h  "
                      f"V_dc={V_dc:.4f} pu  gen={gen_mw:.0f} MW  "
                      f"P_bid={P_bid:.1f} MW  scale={scale:.1f}" if scale else
                      f"    OPF: cost={pre_cost:,.0f} $/h  V_dc={V_dc:.4f} pu")
            else:
                print("    OPF fallback (NR)")

            counts = {"OK":0,"WARNING":0,"VIOLATION":0,"SEVERE":0}
            t_cont = 0.0
            for c_type, c_idx, c_label in self.contingencies:
                r = run_n1(net_pre, load_idx,
                           phys, scale if scale else 1.0,
                           dc_mw, dc_label,
                           c_type, c_idx, c_label,
                           pre_conv, pre_cost, P_bid, t_cont)
                self.results.append(r)
                counts[r.severity] += 1
                done   += 1
                t_cont += 1.0   # advance physics clock between contingencies
                if done % 30 == 0:
                    el  = time.time() - t_all
                    eta = el / done * (n_total - done)
                    print(f"    [{done}/{n_total}] {el:.0f}s  ETA {eta:.0f}s")

            print("    " + "  ".join(f"{k}={v}" for k,v in counts.items()))
            if phys_name:
                deregister(phys_name)

        print(f"\n  Complete in {time.time()-t_all:.1f}s")

    # ─────────────────────────────────────────────────────────────────────────
    def analyse(self):
        print("\n[4/4] Analysis …\n")
        base_viols = {r.c_label for r in self.results
                      if r.dc_label == "S0 Baseline (no DC)" and r.is_violation}

        print(f"  {'Scenario':<24} {'OK':>5} {'WARN':>5} "
              f"{'VIOL':>5} {'SEVE':>5} {'New vs S0':>10} "
              f"{'MaxDR(MW)':>10} {'LVRTevents':>11}")
        for dc_label in DC_SIZES:
            sub  = [r for r in self.results if r.dc_label == dc_label]
            cnts = {s: sum(1 for r in sub if r.severity==s)
                    for s in ("OK","WARNING","VIOLATION","SEVERE")}
            new  = sum(1 for r in sub
                       if r.is_violation and r.c_label not in base_viols)
            max_dr   = max((r.dc_dr_active for r in sub), default=0.0)
            n_lvrt   = sum(1 for r in sub if r.dc_riding_thru)
            print(f"  {dc_label:<24} {cnts['OK']:>5} {cnts['WARNING']:>5} "
                  f"{cnts['VIOLATION']:>5} {cnts['SEVERE']:>5} "
                  f"{new:>10} {max_dr:>10.1f} {n_lvrt:>11}")

        print("\n  Top 5 worst contingencies (S3 — 500 MW):")
        s3 = [r for r in self.results if r.dc_label == "S3  500 MW DC"]
        worst = sorted(s3, key=lambda r:(
            {"OK":0,"WARNING":1,"VIOLATION":2,"SEVERE":3}[r.severity],
            -r.post_maxll), reverse=True)[:5]
        for r in worst:
            print(f"    {r.c_label:<30} {r.severity:<10}  "
                  f"max_ll={r.post_maxll:.1f}%  V_min={r.post_V_min:.4f}  "
                  f"DR={r.dc_dr_active:.1f} MW  ω_r={r.dc_omega_r:.4f}  "
                  f"LVRT={'YES' if r.dc_riding_thru else 'no'}")

    # ─────────────────────────────────────────────────────────────────────────
    def to_df(self) -> pd.DataFrame:
        return pd.DataFrame([{
            "dc_label":r.dc_label, "dc_mw":r.dc_mw,
            "c_type":r.c_type, "c_idx":r.c_idx, "c_label":r.c_label,
            "pre_conv":r.pre_conv, "pre_cost":r.pre_cost,
            "pre_V_dc":r.pre_V_dc, "pre_maxll":r.pre_maxll,
            "post_conv":r.post_conv, "post_V_dc":r.post_V_dc,
            "post_V_min":r.post_V_min, "post_maxll":r.post_maxll,
            "n_ol":r.n_ol, "n_vv":r.n_vv,
            "dc_P_pre":r.dc_P_pre, "dc_P_post":r.dc_P_post,
            "dc_dr_active":r.dc_dr_active,
            "dc_Q_inj":r.dc_Q_inj,
            "dc_dP_fw":r.dc_dP_fw,
            "dc_dQ_vv":r.dc_dQ_vv,
            "dc_omega_r":r.dc_omega_r,
            "dc_riding_thru":r.dc_riding_thru,
            "severity":r.severity,
        } for r in self.results])

    def save_csv(self):
        self.to_df().to_csv(OUT_CSV, index=False, float_format="%.4f")
        print(f"  CSV  →  {OUT_CSV}")

    # ─────────────────────────────────────────────────────────────────────────
    def plot(self):
        BG, BG2, GRC = "#ffffff", "#f8f9fa", "#dee2e6"
        SEV_COL = {"OK":"#2e7d32","WARNING":"#f9a825",
                   "VIOLATION":"#e65100","SEVERE":"#b71c1c"}
        SEV_NUM = {"OK":0,"WARNING":1,"VIOLATION":2,"SEVERE":3}

        dc_labels   = list(DC_SIZES.keys())
        cont_labels = [cl for _,_,cl in self.contingencies]
        n_dc  = len(dc_labels)
        n_cont= len(cont_labels)
        n_lines = sum(1 for t,_,_ in self.contingencies if t=="line")

        sev_mat = np.zeros((n_dc, n_cont))
        for r in self.results:
            i = dc_labels.index(r.dc_label)
            j = next(k for k,(_,_,cl) in enumerate(self.contingencies)
                     if cl==r.c_label)
            sev_mat[i,j] = SEV_NUM[r.severity]

        fig = plt.figure(figsize=(18, 20))
        fig.patch.set_facecolor(BG)
        gs  = gridspec.GridSpec(3, 2, figure=fig,
                                height_ratios=[2.2,1.4,1.4],
                                hspace=0.52, wspace=0.32,
                                top=0.94, bottom=0.05,
                                left=0.07, right=0.97)
        ax_hm = fig.add_subplot(gs[0,:])
        ax_ol = fig.add_subplot(gs[1,0])
        ax_vv = fig.add_subplot(gs[1,1])
        ax_dr = fig.add_subplot(gs[2,0])
        ax_fw = fig.add_subplot(gs[2,1])   # NEW: freq-watt droop (physics-derived)

        DC_COL = ["#2e7d32","#1565c0","#e65100","#b71c1c"]
        MUT    = "#2c3e50"

        def style(ax, y, t):
            ax.set_facecolor(BG2)
            for sp in ax.spines.values():
                sp.set_edgecolor(GRC); sp.set_linewidth(0.8)
            ax.grid(axis="y", color=GRC, lw=0.5, ls=":")
            ax.set_ylabel(y, fontsize=9)
            ax.set_title(t, fontsize=9.5, loc="left", color=MUT,
                         pad=4, fontweight="bold")
            ax.tick_params(colors=MUT, labelsize=8.5)

        # Heatmap
        cmap = mcolors.LinearSegmentedColormap.from_list(
            "sev",["#2e7d32","#f9a825","#e65100","#b71c1c"],N=4)
        ax_hm.imshow(sev_mat, aspect="auto", cmap=cmap,
                     vmin=0, vmax=3, interpolation="nearest")
        ax_hm.set_yticks(range(n_dc))
        ax_hm.set_yticklabels(dc_labels, fontsize=8.5)
        ax_hm.set_xticks(range(n_cont))
        ax_hm.set_xticklabels(cont_labels, rotation=72,
                               ha="right", fontsize=6)
        ax_hm.axvline(n_lines-0.5, color="white", lw=2, ls="--")
        ax_hm.text(n_lines/2-0.5,-1.4,"← Lines",
                   ha="center",fontsize=8.5,color=MUT)
        ax_hm.text(n_lines+(n_cont-n_lines)/2,-1.4,"Generator Outages →",
                   ha="center",fontsize=8.5,color=MUT)
        legend_p = [Patch(color=SEV_COL[s],label=s)
                    for s in ["OK","WARNING","VIOLATION","SEVERE"]]
        ax_hm.legend(handles=legend_p, fontsize=8.5, loc="upper right",
                     ncol=4, facecolor="white", edgecolor=GRC,
                     bbox_to_anchor=(1.0,1.11))
        ax_hm.set_facecolor(BG2)
        ax_hm.set_title(
            "Panel 1 — N-1 Severity Heatmap  "
            "(rows = DC size, columns = tripped element)  "
            "·  v2: DR from DatacenterPhysics",
            fontsize=10, loc="left", color=MUT, pad=4, fontweight="bold")

        x = np.arange(n_dc)

        # Panel 2: overloads
        n_ol_tot = [sum(r.n_ol for r in self.results if r.dc_label==l)
                    for l in dc_labels]
        base_ol  = {r.c_label for r in self.results
                    if r.dc_label==dc_labels[0] and r.n_ol>0}
        n_new_ol = [sum(1 for r in self.results
                        if r.dc_label==l and r.n_ol>0
                        and r.c_label not in base_ol)
                    for l in dc_labels]
        b1 = ax_ol.bar(x, n_ol_tot, 0.55, color=DC_COL,
                       edgecolor="white", alpha=0.85, label="Total")
        ax_ol.bar(x, n_new_ol, 0.55, color="#ff8f00", alpha=0.75,
                  edgecolor="white", label="New vs baseline")
        for bar,v in zip(b1,n_ol_tot):
            ax_ol.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.3,
                       str(v),ha="center",va="bottom",fontsize=9)
        ax_ol.set_xticks(x)
        ax_ol.set_xticklabels([l[:12] for l in dc_labels],fontsize=8.5)
        ax_ol.legend(fontsize=8,facecolor="white",edgecolor=GRC)
        style(ax_ol,"Overloaded line×cont. pairs","Panel 2 — Thermal Violations")

        # Panel 3: voltage violations
        n_vv_tot = [sum(r.n_vv for r in self.results if r.dc_label==l)
                    for l in dc_labels]
        b2 = ax_vv.bar(x, n_vv_tot, 0.55, color=DC_COL,
                       edgecolor="white", alpha=0.85)
        for bar,v in zip(b2,n_vv_tot):
            ax_vv.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.2,
                       str(v),ha="center",va="bottom",fontsize=9)
        ax_vv.set_xticks(x)
        ax_vv.set_xticklabels([l[:12] for l in dc_labels],fontsize=8.5)
        style(ax_vv,"Bus×cont. voltage violations","Panel 3 — Voltage Violations")

        # Panel 4: DR curtailment (physics-derived in v2)
        for dc_label, col in zip(list(DC_SIZES.keys())[1:],
                                  ["#1565c0","#e65100","#b71c1c"]):
            sub = sorted([r for r in self.results if r.dc_label==dc_label],
                         key=lambda r:-r.dc_dr_active)
            dr_vals = [r.dc_dr_active for r in sub]
            ax_dr.fill_between(range(len(dr_vals)),dr_vals,alpha=0.20,
                               color=col,step="mid")
            ax_dr.step(range(len(dr_vals)),dr_vals,color=col,lw=1.2,
                       where="mid",
                       label=f"{dc_label[:9]}: max {max(dr_vals):.0f} MW")
        ax_dr.axhline(0,color=GRC,lw=0.5)
        ax_dr.set_xlabel("Contingencies sorted by DR magnitude",fontsize=8.5)
        ax_dr.legend(fontsize=8,facecolor="white",edgecolor=GRC)
        style(ax_dr,"DR curtailment (MW)",
              "Panel 4 — Active DR (physics: GPU+IM+VSC price response)")

        # Panel 5: freq-watt droop (new in v2 — physics-derived)
        for dc_label, col in zip(list(DC_SIZES.keys())[1:],
                                  ["#1565c0","#e65100","#b71c1c"]):
            sub = sorted([r for r in self.results if r.dc_label==dc_label],
                         key=lambda r:-abs(r.dc_dP_fw))
            fw_vals = [abs(r.dc_dP_fw) for r in sub]
            ax_fw.fill_between(range(len(fw_vals)),fw_vals,alpha=0.20,
                               color=col,step="mid")
            ax_fw.step(range(len(fw_vals)),fw_vals,color=col,lw=1.2,
                       where="mid",
                       label=f"{dc_label[:9]}: max {max(fw_vals):.2f} MW")
        ax_fw.set_xlabel("Contingencies sorted by F-W magnitude",fontsize=8.5)
        ax_fw.legend(fontsize=8,facecolor="white",edgecolor=GRC)
        style(ax_fw,"|ΔP_fw| (MW)",
              "Panel 5 — Frequency–Watt Droop  (physics: K_fw=0.05 pu/Hz) [v2 only]")

        fig.suptitle(
            "N-1 Contingency Analysis for AI Datacenter Interconnection Planning  "
            "(v2 — adapter-integrated)\n"
            "IEEE 39-Bus  ·  Bus 16  ·  pandapower AC-OPF + NR  "
            "·  DR from DatacenterPhysics → CanonicalOutput → pandapower load",
            fontsize=10, color=MUT, y=0.98, fontweight="bold")
        fig.text(0.07, 0.015,
                 "v2 change: dc_response() replaced by DatacenterPhysics.step() "
                 "receiving V_pcc + freq from pandapower result.  "
                 "Panel 5 (freq-watt droop) is new in v2.",
                 fontsize=7.5, color="#6c757d")

        plt.savefig(OUT_PLOT, dpi=160, bbox_inches="tight", facecolor=BG)
        plt.close()
        print(f"  Figure  →  {OUT_PLOT}")

    # ─────────────────────────────────────────────────────────────────────────
    def report(self):
        sep = "="*70
        base_viols = {r.c_label for r in self.results
                      if r.dc_label=="S0 Baseline (no DC)" and r.is_violation}
        lines = [
            sep,
            "  N-1 CONTINGENCY ANALYSIS  (v2 — adapter-integrated)",
            "  IEEE 39-Bus New England  ·  pandapower AC-OPF  ·  Bus 16 POI",
            sep, "",
            "  v2 ARCHITECTURE CHANGE:",
            "  Pre-contingency: OPFAdapter.get_bid() provides physics P/Q bid",
            "  Post-contingency: DatacenterPhysics.step(V_pcc, freq) returns",
            "    P_mw, Q_mvar, dP_droop_mw, dQ_droop_mvar, riding_through, omega_r",
            "  These are fed directly into the pandapower load element.",
            "",
        ]
        for dc_label, dc_mw in DC_SIZES.items():
            sub  = [r for r in self.results if r.dc_label==dc_label]
            cnts = {s:sum(1 for r in sub if r.severity==s)
                    for s in ("OK","WARNING","VIOLATION","SEVERE")}
            new  = [r for r in sub
                    if r.is_violation and r.c_label not in base_viols]
            cost = next((r.pre_cost for r in sub if r.pre_cost>0),0.0)
            V_dc = next((r.pre_V_dc for r in sub if r.pre_V_dc>0),1.0)
            max_dr   = max((r.dc_dr_active for r in sub),default=0.0)
            max_fw   = max((abs(r.dc_dP_fw) for r in sub),default=0.0)
            n_lvrt   = sum(1 for r in sub if r.dc_riding_thru)
            lines += [
                f"  {'─'*66}",
                f"  {dc_label}",
                f"    OPF cost:          {cost:,.0f} $/h",
                f"    Pre-contingency V: {V_dc:.4f} pu at bus {DC_BUS_LABEL}",
                f"    N-1 results:       "+
                "  ".join(f"{k}={v}" for k,v in cnts.items()),
                f"    New violations vs S0: {len(new)}",
                f"    Max DR (active):   {max_dr:.1f} MW  (physics-derived)",
                f"    Max F-W droop:     {max_fw:.3f} MW  (physics-derived, new in v2)",
                f"    LVRT events:       {n_lvrt}",
            ]
            if new:
                lines.append("    New violations:")
                for r in sorted(new,key=lambda r:-r.post_maxll)[:3]:
                    lines.append(
                        f"      {r.c_label:<30} "
                        f"max_ll={r.post_maxll:.1f}%  "
                        f"V_min={r.post_V_min:.4f}  "
                        f"DR={r.dc_dr_active:.1f} MW  "
                        f"ω_r={r.dc_omega_r:.4f}")
            lines.append("")
        lines += [
            sep,
            f"  Figure: {OUT_PLOT}",
            f"  Data:   {OUT_CSV}",
            sep,
        ]
        text = "\n".join(lines)
        OUT_RPT.write_text(text)
        print(text)
        print(f"\n  Report  →  {OUT_RPT}")


# =============================================================================
#  ENTRY POINT
# =============================================================================

def main():
    study = N1Study()
    study.run()
    study.analyse()
    study.save_csv()
    study.plot()
    study.report()
    print("\nDone.")

if __name__ == "__main__":
    main()
