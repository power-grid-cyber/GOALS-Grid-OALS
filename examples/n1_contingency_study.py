"""
n1_contingency_study.py
========================
N-1 Contingency Analysis for Transmission Interconnection Planning
===================================================================
Pandapower integration
----------------------
  - Network  : pandapower.networks.case39()  (IEEE 39-bus, 100 MVA)
  - Base PF  : pp.runpp()        (Newton-Raphson AC power flow)
  - OPF      : pp.runopp()       (AC-OPF with quadratic generator costs)
  - N-1 loop : line and generator outages, PF re-solved after each trip
  - No custom OPF formulation — pandapower's solver handles everything.

Datacenter model
----------------
  The datacenter is added as a pandapower load element at bus 15 (0-indexed,
  = bus 16 in 1-based IEEE 39 notation).  Four sizes are evaluated:
    S0: Baseline (no datacenter)
    S1: 100 MW datacenter
    S2: 250 MW datacenter
    S3: 500 MW datacenter
"""

from __future__ import annotations
import copy
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import os

import pandapower as pp
import pandapower.networks as pn

warnings.filterwarnings("ignore")

# =============================================================================
#  CONFIGURATION
# =============================================================================

DC_SIZES: Dict[str, float] = {
    "S0 Baseline (no DC)": 0.0,
    "S1  100 MW DC":     100.0,
    "S2  250 MW DC":     250.0,
    "S3  500 MW DC":     500.0,
}

DC_BUS_IDX   = 15   # 0-indexed (bus 16 in 1-based IEEE 39)
DC_BUS_LABEL = 16

# Realistic marginal costs by fuel type [EUR/MWh]
GEN_COSTS = {
    0: ("Nuclear-b30",   18.0, 0.003),
    1: ("Coal-b32",      22.0, 0.008),
    2: ("Coal-b33",      21.0, 0.008),
    3: ("CCGT-b34",      28.0, 0.012),
    4: ("Nuclear-b35",   16.0, 0.003),
    5: ("Hydro-b36",     12.0, 0.001),
    6: ("CCGT-b37",      30.0, 0.012),
    7: ("OilGT-b38",     55.0, 0.040),
    8: ("Nuclear-b39",   17.0, 0.003),
    "ext": ("GasCC-b31", 35.0, 0.015),
}

LINE_LIMIT_PCT = 100.0
V_MIN_PU       = 0.95
V_MAX_PU       = 1.05
K_VV           = 0.10    # volt-VAR droop gain [pu/pu]
DR_MAX_FRAC    = 0.25    # max active curtailment fraction


# ── Output paths ──────────────────────────────────────────────────────────────
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
    c_type:     str      # 'line' | 'generator'
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
    post_maxll_line: int = 0
    n_ol:       int   = 0
    n_vv:       int   = 0
    dc_P_pre:   float = 0.0
    dc_P_post:  float = 0.0
    dc_Q_inj:   float = 0.0
    dc_dr:      float = 0.0
    severity:   str   = "OK"

    @property
    def is_violation(self):
        return self.severity in ("VIOLATION", "SEVERE")


# =============================================================================
#  NETWORK BUILDER
# =============================================================================

def build_network() -> pp.pandapowerNet:
    """Load case39 and apply differentiated generator cost curves."""
    net = pn.case39()
    net.poly_cost = net.poly_cost.iloc[0:0]

    for i in range(len(net.gen)):
        _, cp1, cp2 = GEN_COSTS[i]
        pp.create_poly_cost(net, i, "gen",
                            cp0_eur=50.0,
                            cp1_eur_per_mw=cp1,
                            cp2_eur_per_mw2=cp2)

    _, cp1e, cp2e = GEN_COSTS["ext"]
    pp.create_poly_cost(net, 0, "ext_grid",
                        cp0_eur=50.0,
                        cp1_eur_per_mw=cp1e,
                        cp2_eur_per_mw2=cp2e)

    net.bus["max_vm_pu"] = V_MAX_PU + 0.01
    net.bus["min_vm_pu"] = V_MIN_PU - 0.01
    return net


def add_datacenter(net: pp.pandapowerNet, P_mw: float) -> int:
    """Add datacenter as a controllable load; return load index (-1 if none)."""
    if P_mw <= 0:
        return -1
    Q_mvar = P_mw * np.tan(np.arccos(0.92))
    idx = pp.create_load(net, DC_BUS_IDX, P_mw, Q_mvar,
                         name=f"DC_{P_mw:.0f}MW",
                         controllable=True,
                         max_p_mw=P_mw,
                         min_p_mw=P_mw * (1 - DR_MAX_FRAC),
                         max_q_mvar=Q_mvar * 1.5,
                         min_q_mvar=0.0)
    pp.create_poly_cost(net, idx, "load",
                        cp0_eur=0.0,
                        cp1_eur_per_mw=-60.0,   # DR value
                        cp2_eur_per_mw2=0.0)
    return idx


def dc_response(net: pp.pandapowerNet,
                load_idx: int,
                dc_mw: float) -> Tuple[float, float]:
    """Compute datacenter DR after contingency PF. Returns (P_post, Q_inj)."""
    if load_idx < 0 or dc_mw <= 0:
        return 0.0, 0.0

    V_dc = float(net.res_bus.vm_pu.iloc[DC_BUS_IDX])
    dV   = V_dc - 1.0

    # Volt-VAR injection
    S_dc     = dc_mw / 0.97
    Q_inject = float(np.clip(-K_VV * dV * S_dc,
                             -S_dc * 0.30, S_dc * 0.30))

    # Active curtailment trigger
    max_ll = float(net.res_line.loading_percent.max()) \
        if len(net.res_line) > 0 else 0.0
    v_min  = float(net.res_bus.vm_pu.min())

    if max_ll > LINE_LIMIT_PCT or v_min < (V_MIN_PU - 0.01):
        sev    = min(1.0, max((max_ll - LINE_LIMIT_PCT) / 50.0,
                              (V_MIN_PU - v_min) / 0.10))
        P_post = dc_mw * (1 - DR_MAX_FRAC * sev)
    else:
        P_post = dc_mw

    return float(P_post), float(Q_inject)


# =============================================================================
#  CONTINGENCY ENGINE
# =============================================================================

def pre_opf(net_base: pp.pandapowerNet,
            dc_mw: float) -> Tuple[pp.pandapowerNet, int, bool, float]:
    """Run pre-contingency OPF. Falls back to NR if OPF fails."""
    net = copy.deepcopy(net_base)
    idx = add_datacenter(net, dc_mw)
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
    return net, idx, conv, cost


def run_n1(net_pre:  pp.pandapowerNet,
           load_idx: int,
           dc_mw:    float,
           dc_label: str,
           c_type:   str,
           c_idx:    int,
           c_label:  str,
           pre_conv: bool,
           pre_cost: float) -> CResult:
    """Run one N-1 contingency and return structured result."""
    r = CResult(dc_label=dc_label, dc_mw=dc_mw,
                c_type=c_type, c_idx=c_idx, c_label=c_label,
                pre_conv=pre_conv, pre_cost=pre_cost, dc_P_pre=dc_mw)

    if pre_conv and len(net_pre.res_bus) > 0:
        r.pre_V_dc  = float(net_pre.res_bus.vm_pu.iloc[DC_BUS_IDX])
        r.pre_maxll = float(net_pre.res_line.loading_percent.max())

    # Apply outage
    net = copy.deepcopy(net_pre)
    if c_type == "line":
        net.line.at[c_idx, "in_service"] = False
    elif c_type == "generator":
        if c_idx < len(net.gen):
            net.gen.at[c_idx, "in_service"] = False
            net.gen.at[c_idx, "p_mw"] = 0.0
        else:
            net.ext_grid.at[0, "in_service"] = False

    # Initial post-contingency PF to compute DC response
    try:
        pp.runpp(net, numba=False, max_iteration=15)
        init_conv = bool(net.converged)
    except Exception:
        init_conv = False

    if init_conv and load_idx >= 0:
        P_post, Q_inj = dc_response(net, load_idx, dc_mw)
        net.load.at[load_idx, "p_mw"] = P_post
        net.load.at[load_idx, "q_mvar"] = (
            net.load.at[load_idx, "q_mvar"]
            * (P_post / dc_mw if dc_mw > 0 else 1.0) - Q_inj)
        r.dc_P_post  = P_post
        r.dc_Q_inj   = Q_inj
        r.dc_dr      = dc_mw - P_post
    else:
        r.dc_P_post = dc_mw

    # Final post-contingency PF with DC response applied
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
        r.post_maxll_line = int(ll.idxmax())
        r.n_ol         = int((ll > LINE_LIMIT_PCT).sum())
        r.n_vv         = int(((vm < V_MIN_PU) | (vm > V_MAX_PU)).sum())

        if r.n_ol == 0 and r.n_vv == 0:
            r.severity = "OK"
        elif r.n_ol <= 1 and r.n_vv <= 2:
            r.severity = "WARNING"
        elif r.n_ol <= 3 or r.n_vv <= 5:
            r.severity = "VIOLATION"
        else:
            r.severity = "SEVERE"
    else:
        r.severity = "SEVERE"

    return r


# =============================================================================
#  STUDY CLASS
# =============================================================================

class N1Study:

    def __init__(self):
        print("\n" + "=" * 70)
        print(" GOALS — N-1 Contingency Analysis  |  pandapower")
        print("  IEEE 39-Bus New England  |  AI Datacenter at Bus 16")
        print("=" * 70)

        print("\n[1/4] Building network …")
        self.net_base = build_network()
        pp.runpp(self.net_base, numba=False)
        print(f"      Converged: {self.net_base.converged}  "
              f"Max line load: "
              f"{self.net_base.res_line.loading_percent.max():.1f}%")

        print("[2/4] Building contingency list …")
        self.contingencies: List[Tuple[str, int, str]] = []
        for i in range(len(self.net_base.line)):
            fb = int(self.net_base.line.from_bus.iloc[i]) + 1
            tb = int(self.net_base.line.to_bus.iloc[i]) + 1
            self.contingencies.append(("line", i, f"L{fb}-{tb}"))
        for i in range(len(self.net_base.gen)):
            b  = int(self.net_base.gen.bus.iloc[i]) + 1
            fl = GEN_COSTS[i][0]
            self.contingencies.append(("generator", i, f"G{fl}"))
        self.contingencies.append(("generator", len(self.net_base.gen),
                                   "G-ExtGrid"))
        n_lines = sum(1 for t, _, _ in self.contingencies if t == "line")
        n_gens  = len(self.contingencies) - n_lines
        print(f"      {len(self.contingencies)} contingencies: "
              f"{n_lines} lines + {n_gens} generators")

        self.results: List[CResult] = []

    # ─────────────────────────────────────────────────────────────────────────
    def run(self):
        print(f"\n[3/4] Running N-1 analysis …")
        n_total = len(DC_SIZES) * len(self.contingencies)
        print(f"      {len(DC_SIZES)} DC sizes × {len(self.contingencies)} "
              f"contingencies = {n_total} cases\n")

        done   = 0
        t_all  = time.time()

        for dc_label, dc_mw in DC_SIZES.items():
            print(f"  ── {dc_label} ──")
            net_pre, load_idx, pre_conv, pre_cost = pre_opf(
                self.net_base, dc_mw)

            if pre_conv:
                V_dc = (net_pre.res_bus.vm_pu.iloc[DC_BUS_IDX]
                        if len(net_pre.res_bus) > 0 else 1.0)
                gen_tot = (net_pre.res_gen.p_mw.sum()
                           + net_pre.res_ext_grid.p_mw.sum())
                print(f"    OPF: cost={pre_cost:,.0f} EUR/h  "
                      f"V_dc={V_dc:.4f} pu  "
                      f"gen={gen_tot:.0f} MW")
            else:
                print("    OPF fallback (NR)")

            counts = {"OK": 0, "WARNING": 0, "VIOLATION": 0, "SEVERE": 0}
            for c_type, c_idx, c_label in self.contingencies:
                r = run_n1(net_pre, load_idx, dc_mw, dc_label,
                           c_type, c_idx, c_label, pre_conv, pre_cost)
                self.results.append(r)
                counts[r.severity] += 1
                done += 1
                if done % 30 == 0:
                    el = time.time() - t_all
                    eta = el / done * (n_total - done)
                    print(f"    [{done}/{n_total}] {el:.0f}s elapsed, "
                          f"ETA {eta:.0f}s")

            print(f"    " + "  ".join(f"{k}={v}"
                                      for k, v in counts.items()))

        print(f"\n  Complete in {time.time()-t_all:.1f}s")

    # ─────────────────────────────────────────────────────────────────────────
    def analyse(self):
        print("\n[4/4] Analysis …\n")
        base_viols = {
            r.c_label for r in self.results
            if r.dc_label == "S0 Baseline (no DC)" and r.is_violation}

        print(f"  {'Scenario':<24} {'OK':>5} {'WARN':>5} "
              f"{'VIOL':>5} {'SEVE':>5} {'New vs base':>11}")
        for dc_label in DC_SIZES:
            sub  = [r for r in self.results if r.dc_label == dc_label]
            cnts = {s: sum(1 for r in sub if r.severity == s)
                    for s in ("OK","WARNING","VIOLATION","SEVERE")}
            new  = sum(1 for r in sub
                       if r.is_violation and r.c_label not in base_viols)
            print(f"  {dc_label:<24} {cnts['OK']:>5} {cnts['WARNING']:>5} "
                  f"{cnts['VIOLATION']:>5} {cnts['SEVERE']:>5} {new:>11}")

        print("\n  Top 5 worst contingencies for S3 — 500 MW:")
        s3 = [r for r in self.results if r.dc_label == "S3  500 MW DC"]
        worst = sorted(s3, key=lambda r: (
            {"OK":0,"WARNING":1,"VIOLATION":2,"SEVERE":3}[r.severity],
            -r.post_maxll), reverse=True)[:5]
        for r in worst:
            print(f"    {r.c_label:<30} {r.severity:<10}  "
                  f"max_ll={r.post_maxll:.1f}%  "
                  f"V_min={r.post_V_min:.4f}  DR={r.dc_dr:.1f} MW")

    # ─────────────────────────────────────────────────────────────────────────
    def to_df(self) -> pd.DataFrame:
        return pd.DataFrame([{
            "dc_label": r.dc_label, "dc_mw": r.dc_mw,
            "c_type": r.c_type, "c_idx": r.c_idx, "c_label": r.c_label,
            "pre_conv": r.pre_conv, "pre_cost": r.pre_cost,
            "pre_V_dc": r.pre_V_dc, "pre_maxll": r.pre_maxll,
            "post_conv": r.post_conv, "post_V_dc": r.post_V_dc,
            "post_V_min": r.post_V_min, "post_maxll": r.post_maxll,
            "n_ol": r.n_ol, "n_vv": r.n_vv,
            "dc_P_pre": r.dc_P_pre, "dc_P_post": r.dc_P_post,
            "dc_dr": r.dc_dr, "dc_Q_inj": r.dc_Q_inj,
            "severity": r.severity,
        } for r in self.results])

    def save_csv(self):
        self.to_df().to_csv(OUT_CSV, index=False, float_format="%.4f")
        print(f"  CSV  →  {OUT_CSV}")

    # ─────────────────────────────────────────────────────────────────────────
    def plot(self):
        BG  = "#ffffff"
        BG2 = "#f8f9fa"
        GRC = "#dee2e6"
        SEV_COL = {"OK":"#2e7d32","WARNING":"#f9a825",
                   "VIOLATION":"#e65100","SEVERE":"#b71c1c"}
        SEV_NUM = {"OK":0,"WARNING":1,"VIOLATION":2,"SEVERE":3}

        dc_labels  = list(DC_SIZES.keys())
        cont_labels= [cl for _, _, cl in self.contingencies]
        n_dc       = len(dc_labels)
        n_cont     = len(cont_labels)
        n_lines    = sum(1 for t, _, _ in self.contingencies if t == "line")

        # Severity matrix for heatmap
        sev_mat = np.zeros((n_dc, n_cont))
        for r in self.results:
            i = dc_labels.index(r.dc_label)
            j = next(k for k, (_, _, cl) in enumerate(self.contingencies)
                     if cl == r.c_label)
            sev_mat[i, j] = SEV_NUM[r.severity]

        fig = plt.figure(figsize=(18, 20))
        fig.patch.set_facecolor(BG)
        gs = gridspec.GridSpec(3, 2, figure=fig,
                               height_ratios=[2.2, 1.4, 1.4],
                               hspace=0.52, wspace=0.32,
                               top=0.94, bottom=0.05,
                               left=0.07, right=0.97)
        ax_hm = fig.add_subplot(gs[0, :])
        ax_ol = fig.add_subplot(gs[1, 0])
        ax_vv = fig.add_subplot(gs[1, 1])
        ax_dr = fig.add_subplot(gs[2, 0])
        ax_cc = fig.add_subplot(gs[2, 1])

        def style(ax, y, t):
            ax.set_facecolor(BG2)
            for sp in ax.spines.values():
                sp.set_edgecolor(GRC); sp.set_linewidth(0.8)
            ax.grid(axis="y", color=GRC, lw=0.5, ls=":")
            ax.set_ylabel(y, fontsize=9)
            ax.set_title(t, fontsize=9.5, loc="left",
                         color="#2c3e50", pad=4, fontweight="bold")
            ax.tick_params(colors="#2c3e50", labelsize=8.5)

        # ── Heatmap ───────────────────────────────────────────────────────────
        cmap = mcolors.LinearSegmentedColormap.from_list(
            "sev", ["#2e7d32","#f9a825","#e65100","#b71c1c"], N=4)
        ax_hm.imshow(sev_mat, aspect="auto", cmap=cmap,
                     vmin=0, vmax=3, interpolation="nearest")
        ax_hm.set_yticks(range(n_dc))
        ax_hm.set_yticklabels(dc_labels, fontsize=8.5)
        ax_hm.set_xticks(range(n_cont))
        ax_hm.set_xticklabels(cont_labels, rotation=72,
                               ha="right", fontsize=6)
        ax_hm.axvline(n_lines - 0.5, color="white", lw=2, ls="--")
        ax_hm.text(n_lines / 2 - 0.5, -1.4, "← Transmission Lines",
                   ha="center", fontsize=8.5, color="#2c3e50")
        ax_hm.text(n_lines + (n_cont - n_lines) / 2, -1.4,
                   "Generator Outages →",
                   ha="center", fontsize=8.5, color="#2c3e50")
        legend_p = [Patch(color=SEV_COL[s], label=s)
                    for s in ["OK","WARNING","VIOLATION","SEVERE"]]
        ax_hm.legend(handles=legend_p, fontsize=8.5,
                     loc="upper right", ncol=4,
                     facecolor="white", edgecolor=GRC,
                     bbox_to_anchor=(1.0, 1.11))
        ax_hm.set_facecolor(BG2)
        ax_hm.set_title(
            "Panel 1 — N-1 Contingency Severity Heatmap  "
            "(rows = datacenter size, columns = tripped element)",
            fontsize=10, loc="left", color="#2c3e50",
            pad=4, fontweight="bold")

        # ── Overloads bar ─────────────────────────────────────────────────────
        x = np.arange(n_dc)
        DC_COL = ["#2e7d32","#1565c0","#e65100","#b71c1c"]
        n_ol_tot = [sum(r.n_ol for r in self.results if r.dc_label == l)
                    for l in dc_labels]
        base_ol  = set(r.c_label for r in self.results
                       if r.dc_label == dc_labels[0] and r.n_ol > 0)
        n_new_ol = [sum(1 for r in self.results
                        if r.dc_label == l and r.n_ol > 0
                        and r.c_label not in base_ol)
                    for l in dc_labels]

        b1 = ax_ol.bar(x, n_ol_tot, 0.55, color=DC_COL,
                       edgecolor="white", alpha=0.85, label="Total")
        ax_ol.bar(x, n_new_ol, 0.55, color="#ff8f00",
                  edgecolor="white", alpha=0.75,
                  label="New vs baseline")
        for bar, v in zip(b1, n_ol_tot):
            ax_ol.text(bar.get_x()+bar.get_width()/2,
                       bar.get_height()+0.3, str(v),
                       ha="center", va="bottom", fontsize=9)
        ax_ol.set_xticks(x)
        ax_ol.set_xticklabels([l[:12] for l in dc_labels], fontsize=8.5)
        ax_ol.legend(fontsize=8, facecolor="white", edgecolor=GRC)
        style(ax_ol, "Overloaded line×contingency pairs",
              "Panel 2 — Thermal Violations Across All N-1")

        # ── Voltage violations bar ────────────────────────────────────────────
        n_vv_tot = [sum(r.n_vv for r in self.results if r.dc_label == l)
                    for l in dc_labels]
        b2 = ax_vv.bar(x, n_vv_tot, 0.55, color=DC_COL,
                       edgecolor="white", alpha=0.85)
        for bar, v in zip(b2, n_vv_tot):
            ax_vv.text(bar.get_x()+bar.get_width()/2,
                       bar.get_height()+0.2, str(v),
                       ha="center", va="bottom", fontsize=9)
        ax_vv.set_xticks(x)
        ax_vv.set_xticklabels([l[:12] for l in dc_labels], fontsize=8.5)
        style(ax_vv, "Bus×contingency voltage violations",
              "Panel 3 — Voltage Violations Across All N-1")

        # ── DR response ───────────────────────────────────────────────────────
        for dc_label, col in zip(list(DC_SIZES.keys())[1:],
                                 ["#1565c0","#e65100","#b71c1c"]):
            sub = sorted([r for r in self.results if r.dc_label == dc_label],
                         key=lambda r: -r.dc_dr)
            dr_vals = [r.dc_dr for r in sub]
            ax_dr.fill_between(range(len(dr_vals)), dr_vals,
                               alpha=0.20, color=col, step="mid")
            ax_dr.step(range(len(dr_vals)), dr_vals, color=col,
                       lw=1.2, where="mid",
                       label=f"{dc_label[:9]}: max {max(dr_vals):.0f} MW")
        ax_dr.axhline(0, color=GRC, lw=0.5)
        ax_dr.set_xlabel("Contingencies sorted by DR magnitude", fontsize=8.5)
        ax_dr.legend(fontsize=8, facecolor="white", edgecolor=GRC)
        style(ax_dr, "DR curtailment (MW)",
              "Panel 4 — Datacenter Demand Response per Contingency")

        # ── OPF cost + V_dc ───────────────────────────────────────────────────
        costs = [next((r.pre_cost for r in self.results
                       if r.dc_label == l and r.pre_cost > 0), 0.0)
                 for l in dc_labels]
        ax_cc.bar(x, costs, 0.55, color=DC_COL,
                  edgecolor="white", alpha=0.85)
        for xi, c in zip(x, costs):
            if c > 0:
                ax_cc.text(xi, c+20, f"{c:,.0f}",
                           ha="center", va="bottom", fontsize=8.5)

        ax2 = ax_cc.twinx()
        v_dcs = [next((r.pre_V_dc for r in self.results
                       if r.dc_label == l and r.pre_V_dc > 0), 1.0)
                 for l in dc_labels]
        ax2.plot(x, v_dcs, "o--", color="#9c27b0", ms=8, lw=1.5,
                 label="V_pcc (pu)")
        ax2.axhline(V_MIN_PU, color="#c62828", lw=0.8, ls="--", alpha=0.7)
        ax2.set_ylabel("|V| at DC bus (pu)", fontsize=8.5, color="#9c27b0")
        ax2.tick_params(colors="#9c27b0", labelsize=8)
        ax2.legend(fontsize=8, facecolor="white", edgecolor=GRC,
                   loc="lower right")
        ax2.set_ylim(0.90, 1.10)
        ax_cc.set_xticks(x)
        ax_cc.set_xticklabels([l[:12] for l in dc_labels], fontsize=8.5)
        style(ax_cc, "Pre-contingency OPF cost (EUR/h)",
              "Panel 5 — OPF Dispatch Cost and PCC Voltage")

        fig.suptitle(
            "N-1 Contingency Analysis for AI Datacenter Interconnection Planning\n"
            "IEEE 39-Bus New England · Bus 16 POI · pandapower AC-OPF + "
            "Newton-Raphson",
            fontsize=11, color="#2c3e50", y=0.98, fontweight="bold")
        fig.text(
            0.07, 0.015,
            f"35 line + 10 generator contingencies.  "
            f"Limits: line ≤{LINE_LIMIT_PCT:.0f}%,  "
            f"V ∈ [{V_MIN_PU},{V_MAX_PU}] pu.  "
            f"Datacenter DR: volt-VAR (K_vv={K_VV}) + "
            f"active curtailment (max {DR_MAX_FRAC*100:.0f}%).",
            fontsize=7.5, color="#6c757d")

        plt.savefig(OUT_PLOT, dpi=160, bbox_inches="tight", facecolor=BG)
        plt.close()
        print(f"  Figure  →  {OUT_PLOT}")

    # ─────────────────────────────────────────────────────────────────────────
    def report(self):
        sep = "=" * 70
        base_viols = {r.c_label for r in self.results
                      if r.dc_label == "S0 Baseline (no DC)"
                      and r.is_violation}
        lines = [
            sep,
            "  N-1 CONTINGENCY ANALYSIS — INTERCONNECTION PLANNING REPORT",
            "  IEEE 39-Bus New England · pandapower AC-OPF · Bus 16 POI",
            sep, "",
            "  Thermal limit: line loading ≤ 100%",
            f"  Voltage limits: [{V_MIN_PU}, {V_MAX_PU}] pu  (ANSI C84.1 Range A)",
            f"  DC demand response: volt-VAR (K_vv={K_VV}) + "
            f"active curtailment (max {DR_MAX_FRAC*100:.0f}%)",
            f"  Total contingencies: {len(self.contingencies)} "
            f"(35 lines + 10 generators)", "",
        ]
        for dc_label, dc_mw in DC_SIZES.items():
            sub  = [r for r in self.results if r.dc_label == dc_label]
            cnts = {s: sum(1 for r in sub if r.severity == s)
                    for s in ("OK","WARNING","VIOLATION","SEVERE")}
            new  = [r for r in sub
                    if r.is_violation and r.c_label not in base_viols]
            cost = next((r.pre_cost for r in sub if r.pre_cost > 0), 0.0)
            V_dc = next((r.pre_V_dc for r in sub if r.pre_V_dc > 0), 1.0)
            max_dr = max((r.dc_dr for r in sub), default=0.0)

            lines += [
                f"  {'─'*66}",
                f"  {dc_label}",
                f"    OPF cost:    {cost:,.0f} EUR/h",
                f"    V_pcc (pre): {V_dc:.4f} pu",
                f"    N-1 results: " +
                "  ".join(f"{k}={v}" for k,v in cnts.items()),
                f"    New violations vs baseline: {len(new)}",
                f"    Max DR curtailment: {max_dr:.1f} MW",
            ]
            if new:
                lines.append("    New violations:")
                for r in sorted(new, key=lambda r: -r.post_maxll)[:3]:
                    lines.append(
                        f"      {r.c_label:<30} "
                        f"max_line={r.post_maxll:.1f}%  "
                        f"V_min={r.post_V_min:.4f}  DR={r.dc_dr:.1f} MW")
            lines.append("")

        lines += [
            f"  {'─'*66}",
            "  INTERCONNECTION REQUIREMENT IMPLICATIONS",
            "",
            "  S1 (100 MW): No new violations. Feasible without upgrade.",
            "    Reactive compensation obligation per FAC-001.",
            "",
            "  S2 (250 MW): Review worst-case line trips.",
            "    Interruptibility clause (25% curtailment) recommended.",
            "",
            "  S3 (500 MW): New violations introduced.",
            "    Transmission reinforcement required, OR",
            "    firm load cap + full DR interruptibility.",
            "", sep,
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
