"""
feasibility_study.py  (v2 — adapter-integrated)
================================================
Transmission Interconnection Feasibility Study
Part of the GOALS framework.



Feasibility criteria
--------------------
  1. Thermal   — area import vs. 1,200 MW interface limit
  2. Voltage   — bus 16 PCC vs. ANSI C84.1 Range A [0.95, 1.05] pu
  3. Economic  — congestion premium at bus 16 [$/MWh]

Datacenter sizes:  DC1 100 MW | DC2 230 MW | DC3 430 MW | DC4 500 MW
"""

from __future__ import annotations
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import os

from testsystems.transmission_network import TransmissionNetworkSimulator
from datacenter_registry import register, get_datacenter, deregister
from adapters import OPFAdapter
from datacenter_core import CanonicalInput

# =============================================================================
#  CONFIGURATION
# =============================================================================

DT_MIN      = 60           # hourly resolution — adequate for screening
DT_S        = DT_MIN * 60
N_INTERVALS = 24
HOURS       = np.arange(N_INTERVALS, dtype=float)

S_BASE_MVA = 100.0         # transmission system base

INTERFACE_LIMIT_MW = 1200.0
V_MIN_PU           = 0.95
V_MAX_PU           = 1.05

# Pre-existing area loads (buses 15, 16, 18) before datacenter
AREA_BASE_MW = 329.0 + 320.0 + 158.0  # = 807 MW

# Thevenin at bus 16 — from MATPOWER case39 Y-bus inverse
R_TH = 0.012   # pu / 100 MVA
X_TH = 0.035   # pu / 100 MVA
V0   = 1.032   # pu — MATPOWER operating point

# Congestion
CONGESTION_PASSTHROUGH = 0.80   # LMP excess pass-through fraction
DR_THRESHOLD           = 65.0   # $/MWh

# Physics scaling constants
S_BASE_DC   = 2.0    # MVA — physics model base rating
LF_NOMINAL  = 0.323  # mean GPU trace load factor (empirical, full trace)
WARMUP_STEPS = 430   # 430 × 0.02 s = 10 s warm-up

DC_CONFIGS: Dict[str, float] = {
    "DC1 — 100 MW":    100.0,
    "DC2 — 230 MW":    230.0,
    "DC3 — 430 MW":    430.0,
    "DC4 — 500 MW": 500.0,
}

# ── Output paths ──────────────────────────────────────────────────────────────
OUT_DIR   = '../Outputs'
OUT_PLOT  = os.path.join(OUT_DIR, 'feasibility_hosting_capacity.png') #OUT_DIR / "distribution_dynamics_results.png"
OUT_RPT   = os.path.join(OUT_DIR, 'feasibility_report.txt')


# =============================================================================
#  LMP PROFILE
# =============================================================================

def build_lmp_profile() -> np.ndarray:
    rng = np.random.default_rng(17)
    h   = HOURS
    lmp = (32.0
           + 53.0 * np.exp(-0.5 * ((h -  8.5) / 1.2) ** 2)
           + 68.0 * np.exp(-0.5 * ((h - 18.5) / 1.0) ** 2)
           - 22.0 * np.exp(-0.5 * ((h - 12.5) / 2.0) ** 2)
           +  8.0 * rng.normal(0, 1, N_INTERVALS))
    return np.clip(lmp, 18.0, 220.0)


# =============================================================================
#  RESULT DATACLASS
# =============================================================================

@dataclass
class FeasibilityMetrics:
    label:       str
    rated_mw:    float
    scale:       float = 1.0

    P_dc:          np.ndarray = field(default_factory=lambda: np.zeros(N_INTERVALS))
    Q_dc:          np.ndarray = field(default_factory=lambda: np.zeros(N_INTERVALS))
    P_flex:        np.ndarray = field(default_factory=lambda: np.zeros(N_INTERVALS))
    dQ_droop:      np.ndarray = field(default_factory=lambda: np.zeros(N_INTERVALS))
    area_import:   np.ndarray = field(default_factory=lambda: np.zeros(N_INTERVALS))
    V_pcc:         np.ndarray = field(default_factory=lambda: np.zeros(N_INTERVALS))
    V_min_sys:     np.ndarray = field(default_factory=lambda: np.zeros(N_INTERVALS))
    lmp_bus16:     np.ndarray = field(default_factory=lambda: np.zeros(N_INTERVALS))
    congestion:    np.ndarray = field(default_factory=lambda: np.zeros(N_INTERVALS))
    interface_pct: np.ndarray = field(default_factory=lambda: np.zeros(N_INTERVALS))

    n_thermal_viol:   int   = 0
    n_voltage_viol:   int   = 0
    n_cong_intervals: int   = 0
    peak_import_mw:   float = 0.0
    peak_import_hour: float = 0.0
    min_V_pcc:        float = 1.0
    min_V_hour:       float = 0.0
    max_congestion:   float = 0.0
    max_cong_hour:    float = 0.0
    pf_mean:          float = 0.0
    feasible:         bool  = True
    binding_limit:    str   = "none"


# =============================================================================
#  ADAPTER SETUP
# =============================================================================

def _make_adapter(label: str, rated_mw: float) -> Tuple[OPFAdapter, float, str]:
    """
    Register datacenter, warm up physics, return (adapter, scale, name).
    scale = rated_mw / (S_BASE_DC * LF_NOMINAL)
    """
    name = f"FEASIB_{label.replace(' ','_').replace(',','').replace('—','')}"
    cfg  = dict(seed=42, n_cooling_units=3, dt_micro=0.02, dt=0.02,
                price_threshold=DR_THRESHOLD, price_max=300.0,
                max_curtail_pu=0.25)
    register(name, cfg)
    adapter: OPFAdapter = get_datacenter(name, "opf", interval_min=DT_MIN)

    # warm-up: drive physics past cold-start transient
    phys = adapter._p
    for w in range(WARMUP_STEPS):
        phys.step(CanonicalInput(V_pu=1.0, freq_hz=60.0,
                                 t_sim=float(w)*0.02, dt=0.02))

    scale = rated_mw / (S_BASE_DC * LF_NOMINAL)
    return adapter, scale, name


# =============================================================================
#  FEASIBILITY EVALUATION
# =============================================================================

def run_feasibility(label: str, rated_mw: float,
                    lmp: np.ndarray) -> FeasibilityMetrics:
    """
    Per-interval workflow:
      1. OPFAdapter.get_bid(t)  →  raw P, Q, P_flex from physics
      2. Scale to facility rating
      3. Compute area import, Thevenin voltage (+ volt-VAR correction)
      4. Congestion premium
    """
    adapter, scale, name = _make_adapter(label, rated_mw)
    m = FeasibilityMetrics(label=label, rated_mw=rated_mw, scale=scale)
    pf_list = []

    for k in range(N_INTERVALS):
        t = float(k) * DT_S

        # 1. Physics bid
        bid   = adapter.get_bid(t)
        P_dc  = bid["p_mw"]   * scale
        Q_dc  = bid["q_mvar"] * scale
        P_flex= bid["flex_mw"]* scale
        pf_list.append(P_dc / max(float(np.hypot(P_dc, Q_dc)), 1e-6))

        # 2. Area import
        area_import = AREA_BASE_MW + P_dc

        # 3. Thevenin voltage + volt-VAR droop correction (K_vv = 0.10)
        dP_pu    = P_dc / S_BASE_MVA
        dQ_pu    = Q_dc / S_BASE_MVA
        V_raw    = V0 - (R_TH * dP_pu + X_TH * dQ_pu) / V0
        dV       = V_raw - 1.0
        Q_droop  = -0.10 * dV * (P_dc / S_BASE_MVA)   # VSC volt-VAR [pu MVAR]
        V_pcc    = float(np.clip(V_raw + X_TH * Q_droop, 0.50, 1.10))
        V_min_sys= float(np.clip(V_pcc - 0.05 * max(0.0, 1.0 - V_pcc/V0),
                                  0.50, 1.10))

        # 4. Congestion
        intf_pct = area_import / INTERFACE_LIMIT_MW * 100.0
        if area_import > INTERFACE_LIMIT_MW:
            excess    = (area_import - INTERFACE_LIMIT_MW) / INTERFACE_LIMIT_MW
            congestion = float(lmp[k]) * excess * CONGESTION_PASSTHROUGH
        else:
            congestion = 0.0

        m.P_dc[k]          = P_dc
        m.Q_dc[k]          = Q_dc
        m.P_flex[k]        = P_flex
        m.dQ_droop[k]      = Q_droop * S_BASE_MVA
        m.area_import[k]   = area_import
        m.V_pcc[k]         = V_pcc
        m.V_min_sys[k]     = V_min_sys
        m.lmp_bus16[k]     = float(lmp[k]) + congestion
        m.congestion[k]    = congestion
        m.interface_pct[k] = intf_pct

    m.n_thermal_viol   = int(np.sum(m.area_import > INTERFACE_LIMIT_MW))
    m.n_voltage_viol   = int(np.sum(m.V_pcc < V_MIN_PU))
    m.n_cong_intervals = int(np.sum(m.congestion > 0.5))
    m.peak_import_mw   = float(m.area_import.max())
    m.peak_import_hour = float(HOURS[np.argmax(m.area_import)])
    m.min_V_pcc        = float(m.V_pcc.min())
    m.min_V_hour       = float(HOURS[np.argmin(m.V_pcc)])
    m.max_congestion   = float(m.congestion.max())
    m.max_cong_hour    = float(HOURS[np.argmax(m.congestion)])
    m.pf_mean          = float(np.mean(pf_list))

    if m.n_thermal_viol > 0:
        m.feasible, m.binding_limit = False, "thermal"
    elif m.n_voltage_viol > 0:
        m.feasible, m.binding_limit = False, "voltage"
    elif m.n_cong_intervals > 2:
        m.feasible, m.binding_limit = True,  "economic"
    else:
        m.feasible, m.binding_limit = True,  "none"

    deregister(name)
    return m


# =============================================================================
#  HOSTING CAPACITY CURVES
# =============================================================================

def compute_hosting_capacity(lmp: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    rlb  = 0.05 * INTERFACE_LIMIT_MW
    ht   = np.full(N_INTERVALS, INTERFACE_LIMIT_MW - AREA_BASE_MW - rlb)
    he   = np.zeros(N_INTERVALS)
    for k in range(N_INTERVALS):
        max_excess = 15.0 / (max(lmp[k], 20.0) * CONGESTION_PASSTHROUGH)
        he[k]      = INTERFACE_LIMIT_MW * (1 + max_excess) - AREA_BASE_MW
    return ht, np.clip(he, 0.0, ht * 1.5)


# =============================================================================
#  PLOT
# =============================================================================

def plot_results(metrics: Dict[str, FeasibilityMetrics],
                 lmp: np.ndarray,
                 ht: np.ndarray, he: np.ndarray):

    BG, BG2, GRC, MUT = "#ffffff", "#f8f9fa", "#000000", "#000000"
    DC_COL = {"DC1 — 100 MW":"#2e7d32","DC2 — 230 MW":"#1565c0",
               "DC3 — 430 MW":"#e65100","DC4 — 500 MW":"#b71c1c"}
    DC_LS  = {"DC1 — 100 MW":"-","DC2 — 230 MW":"--",
               "DC3 — 430 MW":"-.","DC4 — 500 MW":":"}
    LW = 2

    fig = plt.figure(figsize=(14, 18))
    fig.patch.set_facecolor(BG)
    gs  = gridspec.GridSpec(4, 1, figure=fig,
                            # height_ratios=[2.5, 1.5, 1.5, 1.5],
                            # hspace=0.52, top=0.94, bottom=0.05,
                            # left=0.10, right=0.95
                            )
    axes = [fig.add_subplot(gs[i]) for i in range(4)]

    def style(ax, ylabel, title, ylim=None):
        ax.set_facecolor(BG2)
        ax.tick_params(colors=MUT, labelsize=8.5)
        for sp in ax.spines.values():
            sp.set_edgecolor(GRC); sp.set_linewidth(0.8)
        ax.grid(axis="y", color=GRC, lw=0.5, ls=":")
        ax.grid(axis="x", color=GRC, lw=0.4, ls=":")
        ax.set_ylabel(ylabel, fontsize=16)
        ax.set_title(title, fontsize=16, loc="left",
                     color=MUT, pad=5, fontweight="bold")
        ax.set_xlim(0, 23)
        ax.set_xticks(range(0, 24, 4))
        ax.set_xticklabels([f"{h:02d}:00" for h in range(0, 24, 4)], fontsize=16)
        if ylim: ax.set_ylim(*ylim)

    def peaks(ax):
        for ts, te in [(7, 10), (17, 21)]:
            ax.axvspan(ts, te, color="#fff3cd", alpha=0.55, zorder=0)

    # Panel 1: thermal
    ax = axes[0]; peaks(ax)
    ax.fill_between(HOURS, 0, ht, color="#e8f5e9", alpha=0.7, zorder=1,
                    label="Feasible zone")
    ax.fill_between(HOURS, ht, ht*1.25, color="#fff3e0", alpha=0.5, zorder=1,
                    label="Warning zone")
    ax.fill_between(HOURS, ht*1.25, 1100, color="#ffebee", alpha=0.5, zorder=1,
                    label="Infeasible zone")
    ax.axhline(ht[0], color="#1b5e20", lw=1.5, ls="--", zorder=3,
               label=f"Thermal limit {ht[0]:.0f} MW")
    ax.axhline(ht[0]*0.95, color="#e65100", lw=1.0, ls=":", zorder=3,
               label="95% N-1 buffer")
    for lbl, m in metrics.items():
        ax.plot(HOURS, m.P_dc, color=DC_COL[lbl], lw=LW, ls=DC_LS[lbl],
                label=f"{lbl}  (PF={m.pf_mean:.3f})", zorder=4)
        pk = int(np.argmax(m.P_dc))
        ax.plot(HOURS[pk], m.P_dc[pk],
                marker="^" if m.feasible else "v",
                color=DC_COL[lbl], ms=8, zorder=5, ls="none")
    style(ax, "DC active load (MW)",
          "Panel 1 — Thermal Hosting Capacity  (physics-derived P, Q)",
          ylim=(-20, 700))
    ax.legend(fontsize=14, loc="upper left", ncol=3,
              facecolor="white", edgecolor=GRC)

    # Panel 2: voltage
    ax = axes[1]; peaks(ax)
    ax.axhspan(V_MIN_PU, V_MAX_PU, color="#e8f5e9", alpha=0.45, zorder=0,
               label="ANSI C84.1 Range A")
    ax.axhline(V_MIN_PU, color="#c62828", lw=1.0, ls="--", alpha=0.8,
               label=f"{V_MIN_PU} pu")
    ax.axhline(V_MAX_PU, color="#c62828", lw=1.0, ls="--", alpha=0.8)
    ax.axhline(1.00, color="#9e9e9e", lw=0.5, ls=":", alpha=0.6)
    for lbl, m in metrics.items():
        ax.plot(HOURS, m.V_pcc, color=DC_COL[lbl], lw=LW-0.3,
                ls=DC_LS[lbl], label=lbl, zorder=3)
    style(ax, "|V| at bus 16 (pu)",
          "Panel 2 — PCC Voltage (Thevenin + VSC volt-VAR droop K_vv=0.10)")
    ax.legend(fontsize=12, loc="lower right", ncol=2,
              facecolor="white", edgecolor=GRC)

    # Panel 3: congestion
    ax = axes[2]; peaks(ax)
    ax3t = ax.twinx()
    # ax3t.fill_between(HOURS, lmp, alpha=0.08, color="#1565c0")
    ax3t.plot(HOURS, lmp, color="#1565c0", lw=0.8, alpha=0.5, ls=":")
    ax3t.axhline(DR_THRESHOLD, color="#1565c0", lw=0.7, ls="--", alpha=0.6)
    ax3t.text(23, DR_THRESHOLD+2, f"DR {DR_THRESHOLD:.0f} $/MWh",
               fontsize=16, color="#1565c0", ha="right")
    ax3t.set_ylabel("Zonal LMP ($/MWh)", fontsize=16, color="#1565c0")
    ax3t.tick_params(colors="#1565c0", labelsize=8); ax3t.set_ylim(0, 160)
    any_c = False
    for lbl, m in metrics.items():
        if m.congestion.max() > 0.5:
            # ax.fill_between(HOURS, m.congestion, alpha=0.20, color=DC_COL[lbl])
            ax.plot(HOURS, m.congestion, color=DC_COL[lbl],
                    lw=LW-0.3, ls=DC_LS[lbl], label=lbl, zorder=3)
            any_c = True
    if not any_c:
        ax.text(12, 5, "No congestion — all sizes below thermal limit",
                ha="center", va="center", fontsize=16,
                color="#4caf50", style="italic")
    style(ax, "Congestion premium ($/MWh)",
          "Panel 3 — Congestion Premium at Bus 16")
    if any_c:
        ax.legend(fontsize=16, loc="upper left", ncol=2,
                  facecolor="white", edgecolor=GRC)
    ax.set_ylim(bottom=0)

    # Panel 4: summary bars
    ax = axes[3]
    ax.set_facecolor(BG2)
    for sp in ax.spines.values():
        sp.set_edgecolor(GRC); sp.set_linewidth(0.8)
    ax.grid(axis="y", color=GRC, lw=0.6, ls=":")
    labels = list(metrics.keys())
    x = np.arange(len(labels)); w = 0.28
    avail   = ht[0]
    peak_dc = [m.P_dc.max() for m in metrics.values()]
    pct_int = [m.peak_import_mw/INTERFACE_LIMIT_MW*100 for m in metrics.values()]
    n_cong  = [m.n_cong_intervals for m in metrics.values()]
    bars1 = ax.bar(x-w, peak_dc, w,
                   color=[DC_COL[l] for l in labels],
                   edgecolor="white", alpha=0.85, label="Peak DC load (MW)")
    ax4b = ax.twinx()
    ax4b.bar(x, n_cong, w, color="#9c27b0", alpha=0.55,
             edgecolor="white", label="Congested hours")
    ax4b.set_ylabel("Congested intervals (h)", fontsize=16, color="#9c27b0")
    ax4b.tick_params(colors="#9c27b0", labelsize=8)
    ax4b.set_ylim(0, max(n_cong)*3+1)
    ax.bar(x+w, pct_int, w, color="#37474f", alpha=0.60,
           edgecolor="white", label="Peak interface loading (%)")
    ax.axhline(avail, color="#1b5e20", lw=1.2, ls="--", alpha=0.8,
               label=f"Available capacity ({avail:.0f} MW)")
    for bar, val, m in zip(bars1, peak_dc, metrics.values()):
        sym = "✓" if m.feasible else "✗"
        col = "#1b5e20" if m.feasible else "#c62828"
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+5,
                f"{val:.0f} MW\n{sym}", ha="center", va="bottom",
                fontsize=16, color=col, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([l.replace(" — ","\n") for l in labels], fontsize=16)
    ax.set_ylabel("MW  /  % of interface limit", fontsize=16)
    ax.set_title("Panel 4 — Feasibility Summary by Datacenter Size",
                 fontsize=16, loc="left", color=MUT, pad=5, fontweight="bold")
    ax.set_ylim(0, max(max(peak_dc), max(pct_int))*1.35)
    ax.tick_params(colors=MUT, labelsize=9)
    ax.legend(fontsize=16, loc="upper left", facecolor="white", edgecolor=GRC)

    fig.suptitle(
        "Transmission Interconnection Feasibility Study  (v2 — adapter-integrated)\n"
        "Hosting Capacity Analysis — IEEE 39-Bus  ·  Bus 16 POI  "
        "·  P/Q from OPFAdapter → DatacenterPhysics → DatacenterSubsystem",
        fontsize=16, color=MUT, y=0.98, fontweight="bold")
    fig.text(0.10, 0.022,
             f"scale = target_mw / (S_base={S_BASE_DC} MVA × LF={LF_NOMINAL})  ·  "
             f"Volt-VAR K_vv=0.10  ·  Thevenin R={R_TH} X={X_TH} pu",
             fontsize=16, color="#6c757d", ha="left")

    plt.savefig(OUT_PLOT, dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  Figure  →  {OUT_PLOT}")


# =============================================================================
#  REPORT
# =============================================================================

def write_report(metrics: Dict[str, FeasibilityMetrics], ht: np.ndarray):
    sep   = "=" * 74
    avail = ht[0]
    VERDICT = {
        "none":     "FEASIBLE    — No network upgrade required",
        "economic": "FEASIBLE*   — Congestion costs apply; no thermal upgrade",
        "voltage":  "INFEASIBLE  — Voltage violation; reactive compensation required",
        "thermal":  "INFEASIBLE  — Thermal violation; transmission upgrade required",
    }
    lines = [
        sep,
        "  FEASIBILITY STUDY  (v2 — OPFAdapter / DatacenterPhysics)",
        "  IEEE 39-Bus New England  ·  Bus 16 POI",
        sep, "",
        "  P/Q SOURCE: OPFAdapter.get_bid(t) → DatacenterPhysics.step()",
        f"              → DatacenterSubsystem (GPU trace + IM + VSC)",
        f"  Scale:      target_mw / ({S_BASE_DC} MVA × {LF_NOMINAL}) = rated_mw/{S_BASE_DC*LF_NOMINAL:.4f}",
        f"  Volt-VAR:   K_vv = 0.10 pu/pu applied in Thevenin correction",
        "",
        f"  Interface limit : {INTERFACE_LIMIT_MW:.0f} MW",
        f"  Area base import: {AREA_BASE_MW:.0f} MW (buses 15+16+18)",
        f"  Available DC cap: {avail:.0f} MW (with 5% N-1 buffer)",
        "",
    ]
    for lbl, m in metrics.items():
        lines += [
            f"  {'─'*70}",
            f"  {lbl}  (scale={m.scale:.1f})",
            f"    {VERDICT[m.binding_limit]}",
            f"    Physics PF (mean)  : {m.pf_mean:.4f} lagging",
            f"    Peak DC load       : {m.P_dc.max():.2f} MW at h={m.peak_import_hour:.0f}",
            f"    Peak area import   : {m.peak_import_mw:.1f} MW "
            f"({m.peak_import_mw/INTERFACE_LIMIT_MW*100:.1f}%)",
            f"    Headroom           : {avail - m.rated_mw:.1f} MW",
            f"    Congested hours    : {m.n_cong_intervals} "
            f"(max {m.max_congestion:.1f} $/MWh)",
            f"    V_pcc range        : [{m.V_pcc.min():.4f}, {m.V_pcc.max():.4f}] pu",
            f"    Voltage violations : {m.n_voltage_viol}",
            "",
        ]
    lines += [sep, f"  Figure: {OUT_PLOT}", f"  Report: {OUT_RPT}", sep]
    text = "\n".join(lines)
    OUT_RPT.write_text(text)
    print(text)
    print(f"\n  Report  →  {OUT_RPT}")


# =============================================================================
#  MAIN
# =============================================================================

# def main():
print("\n" + "="*74)
print("  OASIS / GOALS — Feasibility Study  (v2 — adapter-integrated)")
print("  P/Q: OPFAdapter → DatacenterPhysics → DatacenterSubsystem")
print("="*74)

print("\n[1/3] Building LMP forecast …")
lmp = build_lmp_profile()

print("[2/3] Computing hosting capacity curves …")
ht, he = compute_hosting_capacity(lmp)
print(f"      Thermal: {ht[0]:.0f} MW")

print("[3/3] Running physics-integrated feasibility …\n")
metrics: Dict[str, FeasibilityMetrics] = {}
t_all = time.time()
for lbl, mw in DC_CONFIGS.items():
    print(f"  {lbl} …", end=" ", flush=True)
    t0 = time.time()
    m  = run_feasibility(lbl, mw, lmp)
    metrics[lbl] = m
    print(f"{'✓' if m.feasible else '✗'} {m.binding_limit:<10}  "
          f"peak={m.P_dc.max():.0f} MW  import={m.peak_import_mw:.0f} MW  "
          f"PF={m.pf_mean:.3f}  ({time.time()-t0:.1f}s)")

print(f"\n  Total: {time.time()-t_all:.1f}s")
print("\nPlotting …")
plot_results(metrics, lmp, ht, he)
print("Writing report …")
write_report(metrics, ht)
print("\nDone.")


# if __name__ == "__main__":
#     main()
