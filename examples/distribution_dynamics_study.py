"""
===============================
Example 1 — Distribution System Dynamics Study for IEEE 13-bus system
==================================================

The simulation runs for SIM_DURATION seconds and covers five distinct
operating phases, each exercising a different dynamic phenomenon:

  Phase 1  t =   0 – 30 s   Cold start & motor acceleration
  Phase 2  t =  30 – 90 s   Steady-state feeder operation + DG variability
  Phase 3  t =  90 – 110 s  Voltage sag event (capacitor bank switching)
  Phase 4  t = 110 – 140 s  Fault at bus_634 (datacenter PCC), ride-through
  Phase 5  t = 140 – 300 s  Post-fault recovery, solar ramp, frequency events
"""

import sys
import os
import time
import textwrap
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from pathlib import Path
plt.rcParams['lines.linewidth'] = 2.0

# ── Project modules ───────────────────────────────────────────────────────────
from datacenter_registry import register, get_datacenter
from adapters import DistributionAdapter

# =============================================================================
#  STUDY CONFIGURATION
# =============================================================================
SIM_DURATION   = 300.0   # s  — total simulation time
T_MACRO        = 0.1     # s  — network power-flow interval
DT_MICRO       = 0.01    # s  — datacenter internal step
N_MACRO        = int(SIM_DURATION / T_MACRO)
TIME           = np.linspace(0.0, SIM_DURATION - T_MACRO, N_MACRO)

# Gauss-Seidel coupling parameters
MAX_GS_ITER    = 10       # max iterations per macro step
GS_TOL         = 5e-4    # convergence tolerance on |ΔV_pcc| [pu]

# Datacenter registration
DC_NAME        = "DC_FEEDER_634"
DC_CONFIG      = {
    "seed":          42,
    "n_cooling_units": 3,
    "dt_micro":      DT_MICRO,
    "dt":            T_MACRO,
    "price_threshold": 50.0,
    "price_max":     200.0,
    "max_curtail_pu": 0.30,
}

# Network choice
TEST_SYSTEM = 'opendss_13bus_network'
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))
# from TEST_SYSTEM_PATH import OpenDSSNetworkSimulator, NetworkResults
from testsystems.opendss_13bus_network import OpenDSSNetworkSimulator, NetworkResults


# ── Operating phases ─────────────────────────────────────────────────────────
@dataclass
class Phase:
    name:     str
    t_start:  float
    t_end:    float
    color:    str
    description: str

PHASES = [
    Phase("Cold Start & Motor Acceleration", 0.0,   30.0,  "#3fb950",
          "HVAC motors accelerate from standstill; VSC control loop settles; "
          "feeder establishes steady-state voltage profile."),
    Phase("Steady-State Operation",          30.0,  90.0,  "#ffffff", #"#58a6ff",
          "Normal feeder operation. DG1 wind output varies sinusoidally. "
          "Solar DGs ramp up toward midday peak. Voltage regulation observed."),
    Phase("Cap Bank Switch",   90.0,  110.0, "#e3b341",
          "Capacitor bank at bus_671 switched out at t=90s, reinserted at t=105s. "
          "Volt-VAR droop on datacenter VSC responds to voltage excursion."),
    Phase("Fault — Bus 634 PCC",             120.0, 140.0, "#f78166",
          "Line-to-ground fault at datacenter PCC (bus_634) applied t=120s, "
          "cleared t=122s. LVRT ride-through active. Motor stalling risk assessed."),
    # Phase("Post-Fault Recovery",             140.0, 300.0, "#d2a8ff",
    #       "Voltage and frequency recover. Solar reaches peak dispatch at t=150s. "
    #       "Frequency droop event injected at t=200s. Final steady-state observed."),
    # Phase("Steady-State Operation",   140.0, 200.0, "#ffffff",
    #       "Voltage and frequency recover. Solar reaches peak dispatch at t=150s. "
    #       "Frequency droop event injected at t=200s. Final steady-state observed."),
    Phase("Frequency-Event, Generator Trip",             200.0, 210.0, "#d2a8ff",
          "Voltage and frequency recover. Solar reaches peak dispatch at t=150s. "
          "Frequency droop event injected at t=200s. Final steady-state observed."),
]

# ── Event schedule ────────────────────────────────────────────────────────────
@dataclass
class Event:
    t: float
    label: str
    kind: str    # 'fault_on' | 'fault_off' | 'cap_out' | 'cap_in' | 'freq_step'
    bus: Optional[str] = None
    param: float = 0.0

EVENTS = [
    Event(90.0,  "Cap bank switched OUT",  "cap_out",  "bus_671",  0.0),
    Event(105.0, "Cap bank reinserted",    "cap_in",   "bus_671",  0.0),
    Event(120.0, "FAULT applied",          "fault_on", "bus_634",  0.08),
    Event(122.0, "Fault CLEARED",          "fault_off","bus_634",  0.0),
    Event(200.0, "Freq step −0.5 Hz",      "freq_step","",        -0.5),
    Event(210.0, "Freq step restored",     "freq_step","",         0.0),
]

# ── Output paths ──────────────────────────────────────────────────────────────
OUT_DIR   = '../Outputs'
OUT_PLOT  = os.path.join(OUT_DIR, 'distribution_dynamics_results.png') #OUT_DIR / "distribution_dynamics_results.png"
OUT_CSV   = os.path.join(OUT_DIR, 'distribution_dynamics_data.csv')
OUT_RPT   = os.path.join(OUT_DIR, 'distribution_dynamics_report.txt')


#  TIME-SERIES RECORD
@dataclass
class Record:
    """All recorded signals, pre-allocated as NumPy arrays."""
    # Network state
    V_pcc:        np.ndarray = field(default_factory=lambda: np.zeros(N_MACRO))
    V_632:        np.ndarray = field(default_factory=lambda: np.zeros(N_MACRO))
    V_680:        np.ndarray = field(default_factory=lambda: np.zeros(N_MACRO))
    freq:         np.ndarray = field(default_factory=lambda: np.zeros(N_MACRO))
    P_load_total: np.ndarray = field(default_factory=lambda: np.zeros(N_MACRO))
    P_losses:     np.ndarray = field(default_factory=lambda: np.zeros(N_MACRO))
    P_gen_dg:     np.ndarray = field(default_factory=lambda: np.zeros(N_MACRO))

    # Datacenter boundary
    P_dc:         np.ndarray = field(default_factory=lambda: np.zeros(N_MACRO))
    Q_dc:         np.ndarray = field(default_factory=lambda: np.zeros(N_MACRO))

    # Datacenter internals
    P_server:     np.ndarray = field(default_factory=lambda: np.zeros(N_MACRO))
    P_cool:       np.ndarray = field(default_factory=lambda: np.zeros(N_MACRO))
    Q_cool:       np.ndarray = field(default_factory=lambda: np.zeros(N_MACRO))
    omega_r:      np.ndarray = field(default_factory=lambda: np.zeros(N_MACRO))
    slip:         np.ndarray = field(default_factory=lambda: np.zeros(N_MACRO))

    # Grid-support signals
    dP_fw:        np.ndarray = field(default_factory=lambda: np.zeros(N_MACRO))
    dQ_vv:        np.ndarray = field(default_factory=lambda: np.zeros(N_MACRO))

    # Protection / status flags
    riding_through: np.ndarray = field(default_factory=lambda: np.zeros(N_MACRO, dtype=bool))
    fault_active:   np.ndarray = field(default_factory=lambda: np.zeros(N_MACRO, dtype=bool))
    cap_out:        np.ndarray = field(default_factory=lambda: np.zeros(N_MACRO, dtype=bool))

    # Co-simulation metadata
    gs_iters:     np.ndarray = field(default_factory=lambda: np.zeros(N_MACRO, dtype=int))
    gs_converged: np.ndarray = field(default_factory=lambda: np.zeros(N_MACRO, dtype=bool))


#  DISTRIBUTION DYNAMICS STUDY CLASS
class DistributionDynamicsStudy:
    """
    Main study class.  Wires together the network simulator, the datacenter
    adapter, the Gauss-Seidel interface, and all analysis/output routines.

    The __init__ sets up components; run() executes the simulation;
    analyse(), save_csv(), plot(), report() produce outputs.
    """

    def __init__(self):
        self._print_banner()

        # ── Register datacenter via registry ──────────────────────────────────
        print("\n[1/4] Registering datacenter model …")
        register(DC_NAME, DC_CONFIG)
        self.dc_adapter: DistributionAdapter = get_datacenter(
            DC_NAME, "distribution",
            feeder_base_kv = 4.16,
            dc_base_kv     = 0.48,
        )

        # ── Instantiate network simulator ─────────────────────────────────────
        print("[2/4] Initialising IEEE 13-bus network …")
        self.net = OpenDSSNetworkSimulator(
            base_kv  = 4.16,
            base_mva = 5.0,
            freq_hz  = 60.0,
        )

        # ── Boundary signals (Gauss-Seidel state) ─────────────────────────────
        self.V_pcc_pu = 1.0
        self.freq_hz  = 60.0
        self.P_dc_mw  = 0.0
        self.Q_dc_mvar = 0.0

        # ── Pre-allocate record ───────────────────────────────────────────────
        self.rec = Record()

        # ── Event & phase state ───────────────────────────────────────────────
        self._fault_active   = False
        self._cap_out        = False
        self._cap_Y_perturb  = 0.0    # pu susceptance change from cap switching
        self._freq_offset    = 0.0    # Hz offset from synthetic freq event

        # Phase tracking for terminal summary
        self._phase_stats: List[dict] = []

        print("[3/4] Pre-computing event schedule …")
        self._event_idx = {e.t: e for e in EVENTS}

        print("[4/4] Ready.\n")
        print(self.net.summary())

    #  MAIN SIMULATION LOOP
    def run(self):
        print(f"\n{'='*74}")
        print(f"  DISTRIBUTION DYNAMICS STUDY")
        print(f"  Duration: {SIM_DURATION:.0f}s  |  "
              f"T_macro: {T_MACRO}s  |  T_micro: {DT_MICRO}s  |  "
              f"Buses: {self.net.n_bus}")
        print(f"{'='*74}")

        t_wall = time.time()
        current_phase_idx = 0

        for k, t in enumerate(TIME):

            # ── Phase transition announcement ─────────────────────────────────
            if current_phase_idx < len(PHASES):
                ph = PHASES[current_phase_idx]
                if t >= ph.t_start and (k == 0 or
                        TIME[k-1] < ph.t_start):
                    self._announce_phase(ph, t)

                if t >= ph.t_end:
                    self._record_phase_end(ph, k)
                    current_phase_idx += 1

            # ── Process scheduled events ──────────────────────────────────────
            self._process_events(t)

            # ── Gauss-Seidel predictor-corrector interface ────────────────────
            dV_gs = 1.0   # track convergence
            gs_it = 0

            for gs_it in range(MAX_GS_ITER):
                V_prev = self.V_pcc_pu

                # Step 1: advance datacenter one macro step
                P_dc, Q_dc = self.dc_adapter.step(
                    V_pcc_pu = self.V_pcc_pu,
                    freq_hz  = self.freq_hz + self._freq_offset,
                    t        = t,
                    dt       = T_MACRO,
                )
                dc_out = self.dc_adapter.last_output

                # Step 2: solve network power flow
                net_res: NetworkResults = self.net.solve(
                    t         = t,
                    P_dc_mw   = P_dc,
                    Q_dc_mvar = Q_dc,
                    dt        = T_MACRO,
                )

                # Step 3: update boundary
                dV_gs = abs(net_res.V_pcc_pu - self.V_pcc_pu)
                self.V_pcc_pu  = net_res.V_pcc_pu
                self.freq_hz   = net_res.freq_hz
                self.P_dc_mw   = P_dc
                self.Q_dc_mvar = Q_dc

                if dV_gs < GS_TOL:
                    break

            gs_converged = (dV_gs < GS_TOL)

            # ── Record ────────────────────────────────────────────────────────
            r = self.rec
            r.V_pcc[k]        = net_res.V_pcc_pu
            r.V_632[k]        = net_res.V_bus632_pu
            r.V_680[k]        = net_res.V_bus680_pu
            r.freq[k]         = net_res.freq_hz + self._freq_offset
            r.P_load_total[k] = net_res.P_total_load_mw
            r.P_losses[k]     = net_res.P_losses_mw
            r.P_gen_dg[k]     = net_res.P_gen_total_mw
            r.P_dc[k]         = P_dc
            r.Q_dc[k]         = Q_dc
            r.P_server[k]     = dc_out.P_server_kw
            r.P_cool[k]       = dc_out.P_cool_kw
            r.Q_cool[k]       = dc_out.Q_cool_kvar
            r.omega_r[k]      = dc_out.omega_r_pu
            r.slip[k]         = dc_out.slip
            r.dP_fw[k]        = dc_out.dP_droop_mw
            r.dQ_vv[k]        = dc_out.dQ_droop_mvar
            r.riding_through[k] = dc_out.riding_through
            r.fault_active[k]   = self._fault_active
            r.cap_out[k]        = self._cap_out
            r.gs_iters[k]       = gs_it + 1
            r.gs_converged[k]   = gs_converged

            # ── Console progress ──────────────────────────────────────────────
            if k % 200 == 0 or self._fault_active:
                flags = ""
                if self._fault_active:   flags += "  ⚡ FAULT"
                if self._cap_out:        flags += "  🔌 CAP-OUT"
                if dc_out.riding_through: flags += "  🔄 LVRT/FRT"
                if self._freq_offset != 0: flags += f"  Δf={self._freq_offset:+.1f}Hz"

                print(
                    f"  t={t:6.1f}s | "
                    f"V_pcc={net_res.V_pcc_pu:.4f} pu | "
                    f"f={r.freq[k]:.3f} Hz | "
                    f"P_dc={P_dc:.3f} MW | "
                    f"Q_dc={Q_dc:.3f} MVAR | "
                    f"ω_r={dc_out.omega_r_pu:.4f} | "
                    f"GS={gs_it+1}"
                    + flags
                )

        elapsed = time.time() - t_wall
        avg_gs  = self.rec.gs_iters.mean()
        print(f"\n{'='*74}")
        print(f"  Simulation complete — {elapsed:.1f}s wall time  |  "
              f"Avg GS iterations: {avg_gs:.2f}")
        print(f"{'='*74}\n")

    #  EVENT HANDLING
    def _process_events(self, t: float):
        """Apply scheduled network events at the correct simulation time."""
        # Round t to one decimal to match event schedule keys
        t_key = round(t, 1)
        if t_key not in self._event_idx:
            return

        ev = self._event_idx[t_key]
        print(f"\n  ── EVENT t={t:.1f}s: {ev.label} ──")

        if ev.kind == "fault_on":
            self.net.apply_fault(ev.bus, z_fault_pu=ev.param)
            self._fault_active = True

        elif ev.kind == "fault_off":
            self.net.clear_fault(ev.bus)
            self._fault_active = False

        elif ev.kind == "cap_out":
            # Simulate capacitor bank removal by modifying feeder shunt
            # Inject +0.15 pu reactive load at bus_671 (cap bank removed)
            self.net.buses['bus_671'].Q_mvar += 0.30   # MW — extra reactive demand
            self._cap_out = True

        elif ev.kind == "cap_in":
            # Reinsert capacitor: restore reactive load
            self.net.buses['bus_671'].Q_mvar -= 0.30
            self._cap_out = False

        elif ev.kind == "freq_step":
            # Synthetic frequency perturbation (external grid event)
            self._freq_offset = ev.param
            if ev.param != 0:
                self.net._freq_dev = ev.param

    #  POST-SIMULATION ANALYSIS
    def analyse(self):
        """Compute derived metrics for the summary report."""
        r  = self.rec
        t  = TIME

        print("─" * 74)
        print("  POST-SIMULATION ANALYSIS")
        print("─" * 74)

        # ── Voltage profile statistics ────────────────────────────────────────
        print("\n  VOLTAGE PROFILE")
        print(f"    Bus 634 (DC PCC)  : mean={r.V_pcc.mean():.4f}  "
              f"min={r.V_pcc.min():.4f}  max={r.V_pcc.max():.4f}  "
              f"std={r.V_pcc.std():.5f}")
        print(f"    Bus 632 (junction): mean={r.V_632.mean():.4f}  "
              f"min={r.V_632.min():.4f}  max={r.V_632.max():.4f}")
        print(f"    Bus 680 (DG wind) : mean={r.V_680.mean():.4f}  "
              f"min={r.V_680.min():.4f}  max={r.V_680.max():.4f}")

        # IEEE 1547 voltage compliance check (outside fault window)
        nonfault = ~r.fault_active
        V_nonfault = r.V_pcc[nonfault]
        pct_in_band = 100 * np.mean((V_nonfault >= 0.95) & (V_nonfault <= 1.05))
        print(f"\n    ANSI C84.1 Range A (0.95–1.05 pu) compliance "
              f"(non-fault periods): {pct_in_band:.1f}%")

        # ── Frequency ─────────────────────────────────────────────────────────
        print("\n  FREQUENCY")
        print(f"    Mean: {r.freq.mean():.4f} Hz   "
              f"Min: {r.freq.min():.4f} Hz   "
              f"Max: {r.freq.max():.4f} Hz")
        rocof = np.gradient(r.freq, T_MACRO)
        print(f"    Peak ROCOF: {np.abs(rocof).max():.3f} Hz/s  "
              f"(at t={TIME[np.argmax(np.abs(rocof))]:.1f}s)")

        # ── Datacenter power ──────────────────────────────────────────────────
        print("\n  DATACENTER POWER")
        print(f"    P mean={r.P_dc.mean():.4f} MW  max={r.P_dc.max():.4f} MW  "
              f"min={r.P_dc.min():.4f} MW")
        print(f"    Q mean={r.Q_dc.mean():.4f} MVAR")
        total_energy = np.trapezoid(r.P_dc, TIME) / 3600.0
        print(f"    Energy consumed: {total_energy:.4f} MWh "
              f"over {SIM_DURATION:.0f}s")

        # ── Grid-support assessment ────────────────────────────────────────────
        print("\n  GRID SUPPORT")
        dP_peak = np.abs(r.dP_fw).max()
        dQ_peak = np.abs(r.dQ_vv).max()
        rt_steps = r.riding_through.sum()
        print(f"    Freq-Watt peak response : {dP_peak:.4f} MW")
        print(f"    Volt-VAR peak response  : {dQ_peak:.4f} MVAR")
        print(f"    LVRT/FRT ride-through   : {rt_steps} steps "
              f"({100*rt_steps/N_MACRO:.1f}% of simulation)")

        # ── Motor dynamics ────────────────────────────────────────────────────
        print("\n  HVAC MOTOR DYNAMICS")
        print(f"    ω_r mean={r.omega_r.mean():.4f} pu  "
              f"min={r.omega_r.min():.4f} pu  "
              f"(min during fault: "
              f"{r.omega_r[r.fault_active].min() if r.fault_active.any() else float('nan'):.4f} pu)")
        print(f"    Slip mean={r.slip.mean():.4f}  "
              f"max={r.slip.max():.4f}")

        # ── Feeder losses ─────────────────────────────────────────────────────
        print("\n  FEEDER LOSSES")
        print(f"    Mean: {r.P_losses.mean():.4f} MW  "
              f"({100*r.P_losses.mean()/r.P_load_total.mean():.2f}% of load)")
        loss_energy = np.trapezoid(r.P_losses, TIME) / 3600.0
        print(f"    Total loss energy: {loss_energy:.4f} MWh")

        # ── Co-simulation quality ─────────────────────────────────────────────
        print("\n  CO-SIMULATION INTERFACE")
        print(f"    Average GS iterations  : {r.gs_iters.mean():.2f}")
        print(f"    Non-convergence steps  : {(~r.gs_converged).sum()}")
        print(f"    Max iterations in step : {r.gs_iters.max()}")

        print("─" * 74)

    #  PHASE BOOKKEEPING
    def _announce_phase(self, ph: Phase, t: float):
        print(f"\n  ┌{'─'*70}┐")
        print(f"  │  PHASE: {ph.name:<62}│")
        print(f"  │  t = {ph.t_start:.0f}–{ph.t_end:.0f}s"
              f"{'':>57}│")
        print(f"  └{'─'*70}┘")

    def _record_phase_end(self, ph: Phase, k_end: int):
        k_start = int(ph.t_start / T_MACRO)
        k_end   = min(k_end, N_MACRO)
        sl      = slice(k_start, k_end)
        r = self.rec
        self._phase_stats.append({
            'phase':       ph.name,
            't_start':     ph.t_start,
            't_end':       ph.t_end,
            'V_pcc_mean':  r.V_pcc[sl].mean()  if k_end > k_start else np.nan,
            'V_pcc_min':   r.V_pcc[sl].min()   if k_end > k_start else np.nan,
            'freq_mean':   r.freq[sl].mean()   if k_end > k_start else np.nan,
            'P_dc_mean':   r.P_dc[sl].mean()   if k_end > k_start else np.nan,
            'RT_steps':    int(r.riding_through[sl].sum()),
        })

    # ─────────────────────────────────────────────────────────────────────────
    #  SAVE CSV
    # ─────────────────────────────────────────────────────────────────────────
    def save_csv(self):
        r = self.rec
        df = pd.DataFrame({
            'time_s':           TIME,
            'V_pcc_pu':         r.V_pcc,
            'V_bus632_pu':      r.V_632,
            'V_bus680_pu':      r.V_680,
            'freq_hz':          r.freq,
            'P_dc_mw':          r.P_dc,
            'Q_dc_mvar':        r.Q_dc,
            'P_server_kw':      r.P_server,
            'P_cool_kw':        r.P_cool,
            'Q_cool_kvar':      r.Q_cool,
            'omega_r_pu':       r.omega_r,
            'slip':             r.slip,
            'dP_fw_mw':         r.dP_fw,
            'dQ_vv_mvar':       r.dQ_vv,
            'P_load_total_mw':  r.P_load_total,
            'P_losses_mw':      r.P_losses,
            'P_gen_dg_mw':      r.P_gen_dg,
            'riding_through':   r.riding_through.astype(int),
            'fault_active':     r.fault_active.astype(int),
            'cap_out':          r.cap_out.astype(int),
            'gs_iters':         r.gs_iters,
        })
        df.to_csv(OUT_CSV, index=False, float_format='%.6f')
        print(f"  CSV  →  {OUT_CSV}")

    # ─────────────────────────────────────────────────────────────────────────
    #  PLOT  (10-panel time-series)
    # ─────────────────────────────────────────────────────────────────────────
    # def plot(self):
    #     r = self.rec
    #     t = TIME
    #
    #     # ── Colour palette ────────────────────────────────────────────────────
    #     BG  = '#ffffff'; BG2 = '#f6f8fa'; MUT = '#24292f'; WHT = '#24292f'
    #     C   = ['#0969da','#1a7f37','#8250df','#bc4c00',
    #            '#cf222e','#0550ae','#2da44e','#9a6700',
    #            '#a40e26','#0969da']
    #
    #     fig = plt.figure(figsize=(16, 28))
    #     fig.patch.set_facecolor(BG)
    #     gs_layout = gridspec.GridSpec(
    #         10, 1, figure=fig,
    #         hspace=0.52, top=0.96, bottom=0.04,
    #         left=0.09, right=0.97
    #     )
    #     axes = [fig.add_subplot(gs_layout[i]) for i in range(10)]
    #
    #     for ax in axes:
    #         ax.set_facecolor(BG2)
    #         ax.tick_params(colors=MUT, labelsize=8)
    #         ax.yaxis.label.set_color(MUT)
    #         for sp in ax.spines.values():
    #             sp.set_edgecolor('#d0d7de'); sp.set_linewidth(0.5)
    #         ax.grid(axis='y', color='#d0d7de', lw=0.5, ls=':')
    #
    #     def shade_phases(ax):
    #         """Shade each operating phase with its assigned colour."""
    #         for ph in PHASES:
    #             ax.axvspan(ph.t_start, ph.t_end,
    #                        color=ph.color, alpha=0.06, zorder=0)
    #
    #     def shade_events(ax):
    #         """Mark key events with vertical lines."""
    #         ev_styles = {
    #             'fault_on':  ('#f78166', '--', 1.4),
    #             'fault_off': ('#3fb950', '--', 1.0),
    #             'cap_out':   ('#e3b341', ':',  1.0),
    #             'cap_in':    ('#3fb950', ':',  1.0),
    #             'freq_step': ('#d2a8ff', '-.',  1.0),
    #         }
    #         for ev in EVENTS:
    #             style = ev_styles.get(ev.kind, (MUT, '-', 0.8))
    #             ax.axvline(ev.t, color=style[0], ls=style[1],
    #                        lw=style[2], alpha=0.85, zorder=4)
    #
    #     def add_hvoltage_ref(ax, val=0.95, label='0.95 pu'):
    #         ax.axhline(val, color='#f78166', lw=0.7, ls='--', alpha=0.6, label=label)
    #         ax.axhline(2.0 - val, color='#f78166', lw=0.7, ls='--', alpha=0.6)
    #         ax.axhline(1.0, color=WHT, lw=0.3, ls=':', alpha=0.25)
    #
    #     # ── Panel 0: PCC Voltage ──────────────────────────────────────────────
    #     ax = axes[0]
    #     ax.plot(t, r.V_pcc, color=C[0], lw=1.0, label='Bus 634 (DC PCC)', zorder=3)
    #     ax.plot(t, r.V_632, color=C[1], lw=0.7, ls='--', alpha=0.8, label='Bus 632')
    #     ax.plot(t, r.V_680, color=C[3], lw=0.7, ls=':', alpha=0.7, label='Bus 680 (DG)')
    #     add_hvoltage_ref(ax)
    #     ax.set_ylabel('|V| (pu)', fontsize=8)
    #     ax.set_title('Bus Voltage Profiles — IEEE 13-Bus Distribution Feeder',
    #                  color=WHT, fontsize=9, loc='left', pad=4)
    #     ax.legend(fontsize=7, facecolor=BG2, labelcolor=MUT,
    #               edgecolor='#d0d7de', loc='lower right', ncol=3)
    #     shade_phases(ax); shade_events(ax)
    #
    #     # ── Panel 1: Frequency ────────────────────────────────────────────────
    #     ax = axes[1]
    #     ax.plot(t, r.freq, color=C[5], lw=1.0, zorder=3)
    #     ax.axhline(60.0, color=WHT, lw=0.3, ls=':', alpha=0.3)
    #     ax.axhline(59.5, color='#f78166', lw=0.7, ls='--', alpha=0.6,
    #                label='59.5 Hz (FRT soft)')
    #     ax.axhline(60.5, color='#f78166', lw=0.7, ls='--', alpha=0.6)
    #     rocof = np.gradient(r.freq, T_MACRO)
    #     ax2 = ax.twinx()
    #     ax2.plot(t, rocof, color=C[4], lw=0.6, alpha=0.55, label='ROCOF (Hz/s)')
    #     ax2.set_ylabel('ROCOF (Hz/s)', color=MUT, fontsize=7)
    #     ax2.tick_params(colors=MUT, labelsize=7)
    #     ax2.set_facecolor(BG2)
    #     ax.set_ylabel('f (Hz)', fontsize=8)
    #     ax.set_title('System Frequency & Rate-of-Change-of-Frequency (ROCOF)',
    #                  color=WHT, fontsize=9, loc='left', pad=4)
    #     ax.legend(fontsize=7, facecolor=BG2, labelcolor=MUT, edgecolor='#d0d7de')
    #     shade_phases(ax); shade_events(ax)
    #
    #     # ── Panel 2: Datacenter Active Power ──────────────────────────────────
    #     ax = axes[2]
    #     ax.plot(t, r.P_dc, color=C[0], lw=1.0, label='P_DC total', zorder=3)
    #     ax.plot(t, r.P_server / 1e3, color=C[3], lw=0.7, ls='--',
    #             alpha=0.8, label='P_server (MW)')
    #     ax.plot(t, r.P_cool / 1e3,  color=C[4], lw=0.7, ls=':',
    #             alpha=0.7, label='P_HVAC (MW)')
    #     ax.plot(t, r.dP_fw, color=C[7], lw=0.8, ls='-.',
    #             alpha=0.9, label='ΔP freq-watt droop')
    #     ax.axhline(0, color=WHT, lw=0.3, ls=':', alpha=0.2)
    #     ax.set_ylabel('P (MW)', fontsize=8)
    #     ax.set_title('Datacenter Active Power — IT Load / HVAC / Freq-Watt Droop',
    #                  color=WHT, fontsize=9, loc='left', pad=4)
    #     ax.legend(fontsize=7, facecolor=BG2, labelcolor=MUT,
    #               edgecolor='#d0d7de', ncol=2)
    #     shade_phases(ax); shade_events(ax)
    #
    #     # ── Panel 3: Datacenter Reactive Power ────────────────────────────────
    #     ax = axes[3]
    #     ax.plot(t, r.Q_dc, color=C[2], lw=1.0, label='Q_DC total', zorder=3)
    #     ax.plot(t, r.Q_cool / 1e3, color=C[4], lw=0.7, ls='--',
    #             alpha=0.8, label='Q_HVAC (MVAR)')
    #     ax.plot(t, r.dQ_vv, color=C[1], lw=0.8, ls='-.',
    #             alpha=0.9, label='ΔQ volt-VAR droop')
    #     ax.set_ylabel('Q (MVAR)', fontsize=8)
    #     ax.set_title('Datacenter Reactive Power — HVAC / Volt-VAR Droop Response',
    #                  color=WHT, fontsize=9, loc='left', pad=4)
    #     ax.legend(fontsize=7, facecolor=BG2, labelcolor=MUT,
    #               edgecolor='#d0d7de', ncol=2)
    #     shade_phases(ax); shade_events(ax)
    #
    #     # ── Panel 4: Motor Dynamics ───────────────────────────────────────────
    #     ax = axes[4]
    #     ax.plot(t, r.omega_r, color=C[1], lw=1.0, label='ω_r (pu)', zorder=3)
    #     ax.axhline(1.0, color=WHT, lw=0.3, ls=':', alpha=0.25,
    #                label='Synchronous speed')
    #     ax2 = ax.twinx()
    #     ax2.fill_between(t, r.slip * 100, alpha=0.25, color=C[3])
    #     ax2.plot(t, r.slip * 100, color=C[3], lw=0.6, alpha=0.8, label='Slip (%)')
    #     ax2.set_ylabel('Slip (%)', color=MUT, fontsize=7)
    #     ax2.tick_params(colors=MUT, labelsize=7)
    #     ax2.set_facecolor(BG2)
    #     ax.set_ylabel('ω_r (pu)', fontsize=8)
    #     ax.set_title('HVAC Induction Motor Speed & Slip Dynamics',
    #                  color=WHT, fontsize=9, loc='left', pad=4)
    #     ax.legend(fontsize=7, facecolor=BG2, labelcolor=MUT, edgecolor='#d0d7de')
    #     shade_phases(ax); shade_events(ax)
    #
    #     # ── Panel 5: DG Generation ────────────────────────────────────────────
    #     ax = axes[5]
    #     ax.fill_between(t, r.P_gen_dg, alpha=0.25, color=C[1])
    #     ax.plot(t, r.P_gen_dg, color=C[1], lw=1.0, label='Total DG output')
    #     # Overlay feeder total load for context
    #     ax.plot(t, r.P_load_total, color=C[5], lw=0.7, ls='--',
    #             alpha=0.7, label='Total feeder load')
    #     ax.set_ylabel('P (MW)', fontsize=8)
    #     ax.set_title('DG Generation Profile (Wind + Solar) vs. Total Feeder Load',
    #                  color=WHT, fontsize=9, loc='left', pad=4)
    #     ax.legend(fontsize=7, facecolor=BG2, labelcolor=MUT, edgecolor='#d0d7de')
    #     shade_phases(ax); shade_events(ax)
    #
    #     # ── Panel 6: Feeder Losses ────────────────────────────────────────────
    #     ax = axes[6]
    #     ax.fill_between(t, r.P_losses * 1e3, alpha=0.30, color=C[4])
    #     ax.plot(t, r.P_losses * 1e3, color=C[4], lw=0.9, label='I²R losses (kW)')
    #     ax.set_ylabel('Losses (kW)', fontsize=8)
    #     ax.set_title('Feeder Active Power Losses',
    #                  color=WHT, fontsize=9, loc='left', pad=4)
    #     shade_phases(ax); shade_events(ax)
    #
    #     # ── Panel 7: Voltage deviation histogram (inset) + time series ────────
    #     ax = axes[7]
    #     V_dev = (r.V_pcc - 1.0) * 100   # % deviation from nominal
    #     ax.fill_between(t, V_dev, alpha=0.3, color=C[0])
    #     ax.plot(t, V_dev, color=C[0], lw=0.9, label='Bus 634 voltage deviation (%)')
    #     ax.axhline(5,  color='#f78166', lw=0.7, ls='--', alpha=0.6)
    #     ax.axhline(-5, color='#f78166', lw=0.7, ls='--', alpha=0.6,
    #                label='±5% ANSI limit')
    #     ax.axhline(0,  color=WHT, lw=0.3, ls=':', alpha=0.25)
    #     ax.set_ylabel('ΔV (%)', fontsize=8)
    #     ax.set_title('PCC Voltage Deviation from Nominal (%) — ANSI C84.1 Range A',
    #                  color=WHT, fontsize=9, loc='left', pad=4)
    #     ax.legend(fontsize=7, facecolor=BG2, labelcolor=MUT, edgecolor='#d0d7de')
    #     shade_phases(ax); shade_events(ax)
    #
    #     # ── Panel 8: LVRT/FRT status + ride-through ───────────────────────────
    #     ax = axes[8]
    #     ax.fill_between(t, r.riding_through.astype(float),
    #                     step='post', alpha=0.6, color='#00bfff',
    #                     label='LVRT/FRT ride-through active')
    #     ax.fill_between(t, r.fault_active.astype(float),
    #                     step='post', alpha=0.5, color='#f78166',
    #                     label='Fault active')
    #     ax.fill_between(t, r.cap_out.astype(float),
    #                     step='post', alpha=0.4, color=C[7],
    #                     label='Capacitor bank out')
    #     ax.set_ylim(-0.05, 1.5)
    #     ax.set_yticks([0, 1])
    #     ax.set_yticklabels(['OFF', 'ON'], fontsize=7, color=MUT)
    #     ax.set_ylabel('Status', fontsize=8)
    #     ax.set_title('Protection Status — LVRT/FRT / Fault / Capacitor',
    #                  color=WHT, fontsize=9, loc='left', pad=4)
    #     ax.legend(fontsize=7, facecolor=BG2, labelcolor=MUT,
    #               edgecolor='#d0d7de', ncol=3)
    #     shade_phases(ax)
    #
    #     # ── Panel 9: Co-simulation quality (GS iterations) ───────────────────
    #     ax = axes[9]
    #     ax.fill_between(t, r.gs_iters, step='post',
    #                     alpha=0.5, color=C[8])
    #     ax.step(t, r.gs_iters, color=C[8], lw=0.8, where='post',
    #             label='GS iterations per macro step')
    #     ax.axhline(MAX_GS_ITER, color='#f78166', lw=0.7, ls='--', alpha=0.6,
    #                label=f'Max ({MAX_GS_ITER})')
    #     ax.set_ylim(0, MAX_GS_ITER + 0.5)
    #     ax.set_ylabel('Iterations', fontsize=8)
    #     ax.set_xlabel('Time (s)', color=MUT, fontsize=9)
    #     ax.set_title('Co-Simulation Interface — Gauss-Seidel Convergence',
    #                  color=WHT, fontsize=9, loc='left', pad=4)
    #     ax.legend(fontsize=7, facecolor=BG2, labelcolor=MUT, edgecolor='#d0d7de')
    #     shade_phases(ax); shade_events(ax)
    #
    #     # ── Phase legend at top ───────────────────────────────────────────────
    #     ph_patches = [
    #         mpatches.Patch(color=ph.color, alpha=0.5, label=f"Ph{i+1}: {ph.name}")
    #         for i, ph in enumerate(PHASES)
    #     ]
    #     ev_lines = [
    #         mpatches.Patch(color='#f78166', label='Fault on/off'),
    #         mpatches.Patch(color='#e3b341', label='Cap switching'),
    #         mpatches.Patch(color='#d2a8ff', label='Freq event'),
    #     ]
    #     fig.legend(
    #         handles=ph_patches + ev_lines,
    #         loc='upper center', ncol=4,
    #         facecolor=BG2, labelcolor=MUT, edgecolor='#d0d7de',
    #         fontsize=7.5, bbox_to_anchor=(0.5, 0.997)
    #     )
    #
    #     fig.suptitle(
    #         "Distribution System Dynamics Study — IEEE 13-Bus Feeder + Grid-Supporting Datacenter\n"
    #         "OpenDSS Newton-Raphson Power Flow  ·  3rd-Order IM HVAC  ·  "
    #         "VSC Freq-Watt / Volt-VAR  ·  IEEE 1547-2018 LVRT/FRT  ·  "
    #         "5-Phase Scenario",
    #         color=WHT, fontsize=10, y=1.008, fontweight='medium'
    #     )
    #
    #     fig.savefig(OUT_PLOT, dpi=150, bbox_inches='tight',
    #                 facecolor=BG, edgecolor='none')
    #     plt.close()
    #     print(f"  Plot →  {OUT_PLOT}")
    def plot(self):
        r = self.rec
        t = TIME

        # ── Colour palette ────────────────────────────────────────────────────
        BG  = '#ffffff'; BG2 = '#f6f8fa'; MUT = '#000000'; WHT = '#000000'
        C   = ['#0969da','#1a7f37','#8250df','#bc4c00',
               '#cf222e','#0550ae','#2da44e','#9a6700',
               '#a40e26','#0969da']

        fig = plt.figure(figsize=(14, 28))
        fig.patch.set_facecolor(BG)
        gs_layout = gridspec.GridSpec(
            10, 1, figure=fig,
            hspace=0.52, top=0.96, bottom=0.04,
            left=0.09, right=0.97
        )
        axes = [fig.add_subplot(gs_layout[i]) for i in range(7)]

        for ax in axes:
            ax.set_facecolor(BG)
            ax.tick_params(colors=MUT, labelsize=10)
            ax.yaxis.label.set_color(MUT)
            for sp in ax.spines.values():
                sp.set_edgecolor('#d0d7de'); sp.set_linewidth(2)
            ax.grid(axis='y', color='#d0d7de', lw=0.5, ls=':')

        def shade_phases(ax):
            """Shade each operating phase with its assigned colour."""
            for ph in PHASES:
                ax.axvspan(ph.t_start, ph.t_end,
                           color=ph.color, alpha=0.2, zorder=0)

        def shade_events(ax):
            """Mark key events with vertical lines."""
            ev_styles = {
                'fault_on':  ('#f78166', '--', 1.4),
                'fault_off': ('#3fb950', '--', 1.0),
                'cap_out':   ('#e3b341', ':',  1.0),
                'cap_in':    ('#3fb950', ':',  1.0),
                'freq_step': ('#d2a8ff', '-.',  1.0),
            }
            for ev in EVENTS:
                style = ev_styles.get(ev.kind, (MUT, '-', 0.8))
                ax.axvline(ev.t, color=style[0], ls=style[1],
                           lw=style[2], alpha=0.85, zorder=4)

        def add_hvoltage_ref(ax, val=0.95, label='0.95 pu'):
            ax.axhline(val, color='#f78166', lw=0.7, ls='--', alpha=0.6, label=label)
            ax.axhline(2.0 - val, color='#f78166', lw=0.7, ls='--', alpha=0.6)
            ax.axhline(1.0, color=WHT, lw=0.3, ls=':', alpha=0.25)

        # ── Panel 0: PCC Voltage ──────────────────────────────────────────────
        ax = axes[0]
        ax.plot(t, r.V_pcc, color=C[0], lw=2.0, label='Bus 634 (DC PCC)', zorder=3)
        ax.plot(t, r.V_632, color=C[1], lw=2.0, ls='--', alpha=1, label='Bus 632')
        ax.plot(t, r.V_680, color=C[3], lw=2.0, ls=':', alpha=1, label='Bus 680 (DG)')
        add_hvoltage_ref(ax)
        ax.set_ylabel('|V| (pu)', fontsize=14)
        ax.set_title('Bus Voltage Profiles',
                     color=WHT, fontsize=14, loc='left', pad=4)
        ax.legend(fontsize=14, facecolor=BG2, labelcolor=MUT,
                  edgecolor='#d0d7de', loc='lower right', ncol=3)
        shade_phases(ax); shade_events(ax)

        # ── Panel 1: Frequency ────────────────────────────────────────────────
        ax = axes[1]
        ax.plot(t, r.freq, color=C[5], lw=2.0, zorder=3)
        ax.axhline(60.0, color=WHT, lw=2.0, ls=':', alpha=0.3)
        ax.axhline(59.5, color='#f78166', lw=2.0, ls='--', alpha=0.6,
                   label='59.5 Hz (FRT soft)')
        ax.axhline(60.5, color='#f78166', lw=2.0, ls='--', alpha=0.6)
        rocof = np.gradient(r.freq, T_MACRO)
        ax2 = ax.twinx()
        ax2.plot(t, rocof, color=C[4], lw=2.0, alpha=0.55, label='ROCOF (Hz/s)')
        ax2.set_ylabel('ROCOF (Hz/s)', color=MUT, fontsize=14)
        ax2.tick_params(colors=MUT, labelsize=10)
        ax2.set_facecolor(BG)
        ax.set_ylabel('f (Hz)', fontsize=14)
        ax.set_title('System Frequency & Rate-of-Change-of-Frequency (ROCOF)',
                     color=WHT, fontsize=14, loc='left', pad=4)
        ax.legend(fontsize=14, facecolor=BG2, labelcolor=MUT, edgecolor='#d0d7de')
        shade_phases(ax); shade_events(ax)

        # ── Panel 2: Datacenter Active Power ──────────────────────────────────
        ax = axes[2]
        ax.plot(t, r.P_dc, color=C[0], lw=2.0, label='P_DC total', zorder=3)
        ax.plot(t, r.P_server / 1e3, color=C[3], lw=2.0, ls='--',
                alpha=0.8, label='P_server (MW)')
        ax.plot(t, r.P_cool / 1e3,  color=C[4], lw=2.0, ls=':',
                alpha=0.7, label='P_HVAC (MW)')
        ax.plot(t, r.dP_fw, color=C[7], lw=2.0, ls='-.',
                alpha=0.9, label='ΔP freq-watt droop')
        ax.axhline(0, color=WHT, lw=0.3, ls=':', alpha=0.2)
        ax.set_ylabel('P (MW)', fontsize=14)
        ax.set_title('Datacenter Active Power — IT Load / HVAC / Freq-Watt Droop',
                     color=WHT, fontsize=14, loc='left', pad=4)
        ax.legend(fontsize=14, facecolor=BG2, labelcolor=MUT,
                  edgecolor='#d0d7de', ncol=2)
        shade_phases(ax); shade_events(ax)

        # ── Panel 3: Datacenter Reactive Power ────────────────────────────────
        ax = axes[3]
        ax.plot(t, r.Q_dc, color=C[2], lw=2.0, label='Q_DC total', zorder=3)
        ax.plot(t, r.Q_cool / 1e3, color=C[4], lw=2.0, ls='--',
                alpha=0.8, label='Q_HVAC (MVAR)')
        ax.plot(t, r.dQ_vv, color=C[1], lw=2.0, ls='-.',
                alpha=0.9, label='ΔQ volt-VAR droop')
        ax.set_ylabel('Q (MVAR)', fontsize=14)
        ax.set_title('Datacenter Reactive Power — HVAC / Volt-VAR Droop Response',
                     color=WHT, fontsize=14, loc='left', pad=4)
        ax.legend(fontsize=14, facecolor=BG2, labelcolor=MUT,
                  edgecolor='#d0d7de', ncol=2)
        shade_phases(ax); shade_events(ax)

        # ── Panel 4: Motor Dynamics ───────────────────────────────────────────
        ax = axes[4]
        ax.plot(t, r.omega_r, color=C[1], lw=2.0, label='ω_r (pu)', zorder=3)
        ax.axhline(1.0, color=WHT, lw=0.3, ls=':', alpha=0.25,
                   label='Synchronous speed')
        # ax2 = ax.twinx()
        # ax2.fill_between(t, r.slip * 100, alpha=0.25, color=C[3])
        # ax2.plot(t, r.slip * 100, color=C[3], lw=2.0, alpha=0.8, label='Slip (%)')
        # ax2.set_ylabel('Slip (%)', color=MUT, fontsize=7)
        # ax2.tick_params(colors=MUT, labelsize=10)
        # ax2.set_facecolor(BG)
        ax.set_ylabel('ω_r (pu)', fontsize=14)
        ax.set_title('HVAC Induction Motor Speed & Slip Dynamics',
                     color=WHT, fontsize=14, loc='left', pad=4)
        ax.legend(fontsize=14, facecolor=BG2, labelcolor=MUT, edgecolor='#d0d7de')
        shade_phases(ax); shade_events(ax)

        # ── Panel 5: DG Generation ────────────────────────────────────────────
        # ax = axes[5]
        # ax.fill_between(t, r.P_gen_dg, alpha=0.25, color=C[1])
        # ax.plot(t, r.P_gen_dg, color=C[1], lw=1.0, label='Total DG output')
        # # Overlay feeder total load for context
        # ax.plot(t, r.P_load_total, color=C[5], lw=0.7, ls='--',
        #         alpha=0.7, label='Total feeder load')
        # ax.set_ylabel('P (MW)', fontsize=8)
        # ax.set_title('DG Generation Profile (Wind + Solar) vs. Total Feeder Load',
        #              color=WHT, fontsize=9, loc='left', pad=4)
        # ax.legend(fontsize=7, facecolor=BG2, labelcolor=MUT, edgecolor='#d0d7de')
        # shade_phases(ax); shade_events(ax)

        # ── Panel 6: Feeder Losses ────────────────────────────────────────────
        # ax = axes[6]
        # ax.fill_between(t, r.P_losses * 1e3, alpha=0.30, color=C[4])
        # ax.plot(t, r.P_losses * 1e3, color=C[4], lw=0.9, label='I²R losses (kW)')
        # ax.set_ylabel('Losses (kW)', fontsize=8)
        # ax.set_title('Feeder Active Power Losses',
        #              color=WHT, fontsize=9, loc='left', pad=4)
        # shade_phases(ax); shade_events(ax)

        # ── Panel 7: Voltage deviation histogram (inset) + time series ────────
        # ax = axes[7]
        # V_dev = (r.V_pcc - 1.0) * 100   # % deviation from nominal
        # ax.fill_between(t, V_dev, alpha=0.3, color=C[0])
        # ax.plot(t, V_dev, color=C[0], lw=0.9, label='Bus 634 voltage deviation (%)')
        # ax.axhline(5,  color='#f78166', lw=0.7, ls='--', alpha=0.6)
        # ax.axhline(-5, color='#f78166', lw=0.7, ls='--', alpha=0.6,
        #            label='±5% ANSI limit')
        # ax.axhline(0,  color=WHT, lw=0.3, ls=':', alpha=0.25)
        # ax.set_ylabel('ΔV (%)', fontsize=8)
        # ax.set_title('PCC Voltage Deviation from Nominal (%) — ANSI C84.1 Range A',
        #              color=WHT, fontsize=9, loc='left', pad=4)
        # ax.legend(fontsize=7, facecolor=BG2, labelcolor=MUT, edgecolor='#d0d7de')
        # shade_phases(ax); shade_events(ax)

        # ── Panel 8: LVRT/FRT status + ride-through ───────────────────────────
        ax = axes[5]
        ax.fill_between(t, r.riding_through.astype(float),
                        step='post', alpha=0.6, color='#00bfff',
                        label='LVRT/FRT ride-through active')
        ax.fill_between(t, r.fault_active.astype(float),
                        step='post', alpha=0.8, color='#f78166',
                        label='Fault active')
        ax.fill_between(t, r.cap_out.astype(float),
                        step='post', alpha=0.4, color=C[7],
                        label='Capacitor bank out')
        ax.set_ylim(-0.05, 1.5)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['OFF', 'ON'], fontsize=14, color=MUT)
        ax.set_ylabel('Status', fontsize=14)
        ax.set_title('Protection Status — LVRT/FRT / Fault / Capacitor',
                     color=WHT, fontsize=14, loc='left', pad=4)
        ax.legend(fontsize=14, facecolor=BG2, labelcolor=MUT,
                  edgecolor='#d0d7de', ncol=3)
        shade_phases(ax)

        # ── Panel 9: Co-simulation quality (GS iterations) ───────────────────
        ax = axes[6]
        ax.fill_between(t, r.gs_iters, step='post',
                        alpha=0.5, color=C[8])
        # ax.step(t, r.gs_iters, color=C[8], lw=0.8, where='post',
        #         label='GS iterations per macro step')
        ax.axhline(MAX_GS_ITER, color='#f78166', lw=0.7, ls='--', alpha=0.6,
                   label=f'Max ({MAX_GS_ITER})')
        ax.set_ylim(0, MAX_GS_ITER + 0.5)
        ax.set_ylabel('Iterations', fontsize=14)
        ax.set_xlabel('Time (s)', color=MUT, fontsize=14)
        ax.set_title('Co-Simulation Interface — Gauss-Seidel Convergence',
                     color=WHT, fontsize=14, loc='left', pad=4)
        ax.legend(fontsize=14, facecolor=BG2, labelcolor=MUT, edgecolor='#d0d7de')
        shade_phases(ax); shade_events(ax)

        # ── Phase legend at top ───────────────────────────────────────────────
        ph_patches = [
            mpatches.Patch(color=ph.color, alpha=0.5, label=f"Ph{i+1}: {ph.name}")
            for i, ph in enumerate(PHASES)
        ]
        ev_lines = [
            mpatches.Patch(color='#f78166', label='Fault on/off'),
            mpatches.Patch(color='#e3b341', label='Cap switching'),
            mpatches.Patch(color='#d2a8ff', label='Freq event'),
        ]
        fig.legend(
            handles=ph_patches,
            loc='upper center', ncol=3,
            facecolor=BG2, labelcolor=MUT, edgecolor='#d0d7de',
            fontsize=14, bbox_to_anchor=(0.5, 0.997)
        )

        # fig.suptitle(
        #     "Distribution System Dynamics Study — IEEE 13-Bus Feeder + Grid-Supporting Datacenter\n"
        #     "OpenDSS Newton-Raphson Power Flow  ·  3rd-Order IM HVAC  ·  "
        #     "VSC Freq-Watt / Volt-VAR  ·  IEEE 1547-2018 LVRT/FRT  ·  "
        #     "5-Phase Scenario",
        #     color=WHT, fontsize=14, y=1.008, fontweight='medium'
        # )

        fig.savefig(OUT_PLOT, dpi=150, bbox_inches='tight',
                    facecolor=BG, edgecolor='none')
        # plt.show()
        print(f"  Plot →  {OUT_PLOT}")

    #  WRITTEN REPORT
    def report(self):
        r   = self.rec
        t   = TIME
        sep = "=" * 74

        lines = [
            sep,
            "  DISTRIBUTION SYSTEM DYNAMICS STUDY — FINAL REPORT",
            sep,
            "",
            "  STUDY CONFIGURATION",
            f"    Feeder         : IEEE 13-bus, 4.16 kV, 5 MVA base",
            f"    Datacenter     : {DC_NAME}  (2 MVA, bus 634, 0.48 kV service)",
            f"    Duration       : {SIM_DURATION:.0f} s",
            f"    Time step      : T_macro={T_MACRO}s  T_micro={DT_MICRO}s",
            f"    Interface      : Gauss-Seidel (max {MAX_GS_ITER} iter, "
                                  f"tol={GS_TOL} pu)",
            "",
            "  OPERATING PHASES",
            "  " + "─" * 70,
        ]

        for i, ph in enumerate(PHASES):
            lines += [
                f"  Phase {i+1}: {ph.name}  (t={ph.t_start:.0f}–{ph.t_end:.0f}s)",
                f"    {textwrap.fill(ph.description, 68, subsequent_indent='    ')}",
            ]
            if i < len(self._phase_stats):
                ps = self._phase_stats[i]
                lines += [
                    f"    V_pcc: mean={ps['V_pcc_mean']:.4f}  min={ps['V_pcc_min']:.4f} pu  |  "
                    f"f_mean={ps['freq_mean']:.4f} Hz  |  "
                    f"P_dc_mean={ps['P_dc_mean']:.3f} MW  |  "
                    f"RT_steps={ps['RT_steps']}",
                ]
            lines.append("")

        lines += [
            "  " + "─" * 70,
            "  SIMULATION RESULTS SUMMARY",
            "  " + "─" * 70,
            f"  Voltage  — Bus 634 PCC: "
                f"mean={r.V_pcc.mean():.4f}  "
                f"min={r.V_pcc.min():.4f}  "
                f"max={r.V_pcc.max():.4f} pu",
            f"  Frequency:              "
                f"mean={r.freq.mean():.4f}  "
                f"min={r.freq.min():.4f}  "
                f"max={r.freq.max():.4f} Hz",
            f"  Datacenter P:           "
                f"mean={r.P_dc.mean():.4f} MW  "
                f"max={r.P_dc.max():.4f} MW",
            f"  Datacenter Q:           "
                f"mean={r.Q_dc.mean():.4f} MVAR",
            f"  Motor speed ω_r:        "
                f"mean={r.omega_r.mean():.4f}  "
                f"min={r.omega_r.min():.4f} pu",
            f"  Feeder losses:          "
                f"mean={r.P_losses.mean():.4f} MW  "
                f"({100*r.P_losses.mean()/max(r.P_load_total.mean(),0.01):.2f}% of load)",
            f"  LVRT/FRT ride-through:  "
                f"{r.riding_through.sum()} steps  "
                f"({100*r.riding_through.mean():.1f}%)",
            f"  GS avg iterations:      {r.gs_iters.mean():.2f}",
            "",
            "  OUTPUT FILES",
            f"    {OUT_PLOT}",
            f"    {OUT_CSV}",
            f"    {OUT_RPT}",
            "",
            sep,
        ]

        text = "\n".join(lines)
        print("\n" + text)
        with open(OUT_RPT, 'w') as f:
            f.write(text + "\n")
        print(f"\n  Report →  {OUT_RPT}")

    #  BANNER
    def _print_banner(self):
        print()
        print("┌" + "─" * 72 + "┐")
        print("│" + " " * 72 + "│")
        print("│   DISTRIBUTION SYSTEM DYNAMICS STUDY" + " " * 35 + "│")
        print("│   IEEE 13-Bus Feeder  +  Grid-Supporting Datacenter" + " " * 21 + "│")
        print("│" + " " * 72 + "│")
        print("│   Modules used:" + " " * 56 + "│")
        print("│     datacenter_registry.py  →  component registration" + " " * 19 + "│")
        print("│     datacenter_core.py      →  canonical interface" + " " * 22 + "│")
        print("│     adapters.py             →  DistributionAdapter" + " " * 22 + "│")
        print("│     datacenter_subsystem.py →  physics (IM + VSC)" + " " * 23 + "│")
        print("│     opendss_13bus_network.py      →  IEEE 13-bus NR solver" + " " * 20 + "│")
        print("│" + " " * 72 + "│")
        print("└" + "─" * 72 + "┘")


#  ======================================= Main Program ======================================
if __name__ == "__main__":

    # ── Instantiate study
    study = DistributionDynamicsStudy()
    # ── Run simulation ────────────────────────────────────────────────────────
    study.run()
    # ── Post-simulation analysis ──────────────────────────────────────────────
    study.analyse()

    # ── Save outputs ──────────────────────────────────────────────────────────
    print("\nSaving outputs …")
    study.save_csv()
    study.plot()
    study.report()

    print("\nDone. All outputs written to:")
    print(f"  {OUT_DIR}")
    sys.exit(0)
