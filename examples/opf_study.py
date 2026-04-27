"""
optimal_powerflow_study.py
==========================
Main Program — Optimal Power Flow Study
=========================================
Part of the OALS-grid modular datacenter model.  Uses the registry/adapter architecture to perform a
multi-period, network-constrained OPF analysis of the IEEE 13-bus feeder with
an embedded price-responsive datacenter load.

Study Overview
--------------
The datacenter is treated as a *dispatchable load* that submits price-responsive
bids to the system operator and modulates consumption in response to LMPs, voltage
conditions, and feeder congestion.

24-hour rolling horizon, 5-min intervals (288 total).  Four scenarios:

  S1  Baseline         — no DR; datacenter runs at unconstrained operating point
  S2  Price-Responsive — DR enabled; load reduced when LMP > 50 $/MWh
  S3  Volt-VAR Support — DR + reactive injection from VSC when V < 1.0 pu
  S4  Congestion Case  — same as S3 but with 40% peak load increase

OPF Formulation (DC-OPF approximation + AC verification)
---------------------------------------------------------
  min  Σ_g C_g(P_g) + C_loss·P_loss − C_DR·ΔP_DC

  s.t. Power balance per node (NR power flow)
       0.95 ≤ |V_i| ≤ 1.05 pu   (ANSI C84.1 Range A)
       Generator limits P_g ∈ [P_min, P_max]
       DC flexibility  P_DC ∈ [P_committed, P_max]

Physics model notes
-------------------
At OPF timescale (5-min intervals), transient dynamics are negligible and
the datacenter is treated as quasi-static.  The subsystem is stepped at
dt_micro=0.02 s (numerically stable for the 3rd-order IM circuit) with
nsub=1, advancing one micro-step per OPF sample to track the slowly-evolving
operating point (GPU load profile, motor slip, VSC droop state).  This gives:

  • Correct steady-state P/Q, omega_r, slip at each interval
  • No numerical instability (dt < RK4 stability limit of ~0.03 s for this IM)
  • Fast execution: ~0.03 s per scenario (288 × one micro-step)

"""

import sys
import time
import textwrap
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches

from datacenter_registry import register, get_datacenter, deregister
from testsystems.opendss_13bus_network import OpenDSSNetworkSimulator, NetworkResults
from adapters import OPFAdapter
from datacenter_core import CanonicalInput, CanonicalOutput
import os

# =============================================================================
#  STUDY CONFIGURATION
# =============================================================================
HOURS          = 24
DT_MIN         = 5           # OPF interval [min]
DT_S           = DT_MIN * 60 # OPF interval [s]
N_INTERVALS    = HOURS * 60 // DT_MIN    # 288
INTERVAL_TIMES = np.arange(N_INTERVALS, dtype=float) * DT_S
INTERVAL_HOURS = INTERVAL_TIMES / 3600.0

V_MIN   = 0.95       # ANSI C84.1 Range A
V_MAX   = 1.05
BASE_MVA = 5.0
BASE_KV  = 4.16

# Generator cost curves  a·P² + b·P + c  [$/MWh]
GEN_SPECS = {
    'G1_slack':    {'P_min': 0.0, 'P_max': 8.0, 'a': 1.50, 'b': 28.0, 'c': 120},
    'G2_peaker':   {'P_min': 0.2, 'P_max': 2.5, 'a': 4.00, 'b': 52.0, 'c':  80},
    'G3_baseload': {'P_min': 0.5, 'P_max': 2.0, 'a': 0.80, 'b': 22.0, 'c': 200},
}
C_DR    = 45.0   # $/MWh value of demand response
C_LOSS  = 35.0   # $/MWh value of loss reduction

OUT_DIR   = '../Outputs'
OUT_PLOT  = os.path.join(OUT_DIR, 'opf_results.png') #OUT_DIR / "distribution_dynamics_results.png"
OUT_CSV   = os.path.join(OUT_DIR, 'opf_timeseries.csv')
OUT_RPT   = os.path.join(OUT_DIR, 'opf_report.txt')
# OUT_DIR  = Path('/mnt/user-data/outputs')
# OUT_PLOT = OUT_DIR / 'opf_results.png'
# OUT_CSV  = OUT_DIR / 'opf_timeseries.csv'
# OUT_RPT  = OUT_DIR / 'opf_report.txt'

DC_NAME   = 'DC_OPF_634'

# Physics config:
#   dt_micro = 0.02 s → numerically stable for the 3rd-order IM circuit
#   dt = dt_micro     → nsub = 1  (one micro-step per OPF call)
#   This tracks the quasi-static operating point without integrating the full
#   5-min interval at 0.02 s resolution (which would be 15 000 steps/interval).
DC_CONFIG = {
    'seed':            42,
    'n_cooling_units':  3,
    'dt_micro':         0.02,
    'dt':               0.02,   # nsub = dt / dt_micro = 1
    'price_threshold':  50.0,
    'price_max':       200.0,
    'max_curtail_pu':   0.30,
}

# Warm-up duration: run physics for WARMUP_STEPS before the OPF horizon to
# reach steady-state motor speed and VSC operating point.
WARMUP_STEPS = 500


# =============================================================================
#  FORECAST PROFILES
# =============================================================================
def build_lmp_profile() -> np.ndarray:
    """Synthetic 24-hour LMP [$/MWh]: off-peak trough, morning peak, solar dip, evening peak."""
    h   = INTERVAL_HOURS
    rng = np.random.default_rng(7)
    lmp = (30.0
           + 40.0 * np.exp(-0.5 * ((h - 8.5) / 1.5) ** 2)
           + 55.0 * np.exp(-0.5 * ((h - 18.5) / 1.2) ** 2)
           - 18.0 * np.exp(-0.5 * ((h - 12.5) / 2.0) ** 2)
           + 5.0  * rng.normal(0, 1, N_INTERVALS))
    return np.clip(lmp, 15.0, 150.0)


def build_dg_profiles() -> Tuple[np.ndarray, np.ndarray]:
    """Wind [MW] and solar [pu] generation profiles."""
    rng = np.random.default_rng(13)
    h   = INTERVAL_HOURS

    wind = np.clip(0.40 + 0.20 * np.cos(2 * np.pi * h / 24)
                   + rng.normal(0, 0.04, N_INTERVALS), 0.05, 0.60)

    t_rise, t_set = 5.5, 20.0
    solar = np.zeros(N_INTERVALS)
    for i, hh in enumerate(h):
        if t_rise < hh < t_set:
            clear = min((hh - t_rise) / 2, (t_set - hh) / 2, 1.0)
            cloud = rng.uniform(0.85, 1.0) if rng.random() > 0.15 else rng.uniform(0.40, 0.70)
            solar[i] = clear * cloud

    return wind, solar


def build_base_load_profile() -> np.ndarray:
    """Feeder base load [MW] (residential/commercial diurnal profile)."""
    h = INTERVAL_HOURS
    load = (2.20
            + 0.60 * np.exp(-0.5 * ((h -  8.0) / 2.0) ** 2)
            + 0.90 * np.exp(-0.5 * ((h - 19.0) / 1.8) ** 2)
            - 0.30 * np.exp(-0.5 * ((h - 13.0) / 2.5) ** 2))
    return np.clip(load, 1.5, 4.0)


# =============================================================================
#  NETWORK-CONSTRAINED OPF  (DC-OPF merit order + AC verification)
# =============================================================================
class NetworkOPF:
    """
    Simplified network-constrained OPF for the IEEE 13-bus feeder.

    Algorithm per 5-min interval
    ─────────────────────────────
    1. Economic merit-order dispatch → generator set-points + LMP estimate
    2. DR signal: if LMP > threshold, compute curtailed DC load
    3. Volt-VAR: run AC power flow; if V < 1.0 pu, inject reactive via droop
    4. Loss sensitivity: perturb DC load by 1% to estimate ∂P_loss/∂P_DC
    5. Nodal LMP at bus 634 = LMP_ref × (1 + loss_sensitivity)
    """

    def __init__(self, net: OpenDSSNetworkSimulator):
        self.net = net

    def solve_interval(
        self,
        t:            float,
        P_base_mw:    float,
        P_dc_bid:     float,
        P_dc_min:     float,
        P_dc_max:     float,
        Q_dc_bid:     float,
        lmp_ref:      float,
        dr_enabled:   bool,
        volt_support: bool,
    ) -> Dict:

        # ── 1. Generator dispatch ──────────────────────────────────────────────
        P_demand = P_base_mw + P_dc_bid
        gen_dispatch, gen_cost, lmp = self._merit_order(P_demand, lmp_ref)

        # ── 2. Demand response ────────────────────────────────────────────────
        if dr_enabled and lmp > DC_CONFIG['price_threshold']:
            P_dc = self._dr_dispatch(P_dc_bid, P_dc_min, P_dc_max, lmp)
        else:
            P_dc = P_dc_bid
        dr_mw = max(0.0, P_dc_bid - P_dc)

        # ── 3. AC power flow + volt-VAR ───────────────────────────────────────
        Q_dc = Q_dc_bid
        net0: NetworkResults = self.net.solve(t, P_dc, Q_dc, DT_S)

        if volt_support:
            dV   = net0.V_pcc_pu - 1.0
            Q_dc = Q_dc_bid - 0.10 * dV * BASE_MVA   # volt-VAR droop
            net0 = self.net.solve(t, P_dc, Q_dc, DT_S)

        # ── 4. Loss sensitivity ───────────────────────────────────────────────
        net1: NetworkResults = self.net.solve(t, P_dc * 1.01 + 1e-4, Q_dc, DT_S)
        dLoss = ((net1.P_losses_mw - net0.P_losses_mw)
                 / (P_dc * 0.01 + 1e-4 + 1e-9))
        lmp_pcc = lmp * (1.0 + dLoss)

        # ── 5. Economics ──────────────────────────────────────────────────────
        dt_h       = DT_S / 3600.0
        dr_value   = dr_mw * dt_h * C_DR
        loss_cost  = net0.P_losses_mw * dt_h * C_LOSS
        total_cost = gen_cost * dt_h + loss_cost - dr_value

        return {
            'P_dc_dispatch':  P_dc,
            'Q_dc_adjusted':  Q_dc,
            'P_dc_bid':       P_dc_bid,
            'dr_mw':          dr_mw,
            'gen_cost_rate':  gen_cost,
            'lmp_ref':        lmp,
            'lmp_pcc':        lmp_pcc,
            'loss_sensitivity': dLoss,
            'V_pcc':          net0.V_pcc_pu,
            'V_632':          net0.V_bus632_pu,
            'V_680':          net0.V_bus680_pu,
            'freq_hz':        net0.freq_hz,
            'P_losses':       net0.P_losses_mw,
            'P_gen_dg':       net0.P_gen_total_mw,
            'dr_value_usd':   dr_value,
            'loss_cost_usd':  loss_cost,
            'total_cost_usd': total_cost,
            'v_violation':    (net0.V_pcc_pu < V_MIN or net0.V_pcc_pu > V_MAX),
            'lmp_used_dr':    (dr_mw > 1e-4),
        }

    def _merit_order(self, P_demand: float, lmp_ref: float):
        gens     = sorted(GEN_SPECS.items(), key=lambda x: x[1]['b'])
        dispatch = {}
        P_left   = P_demand
        marginal = list(gens)[-1][0]

        for name, spec in gens:
            P_g = float(np.clip(P_left, 0.0, spec['P_max'] - spec['P_min']))
            dispatch[name] = P_g + spec['P_min']
            P_left -= P_g
            if P_left <= 0:
                marginal = name
                break

        total_cost = sum(
            s['a'] * dispatch.get(n, s['P_min'])**2
            + s['b'] * dispatch.get(n, s['P_min'])
            + s['c']
            for n, s in GEN_SPECS.items()
        )
        sm   = GEN_SPECS[marginal]
        Pm   = dispatch.get(marginal, sm['P_min'])
        lmp  = 0.70 * (2 * sm['a'] * Pm + sm['b']) + 0.30 * lmp_ref
        return dispatch, total_cost, lmp

    def _dr_dispatch(self, P_bid, P_min, P_max, lmp):
        cost_b, cost_a = 35.0, 2.5
        if lmp <= cost_b:
            return P_bid
        P_opt = P_bid - (lmp - cost_b) / (2.0 * cost_a) * 0.05
        return float(np.clip(P_opt, P_min, P_max))


# =============================================================================
#  RESULT RECORD
# =============================================================================
@dataclass
class ScenarioRecord:
    label: str
    P_dc:       np.ndarray = field(default_factory=lambda: np.zeros(N_INTERVALS))
    Q_dc:       np.ndarray = field(default_factory=lambda: np.zeros(N_INTERVALS))
    P_dc_bid:   np.ndarray = field(default_factory=lambda: np.zeros(N_INTERVALS))
    dr_mw:      np.ndarray = field(default_factory=lambda: np.zeros(N_INTERVALS))
    P_server:   np.ndarray = field(default_factory=lambda: np.zeros(N_INTERVALS))
    P_cool:     np.ndarray = field(default_factory=lambda: np.zeros(N_INTERVALS))
    omega_r:    np.ndarray = field(default_factory=lambda: np.zeros(N_INTERVALS))
    slip:       np.ndarray = field(default_factory=lambda: np.zeros(N_INTERVALS))
    V_pcc:      np.ndarray = field(default_factory=lambda: np.zeros(N_INTERVALS))
    V_632:      np.ndarray = field(default_factory=lambda: np.zeros(N_INTERVALS))
    V_680:      np.ndarray = field(default_factory=lambda: np.zeros(N_INTERVALS))
    freq_hz:    np.ndarray = field(default_factory=lambda: np.zeros(N_INTERVALS))
    P_losses:   np.ndarray = field(default_factory=lambda: np.zeros(N_INTERVALS))
    P_gen_dg:   np.ndarray = field(default_factory=lambda: np.zeros(N_INTERVALS))
    lmp_ref:    np.ndarray = field(default_factory=lambda: np.zeros(N_INTERVALS))
    lmp_pcc:    np.ndarray = field(default_factory=lambda: np.zeros(N_INTERVALS))
    total_cost: np.ndarray = field(default_factory=lambda: np.zeros(N_INTERVALS))
    dr_value:   np.ndarray = field(default_factory=lambda: np.zeros(N_INTERVALS))
    loss_cost:  np.ndarray = field(default_factory=lambda: np.zeros(N_INTERVALS))
    loss_sens:  np.ndarray = field(default_factory=lambda: np.zeros(N_INTERVALS))
    v_viol:     np.ndarray = field(default_factory=lambda: np.zeros(N_INTERVALS, bool))
    dr_active:  np.ndarray = field(default_factory=lambda: np.zeros(N_INTERVALS, bool))
    # Summaries
    total_cost_day:   float = 0.0
    total_energy_mwh: float = 0.0
    total_dr_mwh:     float = 0.0
    total_dr_value:   float = 0.0
    n_viol:           int   = 0


# =============================================================================
#  OPF STUDY
# =============================================================================
class OPFStudy:

    def __init__(self):
        self._banner()
        print('\n[1/5] Building forecast profiles …')
        self.lmp_profile           = build_lmp_profile()
        self.wind_mw, self.solar_pu = build_dg_profiles()
        self.base_load             = build_base_load_profile()

        print('[2/5] Configuring scenarios …')
        self.scenarios = {
            'S1_Baseline':      {'dr': False, 'volt': False, 'congestion': False},
            'S2_PriceResponse': {'dr': True,  'volt': False, 'congestion': False},
            'S3_VoltSupport':   {'dr': True,  'volt': True,  'congestion': False},
            'S4_Congestion':    {'dr': True,  'volt': True,  'congestion': True},
        }
        self.records: Dict[str, ScenarioRecord] = {}

        print('[3/5] Initialising IEEE 13-bus network …')
        self.net = OpenDSSNetworkSimulator(base_kv=BASE_KV, base_mva=BASE_MVA)
        self.opf = NetworkOPF(self.net)
        print('[4/5] Registering datacenter physics …')
        register(DC_NAME, DC_CONFIG)
        print('[5/5] Ready.\n')
        print(self.net.summary())

    # ─────────────────────────────────────────────────────────────────────────
    def run_all_scenarios(self):
        print(f'\n{"="*74}')
        print(f'  OPF STUDY  ·  24 h  ·  {N_INTERVALS} intervals  ·  {DT_MIN}-min resolution')
        print(f'{"="*74}')
        for sc_name, cfg in self.scenarios.items():
            self._run_scenario(sc_name, cfg)

    def _run_scenario(self, sc_name: str, cfg: Dict):
        print(f'\n  ── {sc_name} ──')
        flags = ('DR=ON' if cfg['dr'] else 'DR=OFF',
                 'Volt-VAR=ON' if cfg['volt'] else 'Volt-VAR=OFF',
                 'Congestion=ON' if cfg['congestion'] else '')
        print('    ' + '  ·  '.join(f for f in flags if f))

        # Fresh physics per scenario (independent trajectory)
        sc_dc_name = f'{DC_NAME}_{sc_name}'
        register(sc_dc_name, DC_CONFIG)
        dc: OPFAdapter = get_datacenter(sc_dc_name, 'opf', interval_min=DT_MIN)
        physics        = dc._p    # direct reference for state probing

        # ── Warm up to steady state ───────────────────────────────────────────
        # Step at dt_micro increments so the IM and VSC reach their
        # quasi-static operating point before the OPF horizon begins.
        for w in range(WARMUP_STEPS):
            physics.step(CanonicalInput(
                V_pu=1.0, freq_hz=60.0,
                t_sim=float(w) * DC_CONFIG['dt_micro'],
                dt=DC_CONFIG['dt_micro']))

        rec     = ScenarioRecord(label=sc_name)
        t_wall  = time.time()
        dt_h    = DT_S / 3600.0

        # Congestion load scale
        load_scale = np.ones(N_INTERVALS)
        if cfg['congestion']:
            h = INTERVAL_HOURS
            load_scale += (0.40 * np.exp(-0.5 * ((h -  8.0) / 1.5)**2)
                         + 0.35 * np.exp(-0.5 * ((h - 19.0) / 1.5)**2))

        # ── Step the physics sequentially at dt_micro, sampling once per OPF
        #    interval to get the quasi-static operating point.
        #    The OPF interval index k maps to physics time:
        #      t_phys = k * DC_CONFIG['dt_micro']
        #    This is NOT the wall-clock OPF time (k * DT_S); the physics simply
        #    advances at its own stable micro-step rate.
        for k in range(N_INTERVALS):
            t_opf  = float(k) * DT_S          # OPF economic time [s]
            t_phys = float(k) * DC_CONFIG['dt_micro']   # physics clock [s]
            lmp    = self.lmp_profile[k]

            # One quasi-static micro-step → current P/Q, motor speed, VSC state
            state: CanonicalOutput = physics.step(CanonicalInput(
                V_pu          = 0.92,          # placeholder; updated after NR
                freq_hz       = 60.0,
                t_sim         = t_phys,
                dt            = DC_CONFIG['dt_micro'],
                price_per_mwh = lmp,
            ))

            P_bid = state.P_mw
            Q_bid = state.Q_mvar

            # Solve OPF interval
            result = self.opf.solve_interval(
                t            = t_opf,
                P_base_mw    = self.base_load[k] * load_scale[k],
                P_dc_bid     = P_bid,
                P_dc_min     = state.P_committed_mw,
                P_dc_max     = P_bid + state.P_flex_mw * 0.10,
                Q_dc_bid     = Q_bid,
                lmp_ref      = lmp,
                dr_enabled   = cfg['dr'],
                volt_support = cfg['volt'],
            )

            # Update state with actual V_pcc from NR power flow
            state2: CanonicalOutput = physics.step(CanonicalInput(
                V_pu          = result['V_pcc'],
                freq_hz       = result['freq_hz'],
                t_sim         = t_phys + DC_CONFIG['dt_micro'],
                dt            = DC_CONFIG['dt_micro'],
                price_per_mwh = result['lmp_pcc'],
            ))

            # Record
            rec.P_dc[k]       = result['P_dc_dispatch']
            rec.Q_dc[k]       = result['Q_dc_adjusted']
            rec.P_dc_bid[k]   = result['P_dc_bid']
            rec.dr_mw[k]      = result['dr_mw']
            rec.P_server[k]   = state2.P_server_kw
            rec.P_cool[k]     = state2.P_cool_kw
            rec.omega_r[k]    = state2.omega_r_pu
            rec.slip[k]       = state2.slip
            rec.V_pcc[k]      = result['V_pcc']
            rec.V_632[k]      = result['V_632']
            rec.V_680[k]      = result['V_680']
            rec.freq_hz[k]    = result['freq_hz']
            rec.P_losses[k]   = result['P_losses']
            rec.P_gen_dg[k]   = result['P_gen_dg']
            rec.lmp_ref[k]    = result['lmp_ref']
            rec.lmp_pcc[k]    = result['lmp_pcc']
            rec.total_cost[k] = result['total_cost_usd']
            rec.dr_value[k]   = result['dr_value_usd']
            rec.loss_cost[k]  = result['loss_cost_usd']
            rec.loss_sens[k]  = result['loss_sensitivity']
            rec.v_viol[k]     = result['v_violation']
            rec.dr_active[k]  = result['lmp_used_dr']

            if k % 24 == 0:
                print(f'    h={k*DT_S/3600:5.1f}  '
                      f'V_pcc={result["V_pcc"]:.4f} pu  '
                      f'LMP_pcc={result["lmp_pcc"]:6.1f} $/MWh  '
                      f'P_DC={result["P_dc_dispatch"]:.3f} MW  '
                      f'DR={result["dr_mw"]*1000:5.0f} kW  '
                      f'ω_r={state2.omega_r_pu:.4f}')

        rec.total_cost_day   = float(rec.total_cost.sum())
        rec.total_energy_mwh = float((rec.P_dc * dt_h).sum())
        rec.total_dr_mwh     = float((rec.dr_mw * dt_h).sum())
        rec.total_dr_value   = float(rec.dr_value.sum())
        rec.n_viol           = int(rec.v_viol.sum())
        self.records[sc_name] = rec

        elapsed = time.time() - t_wall
        print(f'\n    ✓  {elapsed:.1f}s  |  '
              f'Cost=${rec.total_cost_day:,.0f}  |  '
              f'DC energy={rec.total_energy_mwh:.2f} MWh  |  '
              f'DR savings=${rec.total_dr_value:.0f}  |  '
              f'V violations={rec.n_viol}')
        deregister(sc_dc_name)

    # ─────────────────────────────────────────────────────────────────────────
    def analyse(self):
        print(f'\n{"─"*74}\n  POST-SIMULATION ANALYSIS\n{"─"*74}')
        baseline = list(self.records.values())[0]

        print('\n  DAILY ECONOMICS')
        print(f'  {"Scenario":<22} {"Cost/Day":>10} {"DC Energy":>11} '
              f'{"DR MWh":>9} {"DR Value":>10} {"V viol":>7}')
        print('  ' + '─' * 72)
        for sc, rec in self.records.items():
            sav = baseline.total_cost_day - rec.total_cost_day
            tag = f'(−${sav:,.0f})' if sav > 0 else '(baseline)'
            print(f'  {sc:<22} {rec.total_cost_day:>10,.0f} '
                  f'{rec.total_energy_mwh:>11.3f} '
                  f'{rec.total_dr_mwh:>9.4f} '
                  f'{rec.total_dr_value:>10.0f} '
                  f'{rec.n_viol:>7}   {tag}')

        print('\n  VOLTAGE STATISTICS (bus 634 PCC)')
        for sc, rec in self.records.items():
            ok = 100 * np.mean((rec.V_pcc >= V_MIN) & (rec.V_pcc <= V_MAX))
            print(f'  {sc:<22}  mean={rec.V_pcc.mean():.4f}  '
                  f'min={rec.V_pcc.min():.4f}  '
                  f'max={rec.V_pcc.max():.4f}  '
                  f'ANSI={ok:.1f}%')

        print('\n  MOTOR OPERATING POINT')
        for sc, rec in self.records.items():
            print(f'  {sc:<22}  ω_r mean={rec.omega_r.mean():.4f}  '
                  f'slip mean={rec.slip.mean()*100:.2f}%')

        print('\n  FEEDER LOSSES')
        for sc, rec in self.records.items():
            frac = 100 * rec.P_losses.mean() / max(
                (rec.P_dc + self.base_load).mean(), 0.01)
            print(f'  {sc:<22}  mean={rec.P_losses.mean():.4f} MW  ({frac:.2f}%)')

    # ─────────────────────────────────────────────────────────────────────────
    def save_csv(self):
        rows = []
        for sc, rec in self.records.items():
            for k in range(N_INTERVALS):
                rows.append({
                    'scenario':       sc,
                    'hour':           INTERVAL_HOURS[k],
                    'interval':       k,
                    'lmp_ref':        rec.lmp_ref[k],
                    'lmp_pcc':        rec.lmp_pcc[k],
                    'P_dc_bid_mw':    rec.P_dc_bid[k],
                    'P_dc_mw':        rec.P_dc[k],
                    'Q_dc_mvar':      rec.Q_dc[k],
                    'dr_mw':          rec.dr_mw[k],
                    'P_server_kw':    rec.P_server[k],
                    'P_cool_kw':      rec.P_cool[k],
                    'omega_r_pu':     rec.omega_r[k],
                    'slip':           rec.slip[k],
                    'V_pcc_pu':       rec.V_pcc[k],
                    'V_632_pu':       rec.V_632[k],
                    'V_680_pu':       rec.V_680[k],
                    'freq_hz':        rec.freq_hz[k],
                    'P_losses_mw':    rec.P_losses[k],
                    'P_gen_dg_mw':    rec.P_gen_dg[k],
                    'total_cost_usd': rec.total_cost[k],
                    'dr_value_usd':   rec.dr_value[k],
                    'loss_sens':      rec.loss_sens[k],
                    'v_violation':    int(rec.v_viol[k]),
                    'dr_active':      int(rec.dr_active[k]),
                })
        import pandas as pd
        pd.DataFrame(rows).to_csv(OUT_CSV, index=False, float_format='%.6f')
        print(f'  CSV   →  {OUT_CSV}')

    # ─────────────────────────────────────────────────────────────────────────
    def plot(self):
        h = INTERVAL_HOURS
        SC_COL = {
            'S1_Baseline':      '#1a5276',
            'S2_PriceResponse': '#1e8449',
            'S3_VoltSupport':   '#7d6608',
            'S4_Congestion':    '#922b21',
        }
        SC_LBL = {
            'S1_Baseline':      'S1  Baseline',
            'S2_PriceResponse': 'S2  Price-Responsive DR',
            'S3_VoltSupport':   'S3  DR + Volt-VAR',
            'S4_Congestion':    'S4  DR + Congestion',
        }
        LW, AL = 1.3, 0.85

        fig = plt.figure(figsize=(20, 28))
        fig.patch.set_facecolor('white')
        gs  = gridspec.GridSpec(12, 1, figure=fig, hspace=0.55,
                                top=0.96, bottom=0.03, left=0.08, right=0.96)
        axs = [fig.add_subplot(gs[i]) for i in range(12)]

        def style(ax):
            ax.set_facecolor('#f8f9fa')
            ax.tick_params(colors='#2c3e50', labelsize=8)
            for sp in ax.spines.values():
                sp.set_edgecolor('#ced4da'); sp.set_linewidth(0.6)
            ax.grid(axis='y', color='#dee2e6', lw=0.5, ls=':')
            ax.set_xlim(0, 24); ax.set_xticks(range(0, 25, 4))

        def peak(ax):
            ax.axvspan( 7, 10, color='#fff3cd', alpha=0.55, zorder=0)
            ax.axvspan(17, 21, color='#fff3cd', alpha=0.55, zorder=0)

        def vlimits(ax):
            ax.axhline(V_MIN, color='#e74c3c', lw=0.8, ls='--', alpha=0.7)
            ax.axhline(V_MAX, color='#e74c3c', lw=0.8, ls='--', alpha=0.7)
            ax.axhline(1.0,   color='#adb5bd', lw=0.4, ls=':',  alpha=0.5)

        for ax in axs: style(ax)

        # ── 0: LMP ───────────────────────────────────────────────────────────
        r0 = list(self.records.values())[0]
        axs[0].fill_between(h, r0.lmp_ref, alpha=0.18, color='#1a5276')
        axs[0].plot(h, r0.lmp_ref, color='#1a5276', lw=1.5, label='System LMP (ref)')
        for sc, rec in self.records.items():
            if sc != 'S1_Baseline':
                axs[0].plot(h, rec.lmp_pcc, color=SC_COL[sc], lw=LW, ls='--',
                            alpha=AL, label=f'PCC LMP — {SC_LBL[sc]}')
        axs[0].axhline(50, color='#e74c3c', lw=0.8, ls=':', alpha=0.7,
                       label='DR threshold (50 $/MWh)')
        peak(axs[0])
        axs[0].set_ylabel('LMP ($/MWh)', fontsize=8.5)
        axs[0].set_title('Locational Marginal Prices — System vs. Bus 634 PCC',
                         fontsize=9.5, loc='left', color='#1a2744', pad=3)
        axs[0].legend(fontsize=7.5, ncol=2, facecolor='white', edgecolor='#ced4da')

        # ── 1: DC active power ────────────────────────────────────────────────
        axs[1].plot(h, r0.P_dc_bid * 1000, color='#adb5bd', lw=0.8, ls=':',
                    alpha=0.7, label='Unconstrained bid')
        for sc, rec in self.records.items():
            axs[1].plot(h, rec.P_dc * 1000, color=SC_COL[sc],
                        lw=LW, alpha=AL, label=SC_LBL[sc])
        peak(axs[1])
        axs[1].set_ylabel('P_DC (kW)', fontsize=8.5)
        axs[1].set_title('Datacenter Active Power Dispatch', fontsize=9.5,
                         loc='left', color='#1a2744', pad=3)
        axs[1].legend(fontsize=7.5, ncol=2, facecolor='white', edgecolor='#ced4da')

        # ── 2: DR curtailment ─────────────────────────────────────────────────
        any_dr = False
        for sc, rec in self.records.items():
            if rec.dr_mw.max() > 1e-4:
                axs[2].fill_between(h, rec.dr_mw * 1000, alpha=0.22,
                                    color=SC_COL[sc])
                axs[2].plot(h, rec.dr_mw * 1000, color=SC_COL[sc],
                            lw=LW, alpha=AL, label=SC_LBL[sc])
                any_dr = True
        if not any_dr:
            axs[2].text(12, 0.5, 'No DR activated in any scenario',
                        ha='center', va='center', fontsize=9, color='#adb5bd')
        peak(axs[2])
        axs[2].set_ylabel('DR Curtailment (kW)', fontsize=8.5)
        axs[2].set_title('Demand Response Curtailment', fontsize=9.5,
                         loc='left', color='#1a2744', pad=3)
        if any_dr:
            axs[2].legend(fontsize=7.5, facecolor='white', edgecolor='#ced4da')

        # ── 3: DC reactive power ──────────────────────────────────────────────
        for sc, rec in self.records.items():
            axs[3].plot(h, rec.Q_dc * 1000, color=SC_COL[sc],
                        lw=LW, alpha=AL, label=SC_LBL[sc])
        axs[3].axhline(0, color='#adb5bd', lw=0.4, ls=':', alpha=0.5)
        peak(axs[3])
        axs[3].set_ylabel('Q_DC (kVAR)', fontsize=8.5)
        axs[3].set_title('Datacenter Reactive Power — Volt-VAR Droop Effect',
                         fontsize=9.5, loc='left', color='#1a2744', pad=3)
        axs[3].legend(fontsize=7.5, ncol=2, facecolor='white', edgecolor='#ced4da')

        # ── 4: IT / cooling breakdown ─────────────────────────────────────────
        ref = self.records.get('S2_PriceResponse',
                               list(self.records.values())[0])
        axs[4].fill_between(h, ref.P_server, alpha=0.30, color='#0969da',
                            label='IT servers (kW)')
        axs[4].fill_between(h, ref.P_server + ref.P_cool, ref.P_server,
                            alpha=0.30, color='#cf222e', label='HVAC cooling (kW)')
        axs[4].plot(h, ref.P_server, color='#0969da', lw=1.2)
        axs[4].plot(h, ref.P_server + ref.P_cool, color='#cf222e', lw=1.2)
        peak(axs[4])
        axs[4].set_ylabel('Power (kW)', fontsize=8.5)
        axs[4].set_title('Datacenter Internal Load — IT vs. HVAC (S2)',
                         fontsize=9.5, loc='left', color='#1a2744', pad=3)
        axs[4].legend(fontsize=7.5, facecolor='white', edgecolor='#ced4da')

        # ── 5: Motor speed ────────────────────────────────────────────────────
        for sc, rec in self.records.items():
            axs[5].plot(h, rec.omega_r, color=SC_COL[sc],
                        lw=LW, alpha=AL, label=SC_LBL[sc])
        axs[5].axhline(1.0, color='#adb5bd', lw=0.5, ls=':', alpha=0.5)
        ax5r = axs[5].twinx()
        ax5r.fill_between(h, r0.slip * 100, alpha=0.15, color='#9a6700')
        ax5r.plot(h, r0.slip * 100, color='#9a6700', lw=0.8, alpha=0.6)
        ax5r.set_ylabel('Slip (%)', fontsize=7.5, color='#9a6700')
        ax5r.tick_params(colors='#9a6700', labelsize=7)
        axs[5].set_ylabel('ω_r (pu)', fontsize=8.5)
        axs[5].set_title('HVAC Motor Speed (pu) — All Scenarios + Slip Overlay (S1)',
                         fontsize=9.5, loc='left', color='#1a2744', pad=3)
        axs[5].legend(fontsize=7.5, ncol=2, facecolor='white', edgecolor='#ced4da',
                      loc='lower right')

        # ── 6: Bus 634 voltage ────────────────────────────────────────────────
        vlimits(axs[6])
        for sc, rec in self.records.items():
            axs[6].plot(h, rec.V_pcc, color=SC_COL[sc],
                        lw=LW, alpha=AL, label=SC_LBL[sc])
            if rec.v_viol.any():
                axs[6].scatter(h[rec.v_viol], rec.V_pcc[rec.v_viol],
                               color=SC_COL[sc], marker='x', s=40, zorder=5)
        peak(axs[6])
        axs[6].set_ylabel('|V| (pu)', fontsize=8.5)
        axs[6].set_title('Bus 634 (DC PCC) Voltage — ANSI C84.1 Range A',
                         fontsize=9.5, loc='left', color='#1a2744', pad=3)
        axs[6].legend(fontsize=7.5, ncol=2, facecolor='white', edgecolor='#ced4da')

        # ── 7: Multi-bus voltage (S1 vs S3) ──────────────────────────────────
        vlimits(axs[7])
        s1 = self.records.get('S1_Baseline', list(self.records.values())[0])
        s3 = self.records.get('S3_VoltSupport', list(self.records.values())[-1])
        for data, ls, lbl_suf in [(s1, '--', 'S1'), (s3, '-', 'S3')]:
            axs[7].plot(h, data.V_pcc, color='#1a5276', lw=LW, ls=ls,
                        alpha=AL, label=f'Bus 634  {lbl_suf}')
            axs[7].plot(h, data.V_632, color='#1e8449', lw=LW, ls=ls,
                        alpha=AL, label=f'Bus 632  {lbl_suf}')
        axs[7].plot(h, s3.V_680, color='#7d6608', lw=LW, alpha=AL,
                    label='Bus 680  S3')
        peak(axs[7])
        axs[7].set_ylabel('|V| (pu)', fontsize=8.5)
        axs[7].set_title('Multi-Bus Voltage — S1 (baseline) vs. S3 (volt-VAR support)',
                         fontsize=9.5, loc='left', color='#1a2744', pad=3)
        axs[7].legend(fontsize=7.5, ncol=3, facecolor='white', edgecolor='#ced4da')

        # ── 8: DG output ──────────────────────────────────────────────────────
        axs[8].fill_between(h, self.wind_mw, alpha=0.25, color='#0969da',
                            label='DG1 Wind (MW)')
        solar_total = (0.25 + 0.15) * self.solar_pu
        axs[8].fill_between(h, solar_total, alpha=0.25, color='#e3b341',
                            label='DG2+3 Solar (MW)')
        axs[8].plot(h, self.wind_mw, color='#0969da', lw=1.2)
        axs[8].plot(h, solar_total, color='#e3b341', lw=1.2)
        dg_total = self.wind_mw + solar_total
        axs[8].fill_between(h, dg_total, alpha=0.10, color='#2da44e',
                            label='Total DG (MW)')
        axs[8].plot(h, dg_total, color='#2da44e', lw=1.0, ls='--')
        peak(axs[8])
        axs[8].set_ylabel('P_gen (MW)', fontsize=8.5)
        axs[8].set_title('Distributed Generation — Wind + Solar Forecast',
                         fontsize=9.5, loc='left', color='#1a2744', pad=3)
        axs[8].legend(fontsize=7.5, ncol=3, facecolor='white', edgecolor='#ced4da')

        # ── 9: Feeder losses ──────────────────────────────────────────────────
        for sc, rec in self.records.items():
            axs[9].plot(h, rec.P_losses * 1000, color=SC_COL[sc],
                        lw=LW, alpha=AL, label=SC_LBL[sc])
        peak(axs[9])
        axs[9].set_ylabel('P_loss (kW)', fontsize=8.5)
        axs[9].set_title('Feeder I²R Losses by Scenario',
                         fontsize=9.5, loc='left', color='#1a2744', pad=3)
        axs[9].legend(fontsize=7.5, ncol=2, facecolor='white', edgecolor='#ced4da')

        # ── 10: Cumulative DR value ───────────────────────────────────────────
        for sc, rec in self.records.items():
            axs[10].plot(h, np.cumsum(rec.dr_value), color=SC_COL[sc],
                         lw=LW, alpha=AL,
                         label=f'{SC_LBL[sc]} (${rec.total_dr_value:.0f})')
        peak(axs[10])
        axs[10].set_ylabel('Cum. DR Value ($)', fontsize=8.5)
        axs[10].set_title('Cumulative Demand Response Value — 24-Hour Horizon',
                          fontsize=9.5, loc='left', color='#1a2744', pad=3)
        axs[10].legend(fontsize=7.5, ncol=2, facecolor='white', edgecolor='#ced4da')

        # ── 11: Cost bar chart ────────────────────────────────────────────────
        sc_names = list(self.records.keys())
        costs    = [self.records[s].total_cost_day for s in sc_names]
        dr_vals  = [self.records[s].total_dr_value for s in sc_names]
        x        = np.arange(len(sc_names))
        bars = axs[11].bar(x, costs,
                           color=[SC_COL[s] for s in sc_names],
                           alpha=0.75, width=0.55,
                           edgecolor='white', linewidth=1.5, zorder=3)
        for bar, cost, dr in zip(bars, costs, dr_vals):
            axs[11].text(bar.get_x() + bar.get_width() / 2,
                         bar.get_height() + max(costs) * 0.01,
                         f'${cost:,.0f}\n(DR=${dr:.0f})',
                         ha='center', va='bottom', fontsize=8,
                         color='#2c3e50', fontweight='bold')
        base = costs[0]
        for i, (c, s) in enumerate(zip(costs[1:], sc_names[1:]), 1):
            sav = base - c
            if sav > 0:
                axs[11].annotate(
                    f'−${sav:,.0f}',
                    xy=(i, c),
                    xytext=(i + 0.30, c + max(costs) * 0.05),
                    fontsize=7.5, color='#1e8449', fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color='#1e8449', lw=0.8))
        axs[11].set_xticks(x)
        axs[11].set_xticklabels([SC_LBL[s] for s in sc_names], fontsize=8)
        axs[11].set_ylabel('Daily Operating Cost ($)', fontsize=8.5)
        axs[11].set_title('Daily System Operating Cost — Scenario Comparison',
                          fontsize=9.5, loc='left', color='#1a2744', pad=3)
        axs[11].set_xlim(-0.5, len(sc_names) - 0.5)
        axs[11].grid(axis='y', color='#dee2e6', lw=0.5, ls=':')

        for ax in axs[:-2]: ax.tick_params(labelbottom=False)
        axs[-2].set_xlabel('Hour of Day', fontsize=9, color='#2c3e50')

        peak_patch = mpatches.Patch(color='#fff3cd', alpha=0.9,
                                    label='Peak demand periods')
        fig.legend(handles=[peak_patch], loc='upper right', fontsize=7.5,
                   facecolor='white', edgecolor='#ced4da',
                   bbox_to_anchor=(0.96, 0.962))

        fig.suptitle(
            'Optimal Power Flow Study — IEEE 13-Bus Distribution Feeder  ·  '
            'Price-Responsive Datacenter Load\n'
            'DC-OPF + AC Verification  ·  4 Scenarios  ·  '
            '24-Hour Horizon  ·  5-Minute Intervals',
            fontsize=11, color='#1a2744', y=0.968, fontweight='semibold')

        fig.savefig(OUT_PLOT, dpi=150, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.close()
        print(f'  Plot  →  {OUT_PLOT}')

    # ─────────────────────────────────────────────────────────────────────────
    def report(self):
        sep = '=' * 74
        lines = [
            sep,
            '  OPTIMAL POWER FLOW STUDY — FINAL REPORT',
            sep, '',
            '  CONFIGURATION',
            f'    Feeder      :  IEEE 13-bus, {BASE_KV} kV, {BASE_MVA} MVA',
            f'    Datacenter  :  {DC_NAME}  (2 MVA, bus 634, 0.48 kV)',
            f'    Horizon     :  {HOURS} h  ·  {N_INTERVALS} intervals  ·  {DT_MIN}-min',
            f'    OPF method  :  DC merit-order + AC (NR) verification',
            f'    Volt limits :  [{V_MIN}, {V_MAX}] pu  (ANSI C84.1 Range A)',
            f'    DR threshold:  {DC_CONFIG["price_threshold"]:.0f} $/MWh',
            '',
        ]
        base_cost = list(self.records.values())[0].total_cost_day
        desc = {
            'S1_Baseline':      'No demand response. DC at unconstrained operating point.',
            'S2_PriceResponse': 'DR enabled. DC curtails load when LMP > 50 $/MWh.',
            'S3_VoltSupport':   'DR + volt-VAR. VSC injects reactive when V < 1.0 pu.',
            'S4_Congestion':    'DR + volt-VAR + 40% load increase at morning/evening peaks.',
        }
        for sc, rec in self.records.items():
            sav = base_cost - rec.total_cost_day
            lines += [
                f'  {sc}',
                f'    {textwrap.fill(desc[sc], 68, subsequent_indent="    ")}',
                f'    Cost: ${rec.total_cost_day:,.0f}  |  '
                f'Saving: ${sav:,.0f} ({100*sav/base_cost if base_cost else 0:.1f}%)',
                f'    DC energy: {rec.total_energy_mwh:.3f} MWh  |  '
                f'DR curtailed: {rec.total_dr_mwh:.4f} MWh  |  '
                f'DR value: ${rec.total_dr_value:.0f}',
                f'    V_pcc mean={rec.V_pcc.mean():.4f}  '
                f'min={rec.V_pcc.min():.4f}  '
                f'max={rec.V_pcc.max():.4f}  |  '
                f'Violations: {rec.n_viol} intervals',
                '',
            ]
        lines += [
            '  OUTPUT FILES',
            f'    {OUT_PLOT}',
            f'    {OUT_CSV}',
            f'    {OUT_RPT}',
            '', sep,
        ]
        text = '\n'.join(lines)
        print('\n' + text)
        with open(OUT_RPT, 'w') as f:
            f.write(text + '\n')
        print(f'  Report →  {OUT_RPT}')

    # ─────────────────────────────────────────────────────────────────────────
    def _banner(self):
        lines = [
            '',
            '┌' + '─' * 72 + '┐',
            '│  OPTIMAL POWER FLOW STUDY' + ' ' * 46 + '│',
            '│  IEEE 13-Bus  +  Price-Responsive Datacenter Load (OASIS)' + ' ' * 13 + '│',
            '│  DC-OPF Merit Order  ·  AC Verification  ·  4 Scenarios  ·  24 h' + ' ' * 5 + '│',
            '│' + ' ' * 72 + '│',
            '│  Module chain:' + ' ' * 57 + '│',
            '│    OPFStudy → registry → OPFAdapter → DatacenterPhysics' + ' ' * 16 + '│',
            '│             → NetworkOPF → OpenDSSNetworkSimulator (NR)' + ' ' * 15 + '│',
            '│' + ' ' * 72 + '│',
            '└' + '─' * 72 + '┘',
        ]
        print('\n'.join(lines))


# =============================================================================
#  ENTRY POINT
# =============================================================================
if __name__ == '__main__':
    t_total = time.time()
    study = OPFStudy()
    study.run_all_scenarios()
    study.analyse()
    print('\nSaving Outputs …')
    study.save_csv()
    study.plot()
    study.report()
    print(f'\nTotal wall time: {time.time()-t_total:.1f}s')
    print(f'All Outputs → {OUT_DIR}')
    sys.exit(0)