"""
OPF formulation (DC-OPF merit-order + AC verification)
-------------------------------------------------------
  min  Σ_g C_g(P_g) + C_loss·P_loss_pu − C_DR·ΔP_DC

  s.t. Power balance (AC NR per interval)
       0.95 ≤ |V_i| ≤ 1.05 pu   (ANSI C84.1 Range A)
       Generator limits P_g ∈ [P_min, P_max]
       Interface thermal limit (Bus 16 area import)
       DC flexibility P_DC ∈ [P_committed, P_max]

Physics model
-------------
Datacenter stepped at dt_micro=0.02 s, nsub=1 per OPF interval (same as
distribution OPF).  500-step warm-up drives motor and VSC to steady state.
TransmissionAdapter converts P/Q to Norton (I_d, I_q) on 100 MVA base.

Run
---
    python transmission_opf_study.py
"""

import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

from datacenter_registry import register, get_datacenter, deregister
from testsystems.transmission_network import TransmissionNetworkSimulator, TransNetworkResults
from adapters import TransmissionAdapter
from datacenter_core import CanonicalInput, CanonicalOutput

# =============================================================================
#  STUDY CONFIGURATION
# =============================================================================
HOURS          = 24
DT_MIN         = 5
DT_S           = DT_MIN * 60        # 300 s per interval
N_INTERVALS    = HOURS * 60 // DT_MIN   # 288
INTERVAL_HOURS = np.arange(N_INTERVALS, dtype=float) * DT_S / 3600.0

S_BASE_MVA     = 100.0   # transmission system base
V_MIN          = 0.95    # ANSI C84.1 Range A lower bound
V_MAX          = 1.05    # ANSI C84.1 Range A upper bound
DC_PCC_BUS     = 16      # datacenter PCC

# DR / loss pricing
C_DR           = 60.0    # $/MWh value of DR curtailment (higher than distribution
                         # — transmission-level DR has greater market value)
C_LOSS         = 40.0    # $/MWh value of loss reduction
DR_THRESHOLD   = 65.0    # $/MWh LMP threshold to activate DR (zonal price signal)
DR_MAX_FRAC    = 0.25    # max DR curtailment as fraction of bid

# Thermal interface limit at bus 16 area import
INTERFACE_LIMIT_MW = 1200.0   # MW; aggregate import limit into the bus 16 load area

# Physics warm-up
WARMUP_STEPS   = 500

DC_NAME = "DC_TRANS_OPF_B16"
DC_CONFIG = {
    "seed":            42,
    "n_cooling_units":  3,
    "dt_micro":         0.02,
    "dt":               0.02,    # nsub=1
    "price_threshold":  DR_THRESHOLD,
    "price_max":        300.0,
    "max_curtail_pu":   DR_MAX_FRAC,
}

# ── Output paths ──────────────────────────────────────────────────────────────
OUT_DIR   = '../Outputs'
OUT_PLOT  = os.path.join(OUT_DIR, 'transmission_opf_results.png') #OUT_DIR / "distribution_dynamics_results.png"
OUT_CSV   = os.path.join(OUT_DIR, 'transmission_opf_timeseries.csv')
OUT_RPT   = os.path.join(OUT_DIR, 'transmission_opf_report.txt')

# =============================================================================
#  GENERATOR SPECIFICATIONS  (100 MVA system base)
# =============================================================================
# Format: bus, fuel type, P_min [MW], P_max [MW], a [$/MW²h], b [$/MWh], c [$]
GEN_SPECS = {
    "G01_nuclear_b30":  {"bus": 30,  "fuel": "Nuclear",  "P_min":  50., "P_max": 250., "a": 0.10, "b": 18., "c": 800},
    "G02_gas_b31":      {"bus": 31,  "fuel": "Gas CC",   "P_min":  50., "P_max": 573., "a": 0.30, "b": 35., "c": 500},
    "G03_coal_b32":     {"bus": 32,  "fuel": "Coal",     "P_min": 100., "P_max": 650., "a": 0.15, "b": 22., "c": 900},
    "G04_coal_b33":     {"bus": 33,  "fuel": "Coal",     "P_min": 100., "P_max": 632., "a": 0.14, "b": 21., "c": 920},
    "G05_ccgt_b34":     {"bus": 34,  "fuel": "Gas CC",   "P_min":  50., "P_max": 508., "a": 0.25, "b": 28., "c": 600},
    "G06_nuclear_b35":  {"bus": 35,  "fuel": "Nuclear",  "P_min":  50., "P_max": 650., "a": 0.08, "b": 16., "c": 850},
    "G07_hydro_b36":    {"bus": 36,  "fuel": "Hydro",    "P_min":   0., "P_max": 560., "a": 0.05, "b": 12., "c": 100},
    "G08_ccgt_b37":     {"bus": 37,  "fuel": "Gas CC",   "P_min":  50., "P_max": 540., "a": 0.28, "b": 30., "c": 550},
    "G09_oil_b38":      {"bus": 38,  "fuel": "Oil GT",   "P_min":   0., "P_max": 830., "a": 0.80, "b": 55., "c": 300},
    "G10_nuclear_b39":  {"bus": 39,  "fuel": "Nuclear",  "P_min": 200., "P_max":1000., "a": 0.09, "b": 17., "c":1200},
}

# Installed capacity in MW
TOTAL_CAPACITY_MW = sum(s["P_max"] for s in GEN_SPECS.values())

# =============================================================================
#  FORECAST PROFILES
# =============================================================================
def build_lmp_profile() -> np.ndarray:
    """
    Zonal LMP forecast [$/MWh] — New England-style profile.
    Off-peak valley ~30, morning peak ~85, midday solar dip ~55,
    evening peak ~100, overnight valley ~25.
    """
    h   = INTERVAL_HOURS
    rng = np.random.default_rng(17)
    lmp = (32.0
           + 53.0 * np.exp(-0.5 * ((h -  8.5) / 1.2) ** 2)
           + 68.0 * np.exp(-0.5 * ((h - 18.5) / 1.0) ** 2)
           - 22.0 * np.exp(-0.5 * ((h - 12.5) / 2.0) ** 2)
           +  8.0 * rng.normal(0, 1, N_INTERVALS))
    return np.clip(lmp, 18.0, 220.0)


def build_system_load_profile() -> np.ndarray:
    """
    System total load [pu on 100 MVA base].
    Models a typical New England summer weekday demand profile.
    Peak ≈ 62 pu (6200 MW), overnight trough ≈ 44 pu (4400 MW).
    """
    h = INTERVAL_HOURS
    load = (50.0
            +  8.0 * np.exp(-0.5 * ((h -  9.0) / 2.0) ** 2)
            + 12.0 * np.exp(-0.5 * ((h - 19.0) / 1.5) ** 2)
            -  3.0 * np.exp(-0.5 * ((h - 13.0) / 2.5) ** 2))
    return np.clip(load, 42.0, 66.0)   # pu on 100 MVA


def build_renewable_profile() -> Tuple[np.ndarray, np.ndarray]:
    """
    Wind [MW] and offshore solar [MW] generation profiles.
    New England offshore wind: high at night/morning; solar peaks midday.
    """
    h   = INTERVAL_HOURS
    rng = np.random.default_rng(23)

    wind_mw = np.clip(
        280.0 + 120.0 * np.cos(2 * np.pi * (h - 3) / 24)
        + rng.normal(0, 15, N_INTERVALS),
        30.0, 450.0
    )

    t_rise, t_set = 5.5, 20.0
    solar_mw = np.zeros(N_INTERVALS)
    for i, hh in enumerate(h):
        if t_rise < hh < t_set:
            clear  = min((hh - t_rise) / 2.5, (t_set - hh) / 2.5, 1.0)
            cloud  = rng.uniform(0.80, 1.0) if rng.random() > 0.12 else rng.uniform(0.35, 0.65)
            solar_mw[i] = clear * cloud * 350.0   # 350 MW installed solar

    return wind_mw, solar_mw


# =============================================================================
#  TRANSMISSION-LEVEL OPF SOLVER
# =============================================================================
class TransmissionOPF:
    """
    Network-constrained OPF for the IEEE 39-bus system.

    Algorithm per 5-minute interval
    ────────────────────────────────
    1. Economic merit-order dispatch subject to generator P_min/P_max.
       Renewable (wind + solar) dispatched first at zero marginal cost.
    2. Thermal interface check: if area import > INTERFACE_LIMIT_MW,
       re-dispatch by backing down units outside the constrained area
       and activating DR at bus 16.
    3. DR signal: if zonal LMP > DR_THRESHOLD, compute curtailed DC load
       using a price-elastic demand function.
    4. AC power flow verification via NR; volt-VAR injection if V < 1.0 pu.
    5. Loss sensitivity: ∂P_loss/∂P_DC by 1% perturbation.
    6. Nodal LMP at bus 16 = zonal LMP × (1 + loss_sensitivity)
       + congestion_component.

    In S5 (generator trip): G06 is removed from the merit order and its
    capacity is re-dispatched to remaining units; DR provides emergency
    load reduction at the marginal price.
    """

    def __init__(self, net: TransmissionNetworkSimulator):
        self.net           = net
        self._tripped_gens = set()   # names of tripped generators

    def trip_generator(self, gen_name: str):
        """Remove a generator from dispatch (N-1 contingency)."""
        self._tripped_gens.add(gen_name)
        spec = GEN_SPECS.get(gen_name)
        if spec:
            self.net.trip_generator(spec["bus"])

    def restore_generator(self, gen_name: str):
        """Restore a previously tripped generator."""
        self._tripped_gens.discard(gen_name)
        spec = GEN_SPECS.get(gen_name)
        if spec:
            b = self.net.buses.get(spec["bus"])
            if b:
                b.type = 'pv'
                b.P_gen_pu = spec["P_max"] * 0.8 / S_BASE_MVA
        self.net._build_ybus()

    def solve_interval(
        self,
        t:             float,
        P_dc_bid_mw:   float,
        P_dc_min_mw:   float,
        P_dc_max_mw:   float,
        Q_dc_bid_mvar: float,
        lmp_zonal:     float,
        total_load_pu: float,
        wind_mw:       float,
        solar_mw:      float,
        dr_enabled:    bool,
        volt_freq_support: bool,
        congestion:    bool,
        gen_trip:      bool,
    ) -> Dict:

        # ── 1. Renewable dispatch (zero MC) ───────────────────────────────────
        P_renewable_pu = (wind_mw + solar_mw) / S_BASE_MVA

        # ── 2. Residual thermal demand ────────────────────────────────────────
        P_thermal_pu = max(0.0, total_load_pu - P_renewable_pu
                          + P_dc_bid_mw / S_BASE_MVA)

        # ── 3. Merit-order thermal dispatch ───────────────────────────────────
        dispatch_mw, gen_cost_rate, lmp_system = self._merit_order(
            P_thermal_pu * S_BASE_MVA, lmp_zonal, gen_trip)

        # ── 4. Interface thermal limit check ──────────────────────────────────
        # Bus 16 load area import = sum of P flowing into buses 15-18 area
        area_import_mw = (P_dc_bid_mw +
                          329.0 + 320.0 + 158.0)   # bus 16 + 15 + 18 base loads [MW]
        congestion_comp = 0.0
        if area_import_mw > INTERFACE_LIMIT_MW:
            congestion_comp = lmp_system * 0.15   # 15% congestion adder
            # Activate DR to relieve congestion
            if dr_enabled:
                P_dc_bid_mw = max(P_dc_min_mw,
                                  P_dc_bid_mw * (1 - DR_MAX_FRAC * 0.5))

        # ── 5. DR dispatch ────────────────────────────────────────────────────
        lmp_eff = lmp_system + congestion_comp
        if dr_enabled and lmp_eff > DR_THRESHOLD:
            P_dc_mw = self._dr_dispatch(P_dc_bid_mw, P_dc_min_mw,
                                         P_dc_max_mw, lmp_eff)
        else:
            P_dc_mw = P_dc_bid_mw
        dr_mw = max(0.0, P_dc_bid_mw - P_dc_mw)

        # ── 6. AC power flow ──────────────────────────────────────────────────
        # Update generator dispatch in network model
        self._apply_dispatch(dispatch_mw, wind_mw, solar_mw)

        Q_dc = Q_dc_bid_mvar
        net0: TransNetworkResults = self.net.solve(t, P_dc_mw, Q_dc, DT_S, quasi_static=True)

        # Volt-VAR support
        if volt_freq_support:
            dV   = net0.V_pcc_pu - 1.0
            Q_dc = Q_dc_bid_mvar - 0.10 * dV * S_BASE_MVA
            net0 = self.net.solve(t, P_dc_mw, Q_dc, DT_S, quasi_static=True)

        # ── 7. Loss sensitivity ───────────────────────────────────────────────
        net1: TransNetworkResults = self.net.solve(
            t, P_dc_mw * 1.01 + 0.001, Q_dc, DT_S, quasi_static=True)
        dLoss_pu = ((net1.P_losses_pu - net0.P_losses_pu)
                    / (P_dc_mw / S_BASE_MVA * 0.01 + 1e-6))
        lmp_bus16 = lmp_eff * (1.0 + dLoss_pu) + congestion_comp

        # ── 8. Economics ──────────────────────────────────────────────────────
        dt_h        = DT_S / 3600.0
        dr_value    = dr_mw * dt_h * C_DR
        loss_cost   = net0.P_losses_pu * S_BASE_MVA * dt_h * C_LOSS
        total_cost  = gen_cost_rate * dt_h + loss_cost - dr_value

        # ── 9. Voltage compliance ─────────────────────────────────────────────
        v_viol_pcc = (net0.V_pcc_pu < V_MIN or net0.V_pcc_pu > V_MAX)
        v_viol_sys = (net0.V_min_pu  < V_MIN)

        return {
            "P_dc_mw":          P_dc_mw,
            "P_dc_bid_mw":      P_dc_bid_mw,
            "Q_dc_mvar":        Q_dc,
            "dr_mw":            dr_mw,
            "gen_cost_rate":    gen_cost_rate,
            "lmp_zonal":        lmp_system,
            "lmp_bus16":        lmp_bus16,
            "congestion_comp":  congestion_comp,
            "loss_sens":        dLoss_pu,
            "V_pcc":            net0.V_pcc_pu,
            "V_bus14":          net0.V_bus14_pu,
            "V_bus15":          net0.V_bus15_pu,
            "V_min":            net0.V_min_pu,
            "freq_hz":          net0.freq_hz,
            "rocof_hz_s":       net0.rocof_hz_s,
            "P_losses_mw":      net0.P_losses_pu * S_BASE_MVA,
            "P_gen_total_mw":   net0.P_total_gen_pu * S_BASE_MVA,
            "P_renewable_mw":   wind_mw + solar_mw,
            "dr_value_usd":     dr_value,
            "loss_cost_usd":    loss_cost,
            "total_cost_usd":   total_cost,
            "v_viol_pcc":       v_viol_pcc,
            "v_viol_sys":       v_viol_sys,
            "nr_converged":     net0.nr_converged,
            "dr_active":        (dr_mw > 0.001),
            "area_import_mw":   area_import_mw,
            "dispatch_mw":      dispatch_mw,
        }

    # ─────────────────────────────────────────────────────────────────────────
    def _merit_order(
        self,
        P_demand_mw: float,
        lmp_ref: float,
        gen_trip: bool,
    ) -> Tuple[Dict[str, float], float, float]:
        """Economic dispatch by ascending marginal cost."""
        # Sort by linear cost coefficient b ($/MWh)
        available = {
            n: s for n, s in GEN_SPECS.items()
            if n not in self._tripped_gens
        }
        ordered = sorted(available.items(), key=lambda x: x[1]["b"])

        dispatch: Dict[str, float] = {n: s["P_min"] for n, s in GEN_SPECS.items()}
        P_left  = P_demand_mw - sum(s["P_min"] for s in available.values())
        marginal = ordered[-1][0]

        for name, spec in ordered:
            headroom = spec["P_max"] - spec["P_min"]
            P_add    = float(np.clip(P_left, 0.0, headroom))
            dispatch[name] = spec["P_min"] + P_add
            P_left -= P_add
            if P_left <= 1e-3:
                marginal = name
                break

        # Total generation cost [$/h]
        cost = sum(
            s["a"] * dispatch.get(n, 0.0) ** 2
            + s["b"] * dispatch.get(n, 0.0)
            + s["c"]
            for n, s in GEN_SPECS.items()
        )

        # System LMP ≈ blend of marginal unit and forecast
        sm   = GEN_SPECS[marginal]
        Pm   = dispatch.get(marginal, sm["P_min"])
        lmp  = 0.65 * (2 * sm["a"] * Pm + sm["b"]) + 0.35 * lmp_ref

        return dispatch, cost, lmp

    def _dr_dispatch(
        self,
        P_bid: float,
        P_min: float,
        P_max: float,
        lmp: float,
    ) -> float:
        """Price-elastic DR: linear reduction above DR_THRESHOLD."""
        if lmp <= DR_THRESHOLD:
            return P_bid
        excess  = lmp - DR_THRESHOLD
        DR_COST = C_DR   # $/MWh reservation cost
        frac    = float(np.clip(excess / (200.0 - DR_THRESHOLD), 0.0, 1.0))
        P_opt   = P_bid * (1.0 - DR_MAX_FRAC * frac)
        return float(np.clip(P_opt, P_min, P_max))

    def _apply_dispatch(
        self,
        dispatch_mw: Dict[str, float],
        wind_mw: float,
        solar_mw: float,
    ):
        """Write dispatch set-points back to the network model."""
        # Conventional generators
        for name, P_mw in dispatch_mw.items():
            if name in self._tripped_gens:
                continue
            spec = GEN_SPECS[name]
            bus  = self.net.buses.get(spec["bus"])
            if bus and bus.type in ("pv", "slack"):
                bus.P_gen_pu = P_mw / S_BASE_MVA

        # Renewables: distribute proportionally across PV generator buses
        # that have headroom (simplified: add to bus 30 and bus 31)
        renewable_pu = (wind_mw + solar_mw) / S_BASE_MVA
        self.net.buses[30].P_gen_pu = float(np.clip(
            GEN_SPECS["G01_nuclear_b30"]["P_max"] / S_BASE_MVA,
            0.0, GEN_SPECS["G01_nuclear_b30"]["P_max"] / S_BASE_MVA))


# =============================================================================
#  RESULT RECORD
# =============================================================================
@dataclass
class ScenarioRecord:
    label: str
    # Datacenter
    P_dc:         np.ndarray = field(default_factory=lambda: np.zeros(N_INTERVALS))
    P_dc_bid:     np.ndarray = field(default_factory=lambda: np.zeros(N_INTERVALS))
    Q_dc:         np.ndarray = field(default_factory=lambda: np.zeros(N_INTERVALS))
    dr_mw:        np.ndarray = field(default_factory=lambda: np.zeros(N_INTERVALS))
    P_server:     np.ndarray = field(default_factory=lambda: np.zeros(N_INTERVALS))
    P_cool:       np.ndarray = field(default_factory=lambda: np.zeros(N_INTERVALS))
    omega_r:      np.ndarray = field(default_factory=lambda: np.zeros(N_INTERVALS))
    slip:         np.ndarray = field(default_factory=lambda: np.zeros(N_INTERVALS))
    # Network
    V_pcc:        np.ndarray = field(default_factory=lambda: np.zeros(N_INTERVALS))
    V_bus14:      np.ndarray = field(default_factory=lambda: np.zeros(N_INTERVALS))
    V_bus15:      np.ndarray = field(default_factory=lambda: np.zeros(N_INTERVALS))
    V_min:        np.ndarray = field(default_factory=lambda: np.zeros(N_INTERVALS))
    freq_hz:      np.ndarray = field(default_factory=lambda: np.zeros(N_INTERVALS))
    rocof:        np.ndarray = field(default_factory=lambda: np.zeros(N_INTERVALS))
    P_losses_mw:  np.ndarray = field(default_factory=lambda: np.zeros(N_INTERVALS))
    P_gen_mw:     np.ndarray = field(default_factory=lambda: np.zeros(N_INTERVALS))
    P_renew_mw:   np.ndarray = field(default_factory=lambda: np.zeros(N_INTERVALS))
    area_import:  np.ndarray = field(default_factory=lambda: np.zeros(N_INTERVALS))
    # Economics
    lmp_zonal:    np.ndarray = field(default_factory=lambda: np.zeros(N_INTERVALS))
    lmp_bus16:    np.ndarray = field(default_factory=lambda: np.zeros(N_INTERVALS))
    congestion:   np.ndarray = field(default_factory=lambda: np.zeros(N_INTERVALS))
    loss_sens:    np.ndarray = field(default_factory=lambda: np.zeros(N_INTERVALS))
    total_cost:   np.ndarray = field(default_factory=lambda: np.zeros(N_INTERVALS))
    dr_value:     np.ndarray = field(default_factory=lambda: np.zeros(N_INTERVALS))
    # Status flags
    v_viol_pcc:   np.ndarray = field(default_factory=lambda: np.zeros(N_INTERVALS, bool))
    v_viol_sys:   np.ndarray = field(default_factory=lambda: np.zeros(N_INTERVALS, bool))
    dr_active:    np.ndarray = field(default_factory=lambda: np.zeros(N_INTERVALS, bool))
    # Summaries
    total_cost_day:   float = 0.0
    total_energy_mwh: float = 0.0
    total_dr_mwh:     float = 0.0
    total_dr_value:   float = 0.0
    n_viol_pcc:       int   = 0
    n_viol_sys:       int   = 0


# =============================================================================
#  OPF STUDY
# =============================================================================
class TransmissionOPFStudy:
    """
    Main study class.

    Five scenarios on the IEEE 39-bus New England test system with a
    2 MVA AI datacenter at bus 16.  The datacenter physics are identical
    to those used in the distribution and dynamics studies; only the
    network backend (TransmissionNetworkSimulator) and adapter
    (TransmissionAdapter → OPFAdapter via CanonicalInput) differ.
    """

    def __init__(self):
        self._banner()
        print("\n[1/5] Building forecast profiles …")
        self.lmp_profile            = build_lmp_profile()
        self.load_profile           = build_system_load_profile()
        self.wind_mw, self.solar_mw = build_renewable_profile()

        print("[2/5] Configuring scenarios …")
        self.scenarios = {
            "S1_Baseline":       {"dr": False, "vf": False,
                                  "congestion": False, "gen_trip": False},
            "S2_PriceResponse":  {"dr": True,  "vf": False,
                                  "congestion": False, "gen_trip": False},
            "S3_VoltFreqSupport":{"dr": True,  "vf": True,
                                  "congestion": False, "gen_trip": False},
            "S4_Congestion":     {"dr": True,  "vf": True,
                                  "congestion": True,  "gen_trip": False},
            "S5_GenTrip":        {"dr": True,  "vf": True,
                                  "congestion": True,  "gen_trip": True},
        }
        self.records: Dict[str, ScenarioRecord] = {}

        print("[3/5] Initialising IEEE 39-bus transmission network …")
        self.net = TransmissionNetworkSimulator()

        print("[4/5] Initialising OPF solver …")
        self.opf = TransmissionOPF(self.net)

        print("[5/5] Registering datacenter …")
        register(DC_NAME, DC_CONFIG)
        print()
        print(self.net.summary())

    # ─────────────────────────────────────────────────────────────────────────
    def run_all_scenarios(self):
        print(f"\n{'='*74}")
        print(f"  TRANSMISSION OPF STUDY  ·  "
              f"IEEE 39-Bus  ·  "
              f"{N_INTERVALS} intervals  ·  {DT_MIN}-min resolution")
        print(f"{'='*74}")
        for sc_name, cfg in self.scenarios.items():
            self._run_scenario(sc_name, cfg)

    # ─────────────────────────────────────────────────────────────────────────
    def _run_scenario(self, sc_name: str, cfg: Dict):
        print(f"\n  ── {sc_name} ──")
        flags = [
            "DR=ON"          if cfg["dr"]         else "DR=OFF",
            "Volt/Freq=ON"   if cfg["vf"]         else "Volt/Freq=OFF",
            "Congestion=ON"  if cfg["congestion"] else "",
            "G06 tripped"    if cfg["gen_trip"]   else "",
        ]
        print("    " + "  ·  ".join(f for f in flags if f))

        # Fresh physics instance per scenario
        sc_dc_name = f"{DC_NAME}_{sc_name}"
        register(sc_dc_name, DC_CONFIG)

        # Use OPFAdapter for quasi-static stepping (same as distribution OPF)
        from adapters import OPFAdapter
        dc_opf: OPFAdapter = get_datacenter(sc_dc_name, "opf",
                                            interval_min=DT_MIN)
        physics = dc_opf._p

        # Re-initialise network to clean state for each scenario
        self.net = TransmissionNetworkSimulator()
        self.opf = TransmissionOPF(self.net)

        # Apply generator trip for S5
        if cfg["gen_trip"]:
            self.opf.trip_generator("G06_nuclear_b35")

        # ── Warm-up ───────────────────────────────────────────────────────────
        for w in range(WARMUP_STEPS):
            physics.step(CanonicalInput(
                V_pu=1.0, freq_hz=60.0,
                t_sim=float(w) * DC_CONFIG["dt_micro"],
                dt=DC_CONFIG["dt_micro"]))

        rec    = ScenarioRecord(label=sc_name)
        t_wall = time.time()
        dt_h   = DT_S / 3600.0

        # Congestion load scale (extra 25% at morning and evening peaks)
        load_scale = np.ones(N_INTERVALS)
        if cfg["congestion"]:
            h = INTERVAL_HOURS
            load_scale += (0.25 * np.exp(-0.5 * ((h -  8.5) / 1.5) ** 2)
                         + 0.20 * np.exp(-0.5 * ((h - 18.5) / 1.5) ** 2))

        for k in range(N_INTERVALS):
            t_opf  = float(k) * DT_S
            t_phys = float(k) * DC_CONFIG["dt_micro"]
            lmp    = self.lmp_profile[k]

            # Step physics to get quasi-static operating point
            state: CanonicalOutput = physics.step(CanonicalInput(
                V_pu          = 1.0,          # placeholder; updated after NR
                freq_hz       = 60.0,
                t_sim         = t_phys,
                dt            = DC_CONFIG["dt_micro"],
                price_per_mwh = lmp,
            ))

            # Solve OPF interval
            result = self.opf.solve_interval(
                t              = t_opf,
                P_dc_bid_mw    = state.P_mw,
                P_dc_min_mw    = state.P_committed_mw,
                P_dc_max_mw    = state.P_mw + state.P_flex_mw * 0.10,
                Q_dc_bid_mvar  = state.Q_mvar,
                lmp_zonal      = lmp,
                total_load_pu  = self.load_profile[k] * load_scale[k],
                wind_mw        = self.wind_mw[k],
                solar_mw       = self.solar_mw[k],
                dr_enabled     = cfg["dr"],
                volt_freq_support = cfg["vf"],
                congestion     = cfg["congestion"],
                gen_trip       = cfg["gen_trip"],
            )

            # Second physics step with actual V_pcc from NR
            state2: CanonicalOutput = physics.step(CanonicalInput(
                V_pu          = result["V_pcc"],
                freq_hz       = result["freq_hz"],
                t_sim         = t_phys + DC_CONFIG["dt_micro"],
                dt            = DC_CONFIG["dt_micro"],
                price_per_mwh = result["lmp_bus16"],
            ))

            # Record
            rec.P_dc[k]        = result["P_dc_mw"]
            rec.P_dc_bid[k]    = result["P_dc_bid_mw"]
            rec.Q_dc[k]        = result["Q_dc_mvar"]
            rec.dr_mw[k]       = result["dr_mw"]
            rec.P_server[k]    = state2.P_server_kw
            rec.P_cool[k]      = state2.P_cool_kw
            rec.omega_r[k]     = state2.omega_r_pu
            rec.slip[k]        = state2.slip
            rec.V_pcc[k]       = result["V_pcc"]
            rec.V_bus14[k]     = result["V_bus14"]
            rec.V_bus15[k]     = result["V_bus15"]
            rec.V_min[k]       = result["V_min"]
            rec.freq_hz[k]     = result["freq_hz"]
            rec.rocof[k]       = result["rocof_hz_s"]
            rec.P_losses_mw[k] = result["P_losses_mw"]
            rec.P_gen_mw[k]    = result["P_gen_total_mw"]
            rec.P_renew_mw[k]  = result["P_renewable_mw"]
            rec.area_import[k] = result["area_import_mw"]
            rec.lmp_zonal[k]   = result["lmp_zonal"]
            rec.lmp_bus16[k]   = result["lmp_bus16"]
            rec.congestion[k]  = result["congestion_comp"]
            rec.loss_sens[k]   = result["loss_sens"]
            rec.total_cost[k]  = result["total_cost_usd"]
            rec.dr_value[k]    = result["dr_value_usd"]
            rec.v_viol_pcc[k]  = result["v_viol_pcc"]
            rec.v_viol_sys[k]  = result["v_viol_sys"]
            rec.dr_active[k]   = result["dr_active"]

            if k % 24 == 0:
                print(
                    f"    h={t_opf/3600:5.1f}  "
                    f"V_pcc={result['V_pcc']:.4f} pu  "
                    f"f={result['freq_hz']:.3f} Hz  "
                    f"LMP_16={result['lmp_bus16']:6.1f} $/MWh  "
                    f"P_DC={result['P_dc_mw']:.3f} MW  "
                    f"DR={result['dr_mw']*1000:5.0f} kW  "
                    f"Cong={result['congestion_comp']:.1f} $/MWh"
                )

        rec.total_cost_day   = float(rec.total_cost.sum())
        rec.total_energy_mwh = float((rec.P_dc * dt_h).sum())
        rec.total_dr_mwh     = float((rec.dr_mw * dt_h).sum())
        rec.total_dr_value   = float(rec.dr_value.sum())
        rec.n_viol_pcc       = int(rec.v_viol_pcc.sum())
        rec.n_viol_sys       = int(rec.v_viol_sys.sum())
        self.records[sc_name] = rec

        elapsed = time.time() - t_wall
        print(
            f"\n    Cost=${rec.total_cost_day:,.0f}  |  "
            f"DC energy={rec.total_energy_mwh:.2f} MWh  |  "
            f"DR={rec.total_dr_mwh:.4f} MWh (${rec.total_dr_value:.0f})  |  "
            f"V_viol_pcc={rec.n_viol_pcc}  |  "
            f"V_viol_sys={rec.n_viol_sys}  |  "
            f"{elapsed:.1f}s"
        )
        deregister(sc_dc_name)

    # ─────────────────────────────────────────────────────────────────────────
    def analyse(self):
        print(f"\n{'─'*74}\n  POST-SIMULATION ANALYSIS\n{'─'*74}")
        baseline = list(self.records.values())[0]

        print("\n  DAILY ECONOMICS")
        hdr = f"  {'Scenario':<24} {'Cost/Day':>11} {'DC Energy':>11} " \
              f"{'DR MWh':>9} {'DR Value':>10} {'V_pcc viol':>11} {'V_sys viol':>11}"
        print(hdr)
        print("  " + "─" * 92)
        for sc, rec in self.records.items():
            sav = baseline.total_cost_day - rec.total_cost_day
            tag = f"(−${sav:,.0f})" if sav > 0 else f"(+${abs(sav):,.0f})" if sav < 0 else "(baseline)"
            print(
                f"  {sc:<24} {rec.total_cost_day:>11,.0f} "
                f"{rec.total_energy_mwh:>11.3f} "
                f"{rec.total_dr_mwh:>9.4f} "
                f"{rec.total_dr_value:>10.0f} "
                f"{rec.n_viol_pcc:>11} "
                f"{rec.n_viol_sys:>11}   {tag}"
            )

        print("\n  VOLTAGE — BUS 16 PCC")
        for sc, rec in self.records.items():
            ok = 100 * np.mean((rec.V_pcc >= V_MIN) & (rec.V_pcc <= V_MAX))
            print(f"  {sc:<24}  mean={rec.V_pcc.mean():.4f}  "
                  f"min={rec.V_pcc.min():.4f}  "
                  f"max={rec.V_pcc.max():.4f}  "
                  f"ANSI={ok:.1f}%")

        print("\n  VOLTAGE — SYSTEM-WIDE MINIMUM")
        for sc, rec in self.records.items():
            print(f"  {sc:<24}  V_min mean={rec.V_min.mean():.4f}  "
                  f"V_min_abs={rec.V_min.min():.4f} pu")

        print("\n  FREQUENCY")
        for sc, rec in self.records.items():
            print(f"  {sc:<24}  mean={rec.freq_hz.mean():.4f} Hz  "
                  f"min={rec.freq_hz.min():.4f} Hz  "
                  f"ROCOF_max={np.abs(rec.rocof).max():.4f} Hz/s")

        print("\n  TRANSMISSION LOSSES")
        for sc, rec in self.records.items():
            pct = 100 * rec.P_losses_mw.mean() / max(rec.P_gen_mw.mean(), 1.0)
            print(f"  {sc:<24}  mean={rec.P_losses_mw.mean():.1f} MW  ({pct:.2f}%)")

        print("\n  CONGESTION PREMIUM")
        for sc, rec in self.records.items():
            cong_hrs = np.sum(rec.congestion > 0.1) * DT_S / 3600.0
            cong_max = rec.congestion.max()
            print(f"  {sc:<24}  max={cong_max:.2f} $/MWh  "
                  f"duration={cong_hrs:.1f} h")

        print("\n  MOTOR OPERATING POINT")
        for sc, rec in self.records.items():
            print(f"  {sc:<24}  ω_r={rec.omega_r.mean():.4f}  "
                  f"slip={rec.slip.mean()*100:.2f}%")

        print("─" * 74)

    # ─────────────────────────────────────────────────────────────────────────
    def save_csv(self):
        import pandas as pd
        rows = []
        for sc, rec in self.records.items():
            for k in range(N_INTERVALS):
                rows.append({
                    "scenario":         sc,
                    "hour":             INTERVAL_HOURS[k],
                    "interval":         k,
                    "lmp_zonal":        rec.lmp_zonal[k],
                    "lmp_bus16":        rec.lmp_bus16[k],
                    "congestion_comp":  rec.congestion[k],
                    "P_dc_bid_mw":      rec.P_dc_bid[k],
                    "P_dc_mw":          rec.P_dc[k],
                    "Q_dc_mvar":        rec.Q_dc[k],
                    "dr_mw":            rec.dr_mw[k],
                    "P_server_kw":      rec.P_server[k],
                    "P_cool_kw":        rec.P_cool[k],
                    "omega_r_pu":       rec.omega_r[k],
                    "slip":             rec.slip[k],
                    "V_pcc_pu":         rec.V_pcc[k],
                    "V_bus14_pu":       rec.V_bus14[k],
                    "V_bus15_pu":       rec.V_bus15[k],
                    "V_min_pu":         rec.V_min[k],
                    "freq_hz":          rec.freq_hz[k],
                    "rocof_hz_s":       rec.rocof[k],
                    "P_losses_mw":      rec.P_losses_mw[k],
                    "P_gen_mw":         rec.P_gen_mw[k],
                    "P_renewable_mw":   rec.P_renew_mw[k],
                    "area_import_mw":   rec.area_import[k],
                    "total_cost_usd":   rec.total_cost[k],
                    "dr_value_usd":     rec.dr_value[k],
                    "loss_sens":        rec.loss_sens[k],
                    "v_viol_pcc":       int(rec.v_viol_pcc[k]),
                    "v_viol_sys":       int(rec.v_viol_sys[k]),
                    "dr_active":        int(rec.dr_active[k]),
                })
        pd.DataFrame(rows).to_csv(OUT_CSV, index=False, float_format="%.6f")
        print(f"  CSV  →  {OUT_CSV}")

    # ─────────────────────────────────────────────────────────────────────────
    def plot(self):
        h = INTERVAL_HOURS
        SC_COL = {
            "S1_Baseline":        "#1a5276",
            "S2_PriceResponse":   "#1e8449",
            "S3_VoltFreqSupport": "#7d3c98",
            "S4_Congestion":      "#b7770d",
            "S5_GenTrip":         "#922b21",
        }
        SC_LBL = {
            "S1_Baseline":        "S1  Baseline",
            "S2_PriceResponse":   "S2  Price-Responsive DR",
            "S3_VoltFreqSupport": "S3  DR + Volt/Freq Support",
            "S4_Congestion":      "S4  Congestion",
            "S5_GenTrip":         "S5  G06 Trip + DR",
        }
        LW, AL = 1.2, 0.82

        fig = plt.figure(figsize=(20, 32))
        fig.patch.set_facecolor("white")
        gs  = gridspec.GridSpec(12, 1, figure=fig, hspace=0.55,
                                top=0.96, bottom=0.03,
                                left=0.08, right=0.96)
        axs = [fig.add_subplot(gs[i]) for i in range(12)]

        def style(ax):
            ax.set_facecolor("#f8f9fa")
            ax.tick_params(colors="#2c3e50", labelsize=8)
            for sp in ax.spines.values():
                sp.set_edgecolor("#ced4da"); sp.set_linewidth(0.6)
            ax.grid(axis="y", color="#dee2e6", lw=0.5, ls=":")
            ax.set_xlim(0, 24); ax.set_xticks(range(0, 25, 4))

        def peaks(ax):
            ax.axvspan( 7, 10, color="#fff3cd", alpha=0.50, zorder=0,
                        label="Morning peak")
            ax.axvspan(17, 21, color="#fff3cd", alpha=0.50, zorder=0)

        def vlim(ax):
            ax.axhline(V_MIN, color="#e74c3c", lw=0.8, ls="--", alpha=0.7)
            ax.axhline(V_MAX, color="#e74c3c", lw=0.8, ls="--", alpha=0.7)
            ax.axhline(1.00,  color="#adb5bd", lw=0.4, ls=":",  alpha=0.5)

        for ax in axs:
            style(ax)

        r0 = list(self.records.values())[0]

        # ── Panel 0: LMP ──────────────────────────────────────────────────────
        axs[0].fill_between(h, r0.lmp_zonal, alpha=0.15, color="#1a5276")
        axs[0].plot(h, r0.lmp_zonal, color="#1a5276", lw=1.5,
                    label="Zonal LMP (S1 reference)")
        for sc, rec in self.records.items():
            axs[0].plot(h, rec.lmp_bus16, color=SC_COL[sc], lw=LW,
                        ls="--", alpha=AL, label=f"Bus 16 LMP — {SC_LBL[sc]}")
        axs[0].axhline(DR_THRESHOLD, color="#e74c3c", lw=0.8, ls=":",
                       alpha=0.7, label=f"DR threshold ({DR_THRESHOLD} $/MWh)")
        peaks(axs[0])
        axs[0].set_ylabel("LMP ($/MWh)", fontsize=8.5)
        axs[0].set_title("Locational Marginal Prices — Zonal vs. Bus 16",
                         fontsize=9.5, loc="left", color="#1a2744", pad=3)
        axs[0].legend(fontsize=7, ncol=2, facecolor="white",
                      edgecolor="#ced4da")

        # ── Panel 1: Congestion premium ───────────────────────────────────────
        for sc, rec in self.records.items():
            if rec.congestion.max() > 0.1:
                axs[1].fill_between(h, rec.congestion, alpha=0.18,
                                    color=SC_COL[sc])
                axs[1].plot(h, rec.congestion, color=SC_COL[sc],
                            lw=LW, alpha=AL, label=SC_LBL[sc])
        axs[1].set_ylabel("Cong. Comp. ($/MWh)", fontsize=8.5)
        axs[1].set_title("Congestion Premium at Bus 16", fontsize=9.5,
                         loc="left", color="#1a2744", pad=3)
        axs[1].legend(fontsize=7.5, facecolor="white", edgecolor="#ced4da")

        # ── Panel 2: Datacenter active power ──────────────────────────────────
        axs[2].plot(h, r0.P_dc_bid * 1000, color="#adb5bd", lw=0.8,
                    ls=":", alpha=0.7, label="Unconstrained bid")
        for sc, rec in self.records.items():
            axs[2].plot(h, rec.P_dc * 1000, color=SC_COL[sc],
                        lw=LW, alpha=AL, label=SC_LBL[sc])
        peaks(axs[2])
        axs[2].set_ylabel("P_DC (kW)", fontsize=8.5)
        axs[2].set_title("Datacenter Active Power Dispatch — Bus 16",
                         fontsize=9.5, loc="left", color="#1a2744", pad=3)
        axs[2].legend(fontsize=7.5, ncol=2, facecolor="white",
                      edgecolor="#ced4da")

        # ── Panel 3: DR curtailment ───────────────────────────────────────────
        any_dr = False
        for sc, rec in self.records.items():
            if rec.dr_mw.max() > 1e-3:
                axs[3].fill_between(h, rec.dr_mw * 1000, alpha=0.20,
                                    color=SC_COL[sc])
                axs[3].plot(h, rec.dr_mw * 1000, color=SC_COL[sc],
                            lw=LW, alpha=AL, label=SC_LBL[sc])
                any_dr = True
        if not any_dr:
            axs[3].text(12, 0.5, "No DR activated", ha="center",
                        va="center", fontsize=9, color="#adb5bd")
        axs[3].set_ylabel("DR Curtailment (kW)", fontsize=8.5)
        axs[3].set_title("Demand Response Curtailment", fontsize=9.5,
                         loc="left", color="#1a2744", pad=3)
        if any_dr:
            axs[3].legend(fontsize=7.5, facecolor="white", edgecolor="#ced4da")

        # ── Panel 4: Datacenter reactive power ────────────────────────────────
        for sc, rec in self.records.items():
            axs[4].plot(h, rec.Q_dc, color=SC_COL[sc], lw=LW,
                        alpha=AL, label=SC_LBL[sc])
        axs[4].axhline(0, color="#888", lw=0.4, ls=":")
        axs[4].set_ylabel("Q_DC (MVAR)", fontsize=8.5)
        axs[4].set_title("Datacenter Reactive Power — Bus 16", fontsize=9.5,
                         loc="left", color="#1a2744", pad=3)
        axs[4].legend(fontsize=7.5, ncol=2, facecolor="white",
                      edgecolor="#ced4da")

        # ── Panel 5: Bus 16 PCC voltage ───────────────────────────────────────
        for sc, rec in self.records.items():
            axs[5].plot(h, rec.V_pcc, color=SC_COL[sc], lw=LW,
                        alpha=AL, label=SC_LBL[sc])
        vlim(axs[5])
        peaks(axs[5])
        axs[5].set_ylabel("|V| pu", fontsize=8.5)
        axs[5].set_title("Voltage — Bus 16 PCC", fontsize=9.5,
                         loc="left", color="#1a2744", pad=3)
        axs[5].legend(fontsize=7.5, ncol=2, facecolor="white",
                      edgecolor="#ced4da")

        # ── Panel 6: System-wide minimum voltage ──────────────────────────────
        for sc, rec in self.records.items():
            axs[6].plot(h, rec.V_min, color=SC_COL[sc], lw=LW,
                        alpha=AL, label=SC_LBL[sc])
        axs[6].axhline(0.90, color="#e74c3c", lw=0.8, ls="--",
                       alpha=0.7, label="0.90 pu floor")
        axs[6].set_ylabel("V_min (pu)", fontsize=8.5)
        axs[6].set_title("System-Wide Minimum Voltage (all 39 buses)",
                         fontsize=9.5, loc="left", color="#1a2744", pad=3)
        axs[6].legend(fontsize=7.5, ncol=2, facecolor="white",
                      edgecolor="#ced4da")

        # ── Panel 7: System frequency ─────────────────────────────────────────
        for sc, rec in self.records.items():
            axs[7].plot(h, rec.freq_hz, color=SC_COL[sc], lw=LW,
                        alpha=AL, label=SC_LBL[sc])
        axs[7].axhline(60.0, color="#888", lw=0.4, ls=":")
        axs[7].axhline(59.5, color="#e74c3c", lw=0.8, ls="--", alpha=0.7)
        axs[7].set_ylabel("f (Hz)", fontsize=8.5)
        axs[7].set_title("System Frequency (COI)", fontsize=9.5,
                         loc="left", color="#1a2744", pad=3)
        axs[7].legend(fontsize=7.5, ncol=2, facecolor="white",
                      edgecolor="#ced4da")

        # ── Panel 8: HVAC motor rotor speed ──────────────────────────────────
        for sc, rec in self.records.items():
            axs[8].plot(h, rec.omega_r, color=SC_COL[sc], lw=LW,
                        alpha=AL, label=SC_LBL[sc])
        axs[8].axhline(1.0, color="#888", lw=0.4, ls=":")
        axs[8].set_ylabel("ω_r (pu)", fontsize=8.5)
        axs[8].set_title("HVAC Motor Rotor Speed", fontsize=9.5,
                         loc="left", color="#1a2744", pad=3)
        axs[8].legend(fontsize=7.5, ncol=2, facecolor="white",
                      edgecolor="#ced4da")

        # ── Panel 9: Renewable generation ─────────────────────────────────────
        axs[9].fill_between(h, self.wind_mw, alpha=0.25, color="#1e8449",
                            label="Wind (MW)")
        axs[9].fill_between(h, self.solar_mw, alpha=0.25, color="#e67e22",
                            label="Solar (MW)")
        axs[9].plot(h, self.wind_mw,  color="#1e8449", lw=1.0)
        axs[9].plot(h, self.solar_mw, color="#e67e22", lw=1.0)
        axs[9].set_ylabel("P_renew (MW)", fontsize=8.5)
        axs[9].set_title("Renewable Generation (Wind + Solar)",
                         fontsize=9.5, loc="left", color="#1a2744", pad=3)
        axs[9].legend(fontsize=7.5, facecolor="white", edgecolor="#ced4da")

        # ── Panel 10: System load vs generation ───────────────────────────────
        axs[10].plot(h, self.load_profile * S_BASE_MVA, color="#1a5276",
                     lw=1.2, label="Total load (MW)")
        for sc, rec in self.records.items():
            axs[10].plot(h, rec.P_gen_mw, color=SC_COL[sc], lw=0.8,
                         ls="--", alpha=0.7, label=f"Gen — {SC_LBL[sc]}")
        peaks(axs[10])
        axs[10].set_ylabel("P (MW)", fontsize=8.5)
        axs[10].set_title("System Load vs. Total Generation", fontsize=9.5,
                          loc="left", color="#1a2744", pad=3)
        axs[10].legend(fontsize=7, ncol=3, facecolor="white",
                       edgecolor="#ced4da")

        # ── Panel 11: Cumulative cost ─────────────────────────────────────────
        dt_h = DT_S / 3600.0
        for sc, rec in self.records.items():
            axs[11].plot(h, rec.total_cost.cumsum(),
                         color=SC_COL[sc], lw=LW, alpha=AL,
                         label=f"{SC_LBL[sc]}  (${rec.total_cost_day:,.0f}/day)")
        axs[11].set_ylabel("Cumul. Cost ($)", fontsize=8.5)
        axs[11].set_title("Cumulative Daily Generation Cost", fontsize=9.5,
                          loc="left", color="#1a2744", pad=3)
        axs[11].legend(fontsize=7.5, facecolor="white", edgecolor="#ced4da")

        for ax in axs:
            ax.set_xlabel("")
        axs[-1].set_xlabel("Hour of day", fontsize=8.5, color="#2c3e50")

        fig.suptitle(
            "IEEE 39-Bus Transmission OPF — AI Datacenter at Bus 16 (2 MVA)",
            fontsize=11, color="#1a2744", y=0.99)

        plt.savefig(OUT_PLOT, dpi=150, bbox_inches="tight",
                    facecolor="white")
        plt.close()
        print(f"  PNG  →  {OUT_PLOT}")

    # ─────────────────────────────────────────────────────────────────────────
    def report(self):
        sep  = "=" * 74
        recs = self.records
        baseline = list(recs.values())[0]

        lines = [
            sep,
            "  TRANSMISSION OPF STUDY — FINAL REPORT",
            sep,
            "",
            "  CONFIGURATION",
            f"    Network        : IEEE 39-Bus New England, {S_BASE_MVA:.0f} MVA",
            f"    Datacenter     : {DC_NAME}  (2 MVA, bus {DC_PCC_BUS})",
            f"    Horizon        : {HOURS} h  ·  {N_INTERVALS} intervals  ·  {DT_MIN}-min",
            f"    OPF method     : DC merit-order + AC (NR) verification",
            f"    V limits       : [{V_MIN:.2f}, {V_MAX:.2f}] pu  (ANSI C84.1 Range A)",
            f"    DR threshold   : {DR_THRESHOLD:.0f} $/MWh",
            f"    Interface limit: {INTERFACE_LIMIT_MW:.0f} MW  (bus 16 area import)",
            "",
        ]

        for sc, rec in recs.items():
            sav = baseline.total_cost_day - rec.total_cost_day
            ok  = 100 * np.mean((rec.V_pcc >= V_MIN) & (rec.V_pcc <= V_MAX))
            lines += [
                f"  {sc}",
                f"    Cost: ${rec.total_cost_day:,.0f}/day  "
                f"Saving: ${sav:+,.0f} ({100*sav/max(baseline.total_cost_day,1):.1f}%)",
                f"    DC energy: {rec.total_energy_mwh:.3f} MWh  |  "
                f"DR curtailed: {rec.total_dr_mwh*1000:.2f} kWh  |  "
                f"DR value: ${rec.total_dr_value:.0f}",
                f"    V_pcc mean={rec.V_pcc.mean():.4f}  "
                f"min={rec.V_pcc.min():.4f}  max={rec.V_pcc.max():.4f}  "
                f"ANSI={ok:.1f}%",
                f"    V_min_sys={rec.V_min.min():.4f} pu  |  "
                f"f_min={rec.freq_hz.min():.4f} Hz  |  "
                f"ROCOF_max={np.abs(rec.rocof).max():.4f} Hz/s",
                f"    V_viol_pcc={rec.n_viol_pcc}  V_viol_sys={rec.n_viol_sys}  |  "
                f"Cong_max={rec.congestion.max():.2f} $/MWh",
                "",
            ]

        lines += [
            "  OUTPUT FILES",
            f"    {OUT_PLOT}",
            f"    {OUT_CSV}",
            f"    {OUT_RPT}",
            "",
            sep,
        ]

        text = "\n".join(lines)
        OUT_RPT.write_text(text)
        print(text)
        print(f"  RPT  →  {OUT_RPT}")

    # ─────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _banner():
        print("\n" + "=" * 74)
        print("  OASIS — Transmission OPF Study")
        print("  IEEE 39-Bus New England Test System")
        print("  AI Datacenter at Bus 16  (2 MVA, 100 MVA base)")
        print("=" * 74)


# =============================================================================
#  ENTRY POINT
# =============================================================================
def main():
    study = TransmissionOPFStudy()
    study.run_all_scenarios()
    study.analyse()
    study.save_csv()
    study.plot()
    study.report()
    print("\nDone.")


if __name__ == "__main__":
    main()
