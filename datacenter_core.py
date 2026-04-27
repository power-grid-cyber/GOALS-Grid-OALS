"""
Canonical interface layer between the physics model and any study adapter.

CanonicalInput  — what any external study sends to the datacenter
CanonicalOutput — what the datacenter reports back to any study
DatacenterPhysics — thin wrapper around DatacenterSubsystem that speaks
                    the canonical language

All physics remain in datacenter_subsystem.py untouched.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from datacenter_subsystem import (
    DatacenterSubsystem, DatacenterResults,
    InductionMachineEQ, GridSupportVSC, _build_server_trace,
    F_NOM, S_BASE
)


# ─────────────────────────────────────────────────────────────────────────────
#  Canonical signal dataclasses
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class CanonicalInput:
    """Unified input — covers all study types."""
    V_pu: float = 1.0  # PCC voltage magnitude [pu on DC base]
    freq_hz: float = 60.0  # PCC frequency [Hz]
    V_d_pu: float = 1.0  # d-axis voltage component [pu]
    V_q_pu: float = 0.0  # q-axis voltage component [pu]
    price_per_mwh: float = 0.0  # real-time LMP [$/MWh] for demand response
    t_sim: float = 0.0  # current simulation time [s]
    dt: float = 0.1  # requested time step [s]


@dataclass
class CanonicalOutput:
    """Unified output — covers all study types."""
    # Primary power boundary
    P_mw: float = 0.0
    Q_mvar: float = 0.0
    P_flex_mw: float = 0.0  # curtailable load headroom [MW]
    P_committed_mw: float = 0.0  # non-deferrable floor [MW]

    # Internal states for protection/stability analysis
    V_pcc_pu: float = 1.0
    freq_hz: float = 60.0
    omega_r_pu: float = 1.0
    slip: float = 0.02
    vsc_id: float = 0.0
    vsc_iq: float = 0.0
    P_server_kw: float = 0.0
    P_cool_kw: float = 0.0
    Q_cool_kvar: float = 0.0

    # Grid-support signals
    dP_droop_mw: float = 0.0
    dQ_droop_mvar: float = 0.0
    riding_through: bool = False

    # Energy accounting
    E_interval_mwh: float = 0.0
    cost_interval: float = 0.0

    # Metadata
    t_sim: float = 0.0
    dt_actual: float = 0.1
    converged: bool = True


#  Physics core — wraps DatacenterSubsystem with canonical interface
class DatacenterPhysics:
    """
    Canonical-interface wrapper around DatacenterSubsystem.

    Accepts CanonicalInput, returns CanonicalOutput.
    Holds no study-specific logic — all study adaptation is in adapters.py.

    Parameters
    ----------
    config : dict, optional
        'n_cooling_units' : int   (default 3)
        'seed'            : int   (default 42)
        'duration'        : float simulation duration hint for trace pre-gen [s]
        'price_threshold' : float $/MWh below which no DR response
        'price_max'       : float $/MWh at which full curtailment
        'max_curtail_pu'  : float maximum load reduction fraction (default 0.30)
    """

    def __init__(self, config: dict = None):
        cfg = config or {}

        self._subsystem = DatacenterSubsystem(
            dt_micro=cfg.get('dt_micro', 0.01),
            t_macro=cfg.get('dt', 0.1),
            seed=cfg.get('seed', 42),
            n_gpu    = config.get("n_gpu",    810),
            n_cooling_units=config.get("n_cooling_units", 3)
        )

        self._price_threshold = cfg.get('price_threshold', 50.0)
        self._price_max = cfg.get('price_max', 200.0)
        self._max_curtail = cfg.get('max_curtail_pu', 0.30)
        self.n_gpu = cfg.get("n_gpu", 810)
        self.P_rated_mw = self.n_gpu * 180e-6

        # Energy accumulator (for market/OPF settlement)
        self._E_accum_mwh = 0.0
        self._t_prev = 0.0

        # Firm (non-curtailable) floor — idle power of all GPUs
        P_idle_total_w = 20.0 * 810  # P_IDLE * total GPUs
        self._P_firm_mw = P_idle_total_w / 1e6

    # ── Price-elastic demand response ─────────────────────────────────────────
    def _price_response(self, P_ref_pu: float, price: float) -> float:
        """Reduce load linearly when LMP exceeds threshold."""
        if price <= self._price_threshold:
            return P_ref_pu
        ratio = min(1.0, (price - self._price_threshold) /
                    (self._price_max - self._price_threshold))
        return P_ref_pu * (1.0 - self._max_curtail * ratio)

    # ── Main step ─────────────────────────────────────────────────────────────
    def step(self, inp: CanonicalInput) -> CanonicalOutput:
        """Advance one time step; return canonical results."""

        # Apply price-based DR by temporarily scaling VSC droop gain
        # (clean implementation: price signal modifies P_ref before VSC)
        V_use = max(inp.V_pu, 0.01)

        # Run the physics subsystem
        res: DatacenterResults = self._subsystem.step(
            V_pcc_pu=V_use,
            freq_hz=inp.freq_hz,
            t_sim=inp.t_sim,
        )

        # Apply price response on top of physics output
        if inp.price_per_mwh > self._price_threshold:
            scale = self._price_response(1.0, inp.price_per_mwh)
            P_out = res.P_total_mw * scale
            Q_out = res.Q_total_mvar * scale
        else:
            P_out = res.P_total_mw
            Q_out = res.Q_total_mvar

        # Energy accumulation
        dt_h = inp.dt / 3600.0
        self._E_accum_mwh += P_out * dt_h
        cost = P_out * dt_h * inp.price_per_mwh

        P_flex = max(0.0, P_out - self._P_firm_mw)

        return CanonicalOutput(
            P_mw=P_out,
            Q_mvar=Q_out,
            P_flex_mw=P_flex,
            P_committed_mw=self._P_firm_mw,
            V_pcc_pu=V_use,
            freq_hz=inp.freq_hz,
            omega_r_pu=res.omega_r_pu,
            slip=res.slip,
            vsc_id=self._subsystem.vsc.id,
            vsc_iq=self._subsystem.vsc.iq,
            P_server_kw=res.P_server_kw,
            P_cool_kw=res.P_cool_kw,
            Q_cool_kvar=res.Q_cool_kvar,
            dP_droop_mw=res.freq_response_mw,
            dQ_droop_mvar=res.volt_var_mvar,
            riding_through=res.riding_through,
            E_interval_mwh=self._E_accum_mwh,
            cost_interval=cost,
            t_sim=inp.t_sim,
            dt_actual=inp.dt,
            converged=True,
        )

    def reset_energy_accumulator(self):
        self._E_accum_mwh = 0.0
