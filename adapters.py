"""
adapters.py
===========
Study-type adapters. Each translates between the canonical datacenter
interface and the signal conventions of a specific power system study.

One class per study type. Each class:
  - Holds a reference to DatacenterPhysics (never owns the physics)
  - Translates input signals INTO CanonicalInput
  - Translates CanonicalOutput INTO study-native return values
  - Contains zero physics

Available adapters
------------------
  DistributionAdapter   — OpenDSS / distribution power flow co-simulation
  TransmissionAdapter   — PSS/E-style transient stability (Norton injection)
  OPFAdapter            — MATPOWER / pandapower optimal power flow
  MarketAdapter         — ISO day-ahead + real-time market settlement
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict, Any

from datacenter_core import DatacenterPhysics, CanonicalInput, CanonicalOutput


# ─────────────────────────────────────────────────────────────────────────────
class DistributionAdapter:
    """
    For co-simulation with OpenDSS / distribution power flow tools.

    Signal conventions (matches existing cosim_framework.py exactly):
      Input  : V_pcc_pu [pu on feeder base], freq_hz [Hz], t [s], dt [s]
      Output : P_mw [MW], Q_mvar [MVAR] ready for NR injection

    This adapter is a near-passthrough — the distribution study is the
    native environment for which the datacenter model was originally built.
    It adds voltage base checking and exposes the full CanonicalOutput for
    any analysis that needs internal states (motor speed, ride-through, etc.)
    """

    def __init__(self, physics: DatacenterPhysics,
                 feeder_base_kv: float = 4.16,
                 dc_base_kv:     float = 0.48):
        self._p             = physics
        self.feeder_base_kv = feeder_base_kv
        self.dc_base_kv     = dc_base_kv
        self._last_out: CanonicalOutput = CanonicalOutput()

    def step(self, V_pcc_pu: float, freq_hz: float,
             t: float, dt: float = 0.1) -> Tuple[float, float]:
        """
        Called by cosim_framework each macro step.
        Returns (P_mw, Q_mvar) for injection into the network power flow.
        Full state available via .last_output property.
        """
        out = self._p.step(CanonicalInput(
            V_pu    = V_pcc_pu,
            freq_hz = freq_hz,
            V_d_pu  = V_pcc_pu,
            V_q_pu  = 0.0,
            t_sim   = t,
            dt      = dt,
        ))
        self._last_out = out
        return out.P_mw, out.Q_mvar

    @property
    def last_output(self) -> CanonicalOutput:
        """Full state from the most recent step (for monitors, logging)."""
        return self._last_out

    def get_grid_support_signals(self) -> Dict[str, float]:
        """Convenience accessor for grid-support quantities."""
        o = self._last_out
        return {
            'dP_freq_response_mw':  o.dP_droop_mw,
            'dQ_volt_var_mvar':     o.dQ_droop_mvar,
            'omega_r_pu':           o.omega_r_pu,
            'slip':                 o.slip,
            'riding_through':       float(o.riding_through),
        }


# ─────────────────────────────────────────────────────────────────────────────
class TransmissionAdapter:
    """
    For transient/voltage stability in PSS/E-style tools.

    Signal conventions:
      Input  : V_d_sys, V_q_sys [pu on SYSTEM base, e.g. 100 MVA]
      Output : I_d_sys, I_q_sys [pu on system base] — Norton current injection

    The datacenter appears to the transmission network as a Norton equivalent:
      I_Norton = (P − jQ) / V*   projected onto d-q axes
    This is the standard interface for dynamic load models in PSS/E (LDFMAC),
    PowerWorld, and DIgSILENT PowerFactory.
    """

    def __init__(self, physics: DatacenterPhysics,
                 S_sys_mva: float = 100.0):
        self._p       = physics
        self.S_sys    = S_sys_mva
        self.S_dc     = 2.0           # datacenter MVA base
        self._last_out: CanonicalOutput = CanonicalOutput()

    def step(self, V_d_sys: float, V_q_sys: float,
             freq_hz: float, t: float,
             dt: float = 0.005) -> Tuple[float, float, CanonicalOutput]:
        """
        V_d_sys, V_q_sys : terminal voltage in pu on SYSTEM base.
        Returns (I_d_sys, I_q_sys, full_state).
        """
        V_mag = float(np.hypot(V_d_sys, V_q_sys))
        out = self._p.step(CanonicalInput(
            V_pu    = V_mag,
            freq_hz = freq_hz,
            V_d_pu  = V_d_sys,
            V_q_pu  = V_q_sys,
            t_sim   = t,
            dt      = dt,
        ))
        self._last_out = out

        # Convert MW/MVAR to Norton current on system base
        P_sys = out.P_mw   / self.S_sys
        Q_sys = out.Q_mvar / self.S_sys
        V_d   = max(V_d_sys, 0.01)

        # d-axis aligned: Id absorbs P, Iq absorbs Q (motor convention: +Q inductive)
        I_d = P_sys / V_d
        I_q = Q_sys / V_d

        return I_d, I_q, out


# ─────────────────────────────────────────────────────────────────────────────
class OPFAdapter:
    """
    For optimal power flow (MATPOWER, pandapower, Pyomo, GAMS).

    The datacenter is modelled as a price-responsive dispatchable load.
    The OPF solver calls get_bid() to obtain the load cost curve,
    sets the interval dispatch via set_dispatch(), then calls step()
    to advance physics and get actual P/Q consumed.
    """

    def __init__(self, physics: DatacenterPhysics,
                 interval_min: float = 15.0):
        self._p           = physics
        self._interval_s  = interval_min * 60.0
        self._dispatch_mw = None

    def get_bid(self, t: float) -> Dict[str, Any]:
        """
        Returns pandapower-compatible controllable load bid dict.
        Call once per OPF interval before solving.
        """
        out = self._p.step(CanonicalInput(t_sim=t, dt=0.1))
        return {
            'p_mw':         out.P_mw,
            'q_mvar':       out.Q_mvar,
            'min_p_mw':     out.P_committed_mw,
            'max_p_mw':     out.P_mw + out.P_flex_mw * 0.10,
            'controllable': True,
            'cost_a':       2.5,
            'cost_b':       35.0,
            'cost_c':       0.0,
            'flex_mw':      out.P_flex_mw,
            'response_time_s': 30,
        }

    def set_dispatch(self, P_dispatch_mw: float):
        self._dispatch_mw = P_dispatch_mw

    def step(self, t: float, dt: float = 60.0,
             price: float = 0.0) -> Dict[str, float]:
        self._p.reset_energy_accumulator()
        out = self._p.step(CanonicalInput(
            t_sim=t, dt=dt, price_per_mwh=price))
        return {
            'P_actual_mw':   out.P_mw,
            'Q_actual_mvar': out.Q_mvar,
            'E_mwh':         out.E_interval_mwh,
            'cost':          out.cost_interval,
            'flex_mw':       out.P_flex_mw,
        }


# ─────────────────────────────────────────────────────────────────────────────
class MarketAdapter:
    """
    For ISO energy market simulation (day-ahead, real-time, ancillary services).
    """

    def __init__(self, physics: DatacenterPhysics,
                 participant_id: str = "DC_001"):
        self._p  = physics
        self.pid = participant_id

    def day_ahead_bid(self, hour: int,
                      price_forecast: float) -> Dict[str, Any]:
        out = self._p.step(CanonicalInput(
            t_sim=hour * 3600.0, dt=3600.0,
            price_per_mwh=price_forecast))
        return {
            'participant': self.pid, 'hour': hour,
            'P_bid_mw':    out.P_mw,
            'price_cap':   price_forecast * 1.15,
            'price_floor': price_forecast * 0.50,
        }

    def ancillary_offer(self) -> Dict[str, Any]:
        return {
            'participant':     self.pid,
            'reg_up_mw':       0.10,
            'reg_down_mw':     0.10,
            'spin_reserve_mw': 0.15,
            'activation_time_s': 10,
            'cost_reg_mw':     8.50,
            'cost_spin_mw':    3.20,
        }

    def settle_interval(self, t: float, dt_s: float,
                        price_rt: float,
                        reg_signal: float = 0.0) -> Dict[str, float]:
        self._p.reset_energy_accumulator()
        price_adj = price_rt - reg_signal * 50.0
        out = self._p.step(CanonicalInput(
            t_sim=t, dt=dt_s, price_per_mwh=price_adj))
        reg_mw = reg_signal * 0.10
        return {
            'participant':     self.pid,
            'E_mwh':           out.E_interval_mwh,
            'P_avg_mw':        out.P_mw,
            'price_rt':        price_rt,
            'revenue':         -out.cost_interval,
            'reg_provided_mw': reg_mw,
        }
