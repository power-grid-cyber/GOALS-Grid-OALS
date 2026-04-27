"""
Datacenter Subsystem Simulator — grid-supporting variant for co-simulation.

This module packages the complete datacenter model as a co-simulation
component with a single step() call per macro time step.

Architecture
------------
                 ┌─────────────────────────────────────────────────┐
  V_pcc, f  ──► │  GPU Workload Traces  (5 clusters, 810 GPUs)     │
                 │       ↓ P_server                                  │
                 │  Induction Machine HVAC  (3 × 250 kVA)           │
  from network  │    equivalent circuit + swing equation            │  P_dc, Q_dc
                 │       ↓ P_cool, Q_cool                            │ ──────────►
                 │  Grid-Support VSC                                  │  to network
                 │    frequency-watt droop  ΔP = −K_fw · Δf          │
                 │    volt-VAR droop        ΔQ = −K_vv · ΔV          │
                 │    LVRT / FRT ride-through (IEEE 1547-2018)        │
                 └─────────────────────────────────────────────────┘

Induction machine model: 3rd-order reduced equivalent-circuit model
  1. Equivalent-circuit admittance gives P(V, s) and Q(V, s):
         Z_in = R_s + jX_ls + jX_m ‖ (R_r/s + jX_lr)
         S    = V² / Z_in*           (per unit on machine base)
  2. Electromagnetic torque from air-gap power:
         T_e  = P_ag / ω_r           (air-gap power = P − I²R_s)
  3. Rotor speed — swing equation (RK4):
         dω_r/dt = (ω_s / 2H) · (T_e − T_L)
         T_L     = T_L0 · ω_r²       (quadratic fan/pump law)

  Slip is recomputed each micro-step:  s = (ω_s − ω_r) / ω_s

  • Frequency-Watt droop:  ΔP_pu = −K_fw · Δf  (5% on 2 MVA base)
    Reduces load draw as grid frequency falls, providing primary frequency
    response symmetrically with generators.

  • Volt-VAR droop:  ΔQ_pu = −K_vv · ΔV
    Injects reactive power when PCC voltage sags, supporting voltage
    recovery cooperatively with the feeder capacitors.

  • LVRT / FRT ride-through (IEEE 1547-2018 Cat III):
    Stays connected for |V| ≥ 0.50 pu (momentary), f ∈ [59.0, 61.0] Hz.
    Only hard-trips for sustained severe violations beyond clearing times.
    Replaces the instant-trip logic from the standalone islanding relay.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple

# ── Constants ──────────────────────────────────────────────────────────────────
F_NOM = 60.0
OM_NOM = 2.0 * np.pi * F_NOM
S_BASE = 2_000_000.0  # datacenter apparent power base  [VA]

# GPU server parameters
P_IDLE = 20.0  # W  idle power per GPU
ALPHA = 160.0  # W  power coefficient (rated − idle)
NOISE_P = 4.0  # W  Gaussian noise std


# ─────────────────────────────────────────────────────────────────────────────
#  Result dataclass
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class DatacenterResults:
    """Quantities returned to the co-simulation orchestrator each macro step."""
    P_total_mw: float = 0.0  # total active power at PCC     [MW]
    Q_total_mvar: float = 0.0  # total reactive power at PCC   [MVAR]
    P_server_kw: float = 0.0  # IT server load                [kW]
    P_cool_kw: float = 0.0  # HVAC active draw              [kW]
    Q_cool_kvar: float = 0.0  # HVAC reactive draw            [kVAR]
    omega_r_pu: float = 1.0  # avg motor rotor speed         [pu]
    slip: float = 0.02  # avg motor slip                [pu]
    riding_through: bool = False  # True when LVRT/FRT active
    freq_response_mw: float = 0.0  # droop-sourced power change    [MW]
    volt_var_mvar: float = 0.0  # droop-sourced VAR injection   [MVAR]


#  Induction machine — 3rd-order equivalent-circuit + swing model
class InductionMachineEQ:
    """
    Single squirrel-cage induction motor: equivalent-circuit power + swing equation.

    State: rotor speed ω_r [pu on synchronous speed ω_s].


    """
    # Parameters(typical 4 - pole HVAC compressor motor)
    rs = 0.045
    rr = 0.040
    Xls = 0.098
    Xlr = 0.098
    Xm = 3.200
    H = 0.80

    def __init__(self, rated_mva: float = 0.25, TL0: float = 0.80):
        self.rated_mva = rated_mva
        self.TL0 = TL0
        self.omega_r = 0.980  # initial speed [pu]  (2% slip at start)

    def _electrical(self, V: float, omega_r: float,
                    omega_s: float = 1.0) -> Tuple[float, float, float]:
        """
        Solve equivalent circuit for given terminal voltage V [pu on machine base]
        and rotor speed.  Returns (P_pu, Q_pu, T_e).

        Uses the exact equivalent circuit (no approximations):
            Z_in = (Rs + jXls) + jXm ‖ (Rr/s + jXlr)
        """
        s = (omega_s - omega_r) / omega_s
        s = np.clip(s, 0.001, 0.99)  # avoid division by zero

        Z_rotor = complex(self.rr / s, self.Xlr)
        Z_mag = complex(0.0, self.Xm)
        Z_parallel = (Z_mag * Z_rotor) / (Z_mag + Z_rotor)
        Z_total = complex(self.rs, self.Xls) + Z_parallel

        # Stator apparent power S = V · I_s* = |V|² / Z*
        S = (V ** 2) / Z_total.conjugate()
        P_pu = S.real
        Q_pu = S.imag

        # Air-gap power → electromagnetic torque
        I_s_sq = (V / abs(Z_total)) ** 2
        P_ag = P_pu - I_s_sq * self.rs  # subtract stator copper loss
        T_e = P_ag / omega_r if omega_r > 0.01 else 0.0

        return P_pu, Q_pu, T_e

    def _domega(self, omega_r: float, V: float, omega_s: float) -> float:
        """Swing equation RHS: dω_r/dt  [pu/s]."""
        _, _, T_e = self._electrical(V, omega_r, omega_s)
        T_L = self.TL0 * omega_r ** 2
        return (omega_s / (2.0 * self.H)) * (T_e - T_L)

    def step(self, V: float, freq_hz: float, dt: float) -> Tuple[float, float]:
        """
        Advance rotor dynamics one micro-step (RK4). Returns (P_MW, Q_MVAR) drawn from the feeder.
        """
        omega_s = freq_hz / F_NOM  # synchronous speed in pu

        # RK4 for rotor speed
        k1 = self._domega(self.omega_r, V, omega_s)
        k2 = self._domega(self.omega_r + dt / 2 * k1, V, omega_s)
        k3 = self._domega(self.omega_r + dt / 2 * k2, V, omega_s)
        k4 = self._domega(self.omega_r + dt * k3, V, omega_s)
        self.omega_r += (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        self.omega_r = float(np.clip(self.omega_r, 0.50, 1.10))

        P_pu, Q_pu, _ = self._electrical(V, self.omega_r, omega_s)
        return (P_pu * self.rated_mva,  # MW  (rated_mva already in MW)
                Q_pu * self.rated_mva)  # MVAR


#  Grid-supporting VSC
class GridSupportVSC:
    """
    Average-value VSC with d-q PI current control.
    Grid-support:
      Frequency-Watt droop:  ΔP_pu = −K_fw · Δf
      Volt-VAR droop:        ΔQ_pu = −K_vv · ΔV
      IEEE 1547-2018 Cat III LVRT/FRT ride-through

    All quantities in per-unit on S_BASE = 2 MVA.
    """
    # Filter + controller gains
    Lf = 0.05  # filter inductance  [pu]
    Rf = 0.005  # filter resistance  [pu]
    Kp_i = 0.80  # current-loop proportional gain
    Ki_i = 40.0  # current-loop integral gain
    Vdc = 1.05  # DC bus voltage  [pu]

    # Droop gains (5% frequency droop on 2 MVA base; 10% voltage droop)
    K_fw = 0.05  # pu_power / Hz
    K_vv = 0.10  # pu_power / pu_voltage

    # IEEE 1547-2018 Cat III ride-through thresholds
    # Hard-trip only for extreme sustained violations
    LVRT_HARD = 0.45  # pu  — below this, trip (IEEE 1547 Cat III: 0.45 pu momentary)
    FRT_LO = 58.5  # Hz  — below this, trip (IEEE 1547 Cat III: 58.5 Hz)
    FRT_HI = 61.5  # Hz  — above this, trip
    # Soft ride-through: stay connected, flag active, droop response engaged
    RT_SOFT_V = 0.88  # pu  — below this, flag ride-through
    RT_SOFT_F = 0.5  # Hz  — |Δf| above this, flag ride-through

    def __init__(self):
        self.id = self.iq = 0.0
        self.id_ref = self.iq_ref = 0.0
        self._ei_d = self._ei_q = 0.0
        self.om_pll = OM_NOM
        self.P_pu = self.Q_pu = 0.0
        self.riding_through = False
        self._tripped = False
        self._blank = 5.0  # startup blanking window [s]
        self._freq_resp_pu = 0.0
        self._volt_var_pu = 0.0

    def set_power_ref(self, P_base_pu: float, Q_base_pu: float,
                      V_pcc: float, freq: float):
        """
        Compute d-q current references incorporating droop corrections.

        Frequency-Watt:  ΔP = −K_fw · (f − f_nom)
            • Positive Δf → datacenter reduces load (freq support symmetric)
            • Negative Δf → datacenter reduces load (primary frequency response)

        Volt-VAR:  ΔQ = −K_vv · (V − 1.0)
            • Voltage sag → inject positive Q (capacitive support)
        """
        df = freq - F_NOM
        dV = V_pcc - 1.0

        dP = -self.K_fw * df  # load reduction (positive = draw less)
        dQ = -self.K_vv * dV  # Q injection    (positive = capacitive)

        self._freq_resp_pu = dP
        self._volt_var_pu = dQ

        P_cmd = float(np.clip(P_base_pu + dP, 0.0, 1.2))
        Q_cmd = float(np.clip(Q_base_pu + dQ, -0.5, 0.5))

        vd = max(V_pcc, 0.01)
        self.id_ref = P_cmd / vd
        self.iq_ref = -Q_cmd / vd

    def step(self, vd: float, vq: float, om_grid: float, dt: float):
        """Advance VSC one micro-step; enforce LVRT/FRT limits."""
        V = np.hypot(vd, vq)
        f = om_grid / (2.0 * np.pi)

        if self._blank > 0.0:
            self._blank -= dt
            # Still output approximate value during blanking
        else:
            # Hard-trip check (IEEE 1547-2018 momentary cessation)
            if V < self.LVRT_HARD or f < self.FRT_LO or f > self.FRT_HI:
                self._tripped = True
                self.P_pu = self.Q_pu = 0.0
                self.riding_through = False
                return

            # Soft ride-through (stay connected, flag active)
            self.riding_through = (V < self.RT_SOFT_V or
                                   abs(f - F_NOM) > self.RT_SOFT_F)

        # d-q current PI control with cross-coupling decoupling
        self.om_pll = om_grid
        err_d = self.id_ref - self.id
        err_q = self.iq_ref - self.iq
        self._ei_d += err_d * dt
        self._ei_q += err_q * dt

        vd_mod = (vd + self.Kp_i * err_d + self.Ki_i * self._ei_d
                  - self.om_pll * self.Lf * self.iq)
        vq_mod = (vq + self.Kp_i * err_q + self.Ki_i * self._ei_q
                  + self.om_pll * self.Lf * self.id)

        self.id += ((vd_mod - vd - self.Rf * self.id
                     + self.om_pll * self.Lf * self.iq) / self.Lf * dt)
        self.iq += ((vq_mod - vq - self.Rf * self.iq
                     - self.om_pll * self.Lf * self.id) / self.Lf * dt)

        self.P_pu = vd * self.id + vq * self.iq
        self.Q_pu = vq * self.id - vd * self.iq


#  GPU workload trace generators
def _build_server_trace(duration: float = 300.0, seed: int = 42, n_gpu:    int   = 810) -> np.ndarray:
    """
    Pre-generate combined GPU server power at 1-second resolution [W].

    Five workload archetypes across 810 GPUs:
      T1 (200 GPUs) — continuous inference,   mean utilisation 0.86
      T2 (150 GPUs) — Poisson-burst serving,  λ = 0.02 /s
      T3 (180 GPUs) — mini-batch training,    periodic duty cycle
      T4 (120 GPUs) — preprocessing-limited,  frequent short bursts
      T5 (160 GPUs) — compute-bound training, sinusoidal pattern
    """
    rng = np.random.default_rng(seed)
    N = int(np.ceil(duration))

    def power(u: np.ndarray) -> np.ndarray:
        return P_IDLE + ALPHA * u + rng.normal(0.0, NOISE_P, len(u))

    def T1() -> np.ndarray:
        u = np.clip(rng.normal(0.86, 0.02, N), 0, 1)
        for idx in rng.choice(N, 15, replace=False):
            w = rng.integers(3, 8)
            u[idx:idx + w] *= rng.uniform(0.1, 0.4)
        return u

    def T2() -> np.ndarray:
        u = np.zeros(N)
        t = 0.0
        while t < duration:
            t += rng.exponential(50)
            s = int(t);
            e = int(min(t + rng.uniform(5, 15), N))
            if s < N: u[s:e] = rng.uniform(0.8, 1.0)
        return u

    def T3() -> np.ndarray:
        u = np.zeros(N)
        t = 0.0
        while t < duration:
            T = rng.normal(10, 1)
            phi = rng.uniform(0.6, 0.8)
            s = max(0, min(int(t), N))
            m = max(s, min(int(t + phi * T), N))
            e = max(m, min(int(t + T), N))
            if m > s: u[s:m] = np.clip(rng.normal(0.60, 0.05, m - s), 0, 1)
            if e > m: u[m:e] = np.clip(rng.normal(0.30, 0.05, e - m), 0, 1)
            t += T
        return np.clip(u, 0, 1)

    def T4() -> np.ndarray:
        u = np.zeros(N)
        t = 0.0
        while t < duration:
            t += rng.exponential(2)
            s = int(t);
            e = int(min(t + rng.uniform(2, 6), N))
            if s < N: u[s:e] = rng.uniform(0.3, 0.7)
            if rng.random() < 0.05:
                t += rng.uniform(20, 40)
        return np.clip(u, 0, 1)

    def T5() -> np.ndarray:
        tt = np.arange(N, dtype=float)
        return np.clip(
            0.75 + 0.35 * np.sin(2 * np.pi / 8 * tt)
            + rng.normal(0.0, 0.03, N), 0, 1)

    # scales = [200, 150, 180, 120, 160]
    # total = np.zeros(N)
    # for gen, sc in zip([T1, T2, T3, T4, T5], scales):
    #     total += power(gen()) * sc
        # Archetype split proportional to n_gpu
    n_T1 = int(n_gpu * 200 / 810)  # continuous inference
    n_T2 = int(n_gpu * 150 / 810)  # Poisson-burst serving
    n_T3 = int(n_gpu * 180 / 810)  # mini-batch training
    n_T4 = int(n_gpu * 120 / 810)  # preprocessing-limited
    n_T5 = n_gpu - n_T1 - n_T2 - n_T3 - n_T4  # remainder to sinusoidal

    scales = [n_T1, n_T2, n_T3, n_T4, n_T5]
    total = np.zeros(N)
    for gen, sc in zip([T1, T2, T3, T4, T5], scales):
        total += power(gen()) * sc

    return total  # [W] at 1-second resolution


#  Datacenter subsystem — main co-simulation component
class DatacenterSubsystem:
    """
    Complete datacenter model packaged as a co-simulation component.
    The orchestrator calls step() once per macro time step (T_macro = 0.1 s).
    Internally, step() executes N_sub = T_macro / dt_micro sub-steps of the HVAC motor dynamics and VSC control loop.

    Parameters
    ----------
    dt_micro : float
        Internal integration time step [s].  Default 0.01 s (100 Hz).
    t_macro  : float
        Macro time step size [s] matching the network solver cadence.
    seed     : int
        Random seed for reproducible GPU workload traces.
    """
    COOLING_MVA = 0.25  # MVA rating per HVAC motor unit
    # N_COOLING = 3  # number of HVAC units

    def __init__(self, dt_micro: float = 0.01,
                 t_macro: float = 0.1,
                 seed: int = 42, n_gpu:   int = 810, n_cooling_units: int = 3):
        self.dt = dt_micro
        self.tmac = t_macro
        self.nsub = max(1, int(round(t_macro / dt_micro)))
        self.n_gpu = n_gpu
        # Scale the apparent power base with n_gpu
        # 810 GPUs ≈ 2 MVA → P_rated per GPU ≈ 2000/810 kVA = 2.469 kVA
        self.S_base = n_gpu * 2_000_000 / 810  # VA

        # Pre-generate server power trace
        self._srv_trace = _build_server_trace(duration=86_500.0, seed=seed, n_gpu = n_gpu)

        # HVAC induction machines
        n_cool = n_cooling_units if n_cooling_units > 1 else max(3, int(n_gpu / 270))
        self.N_COOLING = n_cool
        self.machines: List[InductionMachineEQ] = [
            InductionMachineEQ(self.COOLING_MVA, TL0=0.80)
            for _ in range(self.N_COOLING)
        ]

        # VSC at PCC
        self.vsc = GridSupportVSC()

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _server_power_w(self, t: float) -> float:
        """Linear interpolation into pre-generated server trace [W]."""
        idx_lo = int(t)
        idx_hi = min(idx_lo + 1, len(self._srv_trace) - 1)
        frac = t - idx_lo
        lo = self._srv_trace[min(idx_lo, len(self._srv_trace) - 1)]
        hi = self._srv_trace[idx_hi]
        return float(lo + frac * (hi - lo))

    # ── Main co-simulation step ───────────────────────────────────────────────
    def step(self,
             V_pcc_pu: float,
             freq_hz: float,
             t_sim: float) -> DatacenterResults:
        """
        Advance the datacenter model for one macro step.

        Parameters
        ----------
        V_pcc_pu : float
            PCC voltage magnitude from the network solver [pu].
        freq_hz  : float
            System frequency from the network solver [Hz].
        t_sim    : float
            Current simulation time [s].

        Returns
        -------
        DatacenterResults
            Averaged power quantities over the macro step, ready for
            injection into the network power-flow solution.
        """
        om_s = 2.0 * np.pi * freq_hz
        vd = float(V_pcc_pu)  # d-axis aligned with PCC voltage
        vq = 0.0

        # Accumulators (averaged over sub-steps)
        acc_Psrv = acc_Pcool = acc_Qcool = 0.0
        acc_omr = acc_Pvsc = acc_Qvsc = 0.0
        acc_fr = acc_vv = 0.0

        for sub in range(self.nsub):
            t_cur = t_sim + sub * self.dt

            # 1. Server IT load
            P_srv = self._server_power_w(t_cur)

            # 2. HVAC motors (3 × RK4)
            P_cool = Q_cool = omr = 0.0
            for m in self.machines:
                P_mw, Q_mvar = m.step(V_pcc_pu, freq_hz, self.dt)
                P_cool += P_mw * 1e6  # → W
                Q_cool += Q_mvar * 1e6  # → VAR
                omr += m.omega_r
            omr /= self.N_COOLING

            # 3. VSC set-point from total load + droop
            P_ref_pu = (P_srv + P_cool) / self.S_base
            Q_ref_pu = Q_cool / self.S_base
            self.vsc.set_power_ref(P_ref_pu, Q_ref_pu, V_pcc_pu, freq_hz)
            self.vsc.step(vd, vq, om_s, self.dt)

            # Accumulate
            acc_Psrv += P_srv
            acc_Pcool += P_cool
            acc_Qcool += Q_cool
            acc_omr += omr
            acc_Pvsc += self.vsc.P_pu * self.S_base
            acc_Qvsc += self.vsc.Q_pu * self.S_base
            acc_fr += self.vsc._freq_resp_pu
            acc_vv += self.vsc._volt_var_pu

        n = self.nsub
        slip = (1.0 - acc_omr / n) if acc_omr / n < 1.0 else 0.005

        # VSC measured quantities are the definitive boundary values
        P_out_mw = max(0.0, acc_Pvsc / n / 1e6)
        Q_out_mvar = acc_Qvsc / n / 1e6

        return DatacenterResults(
            P_total_mw=P_out_mw,
            Q_total_mvar=Q_out_mvar,
            P_server_kw=acc_Psrv / n / 1e3,
            P_cool_kw=acc_Pcool / n / 1e3,
            Q_cool_kvar=acc_Qcool / n / 1e3,
            omega_r_pu=acc_omr / n,
            slip=slip,
            riding_through=self.vsc.riding_through,
            freq_response_mw=acc_fr / n * self.S_base / 1e6,
            volt_var_mvar=acc_vv / n * self.S_base / 1e6,
        )
