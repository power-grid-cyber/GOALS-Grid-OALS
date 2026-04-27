"""
opendss_13bus_network.py  (opendssdirect-aware version)
==================================================
Drops in the real OpenDSS engine when opendssdirect is installed,
falls back to the built-in Newton-Raphson solver automatically.

Usage — no code changes needed in cosim_framework.py:
    # With opendssdirect installed:
    #   pip install opendssdirect.py
    # Point DSS_FILE to the master redirect file:
    #   DSS_FILE = "ieee13_datacenter.dss"   (relative or absolute path)
    # Everything else is identical.

Backend selection
-----------------
At import time the module tries:
    import opendssdirect as dss   → USE_OPENDSS = True
If that fails:
    USE_OPENDSS = False           → built-in NR solver used (no change)

The public interface (NetworkResults, OpenDSSNetworkSimulator) is identical
in both cases so cosim_framework.py and datacenter_subsystem.py are
completely unaffected by which backend is active.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import os

# ── Backend detection ─────────────────────────────────────────────────────────
try:
    import opendssdirect as dss
    USE_OPENDSS = True
except ImportError:
    USE_OPENDSS = False

# Path to the master DSS file (edit this to match your directory layout)
DSS_FILE = os.path.join(os.path.dirname(__file__), "ieee13_datacenter.dss")


# ─────────────────────────────────────────────────────────────────────────────
#  Shared result dataclass — identical for both backends
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class NetworkResults:
    V_pcc_pu:        float = 1.0
    theta_pcc:       float = 0.0
    V_bus632_pu:     float = 1.0
    V_bus680_pu:     float = 1.0
    freq_hz:         float = 60.0
    P_total_load_mw: float = 0.0
    P_losses_mw:     float = 0.0
    P_gen_total_mw:  float = 0.0
    converged:       bool  = True
    iterations:      int   = 0


# ─────────────────────────────────────────────────────────────────────────────
#  Internal data classes (used by built-in NR backend only)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Bus:
    name: str; P_mw: float=0.0; Q_mvar: float=0.0
    P_gen_mw: float=0.0; Q_gen_mvar: float=0.0
    is_slack: bool=False; V_pu: float=1.0; theta: float=0.0

@dataclass
class Branch:
    from_bus: str; to_bus: str; R_pu: float; X_pu: float; B_pu: float=0.0
    tap: float = 1.0   # off-nominal tap ratio (from_bus side); 1.0 = plain line


# ─────────────────────────────────────────────────────────────────────────────
#  OpenDSS Network Simulator
#  Automatically chooses the real OpenDSS engine or the built-in NR solver.
# ─────────────────────────────────────────────────────────────────────────────
class OpenDSSNetworkSimulator:

    def __init__(self, base_kv=4.16, base_mva=5.0, freq_hz=60.0):
        self.base_kv  = base_kv
        self.base_mva = base_mva
        self.freq_hz  = freq_hz
        self._freq_dev   = 0.0
        self._H_sys      = 5.0
        self._D_sys      = 2.0
        self._P_dg_prev  = 1.0
        self._fault_bus: Optional[str] = None
        self._fault_Y:   complex = 0j

        if USE_OPENDSS:
            self._init_opendss()
        else:
            self._build_network()
            self._build_ybus()

        print(f"  Backend: {'OpenDSS (opendssdirect)' if USE_OPENDSS else 'Built-in Newton-Raphson'}")

    # =========================================================================
    #  OPENDSS BACKEND
    # =========================================================================
    def _init_opendss(self):
        """Initialise the real OpenDSS engine from the DSS file set."""
        if not os.path.isfile(DSS_FILE):
            raise FileNotFoundError(
                f"DSS master file not found: {DSS_FILE}\n"
                f"Set DSS_FILE at the top of opendss_13bus_network.py to the correct path.")

        dss.run_command(f"Redirect {DSS_FILE}")
        if not dss.Solution.Converged():
            raise RuntimeError("OpenDSS initial power flow did not converge.")

        # Build a minimal bus dict for frequency model (DG dispatch only)
        self.buses = self._make_dg_bus_dict()
        self.pcc_bus = 'bus_634'

    def _make_dg_bus_dict(self):
        """Minimal bus dict keeping only DG entries (for frequency tracking)."""
        return {
            'bus_680': Bus('bus_680', P_gen_mw=0.36),
            'bus_611': Bus('bus_611', P_mw=0.170, Q_mvar=0.080, P_gen_mw=0.0),
            'bus_652': Bus('bus_652', P_mw=0.128, Q_mvar=0.086, P_gen_mw=0.0),
        }

    def _opendss_apply_fault(self, bus_name: str, z_fault_pu: float):
        # Convert pu impedance to ohms on system base
        Z_base = self.base_kv ** 2 / self.base_mva
        R_ohm  = z_fault_pu * Z_base
        dss.run_command(f"New Fault.F1 Bus1={bus_name} Phases=3 R={R_ohm:.6f} enabled=yes")
        self._freq_dev = -1.2

    def _opendss_clear_fault(self, bus_name: str):
        dss.run_command("Fault.F1.enabled=no")

    def _opendss_solve(self, t: float, P_dc_mw: float,
                       Q_dc_mvar: float, dt: float) -> NetworkResults:
        """Run one OpenDSS power-flow step."""
        # Update DG dispatch and feeder loads
        self._update_dg(t)

        # Push datacenter P/Q into OpenDSS load object
        dss.run_command(f"Load.datacenter.kW={P_dc_mw * 1000:.3f}")
        dss.run_command(f"Load.datacenter.kvar={Q_dc_mvar * 1000:.3f}")

        # Solve
        dss.Solution.Solve()
        self._update_frequency(P_dc_mw, dt)

        # ── Extract bus voltages ──────────────────────────────────────────────
        # VMagAngle() returns [|V|_ph1, ang_ph1, |V|_ph2, ...] in volts and degrees
        # Divide by (kV_LL * 1000 / sqrt(3)) to get per-unit line-to-neutral
        def V_pu_ln(bus_name_dss: str, kv_ll: float) -> float:
            dss.Circuit.SetActiveBus(bus_name_dss)
            V_ln_base = kv_ll * 1000.0 / 3.0 ** 0.5
            mags = dss.Bus.VMagAngle()[0::2]   # magnitudes only
            return float(np.mean(mags) / V_ln_base) if mags else 1.0

        def theta_rad(bus_name_dss: str) -> float:
            dss.Circuit.SetActiveBus(bus_name_dss)
            angs = dss.Bus.VMagAngle()[1::2]   # angles in degrees
            return float(np.radians(np.mean(angs))) if angs else 0.0

        # bus_634 is on 0.48 kV secondary of the step-down transformer
        V_pcc  = V_pu_ln('634',  0.48)
        V_632  = V_pu_ln('632',  4.16)
        V_680  = V_pu_ln('680',  4.16)
        th_pcc = theta_rad('634')

        # ── Losses and total load ─────────────────────────────────────────────
        # Circuit.Losses() → [P_losses_W, Q_losses_VAR]
        losses_mw  = abs(dss.Circuit.Losses()[0]) / 1e6

        # Circuit.TotalPower() → [P_kW, Q_kVAR] — includes all sources (negative = load)
        tp = dss.Circuit.TotalPower()
        tot_load_mw = abs(tp[0]) / 1000.0   # kW → MW

        tot_gen = sum(b.P_gen_mw for b in self.buses.values())

        return NetworkResults(
            V_pcc_pu        = V_pcc,
            theta_pcc       = th_pcc,
            V_bus632_pu     = V_632,
            V_bus680_pu     = V_680,
            freq_hz         = self.freq_hz,
            P_total_load_mw = tot_load_mw,
            P_losses_mw     = losses_mw,
            P_gen_total_mw  = tot_gen,
            converged       = bool(dss.Solution.Converged()),
            iterations      = int(dss.Solution.Iterations()),
        )

    # =========================================================================
    #  BUILT-IN NEWTON-RAPHSON BACKEND  (unchanged from original)
    # =========================================================================
    def _build_network(self):
        self.buses: Dict[str, Bus] = {
            # Slack (substation) at 1.0625 pu — matches the EPRI IEEE 13-bus
            # specification where the 115/4.16 kV transformer tap is set to
            # +6.25 % to compensate for feeder voltage drop (EPRI TR-1000419).
            'bus_sub':  Bus('bus_sub',  is_slack=True, V_pu=1.0625),
            'bus_650':  Bus('bus_650'),
            'bus_632':  Bus('bus_632',  P_mw=0.060,  Q_mvar=0.025),
            'bus_633':  Bus('bus_633',  P_mw=0.170,  Q_mvar=0.080),
            'bus_634':  Bus('bus_634',  P_mw=0.160,  Q_mvar=0.110),
            'bus_645':  Bus('bus_645',  P_mw=0.170,  Q_mvar=0.125),
            'bus_646':  Bus('bus_646',  P_mw=0.230,  Q_mvar=0.132),
            'bus_671':  Bus('bus_671',  P_mw=1.155,  Q_mvar=0.660),
            'bus_692':  Bus('bus_692',  P_mw=0.170,  Q_mvar=0.151),
            'bus_675':  Bus('bus_675',  P_mw=0.485,  Q_mvar=0.190),
            'bus_684':  Bus('bus_684'),
            'bus_680':  Bus('bus_680',  P_gen_mw=0.60),
            'bus_611':  Bus('bus_611',  P_mw=0.170, Q_mvar=0.080, P_gen_mw=0.25),
            'bus_652':  Bus('bus_652',  P_mw=0.128, Q_mvar=0.086, P_gen_mw=0.15),
        }
        self.branches: List[Branch] = [
            Branch('bus_sub',  'bus_650',  0.002,  0.008),
            # Voltage regulator (LTC) between bus_650 and bus_632.
            # EPRI spec: 32-step, ±10%, base-case tap ≈ +2.5 % (a=1.025).
            # Modelled as an off-nominal tap transformer (standard π-model).
            Branch('bus_650',  'bus_632',  0.038,  0.111, tap=1.025),
            Branch('bus_632',  'bus_671',  0.038,  0.111),
            Branch('bus_632',  'bus_633',  0.020,  0.057),
            Branch('bus_633',  'bus_634',  0.012,  0.022),
            Branch('bus_632',  'bus_645',  0.017,  0.049),
            Branch('bus_645',  'bus_646',  0.013,  0.037),
            Branch('bus_671',  'bus_680',  0.019,  0.056),
            Branch('bus_671',  'bus_692',  0.002,  0.005),
            Branch('bus_692',  'bus_675',  0.014,  0.040),
            Branch('bus_671',  'bus_684',  0.017,  0.049),
            Branch('bus_684',  'bus_611',  0.009,  0.025),
            Branch('bus_684',  'bus_652',  0.014,  0.034),
        ]
        self.bus_names = (
            [n for n, b in self.buses.items() if not b.is_slack] +
            [n for n, b in self.buses.items() if b.is_slack]
        )
        self.n_bus  = len(self.bus_names)
        self.n_pq   = self.n_bus - 1
        self.bus_idx = {nm: i for i, nm in enumerate(self.bus_names)}
        self.pcc_bus = 'bus_634'

    def _build_ybus(self):
        """
        Build Y-bus.  Branches with tap != 1.0 use the standard
        off-nominal transformer pi-model (Bergen & Vittal §6.3):
          Y_ii += y / a^2        (from_bus diagonal)
          Y_jj += y              (to_bus diagonal)
          Y_ij = Y_ji = -y / a   (off-diagonal)
        where a = br.tap (from_bus side), y = 1/Z_series.
        For a=1.0 this reduces to the ordinary line pi-model.
        """
        n = self.n_bus; self.Y = np.zeros((n, n), dtype=complex)
        for br in self.branches:
            i = self.bus_idx[br.from_bus]; j = self.bus_idx[br.to_bus]
            y  = 1.0 / complex(br.R_pu, br.X_pu)
            yb = 0.5j * br.B_pu
            a  = br.tap
            self.Y[i,i] += y / a**2 + yb
            self.Y[j,j] += y        + yb
            self.Y[i,j] -= y / a
            self.Y[j,i] -= y / a

    def _nr_build_injections(self, Pdc, Qdc):
        P = np.zeros(self.n_pq); Q = np.zeros(self.n_pq)
        for nm, b in self.buses.items():
            if b.is_slack: continue
            i = self.bus_idx[nm]
            if nm == self.pcc_bus:
                P[i] = -Pdc / self.base_mva; Q[i] = -Qdc / self.base_mva
            else:
                P[i] = (b.P_gen_mw  - b.P_mw)  / self.base_mva
                Q[i] = (b.Q_gen_mvar - b.Q_mvar) / self.base_mva
        return P, Q

    def _nr(self, Ps, Qs, maxiter=20, tol=1e-8):
        n=self.n_bus; npq=self.n_pq; G=self.Y.real; B=self.Y.imag
        Vm=np.array([self.buses[nm].V_pu  for nm in self.bus_names])
        th=np.array([self.buses[nm].theta for nm in self.bus_names])
        for it in range(maxiter):
            Pc=np.zeros(n); Qc=np.zeros(n)
            for i in range(n):
                for j in range(n):
                    a=th[i]-th[j]; ca=np.cos(a); sa=np.sin(a)
                    Pc[i]+=Vm[i]*Vm[j]*(G[i,j]*ca+B[i,j]*sa)
                    Qc[i]+=Vm[i]*Vm[j]*(G[i,j]*sa-B[i,j]*ca)
            dP=Ps-Pc[:npq]; dQ=Qs-Qc[:npq]
            if max(abs(dP).max(),abs(dQ).max())<tol: return Vm,th,True,it
            H=np.zeros((npq,npq)); N=np.zeros((npq,npq))
            M=np.zeros((npq,npq)); L=np.zeros((npq,npq))
            for i in range(npq):
                H[i,i]=-Qc[i]-B[i,i]*Vm[i]**2; N[i,i]=Pc[i]+G[i,i]*Vm[i]**2
                M[i,i]= Pc[i]-G[i,i]*Vm[i]**2; L[i,i]=Qc[i]-B[i,i]*Vm[i]**2
                for k in range(npq):
                    if k==i: continue
                    a=th[i]-th[k]; ca=np.cos(a); sa=np.sin(a)
                    H[i,k]=Vm[i]*Vm[k]*(G[i,k]*sa-B[i,k]*ca)
                    N[i,k]=Vm[i]*Vm[k]*(G[i,k]*ca+B[i,k]*sa)
                    M[i,k]=-N[i,k]; L[i,k]=H[i,k]
            try: dx=np.linalg.solve(np.block([[H,N],[M,L]]),np.r_[dP,dQ])
            except np.linalg.LinAlgError: break
            th[:npq]+=dx[:npq]
            Vm[:npq]*=np.clip(1.0+dx[npq:],0.50,2.00)  # relaxed: allows post-fault recovery
            Vm[:npq]=np.clip(Vm[:npq],0.50,1.50)       # absolute guard
        return Vm,th,False,maxiter

    def _nr_losses(self, Vm, th):
        l=0.0
        for br in self.branches:
            i=self.bus_idx[br.from_bus]; j=self.bus_idx[br.to_bus]
            I=(Vm[i]*np.exp(1j*th[i])-Vm[j]*np.exp(1j*th[j]))/complex(br.R_pu,br.X_pu)
            l+=abs(I)**2*br.R_pu
        return l*self.base_mva

    def _nr_solve(self, t, P_dc_mw, Q_dc_mvar, dt):
        self._update_dg(t)
        Ps, Qs = self._nr_build_injections(P_dc_mw, Q_dc_mvar)
        Vm, th, conv, it = self._nr(Ps, Qs)
        self._update_frequency(P_dc_mw, dt)
        for nm in self.bus_names:
            idx=self.bus_idx[nm]
            self.buses[nm].V_pu=float(Vm[idx]); self.buses[nm].theta=float(th[idx])
        pcc=self.bus_idx[self.pcc_bus]
        b632=self.bus_idx['bus_632']; b680=self.bus_idx['bus_680']
        return NetworkResults(
            V_pcc_pu        = float(Vm[pcc]),
            theta_pcc       = float(th[pcc]),
            V_bus632_pu     = float(Vm[b632]),
            V_bus680_pu     = float(Vm[b680]),
            freq_hz         = self.freq_hz,
            P_total_load_mw = float(sum(b.P_mw for b in self.buses.values())+P_dc_mw),
            P_losses_mw     = float(self._nr_losses(Vm, th)),
            P_gen_total_mw  = float(sum(b.P_gen_mw for b in self.buses.values())),
            converged       = conv,
            iterations      = it,
        )

    # =========================================================================
    #  SHARED METHODS  (identical behaviour for both backends)
    # =========================================================================
    def _update_dg(self, t: float):
        """Update DG dispatch. Pushes to OpenDSS generator objects when live."""
        wind_mw  = 0.60 * (0.60 + 0.30 * np.sin(2 * np.pi * t / 180.0))
        solar_pu = max(0.0, 1.0 - abs(t - 150.0) / 150.0)

        self.buses['bus_680'].P_gen_mw = wind_mw
        self.buses['bus_611'].P_gen_mw = 0.25 * solar_pu
        self.buses['bus_652'].P_gen_mw = 0.15 * solar_pu

        if USE_OPENDSS:
            dss.run_command(f"Generator.DG1_wind.kW={wind_mw * 1000:.2f}")
            dss.run_command(f"Generator.DG2_solar.kW={0.25 * solar_pu * 1000:.2f}")
            dss.run_command(f"Generator.DG3_solar.kW={0.15 * solar_pu * 1000:.2f}")

    def _update_frequency(self, P_dc_mw: float, dt: float):
        """
        Distribution feeder connected to infinite-bus grid.
        Frequency holds 60 Hz at steady state; deviates only from transients
        (fault injection, DG dispatch steps).  AGC restores with τ = 4 s.
        Supports large dt (OPF intervals) by sub-stepping if dt > 10 s.
        """
        # Sub-step for large intervals (OPF study uses dt=300 s)
        dt_sub  = min(max(dt, 1e-6), 5.0)
        n_sub   = max(1, int(np.ceil(dt / dt_sub)))
        dt_sub  = dt / n_sub

        P_dg_now  = sum(b.P_gen_mw for b in self.buses.values())
        dP_dg     = P_dg_now - getattr(self, '_P_dg_prev', P_dg_now)
        self._P_dg_prev = P_dg_now
        dg_pu = dP_dg / self.base_mva

        for _ in range(n_sub):
            d_dev = (60.0 / (2.0 * self._H_sys)) * (
                dg_pu - self._D_sys * self._freq_dev / 60.0) * dt_sub
            self._freq_dev = self._freq_dev * np.exp(-dt_sub / 4.0) + d_dev
            self._freq_dev = float(np.clip(self._freq_dev, -2.0, 2.0))

        self.freq_hz = float(np.clip(60.0 + self._freq_dev, 58.5, 61.5))

    # ── Public API ────────────────────────────────────────────────────────────
    def apply_fault(self, bus_name: str, z_fault_pu: float = 0.08):
        self._fault_bus = bus_name
        if USE_OPENDSS:
            self._opendss_apply_fault(bus_name, z_fault_pu)
        else:
            self._fault_Y = 1.0 / complex(z_fault_pu, z_fault_pu)
            self.Y[self.bus_idx[bus_name], self.bus_idx[bus_name]] += self._fault_Y
        self._freq_dev = -1.2

    def clear_fault(self, bus_name: str):
        if self._fault_bus == bus_name:
            if USE_OPENDSS:
                self._opendss_clear_fault(bus_name)
            else:
                self.Y[self.bus_idx[bus_name], self.bus_idx[bus_name]] -= self._fault_Y
                self._fault_Y = 0j
                # Reset all non-slack buses to a warm flat-start so the NR
                # does not inherit the fault-depressed voltages (0.75 pu).
                # Without this, the 10%-per-iteration step limiter needs 3+
                # iterations to climb from 0.75 → 0.95 pu, but the NR runs
                # out of iterations (maxiter=20 is plenty, but the ×1.10 cap
                # per iteration means it converges to 1.25 pu, not the real
                # solution, from such a poor starting point).
                for b in self.buses.values():
                    if not b.is_slack:
                        b.V_pu = 1.0
                        b.theta = 0.0
            self._fault_bus = None

    def solve(self, t: float, P_dc_mw: float,
              Q_dc_mvar: float, dt: float = 0.1) -> NetworkResults:
        if USE_OPENDSS:
            return self._opendss_solve(t, P_dc_mw, Q_dc_mvar, dt)
        else:
            return self._nr_solve(t, P_dc_mw, Q_dc_mvar, dt)

    def summary(self) -> str:
        backend = "opendssdirect (real OpenDSS)" if USE_OPENDSS else "built-in Newton-Raphson"
        dss_file = DSS_FILE if USE_OPENDSS else "N/A"
        lines = [
            "=" * 72,
            "  OpenDSS Network Simulator — IEEE 13-Bus Distribution Feeder",
            "=" * 72,
            f"  Backend      :  {backend}",
            f"  DSS file     :  {dss_file}",
            f"  Base         :  {self.base_mva} MVA  /  {self.base_kv} kV",
            f"  DGs          :  wind@bus_680 (0.6 MW)  solar@bus_611 (0.25 MW)  solar@bus_652 (0.15 MW)",
            f"  PCC          :  bus_634  (datacenter, 0.48 kV secondary)",
            "=" * 72,
        ]
        return "\n".join(lines)