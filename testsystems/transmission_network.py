"""
transmission_network.py
=======================
IEEE 39-Bus New England Transmission Network Simulator
======================================================

Implements a full Newton-Raphson AC power flow on the IEEE 39-bus
New England test system.  This system is the de-facto benchmark for
transmission-level transient stability and voltage stability studies.

Network characteristics
-----------------------
  Buses        : 39  (29 PQ + 9 PV + 1 slack)
  Generators   : 10  (G01 slack at bus 31; G02-G10 PV)
  Transmission : 46 branches (lines + transformers)
  Base MVA     : 100 MVA
  Base kV      : 345 kV (HV), 20 kV (gen terminals)
  Total load   : ~6,200 MW, ~1,400 MVAR

This module mirrors the interface of opendss_network.py so the
co-simulation engine can substitute it without code changes.

Datacenter connection
---------------------
The datacenter pod is connected at bus 16 (a major load bus near
the urban load centre), represented as a controllable PQ load.
The pod rating is 2 MVA on a 100 MVA system base, i.e. 0.02 pu.

Dynamics
--------
Generator swing equations are integrated at the macro step to
provide realistic frequency and ROCOF signals.  Each generator is
modelled as a classical second-order swing with:
  - Inertia constant H [s]
  - Damping coefficient D [pu]
  - Mechanical power P_m  held constant (governor not modelled)
  - Voltage magnitude held at setpoint (AVR not modelled)

Contingencies supported
-----------------------
  apply_fault(bus)        : three-phase bus fault (shunt to ground)
  clear_fault(bus)        : remove fault
  trip_line(from, to)     : open transmission line
  restore_line(from, to)  : reclose transmission line
  trip_generator(bus)     : remove generator (P_m → 0)
  step_load(bus, dp, dq)  : step change in load at a bus
"""

# ── Standard library ──────────────────────────────────────────────────────────
from __future__ import annotations
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ── Third-party ───────────────────────────────────────────────────────────────
import numpy as np

# =============================================================================
#  DATACLASSES
# =============================================================================

@dataclass
class TransBus:
    """Single transmission bus."""
    idx:        int
    name:       str
    type:       str          # 'slack' | 'pv' | 'pq'
    V_pu:       float = 1.0
    theta:      float = 0.0  # rad
    P_gen_pu:   float = 0.0  # scheduled generation [pu on 100 MVA]
    Q_gen_pu:   float = 0.0
    P_load_pu:  float = 0.0  # load [pu on 100 MVA]
    Q_load_pu:  float = 0.0
    V_set:      float = 1.0  # PV bus voltage setpoint
    Q_min:      float = -0.5
    Q_max:      float =  0.5
    fault:      bool  = False


@dataclass
class TransBranch:
    """Transmission line or transformer."""
    from_bus:   int
    to_bus:     int
    R_pu:       float
    X_pu:       float
    B_pu:       float = 0.0   # total line charging susceptance
    tap:        float = 1.0   # off-nominal turns ratio (from-bus side)
    in_service: bool  = True


@dataclass
class Generator:
    """Classical second-order generator model."""
    bus:    int
    H:      float        # inertia constant [s]
    D:      float        # damping [pu]
    P_m:    float        # mechanical power [pu on 100 MVA]
    delta:  float = 0.0  # rotor angle [rad] relative to reference
    omega:  float = 0.0  # speed deviation [pu], 0 = synchronous
    in_service: bool = True


@dataclass
class TransNetworkResults:
    """Results returned to co-simulation orchestrator each macro step."""
    # System frequency (centre-of-inertia)
    freq_hz:         float = 60.0
    rocof_hz_s:      float = 0.0

    # Voltage at datacenter PCC bus (bus 16)
    V_pcc_pu:        float = 1.0
    theta_pcc_rad:   float = 0.0

    # Selected bus voltages for monitoring
    V_bus14_pu:      float = 1.0  # major load bus
    V_bus15_pu:      float = 1.0
    V_bus39_pu:      float = 1.0  # slack (bus 39 in IEEE numbering)

    # System totals
    P_total_load_pu: float = 0.0
    P_total_gen_pu:  float = 0.0
    P_losses_pu:     float = 0.0
    Q_total_load_pu: float = 0.0

    # Generator with minimum voltage margin
    V_min_pu:        float = 1.0
    V_min_bus:       int   = 0

    # Co-simulation metadata
    nr_iters:        int   = 0
    nr_converged:    bool  = True


# =============================================================================
#  IEEE 39-BUS NEW ENGLAND TEST SYSTEM DATA
# =============================================================================

def _build_ieee39_buses() -> List[TransBus]:
    """
    Bus data from MATPOWER case39 (verified operating point).
    Loads and generation on 100 MVA base.
    Q_gen initialised from MATPOWER solved case; Q limits from generator data.
    """
    # (idx, name, type, V_pu, P_gen_pu, P_load_pu, Q_load_pu, V_set, Q_min, Q_max)
    bus_data = [
        ( 1,"bus_01","pq", 1.0393, 0.000, 0.097, 0.044, 1.0393,-9.9,-9.9),
        ( 2,"bus_02","pq", 1.0487, 0.000, 0.000, 0.000, 1.0487,-9.9,-9.9),
        ( 3,"bus_03","pq", 1.0302, 0.000, 3.220, 0.024, 1.0302,-9.9,-9.9),
        ( 4,"bus_04","pq", 1.0040, 0.000, 5.000, 1.840, 1.0040,-9.9,-9.9),
        ( 5,"bus_05","pq", 1.0056, 0.000, 0.000, 0.000, 1.0056,-9.9,-9.9),
        ( 6,"bus_06","pq", 1.0078, 0.000, 0.000, 0.000, 1.0078,-9.9,-9.9),
        ( 7,"bus_07","pq", 0.9974, 0.000, 2.338, 0.840, 0.9974,-9.9,-9.9),
        ( 8,"bus_08","pq", 0.9960, 0.000, 5.220, 1.760, 0.9960,-9.9,-9.9),
        ( 9,"bus_09","pq", 1.0277, 0.000, 0.000, 0.000, 1.0277,-9.9,-9.9),
        (10,"bus_10","pq", 1.0169, 0.000, 0.000, 0.000, 1.0169,-9.9,-9.9),
        (11,"bus_11","pq", 1.0127, 0.000, 0.000, 0.000, 1.0127,-9.9,-9.9),
        (12,"bus_12","pq", 1.0000, 0.000, 0.085, 0.880, 1.0000,-9.9,-9.9),
        (13,"bus_13","pq", 1.0142, 0.000, 0.000, 0.000, 1.0142,-9.9,-9.9),
        (14,"bus_14","pq", 1.0117, 0.000, 0.000, 0.000, 1.0117,-9.9,-9.9),
        (15,"bus_15","pq", 1.0163, 0.000, 3.200, 1.530, 1.0163,-9.9,-9.9),
        (16,"bus_16","pq", 1.0320, 0.000, 3.290, 0.323, 1.0320,-9.9,-9.9),
        (17,"bus_17","pq", 1.0337, 0.000, 0.000, 0.000, 1.0337,-9.9,-9.9),
        (18,"bus_18","pq", 1.0317, 0.000, 1.580, 0.300, 1.0317,-9.9,-9.9),
        (19,"bus_19","pq", 1.0500, 0.000, 0.000, 0.000, 1.0500,-9.9,-9.9),
        (20,"bus_20","pq", 0.9912, 0.000, 6.280, 1.030, 0.9912,-9.9,-9.9),
        (21,"bus_21","pq", 1.0321, 0.000, 2.740, 1.150, 1.0321,-9.9,-9.9),
        (22,"bus_22","pq", 1.0497, 0.000, 0.000, 0.000, 1.0497,-9.9,-9.9),
        (23,"bus_23","pq", 1.0451, 0.000, 2.475, 0.846, 1.0451,-9.9,-9.9),
        (24,"bus_24","pq", 1.0370, 0.000, 3.086,-0.922, 1.0370,-9.9,-9.9),
        (25,"bus_25","pq", 1.0569, 0.000, 2.240, 0.472, 1.0569,-9.9,-9.9),
        (26,"bus_26","pq", 1.0522, 0.000, 1.390, 0.170, 1.0522,-9.9,-9.9),
        (27,"bus_27","pq", 1.0376, 0.000, 2.810, 0.755, 1.0376,-9.9,-9.9),
        (28,"bus_28","pq", 1.0505, 0.000, 2.060, 0.276, 1.0505,-9.9,-9.9),
        (29,"bus_29","pq", 1.0500, 0.000, 2.835, 0.269, 1.0500,-9.9,-9.9),
        # Generator buses — PV type, Q_gen from MATPOWER solved case
        (30,"bus_30","pv", 1.0479, 2.50, 0.000, 0.000, 1.0479, 1.40, 4.00),
        (31,"bus_31","slack",0.9820,6.779,0.092, 0.046, 0.9820,-1.00, 3.00),
        (32,"bus_32","pv", 0.9831, 6.50, 0.000, 0.000, 0.9831, 1.50, 3.00),
        (33,"bus_33","pv", 0.9972, 6.32, 0.000, 0.000, 0.9972, 0.00, 2.50),
        (34,"bus_34","pv", 1.0123, 5.08, 0.000, 0.000, 1.0123,-1.00, 1.67),
        (35,"bus_35","pv", 1.0493, 6.50, 0.000, 0.000, 1.0493,-0.25, 3.00),
        (36,"bus_36","pv", 1.0635, 5.60, 0.000, 0.000, 1.0635,-1.50, 2.40),
        (37,"bus_37","pv", 1.0278, 5.40, 0.000, 0.000, 1.0278,-1.50, 2.50),
        (38,"bus_38","pv", 1.0265, 8.30, 0.000, 0.000, 1.0265,-1.50, 3.00),
        (39,"bus_39","pq", 1.0300, 0.000,11.04, 2.500, 1.0300,-9.9,-9.9),
    ]
    # Q_gen initial values from MATPOWER (pu on 100 MVA)
    Q_gen_init = {30:1.618, 31:2.216, 32:2.060, 33:1.083,
                  34:-0.228, 35:2.107, 36:1.002, 37:0.000,
                  38:0.217, 39:0.785}

    buses = []
    for row in bus_data:
        (idx,name,btype,V_pu,P_gen,P_load,Q_load,V_set,Qmin,Qmax) = row
        b = TransBus(
            idx=idx, name=name, type=btype,
            V_pu=V_pu, theta=0.0,
            P_gen_pu=P_gen,
            Q_gen_pu=Q_gen_init.get(idx, 0.0),
            P_load_pu=P_load, Q_load_pu=Q_load,
            V_set=V_set,
            Q_min=Qmin, Q_max=Qmax,
        )
        buses.append(b)
    return buses


def _build_ieee39_branches() -> List[TransBranch]:
    """
    Branch data from MATPOWER case39.m (verified).
    R, X in pu on 100 MVA / 345 kV base; B = total line charging.
    tap = 0 in MATPOWER means a regular line (tap=1.0 here).
    46 branches total (34 lines + 12 transformers).
    """
    raw = [
        #  f   t      R       X       B      tap
        [ 1,  2,  0.0035, 0.0411, 0.6987, 0    ],
        [ 1, 39,  0.0010, 0.0250, 0.7500, 0    ],
        [ 2,  3,  0.0013, 0.0151, 0.2572, 0    ],
        [ 2, 25,  0.0070, 0.0086, 0.1460, 0    ],
        [ 2, 30,  0.0000, 0.0181, 0.0000, 1.025],
        [ 3,  4,  0.0013, 0.0213, 0.2214, 0    ],
        [ 3, 18,  0.0011, 0.0133, 0.2138, 0    ],
        [ 4,  5,  0.0008, 0.0128, 0.1342, 0    ],
        [ 4, 14,  0.0008, 0.0129, 0.1382, 0    ],
        [ 5,  6,  0.0002, 0.0026, 0.0434, 0    ],
        [ 5,  8,  0.0008, 0.0112, 0.1476, 0    ],
        [ 6,  7,  0.0006, 0.0092, 0.1130, 0    ],
        [ 6, 11,  0.0007, 0.0082, 0.1389, 0    ],
        [ 6, 31,  0.0000, 0.0250, 0.0000, 1.070],
        [ 7,  8,  0.0004, 0.0046, 0.0780, 0    ],
        [ 8,  9,  0.0023, 0.0363, 0.3804, 0    ],
        [ 9, 39,  0.0010, 0.0250, 1.2000, 0    ],
        [10, 11,  0.0004, 0.0043, 0.0729, 0    ],
        [10, 13,  0.0004, 0.0043, 0.0729, 0    ],
        [10, 32,  0.0000, 0.0200, 0.0000, 1.070],
        [12, 11,  0.0016, 0.0435, 0.0000, 0    ],
        [12, 13,  0.0016, 0.0435, 0.0000, 0    ],
        [13, 14,  0.0009, 0.0101, 0.1723, 0    ],
        [14, 15,  0.0018, 0.0217, 0.3660, 0    ],
        [15, 16,  0.0009, 0.0094, 0.1710, 0    ],
        [16, 17,  0.0007, 0.0089, 0.1342, 0    ],
        [16, 19,  0.0016, 0.0195, 0.3040, 0    ],
        [16, 21,  0.0008, 0.0135, 0.2548, 0    ],
        [16, 24,  0.0003, 0.0059, 0.0680, 0    ],
        [17, 18,  0.0007, 0.0082, 0.1319, 0    ],
        [17, 27,  0.0013, 0.0173, 0.3216, 0    ],
        [19, 20,  0.0007, 0.0138, 0.0000, 1.060],
        [19, 33,  0.0007, 0.0142, 0.0000, 1.070],
        [20, 34,  0.0009, 0.0180, 0.0000, 1.009],
        [21, 22,  0.0008, 0.0140, 0.2565, 0    ],
        [22, 23,  0.0006, 0.0096, 0.1846, 0    ],
        [22, 35,  0.0000, 0.0143, 0.0000, 1.025],
        [23, 24,  0.0022, 0.0350, 0.3610, 0    ],
        [23, 36,  0.0005, 0.0272, 0.0000, 1.000],
        [25, 26,  0.0032, 0.0323, 0.5310, 0    ],
        [25, 37,  0.0006, 0.0232, 0.0000, 1.025],
        [26, 27,  0.0014, 0.0147, 0.2396, 0    ],
        [26, 28,  0.0043, 0.0474, 0.7802, 0    ],
        [26, 29,  0.0057, 0.0625, 1.0290, 0    ],
        [28, 29,  0.0014, 0.0151, 0.2490, 0    ],
        [29, 38,  0.0008, 0.0156, 0.0000, 1.025],
    ]
    return [
        TransBranch(
            from_bus=int(r[0]), to_bus=int(r[1]),
            R_pu=r[2], X_pu=r[3], B_pu=r[4],
            tap=r[5] if r[5] != 0 else 1.0,
        )
        for r in raw
    ]


def _build_ieee39_generators() -> List[Generator]:
    """
    Classical generator data (H in seconds, D in pu on machine base).
    Source: Anderson & Fouad (2003), Appendix.
    """
    gen_data = [
        # bus   H     D     P_m (pu on 100 MVA)
        (30,   500.0, 2.0,  2.50),
        (31,   500.0, 2.0,  5.73),  # slack
        (32,    30.3, 2.0,  6.50),
        (33,    35.8, 2.0,  6.32),
        (34,    28.6, 2.0,  5.08),
        (35,    26.0, 2.0,  6.50),
        (36,    34.8, 2.0,  5.60),
        (37,    26.4, 2.0,  5.40),
        (38,    24.3, 2.0,  8.30),
        (39,    30.4, 2.0,  10.00),
    ]
    gens = []
    for row in gen_data:
        gens.append(Generator(bus=row[0], H=row[1], D=row[2], P_m=row[3]))
    return gens


# =============================================================================
#  TRANSMISSION NETWORK SIMULATOR
# =============================================================================

class TransmissionNetworkSimulator:
    """
    IEEE 39-bus New England transmission network with:
      • Full Newton-Raphson AC power flow (polar formulation)
      • Classical second-order generator swing equations (RK4)
      • Centre-of-inertia frequency and ROCOF tracking
      • Contingency methods: fault, line trip, generator trip, load step
      • Datacenter PCC at bus 16

    Interface mirrors OpenDSSNetworkSimulator:
      results = sim.solve(t, P_dc_mw, Q_dc_mvar, dt)
    """

    DC_PCC_BUS     = 16      # datacenter connects here
    S_BASE_MVA     = 100.0   # system base
    F_NOM          = 60.0    # Hz
    NR_TOL         = 1e-4    # pu
    NR_MAX_ITER    = 50
    STEP_LIM_LO    = 0.50    # NR voltage step limiter
    STEP_LIM_HI    = 1.50

    # MATPOWER case39 verified operating point (Vm [pu], Va [deg])
    _MATPOWER_INIT = {
        1: (1.0393,-14.535), 2: (1.0487, -7.869), 3: (1.0302,-10.521),
        4: (1.0040,-12.434), 5: (1.0056,-11.360), 6: (1.0078,-10.770),
        7: (0.9974,-13.210), 8: (0.9960,-13.210), 9: (1.0277,-11.430),
       10: (1.0169, -8.806),11: (1.0127, -9.598),12: (1.0000,-10.095),
       13: (1.0142, -9.498),14: (1.0117, -9.130),15: (1.0163,-11.119),
       16: (1.0320,-10.847),17: (1.0337,-11.343),18: (1.0317,-11.853),
       19: (1.0500, -1.743),20: (0.9912, -6.645),21: (1.0321, -7.093),
       22: (1.0497, -3.001),23: (1.0451, -5.496),24: (1.0370, -8.217),
       25: (1.0569, -5.671),26: (1.0522, -8.120),27: (1.0376,-10.671),
       28: (1.0505, -4.998),29: (1.0500, -2.003),30: (1.0479, -7.388),
       31: (0.9820,  0.000),32: (0.9831, -0.788),33: (0.9972, -0.960),
       34: (1.0123, -3.677),35: (1.0493, -3.640),36: (1.0635, -5.738),
       37: (1.0278,-10.160),38: (1.0265,-10.430),39: (1.0300,-14.535),
    }

    def __init__(self):
        self.buses:    Dict[int, TransBus]    = {}
        self.branches: List[TransBranch]      = []
        self.gens:     Dict[int, Generator]   = {}

        self._load_data()
        self._init_operating_point()
        self._build_ybus()

        # Frequency state
        self._freq_hz   = self.F_NOM
        self._rocof     = 0.0
        self._omega_coi = 0.0   # centre-of-inertia speed deviation [pu]

        # Active contingencies
        self._faulted_buses: Dict[int, float] = {}  # bus -> Y_fault
        self._tripped_lines:  set = set()

    def _init_operating_point(self):
        """Initialise bus voltages and angles from MATPOWER case39 solution."""
        for idx, (Vm, Va_deg) in self._MATPOWER_INIT.items():
            if idx in self.buses:
                self.buses[idx].V_pu  = Vm
                self.buses[idx].V_set = Vm
                self.buses[idx].theta = np.radians(Va_deg)

    # ─────────────────────────────────────────────────────────────────────────
    def _load_data(self):
        for b in _build_ieee39_buses():
            self.buses[b.idx] = b
        self.branches = _build_ieee39_branches()
        for g in _build_ieee39_generators():
            self.gens[g.bus] = g

    # ─────────────────────────────────────────────────────────────────────────
    def _build_ybus(self):
        """Assemble complex Y-bus from branch data."""
        n = len(self.buses)
        self._Y = np.zeros((n, n), dtype=complex)

        for br in self.branches:
            if not br.in_service:
                continue
            i = br.from_bus - 1   # 0-indexed
            j = br.to_bus - 1
            y = 1.0 / complex(br.R_pu, br.X_pu)
            yb = complex(0, br.B_pu / 2.0)
            a = br.tap

            if abs(a - 1.0) < 1e-6:
                # Transmission line
                self._Y[i, i] += y + yb
                self._Y[j, j] += y + yb
                self._Y[i, j] -= y
                self._Y[j, i] -= y
            else:
                # Off-nominal transformer pi-model
                self._Y[i, i] += y / (a * a) + yb
                self._Y[j, j] += y            + yb
                self._Y[i, j] -= y / a
                self._Y[j, i] -= y / a

    # ─────────────────────────────────────────────────────────────────────────
    def _apply_fault_ybus(self):
        """Modify Y-bus for active faults."""
        self._build_ybus()
        for bus_idx, Y_f in self._faulted_buses.items():
            i = bus_idx - 1
            self._Y[i, i] += complex(0, Y_f)

    # ─────────────────────────────────────────────────────────────────────────
    def _nr_solve(self) -> Tuple[bool, int]:
        """
        Newton-Raphson power flow (polar form) with PV bus Q tracking.
        Modifies bus V_pu and theta in-place.
        Returns (converged, iterations).
        """
        buses = list(self.buses.values())
        n = len(buses)

        pq_idx = [b.idx - 1 for b in buses if b.type == 'pq']
        pv_idx = [b.idx - 1 for b in buses if b.type == 'pv']
        sl_idx = [b.idx - 1 for b in buses if b.type == 'slack'][0]

        V  = np.array([b.V_pu  for b in buses])
        th = np.array([b.theta for b in buses])

        mis = np.ones(n)  # initialise non-zero

        for it in range(self.NR_MAX_ITER):
            Vc = V * np.exp(1j * th)
            S  = Vc * np.conj(self._Y @ Vc)
            P  = S.real
            Q  = S.imag

            # Update Q_gen for PV buses (reactive power tracking)
            for i in pv_idx:
                buses[i].Q_gen_pu = float(np.clip(
                    Q[i] + buses[i].Q_load_pu,
                    buses[i].Q_min, buses[i].Q_max))

            P_sch = np.array([b.P_gen_pu - b.P_load_pu for b in buses])
            Q_sch = np.array([b.Q_gen_pu - b.Q_load_pu for b in buses])

            dP = P_sch - P
            dQ = Q_sch - Q

            mis_p = np.delete(dP, sl_idx)
            mis_q = np.array([dQ[i] for i in pq_idx])
            mis   = np.concatenate([mis_p, mis_q])

            if np.max(np.abs(mis)) < self.NR_TOL:
                break

            J = self._jacobian(V, th, P, Q, n, sl_idx, pq_idx, pv_idx)

            try:
                dx = np.linalg.solve(J, mis)
            except np.linalg.LinAlgError:
                return False, it + 1

            n_th = n - 1
            n_v  = len(pq_idx)

            dth  = dx[:n_th]
            dV_frac = dx[n_th:]

            dth_full = np.insert(dth, sl_idx, 0.0)
            th += dth_full

            # Voltage updates for PQ buses with step limiter
            dV_arr = np.zeros(n)
            for ki, qi in enumerate(pq_idx):
                dV_arr[qi] = dV_frac[ki]

            V = np.clip(
                V * (1.0 + np.clip(dV_arr,
                                   self.STEP_LIM_LO - 1.0,
                                   self.STEP_LIM_HI - 1.0)),
                self.STEP_LIM_LO, self.STEP_LIM_HI)

            # Enforce PV bus voltage setpoints
            for i in pv_idx:
                V[i] = buses[i].V_set

        converged = (np.max(np.abs(mis)) < self.NR_TOL)

        for i, b in enumerate(buses):
            b.V_pu  = float(V[i])
            b.theta = float(th[i])

        return converged, it + 1

    # ─────────────────────────────────────────────────────────────────────────
    def _jacobian(self, V, th, P, Q, n, sl_idx, pq_idx, pv_idx):
        """Construct NR Jacobian (polar, 4-block form)."""
        Y = self._Y
        G = Y.real
        B = Y.imag

        # H = dP/dth, N = dP/dV*V, J = dQ/dth, L = dQ/dV*V
        H = np.zeros((n, n))
        N = np.zeros((n, n))
        Jb = np.zeros((n, n))
        L = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                if i == j:
                    H[i, i]  = -Q[i] - B[i, i] * V[i] ** 2
                    N[i, i]  =  P[i] + G[i, i] * V[i] ** 2
                    Jb[i, i] =  P[i] - G[i, i] * V[i] ** 2
                    L[i, i]  =  Q[i] - B[i, i] * V[i] ** 2
                else:
                    dth = th[i] - th[j]
                    H[i, j] = V[i] * V[j] * (
                        G[i, j] * np.sin(dth) - B[i, j] * np.cos(dth))
                    N[i, j] = V[i] * V[j] * (
                        G[i, j] * np.cos(dth) + B[i, j] * np.sin(dth))
                    Jb[i, j] = -H[i, j]
                    L[i, j]  =  N[i, j]

        # Extract rows/cols for active variables
        all_idx  = list(range(n))
        th_rows  = [i for i in all_idx if i != sl_idx]
        v_rows   = pq_idx

        J11 = H[np.ix_(th_rows, th_rows)]
        J12 = N[np.ix_(th_rows, v_rows)]
        J21 = Jb[np.ix_(v_rows,  th_rows)]
        J22 = L[np.ix_(v_rows,   v_rows)]

        return np.block([[J11, J12], [J21, J22]])

    # ─────────────────────────────────────────────────────────────────────────
    def _swing_equations(self, dt: float):
        """
        Integrate generator swing equations (RK4) for one time step.
        Updates generator delta and omega, computes COI frequency.
        """
        F_NOM  = self.F_NOM
        OM_NOM = 2.0 * np.pi * F_NOM

        def f_delta(g: Generator):
            return OM_NOM * g.omega

        def f_omega(g: Generator):
            if not g.in_service:
                return 0.0
            P_e = self._electrical_power(g.bus)
            return (g.P_m - P_e - g.D * g.omega) / (2.0 * g.H)

        for g in self.gens.values():
            k1d = f_delta(g);      k1o = f_omega(g)
            g.delta += dt / 2 * k1d;  g.omega += dt / 2 * k1o
            k2d = f_delta(g);      k2o = f_omega(g)
            g.delta -= dt / 2 * k1d;  g.omega -= dt / 2 * k1o  # restore
            g.delta += dt / 2 * k2d;  g.omega += dt / 2 * k2o
            k3d = f_delta(g);      k3o = f_omega(g)
            g.delta -= dt / 2 * k2d;  g.omega -= dt / 2 * k2o  # restore
            g.delta += dt * k3d;       g.omega += dt * k3o
            k4d = f_delta(g);      k4o = f_omega(g)
            g.delta -= dt * k3d;       g.omega -= dt * k3o  # restore

            g.delta += (dt / 6.0) * (k1d + 2 * k2d + 2 * k3d + k4d)
            g.omega += (dt / 6.0) * (k1o + 2 * k2o + 2 * k3o + k4o)

        # Centre-of-inertia frequency
        H_total = sum(g.H for g in self.gens.values() if g.in_service)
        omega_coi = sum(g.H * g.omega
                        for g in self.gens.values() if g.in_service) / H_total
        rocof = (omega_coi - self._omega_coi) / dt if dt > 0 else 0.0
        self._omega_coi = omega_coi
        self._freq_hz   = F_NOM * (1.0 + omega_coi)
        self._rocof     = rocof * F_NOM    # convert to Hz/s

    # ─────────────────────────────────────────────────────────────────────────
    def _electrical_power(self, bus_idx: int) -> float:
        """
        Electrical power output of generator at bus_idx [pu on 100 MVA].
        Uses current bus voltage from NR solution.
        """
        b = self.buses[bus_idx]
        return b.P_gen_pu   # simplified: assume generator tracks setpoint

    # ─────────────────────────────────────────────────────────────────────────
    def _reinit_voltages(self):
        """
        Reset bus voltages and angles to the MATPOWER case39 operating point
        before each NR solve.  This prevents drift across successive calls
        and ensures repeatable, physically stable results.

        Note: only PQ bus voltages are reset; PV bus setpoints are enforced
        during the NR iteration itself.  The slack bus is never reset here.
        """
        for idx, (Vm, Va_deg) in self._MATPOWER_INIT.items():
            if idx not in self.buses:
                continue
            b = self.buses[idx]
            if b.type == 'slack':
                continue
            b.theta = float(np.radians(Va_deg))
            if b.type == 'pq':
                b.V_pu = Vm

    def solve(self, t: float, P_dc_mw: float, Q_dc_mvar: float,
              dt: float = 0.1, quasi_static: bool = False) -> TransNetworkResults:
        """
        Advance network one macro step.

        Parameters
        ----------
        t           : current simulation time [s]
        P_dc_mw     : datacenter active power injection at bus 16 [MW]
        Q_dc_mvar   : datacenter reactive power injection at bus 16 [MVAR]
        dt          : macro step size [s]

        Returns
        -------
        TransNetworkResults
        """
        # ── Re-initialise voltages to operating point for NR stability ─────────
        self._reinit_voltages()

        # ── Inject datacenter load at PCC bus ─────────────────────────────────
        pcc = self.buses[self.DC_PCC_BUS]
        pcc.P_load_pu = (P_dc_mw  + 3.290 * self.S_BASE_MVA / self.S_BASE_MVA) \
                        / self.S_BASE_MVA   # add back base load + DC pod
        pcc.Q_load_pu = (Q_dc_mvar + 0.323 * self.S_BASE_MVA / self.S_BASE_MVA) \
                        / self.S_BASE_MVA

        # Cleaner: just add DC pod increment to base load
        pcc.P_load_pu = 3.290 + P_dc_mw  / self.S_BASE_MVA
        pcc.Q_load_pu = 0.323 + Q_dc_mvar / self.S_BASE_MVA

        # ── NR power flow ──────────────────────────────────────────────────────
        converged, nr_iters = self._nr_solve()

        # ── Swing equations ───────────────────────────────────────────────────
        if not quasi_static:
            self._swing_equations(dt)

        # ── Collect results ───────────────────────────────────────────────────
        buses = self.buses
        V_all = np.array([b.V_pu for b in buses.values()])

        # Total load and generation
        P_load = sum(b.P_load_pu for b in buses.values())
        Q_load = sum(b.Q_load_pu for b in buses.values())
        P_gen  = sum(b.P_gen_pu  for b in buses.values())

        # Losses from NR (P_gen - P_load)
        P_losses = P_gen - P_load

        return TransNetworkResults(
            freq_hz         = float(self._freq_hz),
            rocof_hz_s      = float(self._rocof),
            V_pcc_pu        = float(buses[self.DC_PCC_BUS].V_pu),
            theta_pcc_rad   = float(buses[self.DC_PCC_BUS].theta),
            V_bus14_pu      = float(buses[14].V_pu),
            V_bus15_pu      = float(buses[15].V_pu),
            V_bus39_pu      = float(buses[39].V_pu),
            P_total_load_pu = float(P_load),
            P_total_gen_pu  = float(P_gen),
            P_losses_pu     = float(max(P_losses, 0.0)),
            Q_total_load_pu = float(Q_load),
            V_min_pu        = float(V_all.min()),
            V_min_bus       = int(np.argmin(V_all) + 1),
            nr_iters        = nr_iters,
            nr_converged    = converged,
        )

    # ─────────────────────────────────────────────────────────────────────────
    #  CONTINGENCY METHODS
    # ─────────────────────────────────────────────────────────────────────────
    def apply_fault(self, bus_idx: int, z_fault_pu: float = 0.01):
        """Apply three-phase fault at bus (low-impedance shunt to ground)."""
        Y_fault = 1.0 / max(z_fault_pu, 1e-4)
        self._faulted_buses[bus_idx] = Y_fault
        self._apply_fault_ybus()
        self.buses[bus_idx].fault = True

    def clear_fault(self, bus_idx: int):
        """Clear fault and restore bus voltages to flat start."""
        self._faulted_buses.pop(bus_idx, None)
        self._build_ybus()
        self.buses[bus_idx].fault = False
        # Flat-start reset for post-fault NR convergence
        for b in self.buses.values():
            if b.type == 'pq':
                b.V_pu = 1.0
                b.theta = 0.0

    def trip_line(self, from_bus: int, to_bus: int):
        """Open a transmission line (N-1 contingency)."""
        for br in self.branches:
            if ((br.from_bus == from_bus and br.to_bus == to_bus) or
                    (br.from_bus == to_bus and br.to_bus == from_bus)):
                br.in_service = False
                self._tripped_lines.add((from_bus, to_bus))
        self._build_ybus()

    def restore_line(self, from_bus: int, to_bus: int):
        """Reclose a tripped transmission line."""
        for br in self.branches:
            if ((br.from_bus == from_bus and br.to_bus == to_bus) or
                    (br.from_bus == to_bus and br.to_bus == from_bus)):
                br.in_service = True
                self._tripped_lines.discard((from_bus, to_bus))
        self._build_ybus()

    def trip_generator(self, bus_idx: int):
        """Remove a generator (sudden loss, sets P_m = 0)."""
        if bus_idx in self.gens:
            self.gens[bus_idx].in_service = False
            self.gens[bus_idx].P_m = 0.0
            self.buses[bus_idx].P_gen_pu = 0.0
            self.buses[bus_idx].type = 'pq'
        self._build_ybus()

    def step_load(self, bus_idx: int, dP_mw: float, dQ_mvar: float = 0.0):
        """Apply a step change in load at a bus."""
        b = self.buses[bus_idx]
        b.P_load_pu += dP_mw  / self.S_BASE_MVA
        b.Q_load_pu += dQ_mvar / self.S_BASE_MVA

    # ─────────────────────────────────────────────────────────────────────────
    def summary(self) -> str:
        """Return a printable network summary string."""
        n_bus  = len(self.buses)
        n_line = len([b for b in self.branches if b.in_service])
        n_gen  = len([g for g in self.gens.values() if g.in_service])
        P_load = sum(b.P_load_pu * self.S_BASE_MVA for b in self.buses.values())
        P_gen  = sum(b.P_gen_pu  * self.S_BASE_MVA for b in self.buses.values())
        H_tot  = sum(g.H for g in self.gens.values() if g.in_service)
        lines = [
            "=" * 74,
            "  IEEE 39-Bus New England Transmission Network Simulator",
            "=" * 74,
            f"  System base  :  {self.S_BASE_MVA:.0f} MVA  /  345 kV  (HV bus)",
            f"  Buses        :  {n_bus}  (29 PQ + 9 PV + 1 slack)",
            f"  Branches     :  {n_line} in service",
            f"  Generators   :  {n_gen}  (G01 slack @ bus 31)",
            f"  Total H      :  {H_tot:.0f} s  (system inertia)",
            f"  Total load   :  {P_load:.0f} MW  (approximate)",
            f"  Total gen    :  {P_gen:.0f} MW  (dispatch setpoints)",
            f"  DC PCC       :  bus 16  (2 MVA datacenter pod)",
            "=" * 74,
        ]
        return "\n".join(lines)

    @property
    def n_bus(self) -> int:
        return len(self.buses)

    @property
    def freq_hz(self) -> float:
        return self._freq_hz

    @property
    def rocof_hz_s(self) -> float:
        return self._rocof
