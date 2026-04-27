# GOALS: A Grid-aware Open source AI-Datacenter Load Simulator for Impact Analysis
**A modular, open-source co-simulation framework for quantifying the grid impact of AI datacenters on power systems.**


The framework is the software companion to the paper:

> *"A Modular Open-Source AI-Datacenter Simulator for Grid Impact Analysis"*  
> IEEE Transactions on Power Delivery (under review)

### What OASIS does

- Models an AI datacenter with stochastic **GPU workload traces** (4 archetypes), a **3rd-order induction-machine HVAC model**, and an **IEEE 1547-2018 Category III grid-supporting VSC**
- Couples the datacenter to a **distribution network** (IEEE 13-bus, 4.16 kV) via a Gauss–Seidel predictor-corrector co-simulation engine
- Couples the datacenter to a **transmission network** (IEEE 39-bus New England, 345 kV) via pandapower's native AC-OPF and Newton–Raphson solvers
---

---

## Requirements

| Package | Version | Purpose |
|---|---|---|
| Python | ≥ 3.10 | Runtime |
| numpy | ≥ 1.24 | Numerical core |
| pandas | ≥ 2.0 | Data handling and CSV output |
| matplotlib | ≥ 3.7 | All figures |
| pandapower | ≥ 2.13 | IEEE 39-bus AC-OPF and NR power flow |
| scipy | ≥ 1.10 | Optional: OPF solver acceleration |
| opendssdirect.py | ≥ 0.8 | Optional: OpenDSS backend for distribution network |

Install all required packages:

```bash
pip install -r requirements.txt
```

`opendssdirect.py` is optional. If not installed, `opendss_network.py` automatically falls back to the built-in pure-Python Newton–Raphson solver. All case studies in the paper use the built-in solver.

---

## Module Reference

### Core physics

#### `datacenter_subsystem.py`
The physics engine. Contains three coupled models:
- **GPU server trace**: 810 GPUs across 5 archetypes (inference, burst serving, mini-batch training, preprocessing, sinusoidal training). Pre-computed at 1 s resolution; retrieved by linear interpolation during simulation.
- **Induction machine HVAC**: 3 × 250 kVA squirrel-cage motors. Third-order equivalent-circuit + RK4 swing equation. States: rotor speed ω_r.
- **Grid-supporting VSC**: 2 MVA, d–q PI current control, frequency–watt droop (K_fw = 0.05 pu/Hz), volt–VAR droop (K_vv = 0.10 pu/pu), IEEE 1547-2018 Category III LVRT/FRT.

Key parameter: `dt_micro` (inner RK4 step, default 10 ms). Justified by VSC closed-loop bandwidth of 16 rad/s; see Section III-D of the paper.

```python
from datacenter_subsystem import DatacenterSubsystem

dc = DatacenterSubsystem(dt_micro=0.01, t_macro=0.1, seed=42)
result = dc.step(V_pcc_pu=1.0, freq_hz=60.0, t_sim=0.0)
print(result.P_total_mw, result.Q_total_mvar, result.omega_r_pu)
```

#### `datacenter_core.py`
Canonical interface layer. Wraps `DatacenterSubsystem` with `CanonicalInput` / `CanonicalOutput` dataclasses and price-elastic demand response logic.

```python
from datacenter_core import DatacenterPhysics, CanonicalInput

physics = DatacenterPhysics(config={"seed": 42, "dt_micro": 0.01})
out = physics.step(CanonicalInput(V_pu=1.0, freq_hz=60.0,
                                   price_per_mwh=75.0, t_sim=0.0, dt=0.1))
print(out.P_mw, out.dP_droop_mw, out.riding_through)
```

#### `datacenter_registry.py`
Central registry. Allows multiple study modules to share or independently instantiate datacenter instances by name.

```python
from datacenter_registry import register, get_datacenter, deregister

register("my_dc", {"seed": 42, "dt_micro": 0.01, "n_cooling_units": 3})
adapter = get_datacenter("my_dc", "distribution")
P_mw, Q_mvar = adapter.step(V_pcc_pu=0.98, freq_hz=59.8, t=120.0, dt=0.1)
deregister("my_dc")
```

#### `adapters.py`
Study-specific signal translators. Four adapters, all wrapping the same physics:

| Adapter | Signal convention | Primary use |
|---|---|---|
| `DistributionAdapter` | V_pcc [pu on feeder base], f [Hz] → P_mw, Q_mvar | OpenDSS co-simulation |
| `TransmissionAdapter` | V_d, V_q [pu on system base] → I_d, I_q (Norton) | PSS/E-style studies |
| `OPFAdapter` | LMP [$/MWh] → P_bid, Q_bid, P_flex, P_committed | MATPOWER/pandapower OPF |
| `MarketAdapter` | Price signals → market settlement quantities | Day-ahead / real-time |

---


### Study modules

#### `distribution_dynamics_study.py`
Five-phase 300 s transient simulation on the IEEE 13-bus feeder.
Key results: V_pcc = 0.954 pu (pre-fault), motor speed nadir 0.842 pu → 0.963 pu (18 s recovery), ANSI C84.1 compliance 87.2%.

#### `optimal_powerflow_study.py`
24-hour, 5-minute-interval DC merit-order OPF with AC NR verification on the IEEE 13-bus feeder. Four scenarios: Baseline, PriceResponse, VoltSupport, Congestion (+40% peak load).

#### `transmission_opf_study.py`
Same OPF formulation on the IEEE 39-bus transmission system via pandapower. Five scenarios including generator trip (G06, 650 MW forced outage). Demonstrates datacenter DR as contingency reserve resource under FERC Order 2222.

#### `feasibility_study.py`
Transmission interconnection feasibility sweep across four datacenter sizes (100, 250, 500, 1,000 MW) at bus 16 of the IEEE 39-bus system. Computes thermal hosting capacity, voltage sensitivity, and congestion premium without full NR in the inner loop — uses Thevenin equivalent for speed.

```bash
python feasibility_study.py
# Runtime: < 1 second for all four sizes
```

#### `n1_contingency_study.py`
Full N-1 contingency analysis using pandapower's native AC-OPF (`pp.runopp()`) and Newton–Raphson (`pp.runpp()`). Tests all 45 contingencies (35 line trips + 10 generator trips) across four datacenter sizes. Datacenter DR response (volt–VAR + active curtailment) applied before post-contingency power flow.

```bash
python n1_contingency_study.py
# Runtime: ~35 seconds (180 OPF + NR solves)
```

## Extending GOALS

### Adding a new network backend
Subclass `NetworkModel` and implement three methods:

```python
class MyNetworkBackend(NetworkModel):
    def set_pcc_injection(self, P_mw: float, Q_mvar: float): ...
    def solve(self) -> bool: ...                # returns converged flag
    def get_pcc_voltage(self) -> tuple: ...     # returns (V_pu, f_hz)
```

The Gauss–Seidel orchestrator and all study modules are agnostic to the backend.

### Scaling the datacenter size
Pass `target_mw` in the registry config to scale the physics output without changing the GPU count:

```python
register("dc_500mw", {
    "seed":       42,
    "target_mw":  500.0,    # facility rating in MW
    "dt_micro":   0.02,
})
```

## Citation

If you use OASIS in your research, please cite:

```bibtex
@article{goals2026,
  title   = {GOALS: A Grid-aware Open source AI-Datacenter Load Simulator for Impact Analysis},
  author  = {{GOALS Development Team}},
  journal = {IEEE Transactions on Power Delivery},
  year    = {2026},
  note    = {Under review. Code: https://github.com/example-org/oasis-simulator}
}
```

---

## Contributing

Contributions are welcome. Please open an issue before submitting a pull request for new features. The priority areas for future work are:

- Three-phase unbalanced distribution network model
- Hardware-in-the-loop (PHIL) interface
- Reinforcement learning demand response agent
- Integration of measured hyperscale cluster traces (Azure, Google)
- Transmission-level hierarchical multi-campus co-simulation

