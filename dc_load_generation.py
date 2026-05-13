"""
generate_dc_load_profile.py
Generate a 24-hour AI datacenter load profile (P, Q, motor speed,
droop signals) and save to CSV.  Uses the existing DatacenterPhysics
stack — no network solver needed.
"""

import numpy as np
import pandas as pd
from datacenter_registry import register, get_datacenter, deregister
from datacenter_core import CanonicalInput

# ── Configuration ─────────────────────────────────────────────────
TARGET_MW   = 100.0     # facility rating [MW] — change this
DURATION_H  = 24        # hours
DT_S        = 300.0     # 5-minute intervals (300 s)
DT_MICRO    = 0.02      # inner physics step [s]
WARMUP      = 500       # warm-up steps before recording
SEED        = 42

# Scale from 2 MVA physics base to target facility
S_BASE_DC   = 2.0       # MVA
LF_NOMINAL  = 0.323     # mean load factor of GPU trace
scale       = TARGET_MW / (S_BASE_DC * LF_NOMINAL)

# Representative LMP profile ($/MWh) — used only for DR price response
# Replace with your own forecast if needed
h   = np.arange(DURATION_H * 3600 / DT_S) * DT_S / 3600.0
lmp = np.clip(32 + 53*np.exp(-0.5*((h-8.5)/1.2)**2)
                 + 68*np.exp(-0.5*((h-18.5)/1.0)**2)
                 - 22*np.exp(-0.5*((h-12.5)/2.0)**2), 18, 220)

# ── Register and warm up ───────────────────────────────────────────
register("profile_dc", {
    "seed":             SEED,
    "n_cooling_units":  3,
    "dt_micro":         DT_MICRO,
    "dt":               DT_MICRO,
    "price_threshold":  65.0,
    "price_max":        300.0,
    "max_curtail_pu":   0.25,
})
adapter = get_datacenter("profile_dc", "opf", interval_min=int(DT_S/60))
physics = adapter._p

print(f"Warming up ({WARMUP} steps) …")
for w in range(WARMUP):
    physics.step(CanonicalInput(V_pu=1.0, freq_hz=60.0,
                                t_sim=float(w)*DT_MICRO, dt=DT_MICRO))

# ── Sample load profile ────────────────────────────────────────────
print(f"Sampling {len(h)} intervals …")
rows = []
for k, (t_h, price) in enumerate(zip(h, lmp)):
    t_sim = float(WARMUP * DT_MICRO) + float(k) * DT_S

    bid = adapter.get_bid(t_sim)

    rows.append({
        "hour":           round(t_h, 4),
        "time_s":         round(t_sim, 2),
        "lmp_usd_mwh":    round(price, 2),
        # Raw physics (2 MVA base)
        "P_raw_mw":       round(bid["p_mw"],    6),
        "Q_raw_mvar":     round(bid["q_mvar"],  6),
        "flex_raw_mw":    round(bid["flex_mw"], 6),
        # Scaled to target facility
        "P_mw":           round(bid["p_mw"]    * scale, 4),
        "Q_mvar":         round(bid["q_mvar"]  * scale, 4),
        "P_flex_mw":      round(bid["flex_mw"] * scale, 4),
        # Computed quantities
        "pf":             round(bid["p_mw"] /
                                max(np.hypot(bid["p_mw"],bid["q_mvar"]),1e-9), 4),
        "scale":          round(scale, 2),
        "target_mw":      TARGET_MW,
    })

deregister("profile_dc")

# ── Save ───────────────────────────────────────────────────────────
df = pd.DataFrame(rows)
out = f"dc_load_profile_{int(TARGET_MW)}MW.csv"
df.to_csv(out, index=False)

print(f"\nSaved {len(df)} rows → {out}")
print(df[["hour","P_mw","Q_mvar","P_flex_mw","pf","lmp_usd_mwh"]].describe().round(3))