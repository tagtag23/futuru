# BOxCrete — Concrete Strength Optimizer
# Built on Meta's Ax/BoTorch (MIT License)
# ==========================================

import pandas as pd
from ax.service.ax_client import AxClient, ObjectiveProperties

# ── 1. DEFINE THE CONCRETE MIX PARAMETERS ──────────────────
# These are the ingredients we want to optimize.
# Min/max ranges are typical ready-mix values (kg/m³).

parameters = [
    {"name": "cement",          "type": "range", "bounds": [150.0, 450.0]},
    {"name": "water",           "type": "range", "bounds": [120.0, 220.0]},
    {"name": "fine_aggregate",  "type": "range", "bounds": [600.0, 900.0]},
    {"name": "coarse_aggregate","type": "range", "bounds": [800.0, 1100.0]},
    {"name": "fly_ash",         "type": "range", "bounds": [0.0,   200.0]},
    {"name": "superplasticizer","type": "range", "bounds": [0.0,   20.0]},
    {"name": "age_days",        "type": "range", "bounds": [7.0,   90.0]},
]

# ── 2. STRENGTH PREDICTION FUNCTION ────────────────────────
# This is a simplified model based on research data.
# A real version would use a trained ML model or lab data.

def predict_strength(cement, water, fine_aggregate,
                     coarse_aggregate, fly_ash,
                     superplasticizer, age_days):
    """
    Estimates compressive strength in MPa.
    Based on Yeh (1998) concrete dataset relationships.
    """
    w_c_ratio = water / (cement + 0.5 * fly_ash + 1e-6)
    binder    = cement + 0.8 * fly_ash
    strength  = (
        0.35 * binder
        - 25.0 * w_c_ratio
        + 0.008 * fine_aggregate * 0.1
        + 0.005 * coarse_aggregate * 0.1
        + 2.5   * superplasticizer
        + 0.18  * age_days
        - 10.0
    )
    return max(strength, 0.0)

# ── 3. WRAP IT FOR AX ──────────────────────────────────────

def evaluate_mix(params):
    strength = predict_strength(
        cement           = params["cement"],
        water            = params["water"],
        fine_aggregate   = params["fine_aggregate"],
        coarse_aggregate = params["coarse_aggregate"],
        fly_ash          = params["fly_ash"],
        superplasticizer = params["superplasticizer"],
        age_days         = params["age_days"],
    )
    return {"strength_MPa": (strength, 0.5)}  # (value, noise)

# ── 4. RUN BAYESIAN OPTIMIZATION ───────────────────────────

print("\n🔬 BOxCrete — Starting Bayesian Optimization")
print("=" * 50)
print("Goal: Maximize compressive strength (MPa)")
print("Trials: 20 iterations\n")

ax_client = AxClient(verbose_logging=False)

ax_client.create_experiment(
    name       = "boxcrete_iceland_v1",
    parameters = parameters,
    objectives = {"strength_MPa": ObjectiveProperties(minimize=False)},
)

results = []

for i in range(20):
    params, trial_index = ax_client.get_next_trial()
    result = evaluate_mix(params)
    ax_client.complete_trial(trial_index=trial_index, raw_data=result)

    strength = result["strength_MPa"][0]
    results.append({**params, "strength_MPa": round(strength, 1)})
    print(f"  Trial {i+1:02d} | Strength: {strength:.1f} MPa"
          f" | Cement: {params['cement']:.0f} | W/C: {params['water']/params['cement']:.2f}")

# ── 5. SHOW BEST RESULT ─────────────────────────────────────

best_params, best_vals = ax_client.get_best_parameters()
best_strength = best_vals[0]

print("\n" + "=" * 50)
print("✅ OPTIMAL MIX FOUND")
print("=" * 50)
print(f"\n  Predicted strength : {best_strength['strength_MPa']:.1f} MPa")
print(f"  Strength class     : C{int(best_strength['strength_MPa'] * 0.8 / 5) * 5}/...")
print()
print("  Mix design (kg/m³):")
for k, v in best_params.items():
    print(f"    {k:<22} {v:.1f}")

# ── 6. EXPORT TO CSV ────────────────────────────────────────

df = pd.DataFrame(results)
df.to_csv("optimization_results.csv", index=False)
print("\n📄 All trials saved to: optimization_results.csv")
print("=" * 50)