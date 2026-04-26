"""Generate a synthetic A/B-test dataset.

Simulates a checkout-flow experiment: control (A) vs. simplified-cart
treatment (B). For each user, generates a binary `converted` outcome, a
continuous `revenue` (zero if not converted), a continuous `time_on_page`,
and a pre-experiment `pre_revenue` covariate that's correlated with
treatment-period revenue (used by CUPED).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULTS = {
    "n_per_arm": 10_000,
    "baseline_conversion": 0.150,
    "treatment_conversion": 0.165,  # ~10% relative lift
    "baseline_aov": 75.0,            # average order value
    "treatment_aov": 76.5,           # treatment slightly increases AOV too
    "aov_std": 18.0,
    "baseline_time_on_page": 90.0,   # seconds
    "treatment_time_on_page": 78.0,  # 13% time reduction
    "time_std": 25.0,
    "pre_revenue_correlation": 0.35,
    "seed": 42,
}


def generate(
    n_per_arm: int = DEFAULTS["n_per_arm"],
    baseline_conversion: float = DEFAULTS["baseline_conversion"],
    treatment_conversion: float = DEFAULTS["treatment_conversion"],
    baseline_aov: float = DEFAULTS["baseline_aov"],
    treatment_aov: float = DEFAULTS["treatment_aov"],
    aov_std: float = DEFAULTS["aov_std"],
    baseline_time_on_page: float = DEFAULTS["baseline_time_on_page"],
    treatment_time_on_page: float = DEFAULTS["treatment_time_on_page"],
    time_std: float = DEFAULTS["time_std"],
    pre_revenue_correlation: float = DEFAULTS["pre_revenue_correlation"],
    seed: int = DEFAULTS["seed"],
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []

    for variant, p_conv, aov_mean, time_mean in [
        ("A", baseline_conversion, baseline_aov, baseline_time_on_page),
        ("B", treatment_conversion, treatment_aov, treatment_time_on_page),
    ]:
        n = n_per_arm
        # Pre-experiment per-user revenue - drawn from same baseline distribution
        # for both arms (because randomisation happens BEFORE the experiment).
        pre_revenue = np.clip(
            rng.normal(loc=baseline_aov * baseline_conversion,
                        scale=aov_std * 0.5, size=n),
            0, None,
        )
        # In-experiment conversion: probability shifts a bit with pre_revenue
        # so CUPED has signal to use.
        z = (pre_revenue - pre_revenue.mean()) / pre_revenue.std()
        logit = np.log(p_conv / (1 - p_conv)) + 0.4 * z
        prob = 1 / (1 + np.exp(-logit))
        converted = rng.uniform(size=n) < prob
        # Revenue: zero if not converted; otherwise drawn from the AOV distribution
        # but correlated with pre_revenue via the same z-score.
        rev_noise = rng.normal(loc=aov_mean, scale=aov_std, size=n) + \
                    pre_revenue_correlation * (pre_revenue - pre_revenue.mean())
        revenue = np.where(converted, np.clip(rev_noise, 0, None), 0.0)
        # Time on page: log-normal-ish, roughly Gaussian for simplicity.
        time_on_page = np.clip(
            rng.normal(loc=time_mean, scale=time_std, size=n),
            5, None,
        )
        for i in range(n):
            rows.append({
                "user_id": f"{variant}-{i:06d}",
                "variant": variant,
                "converted": int(converted[i]),
                "revenue": float(revenue[i]),
                "time_on_page": float(time_on_page[i]),
                "pre_revenue": float(pre_revenue[i]),
            })

    df = pd.DataFrame(rows)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-per-arm", type=int, default=DEFAULTS["n_per_arm"])
    parser.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    parser.add_argument("--out", type=Path, default=Path("experiment_data.csv"))
    args = parser.parse_args()
    df = generate(n_per_arm=args.n_per_arm, seed=args.seed)
    df.to_csv(args.out, index=False)
    print(f"Wrote {args.out}")
    print(df.groupby("variant").agg(
        users=("user_id", "count"),
        conversion=("converted", "mean"),
        revenue_per_user=("revenue", "mean"),
        time_on_page=("time_on_page", "mean"),
    ).round(3))


if __name__ == "__main__":
    main()
