"""Streamlit dashboard for A/B-test design and analysis.

Three panels:
- **Pre-experiment design** — sample-size calculator with sliders for
  baseline rate, MDE, alpha, and power.
- **Live analysis** — upload an experiment CSV (or use the bundled
  default), see SRM check + per-metric tests + multiple-comparison
  correction.
- **CUPED reanalysis** — re-run the revenue test with pre-experiment
  covariate adjustment and see how much variance was removed.
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from ab_testing import (
    required_sample_size_proportion,
    two_proportion_ztest,
    welch_ttest,
    bootstrap_difference_ci,
    bonferroni,
    benjamini_hochberg,
    srm_check,
    cuped_adjust,
)

st.set_page_config(page_title="A/B Testing Workbench", layout="wide")

st.title("A/B Testing Workbench")
st.caption(
    "End-to-end experimentation analysis — sample-size design, hypothesis "
    "tests, multiple-comparison correction, SRM check, and CUPED variance "
    "reduction."
)

with st.expander("How to use this app", expanded=False):
    st.markdown("""
**What this app does in plain English.**
Companies regularly run experiments to test changes — like "does a
simpler checkout button increase sales?" They show version A to half
the users and version B to the other half, then compare the results.
This app does the math behind those experiments: deciding how many
users you need, checking the data is clean, computing whether the
results are real or random luck, and adjusting for the fact that you
tested multiple things at once.

**The three tabs.**

**1. Sample-size design** — *use this BEFORE running an experiment.*
Tells you how many users you need to detect a real difference. Move
the sliders to match your situation:
- **Baseline conversion rate** — your current success rate (e.g. 15 %
  of visitors check out today).
- **Minimum detectable effect** — the smallest improvement you care
  about (e.g. 10 % means going from 15 % to 16.5 %).
- **Significance level α** — how strict you are about avoiding false
  positives. 0.05 (5 %) is the common default.
- **Statistical power** — how reliable you want the test to be at
  finding a real effect. 0.80 (80 %) is the common default.

**2. Experiment analysis** — *use this AFTER your experiment runs.*
Upload a CSV or use the built-in example data. The app shows:
- **SRM check** — was the user split actually 50/50? If not, the
  experiment is broken and results aren't trustworthy.
- **Per-metric tests** — for each thing you measured (conversions,
  revenue, time on page), did A vs B differ enough to be statistically
  significant? Look at the **p-value column**: under 0.05 generally
  means "real effect, not random luck."
- **Multiple-comparison correction** — when you test multiple things,
  random luck eventually gives you a false hit. Bonferroni and
  Benjamini-Hochberg both adjust for this — the **rejected** column
  tells you which results survive after correction.

**3. CUPED reanalysis** — an advanced trick to make experiments more
sensitive. Don't worry about this unless you're going deep.

**Try this.** Open Tab 2, accept the bundled data, and look at the
p-values: all three metrics (conversion, revenue, time-on-page) come
out under 0.05 even after Bonferroni correction. The treatment wins.
""")

tab_design, tab_analyze, tab_cuped = st.tabs(
    ["Sample-size design", "Experiment analysis", "CUPED reanalysis"]
)

# =====================================================================
# Tab 1 — Pre-experiment design
# =====================================================================
with tab_design:
    st.subheader("Sample size for a two-proportion z-test")
    st.markdown(
        "Use this **before** the experiment runs. Pick the baseline "
        "conversion rate, the smallest lift you care about (MDE), and the "
        "decision-error tolerances."
    )

    col1, col2 = st.columns(2)
    with col1:
        p_baseline = st.slider("Baseline conversion rate", 0.01, 0.50, 0.15, 0.01)
        relative_mde = st.slider(
            "Minimum detectable effect (relative)", 0.01, 0.50, 0.10, 0.01,
            help="Smallest relative lift you want to be able to detect.",
        )
    with col2:
        alpha = st.slider("Significance level α", 0.01, 0.20, 0.05, 0.01)
        power = st.slider("Statistical power (1 − β)", 0.50, 0.99, 0.80, 0.01)

    ss = required_sample_size_proportion(
        p_baseline=p_baseline, relative_mde=relative_mde,
        alpha=alpha, power=power,
    )

    cols = st.columns(4)
    cols[0].metric("Required N per arm", f"{ss.n_per_arm:,}")
    cols[1].metric("Total users", f"{ss.n_per_arm * 2:,}")
    cols[2].metric("Detect lift from", f"{ss.p_baseline:.2%}")
    cols[3].metric("To at least", f"{ss.p_treatment:.2%}")

    st.info(
        f"At a daily traffic of 5,000 users, this experiment needs "
        f"~**{math.ceil(ss.n_per_arm * 2 / 5000)} days** to reach the "
        f"required sample size. Stop early at your peril — peeking inflates "
        f"false-positive rates without correction."
    )

# =====================================================================
# Tab 2 — Live experiment analysis
# =====================================================================
with tab_analyze:
    st.subheader("Per-metric test, with multiple-comparison correction")

    default_path = Path("experiment_data.csv")
    upload = st.file_uploader("Upload experiment CSV (cols: variant, converted, revenue, time_on_page, pre_revenue)", type=["csv"])
    if upload is not None:
        df = pd.read_csv(upload)
    elif default_path.exists():
        df = pd.read_csv(default_path)
        st.caption(f"Using bundled `{default_path.name}` ({len(df):,} rows). Upload your own to override.")
    else:
        st.warning("Run `python generate_data.py` first or upload a CSV.")
        st.stop()

    if "variant" not in df.columns:
        st.error("CSV must include a `variant` column with values A/B (or control/treatment).")
        st.stop()

    a = df[df.variant == df.variant.unique()[0]]
    b = df[df.variant == df.variant.unique()[1]]
    a_label, b_label = df.variant.unique()[0], df.variant.unique()[1]

    # ----------------------------------------------- SRM
    st.markdown("##### Sample-ratio mismatch check")
    srm = srm_check(len(a), len(b))
    cols = st.columns(4)
    cols[0].metric(f"N ({a_label})", f"{len(a):,}")
    cols[1].metric(f"N ({b_label})", f"{len(b):,}")
    cols[2].metric("Chi-squared", f"{srm.chi2:.3f}")
    cols[3].metric("p-value", f"{srm.p_value:.4f}")
    if srm.flagged:
        st.error("**SRM flagged.** Sample split deviates from 50/50 more than expected. Pause and root-cause before interpreting results.")
    else:
        st.success("SRM passed (p > 0.001).")

    # ----------------------------------------------- per-metric tests
    st.markdown("##### Per-metric tests")
    results = {}

    if "converted" in df.columns:
        ct = two_proportion_ztest(int(a.converted.sum()), len(a),
                                   int(b.converted.sum()), len(b))
        results["conversion"] = ct
    if "revenue" in df.columns:
        rt = welch_ttest(a.revenue, b.revenue)
        results["revenue"] = rt
    if "time_on_page" in df.columns:
        tt = welch_ttest(a.time_on_page, b.time_on_page)
        results["time_on_page"] = tt

    rows = []
    for name, r in results.items():
        if hasattr(r, "p_a"):
            rows.append({
                "metric": name, "type": "binary",
                f"{a_label}": f"{r.p_a:.4f}", f"{b_label}": f"{r.p_b:.4f}",
                "abs lift": f"{r.absolute_lift:+.4f}",
                "rel lift": f"{r.relative_lift*100:+.2f}%",
                "p-value": f"{r.p_value:.5f}",
                "95% CI on lift": f"[{r.ci95_lower:+.4f}, {r.ci95_upper:+.4f}]",
            })
        else:
            rows.append({
                "metric": name, "type": "continuous",
                f"{a_label}": f"{r.mean_a:.3f}", f"{b_label}": f"{r.mean_b:.3f}",
                "abs lift": f"{r.absolute_lift:+.3f}",
                "rel lift": f"{r.relative_lift*100:+.2f}%",
                "p-value": f"{r.p_value:.5f}",
                "95% CI on lift": f"[{r.ci95_lower:+.3f}, {r.ci95_upper:+.3f}]",
            })

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ----------------------------------------------- multiple-comparison correction
    st.markdown("##### Multiple-comparison correction")
    pvals = [r.p_value for r in results.values()]
    metric_names = list(results.keys())

    correction_alpha = st.slider("Family-wise α", 0.01, 0.20, 0.05, 0.01, key="correction_alpha")
    bonf = bonferroni(pvals, correction_alpha)
    bh = benjamini_hochberg(pvals, correction_alpha)

    correction_table = pd.DataFrame({
        "metric": metric_names,
        "p-value": [f"{p:.5f}" for p in pvals],
        f"reject @ α={correction_alpha} (no correction)": [p < correction_alpha for p in pvals],
        f"Bonferroni (α/m = {bonf['alpha_corrected']:.4f})": bonf["rejected"],
        "Benjamini-Hochberg (FDR)": bh["rejected"],
    })
    st.dataframe(correction_table, use_container_width=True, hide_index=True)
    st.caption(
        "Bonferroni controls the **family-wise error rate** (probability of "
        "any false positive across all metrics) — strictest. "
        "Benjamini-Hochberg controls the **false discovery rate** (expected "
        "share of false positives among rejections) — looser, more powerful."
    )

# =====================================================================
# Tab 3 — CUPED
# =====================================================================
with tab_cuped:
    st.subheader("CUPED variance reduction (pre-experiment covariate adjustment)")
    st.markdown(
        "**CUPED** (Deng et al. 2013) reduces metric variance by adjusting "
        "for a pre-experiment covariate that's correlated with the in-experiment "
        "metric. Lower variance → tighter CI → smaller required N for the same power."
    )

    if "pre_revenue" not in df.columns:
        st.warning("Dataset has no `pre_revenue` column — skipping CUPED demo.")
    else:
        adj_a, theta_a = cuped_adjust(a.revenue, a.pre_revenue)
        adj_b, theta_b = cuped_adjust(b.revenue, b.pre_revenue)

        # Re-run the test on adjusted metric.
        rt_orig = welch_ttest(a.revenue, b.revenue)
        rt_adj = welch_ttest(adj_a, adj_b)

        original_var = (np.var(a.revenue, ddof=1) + np.var(b.revenue, ddof=1)) / 2
        adjusted_var = (np.var(adj_a, ddof=1) + np.var(adj_b, ddof=1)) / 2
        reduction_pct = (1 - adjusted_var / original_var) * 100

        cols = st.columns(4)
        cols[0].metric("θ (control side)", f"{theta_a:.3f}")
        cols[1].metric("Variance reduction", f"{reduction_pct:.1f}%")
        cols[2].metric("Original p-value", f"{rt_orig.p_value:.5f}")
        cols[3].metric("CUPED p-value", f"{rt_adj.p_value:.5f}")

        st.markdown("##### Original vs CUPED-adjusted CIs on revenue lift")
        st.dataframe(
            pd.DataFrame([
                {"version": "Original", "lift": rt_orig.absolute_lift,
                 "ci95_lower": rt_orig.ci95_lower, "ci95_upper": rt_orig.ci95_upper,
                 "ci_width": rt_orig.ci95_upper - rt_orig.ci95_lower},
                {"version": "CUPED",    "lift": rt_adj.absolute_lift,
                 "ci95_lower": rt_adj.ci95_lower, "ci95_upper": rt_adj.ci95_upper,
                 "ci_width": rt_adj.ci95_upper - rt_adj.ci95_lower},
            ]).round(3),
            use_container_width=True, hide_index=True,
        )
        st.caption(
            "When pre/post correlation is mild (~0.35 in the bundled data), "
            "CUPED removes a few % of variance. With stronger correlations "
            "(0.7+, common in mature products), CUPED routinely cuts required "
            "sample size by 30-50%."
        )
