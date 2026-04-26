"""A/B testing analysis library.

Self-contained statistical functions for online experimentation:

- **Power / sample size** — closed-form sample-size calculation for a
  two-proportion z-test under a target minimum-detectable effect.
- **Hypothesis tests** — two-proportion z-test (binary metric) and
  Welch's t-test (continuous metric).
- **Bootstrap CIs** — resampling-based confidence interval for the
  difference between two groups.
- **Multiple-comparison correction** — Bonferroni and Benjamini-Hochberg.
- **Sample-ratio mismatch (SRM) check** — chi-squared test that the
  observed traffic split matches the expected one.
- **CUPED variance reduction** — pre-experiment covariate adjustment.

All functions return small dataclasses so they compose cleanly into a
notebook narrative or a dashboard table.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy import stats


# =====================================================================
# Power / sample size
# =====================================================================


@dataclass
class SampleSizeResult:
    n_per_arm: int
    p_baseline: float
    p_treatment: float
    absolute_mde: float
    relative_mde: float
    alpha: float
    power: float


def required_sample_size_proportion(
    p_baseline: float,
    relative_mde: float,
    alpha: float = 0.05,
    power: float = 0.80,
    two_sided: bool = True,
) -> SampleSizeResult:
    """Required sample size per arm for a two-proportion z-test.

    Closed-form formula:
        n = (z_{1-alpha/2} sqrt(2 p_bar q_bar) + z_{1-beta} sqrt(p1 q1 + p2 q2))^2 / delta^2
    where p_bar = (p1 + p2) / 2 and delta = |p1 - p2|.
    """
    if not (0 < p_baseline < 1):
        raise ValueError("p_baseline must be in (0, 1)")
    p1 = p_baseline
    p2 = p_baseline * (1 + relative_mde)
    delta = abs(p2 - p1)
    p_bar = (p1 + p2) / 2
    q_bar = 1 - p_bar
    q1, q2 = 1 - p1, 1 - p2

    z_alpha = stats.norm.ppf(1 - alpha / 2) if two_sided else stats.norm.ppf(1 - alpha)
    z_beta = stats.norm.ppf(power)

    n = ((z_alpha * math.sqrt(2 * p_bar * q_bar) +
          z_beta * math.sqrt(p1 * q1 + p2 * q2)) ** 2) / (delta ** 2)
    return SampleSizeResult(
        n_per_arm=int(math.ceil(n)),
        p_baseline=p_baseline,
        p_treatment=p2,
        absolute_mde=delta,
        relative_mde=relative_mde,
        alpha=alpha,
        power=power,
    )


# =====================================================================
# Hypothesis tests
# =====================================================================


@dataclass
class ProportionTestResult:
    n_a: int
    n_b: int
    p_a: float
    p_b: float
    absolute_lift: float
    relative_lift: float
    z_stat: float
    p_value: float
    ci95_lower: float
    ci95_upper: float


def two_proportion_ztest(
    successes_a: int, n_a: int, successes_b: int, n_b: int
) -> ProportionTestResult:
    """Two-proportion z-test (Wald), with 95% CI on the difference."""
    p_a = successes_a / n_a
    p_b = successes_b / n_b
    p_pool = (successes_a + successes_b) / (n_a + n_b)
    se_pool = math.sqrt(p_pool * (1 - p_pool) * (1 / n_a + 1 / n_b))
    z = (p_b - p_a) / se_pool if se_pool > 0 else 0.0
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))

    se_unpool = math.sqrt(p_a * (1 - p_a) / n_a + p_b * (1 - p_b) / n_b)
    margin = 1.96 * se_unpool
    return ProportionTestResult(
        n_a=n_a, n_b=n_b, p_a=p_a, p_b=p_b,
        absolute_lift=p_b - p_a,
        relative_lift=(p_b - p_a) / p_a if p_a > 0 else float("nan"),
        z_stat=z, p_value=p_value,
        ci95_lower=(p_b - p_a) - margin,
        ci95_upper=(p_b - p_a) + margin,
    )


@dataclass
class ContinuousTestResult:
    mean_a: float
    mean_b: float
    std_a: float
    std_b: float
    n_a: int
    n_b: int
    absolute_lift: float
    relative_lift: float
    t_stat: float
    df: float
    p_value: float
    ci95_lower: float
    ci95_upper: float


def welch_ttest(
    group_a: Sequence[float], group_b: Sequence[float]
) -> ContinuousTestResult:
    """Welch's t-test (unequal variances) on two continuous samples."""
    a = np.asarray(group_a, dtype=float)
    b = np.asarray(group_b, dtype=float)
    res = stats.ttest_ind(b, a, equal_var=False)
    df = (a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)) ** 2 / (
        (a.var(ddof=1) / len(a)) ** 2 / (len(a) - 1)
        + (b.var(ddof=1) / len(b)) ** 2 / (len(b) - 1)
    )
    se = math.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    t_crit = stats.t.ppf(0.975, df)
    diff = b.mean() - a.mean()
    return ContinuousTestResult(
        mean_a=float(a.mean()), mean_b=float(b.mean()),
        std_a=float(a.std(ddof=1)), std_b=float(b.std(ddof=1)),
        n_a=len(a), n_b=len(b),
        absolute_lift=diff,
        relative_lift=diff / a.mean() if a.mean() != 0 else float("nan"),
        t_stat=float(res.statistic), df=float(df),
        p_value=float(res.pvalue),
        ci95_lower=diff - t_crit * se,
        ci95_upper=diff + t_crit * se,
    )


# =====================================================================
# Bootstrap CIs
# =====================================================================


def bootstrap_difference_ci(
    group_a: Sequence[float],
    group_b: Sequence[float],
    n_resamples: int = 5000,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float]:
    """Bootstrap CI for `mean(B) - mean(A)`. Returns (point_estimate, lower, upper)."""
    rng = rng or np.random.default_rng(42)
    a = np.asarray(group_a, dtype=float)
    b = np.asarray(group_b, dtype=float)
    diffs = np.empty(n_resamples)
    for i in range(n_resamples):
        diffs[i] = b[rng.integers(0, len(b), len(b))].mean() - \
                  a[rng.integers(0, len(a), len(a))].mean()
    point = float(b.mean() - a.mean())
    lower = float(np.quantile(diffs, alpha / 2))
    upper = float(np.quantile(diffs, 1 - alpha / 2))
    return point, lower, upper


# =====================================================================
# Multiple-comparison correction
# =====================================================================


def bonferroni(pvalues: Sequence[float], alpha: float = 0.05) -> dict:
    """Bonferroni: divide alpha by number of tests."""
    m = len(pvalues)
    threshold = alpha / m
    return {
        "alpha_corrected": threshold,
        "rejected": [p < threshold for p in pvalues],
    }


def benjamini_hochberg(pvalues: Sequence[float], alpha: float = 0.05) -> dict:
    """Benjamini-Hochberg FDR control. Less conservative than Bonferroni."""
    m = len(pvalues)
    sorted_idx = np.argsort(pvalues)
    sorted_p = np.array(pvalues)[sorted_idx]
    bh_thresholds = (np.arange(1, m + 1) / m) * alpha
    # Largest k such that p(k) <= (k/m) * alpha; everything up to k is rejected.
    below = sorted_p <= bh_thresholds
    if not below.any():
        rejected_sorted = np.zeros(m, dtype=bool)
    else:
        k = np.where(below)[0].max()
        rejected_sorted = np.zeros(m, dtype=bool)
        rejected_sorted[: k + 1] = True
    rejected = np.zeros(m, dtype=bool)
    rejected[sorted_idx] = rejected_sorted
    return {
        "thresholds_sorted": bh_thresholds.tolist(),
        "rejected": rejected.tolist(),
    }


# =====================================================================
# Sample-ratio mismatch (SRM)
# =====================================================================


@dataclass
class SRMResult:
    observed_a: int
    observed_b: int
    expected_ratio_a: float
    chi2: float
    p_value: float
    flagged: bool


def srm_check(
    n_a: int, n_b: int, expected_ratio_a: float = 0.5, alpha: float = 0.001
) -> SRMResult:
    """Chi-squared test that the observed split matches the expected one.

    A statistically-significant mismatch means the experiment instrumentation
    is broken — the analysis should be paused and root-caused, not interpreted.
    Default alpha = 0.001 because false positives on SRM are noisy.
    """
    total = n_a + n_b
    expected_a = total * expected_ratio_a
    expected_b = total * (1 - expected_ratio_a)
    chi2 = (n_a - expected_a) ** 2 / expected_a + (n_b - expected_b) ** 2 / expected_b
    p_value = 1 - stats.chi2.cdf(chi2, df=1)
    return SRMResult(
        observed_a=n_a, observed_b=n_b,
        expected_ratio_a=expected_ratio_a,
        chi2=float(chi2), p_value=float(p_value),
        flagged=p_value < alpha,
    )


# =====================================================================
# CUPED variance reduction
# =====================================================================


def cuped_adjust(
    metric: Sequence[float], pre_metric: Sequence[float]
) -> tuple[np.ndarray, float]:
    """CUPED-adjusted metric (Deng et al. 2013).

    Subtracts theta * (X - mean(X)) from Y, where theta = Cov(Y, X) / Var(X).
    The adjusted metric is unbiased for E[Y] and has lower variance whenever
    Y and X are correlated. Returns (adjusted_metric, theta).
    """
    y = np.asarray(metric, dtype=float)
    x = np.asarray(pre_metric, dtype=float)
    var_x = x.var(ddof=1)
    if var_x == 0:
        return y.copy(), 0.0
    theta = float(np.cov(y, x, ddof=1)[0, 1] / var_x)
    y_adj = y - theta * (x - x.mean())
    return y_adj, theta
