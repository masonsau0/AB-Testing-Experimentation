# A/B Testing & Experimentation Workbench

**[Live demo](https://mason-ab-testing.streamlit.app/)** : runs in the browser, no install required.

End-to-end **online experimentation analysis** for an e-commerce checkout
flow A/B test. The bundled synthetic dataset (10,000 users per arm)
simulates a control vs. simplified-cart treatment with a ~10% conversion
lift, $1.20 revenue-per-user lift, and a 13% reduction in time-on-page.

The workbench covers the full lifecycle of a real experiment:

| Stage | What it answers |
|---|---|
| **Sample-size design** | How many users do we need to detect the smallest lift we care about? |
| **Validity checks** | Did the randomisation work? (Sample-ratio mismatch / SRM) |
| **Primary analysis** | Did conversion rate move? (two-proportion z-test) |
| **Secondary analyses** | Did revenue and time-on-page move? (Welch's t-test) |
| **Multiple-comparison correction** | We tested 3 metrics : what's the family-wise error rate? |
| **Bootstrap CIs** | Sanity-check the closed-form CI on the primary metric |
| **CUPED variance reduction** | Use a pre-experiment covariate to cut variance |
| **Decision** | Ship, hold, or kill : with caveats for the rollout doc |

Three layers of access:

1. A **statistics library** (`ab_testing.py`) : power analysis, hypothesis
   tests, multiple-comparison correction, SRM check, CUPED. Self-contained
   functions that compose into a notebook or dashboard.
2. A **walkthrough notebook** (`experiment_analysis.ipynb`) : narrative +
   results for the bundled dataset, in the order a real PM / DS would
   present them.
3. An **interactive Streamlit dashboard** (`ab_testing_app.py`) : three
   tabs: pre-experiment design (sliders for baseline / MDE / α / power),
   live analysis (upload your own CSV), and CUPED reanalysis.

## Key results on the bundled dataset

| Metric | Control (A) | Treatment (B) | Lift | p-value | Bonferroni reject (α=0.05) |
|---|---:|---:|---:|---:|:---:|
| Conversion rate | 15.44% | 16.77% | **+8.6%** rel | 0.011 | ✓ |
| Revenue / user | $11.72 | $12.96 | **+10.5%** rel | 0.003 | ✓ |
| Time on page | 89.8 s | 77.6 s | **−13.6%** rel | <0.001 | ✓ |

All three metrics move in the same direction (conversion + revenue up,
time down), all reach significance after multiple-comparison correction.
**Recommendation: ship.**

## What's actually demonstrated

| Concept | Function in `ab_testing.py` |
|---|---|
| Closed-form sample-size formula for two-proportion z-test | `required_sample_size_proportion` |
| Two-proportion z-test (binary metric) | `two_proportion_ztest` |
| Welch's t-test (continuous metric, unequal variances) | `welch_ttest` |
| Bootstrap CI for the difference between groups | `bootstrap_difference_ci` |
| Bonferroni correction (FWER) | `bonferroni` |
| Benjamini-Hochberg correction (FDR) | `benjamini_hochberg` |
| Sample-ratio mismatch chi-squared check | `srm_check` |
| CUPED variance reduction (Deng et al. 2013) | `cuped_adjust` |

## Run it

```bash
pip install -r requirements.txt
python generate_data.py             # writes experiment_data.csv (20,000 rows)
streamlit run ab_testing_app.py
# or:
jupyter notebook experiment_analysis.ipynb
```

## Repository layout

```
.
├── ab_testing.py               ← statistics library (8 functions)
├── ab_testing_app.py           ← Streamlit dashboard
├── experiment_analysis.ipynb   ← narrative walkthrough notebook
├── generate_data.py            ← synthesises the experiment dataset
├── experiment_data.csv         ← 20,000-row demo dataset (committed)
├── requirements.txt
├── LICENSE
└── README.md
```

## Why this matters

A/B testing is the workhorse of product-decision-making, but most
practitioners stop at "p < 0.05 → ship." Real experimentation is a
pipeline of **design → validity → analysis → correction → reanalysis →
decision** : and the failure modes (peeking, SRM, multiple-comparison
inflation, biased estimators) all live in the steps that get skipped.
This project demonstrates each of them on real-shaped data.

## Stack

Python · **NumPy** / **SciPy** / **statsmodels** (statistics) ·
**pandas** (data) · **Matplotlib** / **seaborn** (figures) ·
**Streamlit** (dashboard) · **Jupyter** (narrative)
