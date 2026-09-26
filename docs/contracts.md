# Report Contracts

This file defines the exact field names each module produces, so everyone's code
can read each other's output without guessing.

## drift_report (produced by B)
Source: `src/monitoring/drift_metrics.py` → `detect_drift_all_features()`
Returns: a list of dicts (one dict per feature)

Columns:
- feature (string)
- type ("numeric" or "categorical")
- psi (float)
- psi_severity ("no_significant_drift" / "moderate_drift" / "significant_drift")
- test_stat (float — KS statistic for numeric, chi-square statistic for categorical)
- p_value (float)
- drifted (true/false)
- scenario (string, e.g. "S0", "S1", "S2")
- seed (int)

Example row:
{"feature": "MonthlyCharges", "type": "numeric", "psi": 1.02, "psi_severity": "significant_drift",
 "test_stat": 0.42, "p_value": 0.0, "drifted": true, "scenario": "S1", "seed": 0}

## rca_report (produced by B)
Source: `src/monitoring/rca.py` → `compute_rca()`
Returns: a pandas DataFrame (one row per feature)

Columns:
- feature (string)
- psi (float)
- shap_importance (float, raw SHAP importance from A)
- drift_score_norm (float, 0-1 normalized PSI)
- importance_score_norm (float, 0-1 normalized SHAP importance, floor 0.05)
- rca_score (float, drift_score_norm * importance_score_norm)
- rca_rank (int, 1 = most likely contributor)
- psi_only_rank (int, ranking by PSI alone)
- shap_only_rank (int, ranking by SHAP importance alone)
- scenario (string)
- seed (int)

Example row:
{"feature": "MonthlyCharges", "psi": 1.02, "shap_importance": 0.39, "rca_score": 1.0,
 "rca_rank": 1, "psi_only_rank": 1, "shap_only_rank": 1, "scenario": "S1", "seed": 0}

## shap_report (produced by A)
Source: `results/tables/shap_importance.json`
Format: {feature_name: mean_abs_shap_value}

## perf_report (produced by A)
Source: `results/tables/baseline_metrics.json`
Columns: accuracy, auc, f1, precision, recall

## decision (produced by C)
[C to fill in once decision engine output format is finalized]

## llm_explanation (produced by C)
[C to fill in once LLM output format is finalized]