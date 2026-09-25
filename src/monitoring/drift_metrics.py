import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, chi2_contingency


def calculate_psi(reference, current, bins=10):
    """
    Population Stability Index for one numeric feature.
    """
    reference = np.array(reference)
    current = np.array(current)

    breakpoints = np.linspace(0, 100, bins + 1)
    bin_edges = np.percentile(reference, breakpoints)
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    ref_counts, _ = np.histogram(reference, bins=bin_edges)
    cur_counts, _ = np.histogram(current, bins=bin_edges)

    ref_pct = ref_counts / len(reference)
    cur_pct = cur_counts / len(current)

    epsilon = 1e-4
    ref_pct = np.where(ref_pct == 0, epsilon, ref_pct)
    cur_pct = np.where(cur_pct == 0, epsilon, cur_pct)

    psi_value = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return psi_value


def psi_severity(psi_value):
    if psi_value < 0.1:
        return "no_significant_drift"
    elif psi_value < 0.25:
        return "moderate_drift"
    else:
        return "significant_drift"


def ks_test(reference, current):
    stat, p_value = ks_2samp(reference, current)
    return stat, p_value


def chi_square_test(reference, current):
    ref_counts = pd.Series(reference).value_counts()
    cur_counts = pd.Series(current).value_counts()

    all_categories = sorted(set(ref_counts.index) | set(cur_counts.index))
    ref_aligned = [ref_counts.get(cat, 0) for cat in all_categories]
    cur_aligned = [cur_counts.get(cat, 0) for cat in all_categories]

    contingency_table = np.array([ref_aligned, cur_aligned])
    stat, p_value, dof, expected = chi2_contingency(contingency_table)
    return stat, p_value


def detect_drift_all_features(reference_df, current_df, numeric_features, categorical_features, p_threshold=0.05):
    results = []

    for feature in numeric_features:
        psi_val = calculate_psi(reference_df[feature], current_df[feature])
        ks_stat, ks_p = ks_test(reference_df[feature], current_df[feature])
        results.append({
            "feature": feature,
            "type": "numeric",
            "psi": round(psi_val, 4),
            "psi_severity": psi_severity(psi_val),
            "test_stat": round(ks_stat, 4),
            "p_value": round(ks_p, 4),
            "drifted": bool(psi_val >= 0.1 or ks_p < p_threshold)
        })

    for feature in categorical_features:
        psi_val = calculate_psi(
            pd.factorize(reference_df[feature])[0],
            pd.factorize(current_df[feature])[0]
        )
        chi_stat, chi_p = chi_square_test(reference_df[feature], current_df[feature])
        results.append({
            "feature": feature,
            "type": "categorical",
            "psi": round(psi_val, 4),
            "psi_severity": psi_severity(psi_val),
            "test_stat": round(chi_stat, 4),
            "p_value": round(chi_p, 4),
            "drifted": bool(psi_val >= 0.1 or chi_p < p_threshold)
        })

    return results