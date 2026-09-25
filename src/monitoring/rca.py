import pandas as pd


def normalize_scores(values_dict, floor=0.05):
    """
    Scale a dict of {feature: value} to [floor, 1] range instead of [0, 1],
    so the lowest-importance feature doesn't get multiplied by exactly zero.
    """
    values = list(values_dict.values())
    min_val, max_val = min(values), max(values)
    if max_val == min_val:
        return {k: 1.0 for k in values_dict}
    return {
        k: floor + (1 - floor) * (v - min_val) / (max_val - min_val)
        for k, v in values_dict.items()
    }


def compute_rca(drift_results, shap_importance):
    """
    drift_results: output from detect_drift_all_features() — a list of dicts,
                   each with 'feature' and 'psi'.
    shap_importance: dict of {feature: mean_abs_shap_value} from Member A.

    Returns a list of dicts, one per feature, each with:
      - drift_score (normalized PSI)
      - importance_score (normalized SHAP importance)
      - rca_score (combined: drift_score * importance_score)
      - psi_only_rank, shap_only_rank, rca_rank (for comparison)
    """
    # pull psi per feature from drift results
    psi_raw = {r["feature"]: r["psi"] for r in drift_results}

    # keep only features present in both drift results and shap importance
    common_features = [f for f in psi_raw if f in shap_importance]

    psi_common = {f: psi_raw[f] for f in common_features}
    shap_common = {f: shap_importance[f] for f in common_features}

    drift_norm = normalize_scores(psi_common)
    importance_norm = normalize_scores(shap_common)

    rows = []
    for f in common_features:
        drift_score = drift_norm[f]
        importance_score = importance_norm[f]
        rca_score = drift_score * importance_score
        rows.append({
            "feature": f,
            "psi": psi_common[f],
            "shap_importance": shap_common[f],
            "drift_score_norm": round(drift_score, 4),
            "importance_score_norm": round(importance_score, 4),
            "rca_score": round(rca_score, 4)
        })

    df = pd.DataFrame(rows)

    # add rankings for comparison (rank 1 = most likely contributor)
    df["rca_rank"] = df["rca_score"].rank(ascending=False, method="min").astype(int)
    df["psi_only_rank"] = df["psi"].rank(ascending=False, method="min").astype(int)
    df["shap_only_rank"] = df["shap_importance"].rank(ascending=False, method="min").astype(int)

    df = df.sort_values("rca_rank").reset_index(drop=True)
    return df


def top_k_contributors(rca_df, k=3, rank_column="rca_rank"):
    """
    Returns the top-k feature names by a given ranking column.
    Use this to check if the injected drift feature shows up in the top-k.
    """
    return rca_df.sort_values(rank_column).head(k)["feature"].tolist()