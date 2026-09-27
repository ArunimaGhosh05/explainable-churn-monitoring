import numpy as np
import pandas as pd


def scenario_s0_no_drift(current_pool_df, seed=0):
    """
    S0: No drift control.
    Just take a random subsample of the current pool, unchanged.
    """
    return current_pool_df.sample(frac=0.8, random_state=seed).reset_index(drop=True)


def scenario_s1_drift_important_feature(current_pool_df, feature="MonthlyCharges", std_multiplier=1.0, seed=0):
    """
    S1: Inject drift on an important feature by shifting it.
    std_multiplier: how many standard deviations to shift by (e.g. 0.5, 1.0)
    """
    df = current_pool_df.sample(frac=0.8, random_state=seed).reset_index(drop=True).copy()
    shift_amount = df[feature].std() * std_multiplier
    df[feature] = df[feature] + shift_amount
    return df


def scenario_s2_drift_unimportant_feature(current_pool_df, feature, std_multiplier=1.0, seed=0):
    """
    S2: Inject the same kind of drift, but on a feature with low SHAP importance.
    'feature' should be picked using A's SHAP importance ranking (lowest importance feature).
    Only suitable for continuous numeric features — use scenario_shift_binary_feature for 0/1 features.
    """
    df = current_pool_df.sample(frac=0.8, random_state=seed).reset_index(drop=True).copy()
    shift_amount = df[feature].std() * std_multiplier
    df[feature] = df[feature] + shift_amount
    return df


def scenario_shift_binary_feature(current_pool_df, feature, flip_fraction=0.3, seed=0):
    """
    For binary (0/1) features: a constant std-shift doesn't change the distribution shape,
    so PSI barely reacts. Instead, flip a fraction of values to change the proportion of 1s,
    which is the meaningful way to simulate drift on a binary feature.
    """
    df = current_pool_df.sample(frac=0.8, random_state=seed).reset_index(drop=True).copy()
    rng = np.random.default_rng(seed)
    n_flip = int(len(df) * flip_fraction)
    flip_idx = rng.choice(df.index, size=n_flip, replace=False)
    df.loc[flip_idx, feature] = 1 - df.loc[flip_idx, feature]
    return df

def scenario_s3_concept_drift(current_pool_df, label_col="Churn", segment_col=None,
                               segment_value=None, flip_fraction=0.3, seed=0):
    """
    S3: Concept drift.
    Features stay unchanged, but we flip a fraction of the churn labels
    within a chosen segment (or the whole dataset if segment_col is None).
    This simulates the relationship between features and churn changing,
    without any change in the input feature distributions.
    """
    df = current_pool_df.sample(frac=0.8, random_state=seed).reset_index(drop=True).copy()
    rng = np.random.default_rng(seed)

    if segment_col is not None and segment_value is not None:
        segment_mask = df[segment_col] == segment_value
    else:
        segment_mask = pd.Series([True] * len(df))

    segment_idx = df[segment_mask].index
    n_flip = int(len(segment_idx) * flip_fraction)
    flip_idx = rng.choice(segment_idx, size=n_flip, replace=False)
    df.loc[flip_idx, label_col] = 1 - df.loc[flip_idx, label_col]

    return df

def get_scenario_data(scenario_id, current_pool_df, important_feature="MonthlyCharges",
                       unimportant_feature=None, std_multiplier=1.0, flip_fraction=0.3,
                       label_col="Churn", segment_col=None, segment_value=None, seed=0):
    if scenario_id == "S0":
        return scenario_s0_no_drift(current_pool_df, seed=seed)
    elif scenario_id == "S1":
        return scenario_s1_drift_important_feature(current_pool_df, feature=important_feature,
                                                     std_multiplier=std_multiplier, seed=seed)
    elif scenario_id == "S2":
        if unimportant_feature is None:
            raise ValueError("S2 requires 'unimportant_feature'.")
        unique_vals = set(current_pool_df[unimportant_feature].dropna().unique())
        if unique_vals <= {0, 1}:
            return scenario_shift_binary_feature(current_pool_df, feature=unimportant_feature,
                                                  flip_fraction=flip_fraction, seed=seed)
        return scenario_s2_drift_unimportant_feature(current_pool_df, feature=unimportant_feature,
                                                       std_multiplier=std_multiplier, seed=seed)
    elif scenario_id == "S3":
        return scenario_s3_concept_drift(current_pool_df, label_col=label_col,
                                          segment_col=segment_col, segment_value=segment_value,
                                          flip_fraction=flip_fraction, seed=seed)
    else:
        raise ValueError(f"Unknown scenario: {scenario_id}")