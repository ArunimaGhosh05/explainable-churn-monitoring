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
    """
    df = current_pool_df.sample(frac=0.8, random_state=seed).reset_index(drop=True).copy()
    shift_amount = df[feature].std() * std_multiplier
    df[feature] = df[feature] + shift_amount
    return df


def get_scenario_data(scenario_id, current_pool_df, important_feature="MonthlyCharges",
                       unimportant_feature=None, std_multiplier=1.0, seed=0):
    """
    Convenience function: pass a scenario ID, get back the current-data version for that scenario.
    """
    if scenario_id == "S0":
        return scenario_s0_no_drift(current_pool_df, seed=seed)
    elif scenario_id == "S1":
        return scenario_s1_drift_important_feature(current_pool_df, feature=important_feature,
                                                     std_multiplier=std_multiplier, seed=seed)
    elif scenario_id == "S2":
        if unimportant_feature is None:
            raise ValueError("S2 requires 'unimportant_feature' (from SHAP ranking, lowest importance).")
        return scenario_s2_drift_unimportant_feature(current_pool_df, feature=unimportant_feature,
                                                       std_multiplier=std_multiplier, seed=seed)
    else:
        raise ValueError(f"Unknown scenario: {scenario_id}")
