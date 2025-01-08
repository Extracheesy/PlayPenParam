import os
import pandas as pd
import numpy as np

# Scikit-Optimize
# pip install scikit-optimize if not installed
from skopt import gp_minimize
from skopt.space import Categorical, Integer, Real
from sklearn.model_selection import KFold

###############################################################################
# 1) READ CSV & DROP UNUSED
###############################################################################
path = "./batch_stats_df_merged"
input_csv_filename = "output_batch_stats_df_merged.csv"
input_csv = os.path.join(path, input_csv_filename)

df = pd.read_csv(input_csv)

# Drop only what you truly don't need. Keep TOTAL ORDERS, TOTAL DURATION,
# since we want them in the final output.
unused_cols = [
    "TYPE",
    "TRADE_TYPE",
    "TOTAL FEES PAID",
    "ICHIMOKU_PARAMS"
]
for c in unused_cols:
    if c in df.columns:
        df.drop(columns=[c], inplace=True)

###############################################################################
# 2) DEFINE THE METRICS AND THEIR PRIORITIES FOR THE COMPOSITE SCORE
###############################################################################
# Example priority tiers (adjust to your real metric names and weighting):
priority_1 = ["COMPARE RETURN"]  # highest weight
priority_2 = ["SHARPE RATIO", "TOTAL RETURN [%]", "WIN RATE [%]", "TOTAL TRADES", "MAX DRAWDOWN [%]"]
priority_3 = [
    "CALMAR RATIO",
    "SORTINO RATIO",
    "AVG WINNING TRADE [%]",
    "MIN VALUE",
    "MAX VALUE",
    "BEST TRADE [%]",
    "PROFIT FACTOR",
    "AVG LOSING TRADE [%]",
    "WORST TRADE [%]",
    "OMEGA RATIO",
    "EXPECTANCY"
]

all_metrics = priority_1 + priority_2 + priority_3

# Assign numeric weights
weight_map = {}
for m in priority_1:
    weight_map[m] = 3.0
for m in priority_2:
    weight_map[m] = 2.0
for m in priority_3:
    weight_map[m] = 1.0

# Convert these metrics to numeric and drop rows with missing values in them
for m in all_metrics:
    if m not in df.columns:
        raise ValueError(f"Missing metric column '{m}' in your CSV!")
    df[m] = pd.to_numeric(df[m], errors="coerce")

df.dropna(subset=all_metrics, inplace=True)

###############################################################################
# 3) COMPOSITE SCORE FUNCTION
###############################################################################
def row_composite_score(row):
    """
    Weighted sum of your priority metrics.
    We'll invert 'MAX DRAWDOWN [%]' so smaller => higher contribution.
    Adjust if you have other "smaller is better" metrics.
    """
    score = 0.0
    for metric in all_metrics:
        val = row[metric]
        w = weight_map[metric]
        if metric == "MAX DRAWDOWN [%]":
            val = -val  # invert drawdown
        score += w * val
    return score

###############################################################################
# 4) K-FOLD SCORING
###############################################################################
def kfold_score(group_df, params, n_splits=4):
    """
    1. Split group_df into K folds.
    2. For each test fold, filter rows by the proposed param values.
    3. Compute average row_composite_score(...) for that fold.
    4. Return mean across folds. If no rows match => -999999 penalty.
    """
    g = group_df.reset_index(drop=True)
    if len(g) < n_splits:
        return -999999

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []

    for _, test_idx in kf.split(g):
        test_data = g.iloc[test_idx].copy()
        for key, val in params.items():
            test_data = test_data.loc[test_data[key] == val]

        if test_data.empty:
            scores.append(-999999)
        else:
            fold_score = test_data.apply(row_composite_score, axis=1).mean()
            scores.append(fold_score)

    return np.mean(scores)

###############################################################################
# 5) MAKE OBJECTIVE FUNCTION (No Decorator)
###############################################################################
def make_objective_fn(group_df, param_cols, n_splits=4):
    """
    Return a function that gp_minimize calls with x (list of param values).
    We convert x -> param_dict -> feed to kfold_score -> return negative.
    """
    def objective_fn(x):
        param_dict = {}
        for i, col in enumerate(param_cols):
            param_dict[col] = x[i]

        avg_score = kfold_score(group_df, param_dict, n_splits=n_splits)
        return -avg_score  # negative => we want to maximize
    return objective_fn

###############################################################################
# 6) OPTIMIZE FOR ONE GROUP
###############################################################################
def optimize_for_group(group_df, param_cols):
    """
    1) Build 'dims' for each param col (discrete or numeric).
    2) Make objective_fn => gp_minimize => best param combo.
    3) Return best_score, best_params.
    """
    dims = []
    for col in param_cols:
        uniques = sorted(group_df[col].unique())
        if len(uniques) < 20:
            dims.append(Categorical(uniques, name=col))
        else:
            col_min, col_max = min(uniques), max(uniques)
            if np.allclose(uniques, np.round(uniques)):
                dims.append(Integer(int(col_min), int(col_max), name=col))
            else:
                dims.append(Real(float(col_min), float(col_max), name=col))

    objective_fn = make_objective_fn(group_df, param_cols, n_splits=4)

    res = gp_minimize(
        func=objective_fn,
        dimensions=dims,
        n_calls=30,
        n_random_starts=5,
        random_state=42
    )

    best_loss = res.fun
    best_score = -best_loss
    best_values = res.x

    best_params = {}
    for i, dim in enumerate(dims):
        best_params[dim.name] = best_values[i]

    return best_score, best_params

###############################################################################
# 7) MAIN: GROUP-BY => BEST PARAMS, THEN CAPTURE MORE COLUMNS
###############################################################################
if __name__ == "__main__":
    # The param columns to optimize
    param_cols = [
        "HIGH_OFFSET",
        "LOW_OFFSET",
        "ZEMA_LEN_BUY",
        "ZEMA_LEN_SELL",
        "SSL_ATR_PERIOD"
    ]

    ###########################################################################
    # The user wants the following columns in final output:
    # (make sure they exist in your CSV):
    ###########################################################################
    additional_cols = [
        "SHARPE RATIO",
        "END VALUE",
        "TOTAL RETURN [%]",
        "BENCHMARK RETURN [%]",
        "WIN RATE [%]",
        "TOTAL TRADES",
        "MAX DRAWDOWN [%]",
        "CALMAR RATIO",
        "SORTINO RATIO",
        "TOTAL ORDERS",
        "AVG WINNING TRADE [%]",
        "MIN VALUE",
        "MAX VALUE",
        "TOTAL DURATION",
        "BEST TRADE [%]",
        "PROFIT FACTOR",
        "AVG LOSING TRADE [%]",
        "WORST TRADE [%]",
        "OMEGA RATIO",
        "EXPECTANCY",
        "COMPARE RETURN"
    ]

    # Convert them to numeric if possible (some might be int, float, etc.)
    for c in additional_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    results = []
    grouped = df.groupby(["SYMBOL", "TIMEFRAME", "MA_TYPE"], as_index=False)

    for _, group_data in grouped:
        sym = group_data["SYMBOL"].iloc[0]
        tf = group_data["TIMEFRAME"].iloc[0]
        ma_t = group_data["MA_TYPE"].iloc[0]

        best_score, best_params = optimize_for_group(group_data, param_cols)

        # Now, find the matching rows in group_data for the best param combo
        matching_data = group_data.copy()
        for pcol, pval in best_params.items():
            matching_data = matching_data.loc[matching_data[pcol] == pval]

        # Build the output row
        row = {
            "SYMBOL": sym,
            "TIMEFRAME": tf,
            "MA_TYPE": ma_t,
            "best_score": best_score
        }
        row.update(best_params)

        if matching_data.empty:
            # No exact match => store NaN
            for c in additional_cols:
                row[c] = np.nan
        else:
            # For each additional col, store the mean (or any other stat you prefer)
            for c in additional_cols:
                if c in matching_data.columns:
                    row[c] = matching_data[c].mean()
                else:
                    row[c] = np.nan

        results.append(row)

    results_df = pd.DataFrame(results)
    print("\n=== BEST PARAMS PER (SYMBOL, TIMEFRAME, MA_TYPE) ===")
    print(results_df)

    # Save to CSV
    results_df.to_csv("best_params_per_group.csv", index=False)
