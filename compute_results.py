import os
import pandas as pd
import numpy as np

# Scikit-Optimize
# If you don't have it installed:
# !pip install scikit-optimize

from skopt import gp_minimize
from skopt.space import Categorical, Integer, Real
from skopt.utils import use_named_args

# For K-Fold cross-validation splits
from sklearn.model_selection import KFold

###############################################################################
# 1) LOAD CSV AND DROP UNUSED COLUMNS
###############################################################################

path = "./batch_stats_df_merged"
input_csv_filename = "output_batch_stats_df_merged.csv"  # Replace with your actual input file if needed

# Construct full path
input_csv = os.path.join(path, input_csv_filename)

df = pd.read_csv(input_csv)  # <-- Now reading from the constructed path

unused_cols = ["TYPE", "TRADE_TYPE", "TOTAL FEES PAID", "TOTAL ORDERS", "TOTAL DURATION"]
for c in unused_cols:
    if c in df.columns:
        df.drop(columns=[c], inplace=True)

# Check for required grouping columns
for col in ["SYMBOL", "TIMEFRAME", "MA_TYPE"]:
    if col not in df.columns:
        raise ValueError(f"Missing required column '{col}' in CSV!")

# Parameter columns you want to optimize
param_cols = ["HIGH_OFFSET", "LOW_OFFSET", "ZEMA_LEN_BUY", "ZEMA_LEN_SELL", "SSL_ATR_PERIOD"]
for col in param_cols:
    if col not in df.columns:
        raise ValueError(f"Missing parameter column '{col}' in CSV!")

###############################################################################
# 2) DEFINE METRICS & THEIR WEIGHTS (THREE PRIORITY TIERS)
###############################################################################

priority_1 = ["COMPARE RETURN"]
priority_2 = ["SHARPE RATIO", "TOTAL RETURN [%]", "WIN RATE [%]", "TOTAL TRADES", "MAX DRAWDOWN [%]"]
priority_3 = [
    "CALMAR RATIO", "SORTINO RATIO", "AVG WINNING TRADE [%]", "MIN VALUE", "MAX VALUE",
    "BEST TRADE [%]", "PROFIT FACTOR", "AVG LOSING TRADE [%]", "WORST TRADE [%]",
    "OMEGA RATIO", "EXPECTANCY"
]
all_metrics = priority_1 + priority_2 + priority_3

weight_map = {}
for m in priority_1:
    weight_map[m] = 3.0
for m in priority_2:
    weight_map[m] = 2.0
for m in priority_3:
    weight_map[m] = 1.0

# Ensure these columns exist & are numeric
for m in all_metrics:
    if m not in df.columns:
        raise ValueError(f"Missing metric column '{m}' in your CSV!")
    df[m] = pd.to_numeric(df[m], errors="coerce")

df.dropna(subset=all_metrics, inplace=True)

###############################################################################
# 3) COMPOSITE SCORE FUNCTION
###############################################################################
# - We'll do a direct weighted sum,
# - We'll invert "MAX DRAWDOWN [%]" so that a smaller drawdown => higher score.

def row_composite_score(row):
    score = 0.0
    for metric in all_metrics:
        w = weight_map[metric]
        val = row[metric]
        if metric == "MAX DRAWDOWN [%]":
            # invert so smaller drawdown => higher contribution
            val = -val
        score += w * val
    return score

###############################################################################
# 4) K-FOLD SCORING FUNCTION (instead of walk-forward)
###############################################################################
def kfold_score(group, params, n_splits=4):
    """
    Perform K-fold cross-validation in the group.
    For each fold, we treat the test fold as "out-of-sample" for these parameters.

    Steps:
    1. Split the rows of the group into K folds.
    2. For each fold, filter rows that match `params` exactly.
    3. Compute the average composite score in the test data.
    4. Collect these scores and average them.

    Return that average (larger is better).
    If no rows match the param set in some test fold, we penalize heavily.
    """
    g = group.reset_index(drop=True)
    n = len(g)
    if n < n_splits:
        # Not enough data => big penalty
        return -999999

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []

    for train_idx, test_idx in kf.split(g):
        test_data = g.iloc[test_idx].copy()

        # Filter test_data to rows that match 'params'
        for param_col, param_val in params.items():
            test_data = test_data.loc[test_data[param_col] == param_val]

        if test_data.empty:
            scores.append(-999999)
        else:
            comp_scores = test_data.apply(row_composite_score, axis=1)
            scores.append(comp_scores.mean())

    return np.mean(scores)

###############################################################################
# 5) OBJECTIVE FUNCTION FOR GP_MINIMIZE
###############################################################################
def make_objective_fn(group, n_splits=4):
    """
    Build a closure that scikit-optimize will call with param values.
    We'll do K-fold scoring, then return negative to maximize our composite.
    """

    @use_named_args(dimensions=[])
    def objective_fn(**params):
        avg_score = kfold_score(group, params, n_splits=n_splits)
        return -avg_score  # negative => we want to maximize

    return objective_fn

###############################################################################
# 6) BUILD PARAM SPACE
###############################################################################
def build_param_space_for_group(group_df, param_cols):
    dims = []
    for col in param_cols:
        uniques = group_df[col].unique()
        # if fewer than 20 unique => treat as discrete
        # else we assume numeric range
        if len(uniques) < 20:
            sorted_vals = sorted(uniques)
            dims.append(Categorical(sorted_vals, name=col))
        else:
            col_min, col_max = group_df[col].min(), group_df[col].max()
            if np.allclose(uniques, uniques.astype(int)):
                dims.append(Integer(int(col_min), int(col_max), name=col))
            else:
                dims.append(Real(float(col_min), float(col_max), name=col))
    return dims

###############################################################################
# 7) MAIN LOOP OVER (SYMBOL, TIMEFRAME, MA_TYPE)
###############################################################################
results = []

grouped = df.groupby(["SYMBOL", "TIMEFRAME", "MA_TYPE"], as_index=False)
for _, group_data in grouped:
    sym = group_data["SYMBOL"].iloc[0]
    tf = group_data["TIMEFRAME"].iloc[0]
    ma_t = group_data["MA_TYPE"].iloc[0]

    dims = build_param_space_for_group(group_data, param_cols)
    objective_fn = make_objective_fn(group_data, n_splits=4)
    objective_fn.__skopt_dimensions__ = dims  # Attach dimension info

    res = gp_minimize(
        func=objective_fn,
        dimensions=dims,
        n_calls=30,         # adjust as needed
        n_random_starts=5,  # random exploration
        random_state=42
    )

    best_loss = res.fun
    best_score = -best_loss
    best_params = res.x

    param_dict = {}
    for dim, val in zip(dims, best_params):
        param_dict[dim.name] = val

    row = {
        "SYMBOL": sym,
        "TIMEFRAME": tf,
        "MA_TYPE": ma_t,
        "best_score": best_score
    }
    row.update(param_dict)
    results.append(row)

results_df = pd.DataFrame(results)
print("\n=== BEST PARAMS PER (SYMBOL, TIMEFRAME, MA_TYPE) ===")
print(results_df)

# Save to CSV
results_df.to_csv("best_params_kfold.csv", index=False)
