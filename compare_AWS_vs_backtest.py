import os
import pandas as pd
import numpy as np

# 1) Filenames and output directory
filename_1 = "49_BITGET_BTCUSDT.csv"
filename_2 = "StrategyObelix_prod_BTC_BTC_JFN9_1m.csv"

filename_1 = "129_BINANCE_BTCUSDT.csv"
filename_2 = "StrategyObelix_prod_BTC_BTC_8EEA_1m.csv"

output_dir = "./test_multi_trend_selection_test_7/compare_AWS_vs_backtest"
os.makedirs(output_dir, exist_ok=True)

# 2) Read AWS and backtest CSVs
file1 = os.path.join(output_dir, filename_1)
file2 = os.path.join(output_dir, filename_2)
df_aws = pd.read_csv(file1, index_col=0, parse_dates=True)
df_bt  = pd.read_csv(file2, index_col="released_dt", parse_dates=True)

# 3) Shift backtest up by one row, then drop the all-NaN tail
# df_bt = df_bt.shift(-1).dropna(how="all")

df_bt.index = df_bt.index.tz_localize('UTC')

# 4) Trim both to their overlapping time span
start = max(df_aws.index.min(), df_bt.index.min())
end   = min(df_aws.index.max(), df_bt.index.max())
df_aws = df_aws.loc[start:end].copy()
df_bt  = df_bt.loc[start:end].copy()

# 5) Suffix columns and concat side by side
aws = df_aws.add_suffix("_aws")
bt  = df_bt.add_suffix("_backtest")
combined = pd.concat([aws, bt], axis=1)

# 6) Comparison helper functions
def pct_compare(a, b):
    if pd.isna(a) or pd.isna(b):
        return None
    return "IDENTICAL" if a == b else f"{(b - a) / a * 100:.2f}%"

def eq_compare(a, b):
    if pd.isna(a) or pd.isna(b):
        return None
    # numeric (e.g. TREND = -1/0/1)?
    if isinstance(a, (int, float, np.integer, np.floating)) and \
       isinstance(b, (int, float, np.integer, np.floating)):
        return "IDENTICAL" if a == b else "DIFFERENT"
    # otherwise coerce to lowercase strings for TRUE/FALSE
    a_str = str(a).strip().lower()
    b_str = str(b).strip().lower()
    return "IDENTICAL" if a_str == b_str else "DIFFERENT"

# 7) Define column mappings
numeric_pairs = {
    "close"          : "close",
    "buy_adj"        : "zerolag_ma_buy_adj",
    "sell_adj"       : "zerolag_ma_sell_adj",
    "close_high_tf"  : "high_tf_close",
    "highest_window" : "s_highest_window",
    "lowest_window"  : "s_lowest_window",
}

signal_pairs = {
    "TREND"       : "trend_signal",
    "buy_signal"  : "signal_buy",
    "sell_signal" : "signal_sell",
}

# 8) Build comparison columns
for aws_base, bt_base in numeric_pairs.items():
    c_aws, c_bt = f"{aws_base}_aws", f"{bt_base}_backtest"
    comp_col = f"{aws_base}_comparison"
    combined[comp_col] = combined.apply(
        lambda r: pct_compare(r[c_aws], r[c_bt]), axis=1
    )

for aws_base, bt_base in signal_pairs.items():
    c_aws, c_bt = f"{aws_base}_aws", f"{bt_base}_backtest"
    comp_col = f"{aws_base}_match"
    combined[comp_col] = combined.apply(
        lambda r: eq_compare(r[c_aws], r[c_bt]), axis=1
    )

# 9) Reorder so each trio is grouped, then keep all other columns
ordered = []
# numeric trios
for aws_base, bt_base in numeric_pairs.items():
    ordered += [
        f"{aws_base}_aws",
        f"{bt_base}_backtest",
        f"{aws_base}_comparison",
    ]
# signal trios
for aws_base, bt_base in signal_pairs.items():
    ordered += [
        f"{aws_base}_aws",
        f"{bt_base}_backtest",
        f"{aws_base}_match",
    ]
# any remaining columns
remaining = [c for c in combined.columns if c not in ordered]
combined = combined[ordered + remaining]

# 10) Save both comma- and semicolon-delimited outputs
base = "aws_vs_backtest_grouped_comparisons"
combined.to_csv(os.path.join(output_dir, f"{base}.csv"))
combined.to_csv(os.path.join(output_dir, f"{base}_semicolon.csv"), sep=";")

# 11) Preview
print(combined.head())
