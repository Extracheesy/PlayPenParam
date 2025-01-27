import pandas as pd
import numpy as np
import os

def main():
    # Step 1: Load the CSV file
    file_path = './batch_stats_df_merged/batch_stats_df_merged.csv'
    file_path = "./results_vbtpro/portfolio_stats_summary.csv"
    df = pd.read_csv(file_path)

    # Step 2: Drop unwanted columns
    columns_to_drop = [
        'TYPE', 'TRADE_TYPE', 'TOTAL FEES PAID', 'TOTAL ORDERS', 'TOTAL DURATION', 'ICHIMOKU_PARAMS'
    ]
    df = df.drop(columns=columns_to_drop, errors='ignore')

    # Step 3: Compute Compare Return
    df['COMPARE RETURN'] = df['TOTAL RETURN [%]'] - df['BENCHMARK RETURN [%]']
    df['COMPARE RETURN FLAG'] = df['TOTAL RETURN [%]'] > df['BENCHMARK RETURN [%]']

    # Step 4: Define scoring weights
    weights_1 = {
        'SHARPE RATIO': 5,
        'TOTAL RETURN [%]': 5,
        'WIN RATE [%]': 5,
        'MAX DRAWDOWN [%]': -5,  # Negative weight as lower is better
        'COMPARE RETURN': 5,
        'TOTAL TRADES': 3,
        'CALMAR RATIO': 3,
        'SORTINO RATIO': 3,
        'AVG WINNING TRADE [%]': 3,
        'MAX VALUE': 3,
        'BEST TRADE [%]': 2,
        'PROFIT FACTOR': 2,
        'AVG LOSING TRADE [%]': -2,
        'WORST TRADE [%]': -2,
        'OMEGA RATIO': 2,
        'EXPECTANCY': 2
    }

    weights_2 = {
        'SHARPE RATIO': 10,
        'TOTAL RETURN [%]': 15,
        'WIN RATE [%]': 10,
        'MAX DRAWDOWN [%]': -15,  # Negative weight as lower is better
        'COMPARE RETURN': 10,
        'TOTAL TRADES': 5,
        'CALMAR RATIO': 5,
        'SORTINO RATIO': 5,
        'AVG WINNING TRADE [%]': 5,
        'MAX VALUE': 3,
        'BEST TRADE [%]': 5,
        'PROFIT FACTOR': 5,
        'AVG LOSING TRADE [%]': -5,
        'WORST TRADE [%]': -5,
        'OMEGA RATIO': 3,
        'EXPECTANCY': 5
    }

    weights_3 = {
        'SHARPE RATIO': 10,
        'TOTAL RETURN [%]': 15,
        'WIN RATE [%]': 15,
        'MAX DRAWDOWN [%]': -15,  # Negative weight as lower is better
        'COMPARE RETURN': 10,
        'TOTAL TRADES': 10,  # Weight is 0 for values below 50
        'CALMAR RATIO': 5,
        'SORTINO RATIO': 5,
        'AVG WINNING TRADE [%]': 5,
        'MAX VALUE': 3,
        'BEST TRADE [%]': 5,
        'PROFIT FACTOR': 5,
        'AVG LOSING TRADE [%]': -5,
        'WORST TRADE [%]': -5,
        'OMEGA RATIO': 3,
        'EXPECTANCY': 5
    }

    # Step 5: Normalize columns and calculate scores per SYMBOL
    normalized_df_list = []
    for symbol, group in df.groupby('SYMBOL'):
        normalized_group = group.copy()
        for weights, score_col in [(weights_1, 'SCORE_1'), (weights_2, 'SCORE_2'), (weights_3, 'SCORE_3')]:
            temp_group = normalized_group.copy()
            for column, weight in weights.items():
                if column == 'TOTAL TRADES' and score_col == 'SCORE_3':
                    # Set weight to 0 for TOTAL TRADES below 50 in SCORE_3
                    temp_group[column + '_WEIGHT'] = temp_group[column].apply(lambda x: 0 if x < 50 else weight)
                else:
                    temp_group[column + '_WEIGHT'] = weight

                if column in group.columns:
                    # Normalize each column to range [0, 1]
                    min_val = group[column].min()
                    max_val = group[column].max()
                    if min_val != max_val:  # Avoid division by zero
                        temp_group[column] = (group[column] - min_val) / (max_val - min_val)
                    else:
                        temp_group[column] = 0
                    # Apply weights (negative for reverse scoring)
                    temp_group[column] *= temp_group[column + '_WEIGHT']

            # Compute final score for each row
            temp_group[score_col] = temp_group[list(weights.keys())].sum(axis=1)
            # Scale SCORE to range 0-100
            min_score = temp_group[score_col].min()
            max_score = temp_group[score_col].max()
            if min_score != max_score:  # Avoid division by zero
                temp_group[score_col] = 100 * (temp_group[score_col] - min_score) / (max_score - min_score)
            else:
                temp_group[score_col] = 100
            normalized_group[score_col] = temp_group[score_col]
        normalized_df_list.append(normalized_group)

    # Combine all groups back into a single dataframe
    normalized_df = pd.concat(normalized_df_list)

    # Step 6: Sort by SYMBOL and SCORE_1
    normalized_df = normalized_df.sort_values(by=['SYMBOL', 'SCORE_1'], ascending=[True, False])

    # Step 7: Save the result to a new CSV in the weighted_score directory
    output_dir = './weighted_score'
    output_dir = "./results_vbtpro/"

    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'scored_results.csv')
    normalized_df.to_csv(output_file, index=False)

    print(f"Scored results saved to {output_file}")

if __name__ == '__main__':
    main()
