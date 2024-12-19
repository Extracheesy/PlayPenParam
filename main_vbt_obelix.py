import input_data
import main_obelix
import main_obelix_pro

import os


if __name__ == "__main__":
    results = {}
    stats_list = []  # To collect stats for all symbols and timeframes

    # data = fetch_data(symbols, tf[0], start_date, end_date)
    # Ensure the results directory exists
    if not os.path.exists('results'):
        os.makedirs('results')



    lst_stats = []
    for trade_type in input_data.lst_trade_type:
        for timeframe in input_data.tf:
            print(f"Running strategy for {input_data.symbols} on {timeframe} timeframe with trade_type='{trade_type}'.")
            strategy = main_obelix_pro.IchimokuZemaStrategy(input_data.symbols, timeframe, input_data.start_date, input_data.end_date, trade_type)
            lst_stats.extend(strategy.get_results(timeframe, trade_type))


