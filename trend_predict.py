import os
import pandas as pd

import time

from compare_AWS_vs_backtest import df_bt
from trend_model_manager import ModelManager
from trend_boost_model_manager import BoostModelTrainer

def eval_predictions(df, result_col='result', pred_col='preds'):
    """
    Compute:
      - accuracy: % of preds == result
      - recall  : % of true-1s correctly predicted (sensitivity)
      - precision: % of predicted-1s that are true-1s

    Returns:
      {'accuracy': ..., 'recall': ..., 'precision': ...}
    """
    # True positives: result==1 and preds==1
    tp = ((df[result_col] == 1) & (df[pred_col] == 1)).sum()
    # Counts of actual positives and predicted positives
    actual_pos = (df[result_col] == 1).sum()
    pred_pos   = (df[pred_col] == 1).sum()
    # Overall accuracy
    accuracy  = (df[pred_col] == df[result_col]).mean() * 100
    # Avoid division by zero
    recall    = (tp / actual_pos * 100) if actual_pos else float('nan')
    precision = (tp / pred_pos * 100) if pred_pos else float('nan')
    return {
        'accuracy': accuracy,
        'recall': recall,
        'precision': precision
    }

def print_df_info(df: pd.DataFrame) -> None:
    """
    Prints the first index, last index, and number of rows of the DataFrame.
    If the DataFrame is empty, it prints a message indicating that.
    """
    if df.empty:
        print("DataFrame is empty!")
        return

    first_idx = df.index[0]
    last_idx  = df.index[-1]
    size      = df.shape[0]

    print(f"First index: {first_idx}")
    print(f"Last index:  {last_idx}")
    print(f"Size (# rows): {size}")

if __name__ == "__main__":
    # your existing base path
    path = r'C:\Users\INTRADE\PycharmProjects\Analysis\ObelixParam\tend_analysis'
    os.makedirs(path, exist_ok=True)  # ensure the base exists
    os.chdir(path)  # cd into it
    print(f"Now in directory: {os.getcwd()}")

    # name of your sub-folder
    directory_result = "trend_predict"

    # join and create it
    result_path = os.path.join(path, directory_result)
    os.makedirs(result_path, exist_ok=True)  # makes trnd_results (and parents) if needed

    # optionally cd into the new results dir
    os.chdir(result_path)
    print(f"Now in directory: {os.getcwd()}")

    lst_start_date = [
        '2025-04-01T00:00:00Z',
        '2024-01-01T00:00:00Z',
        '2024-12-17T00:00:00Z',
        '2025-01-01T00:00:00Z',
        '2024-03-13T00:00:00Z',
        '2025-03-01T00:00:00Z'

    ]

    dct_start_date = {
        1: "2024-01-01T00:00:00Z",
        2: "2024-12-17T00:00:00Z",
        3: "2025-01-01T00:00:00Z",
        4: "2024-03-13T00:00:00Z",
        5: "2025-03-01T00:00:00Z",
        6: "2025-04-01T00:00:00Z"
    }

    lst_start_date = [
        # '2025-04-01T00:00:00Z'
        "2024-01-01T00:00:00Z"
    ]

    dct_start_date = {
        1: "2024-01-01T00:00:00Z"
    }

    str_cutoff = '2025-04-01T00:00:00Z'

    for start_date in lst_start_date:
        symbol = "BTCUSDT"

        # start_date = dct_start_date[6]

        data_dir = os.path.join(result_path, symbol + "_" + start_date.rstrip('Z')[:16].replace('T', '-').replace(':',
                                                                                                                  '-') + "_predict_trend")
        os.makedirs(data_dir, exist_ok=True)

        filename = symbol + "_" + start_date.rstrip('Z')[:16].replace('T', '-').replace(':', '-') + "_combined.csv"
        file_path = os.path.join(data_dir, filename)

        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            print("File loaded into DataFrame!")
        else:
            print(f"❌ File not found: {file_path}")
            exit(1)

        # cutoff = pd.to_datetime(str_cutoff)
        # df_pruned = df[df.index <= cutoff]
        df.set_index('timestamp', inplace=True, drop=True)
        df = df.loc[:str_cutoff]
        df_training = df.copy()

        data_dir_test = os.path.join(result_path, symbol + "_" + str_cutoff.rstrip('Z')[:16].replace('T', '-').replace(':',
                                                                                                                       '-') + "_predict_trend")
        filename_test = symbol + "_" + str_cutoff.rstrip('Z')[:16].replace('T', '-').replace(':', '-') + "_combined.csv"

        file_path_test = os.path.join(data_dir_test, filename_test)

        if os.path.exists(file_path_test):
            df = pd.read_csv(file_path)
            print("File loaded into DataFrame!")
        else:
            print(f"❌ File not found: {file_path}")
            exit(1)
        df.set_index('timestamp', inplace=True, drop=True)
        df = df.loc[str_cutoff:]
        df_test = df.copy()

        feature_cols = [
            'kama_1m', 'kama_5m', 'kama_15m', 'kama_30m', 'kama_1h',
            'adx_tsi_1m', 'adx_tsi_5m', 'adx_tsi_15m', 'adx_tsi_30m', 'adx_tsi_1h',
            'aroon_tsi_1m', 'aroon_tsi_5m', 'aroon_tsi_15m', 'aroon_tsi_30m', 'aroon_tsi_1h',
            'donchian_obv_1m', 'donchian_obv_5m', 'donchian_obv_15m', 'donchian_obv_30m', 'donchian_obv_1h',
            'double_supertrend_1m', 'double_supertrend_5m', 'double_supertrend_15m',
            'double_supertrend_30m', 'double_supertrend_1h',
            'hma_wavetrend_1m', 'hma_wavetrend_5m', 'hma_wavetrend_15m',
            'hma_wavetrend_30m', 'hma_wavetrend_1h'
        ]

        keep_cols = feature_cols + ['profit_flag']

        df_training = df_training[keep_cols].copy()
        df_training = df_training.rename(columns={'profit_flag': 'result'})

        df_test = df_test[keep_cols].copy()
        df_test = df_test.rename(columns={'profit_flag': 'result'})

        print("df_training: ")
        print_df_info(df_training)
        print("df_test: ")
        print_df_info(df_test)

        PREDICT_BOOST_MODEL = False

        if PREDICT_BOOST_MODEL:
            manager = ModelManager(df)
            TRAIN_SAVE = False
            if TRAIN_SAVE:
                manager.train_and_save_all()
            else:
                manager.load_all()
            preds = manager.predict()

            N = len(preds["xgb"])
            df_results = df.tail(N)
            df_results["xgb"] = preds["xgb"]
            df_results["rf"] = preds["rf"]
            df_results["lstm_a"] = preds["lstm_a"]
            df_results["lstm_b"] = preds["lstm_b"]
            df_results["transformer"] = preds["lstm_b"]
            df_results["vote"] =preds["vote"]

            # Assuming df_results is your DataFrame and 'result' is the actual outcome column
            prediction_columns = ['xgb', 'rf', 'lstm_a', 'lstm_b', 'transformer', 'vote']

            # Initialize dictionaries to store match percentages
            overall_match_percentages = {}
            positive_match_percentages = {}
            positive_found_total = {}
            positive_match = {}

            # Total number of rows in the DataFrame
            total_rows = len(df_results)

            # Total number of positive cases in the actual results
            total_positive = (df_results['result'] == 1).sum()

            for col in prediction_columns:
                # Overall match percentage
                overall_matches = (df_results[col] == df_results['result']).sum()
                overall_match_percentages[col] = (overall_matches / total_rows) * 100

                total_positive_found = (df_results[col] == 1).sum()
                # Positive class match percentage
                positive_matches = ((df_results[col] == 1) & (df_results['result'] == 1)).sum()
                positive_match_percentages[col] = (positive_matches / total_positive_found) * 100 if total_positive_found > 0 else 0

                positive_found_total[col] = total_positive_found
                positive_match[col] = positive_matches

            # Convert the results into DataFrames for better readability
            overall_match_df = pd.DataFrame.from_dict(overall_match_percentages, orient='index',
                                                      columns=['Overall Match %'])

            positive_match_df = pd.DataFrame.from_dict(positive_match_percentages, orient='index',
                                                       columns=['Positive Match %'])

            positive_found_df = pd.DataFrame.from_dict(positive_found_total, orient='index',
                                                       columns=['Positive Found Tot'])

            positive_match_tot_df = pd.DataFrame.from_dict(positive_match, orient='index',
                                                           columns=['Positive Match Tot'])

            match_df = pd.concat(
                [overall_match_df, positive_match_df, positive_found_df, positive_match_tot_df],
                axis=1
            )

            # Optionally, sort the DataFrame by 'Overall Match %' in descending order
            match_df = match_df.sort_values(by='Overall Match %', ascending=False)

            print(match_df.to_string())
        else:
            # df.replace(1.0, "UP", inplace=True)
            # df.replace(0.0, "FLAT", inplace=True)
            # df.replace(-1.0, "DOWN", inplace=True)
            # 2) Instantiate & train

            start = time.perf_counter()

            # trainer = BoostModelTrainer(df_training, feature_cols=feature_cols, target_col="result")

            X_train = df_training.drop(columns=['result'])
            y_train = df_training['result']
            trainer = BoostModelTrainer(
                pd.concat([X_train, y_train], axis=1),
                apply_smote=False,
                feature_selection=False
            )

            TRAIN_MODEL = True

            if TRAIN_MODEL:
                trainer.train_all()

            end = time.perf_counter()
            elapsed = end - start

            # split into h : m : s
            hours, rem = divmod(elapsed, 3600)
            minutes, seconds = divmod(rem, 60)

            print(
                f"Training completed: {hours:.0f}h "
                f"{minutes:.0f}m "
                f"{seconds:.4f}s "
                f"({elapsed:.4f} seconds total)"
            )

            if TRAIN_MODEL:
                trainer.save_all()

            if TRAIN_MODEL:
                trainer.load_all()

            # X_test = pd.read_csv("my_test_features.csv")
            # y_test = pd.read_csv("my_test_labels.csv")["result"]

            X_test = df_test
            y_test = df_test["result"]

            X_test_aligned = X_test.reindex(columns=trainer.orig_feature_cols, fill_value=0)

            print("X_test_aligned:")
            print_df_info(X_test_aligned)
            print("y_test:")
            print_df_info(y_test)

            # metrics_df = trainer.evaluate(X_test, y_test)
            metrics_df = trainer.evaluate(X_test_aligned, y_test)

            print(metrics_df.to_string())

            new_samples = X_test.head(5)  # example subset
            new_samples = X_test  # example subset
            preds = trainer.predict_ensemble(new_samples)

            X_test["preds"] = preds

            out_fn = f"{symbol}_X_test_predict_excel.csv"
            X_test.to_csv(
                os.path.join(data_dir, out_fn),
                sep=';',  # semicolon as field separator
                decimal=',',  # comma for the decimal point
                index=True
            )

            metrics = eval_predictions(X_test)
            print(f"Accuracy:  {metrics['accuracy']:.2f}%")
            print(f"Recall :    {metrics['recall']:.2f}%")
            print(f"Precision: {metrics['precision']:.2f}%")
