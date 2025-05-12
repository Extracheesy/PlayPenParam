import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)
from crypto_trend_predictor import CryptoPredictor

def evaluate_model(model_type, file_paths,
                   horizon='15m', test_ratio=0.2,
                   feature_selection=True,
                   hyperparam_tuning=False):
    """
    Trains and backtests one model type, prints confusion matrix and returns a dict of backtest metrics.
    Also returns the confusion matrix array for further inspection.
    """
    predictor = CryptoPredictor(model_type=model_type, mode='classification')
    predictor.load_data(file_paths)
    predictor.train(
        horizon=horizon,
        test_ratio=test_ratio,
        feature_selection=feature_selection,
        hyperparam_tuning=hyperparam_tuning
    )
    # Predictions
    y_true = predictor.y_test
    y_pred = predictor.pipeline.predict(predictor.X_test)

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Print
    print(f"\nConfusion Matrix for {model_type}:")
    print(cm)
    print(f"\nClassification Report for {model_type}:")
    print(classification_report(y_true, y_pred, zero_division=0))

    # Compute additional metrics
    acc   = accuracy_score(y_true, y_pred)
    prec  = precision_score(y_true, y_pred, zero_division=0)
    rec   = recall_score(y_true, y_pred, zero_division=0)
    f1    = f1_score(y_true, y_pred, zero_division=0)
    try:
        proba = predictor.pipeline.predict_proba(predictor.X_test)[:, 1]
        roc   = roc_auc_score(y_true, proba)
    except Exception:
        roc = None

    return {
        'model':           model_type,
        'accuracy':        acc,
        'precision':       prec,
        'recall':          rec,
        'f1_score':        f1,
        'roc_auc':         roc,
        'confusion_matrix': cm  # numpy array
    }

if __name__ == "__main__":
    # 1. Prepare file paths
    BASE_DIR = Path(
        r"C:\Users\INTRADE\PycharmProjects\Analysis\ObelixParam"
        r"\tend_analysis\trend_predict\crypto_predictor"
        r"\BTCUSDT_2025-03-01-00-00_data_ohlcv"
    )
    intervals = ['1m', '5m', '15m', '30m', '1h']
    file_paths = {iv: BASE_DIR / 'data' / f"BTCUSDT_{iv}.csv" for iv in intervals}

    # 2. Compare static backtest across models
    model_types = ['xgboost', 'lightgbm', 'random_forest', 'mlp', 'stacked']
    static_results = []

    for mt in model_types:
        print(f"\n→ Evaluating {mt} (static backtest)...")
        res = evaluate_model(
            mt,
            file_paths,
            horizon='15m',
            test_ratio=0.2,
            feature_selection=True,
            hyperparam_tuning=True
        )
        static_results.append(res)

    # 3. Show comparison table
    df_static = pd.DataFrame([{k: v for k, v in r.items() if k != 'confusion_matrix'} for r in static_results])
    print("\n=== Static Backtest Comparison ===")
    print(df_static.to_string(index=False))
    df_static.to_csv("static_backtest_results.csv", index=False)

    # 4. Display all confusion matrices
    print("\n=== Confusion Matrices ===")
    for res in static_results:
        print(f"\nModel: {res['model']}")
        print(res['confusion_matrix'])

    # 5. Walk-forward backtest on the top model by F1-score
    best_model = df_static.sort_values('f1_score', ascending=False).iloc[0]['model']
    print(f"\n=== Walk-Forward Backtest on Best Model: {best_model} ===")

    wf_predictor = CryptoPredictor(model_type=best_model, mode='classification')
    wf_predictor.load_data(file_paths)
    wf_predictor.train(
        horizon='15m',
        test_ratio=0.2,
        feature_selection=True,
        hyperparam_tuning=False
    )
    wf_predictor.walk_forward_backtest(
        horizon='15m',
        initial_train_frac=0.6,
        step_bars=500
    )
