import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")

warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

# Imputation and preprocessing
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Model selection & tuning
from sklearn.model_selection import TimeSeriesSplit, KFold, cross_val_score
import optuna

# Metrics
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_auc_score, classification_report
)

# Base learners and stacking
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.neural_network import MLPClassifier

# Persistence
import joblib


class CryptoPredictor:
    def __init__(self, model_type='xgboost', mode='classification'):
        self.model_type = model_type
        self.mode = mode
        self.pipeline = None
        self.data_frames = {}
        self.merged_data = None
        self.features = []
        self.X_train = self.X_test = None
        self.y_train = self.y_test = None
        self.horizon_minutes = None

    def _parse_timeframe(self, tf_str):
        if isinstance(tf_str, str) and tf_str.endswith('m'):
            return int(tf_str[:-1])
        if isinstance(tf_str, str) and tf_str.endswith('h'):
            return int(tf_str[:-1]) * 60
        return int(tf_str)

    def load_data(self, file_paths):
        for tf, path in file_paths.items():
            df = pd.read_csv(path)
            df.columns = [c.capitalize() for c in df.columns]
            time_col = next((c for c in df.columns if c.lower() in ['time','timestamp','date']), None)
            if time_col:
                df.rename(columns={time_col:'Timestamp'}, inplace=True)
                try:
                    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms')
                except:
                    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
                df.sort_values('Timestamp', inplace=True)
            else:
                df['Timestamp'] = np.arange(len(df))
            self.data_frames[tf] = df.reset_index(drop=True)

    def _compute_indicators_for_df(self, df, prefix=""):
        data = df.sort_values('Timestamp').reset_index(drop=True).copy()
        close = data['Close']
        delta = close.diff(); up = delta.clip(lower=0); down = -delta.clip(upper=0)
        data[f'{prefix}RSI'] = 100 - (100 / (1 + up.rolling(14).mean() / (down.rolling(14).mean() + 1e-10)))
        data[f'{prefix}EMA_fast'] = close.ewm(span=10, adjust=False).mean()
        data[f'{prefix}EMA_slow'] = close.ewm(span=50, adjust=False).mean()
        data[f'{prefix}SMA_20'] = close.rolling(20).mean()
        m20 = close.rolling(20).mean(); s20 = close.rolling(20).std()
        data[f'{prefix}BB_upper'] = m20 + 2*s20
        data[f'{prefix}BB_lower'] = m20 - 2*s20
        data[f'{prefix}BB_width'] = (data[f'{prefix}BB_upper'] - data[f'{prefix}BB_lower']) / (m20 + 1e-10)
        exp12 = close.ewm(span=12, adjust=False).mean()
        exp26 = close.ewm(span=26, adjust=False).mean()
        macd_line = exp12 - exp26
        sig = macd_line.ewm(span=9, adjust=False).mean()
        data[f'{prefix}MACD_line'] = macd_line
        data[f'{prefix}MACD_signal'] = sig
        data[f'{prefix}MACD_hist'] = macd_line - sig
        cols = ['Timestamp'] + [c for c in data.columns if c.startswith(prefix)]
        return data[cols]

    def _build_merged(self, data_frames):
        base_tf = min(data_frames.keys(), key=lambda t: self._parse_timeframe(t))
        merged = pd.merge_asof(
            data_frames[base_tf].sort_values('Timestamp'),
            self._compute_indicators_for_df(data_frames[base_tf], prefix=f"{base_tf}_").sort_values('Timestamp'),
            on='Timestamp', direction='forward'
        )
        for tf, df in data_frames.items():
            if tf == base_tf: continue
            ind = self._compute_indicators_for_df(df, prefix=f"{tf}_").sort_values('Timestamp')
            merged = pd.merge_asof(
                merged.sort_values('Timestamp'), ind,
                on='Timestamp', direction='nearest'
            )
        return merged.reset_index(drop=True)

    def prepare_features(self, horizon_minutes):
        df = self.merged_data.copy()
        base_int = self._parse_timeframe(min(self.data_frames.keys(), key=lambda t: self._parse_timeframe(t)))
        steps = max(1, int(round(horizon_minutes / base_int)))
        df['FuturePrice'] = df['Close'].shift(-steps)
        if self.mode == 'classification':
            df['Target'] = (df['FuturePrice'] > df['Close']).astype(int)
        else:
            df['Target'] = df['FuturePrice'] - df['Close']
        df.dropna(subset=['Target'], inplace=True)
        drop_cols = ['Timestamp','Open','High','Low','Close','Volume','FuturePrice','Target']
        X = df.drop(columns=[c for c in drop_cols if c in df])
        y = df['Target']
        return X, y

    def train(self, horizon='15m', test_ratio=0.2, feature_selection=False, hyperparam_tuning=False):
        assert self.data_frames, "Call load_data first."
        self.horizon_minutes = self._parse_timeframe(horizon)
        self.merged_data = self._build_merged(self.data_frames)
        X, y = self.prepare_features(self.horizon_minutes)
        split = int(len(X) * (1 - test_ratio))
        X_train_full, X_test = X.iloc[:split], X.iloc[split:]
        y_train_full, y_test = y.iloc[:split], y.iloc[split:]
        if feature_selection:
            temp = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
            temp.fit(X_train_full, y_train_full)
            imp = pd.Series(temp.feature_importances_, index=X_train_full.columns)
            top = imp.nlargest(10).index
            X_train_full, X_test = X_train_full[top], X_test[top]
            self.features = list(top)
        else:
            self.features = list(X_train_full.columns)
        if self.model_type == 'stacked':
            base = [
                ('xgb', XGBClassifier(eval_metric='logloss', verbosity=0)),
                ('lgbm', LGBMClassifier()),
                ('rf', RandomForestClassifier())
            ]
            estimator = StackingClassifier(
                estimators=base,
                final_estimator=LogisticRegression(),
                cv=KFold(n_splits=5), n_jobs=-1
            )
        else:
            if self.mode == 'classification':
                opts = {
                    'xgboost': XGBClassifier(eval_metric='logloss', verbosity=0),
                    'lightgbm': LGBMClassifier(),
                    'random_forest': RandomForestClassifier(),
                    'mlp': MLPClassifier(max_iter=500)
                }
            else:
                opts = {
                    'xgboost': XGBRegressor(),
                    'lightgbm': LGBMRegressor(),
                    'random_forest': XGBRegressor()
                }
            estimator = opts.get(self.model_type, list(opts.values())[0])
        steps = [('imputer', SimpleImputer(strategy='mean'))]
        if self.model_type == 'mlp':
            steps.append(('scaler', StandardScaler()))
        steps.append(('model', estimator))
        self.pipeline = Pipeline(steps)
        if hyperparam_tuning:
            def objective(trial):
                raw = {}
                if self.model_type in ['xgboost', 'lightgbm', 'random_forest']:
                    raw['n_estimators'] = trial.suggest_int('n_estimators', 50, 500)
                    raw['max_depth']    = trial.suggest_int('max_depth', 3, 12)
                    if self.model_type in ['xgboost', 'lightgbm']:
                        raw['learning_rate'] = trial.suggest_loguniform('learning_rate', 1e-3, 1e-1)
                elif self.model_type == 'mlp':
                    raw['hidden_layer_sizes'] = trial.suggest_categorical('hidden_layer_sizes', [(50,), (100,), (50,50)])
                    raw['alpha'] = trial.suggest_loguniform('alpha', 1e-5, 1e-1)
                prefixed = {f"model__{k}": v for k, v in raw.items()}
                self.pipeline.set_params(**prefixed)
                tscv = TimeSeriesSplit(n_splits=5)
                scores = cross_val_score(self.pipeline, X_train_full, y_train_full,
                                         cv=tscv, scoring='accuracy')
                return scores.mean()
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=30)
            best = study.best_params
            best_prefixed = {f"model__{k}": v for k, v in best.items()}
            self.pipeline.set_params(**best_prefixed)
        self.pipeline.fit(X_train_full, y_train_full)
        self.X_train, self.X_test = X_train_full, X_test
        self.y_train, self.y_test = y_train_full, y_test

    def backtest(self):
        assert self.pipeline, "Train first"
        y_pred = self.pipeline.predict(self.X_test)
        print("Confusion matrix (all):")
        print(confusion_matrix(self.y_test, y_pred))
        print(classification_report(self.y_test, y_pred, zero_division=0))
        if self.model_type == 'stacked':
            stack = self.pipeline.named_steps['model']
            for name, est in stack.estimators:
                y_b = est.predict(self.X_test)
                print(f"Confusion matrix ({name}):")
                print(confusion_matrix(self.y_test, y_b))

    def save_model(self, filepath):
        joblib.dump({
            'pipeline': self.pipeline,
            'features': self.features,
            'mode': self.mode,
            'model_type': self.model_type,
            'horizon': self.horizon_minutes
        }, filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        data = joblib.load(filepath)
        self.pipeline = data['pipeline']
        self.features = data['features']
        self.mode = data['mode']
        self.model_type = data['model_type']
        self.horizon_minutes = data['horizon']
        print(f"Model loaded from {filepath}")
