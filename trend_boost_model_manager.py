import os
import joblib
import pandas as pd
from typing import Optional, Dict, Any, List, Tuple

# Boosters
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

# Random Forest & Stacking
from sklearn.ensemble import RandomForestClassifier, StackingClassifier

# Model selection & tuning
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.feature_selection import SelectFromModel

# Imbalance handling
from imblearn.over_sampling import SMOTE

# Metrics
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

class BoostModelTrainer:
    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: Optional[List[str]] = None,
        target_col: str = "result",
        save_dir: str = "saved_models",
        apply_smote: bool = False,
        feature_selection: bool = False,
        fs_threshold: float = 0.01
    ):
        """
        df: DataFrame with data
        feature_cols: list of features; default = all except target_col
        target_col: name of label column
        save_dir: where to persist models
        apply_smote: whether to apply SMOTE on training data
        feature_selection: whether to perform automated feature selection
        fs_threshold: threshold for SelectFromModel importance
        """
        self.df = df.copy()
        self.target_col = target_col
        self.orig_feature_cols = feature_cols or [c for c in df.columns if c != target_col]
        # current feature set (may be reduced if FS applied)
        self.selected_features = list(self.orig_feature_cols)

        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        # Input X and y (original)
        self.X_orig = self.df[self.orig_feature_cols]
        self.y = self.df[target_col]

        # Flags
        self.apply_smote = apply_smote
        self.feature_selection = feature_selection
        self.fs_threshold = fs_threshold

        # Feature importances log
        self.feature_importances_: Dict[str, pd.Series] = {}

        # Base learners
        self.models: Dict[str, Any] = {
            "catboost": CatBoostClassifier(verbose=False, early_stopping_rounds=50),
            "lightgbm": LGBMClassifier(n_estimators=1000),
            "xgboost": XGBClassifier(eval_metric='mlogloss', n_estimators=1000),
            "random_forest": RandomForestClassifier(n_estimators=200)
        }

    def _align_features(self, X: pd.DataFrame) -> pd.DataFrame:
        # use selected_features for consistency
        return X.reindex(columns=self.selected_features, fill_value=0)

    def _prepare_training_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        # start with original X
        X_train = self.X_orig.copy()
        y_train = self.y.copy()
        if self.apply_smote:
            sm = SMOTE(random_state=42)
            X_train, y_train = sm.fit_resample(X_train, y_train)
        if self.feature_selection:
            # select via random forest
            selector = SelectFromModel(
                RandomForestClassifier(n_estimators=100, random_state=42),
                threshold=self.fs_threshold
            )
            selector.fit(X_train, y_train)
            mask = selector.get_support()
            self.selected_features = [f for f, m in zip(self.orig_feature_cols, mask) if m]
            X_train = pd.DataFrame(selector.transform(X_train), columns=self.selected_features)
        else:
            # no selection, keep all
            self.selected_features = list(self.orig_feature_cols)
            X_train = X_train[self.selected_features]
        return X_train, y_train

    def train_all(
        self,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
    ):
        """Train each model, with optional validation for early stopping."""
        X_train, y_train = self._prepare_training_data()
        X_val_aligned, y_val_aligned = None, None
        if X_val is not None and y_val is not None:
            X_val_aligned = self._align_features(X_val)
            y_val_aligned = y_val

        for name, model in self.models.items():
            print(f"Training {name} on {len(self.selected_features)} features...")
            if hasattr(model, 'early_stopping_rounds') and X_val_aligned is not None:
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val_aligned, y_val_aligned)],
                    verbose=False
                )
            else:
                model.fit(X_train, y_train)
            # log importances
            if hasattr(model, 'feature_importances_'):
                imp = pd.Series(model.feature_importances_, index=self.selected_features)
                self.feature_importances_[name] = imp.sort_values(ascending=False)
            print(f"  -> done.")

    def save_all(self):
        """Save models and feature importances."""
        for name, model in self.models.items():
            joblib.dump(model, os.path.join(self.save_dir, f"{name}.joblib"))
        # save importances
        fi_df = pd.DataFrame(self.feature_importances_)
        fi_df.to_csv(os.path.join(self.save_dir, 'feature_importances.csv'))
        print(f"Saved models & importances to {self.save_dir}")

    def load_all(self):
        """Load models and importances."""
        for name in list(self.models.keys()):
            path = os.path.join(self.save_dir, f"{name}.joblib")
            if os.path.exists(path):
                self.models[name] = joblib.load(path)
        fi_path = os.path.join(self.save_dir, 'feature_importances.csv')
        if os.path.exists(fi_path):
            df = pd.read_csv(fi_path, index_col=0)
            self.feature_importances_ = {col: df[col].dropna() for col in df.columns}
        print(f"Loaded models & importances from {self.save_dir}")

    def predict_ensemble(self, X_new: pd.DataFrame) -> pd.Series:
        Xn = self._align_features(X_new)
        probas = [m.predict_proba(Xn) for m in self.models.values()]
        avg = sum(probas) / len(probas)
        classes = next(iter(self.models.values())).classes_
        idx = avg.argmax(axis=1)
        return pd.Series(classes[idx], index=X_new.index)

    def evaluate(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        do_print: bool = True
    ) -> pd.DataFrame:
        Xt = self._align_features(X_test)
        results = []
        for name, model in self.models.items():
            y_pred = model.predict(Xt)
            results.append({
                'model': name,
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='macro', zero_division=0),
                'recall': recall_score(y_test, y_pred, average='macro', zero_division=0),
                'f1': f1_score(y_test, y_pred, average='macro', zero_division=0)
            })
        # ensemble
        ens = self.predict_ensemble(X_test)
        results.append({
            'model': 'ensemble',
            'accuracy': accuracy_score(y_test, ens),
            'precision': precision_score(y_test, ens, average='macro', zero_division=0),
            'recall': recall_score(y_test, ens, average='macro', zero_division=0),
            'f1': f1_score(y_test, ens, average='macro', zero_division=0)
        })
        df_res = pd.DataFrame(results).set_index('model')
        if do_print:
            print(df_res)
        return df_res