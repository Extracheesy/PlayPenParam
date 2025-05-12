import os
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
import torch
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class XGBoostModel:
    def __init__(self):
        self.model = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            use_label_encoder=False,
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1
        )

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]

    def save(self, path):
        joblib.dump(self.model, path)

    def load(self, path):
        self.model = joblib.load(path)


class RandomForestModel:
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=6,
            random_state=42
        )

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]

    def save(self, path):
        joblib.dump(self.model, path)

    def load(self, path):
        self.model = joblib.load(path)


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        return self.sigmoid(self.fc(last_output))


class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=8, num_layers=2, dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                   dim_feedforward=dim_feedforward,
                                                   dropout=dropout,
                                                   batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        x = self.input_proj(x)  # (batch, seq_len, d_model)
        enc = self.transformer(x)  # (batch, seq_len, d_model)
        out = enc[:, -1, :]       # take last time-step
        return self.sigmoid(self.fc(out))


class LSTMWrapper:
    def __init__(self, lookback, input_dim, hidden_dim, num_layers, model_name, epochs=10):
        self.lookback = lookback
        self.model_name = model_name
        self.epochs = epochs
        self.model = LSTMModel(input_dim, hidden_dim, num_layers).to(DEVICE)
        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

    def create_sequences(self, df, target_col='result'):
        data = df.drop(columns=[target_col]).values
        labels = df[target_col].values
        X, y = [], []
        for i in range(len(df) - self.lookback):
            X.append(data[i:i+self.lookback])
            y.append(labels[i+self.lookback])
        return np.array(X), np.array(y)

    def fit(self, df):
        X, y = self.create_sequences(df)
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, shuffle=False)
        X_train = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
        y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(DEVICE)

        self.model.train()
        for epoch in range(self.epochs):
            self.optimizer.zero_grad()
            out = self.model(X_train)
            loss = self.criterion(out, y_train)
            loss.backward()
            self.optimizer.step()
            print(f"{self.model_name} Epoch {epoch+1}/{self.epochs} — Loss: {loss.item():.4f}")

    def predict_proba(self, df):
        self.model.eval()
        X, _ = self.create_sequences(df)
        X_t = torch.tensor(X, dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            probs = self.model(X_t).cpu().numpy().flatten()
        return probs

    def predict(self, df):
        probs = self.predict_proba(df)
        return (probs > 0.5).astype(int)

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path, map_location=DEVICE))
        self.model.eval()


class TransformerWrapper:
    def __init__(self, lookback, input_dim, d_model=64, nhead=8, num_layers=2,
                 dim_feedforward=256, dropout=0.1, model_name='transformer', epochs=10):
        self.lookback = lookback
        self.model_name = model_name
        self.epochs = epochs
        self.model = TransformerModel(input_dim, d_model, nhead, num_layers,
                                     dim_feedforward, dropout).to(DEVICE)
        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

    def create_sequences(self, df, target_col='result'):
        data = df.drop(columns=[target_col]).values
        labels = df[target_col].values
        X, y = [], []
        for i in range(len(df) - self.lookback):
            X.append(data[i:i+self.lookback])
            y.append(labels[i+self.lookback])
        return np.array(X), np.array(y)

    def fit(self, df):
        X, y = self.create_sequences(df)
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, shuffle=False)
        X_train = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
        y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(DEVICE)

        self.model.train()
        for epoch in range(self.epochs):
            self.optimizer.zero_grad()
            out = self.model(X_train)
            loss = self.criterion(out, y_train)
            loss.backward()
            self.optimizer.step()
            print(f"{self.model_name} Epoch {epoch+1}/{self.epochs} — Loss: {loss.item():.4f}")

    def predict_proba(self, df):
        self.model.eval()
        X, _ = self.create_sequences(df)
        X_t = torch.tensor(X, dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            probs = self.model(X_t).cpu().numpy().flatten()
        return probs

    def predict(self, df):
        probs = self.predict_proba(df)
        return (probs > 0.5).astype(int)

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path, map_location=DEVICE))
        self.model.eval()


class ModelManager:
    def __init__(self, df, models_path="./models/"):
        """
        df: pandas DataFrame with feature columns and a 'result' target column.
        models_path: directory to save/load model files.
        """
        self.data = df.copy()
        self.models_path = models_path
        os.makedirs(self.models_path, exist_ok=True)

        self.xgb   = XGBoostModel()
        self.rf    = RandomForestModel()
        self.lstm_a = LSTMWrapper(lookback=60, input_dim=df.shape[1]-1, hidden_dim=64, num_layers=1, model_name='lstm_a')
        self.lstm_b = LSTMWrapper(lookback=30, input_dim=df.shape[1]-1, hidden_dim=128, num_layers=2, model_name='lstm_b')
        self.transformer = TransformerWrapper(lookback=60, input_dim=df.shape[1]-1, d_model=64,
                                              nhead=8, num_layers=2, dim_feedforward=256,
                                              dropout=0.1, model_name='transformer', epochs=10)

    def train_and_save_all(self):
        df = self.data
        X = df.drop(columns=['result'])
        y = df['result']
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, shuffle=False)

        # XGBoost
        self.xgb.fit(X_train, y_train)
        self.xgb.save(os.path.join(self.models_path, "xgb.pkl"))

        # Random Forest
        self.rf.fit(X_train, y_train)
        self.rf.save(os.path.join(self.models_path, "rf.pkl"))

        # LSTMs
        self.lstm_a.fit(df)
        self.lstm_a.save(os.path.join(self.models_path, "lstm_a.pt"))
        self.lstm_b.fit(df)
        self.lstm_b.save(os.path.join(self.models_path, "lstm_b.pt"))

        # Transformer
        self.transformer.fit(df)
        self.transformer.save(os.path.join(self.models_path, "transformer.pt"))

    def load_all(self):
        self.xgb.load(os.path.join(self.models_path, "xgb.pkl"))
        self.rf.load(os.path.join(self.models_path, "rf.pkl"))
        self.lstm_a.load(os.path.join(self.models_path, "lstm_a.pt"))
        self.lstm_b.load(os.path.join(self.models_path, "lstm_b.pt"))
        self.transformer.load(os.path.join(self.models_path, "transformer.pt"))

    def predict(self):
        df = self.data
        X = df.drop(columns=['result'])

        # hard predictions
        px = self.xgb.predict(X)
        pr = self.rf.predict(X)
        pa = self.lstm_a.predict(df)
        pb = self.lstm_b.predict(df)
        pt = self.transformer.predict(df)

        # probabilities for tie-breaking
        qx = self.xgb.predict_proba(X)
        qr = self.rf.predict_proba(X)
        qa = self.lstm_a.predict_proba(df)
        qb = self.lstm_b.predict_proba(df)
        qt = self.transformer.predict_proba(df)

        # align lengths
        lengths = [len(px), len(pr), len(pa), len(pb), len(pt)]
        N = min(lengths)
        px, pr, pa, pb, pt = px[-N:], pr[-N:], pa[-N:], pb[-N:], pt[-N:]
        qx, qr, qa, qb, qt = qx[-N:], qr[-N:], qa[-N:], qb[-N:], qt[-N:]

        # majority vote with tie-breaker
        ensemble = []
        for i in range(N):
            votes = px[i] + pr[i] + pa[i] + pb[i] + pt[i]
            if votes > 2:
                ensemble.append(1)
            elif votes < 2:
                ensemble.append(0)
            else:
                avg_prob = (qx[i] + qr[i] + qa[i] + qb[i] + qt[i]) / 5
                ensemble.append(1 if avg_prob >= 0.5 else 0)

        return {
            "xgb":    px,
            "rf":     pr,
            "lstm_a": pa,
            "lstm_b": pb,
            "transformer": pt,
            "vote":   np.array(ensemble)
        }

# Usage example:
# df = pd.read_csv('your_data.csv', index_col=0, parse_dates=True)
# manager = ModelManager(df)
# manager.train_and_save_all()
# preds = manager.predict()
# print(preds)
