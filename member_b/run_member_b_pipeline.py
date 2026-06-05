from __future__ import annotations

import json
import math
import random
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


SEED = 42
INPUT_WINDOW = 30
HORIZON = 7
AE_WINDOW = 14
CATEGORIES = ["飲食", "交通", "購物", "娛樂", "教育", "醫療健康"]
CATEGORY_EN = {
    "飲食": "Food",
    "交通": "Transport",
    "購物": "Shopping",
    "娛樂": "Entertainment",
    "教育": "Education",
    "醫療健康": "Medical",
}

ROOT = Path(__file__).resolve().parents[2]
A_DIR = ROOT / "文件" / "成員 A：資料工程與 Task 1 分類模型結果"
OUT_DIR = ROOT / "文件" / "成員 B：時序預測與異常偵測結果"
FIG_DIR = OUT_DIR / "figures"

INPUT_CSV = A_DIR / "all_user_6_label.csv"
EMBEDDING_NPY = A_DIR / "receipt_embeddings.npy"


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)


def ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.abs(y_true) + np.abs(y_pred)
    mask = denom > 1e-8
    if not mask.any():
        return 0.0
    return float(np.mean(2.0 * np.abs(y_pred[mask] - y_true[mask]) / denom[mask]))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(math.sqrt(mean_squared_error(y_true, y_pred)))


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    dt = out["date"]
    out["day_of_week"] = dt.dt.dayofweek
    out["is_weekend"] = out["day_of_week"].isin([5, 6]).astype(int)
    out["day_of_month"] = dt.dt.day
    out["week_of_month"] = ((dt.dt.day - 1) // 7 + 1).astype(int)
    out["days_to_month_end"] = (dt.dt.days_in_month - dt.dt.day).astype(int)
    out["month"] = dt.dt.month
    out["dow_sin"] = np.sin(2 * np.pi * out["day_of_week"] / 7.0)
    out["dow_cos"] = np.cos(2 * np.pi * out["day_of_week"] / 7.0)
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12.0)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12.0)
    return out


def load_and_build_daily() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    df = pd.read_csv(INPUT_CSV)
    df["date"] = pd.to_datetime(df["發票日期"])
    df["amount"] = pd.to_numeric(df["消費明細_金額"], errors="coerce").fillna(0.0)
    df["quantity"] = pd.to_numeric(df["消費明細_數量"], errors="coerce")
    df["unit_price"] = pd.to_numeric(df["消費明細_單價"], errors="coerce").fillna(0.0)

    raw = df[
        [
            "user_id",
            "date",
            "label",
            "amount",
            "quantity",
            "unit_price",
            "store_clean",
            "item_clean",
            "price_bucket",
            "label_id",
        ]
    ].copy()

    rows = []
    for uid, g in raw.groupby("user_id"):
        start, end = g["date"].min(), g["date"].max()
        calendar = pd.DataFrame({"date": pd.date_range(start, end, freq="D")})
        pivot = (
            g.pivot_table(
                index="date",
                columns="label",
                values="amount",
                aggfunc="sum",
                fill_value=0.0,
            )
            .reset_index()
            .rename_axis(None, axis=1)
        )
        merged = calendar.merge(pivot, on="date", how="left")
        for cat in CATEGORIES:
            if cat not in merged.columns:
                merged[cat] = 0.0
        merged[CATEGORIES] = merged[CATEGORIES].fillna(0.0)
        merged["total"] = merged[CATEGORIES].sum(axis=1)
        merged["has_transaction"] = (merged["total"] > 0).astype(int)
        merged.insert(0, "user_id", uid)
        rows.append(merged[["user_id", "date", *CATEGORIES, "total", "has_transaction"]])

    daily = pd.concat(rows, ignore_index=True)
    daily_feat = add_time_features(daily)

    meta = {
        "source_csv": str(INPUT_CSV),
        "rows_item_level": int(len(raw)),
        "users": sorted([int(x) for x in raw["user_id"].unique()]),
        "categories": CATEGORIES,
        "input_window_days": INPUT_WINDOW,
        "forecast_horizon_days": HORIZON,
        "ae_window_days": AE_WINDOW,
    }
    return raw, daily_feat, meta


def build_supervised(
    user_daily: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "total",
    input_window: int = INPUT_WINDOW,
    horizon: int = HORIZON,
) -> tuple[np.ndarray, np.ndarray, list[pd.Timestamp]]:
    x = user_daily[feature_cols].to_numpy(dtype=np.float32)
    y = user_daily[target_col].to_numpy(dtype=np.float32)
    xs, ys, dates = [], [], []
    for start in range(0, len(user_daily) - input_window - horizon + 1):
        end = start + input_window
        target_end = end + horizon
        xs.append(x[start:end])
        ys.append(y[end:target_end])
        dates.append(user_daily["date"].iloc[end])
    return np.asarray(xs), np.asarray(ys), dates


def split_indices(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_end = max(1, int(n * 0.70))
    val_end = max(train_end + 1, int(n * 0.85))
    val_end = min(val_end, n - 1)
    idx = np.arange(n)
    return idx[:train_end], idx[train_end:val_end], idx[val_end:]


def scale_windows(
    x: np.ndarray, train_idx: np.ndarray
) -> tuple[np.ndarray, StandardScaler]:
    scaler = StandardScaler()
    n, t, f = x.shape
    scaler.fit(x[train_idx].reshape(-1, f))
    scaled = scaler.transform(x.reshape(-1, f)).reshape(n, t, f).astype(np.float32)
    return scaled, scaler


def baseline_moving_average(user_daily: pd.DataFrame, dates: list[pd.Timestamp]) -> np.ndarray:
    totals = user_daily.set_index("date")["total"]
    preds = []
    for d in dates:
        hist = totals.loc[: d - pd.Timedelta(days=1)].tail(7)
        value = float(hist.mean()) if len(hist) else 0.0
        preds.append(np.full(HORIZON, value, dtype=np.float32))
    return np.asarray(preds)


def baseline_weekday_average(user_daily: pd.DataFrame, dates: list[pd.Timestamp]) -> np.ndarray:
    df = user_daily.set_index("date")
    preds = []
    for d in dates:
        horizon_preds = []
        hist = df.loc[: d - pd.Timedelta(days=1)].copy()
        fallback = float(hist["total"].tail(14).mean()) if len(hist) else 0.0
        for h in range(HORIZON):
            target_date = d + pd.Timedelta(days=h)
            same_weekday = hist[hist["day_of_week"] == target_date.dayofweek]["total"].tail(6)
            horizon_preds.append(float(same_weekday.mean()) if len(same_weekday) else fallback)
        preds.append(horizon_preds)
    return np.asarray(preds, dtype=np.float32)


def baseline_arima(user_daily: pd.DataFrame, dates: list[pd.Timestamp]) -> np.ndarray:
    totals = user_daily.set_index("date")["total"]
    preds = []
    for d in dates:
        hist = totals.loc[: d - pd.Timedelta(days=1)].astype(float)
        if len(hist) < 45 or hist.std() < 1e-8:
            value = float(hist.tail(7).mean()) if len(hist) else 0.0
            preds.append(np.full(HORIZON, value, dtype=np.float32))
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = ARIMA(hist, order=(1, 0, 1))
                fitted = model.fit()
                pred = fitted.forecast(steps=HORIZON).to_numpy(dtype=np.float32)
        except Exception:
            pred = np.full(HORIZON, float(hist.tail(7).mean()), dtype=np.float32)
        preds.append(np.maximum(pred, 0.0))
    return np.asarray(preds, dtype=np.float32)


def baseline_mlp(x_scaled: np.ndarray, y: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        model = MLPRegressor(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            alpha=1e-3,
            learning_rate_init=1e-3,
            max_iter=800,
            random_state=SEED,
            early_stopping=True,
            validation_fraction=0.2,
        )
        model.fit(x_scaled[train_idx].reshape(len(train_idx), -1), y[train_idx])
    return np.maximum(model.predict(x_scaled[test_idx].reshape(len(test_idx), -1)), 0.0)


class LSTMForecaster(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 1, horizon: int = HORIZON):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=0.0)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(64, horizon),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


def train_lstm_forecaster(
    x_scaled: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    max_epochs: int = 220,
) -> tuple[np.ndarray, dict]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMForecaster(input_size=x_scaled.shape[-1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.SmoothL1Loss()

    y_scaler = StandardScaler()
    y_train = y_scaler.fit_transform(y[train_idx])
    y_all = y_scaler.transform(y)

    train_ds = TensorDataset(
        torch.tensor(x_scaled[train_idx], dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

    x_val = torch.tensor(x_scaled[val_idx], dtype=torch.float32).to(device)
    y_val = torch.tensor(y_all[val_idx], dtype=torch.float32).to(device)

    best_state = None
    best_val = float("inf")
    patience, stale = 25, 0
    history = []
    for epoch in range(max_epochs):
        model.train()
        losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.item()))

        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(x_val), y_val).item()) if len(val_idx) else float(np.mean(losses))
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)), "val_loss": val_loss})
        if val_loss < best_val - 1e-5:
            best_val = val_loss
            stale = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if stale >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred_scaled = model(torch.tensor(x_scaled[test_idx], dtype=torch.float32).to(device)).cpu().numpy()
    pred = y_scaler.inverse_transform(pred_scaled)
    info = {
        "device": str(device),
        "epochs": len(history),
        "best_val_loss_scaled": float(best_val),
    }
    return np.maximum(pred, 0.0), info


class LSTMAutoencoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 48, latent_size: int = 24):
        super().__init__()
        self.encoder = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.to_latent = nn.Linear(hidden_size, latent_size)
        self.from_latent = nn.Linear(latent_size, hidden_size)
        self.decoder = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.output = nn.Linear(hidden_size, input_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h, _) = self.encoder(x)
        z = torch.relu(self.to_latent(h[-1]))
        hidden = torch.relu(self.from_latent(z)).unsqueeze(0)
        cell = torch.zeros_like(hidden)
        decoder_input = torch.zeros_like(x)
        out, _ = self.decoder(decoder_input, (hidden, cell))
        return self.output(out)


def train_autoencoder_anomaly(user_daily: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    values = user_daily[feature_cols].to_numpy(dtype=np.float32)
    n = len(values)
    train_end = int(n * 0.70)
    scaler = StandardScaler()
    scaler.fit(values[:train_end])
    scaled = scaler.transform(values).astype(np.float32)

    seqs, seq_starts = [], []
    for start in range(0, n - AE_WINDOW + 1):
        seqs.append(scaled[start : start + AE_WINDOW])
        seq_starts.append(start)
    seqs = np.asarray(seqs, dtype=np.float32)
    train_seq_idx = [i for i, s in enumerate(seq_starts) if s + AE_WINDOW <= train_end]
    if len(train_seq_idx) < 10:
        raise ValueError("Not enough sequences for autoencoder")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMAutoencoder(input_size=seqs.shape[-1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    loader = DataLoader(
        TensorDataset(torch.tensor(seqs[train_seq_idx], dtype=torch.float32)),
        batch_size=32,
        shuffle=True,
    )

    best_state = None
    best_loss = float("inf")
    stale = 0
    for _epoch in range(180):
        model.train()
        losses = []
        for (xb,) in loader:
            xb = xb.to(device)
            opt.zero_grad()
            recon = model(xb)
            loss = loss_fn(recon, xb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.item()))
        epoch_loss = float(np.mean(losses))
        if epoch_loss < best_loss - 1e-5:
            best_loss = epoch_loss
            stale = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if stale >= 25:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        recon = model(torch.tensor(seqs, dtype=torch.float32).to(device)).cpu().numpy()
    seq_errors = np.mean((recon - seqs) ** 2, axis=2)

    day_scores = [[] for _ in range(n)]
    for seq_i, start in enumerate(seq_starts):
        for offset in range(AE_WINDOW):
            day_scores[start + offset].append(float(seq_errors[seq_i, offset]))
    score = np.asarray([np.mean(s) if s else 0.0 for s in day_scores])
    train_scores = score[:train_end]
    threshold = float(np.quantile(train_scores, 0.95))

    extra_cols = [
        c
        for c in [
            "is_normal_day",
            "is_high_spend_day",
            "normal_day_threshold_p75",
            "high_spend_threshold_p90",
        ]
        if c in user_daily.columns
    ]
    out = user_daily[["user_id", "date", "total", "has_transaction", *CATEGORIES, *extra_cols]].copy()
    out["ae_reconstruction_error"] = score
    out["ae_threshold_p95_train"] = threshold
    out["is_anomaly"] = (out["ae_reconstruction_error"] > threshold).astype(int)
    active_total = out.loc[out["has_transaction"] == 1, "total"]
    active_mean = float(active_total.mean()) if len(active_total) else 0.0
    active_std = float(active_total.std(ddof=0)) if len(active_total) else 0.0
    out["active_day_avg"] = active_mean
    out["amount_zscore_active"] = (
        (out["total"] - active_mean) / active_std if active_std > 1e-8 else 0.0
    )
    out["is_high_spend_proxy"] = ((out["total"] >= active_mean * 1.5) & (out["has_transaction"] == 1)).astype(int)
    out["dominant_category"] = out[CATEGORIES].idxmax(axis=1)
    out.loc[out[CATEGORIES].sum(axis=1) <= 0, "dominant_category"] = "無消費紀錄"
    return out


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    actual_flat = y_true.reshape(-1)
    pred_flat = y_pred.reshape(-1)
    actual_std = float(np.std(actual_flat))
    pred_std = float(np.std(pred_flat))
    std_ratio = float(pred_std / actual_std) if actual_std > 1e-8 else 0.0
    return {
        "MAE": float(mean_absolute_error(actual_flat, pred_flat)),
        "RMSE": rmse(actual_flat, pred_flat),
        "sMAPE": smape(actual_flat, pred_flat),
        "MAE_day1": float(mean_absolute_error(y_true[:, 0], y_pred[:, 0])),
        "MAE_day7": float(mean_absolute_error(y_true[:, -1], y_pred[:, -1])),
        "actual_std": actual_std,
        "prediction_std": pred_std,
        "std_ratio": std_ratio,
        "collapsed_prediction": int(std_ratio < 0.10),
    }


def add_routine_event_flags(daily: pd.DataFrame) -> pd.DataFrame:
    out = daily.copy()
    out["normal_day_threshold_p75"] = 0.0
    out["high_spend_threshold_p90"] = 0.0
    out["is_normal_day"] = 0
    out["is_high_spend_day"] = 0
    for uid, idx in out.groupby("user_id").groups.items():
        g = out.loc[idx]
        active = g.loc[g["has_transaction"] == 1, "total"]
        if len(active) == 0:
            continue
        p75 = float(active.quantile(0.75))
        p90 = float(active.quantile(0.90))
        out.loc[idx, "normal_day_threshold_p75"] = p75
        out.loc[idx, "high_spend_threshold_p90"] = p90
        out.loc[idx, "is_normal_day"] = ((g["has_transaction"] == 1) & (g["total"] <= p75)).astype(int)
        out.loc[idx, "is_high_spend_day"] = ((g["has_transaction"] == 1) & (g["total"] >= p90)).astype(int)
    return out


def evaluate_routine_event_forecasts(forecasts: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    flags = daily[
        [
            "user_id",
            "date",
            "is_normal_day",
            "is_high_spend_day",
            "normal_day_threshold_p75",
            "high_spend_threshold_p90",
        ]
    ].copy()
    flags["target_date"] = flags["date"].dt.date.astype(str)
    merged = forecasts.merge(
        flags.drop(columns=["date"]), on=["user_id", "target_date"], how="left"
    )
    methods = [
        c
        for c in merged.columns
        if c
        in ["MovingAverage7", "WeekdayAverage", "ARIMA_101", "MLP", "LSTM_no_time", "LSTM_time"]
    ]
    rows = []
    for uid in sorted(merged["user_id"].unique()):
        g = merged[merged["user_id"] == uid]
        for method in methods:
            pred = g[method].to_numpy(dtype=float)
            actual = g["actual"].to_numpy(dtype=float)
            normal_mask = g["is_normal_day"].fillna(0).to_numpy(dtype=int) == 1
            high_mask = g["is_high_spend_day"].fillna(0).to_numpy(dtype=int) == 1
            rows.append(
                {
                    "user_id": int(uid),
                    "method": method,
                    "overall_MAE": float(mean_absolute_error(actual, pred)),
                    "normal_day_MAE": float(mean_absolute_error(actual[normal_mask], pred[normal_mask]))
                    if normal_mask.any()
                    else np.nan,
                    "high_spend_day_MAE": float(mean_absolute_error(actual[high_mask], pred[high_mask]))
                    if high_mask.any()
                    else np.nan,
                    "n_normal_targets": int(normal_mask.sum()),
                    "n_high_spend_targets": int(high_mask.sum()),
                    "normal_day_threshold_p75": float(g["normal_day_threshold_p75"].dropna().iloc[0])
                    if g["normal_day_threshold_p75"].notna().any()
                    else np.nan,
                    "high_spend_threshold_p90": float(g["high_spend_threshold_p90"].dropna().iloc[0])
                    if g["high_spend_threshold_p90"].notna().any()
                    else np.nan,
                }
            )
    return pd.DataFrame(rows)


def evaluate_spike_detection(anomaly: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for uid, g in anomaly.groupby("user_id"):
        spike = g["is_high_spend_day"].astype(int) == 1
        detected = g["is_anomaly"].astype(int) == 1
        tp = int((spike & detected).sum())
        fp = int((~spike & detected).sum())
        fn = int((spike & ~detected).sum())
        rows.append(
            {
                "user_id": int(uid),
                "high_spend_threshold_p90": float(g["high_spend_threshold_p90"].iloc[0]),
                "n_high_spend_days": int(spike.sum()),
                "n_detected_anomaly_days": int(detected.sum()),
                "true_positive_spikes": tp,
                "missed_spikes": fn,
                "non_spike_anomalies": fp,
                "spike_recall": float(tp / (tp + fn)) if (tp + fn) else np.nan,
                "spike_precision": float(tp / (tp + fp)) if (tp + fp) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def evaluate_threshold_sensitivity(anomaly: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for uid, g in anomaly.groupby("user_id"):
        train_end = int(len(g) * 0.70)
        train_scores = g["ae_reconstruction_error"].iloc[:train_end].to_numpy(dtype=float)
        spike = g["is_high_spend_day"].astype(int) == 1
        for q in [0.90, 0.95, 0.97]:
            threshold = float(np.quantile(train_scores, q))
            detected = g["ae_reconstruction_error"].to_numpy(dtype=float) > threshold
            tp = int((spike & detected).sum())
            fp = int((~spike & detected).sum())
            fn = int((spike & ~detected).sum())
            rows.append(
                {
                    "user_id": int(uid),
                    "threshold_quantile": q,
                    "ae_threshold": threshold,
                    "n_high_spend_days": int(spike.sum()),
                    "n_detected_anomaly_days": int(detected.sum()),
                    "true_positive_spikes": tp,
                    "missed_spikes": fn,
                    "non_spike_anomalies": fp,
                    "spike_recall": float(tp / (tp + fn)) if (tp + fn) else np.nan,
                    "spike_precision": float(tp / (tp + fp)) if (tp + fp) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def build_model_behavior_diagnostics(metrics: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "user_id",
        "method",
        "MAE",
        "MAE_day1",
        "MAE_day7",
        "actual_std",
        "prediction_std",
        "std_ratio",
        "collapsed_prediction",
    ]
    return metrics[cols].copy().sort_values(["user_id", "MAE"])


def build_model_suitability(metrics: pd.DataFrame, routine_event_metrics: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for uid in sorted(metrics["user_id"].unique()):
        user_metrics = metrics[metrics["user_id"] == uid].copy()
        usable = user_metrics[user_metrics["collapsed_prediction"] == 0]
        if usable.empty:
            usable = user_metrics
        best = usable.sort_values("MAE").iloc[0]
        raw_best = user_metrics.sort_values("MAE").iloc[0]
        re_best = routine_event_metrics[
            (routine_event_metrics["user_id"] == uid) & (routine_event_metrics["method"] == best["method"])
        ]
        s = summary[summary["user_id"] == uid].iloc[0]
        if best["collapsed_prediction"] == 1:
            reason = "所有候選模型皆有低變異風險，只能保守解讀"
        elif best["method"].startswith("LSTM"):
            reason = "序列模型未退化且整體 MAE 最低，代表此 user 有可學習的短期時序規律"
        elif best["method"] == "MLP":
            reason = "非序列深度模型較穩，代表短期 window 的整體形狀比逐日遞迴關係更有用"
        elif best["method"] in ["MovingAverage7", "WeekdayAverage"]:
            reason = "簡單統計 baseline 已足夠，深度模型未提供穩定改善"
        else:
            reason = "傳統時序模型在此 user 的整體誤差較低"
        rows.append(
            {
                "user_id": int(uid),
                "calendar_days": int(s["calendar_days"]),
                "active_day_rate": float(s["active_day_rate"]),
                "dominant_category": s["dominant_category"],
                "reporting_method": best["method"],
                "raw_lowest_MAE_method": raw_best["method"],
                "MAE": float(best["MAE"]),
                "normal_day_MAE": float(re_best["normal_day_MAE"].iloc[0]) if not re_best.empty else np.nan,
                "high_spend_day_MAE": float(re_best["high_spend_day_MAE"].iloc[0]) if not re_best.empty else np.nan,
                "std_ratio": float(best["std_ratio"]),
                "collapsed_prediction": int(best["collapsed_prediction"]),
                "selection_reason": reason,
            }
        )
    return pd.DataFrame(rows)


def plot_user_series(daily: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid")
    for uid, g in daily.groupby("user_id"):
        plt.figure(figsize=(12, 4))
        plt.plot(g["date"], g["total"], linewidth=1.4)
        plt.title(f"User {uid} daily total spending")
        plt.xlabel("Date")
        plt.ylabel("Amount")
        plt.tight_layout()
        plt.savefig(FIG_DIR / f"user_{uid}_daily_total.png", dpi=160)
        plt.close()

        cat_month = g.set_index("date")[CATEGORIES].resample("MS").sum().rename(columns=CATEGORY_EN)
        plt.figure(figsize=(12, 5))
        cat_month.plot(kind="bar", stacked=True, ax=plt.gca())
        plt.title(f"User {uid} monthly spending by category")
        plt.xlabel("Month")
        plt.ylabel("Amount")
        plt.legend(loc="upper right", fontsize=8)
        plt.tight_layout()
        plt.savefig(FIG_DIR / f"user_{uid}_monthly_category.png", dpi=160)
        plt.close()


def plot_forecast_examples(forecasts: pd.DataFrame) -> None:
    for uid, g in forecasts.groupby("user_id"):
        day1 = g[g["horizon_day"] == 1].copy()
        if day1.empty:
            continue
        day1["target_date"] = pd.to_datetime(day1["target_date"])
        plt.figure(figsize=(12, 4))
        plt.plot(day1["target_date"], day1["actual"], label="Actual day+1", linewidth=1.5)
        for method in ["MovingAverage7", "WeekdayAverage", "MLP", "LSTM_time"]:
            if method in day1.columns:
                plt.plot(day1["target_date"], day1[method], label=method, alpha=0.8)
        plt.title(f"User {uid} one-day-ahead forecast comparison")
        plt.xlabel("Target date")
        plt.ylabel("Amount")
        ax = plt.gca()
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=7))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
        plt.xticks(rotation=35, ha="right")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(FIG_DIR / f"user_{uid}_forecast_day1.png", dpi=160)
        plt.close()


def plot_anomalies(anomaly: pd.DataFrame) -> None:
    for uid, g in anomaly.groupby("user_id"):
        plt.figure(figsize=(12, 4))
        plt.plot(g["date"], g["total"], label="Daily total", linewidth=1.4)
        marked = g[g["is_anomaly"] == 1]
        plt.scatter(marked["date"], marked["total"], color="red", label="AE anomaly", s=28, zorder=3)
        plt.title(f"User {uid} anomaly detection")
        plt.xlabel("Date")
        plt.ylabel("Amount")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(FIG_DIR / f"user_{uid}_anomaly.png", dpi=160)
        plt.close()


def run_forecasting(daily: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    base_feature_cols = [*CATEGORIES, "total", "has_transaction"]
    time_feature_cols = [
        *base_feature_cols,
        "is_weekend",
        "day_of_month",
        "week_of_month",
        "days_to_month_end",
        "dow_sin",
        "dow_cos",
        "month_sin",
        "month_cos",
    ]

    metric_rows = []
    forecast_rows = []
    training_info = []

    for uid, g in daily.groupby("user_id"):
        g = g.sort_values("date").reset_index(drop=True)
        x_base, y, start_dates = build_supervised(g, base_feature_cols)
        x_time, _, _ = build_supervised(g, time_feature_cols)
        if len(y) < 30:
            continue
        train_idx, val_idx, test_idx = split_indices(len(y))
        test_dates = [start_dates[i] for i in test_idx]
        target_dates = [[d + pd.Timedelta(days=h) for h in range(HORIZON)] for d in test_dates]

        x_base_scaled, _ = scale_windows(x_base, train_idx)
        x_time_scaled, _ = scale_windows(x_time, train_idx)

        method_preds = {
            "MovingAverage7": baseline_moving_average(g, test_dates),
            "WeekdayAverage": baseline_weekday_average(g, test_dates),
            "ARIMA_101": baseline_arima(g, test_dates),
            "MLP": baseline_mlp(x_time_scaled, y, train_idx, test_idx),
        }
        lstm_base_pred, info_base = train_lstm_forecaster(x_base_scaled, y, train_idx, val_idx, test_idx)
        lstm_time_pred, info_time = train_lstm_forecaster(x_time_scaled, y, train_idx, val_idx, test_idx)
        method_preds["LSTM_no_time"] = lstm_base_pred
        method_preds["LSTM_time"] = lstm_time_pred

        training_info.append({"user_id": int(uid), "model": "LSTM_no_time", **info_base})
        training_info.append({"user_id": int(uid), "model": "LSTM_time", **info_time})

        y_test = y[test_idx]
        for method, pred in method_preds.items():
            row = {
                "user_id": int(uid),
                "method": method,
                "n_train_windows": int(len(train_idx)),
                "n_val_windows": int(len(val_idx)),
                "n_test_windows": int(len(test_idx)),
                **evaluate_predictions(y_test, pred),
            }
            metric_rows.append(row)

        for sample_i, window_start_date in enumerate(test_dates):
            for h in range(HORIZON):
                record = {
                    "user_id": int(uid),
                    "forecast_origin": window_start_date.date().isoformat(),
                    "horizon_day": h + 1,
                    "target_date": target_dates[sample_i][h].date().isoformat(),
                    "actual": float(y_test[sample_i, h]),
                }
                for method, pred in method_preds.items():
                    record[method] = float(pred[sample_i, h])
                forecast_rows.append(record)

    return pd.DataFrame(metric_rows), pd.DataFrame(forecast_rows), training_info


def run_anomaly_detection(daily: pd.DataFrame) -> pd.DataFrame:
    ae_feature_cols = [*CATEGORIES, "total", "has_transaction", "is_weekend", "dow_sin", "dow_cos"]
    rows = []
    for uid, g in daily.groupby("user_id"):
        g = g.sort_values("date").reset_index(drop=True)
        rows.append(train_autoencoder_anomaly(g, ae_feature_cols))
    return pd.concat(rows, ignore_index=True)


def summarize_anomaly_reasons(anomaly: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for uid, g in anomaly.groupby("user_id"):
        active_avg = g.loc[g["has_transaction"] == 1, "total"].mean()
        high_pattern = g[(g["is_anomaly"] == 1) & (g["is_high_spend_proxy"] == 1)].copy()
        high_pattern = high_pattern.sort_values(
            ["amount_zscore_active", "ae_reconstruction_error"], ascending=False
        )
        remaining = g.drop(index=high_pattern.index).sort_values("ae_reconstruction_error", ascending=False)
        top = pd.concat([high_pattern, remaining], axis=0).head(10)
        for _, r in top.iterrows():
            cat_amounts = r[CATEGORIES].astype(float)
            dominant = str(cat_amounts.idxmax()) if cat_amounts.max() > 0 else "無消費紀錄"
            rows.append(
                {
                    "user_id": int(uid),
                    "date": r["date"].date().isoformat(),
                    "total": float(r["total"]),
                    "active_day_avg": float(active_avg),
                    "ratio_to_active_avg": float(r["total"] / active_avg) if active_avg and active_avg > 0 else np.nan,
                    "amount_zscore_active": float(r["amount_zscore_active"]),
                    "is_ae_anomaly": int(r["is_anomaly"]),
                    "is_high_spend_proxy": int(r["is_high_spend_proxy"]),
                    "ae_reconstruction_error": float(r["ae_reconstruction_error"]),
                    "dominant_category": dominant,
                    "dominant_category_amount": float(cat_amounts.max()),
                    "interpretation": make_anomaly_interpretation(r, active_avg, dominant),
                }
            )
    return pd.DataFrame(rows)


def make_anomaly_interpretation(row: pd.Series, active_avg: float, dominant: str) -> str:
    total = float(row["total"])
    if total <= 0:
        return "no recorded spending; check missing-receipt pattern"
    ratio = total / active_avg if active_avg and active_avg > 0 else 0.0
    if ratio >= 2.0:
        return f"high spending day: {ratio:.1f}x active-day average; dominant category={dominant}"
    if dominant != "no_spending":
        return f"unusual sequence pattern; dominant category={dominant}"
    return "unusual sequence pattern; review transaction details"

def make_summary_tables(
    raw: pd.DataFrame,
    daily: pd.DataFrame,
    metrics: pd.DataFrame,
    anomaly_top: pd.DataFrame,
    routine_event_metrics: pd.DataFrame | None = None,
    spike_metrics: pd.DataFrame | None = None,
) -> dict:
    user_summary = []
    for uid, g in daily.groupby("user_id"):
        raw_g = raw[raw["user_id"] == uid]
        user_summary.append(
            {
                "user_id": int(uid),
                "item_rows": int(len(raw_g)),
                "date_start": g["date"].min().date().isoformat(),
                "date_end": g["date"].max().date().isoformat(),
                "calendar_days": int(len(g)),
                "active_days": int(g["has_transaction"].sum()),
                "zero_days": int((g["has_transaction"] == 0).sum()),
                "active_day_rate": float(g["has_transaction"].mean()),
                "total_spending": float(g["total"].sum()),
                "avg_active_day_spending": float(g.loc[g["has_transaction"] == 1, "total"].mean()),
                "max_daily_spending": float(g["total"].max()),
                "dominant_category": raw_g.groupby("label")["amount"].sum().sort_values(ascending=False).index[0],
            }
        )
    user_summary_df = pd.DataFrame(user_summary)
    user_summary_df.to_csv(OUT_DIR / "user_data_summary.csv", index=False, encoding="utf-8-sig")

    best = (
        metrics.sort_values(["user_id", "MAE"])
        .groupby("user_id")
        .head(1)
        .rename(columns={"method": "best_method"})
    )
    best.to_csv(OUT_DIR / "best_forecast_method_by_user.csv", index=False, encoding="utf-8-sig")

    return {
        "user_summary": user_summary_df,
        "best_methods": best,
        "metric_table": metrics,
        "anomaly_top": anomaly_top,
        "routine_event_metrics": routine_event_metrics if routine_event_metrics is not None else pd.DataFrame(),
        "spike_metrics": spike_metrics if spike_metrics is not None else pd.DataFrame(),
    }




def main() -> None:
    set_seed()
    ensure_dirs()

    raw, daily, meta = load_and_build_daily()
    daily = add_routine_event_flags(daily)
    raw.to_csv(OUT_DIR / "item_level_input_for_B.csv", index=False, encoding="utf-8-sig")
    daily[["user_id", "date", *CATEGORIES, "total", "has_transaction"]].to_csv(
        OUT_DIR / "daily_user_spending.csv", index=False, encoding="utf-8-sig"
    )
    daily.to_csv(OUT_DIR / "daily_user_spending_with_features.csv", index=False, encoding="utf-8-sig")

    plot_user_series(daily)

    metrics, forecasts, training_info = run_forecasting(daily)
    metrics.to_csv(OUT_DIR / "forecasting_metrics.csv", index=False, encoding="utf-8-sig")
    forecasts.to_csv(OUT_DIR / "forecast_results.csv", index=False, encoding="utf-8-sig")
    routine_event_metrics = evaluate_routine_event_forecasts(forecasts, daily)
    routine_event_metrics.to_csv(
        OUT_DIR / "routine_vs_event_forecast_metrics.csv", index=False, encoding="utf-8-sig"
    )
    model_behavior = build_model_behavior_diagnostics(metrics)
    model_behavior.to_csv(OUT_DIR / "model_behavior_diagnostics.csv", index=False, encoding="utf-8-sig")
    user_summary_for_suitability = make_summary_tables(raw, daily, metrics, pd.DataFrame())["user_summary"]
    model_suitability = build_model_suitability(metrics, routine_event_metrics, user_summary_for_suitability)
    model_suitability.to_csv(OUT_DIR / "model_suitability_by_user.csv", index=False, encoding="utf-8-sig")
    plot_forecast_examples(forecasts)

    anomaly = run_anomaly_detection(daily)
    anomaly.to_csv(OUT_DIR / "anomaly_scores.csv", index=False, encoding="utf-8-sig")
    spike_metrics = evaluate_spike_detection(anomaly)
    spike_metrics.to_csv(OUT_DIR / "spike_detection_metrics.csv", index=False, encoding="utf-8-sig")
    threshold_sensitivity = evaluate_threshold_sensitivity(anomaly)
    threshold_sensitivity.to_csv(OUT_DIR / "anomaly_threshold_sensitivity.csv", index=False, encoding="utf-8-sig")
    anomaly_top = summarize_anomaly_reasons(anomaly)
    anomaly_top.to_csv(OUT_DIR / "top_anomaly_cases.csv", index=False, encoding="utf-8-sig")
    plot_anomalies(anomaly)

    tables = make_summary_tables(raw, daily, metrics, anomaly_top, routine_event_metrics, spike_metrics)
    with (OUT_DIR / "run_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(meta | {"training_info": training_info}, f, ensure_ascii=False, indent=2)

    print("Member B pipeline completed.")
    print(f"Outputs: {OUT_DIR}")


if __name__ == "__main__":
    main()
