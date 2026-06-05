from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, precision_score, recall_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn


BASE = Path(__file__).resolve().parent
OUT = BASE / "synthetic_validation"
FIG = OUT / "figures"
INPUT_WINDOW = 30
HORIZON = 7
RNG = np.random.default_rng(42)
torch.manual_seed(42)
AE_WINDOW = 14


class LSTMForecaster(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 48, horizon: int = HORIZON):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 48),
            nn.ReLU(),
            nn.Linear(48, horizon),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


class LSTMAutoencoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 40, latent_size: int = 20):
        super().__init__()
        self.encoder = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.to_latent = nn.Linear(hidden_size, latent_size)
        self.from_latent = nn.Linear(latent_size, hidden_size)
        self.decoder = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.output = nn.Linear(hidden_size, input_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h, _) = self.encoder(x)
        latent = torch.relu(self.to_latent(h[-1]))
        h0 = torch.relu(self.from_latent(latent)).unsqueeze(0)
        c0 = torch.zeros_like(h0)
        decoder_input = torch.zeros_like(x)
        out, _ = self.decoder(decoder_input, (h0, c0))
        return self.output(out)


def generate_synthetic_daily() -> pd.DataFrame:
    dates = pd.date_range("2025-09-01", "2026-05-31", freq="D")
    rows = []
    configs = {
        0: {"name": "regular_weekly", "base": 260, "weekend": 180, "noise": 35, "sparse": 0.03, "spike": 900},
        1: {"name": "very_regular", "base": 520, "weekend": 260, "noise": 45, "sparse": 0.01, "spike": 1400},
        2: {"name": "sparse_irregular", "base": 180, "weekend": 120, "noise": 95, "sparse": 0.38, "spike": 1100},
    }

    for uid, cfg in configs.items():
        anomaly_idx = set(RNG.choice(np.arange(35, len(dates) - 10), size=9, replace=False).tolist())
        for i, date in enumerate(dates):
            dow = date.dayofweek
            month_cycle = 55 * np.sin(2 * np.pi * date.day / 30.5)
            trend = 0.25 * i if uid == 1 else 0.08 * i
            weekly = cfg["weekend"] if dow >= 5 else 0
            amount = cfg["base"] + weekly + month_cycle + trend + RNG.normal(0, cfg["noise"])

            has_transaction = 1
            if RNG.random() < cfg["sparse"]:
                amount = 0
                has_transaction = 0

            known_anomaly = 0
            anomaly_type = ""
            if i in anomaly_idx:
                known_anomaly = 1
                anomaly_type = "injected_spike"
                amount += cfg["spike"] + RNG.normal(0, cfg["spike"] * 0.15)
                has_transaction = 1

            amount = max(0, amount)
            food = amount * (0.62 + RNG.normal(0, 0.04))
            shopping = amount * (0.24 + RNG.normal(0, 0.05))
            transport = amount * (0.08 + RNG.normal(0, 0.02))
            medical = max(0, amount - food - shopping - transport)
            rows.append(
                {
                    "user_id": uid,
                    "profile": cfg["name"],
                    "date": date,
                    "total": round(amount, 2),
                    "has_transaction": has_transaction,
                    "飲食": round(max(0, food), 2),
                    "購物": round(max(0, shopping), 2),
                    "交通": round(max(0, transport), 2),
                    "醫療健康": round(max(0, medical), 2),
                    "known_anomaly": known_anomaly,
                    "anomaly_type": anomaly_type,
                }
            )
    return pd.DataFrame(rows)


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["day_of_week"] = df["date"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["day_of_month"] = df["date"].dt.day
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["day_of_month"] / 31)
    df["month_cos"] = np.cos(2 * np.pi * df["day_of_month"] / 31)
    return df


def build_windows(g: pd.DataFrame):
    totals = g["total"].to_numpy(dtype=float)
    time_feats = g[["dow_sin", "dow_cos", "month_sin", "month_cos", "is_weekend", "has_transaction"]].to_numpy(dtype=float)
    x_amount, x_time, x_seq_time, y, y_anomaly, origins = [], [], [], [], [], []
    for start in range(0, len(g) - INPUT_WINDOW - HORIZON + 1):
        end = start + INPUT_WINDOW
        x_amount.append(totals[start:end])
        x_time.append(np.column_stack([totals[start:end], time_feats[start:end]]).reshape(-1))
        x_seq_time.append(np.column_stack([totals[start:end], time_feats[start:end]]))
        y.append(totals[end : end + HORIZON])
        y_anomaly.append(g["known_anomaly"].iloc[end : end + HORIZON].to_numpy(dtype=int))
        origins.append(g["date"].iloc[end])
    return np.array(x_amount), np.array(x_time), np.array(x_seq_time), np.array(y), np.array(y_anomaly), np.array(origins)


def train_lstm_forecaster(x_seq: np.ndarray, y: np.ndarray, train_end: int, val_end: int) -> np.ndarray:
    scaler = StandardScaler()
    flat_train = x_seq[:train_end].reshape(-1, x_seq.shape[-1])
    scaler.fit(flat_train)
    x_scaled = scaler.transform(x_seq.reshape(-1, x_seq.shape[-1])).reshape(x_seq.shape).astype(np.float32)
    y_scaled = y.astype(np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMForecaster(input_size=x_seq.shape[-1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.SmoothL1Loss()

    x_train = torch.tensor(x_scaled[:train_end], device=device)
    y_train = torch.tensor(y_scaled[:train_end], device=device)
    x_val = torch.tensor(x_scaled[train_end:val_end], device=device)
    y_val = torch.tensor(y_scaled[train_end:val_end], device=device)

    best_state = None
    best_val = float("inf")
    patience = 0
    for _ in range(180):
        model.train()
        opt.zero_grad()
        loss = loss_fn(model(x_train), y_train)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(x_val), y_val).item() if len(x_val) else loss.item()
        if val_loss < best_val - 1e-4:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
        if patience >= 25:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(x_scaled, device=device)).cpu().numpy()
    return np.maximum(0, pred)


def evaluate_forecasting(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows = []
    result_rows = []

    for uid, g in df.groupby("user_id"):
        g = g.sort_values("date").reset_index(drop=True)
        x_amount, x_time, x_seq_time, y, y_anomaly, origins = build_windows(g)
        n = len(y)
        train_end = int(n * 0.70)
        val_end = int(n * 0.85)
        test_idx = np.arange(val_end, n)

        preds = {}
        moving = []
        weekday = []
        for idx in range(n):
            origin_pos = idx + INPUT_WINDOW
            hist = g.iloc[idx:origin_pos]
            future_dates = g["date"].iloc[origin_pos : origin_pos + HORIZON]
            moving.append(np.repeat(hist["total"].tail(7).mean(), HORIZON))
            weekday.append(
                [
                    hist.loc[hist["date"].dt.dayofweek == d.dayofweek, "total"].tail(6).mean()
                    if len(hist.loc[hist["date"].dt.dayofweek == d.dayofweek]) > 0
                    else hist["total"].tail(7).mean()
                    for d in future_dates
                ]
            )
        preds["MovingAverage7"] = np.array(moving)
        preds["WeekdayAverage"] = np.array(weekday)

        for name, x in [("MLP_amount_only", x_amount), ("MLP_time_features", x_time)]:
            scaler = StandardScaler()
            x_train = scaler.fit_transform(x[:train_end])
            x_all = scaler.transform(x)
            model = MLPRegressor(hidden_layer_sizes=(64, 32), activation="relu", max_iter=900, random_state=42)
            model.fit(x_train, y[:train_end])
            preds[name] = np.maximum(0, model.predict(x_all))

        preds["LSTM_time_features"] = train_lstm_forecaster(x_seq_time, y, train_end, val_end)

        for method, pred in preds.items():
            actual_test = y[test_idx].reshape(-1)
            pred_test = pred[test_idx].reshape(-1)
            anomaly_test = y_anomaly[test_idx].reshape(-1).astype(bool)
            mae = mean_absolute_error(actual_test, pred_test)
            rmse = float(np.sqrt(mean_squared_error(actual_test, pred_test)))
            normal_mae = mean_absolute_error(actual_test[~anomaly_test], pred_test[~anomaly_test]) if (~anomaly_test).any() else np.nan
            anomaly_mae = mean_absolute_error(actual_test[anomaly_test], pred_test[anomaly_test]) if anomaly_test.any() else np.nan
            actual_std = float(np.std(actual_test))
            pred_std = float(np.std(pred_test))
            std_ratio = pred_std / actual_std if actual_std > 0 else 0.0
            metric_rows.append(
                {
                    "user_id": uid,
                    "profile": g["profile"].iloc[0],
                    "method": method,
                    "MAE": mae,
                    "normal_day_MAE": normal_mae,
                    "known_anomaly_day_MAE": anomaly_mae,
                    "RMSE": rmse,
                    "actual_std": actual_std,
                    "prediction_std": pred_std,
                    "std_ratio": std_ratio,
                    "collapsed_prediction": int(std_ratio < 0.10),
                    "n_test_windows": len(test_idx),
                }
            )
            for local_i, idx in enumerate(test_idx):
                for h in range(HORIZON):
                    result_rows.append(
                        {
                            "user_id": uid,
                            "profile": g["profile"].iloc[0],
                            "forecast_origin": origins[idx],
                            "horizon_day": h + 1,
                            "target_date": g["date"].iloc[idx + INPUT_WINDOW + h],
                            "actual": y[idx, h],
                            "known_anomaly": int(y_anomaly[idx, h]),
                            "method": method,
                            "prediction": pred[idx, h],
                        }
                    )
    return pd.DataFrame(metric_rows), pd.DataFrame(result_rows)


def evaluate_anomaly_detection(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = []
    scores = []
    feature_cols = ["total", "飲食", "購物", "交通", "醫療健康", "has_transaction", "dow_sin", "dow_cos"]

    for uid, g in df.groupby("user_id"):
        g = g.sort_values("date").reset_index(drop=True)
        train_end = int(len(g) * 0.70)
        scaler = StandardScaler()
        values = g[feature_cols].to_numpy(dtype=np.float32)
        scaled = scaler.fit_transform(values[:train_end]).astype(np.float32)
        scaled_all = scaler.transform(values).astype(np.float32)

        seqs, starts = [], []
        for start in range(0, len(g) - AE_WINDOW + 1):
            seqs.append(scaled_all[start : start + AE_WINDOW])
            starts.append(start)
        seqs = np.array(seqs, dtype=np.float32)
        starts = np.array(starts)
        train_seq_mask = starts + AE_WINDOW <= train_end

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = LSTMAutoencoder(input_size=seqs.shape[-1]).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        loss_fn = nn.MSELoss()
        train_x = torch.tensor(seqs[train_seq_mask], device=device)

        for _ in range(180):
            model.train()
            opt.zero_grad()
            recon = model(train_x)
            loss = loss_fn(recon, train_x)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        with torch.no_grad():
            recon = model(torch.tensor(seqs, device=device)).cpu().numpy()
        seq_err = ((recon - seqs) ** 2).mean(axis=2)
        raw_score = np.zeros(len(g), dtype=float)
        counts = np.zeros(len(g), dtype=float)
        for seq_i, start in enumerate(starts):
            for offset in range(AE_WINDOW):
                day_i = start + offset
                raw_score[day_i] += seq_err[seq_i, offset]
                counts[day_i] += 1
        raw_score = raw_score / np.maximum(counts, 1)
        threshold = np.quantile(raw_score[AE_WINDOW:train_end], 0.965)
        pred = (raw_score >= threshold).astype(int)
        true = g["known_anomaly"].to_numpy(dtype=int)

        metrics.append(
            {
                "user_id": uid,
                "profile": g["profile"].iloc[0],
                "method": "LSTM_Autoencoder",
                "known_anomaly_days": int(true.sum()),
                "detected_days": int(pred.sum()),
                "true_positive": int(((pred == 1) & (true == 1)).sum()),
                "false_positive": int(((pred == 1) & (true == 0)).sum()),
                "missed_anomaly": int(((pred == 0) & (true == 1)).sum()),
                "recall": recall_score(true, pred, zero_division=0),
                "precision": precision_score(true, pred, zero_division=0),
            }
        )
        tmp = g[["user_id", "profile", "date", "total", "known_anomaly"]].copy()
        tmp["anomaly_score"] = raw_score
        tmp["threshold"] = threshold
        tmp["detected_anomaly"] = pred
        scores.append(tmp)
    return pd.DataFrame(metrics), pd.concat(scores, ignore_index=True)


def make_figures(df: pd.DataFrame, forecast_results: pd.DataFrame, forecast_metrics: pd.DataFrame, anomaly_scores: pd.DataFrame) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    for uid, g in df.groupby("user_id"):
        best_method = select_reporting_method(forecast_metrics[forecast_metrics["user_id"] == uid])
        show_methods = ["MovingAverage7", "WeekdayAverage", "LSTM_time_features", best_method]
        day1 = forecast_results[
            (forecast_results["user_id"] == uid)
            & (forecast_results["horizon_day"] == 1)
            & (forecast_results["method"].isin(list(dict.fromkeys(show_methods))))
        ].copy()
        pivot = day1.pivot_table(index="target_date", columns="method", values="prediction", aggfunc="first")
        actual = day1.drop_duplicates("target_date").set_index("target_date")["actual"]
        plt.figure(figsize=(11, 3.8))
        plt.plot(actual.index, actual.values, label="Actual day+1", color="#1f77b4", linewidth=1.8)
        for method in pivot.columns:
            color = "#d62728" if method == best_method else ("#9467bd" if method == "LSTM_time_features" else "#ff7f0e")
            style = "-" if method == best_method else "--"
            plt.plot(pivot.index, pivot[method], label=method, linewidth=1.5, alpha=0.85, color=color, linestyle=style)
        plt.title(f"Synthetic user {uid}: day+1 forecast, best={best_method}")
        plt.ylabel("Amount")
        ax = plt.gca()
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=7))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
        plt.xticks(rotation=35, ha="right")
        plt.grid(alpha=0.25)
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(FIG / f"synthetic_user_{uid}_forecast.png", dpi=170)
        plt.close()

        s = anomaly_scores[anomaly_scores["user_id"] == uid].copy()
        plt.figure(figsize=(11, 3.8))
        plt.plot(s["date"], s["total"], color="#1f77b4", linewidth=1.5, label="Daily total")
        known = s[s["known_anomaly"] == 1]
        detected = s[s["detected_anomaly"] == 1]
        plt.scatter(known["date"], known["total"], color="#d62728", s=35, label="Injected anomaly", zorder=3)
        plt.scatter(detected["date"], detected["total"], facecolors="none", edgecolors="#111111", s=80, label="Detected", zorder=4)
        plt.title(f"Synthetic user {uid}: injected vs detected anomalies")
        plt.ylabel("Amount")
        ax = plt.gca()
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=7))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
        plt.xticks(rotation=35, ha="right")
        plt.grid(alpha=0.25)
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(FIG / f"synthetic_user_{uid}_anomaly.png", dpi=170)
        plt.close()


def select_reporting_method(metrics_for_user: pd.DataFrame) -> str:
    usable = metrics_for_user[metrics_for_user["collapsed_prediction"] == 0]
    if usable.empty:
        usable = metrics_for_user
    return usable.sort_values("MAE").iloc[0]["method"]



def main() -> None:
    OUT.mkdir(exist_ok=True)
    df = add_time_features(generate_synthetic_daily())
    forecast_metrics, forecast_results = evaluate_forecasting(df)
    anomaly_metrics, anomaly_scores = evaluate_anomaly_detection(df)
    make_figures(df, forecast_results, forecast_metrics, anomaly_scores)

    df.to_csv(OUT / "synthetic_daily_spending.csv", index=False, encoding="utf-8-sig")
    forecast_metrics.to_csv(OUT / "synthetic_forecast_metrics.csv", index=False, encoding="utf-8-sig")
    forecast_results.to_csv(OUT / "synthetic_forecast_results.csv", index=False, encoding="utf-8-sig")
    anomaly_metrics.to_csv(OUT / "synthetic_anomaly_detection_metrics.csv", index=False, encoding="utf-8-sig")
    anomaly_scores.to_csv(OUT / "synthetic_anomaly_scores.csv", index=False, encoding="utf-8-sig")
    print(OUT)


if __name__ == "__main__":
    main()
