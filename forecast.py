"""Export Demand Forecasting for International Trade.

Compares a seasonal-naive baseline with ML models (Ridge, Random Forest,
Gradient Boosting) on monthly export values per commodity.

Usage:
  python forecast.py                      # runs on simulated demo data
  python forecast.py --csv your_data.csv  # needs columns: date, commodity, value
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error


def simulate(seed=42):
    """Simulated monthly export values (USD mn) for 3 commodities, 2015-2024."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2015-01-01", "2024-12-01", freq="MS")
    rows = []
    for name, base, amp in [("Textiles", 1200, 0.18), ("Pharmaceuticals", 2000, 0.08), ("Engineering Goods", 3000, 0.12)]:
        t = np.arange(len(dates))
        v = base * (1 + 0.004 * t) * (1 + amp * np.sin(2 * np.pi * (dates.month - 3) / 12))
        v = v * (1 + rng.normal(0, 0.04, len(dates)))
        rows += [(d, name, x) for d, x in zip(dates, v)]
    return pd.DataFrame(rows, columns=["date", "commodity", "value"])


def make_features(df, lags=(1, 2, 3, 6, 12)):
    df = df.sort_values(["commodity", "date"]).copy()
    for l in lags:
        df[f"lag_{l}"] = df.groupby("commodity")["value"].shift(l)
    df["roll_3"] = df.groupby("commodity")["value"].transform(lambda s: s.shift(1).rolling(3).mean())
    df["month_sin"] = np.sin(2 * np.pi * df["date"].dt.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["date"].dt.month / 12)
    df["t"] = (df["date"].dt.year - df["date"].dt.year.min()) * 12 + df["date"].dt.month
    df = pd.get_dummies(df, columns=["commodity"], prefix="c")
    return df.dropna()


def main(csv=None, test_months=12):
    raw = pd.read_csv(csv, parse_dates=["date"]) if csv else simulate()
    commodity = raw["commodity"].copy()
    data = make_features(raw)
    feats = [c for c in data.columns if c not in ("date", "value")]
    cutoff = data["date"].max() - pd.DateOffset(months=test_months)
    train, test = data[data["date"] <= cutoff], data[data["date"] > cutoff]

    models = {
        "Ridge": Ridge(alpha=1.0),
        "RandomForest": RandomForestRegressor(n_estimators=300, random_state=0),
        "GradientBoosting": GradientBoostingRegressor(random_state=0),
    }
    results, preds = {}, {}
    # seasonal-naive baseline = value 12 months ago
    base = test["lag_12"].values
    results["Seasonal naive (baseline)"] = (
        mean_absolute_percentage_error(test["value"], base),
        mean_squared_error(test["value"], base) ** 0.5,
    )
    for n, m in models.items():
        m.fit(train[feats], train["value"])
        p = m.predict(test[feats])
        preds[n] = p
        results[n] = (mean_absolute_percentage_error(test["value"], p), mean_squared_error(test["value"], p) ** 0.5)

    res = pd.DataFrame(results, index=["MAPE", "RMSE"]).T.sort_values("MAPE")
    res["MAPE"] = (res["MAPE"] * 100).round(2)
    res["RMSE"] = res["RMSE"].round(1)
    print(res.to_string())
    res.to_csv("results.csv")

    best = res.drop(index="Seasonal naive (baseline)").index[0]
    plt.figure(figsize=(9, 4))
    plt.plot(test["date"], test["value"], ".", label="Actual", alpha=0.6)
    plt.plot(test["date"], preds[best], "x", label=f"{best} forecast", alpha=0.6)
    plt.title(f"Hold-out forecast vs actual ({best})")
    plt.legend(); plt.tight_layout(); plt.savefig("forecast.png", dpi=120)

    if best in ("RandomForest", "GradientBoosting"):
        imp = pd.Series(models[best].feature_importances_, index=feats).sort_values()
        plt.figure(figsize=(6, 4)); imp.tail(8).plot.barh()
        plt.title("Top feature importances"); plt.tight_layout(); plt.savefig("importance.png", dpi=120)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None)
    main(ap.parse_args().csv)
