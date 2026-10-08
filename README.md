# Export Demand Forecasting for International Trade

Forecasts monthly export values per commodity using time-series feature engineering and machine learning, and benchmarks it against a seasonal-naive baseline.

## Approach
- Features: lags (1, 2, 3, 6, 12 months), 3-month rolling mean, cyclical month encoding, trend, commodity one-hot
- Models: Ridge Regression, Random Forest, Gradient Boosting
- Validation: time-based hold-out (last 12 months), no shuffling
- Metrics: MAPE, RMSE

## Run
```
pip install -r requirements.txt
python forecast.py                       # demo on simulated data
python forecast.py --csv trade_data.csv  # columns: date, commodity, value
```
Outputs: `results.csv`, `forecast.png`, `importance.png`.

## Data
Demo uses simulated monthly export data. Replace with real data (e.g. India's Ministry of Commerce TradeStat or UN Comtrade) by exporting a CSV with `date, commodity, value`.

## Results (simulated demo data)
| Model | MAPE % |
|---|---|
| Ridge | 3.88 |
| Gradient Boosting | 4.61 |
| Random Forest | 4.83 |
| Seasonal naive | 4.90 |
