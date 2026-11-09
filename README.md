# Daily Climate Forecasting (Delhi) with RNN, LSTM and GRU

A data mining course project that explores four years of daily weather data from Delhi, India, and compares three recurrent neural networks (SimpleRNN, LSTM and GRU) for forecasting from past observations. Everything is in one Jupyter notebook: exploratory analysis, preprocessing, model training and evaluation.

## Dataset

Daily Delhi climate data, 1 January 2013 to 24 April 2017, collected from the Weather Underground API.

| File | Rows | Period |
|---|---|---|
| `DailyDelhiClimateTrain.csv` | 1,462 | 2013-01-01 to 2017-01-01 |
| `DailyDelhiClimateTest.csv` | 114 | 2017-01-01 to 2017-04-24 |

Features: `meantemp`, `humidity`, `wind_speed`, `meanpressure`. There are no missing values. 

## What I did

1. **Exploration:** line plots of each variable over time, average yearly pattern by month, histograms with KDE, and a correlation heatmap.
2. **Seasonal decomposition:** split mean temperature into trend, seasonal and residual parts (additive model, 365-day period) with `statsmodels`.
3. **Feature engineering:** extracted year, month, day and day of week from the date index.
4. **Feature selection:** dropped `meanpressure`, because it has a weak correlation with temperature and contains extreme outliers and spikes. The models use `meantemp`, `humidity` and `wind_speed`.
5. **Preprocessing:** 80/20 train/test split, Min-Max scaling to the 0-1 range (scaler fitted on the training part only), and sliding windows of 12 days as model input.
6. **Modeling:** trained and compared three recurrent architectures, trying different activation functions (tanh and relu), sequence lengths (3, 7, 12, 30) and layer sizes (50 and 100 units for the RNN).

## Findings from the data

- Temperature, humidity and wind speed all show a clear yearly seasonal cycle.
- Temperature and humidity are negatively correlated (about -0.57).
- Wind speed and pressure are weakly correlated (about 0.29).
- The pressure series has several extreme spikes that look like measurement errors.

## Models

| Model | Architecture | Training |
|---|---|---|
| SimpleRNN | SimpleRNN(128) → Dropout(0.2) → Dense(64) → Dropout(0.2) → Dense(32) → Dropout(0.2) → Dense(1), tanh | Adam, MSE loss, 30 epochs |
| LSTM | LSTM(64) → Dropout → LSTM(32) → Dropout → LSTM(16) → Dropout → Dense(1), tanh and relu versions | Adam, MSE loss, up to 20 epochs, early stopping (patience 5) |
| GRU | GRU(128) → Dense(1) | Adam, MSE loss, up to 30 epochs, early stopping (patience 5) |

## Results

Error metrics from the project report, computed on the scaled (0-1) data:

| Model | MAE | MSE | RMSE |
|---|---|---|---|
| SimpleRNN | 0.0714 | 0.0093 | 0.0965 |
| LSTM | 0.0465 | 0.0034 | 0.0581 |
| GRU | 0.0428 | 0.0027 | 0.0520 |

The GRU had the lowest error on all three metrics, with the LSTM close behind and the SimpleRNN last.



Requires Python 3 and these libraries:

```
pip install pandas numpy matplotlib seaborn plotly cufflinks statsmodels scikit-learn tensorflow jupyter
```

Download the two CSV files, put them next to the notebook, then open the notebook and run the cells in order.



## Notes

Course project for Data Mining, University of Tehran. Dataset credit goes to the Weather Underground API.
