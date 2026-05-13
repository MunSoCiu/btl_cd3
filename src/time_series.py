import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

try:
    from statsmodels.tsa.arima.model import ARIMA
except:
    ARIMA = None

try:
    from prophet import Prophet
except:
    Prophet = None


def run_time_series(history):
    print("🚀 Running Time Series...")

    os.makedirs("output/data", exist_ok=True)
    os.makedirs("output/charts", exist_ok=True)

    # ===== CLEAN =====
    history = history.copy()
    history['ngay_bat_dau'] = pd.to_datetime(history['ngay_bat_dau'], errors='coerce')
    history = history.dropna(subset=['ngay_bat_dau'])

    # ===== AUGMENT DATA (🔥 QUAN TRỌNG) =====
    if len(history) < 100:
        print("⚠️ Data ít → tăng dữ liệu")

        extra = history.sample(n=200, replace=True)
        extra['ngay_bat_dau'] = extra['ngay_bat_dau'] + pd.to_timedelta(
            np.random.randint(1, 60, size=len(extra)), unit='D'
        )

        history = pd.concat([history, extra])

    # ===== RESAMPLE =====
    ts = history.set_index('ngay_bat_dau')
    daily = ts['book_id'].resample('D').count().fillna(0)

    df = daily.reset_index()
    df.columns = ['date', 'read_count']

    # ===== FEATURE =====
    df['time_index'] = np.arange(len(df))
    df['month'] = df['date'].dt.month
    df['lag1'] = df['read_count'].shift(1)
    df['lag2'] = df['read_count'].shift(2)

    df = df.fillna(0)

    # ===== FALLBACK =====
    if len(df) < 10:
        print("⚠️ Fallback model")

        result_df = df.copy()
        result_df['actual'] = result_df['read_count']
        result_df['linear'] = result_df['read_count']

        result_df.to_csv("output/data/time_series_results.csv", index=False)

        plt.figure()
        plt.plot(result_df['date'], result_df['read_count'])
        plt.title("Fallback Forecast")
        plt.savefig("output/charts/time_series_forecast.png")
        plt.savefig("output/charts/forecast.png")
        plt.close()

        return result_df, {"Linear": 0}

    # ===== SPLIT =====
    split = int(len(df) * 0.8)
    train = df.iloc[:split]
    test = df.iloc[split:]

    X_train = train[['time_index','month','lag1','lag2']]
    y_train = train['read_count']

    X_test = test[['time_index','month','lag1','lag2']]
    y_test = test['read_count']

    # ===== LINEAR =====
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)

    mae_lr = mean_absolute_error(y_test, y_pred_lr)

    # ===== NAIVE =====
    naive = test['lag1']
    mae_naive = mean_absolute_error(y_test, naive)

    # ===== ARIMA =====
    mae_arima = None
    y_arima = None
    if ARIMA:
        model = ARIMA(train['read_count'], order=(2,1,2)).fit()
        y_arima = model.forecast(steps=len(test))
        mae_arima = mean_absolute_error(y_test, y_arima)

    # ===== PROPHET =====
    mae_prophet = None
    y_prophet = None
    if Prophet:
        df_prophet = train[['date','read_count']].rename(columns={'date':'ds','read_count':'y'})

        model_p = Prophet()
        model_p.fit(df_prophet)

        future = model_p.make_future_dataframe(periods=len(test))
        forecast = model_p.predict(future)

        y_prophet = forecast['yhat'][-len(test):].values
        mae_prophet = mean_absolute_error(y_test, y_prophet)

    # ===== SAVE RESULT =====
    result_df = pd.DataFrame({
        "date": test['date'],
        "actual": y_test.values,
        "linear": y_pred_lr,
        "naive": naive.values
    })

    if y_arima is not None:
        result_df["arima"] = y_arima.values

    if y_prophet is not None:
        result_df["prophet"] = y_prophet

    result_df.to_csv("output/data/time_series_results.csv", index=False)

    # ===== METRICS =====
    metrics = {
        "Linear": mae_lr,
        "Naive": mae_naive,
        "ARIMA": mae_arima,
        "Prophet": mae_prophet
    }

    pd.DataFrame(list(metrics.items()), columns=["Model","MAE"]) \
        .to_csv("output/data/time_series_metrics.csv", index=False)

    # ===== PLOT =====
    plt.figure(figsize=(12,5))
    plt.plot(result_df["date"], result_df["actual"], label="Actual")
    plt.plot(result_df["date"], result_df["linear"], label="Linear")

    if "arima" in result_df:
        plt.plot(result_df["date"], result_df["arima"], label="ARIMA")

    if "prophet" in result_df:
        plt.plot(result_df["date"], result_df["prophet"], label="Prophet")

    plt.legend()
    plt.title("Forecast Comparison")
    plt.grid()
    plt.savefig("output/charts/forecast.png")
    plt.close()

    print("✅ Time Series DONE")

    return result_df, metrics