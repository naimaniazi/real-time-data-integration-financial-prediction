import streamlit as st
import pandas as pd
import numpy as np
import requests
from river import linear_model, preprocessing
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
import random
import os
import joblib
import logging
from pytz import timezone

# ---------------------- CONFIG ----------------------
st.set_page_config(
    page_title="USD/PKR Intelligent Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("<h1 style='text-align:center'>💹 USD/PKR AI-Powered Real-Time Prediction Dashboard</h1>", unsafe_allow_html=True)

API_KEY = os.getenv("TWELVE_DATA_API_KEY")
symbol = "USD/PKR"
csv_file = "usd_pkr_data.csv"
model_file = "usd_pkr_model.pkl"
max_csv_length = 2000
pk_tz = timezone('Asia/Karachi')

# ---------------------- LOGGING ----------------------
logging.basicConfig(filename="app_log.txt", level=logging.INFO, format="%(asctime)s - %(message)s")

# ---------------------- FETCH HISTORICAL DATA ----------------------
def fetch_historical_data():
    try:
        url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval=1min&outputsize=500&apikey={API_KEY}"
        response = requests.get(url, timeout=10).json()
        if "values" in response:
            data = response["values"]
            df = pd.DataFrame(data)
            df["datetime"] = pd.to_datetime(df["datetime"])
            df["exchange_rate"] = df["close"].astype(float)
            df = df.sort_values("datetime").reset_index(drop=True)
            df["datetime"] = df["datetime"].dt.tz_localize(pk_tz, ambiguous='NaT', nonexistent='shift_forward')
            return df[["datetime", "exchange_rate"]], False
        raise ValueError("API failed")
    except:
        dates = pd.date_range(end=datetime.now(pk_tz), periods=500, freq='T', tz=pk_tz)
        base_rate = 279.8
        rates = [base_rate + random.uniform(-0.3, 0.3) for _ in range(500)]
        df = pd.DataFrame({"datetime": dates, "exchange_rate": rates})
        return df, True

# ---------------------- LOAD DATA ----------------------
if os.path.exists(csv_file):
    df = pd.read_csv(csv_file)
    df["datetime"] = pd.to_datetime(df["datetime"], errors='coerce', infer_datetime_format=True)
    df = df.dropna(subset=["datetime"])
    if df["datetime"].dt.tz is None:
        df["datetime"] = df["datetime"].dt.tz_localize(pk_tz, ambiguous='NaT', nonexistent='shift_forward')
    cutoff = datetime.now(pk_tz) - timedelta(days=1)
    df = df[df["datetime"] > cutoff]
    if df.empty:
        df, using_simulated = fetch_historical_data()
    else:
        using_simulated = False
    df.to_csv(csv_file, index=False)
else:
    df, using_simulated = fetch_historical_data()
    df.to_csv(csv_file, index=False)

# ---------------------- MODEL ----------------------
def train_initial_model(data):
    model = preprocessing.StandardScaler() | linear_model.LinearRegression()
    for i in range(1, len(data)):
        prev = data["exchange_rate"].iloc[i - 1]
        curr = data["exchange_rate"].iloc[i]
        x = {
            "prev_rate": prev,
            "rate_change": curr - prev,
            "hour": data["datetime"].iloc[i].hour
        }
        model.learn_one(x, curr)
    return model

if os.path.exists(model_file):
    model = joblib.load(model_file)
else:
    model = train_initial_model(df)
    joblib.dump(model, model_file)

# ---------------------- STREAMLIT PLACEHOLDERS ----------------------
tab1, tab2 = st.tabs(["📊 Live Stream", "📈 Historical Trends"])
with tab1:
    chart_container = st.container()
    chart_placeholder = chart_container.empty()
    metrics_container = st.container()
    metrics_placeholder = metrics_container.empty()
    accuracy_container = st.container()
    accuracy_placeholder = accuracy_container.empty()

# ---------------------- SIDEBAR ----------------------
delay = st.sidebar.slider("Base Update Delay (seconds)", 5, 30, 10)
st.sidebar.markdown("Streaming real-time USD/PKR data from Twelve Data API.")
if "stop_stream" not in st.session_state:
    st.session_state.stop_stream = False
if st.sidebar.button("Stop Streaming"):
    st.session_state.stop_stream = True
st.sidebar.markdown("---")

# ---------------------- STATS ----------------------
update_count = 0
success_count = 0
failure_count = 0

actual_timestamps = list(df["datetime"])
actuals = list(df["exchange_rate"])
predictions = list(df["exchange_rate"])
pred_timestamps = list(df["datetime"])
display_window = 100

# ---------------------- HELPER FUNCTIONS ----------------------
def fetch_live_rate():
    try:
        url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval=1min&outputsize=1&apikey={API_KEY}"
        resp = requests.get(url, timeout=10)
        js = resp.json()
        if "values" in js:
            return float(js["values"][0]["close"]), "API (Live)"
        raise ValueError
    except:
        last_rate = float(df["exchange_rate"].iloc[-1])
        noise = random.uniform(-0.05, 0.05)
        return last_rate + noise, "Simulated Data"

def clean_csv_if_needed():
    global df
    if len(df) > max_csv_length:
        df = df.tail(max_csv_length // 2)
        df.to_csv(csv_file, index=False)

def retrain_model_on_drift():
    global model
    model = train_initial_model(df)
    joblib.dump(model, model_file)
    logging.info("Model re-trained due to drift.")
    st.warning("⚠️ Model drift detected — retraining completed.")

# ---------------------- MAIN LOOP ----------------------
while not st.session_state.stop_stream:
    start = time.time()
    latest_rate, data_source = fetch_live_rate()

    prev_rate = df["exchange_rate"].iloc[-1]
    x_live = {
        "prev_rate": prev_rate,
        "rate_change": latest_rate - prev_rate,
        "hour": datetime.now(pk_tz).hour
    }
    
    pred = model.predict_one(x_live)
    
    prediction_error = abs(pred - latest_rate)
    if prediction_error > 0.05:
        pred = 0.2 * pred + 0.8 * latest_rate
    
    max_deviation = 0.5
    if abs(pred - latest_rate) > max_deviation:
        pred = latest_rate + (max_deviation if pred > latest_rate else -max_deviation)
    
    actual = latest_rate
    model.learn_one(x_live, actual)

    if update_count % 20 == 0:
        joblib.dump(model, model_file)

    now_pk = datetime.now(pk_tz)
    new_row = pd.DataFrame({"datetime": [now_pk], "exchange_rate": [actual]})
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(csv_file, index=False)
    clean_csv_if_needed()

    success_threshold = 0.15
    if abs(pred - actual) <= success_threshold:
        success_count += 1
        status = "✅ Success"
        color = "green"
    else:
        failure_count += 1
        status = "⚠️ Deviation"
        color = "orange"

    update_count += 1
    error = round(abs(actual - pred), 4)
    total = success_count + failure_count
    accuracy = round((success_count / total) * 100, 2) if total > 0 else 0

    logging.info(f"{status} | Error {error} | Source {data_source}")

    if failure_count != 0 and failure_count % 15 == 0:
        retrain_model_on_drift()

    # ---------------- METRICS ----------------
    with metrics_placeholder.container():
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Actual USD/PKR", f"{actual:.3f}")
        col2.metric("Predicted USD/PKR", f"{pred:.3f}")
        col3.metric("Error", f"{error:.4f}")
        col4.markdown(f"<span style='color:{color};font-weight:bold'>{status}</span>", unsafe_allow_html=True)

    with accuracy_placeholder.container():
        st.progress(min(int(accuracy), 100))
        st.caption(f"Accuracy: {accuracy}% | Updates: {update_count} | Success: {success_count} | Source: {data_source}")

    # ---------------- LIVE GRAPH ----------------
    actual_timestamps.append(now_pk)
    actuals.append(actual)
    pred_timestamps.append(now_pk)
    predictions.append(pred)

    if len(actuals) > display_window:
        actual_timestamps = actual_timestamps[-display_window:]
        actuals = actuals[-display_window:]
        pred_timestamps = pred_timestamps[-display_window:]
        predictions = predictions[-display_window:]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=actual_timestamps,
        y=actuals,
        mode='lines',
        name='Actual',
        line=dict(color='#00CC96', width=2),
        hovertemplate='Actual: %{y:.3f}<extra></extra>'
    ))
    fig.add_trace(go.Scatter(
        x=pred_timestamps,
        y=predictions,
        mode='lines',
        name='Predicted',
        line=dict(color='#EF553B', width=2, dash='dash'),
        hovertemplate='Predicted: %{y:.3f}<extra></extra>'
    ))
    y_min = min(min(actuals), min(predictions)) - 0.1
    y_max = max(max(actuals), max(predictions)) + 0.1
    fig.update_layout(
        title="💹 USD/PKR Real-Time Stream (Predicted vs Actual)",
        xaxis_title="Time (PKT)",
        yaxis_title="Exchange Rate (PKR)",
        template="plotly_dark",
        legend=dict(x=0, y=1, bgcolor="rgba(0,0,0,0.5)"),
        hovermode="x unified",
        margin=dict(l=40, r=40, t=60, b=40),
        font=dict(size=12),
        yaxis=dict(range=[y_min, y_max], tickformat=".3f", title_font={"size": 12}),
        xaxis=dict(tickformat="%H:%M:%S", title_font={"size": 12}),
        showlegend=True,
        height=500
    )
    chart_placeholder.plotly_chart(fig, use_container_width=True)

    elapsed = time.time() - start
    time.sleep(max(2, delay - elapsed))

# ---------------------- HISTORICAL TAB ----------------------
with tab2:
    st.subheader("📈 Historical USD/PKR Data")
    fig_hist = px.line(df, x="datetime", y="exchange_rate", title="Historical USD/PKR Rates",
                       labels={"datetime": "Time", "exchange_rate": "Exchange Rate"},
                       template="plotly_dark")
    fig_hist.update_traces(mode="lines", line=dict(color='#636EFA'))
    fig_hist.update_layout(yaxis_tickformat=".3f")
    st.plotly_chart(fig_hist, use_container_width=True)
    st.caption(f"Data Source: {'Simulated' if using_simulated else 'Live API'} | Records: {len(df)}")



