# Real-Time-Data-Integration-Financial-Prediction
A Streamlit-based real-time dashboard for monitoring USD/PKR exchange rate. The app fetches live Forex data, predicts the next value using a machine learning model (River library), and displays both actual and predicted rates on a dynamic, interactive graph.
## **Purpose**
Track USD/PKR rates in real-time and predict future movement using an adaptive online learning model.

---

## **Tech Stack**
- **Python 3.11+**
- **Streamlit** (frontend/dashboard)
- **River** (online machine learning)
- **Plotly** (interactive charts)
- **Aiohttp + Asyncio** (asynchronous API calls)
- **Pandas/Numpy** (data handling)
- **Joblib** (model saving/loading)
- **Twelve Data API** (live Forex data)
- **Pytz** (timezone handling)

---

## **Key Features**
- Real-time live streaming of USD/PKR rates
- Online learning model updates continuously
- Predicted vs Actual overlay graph
- Metrics and accuracy display
- Historical trends tab
- Rolling window graph for smooth performance

---

## **Installation**

```bash
# 1. Create virtual environment
python -m venv venv

# 2. Activate environment
source venv/bin/activate   # Linux / Mac
# venv\Scripts\activate    # Windows

# 3. Install dependencies
pip install streamlit pandas numpy aiohttp river plotly pytz joblib requests

# 4. Run the Streamlit app
streamlit run app.py
 ```
