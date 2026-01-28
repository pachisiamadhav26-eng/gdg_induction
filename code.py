import streamlit as st
import yfinance as yf
import google.generativeai as genai
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.graph_objects as go
from datetime import date
import pandas as pd
import time
from duckduckgo_search import DDGS
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# --- CONFIGURATION & SETUP ---
st.set_page_config(layout="wide", page_title="StockAI Real-Time")

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 1. SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input("Gemini API Key", type="password")
    
    ticker_choice = st.selectbox(
        "Choose a Stock:",
        ["AAPL", "TSLA", "GOOGL", "MSFT", "NVDA", "AMZN", "BTC-USD", "ETH-USD", "Other"]
    )
    
    if ticker_choice == "Other":
        ticker = st.text_input("Enter Custom Ticker", "SPY").upper()
    else:
        ticker = ticker_choice

    start_date = st.date_input("Start Date", value=pd.to_datetime("2020-01-01"))
    end_date = date.today()
    st.info(f"Analyzing: **{ticker}**")

# --- 2. HELPER FUNCTIONS ---

@st.cache_data(ttl=3600) 
def fetch_stock_data(symbol, start, end):
    try:
        data = yf.download(symbol, start=start, end=end)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        return data
    except Exception:
        return None


@st.cache_data(ttl=86400)
def generate_forecast(data, days=7):
    # 1. Prepare Data
    df_train = data.reset_index()[['Date', 'Close']]
    df_train.columns = ['ds', 'y']
    df_train['ds'] = df_train['ds'].dt.tz_localize(None)
    
    # 2. Train Model
    m = Prophet(daily_seasonality=True)
    m.fit(df_train)
    
    # 3. Predict Future + History
    future = m.make_future_dataframe(periods=days)
    forecast = m.predict(future)
    
    # --- FIXED: CALCULATE ACCURACY METRICS ---
    # Merge actuals and predictions so we can drop missing rows safely
    forecast_past = forecast[['ds', 'yhat']]
    merged = df_train.merge(forecast_past, on='ds', how='inner')
    
    # Drop any rows where data is missing (NaN)
    merged = merged.dropna()

    y_true = merged['y']
    y_pred = merged['yhat']
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # Calculate MAPE (and avoid division by zero)
    # We add a tiny number (1e-10) to y_true to prevent crash if price is 0
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100
    
    metrics = {"MAE": mae, "RMSE": rmse, "MAPE": mape}
    # ---------------------------------------

    fig = plot_plotly(m, forecast)
    return forecast, fig, metrics



@st.cache_data(ttl=3600)
def fetch_stock_news(symbol):
    try:
        results = DDGS().text(f"{symbol} stock news", max_results=5)
        if not results:
            return "No news found."
        
        formatted_news = ""
        for result in results:
            formatted_news += f"- Title: {result['title']}\n"
            formatted_news += f"  Link: {result['href']}\n"
            formatted_news += f"  Summary: {result.get('body', '')}\n\n"
            
        return formatted_news
    except Exception as e:
        return f"Error fetching news: {e}"

# --- NEW TASK 3: LIVE DATA FUNCTION ---
def fetch_live_data(symbol):
    data = yf.download(symbol, period="1d", interval="1m", progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    return data

# --- 3. MAIN APP LOGIC ---
st.title(f"📈 {ticker} Intelligence Platform")

data = fetch_stock_data(ticker, start_date, end_date)

if data is None or data.empty:
    st.error("Error fetching data.")
else:
    # THE 4 MAIN TABS
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Market", "🔮 Forecast", "🤖 AI Analyst", "🔴 Live Stream"])

    # --- TAB 1: Market ---
    with tab1:
        fig = go.Figure(data=[go.Candlestick(x=data.index,
                        open=data['Open'], high=data['High'],
                        low=data['Low'], close=data['Close'])])
        fig.update_layout(xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)


# --- TAB 2: Forecast ---
    with tab2:
        if st.button("Generate Forecast"):
            with st.spinner("Forecasting..."):
                # Note: We now unpack 3 values (df, fig, metrics)
                f_df, f_fig, metrics = generate_forecast(data)
                
                # --- NEW: Display Metrics in Columns ---
                st.subheader("Model Accuracy (Backtest on Historical Data)")
                col1, col2, col3 = st.columns(3)
                col1.metric("MAE (Avg Error)", f"${metrics['MAE']:.2f}")
                col2.metric("RMSE (Large Error Penalty)", f"${metrics['RMSE']:.2f}")
                col3.metric("Model Accuracy", f"{100 - metrics['MAPE']:.2f}%")
                # ---------------------------------------

                st.plotly_chart(f_fig, use_container_width=True)
                
                st.subheader("7-Day Price Prediction")
                future_data = f_df.tail(7)[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
                future_data.columns = ['Date', 'Predicted Price', 'Low Estimate', 'High Estimate']
                st.dataframe(future_data)



# --- TAB 3: AI Analyst (Fixed Date) ---
    with tab3:
        if not api_key:
            st.warning("Enter Gemini Key in Sidebar.")
        else:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemma-3-27b-it')
            
            # --- NEW: IMPORT DATE MODULE ---
            import datetime
            
            for msg in st.session_state.messages:
                st.chat_message(msg["role"]).write(msg["content"])

            if prompt := st.chat_input("Ask about this stock..."):
                st.session_state.messages.append({"role": "user", "content": prompt})
                st.chat_message("user").write(prompt)
                
                with st.spinner("Analyzing..."):
                    # 1. Fetch Basic Data
                    price = data['Close'].iloc[-1]
                    news = fetch_stock_news(ticker)
                    
                    # 2. Get TODAY'S DATE (The Fix)
                    today_date = datetime.date.today().strftime("%B %d, %Y") # e.g., "January 28, 2026"
                    
                    # 3. Historical Data
                    yearly_stats = data.resample('Y').agg({'Close': ['min', 'max']})
                    yearly_stats.columns = ['Year_Low', 'Year_High']
                    yearly_stats.index = yearly_stats.index.strftime('%Y')
                    yearly_text = yearly_stats.to_string()
                    
                    # 4. The Context with Date Injection
                    context = f"""
                    You are a strict Financial Analyst.
                    
                    ### SYSTEM INFO
                    - **TODAY'S DATE:** {today_date} (Use this for all time-based answers)
                    - **CURRENT STOCK:** {ticker}
                    
                    ### DATA PROVIDED
                    - Current Price: ${price:.2f}
                    - Yearly Records:
                    {yearly_text}
                    - Latest News:
                    {news}
                    
                    ### USER QUESTION
                    {prompt}
                    
                    ### INSTRUCTIONS
                    1. Always assume the current date is **{today_date}**.
                    2. If asked for the "Current Price", state the price and reference **{today_date}**, not 2023.
                    3. If the user asks about a different stock, refuse to answer.
                    """
                    
                    try:
                        response = model.generate_content(context)
                        st.chat_message("assistant").write(response.text)
                        st.session_state.messages.append({"role": "assistant", "content": response.text})
                    except Exception as e:
                        if "429" in str(e):
                            st.warning("🚦 Speed Limit Hit! Please wait 1 minute.")
                        else:
                            st.error(f"Error: {e}")

    # --- TAB 4: TASK 3 (LIVE STREAMING) ---
    with tab4:
        st.subheader("🔴 Real-Time Data Stream")
        
        col1, col2 = st.columns(2)
        with col1:
            curr_val = float(data['Close'].iloc[-1])
            threshold = st.number_input("Set Alert Threshold ($)", value=curr_val)
        with col2:
            st.write("") 
            start_stream = st.button("Start Live Stream 🚀")
            
        price_placeholder = st.empty()
        alert_placeholder = st.empty()
        chart_placeholder = st.empty()

        if start_stream:
            st.success("Streaming started... (Updates every 60s)")
            
            while True:
                live_data = fetch_live_data(ticker)
                
                if not live_data.empty:
                    current_price = live_data['Close'].iloc[-1]
                    
                    with price_placeholder.container():
                        st.metric(
                            label=f"Live Price ({ticker})", 
                            value=f"${current_price:.2f}",
                            delta=f"{current_price - live_data['Open'].iloc[-1]:.2f}"
                        )
                    
                    if current_price > threshold:
                        alert_placeholder.error(f"🚨 ALERT: Price crossed above ${threshold}!")
                    elif current_price < threshold:
                         alert_placeholder.warning(f"📉 ALERT: Price dropped below ${threshold}!")
                    else:
                        alert_placeholder.info("✅ Price within normal range.")

                    with chart_placeholder.container():
                        fig_live = go.Figure()
                        fig_live.add_trace(go.Scatter(
                            x=live_data.index, 
                            y=live_data['Close'],
                            mode='lines+markers',
                            name='Live Price'
                        ))
                        fig_live.update_layout(title="Intraday Live Chart (1m Interval)")
                        
                        # --- UNIQUE KEY FIX FOR DUPLICATE ERROR ---
                        st.plotly_chart(fig_live, use_container_width=True, key=f"live_chart_{time.time()}")
                
                time.sleep(60)
