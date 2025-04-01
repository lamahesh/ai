import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, GRU
from sklearn.preprocessing import MinMaxScaler
import requests
from textblob import TextBlob
import os
import http.client
import requests
import streamlit as st
# Disable ONEDNN for TensorFlow optimization
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'




# Streamlit App Title
st.title("\U0001F4C8 Stock Vista : Smart stock insights & predictions in real-time.")

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Stock Prediction", "Market Dashboard",  "Commodities Dashboard", "Cryptocurrency Dashboard","Stock News","Stock Price Alerts" , "Market Sentiment" ,"Portfolio Tracker"  ])
# ---------- Market Dashboard ----------
import yfinance as yf

if page == "Market Dashboard":
    st.header("\U0001F4CA Real-Time Market Dashboard")
    
    indices = {
        "NIFTY 50": "^NSEI",
        "SENSEX": "^BSESN"
    }

    for name, ticker in indices.items():
        try:
            data = yf.download(ticker, period="1d", interval="1m")
            if not data.empty:
                price = float(data["Close"].iloc[-1])  # Convert Series to float
                st.metric(label=name, value=f"{price:.2f} INR")
            else:
                st.write(f"⚠ No data found for {name}")
        except Exception as e:
            st.write(f"⚠ Could not retrieve data for {name}: {e}")

# ---------- Stock Prediction ----------
elif page == "Stock Prediction":
    st.header("\U0001F4C9 AI-Based Stock Price Prediction")
    user_input = st.text_input("Enter Stock Ticker (e.g., AAPL, MSFT):", "AAPL")
    
    if st.button("Fetch Data"):
        stock = user_input.upper()
        start = '2019-01-01'  # Last 5 years
        end = '2024-12-31'
        df = yf.download(stock, start, end)
        
        st.subheader('Stock Data')
        st.write(df.tail())
        st.line_chart(df['Close'])
        
        data_train = df.Close[:int(len(df) * 0.80)]
        data_test = df.Close[int(len(df) * 0.80):]
        scaler = MinMaxScaler(feature_range=(0, 1))
        data_train_scaled = scaler.fit_transform(np.array(data_train).reshape(-1, 1))
        
        x_train, y_train = [], []
        for i in range(100, len(data_train_scaled)):
            x_train.append(data_train_scaled[i-100:i])
            y_train.append(data_train_scaled[i, 0])
        x_train, y_train = np.array(x_train), np.array(y_train)
        
        model = Sequential([
            GRU(50, activation='relu', return_sequences=True, input_shape=(x_train.shape[1], 1)),
            Dropout(0.2),
            GRU(60, activation='relu', return_sequences=True),
            Dropout(0.3),
            GRU(80, activation='relu', return_sequences=True),
            Dropout(0.4),
            GRU(120, activation='relu'),
            Dropout(0.5),
            Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(x_train, y_train, epochs=20, batch_size=64, verbose=1)
        st.write("Model training complete.")
        
        # Generate Plot of Actual vs Predicted
        predictions = model.predict(x_train)
        plt.figure(figsize=(12, 6))
        plt.plot(range(len(y_train)), y_train, label="Actual Prices", color='blue')
        plt.plot(range(len(predictions)), predictions, label="Predicted Prices", color='red')
        plt.xlabel("Days")
        plt.ylabel("Stock Price")
        plt.legend()
        st.pyplot(plt)

# ---------- Portfolio Tracker ----------
# ---------- Portfolio Tracker ----------
elif page == "Portfolio Tracker":
    st.header("\U0001F4BC Portfolio Tracker")
    portfolio = st.session_state.get("portfolio", [])
    
    stock_symbol = st.text_input("Enter Stock Ticker (e.g., AAPL, TSLA):").upper()
    shares = st.number_input("Enter Number of Shares:", min_value=1, step=1)
    add_stock = st.button("Add to Portfolio")
    
    if add_stock and stock_symbol:
        portfolio.append({"symbol": stock_symbol, "shares": shares})
        st.session_state["portfolio"] = portfolio
    
    if portfolio:
        st.subheader("Your Portfolio")
        portfolio_df = pd.DataFrame(portfolio)
        prices = []
        
        for stock in portfolio:
            try:
                stock_data = yf.Ticker(stock["symbol"]).history(period="1d")
                current_price = stock_data["Close"].iloc[-1] if not stock_data.empty else None
                prices.append(current_price)
            except Exception:
                prices.append(None)
        
        portfolio_df["Current Price"] = prices
        portfolio_df["Total Value"] = portfolio_df["shares"] * portfolio_df["Current Price"]
        
        st.dataframe(portfolio_df)
        
        total_value = portfolio_df["Total Value"].sum()
        st.metric(label="Total Portfolio Value", value=f"${total_value:.2f}")

# ---------- Stock News ----------
elif page == "Stock News":
    st.header("\U0001F4F0 Latest Stock Market News & Sentiment Analysis")
    api_key = "a7a81622a96e40c59cdb9bda9aed36ff"
    url = f"https://newsapi.org/v2/top-headlines?category=business&language=en&apiKey={api_key}"
    
    try:
        response = requests.get(url)
        news_data = response.json()
        
        if news_data.get("status") == "ok":
            for article in news_data.get("articles", [])[:5]:
                sentiment = TextBlob(article['title']).sentiment.polarity
                sentiment_label = "Positive" if sentiment > 0 else "Negative" if sentiment < 0 else "Neutral"
                st.markdown(f"[{article['title']}]({article['url']}) ({sentiment_label})")
                st.write(article['description'])
                if article.get('urlToImage'):
                    st.image(article['urlToImage'], width=400)
                st.write("---")
        else:
            st.write(f"⚠ Error fetching news: {news_data.get('message', 'Unknown error')}")
    except requests.exceptions.RequestException as e:
        st.write(f"⚠ Network error: {e}")
import yfinance as yf
#----------------Cryptocurrency Dashboard------------------#
if page == "Cryptocurrency Dashboard":
    st.header("\U0001F4B0 Cryptocurrency Prices")

    cryptos = {
        "Bitcoin": "BTC-USD",
        "Ethereum": "ETH-USD"
    }

    for name, ticker in cryptos.items():
        try:
            data = yf.download(ticker, period="1d", interval="1m")
            if not data.empty:
                price = float(data["Close"].iloc[-1])  # Convert Series to float
                st.metric(label=name, value=f"${price:.2f} USD")
            else:
                st.write(f"⚠ No data found for {name}")
        except Exception as e:
            st.write(f"⚠ Could not retrieve data for {name}: {e}")




# ---------- Commodities Dashboard ----------
# ---------- Commodities Dashboard ----------

    st.header("\U0001F4B0 Gold & Silver Prices in INR")

        
elif page == "Commodities Dashboard":
    st.header("\U0001F4B0 Gold & Silver Prices in INR")

    # API Key for GoldAPI.io
    api_key = "goldapi-9zv0b7sm8x1u56y-io"
    base_url = "https://www.goldapi.io/api"

    commodities = {"Gold": "XAU/INR", "Silver": "XAG/INR"}

    for name, ticker in commodities.items():
        try:
            url = f"{base_url}/{ticker}"
            headers = {"x-access-token": api_key, "Content-Type": "application/json"}
            response = requests.get(url, headers=headers)
            data = response.json()

            if response.status_code == 200 and "price" in data:
                price = data["price"]
                st.metric(label=name, value=f"₹{price:.2f}")
            else:
                st.write(f"⚠ Could not fetch {name} price at the moment. Try again later.")

        except Exception as e:
            st.write(f"⚠ Could not retrieve data for {name}: {e}")
# ---------- Stock Price Alerts ----------
if page == "Stock Price Alerts":
    st.header("\U0001F514 Set Stock Price Alerts")

    stock_alert_ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, TSLA):").upper()
    target_price = st.number_input("Set Target Price:", min_value=0.0, format="%.2f")
    check_alert = st.button("Check Price")

    if check_alert and stock_alert_ticker:
        try:
            stock_data = yf.Ticker(stock_alert_ticker).history(period="1d")
            current_price = stock_data["Close"].iloc[-1] if not stock_data.empty else None

            if current_price:
                st.metric(label=f"{stock_alert_ticker} Current Price", value=f"${current_price:.2f}")

                if current_price >= target_price:
                    st.success(f"\U0001F389 Alert! {stock_alert_ticker} has reached your target price of ${target_price:.2f}")
                else:
                    st.info(f"\U0001F4C8 {stock_alert_ticker} is currently at ${current_price:.2f}. Waiting to reach ${target_price:.2f}.")

            else:
                st.write(f"⚠ No data found for {stock_alert_ticker}")

        except Exception as e:
            st.write(f"⚠ Could not retrieve data for {stock_alert_ticker}: {e}")
# ---------- Fear & Greed Index ----------



# Streamlit Page for Fear & Greed Index
if page == "Market Sentiment":
    st.header("📊 Fear & Greed Index ")

    # API URL
    url = "https://api.alternative.me/fng/"

    try:
        # API Request
        response = requests.get(url)
        data = response.json()

        # Check if response contains data
        if "data" in data and len(data["data"]) > 0:
            index_value = data["data"][0]["value"]
            index_label = data["data"][0]["value_classification"]

            # Display Fear & Greed Index
            st.metric(label="Fear & Greed Index", value=index_value, delta=index_label)

            # Market Sentiment Description
            sentiment_desc = {
                "Extreme Greed": "🟢 Investors are highly optimistic. Possible overvaluation.",
                "Greed": "🟡 Investors are confident. Stocks may be overbought.",
                "Neutral": "⚪ Balanced market sentiment.",
                "Fear": "🔴 Investors are cautious. Buying opportunities may arise.",
                "Extreme Fear": "🔥 Panic in the market. Potentially undervalued stocks."
            }
            st.write(f"**Market Sentiment:** {sentiment_desc.get(index_label, 'Unknown')}")

        else:
            st.warning("⚠ Unable to fetch Fear & Greed Index. Try again later.")

    except Exception as e:
        st.error(f"⚠ Error fetching data: {e}")
