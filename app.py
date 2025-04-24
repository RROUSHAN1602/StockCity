
import streamlit as st
import yfinance as yf
from SmartApi.smartConnect import SmartConnect
import pyotp
import pandas as pd
import plotly.graph_objects as go
import datetime as dt
import ta
from sklearn.linear_model import LinearRegression
import numpy as np
import requests
from bs4 import BeautifulSoup
import os
from dotenv import load_dotenv
import urllib.parse
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense # type: ignore
from backtesting import Backtest, Strategy
from prophet import Prophet
from streamlit_lottie import st_lottie


# --- PAGE SETUP ---
st.set_page_config(page_title="📊 Angel One Screener", layout="wide")

# --- SIDEBAR MENU ---
st.sidebar.image("https://img.icons8.com/external-flat-icons-inmotus-design/512/external-stock-stock-market-flat-icons-inmotus-design.png", width=100)
st.sidebar.title("📈 Stock Screener")
st.sidebar.caption("Powered by Angel One + Streamlit")

page = st.sidebar.radio(
    "Navigation", 
    ["🏠 Home", "🔍 Fundamentals", "📉 Charts & Indicators", "🤖 AI Prediction", "📰 News Sentiment", "📊 Advanced Analysis", "⭐ Watchlist"]
)

@st.cache_data(ttl=300)
def fetch_yahoo_indices():
    tickers = {
        "NIFTY 50": "^NSEI",
        "SENSEX": "^BSESN",
        "BANK NIFTY": "^NSEBANK",
        "MIDCAP 100": "^CNXMDCP",
        "FIN NIFTY": "^CNXFIN"
    }
    result = {}
    for name, symbol in tickers.items():
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="1d", interval="1m")
            if not data.empty:
                last = round(data["Close"].iloc[-1], 2)
                prev_close = round(data["Close"].iloc[-2], 2)
                change = round(last - prev_close, 2)
                p_change = round((change / prev_close) * 100, 2)
                result[name] = {"last": last, "change": change, "pChange": p_change}
        except Exception as e:
            continue
    return result

@st.cache_data
def get_nse_stock_list():
    return [
        "RELIANCE", "TCS", "INFY", "SBIN", "HDFCBANK",
        "ITC", "AXISBANK", "BAJFINANCE", "HCLTECH", "LT",
        "WIPRO", "ICICIBANK", "KOTAKBANK", "ONGC", "ADANIENT",
        "POWERGRID", "NTPC", "COALINDIA", "HINDUNILVR", "ULTRACEMCO"
    ]

symbol_map = {
    "RELIANCE": "2885", "TCS": "11536", "INFY": "1594", "SBIN": "3045",
    "HDFCBANK": "1333", "ITC": "1660", "AXISBANK": "2751",
    "BAJFINANCE": "8949", "HCLTECH": "1330", "LT": "11483",
    "WIPRO": "3787", "ICICIBANK": "4963", "KOTAKBANK": "1922",
    "ONGC": "2475", "ADANIENT": "25", "POWERGRID": "14977",
    "NTPC": "2973", "COALINDIA": "20374", "HINDUNILVR": "1394",
    "ULTRACEMCO": "11532"
}

interval_map = {
    "1 Month ": ("ONE_DAY", 30),
    "3 Months ": ("ONE_DAY", 90),
    "6 Months ": ("ONE_DAY", 180),
    "1 Year ": ("ONE_DAY", 365),
    "5 Days ": ("FIVE_MINUTE", 5)
}

all_tickers = get_nse_stock_list()
stock_choice = st.sidebar.selectbox("Select NSE Stock", all_tickers)
selected_label = st.sidebar.selectbox("Select Time Frame for Chart", list(interval_map.keys()))
interval_choice, days = interval_map[selected_label]

load_dotenv()

API_KEY = os.getenv("ANGEL_API_KEY")
CLIENT_ID = os.getenv("ANGEL_CLIENT_ID")
PASSWORD = os.getenv("ANGEL_PASSWORD")
TOTP_SECRET = os.getenv("ANGEL_TOTP_SECRET")

try:
    totp = pyotp.TOTP(TOTP_SECRET).now()
    obj = SmartConnect(api_key=API_KEY)
    session = obj.generateSession(CLIENT_ID, PASSWORD, totp)
except Exception as e:
    st.error(f"Login failed: {e}")
    st.stop()


st.markdown("""
    <div style='background-color:#111; color:#f39c12; padding:10px; border-radius:5px; margin-top:30px;
    overflow:hidden; white-space:nowrap; animation: scroll-left 12s linear infinite;'>
        ⚠️ Disclaimer: This dashboard is for informational purposes only and not a SEBI-registered advisory. Please consult a certified financial advisor before investing.
    </div>

    <style>
    @keyframes scroll-left {
    0%   {transform: translateX(100%);}
    100% {transform: translateX(-100%);}
    }
    </style>
    """, unsafe_allow_html=True)


# --- FETCH LIVE PRICE ---
try:
    token = symbol_map.get(stock_choice)
    live_data = obj.ltpData(exchange="NSE", tradingsymbol=stock_choice, symboltoken=token)
    ltp = round(live_data["data"]["ltp"], 2)
    st.metric(label=f"📈 Live Price of {stock_choice}", value=f"₹ {ltp}")
except:
    st.warning("⚠️ Could not fetch live price.")

# --- PAGE: Home ---
if page == "🏠 Home":
    st.title("📈 Market Overview Dashboard")


    # --- Index Prices from NSE India ---
    def fetch_all_indices():
        try:
            session = requests.Session()
            headers = {
                "User-Agent": "Mozilla/5.0",
                "Referer": "https://www.nseindia.com"
            }
            session.get("https://www.nseindia.com", headers=headers)
            url = "https://www.nseindia.com/api/allIndices"
            r = session.get(url, headers=headers)
            r.raise_for_status()
            return r.json().get("data", [])
        except Exception as e:
            print("Error fetching all indices:", e)
            return []

    def fetch_index_from_nse(index_symbol):
        try:
            data = fetch_all_indices()
            for index in data:
                if index_symbol.upper() in index['index'].upper():
                    return index['last'], index['percentChange']
            return "NA", "NA"
        except:
            return "NA", "NA"


    # --- Bing News Scraper ---
    def fetch_bing_news(query):
        try:
            headers = {"User-Agent": "Mozilla/5.0"}
            url = f"https://www.bing.com/news/search?q={query.replace(' ', '+')}"
            r = requests.get(url, headers=headers)
            soup = BeautifulSoup(r.text, "html.parser")
            titles = soup.find_all("a", class_="title")
            return [t.text.strip() for t in titles[:5]]
        except:
            return []


    # --- Commodity from investing.com ---
    def get_commodity_price(name):
        try:
            headers = {"User-Agent": "Mozilla/5.0"}
            url = f"https://www.investing.com/commodities/{name}"
            r = requests.get(url, headers=headers)
            soup = BeautifulSoup(r.text, "html.parser")
            price = soup.select_one("div[data-test='instrument-price-last'] span")
            return price.text.strip() if price else "NA"
        except:
            return "NA"


    # --- Fetch Index Data ---
    nifty, nifty_change = fetch_index_from_nse("NIFTY 50")
    niftyfinservice, niftyfinservice_change = fetch_index_from_nse("NIFTY FINANCIAL SERVICES")
    banknifty, banknifty_change = fetch_index_from_nse("NIFTY BANK")
    midcap100, midcap_change = fetch_index_from_nse("NIFTY MIDCAP 100")



    # 📊 Index Cards
    st.subheader("📊 Key Indices")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("NIFTY 50", f"₹ {nifty}", f"{nifty_change}%")
    col2.metric("BANK NIFTY", f"₹ {banknifty}", f"{banknifty_change}%")
    col3.metric("FINNIFTY", f"₹ {niftyfinservice}", f"{niftyfinservice_change}%")
    col4.metric("MIDCAP 100", f"₹ {midcap100}", f"{midcap_change}%")


    # ──────────────────────────────────────────────
    # 📥 Scraping Functions for Each Corporate Action
    # ──────────────────────────────────────────────

    # --- Animated Title ---
    st.markdown("""
    <style>
    @keyframes popZoom {
    0% {transform: scale(0.95); opacity: 0;}
    100% {transform: scale(1); opacity: 1;}
    }
    .animated-heading {
        animation: popZoom 1s ease-out forwards;
        font-size: 36px;
        text-align: center;
        color: #ffffff;
        margin-bottom: 20px;
        font-weight: bold;
    }
    </style>
    <h2 class="animated-heading">📅 Market Event Highlights</h2>
    """, unsafe_allow_html=True)

    # --- Lottie Loader ---
    def load_lottie_url(url):
        try:
            r = requests.get(url)
            return r.json() if r.status_code == 200 else None
        except:
            return None

    # --- Bing News Fetcher ---
    def bing_news_search(query, count=4):
        headers = {"User-Agent": "Mozilla/5.0"}
        url = f"https://www.bing.com/news/search?q={query.replace(' ', '+')}+site:moneycontrol.com&FORM=HDRSC6"
        r = requests.get(url, headers=headers)
        soup = BeautifulSoup(r.text, "html.parser")
        results = []
        for item in soup.select("a.title")[:count]:
            results.append({
                "title": item.text.strip(),
                "url": item["href"]
            })
        return results

    # --- Load Working Animations ---
    dividend_lottie = load_lottie_url("https://assets2.lottiefiles.com/packages/lf20_jcikwtux.json")
    split_lottie = load_lottie_url("https://assets2.lottiefiles.com/packages/lf20_5ngs2ksb.json")
    meeting_lottie = load_lottie_url("https://assets2.lottiefiles.com/packages/lf20_ol7a7z.json")
    rights_lottie = load_lottie_url("https://assets2.lottiefiles.com/packages/lf20_kkflmtur.json")

    # --- Define Categories ---
    topics = {
        "📦 Dividends": {"query": "stock dividend announcement India", "anim": dividend_lottie},
        "🔀 Stock Splits": {"query": "stock split announced India", "anim": split_lottie},
        "📅 Board Meetings": {"query": "stock market board meeting agenda India", "anim": meeting_lottie},
        "🎫 Rights Issues": {"query": "stock market rights issue declared India", "anim": rights_lottie},
    }

    # --- Render News in 4 Columns with Vertical Scroll ---
    cols = st.columns(4)
    for i, (label, details) in enumerate(topics.items()):
        with cols[i]:
            st.markdown(f"<div style='text-align:center; font-size:18px; font-weight:bold; margin-bottom:10px;'>{label}</div>", unsafe_allow_html=True)

            if details["anim"]:
                st_lottie(details["anim"], height=100, key=label)

            news = bing_news_search(details["query"])

            if news:
                scroll_css = f"""
                <style>
                .scroll-box-{i} {{
                    height: 200px;
                    overflow: hidden;
                    position: relative;
                }}
                .scroll-content-{i} {{
                    display: block;
                    position: absolute;
                    top: 100%;
                    animation: scroll-up-{i} 15s linear infinite;
                }}
                .scroll-content-{i} div {{
                    margin-bottom: 18px;
                    line-height: 1.4;
                }}
                @keyframes scroll-up-{i} {{
                    0%   {{ top: 100%; }}
                    100% {{ top: -100%; }}
                }}
                </style>
                """
                st.markdown(scroll_css, unsafe_allow_html=True)
                st.markdown(f"<div class='scroll-box-{i}'><div class='scroll-content-{i}'>", unsafe_allow_html=True)
                for item in news:
                    st.markdown(
                        f"<div><a href='{item['url']}' target='_blank' style='text-decoration: none; color: #4ADE80;'>🔗 {item['title']}</a></div>",
                        unsafe_allow_html=True
                    )
                st.markdown("</div></div>", unsafe_allow_html=True)
            else:
                st.markdown("<div style='color: gray;'>No recent news found.</div>", unsafe_allow_html=True)

    # --- Footer ---
    st.markdown(
        "<br><div style='text-align:center; font-size:13px; color:gray;'>📱 Powered by Bing News • Source: Moneycontrol</div>",
        unsafe_allow_html=True
    )
  


    # 📂 Market Sentiment Summary
    st.subheader("📂 Market Sentiment Summary")
    try:
        change = float(nifty_change)
        if change > 0:
            st.success("📈 Market is Bullish")
        elif change < 0:
            st.error("📉 Market is Bearish")
        else:
            st.info("📊 Market is Flat")
    except:
        st.warning("Could not determine sentiment")

    # 🗞️ Top Headlines
    st.subheader("🗞️ Top Headlines")
    headlines = fetch_bing_news("Nifty 50")
    if headlines:
        for i, news in enumerate(headlines, 1):
            st.markdown(f"**{i}. {news}**")
    else:
        st.info("No headlines available.")

    # 📈 Top Gainers & Losers
    st.subheader("📈 Top Gainers & Losers")
    g1, g2 = st.columns(2)
    g1.markdown("**Top Gainers**")
    for g in fetch_bing_news("nifty top gainers"):
        g1.write(f"- {g}")
    g2.markdown("**Top Losers**")
    for l in fetch_bing_news("nifty top losers"):
        g2.write(f"- {l}")

    # 📤 Portfolio Upload & Risk Analyzer
    st.subheader("📤 Portfolio Risk Analyzer")
    uploaded_file = st.file_uploader("Upload your portfolio CSV", type=["csv"])
    if uploaded_file is not None:
        try:
            pf_df = pd.read_csv(uploaded_file)
            st.write("Your Portfolio:", pf_df)
            risk_score = pf_df["Weight"].std() if "Weight" in pf_df.columns else "N/A"
            st.success(f"📊 Risk Score: {round(risk_score, 2)}")
        except:
            st.error("❌ Failed to read the uploaded file.")

    # 🔎 Watchlist-based News
    st.subheader("🔎 News Based on Your Watchlist")
    if "watchlist" in st.session_state and st.session_state.watchlist:
        for stock in st.session_state.watchlist[:3]:
            st.markdown(f"**📌 {stock}**")
            for item in fetch_bing_news(stock):
                st.write(f"- {item}")
    else:
        st.info("Add stocks to your Watchlist to get personalized news.")


# --- PAGE: Fundamentals ---
elif page == "🔍 Fundamentals":
    st.title("📊 Key Financial Ratios")

    # Screener-compatible company slugs
    screener_slug_map = {
        "RELIANCE": "RELIANCE",
        "TCS": "TCS",
        "INFY": "INFOSYS",
        "SBIN": "SBIN",
        "HDFCBANK": "HDFCBANK",
        "ITC": "ITC",
        "AXISBANK": "AXISBANK",
        "BAJFINANCE": "BAJFINANCE",
        "HCLTECH": "HCLTECH",
        "LT": "LT",
        "WIPRO": "WIPRO",
        "ICICIBANK": "ICICIBANK",
        "KOTAKBANK": "KOTAKBANK",
        "ONGC": "ONGC",
        "ADANIENT": "ADANIENT",
        "POWERGRID": "POWERGRID",
        "NTPC": "NTPC",
        "COALINDIA": "COALINDIA",
        "HINDUNILVR": "HINDUNILVR",
        "ULTRACEMCO": "ULTRACEMCO"
    }

    # ✅ CORRECTLY INDENTED FUNCTION
    def fetch_fundamentals(ticker):
        try:
            slug = screener_slug_map.get(ticker, ticker)
            url = f"https://www.screener.in/company/{slug}/"
            headers = {"User-Agent": "Mozilla/5.0"}
            response = requests.get(url, headers=headers)

            if response.status_code != 200:
                st.warning("⚠️ Screener is currently not reachable.")
                return pd.DataFrame()

            soup = BeautifulSoup(response.text, "html.parser")
            data = []

            for item in soup.select("ul#top-ratios li"):
                label_elem = item.select_one("span.name")
                value_elem = item.select_one("span.value")
                if label_elem and value_elem:
                    label = label_elem.text.strip()
                    value = value_elem.text.strip()
                    data.append({"Metric": label, "Value": value})

            return pd.DataFrame(data)
        except Exception as e:
            st.error(f"❌ Error fetching data: {e}")
            return pd.DataFrame()

    # Fetch data
    df = fetch_fundamentals(stock_choice)

    if not df.empty:
        num_cols = 2
        total_metrics = len(df)
        rows = total_metrics // num_cols + int(total_metrics % num_cols != 0)

        st.markdown("### 🧮 Organized Ratio Grid")
        for row in range(rows):
            cols = st.columns(num_cols)
            for i in range(num_cols):
                idx = row * num_cols + i
                if idx < total_metrics:
                    metric = df.iloc[idx]["Metric"]
                    value = df.iloc[idx]["Value"]
                    with cols[i]:
                        st.markdown(
                            f"""
                            <div style='background-color:#111111; padding:20px; border-radius:12px;
                            border:1px solid #333333; text-align:center; margin-bottom:10px;'>
                                <h5 style='color:#aaaaaa; margin-bottom:8px;'>{metric}</h5>
                                <h3 style='color:#29b6f6;'>{value}</h3>
                            </div>
                            """, unsafe_allow_html=True
                        )
    else:
        st.warning("⚠️ Could not fetch fundamentals for this stock.")

# --- PAGE: Charts & Indicators ---
elif page == "📉 Charts & Indicators":
    try:
        st.title("📉 Technical Charts & AI Signals")

        end_time = dt.datetime.now()
        start_time = end_time - dt.timedelta(days=days)

        candle_data = obj.getCandleData({
            "exchange": "NSE",
            "symboltoken": token,
            "interval": interval_choice,
            "fromdate": start_time.strftime("%Y-%m-%d %H:%M"),
            "todate": end_time.strftime("%Y-%m-%d %H:%M")
        })

        df = pd.DataFrame(candle_data["data"], columns=["Datetime", "Open", "High", "Low", "Close", "Volume"])
        df["Datetime"] = pd.to_datetime(df["Datetime"])
        df.set_index("Datetime", inplace=True)

        # Indicators
        df["SMA_20"] = df["Close"].rolling(window=20).mean()
        df["EMA_10"] = df["Close"].ewm(span=10, adjust=False).mean()
        df["RSI"] = ta.momentum.RSIIndicator(df["Close"]).rsi()
        macd = ta.trend.MACD(df["Close"])
        df["MACD"] = macd.macd()
        df["MACD_Signal"] = macd.macd_signal()

        # AI-style signal generation
        df["Signal"] = "Hold"
        for i in range(1, len(df)):
            if df["EMA_10"].iloc[i] > df["SMA_20"].iloc[i] and df["EMA_10"].iloc[i - 1] <= df["SMA_20"].iloc[i - 1] and df["RSI"].iloc[i] < 70:
                df.at[df.index[i], "Signal"] = "Buy"
            elif df["EMA_10"].iloc[i] < df["SMA_20"].iloc[i] and df["EMA_10"].iloc[i - 1] >= df["SMA_20"].iloc[i - 1] and df["RSI"].iloc[i] > 30:
                df.at[df.index[i], "Signal"] = "Sell"

        # Buy/Sell points for plotting
        buy_signals = df[df["Signal"] == "Buy"]
        sell_signals = df[df["Signal"] == "Sell"]

        # --- Chart with OHLC + Volume + Signals ---
        fig = go.Figure()

        # Candlestick
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="Candles"
        ))

        # EMA/SMA overlays
        fig.add_trace(go.Scatter(x=df.index, y=df["EMA_10"], mode="lines", name="EMA 10", line=dict(color="orange")))
        fig.add_trace(go.Scatter(x=df.index, y=df["SMA_20"], mode="lines", name="SMA 20", line=dict(color="blue")))

        # --- Buy markers with text ---
        fig.add_trace(go.Scatter(
            x=buy_signals.index,
            y=buy_signals["Close"],
            mode="markers+text",
            name="Buy",
            text=["Buy"] * len(buy_signals),
            textposition="top center",
            marker=dict(color="lime", size=10, symbol="triangle-up")
        ))

        # --- Sell markers with text ---
        fig.add_trace(go.Scatter(
            x=sell_signals.index,
            y=sell_signals["Close"],
            mode="markers+text",
            name="Sell",
            text=["Sell"] * len(sell_signals),
            textposition="bottom center",
            marker=dict(color="red", size=10, symbol="triangle-down")
        ))

        # Volume bars
        fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume", marker=dict(color="lightblue"), yaxis="y2"))

        # Layout adjustments for volume axis
        fig.update_layout(
            yaxis=dict(title="Price"),
            yaxis2=dict(title="Volume", overlaying="y", side="right", showgrid=False),
            xaxis_rangeslider_visible=False,
            height=600,
            template="plotly_dark",
            title=f"{stock_choice} - Chart with AI Signals"
        )

        st.plotly_chart(fig, use_container_width=True)

        # --- RSI + MACD ---
        st.subheader("📈 RSI & MACD Indicators")
        st.line_chart(df[["RSI"]])
        st.line_chart(df[["MACD", "MACD_Signal"]])

    except Exception as e:
        st.error(f"Error: {e}")
elif page == "🤖 AI Prediction":
    try:
        st.title("🤖 AI-Based Advanced Signal Engine v2.0")

        end_time = dt.datetime.now()
        start_time = end_time - dt.timedelta(days=days)

        candle_data = obj.getCandleData({
            "exchange": "NSE",  
            "symboltoken": token,
            "interval": interval_choice,
            "fromdate": start_time.strftime("%Y-%m-%d %H:%M"),
            "todate": end_time.strftime("%Y-%m-%d %H:%M")
        })

        df = pd.DataFrame(candle_data["data"], columns=["Datetime", "Open", "High", "Low", "Close", "Volume"])
        df["Datetime"] = pd.to_datetime(df["Datetime"])
        df.set_index("Datetime", inplace=True)

        # 🛡️ Check if we have at least 20 rows to proceed
        if df.shape[0] < 20:
            st.warning("⚠️ Not enough data to generate prediction. Please select a longer time frame.")
        else:
            # --- Indicators ---
            df["EMA_20"] = df["Close"].ewm(span=20).mean()
            df["EMA_50"] = df["Close"].ewm(span=50).mean()
            df["EMA_200"] = df["Close"].ewm(span=200).mean()
            df["RSI"] = ta.momentum.RSIIndicator(df["Close"]).rsi()
            macd = ta.trend.MACD(df["Close"])
            df["MACD"] = macd.macd()
            df["MACD_Signal"] = macd.macd_signal()
            df["ATR"] = ta.volatility.AverageTrueRange(df["High"], df["Low"], df["Close"]).average_true_range()
            df["ADX"] = ta.trend.ADXIndicator(df["High"], df["Low"], df["Close"]).adx()
            df["PSAR"] = ta.trend.PSARIndicator(df["High"], df["Low"], df["Close"]).psar()
            df["Bollinger High"] = df["Close"].rolling(20).mean() + 2 * df["Close"].rolling(20).std()
            df["Bollinger Low"] = df["Close"].rolling(20).mean() - 2 * df["Close"].rolling(20).std()

            latest = df.iloc[-1]
            prev = df.iloc[-2]
            signal = "Hold"
            score = 0
            reasons = []

            # --- Signal Decision Logic (same as before) ---
            if latest["EMA_20"] > latest["EMA_50"] > latest["EMA_200"]:
                score += 2
                reasons.append("✅ Strong bullish EMA alignment")
            elif latest["EMA_20"] < latest["EMA_50"] < latest["EMA_200"]:
                score -= 2
                reasons.append("⛔ Bearish EMA alignment")

            if latest["MACD"] > latest["MACD_Signal"] and prev["MACD"] <= prev["MACD_Signal"]:
                score += 2
                reasons.append("✅ MACD Bullish Crossover")
            elif latest["MACD"] < latest["MACD_Signal"] and prev["MACD"] >= prev["MACD_Signal"]:
                score -= 2
                reasons.append("⛔ MACD Bearish Crossover")

            if latest["RSI"] < 30:
                score += 1
                reasons.append("✅ Oversold RSI (<30)")
            elif latest["RSI"] > 70:
                score -= 1
                reasons.append("⛔ Overbought RSI (>70)")

            if latest["Close"] > latest["PSAR"]:
                score += 1
                reasons.append("✅ PSAR indicates uptrend")
            else:
                score -= 1
                reasons.append("⛔ PSAR indicates downtrend")

            if latest["ADX"] > 25:
                score += 1
                reasons.append("✅ Strong trend confirmed by ADX")
            else:
                reasons.append("ℹ️ Weak trend (ADX < 25)")

            avg_volume = df["Volume"].rolling(20).mean().iloc[-1]
            if latest["Volume"] > avg_volume:
                score += 1
                reasons.append("✅ Volume supports price movement")
            else:
                reasons.append("⚠️ Weak volume")

            # Final Signal Decision
            if score >= 4:
                signal = "Buy"
            elif score <= -3:
                signal = "Sell"
            else:
                signal = "Hold"

            recent_high = df["High"].rolling(10).max().iloc[-1]
            recent_low = df["Low"].rolling(10).min().iloc[-1]
            target = round(recent_high * 1.02, 2)
            stop_loss = round(recent_low, 2)
            reward = round(target - latest["Close"], 2)
            risk = round(latest["Close"] - stop_loss, 2)
            rr_ratio = round(reward / risk, 2) if risk > 0 else "∞"
            confidence = f"{min(abs(score) * 20, 95)}%"

            # Display Results
            st.metric("📍 Final Signal", signal)
            st.markdown(f"**🎯 Confidence Level:** `{confidence}`")
            for r in reasons:
                st.markdown(f"- {r}")

            if signal == "Buy":
                st.success(f"🎯 Target: ₹{target}")
                st.warning(f"🛡️ Stop Loss: ₹{stop_loss}")
            elif signal == "Sell":
                st.error(f"📉 Downside Alert: Below ₹{stop_loss}")
                st.warning(f"🛡️ Resistance: ₹{target}")
            else:
                st.info("⚖️ Market indecisive. Wait for confirmation.")

            # Extra Metrics
            st.subheader("📈 Technical Stats")
            col1, col2, col3 = st.columns(3)
            col1.metric("🔄 ATR", round(latest["ATR"], 2))
            col2.metric("📊 ADX", round(latest["ADX"], 2))
            col3.metric("⚖️ Risk/Reward", rr_ratio)

            # Chart
            st.subheader("📉 Price Trend Chart")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Close", line=dict(color="white")))
            fig.add_trace(go.Scatter(x=df.index, y=df["EMA_20"], name="EMA 20", line=dict(color="orange")))
            fig.add_trace(go.Scatter(x=df.index, y=df["EMA_50"], name="EMA 50", line=dict(color="blue")))
            fig.add_trace(go.Scatter(x=df.index, y=df["EMA_200"], name="EMA 200", line=dict(color="lime")))
            fig.update_layout(title=f"{stock_choice} | Advanced AI Prediction", template="plotly_dark", height=500)
            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Prediction Error: {e}")


# News Sentiment page

elif page == "📰 News Sentiment":
    st.title("📰 News Sentiment")
    st.subheader("🔍 Latest Headlines")
    query = st.text_input("Search news for:", value=stock_choice)

    # --- Inline CSS ---
    st.markdown("""
        <style>
        .news-card {
            background-color: #1e1e1e;
            padding: 15px;
            margin-bottom: 20px;
            border-radius: 12px;
            box-shadow: 0 0 10px rgba(255, 255, 255, 0.05);
            transition: transform 0.2s ease;
            height: 100%;
        }
        .news-card:hover {
            transform: scale(1.02);
        }
        .news-img {
            width: 100%;
            height: 160px;
            object-fit: cover;
            border-radius: 10px;
            margin-bottom: 10px;
        }
        .news-title {
            color: #29b6f6 !important;
            font-size: 1rem;
            font-weight: 600;
            line-height: 1.4;
            text-decoration: none;
        }
        .news-title:hover {
            text-decoration: underline;
        }
        </style>
    """, unsafe_allow_html=True)

    def fetch_news_sentiment(query):
        try:
            from datetime import datetime, timedelta
            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d")
            url = f"https://www.bing.com/news/search?q={query}+stock&form=QBNH&sp=-1&ghc=1&filters=ex1%3a%22ez5_{start_date}_{end_date}%22"
            headers = {"User-Agent": "Mozilla/5.0"}
            response = requests.get(url, headers=headers)
            soup = BeautifulSoup(response.text, "html.parser")
            articles = soup.find_all("a", class_="title", href=True)

            news_data = []
            for a in articles[:6]:
                title = a.text.strip()
                link = a["href"]

                # Try to get image
                img_url = "https://i.imgur.com/EO8a4Tz.png"
                parent = a.find_parent("div")
                if parent:
                    img_tag = parent.find("img")
                    if img_tag and "src" in img_tag.attrs:
                        img_url = img_tag["src"]

                # Sentiment detection
                title_lower = title.lower()
                if any(x in title_lower for x in ["profit rises", "record high", "beats estimates", "surges", "jumps", "growth"]):
                    sentiment = "📈 Bullish"
                    tag_color = "green"
                elif any(x in title_lower for x in ["falls", "layoff", "misses", "dips", "plunge", "cut", "drops", "decline", "loss"]):
                    sentiment = "📉 Bearish"
                    tag_color = "red"
                else:
                    sentiment = "❕ Neutral"
                    tag_color = "gray"

                # Category Tag
                if "result" in title_lower:
                    category = "📊 Results"
                elif "alert" in title_lower:
                    category = "🚨 Alert"
                elif "price" in title_lower or "share" in title_lower:
                    category = "💹 Price"
                else:
                    category = "📰 General"

                news_data.append((title, link, img_url, sentiment, tag_color, category))
            return news_data
        except Exception as e:
            print("News fetch error:", e)
            return []

    news_items = fetch_news_sentiment(query)

    if news_items:
        cols = st.columns(3)
        for i, (title, link, img, sentiment, color, category) in enumerate(news_items):
            with cols[i % 3]:
                st.markdown(f"""
                    <div class="news-card">
                        <img src="{img}" class="news-img" alt="thumbnail">
                        <a href="{link}" target="_blank" class="news-title">{title}</a>
                        <div style="margin-top:8px;">
                            <span style="background-color:{color}; color:white; padding:4px 8px; border-radius:5px; font-size: 0.75rem;">{sentiment}</span>
                            <span style="background-color:#444; color:white; padding:4px 8px; border-radius:5px; font-size: 0.75rem; margin-left:5px;">{category}</span>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
    else:
        st.warning("No recent news found.")


# --- PAGE: Watchlist ---
elif page == "⭐ Watchlist":
    st.title("⭐ Your Watchlist")
    if "watchlist" not in st.session_state:
        st.session_state.watchlist = []
    if st.button("Add to Watchlist"):
        if stock_choice not in st.session_state.watchlist:
            st.session_state.watchlist.append(stock_choice)
    st.write("Watchlist:", st.session_state.watchlist)

# --- PAGE: Advanced Analysis ---
elif page == "📊 Advanced Analysis":
    st.title("📊 Advanced Market Analysis")

    try:
        end_time = dt.datetime.now()
        start_time = end_time - dt.timedelta(days=days)

        candle_data = obj.getCandleData({
            "exchange": "NSE",
            "symboltoken": token,
            "interval": interval_choice,
            "fromdate": start_time.strftime("%Y-%m-%d %H:%M"),
            "todate": end_time.strftime("%Y-%m-%d %H:%M")
        })

        df = pd.DataFrame(candle_data["data"], columns=["Datetime", "Open", "High", "Low", "Close", "Volume"])
        df["Datetime"] = pd.to_datetime(df["Datetime"])
        df.set_index("Datetime", inplace=True)

        # --- Advanced Indicators ---
        df["ATR"] = ta.volatility.AverageTrueRange(high=df["High"], low=df["Low"], close=df["Close"], window=14).average_true_range()
        df["EMA_20"] = df["Close"].ewm(span=20).mean()
        df["Bollinger High"] = df["Close"].rolling(window=20).mean() + 2 * df["Close"].rolling(window=20).std()
        df["Bollinger Low"] = df["Close"].rolling(window=20).mean() - 2 * df["Close"].rolling(window=20).std()

        # --- Swing Zones ---
        recent_high = df["High"].rolling(10).max().iloc[-1]
        recent_low = df["Low"].rolling(10).min().iloc[-1]

        # --- Price Gaps ---
        gap = df["Open"].iloc[-1] - df["Close"].iloc[-2]
        gap_msg = "⬆️ Up Gap" if gap > 0 else "⬇️ Down Gap" if gap < 0 else "No Gap"

        # --- Volatility Summary ---
        st.subheader("📈 Price Volatility & Ranges")
        st.metric("🔄 ATR (14)", round(df["ATR"].iloc[-1], 2))
        st.metric("🔍 Bollinger Band Width", round(df["Bollinger High"].iloc[-1] - df["Bollinger Low"].iloc[-1], 2))

        # --- Price Action Summary ---
        st.subheader("📍 Swing Analysis")
        st.write(f"🔺 Recent Resistance: ₹ {round(recent_high, 2)}")
        st.write(f"🔻 Recent Support: ₹ {round(recent_low, 2)}")
        st.write(f"📊 Gap Detected: {gap_msg} ({round(gap, 2)})")

        # --- Chart with Bands ---
        st.subheader("📊 Technical View (Bollinger Bands)")
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=df.index, open=df.Open, high=df.High, low=df.Low, close=df.Close, name="Price"))
        fig.add_trace(go.Scatter(x=df.index, y=df["EMA_20"], name="EMA 20", line=dict(color="orange")))
        fig.add_trace(go.Scatter(x=df.index, y=df["Bollinger High"], name="Upper Band", line=dict(color="green")))
        fig.add_trace(go.Scatter(x=df.index, y=df["Bollinger Low"], name="Lower Band", line=dict(color="red")))
        st.plotly_chart(fig, use_container_width=True)

        # --- Summary AI Signal ---
        st.subheader("🧠 AI Summary & Recommendation")
        if gap > 0 and df["Close"].iloc[-1] > recent_high:
            st.success("🚀 Strong Bullish Breakout Detected – Buy Signal")
        elif gap < 0 and df["Close"].iloc[-1] < recent_low:
            st.error("⚠️ Bearish Breakdown – Caution Advised")
        else:
            st.info("🔄 Sideways/Neutral Trend – Wait for confirmation")

             # --- 📊 Advanced Add-ons: LSTM Forecast + News Sentiment + Backtesting ---

        

        st.markdown("## 🔁 AI Enhancements: LSTM | Sentiment | Backtesting")

        # --- 1️⃣ LSTM Price Prediction ---
        try:
            st.subheader("🔮 LSTM Forecast (Next Day)")
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(df[["Close"]])
            look_back = 60

            X, y = [], []
            for i in range(look_back, len(scaled_data)):
                X.append(scaled_data[i - look_back:i, 0])
                y.append(scaled_data[i, 0])
            X = np.array(X)
            X = np.reshape(X, (X.shape[0], X.shape[1], 1))

            model = Sequential()
            model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)))
            model.add(LSTM(50))
            model.add(Dense(1))
            model.compile(loss='mean_squared_error', optimizer='adam')
            model.fit(X, y, epochs=10, batch_size=8, verbose=0)

            last_sequence = scaled_data[-look_back:]
            last_sequence = np.reshape(last_sequence, (1, look_back, 1))
            next_day_scaled = model.predict(last_sequence)
            next_day_price = scaler.inverse_transform(next_day_scaled)[0][0]

            st.metric("📈 Predicted Price (Tomorrow)", f"₹ {round(next_day_price, 2)}")
        except Exception as e:
            st.warning(f"LSTM prediction error: {e}")

        # --- 2️⃣ News Sentiment Analysis (Bing + FinBERT Simplified) ---
        try:
            st.subheader("📰 News Sentiment (Bing Headlines)")
            def fetch_headlines(stock):
                headers = {"User-Agent": "Mozilla/5.0"}
                url = f"https://www.bing.com/news/search?q={stock}+stock&FORM=HDRSC6"
                r = requests.get(url, headers=headers)
                soup = BeautifulSoup(r.text, "html.parser")
                links = soup.find_all("a", class_="title")
                return [link.text.strip() for link in links[:5]]

            headlines = fetch_headlines(stock_choice)
            positive, negative = 0, 0

            for title in headlines:
                title_lower = title.lower()
                if any(x in title_lower for x in ["surge", "beat", "gain", "profit", "up", "grow"]):
                    positive += 1
                elif any(x in title_lower for x in ["fall", "loss", "drop", "down", "miss", "cut"]):
                    negative += 1

            total = positive + negative if (positive + negative) > 0 else 1
            score = round((positive - negative) / total * 100, 2)

            sentiment = "📈 Bullish" if score > 20 else "📉 Bearish" if score < -20 else "⚖️ Neutral"
            st.metric("🧠 Sentiment Score", f"{score}%", delta=sentiment)

            for i, h in enumerate(headlines, 1):
                st.markdown(f"**{i}.** {h}")

        except Exception as e:
            st.warning(f"Sentiment scraping failed: {e}")

        # --- 3️⃣ Backtesting with backtesting.py ---
        try:
            st.subheader("📉 Strategy Backtesting")

            class MAStrategy(Strategy):
                def init(self):
                    self.ema20 = self.I(lambda x: x.ewm(span=20).mean(), self.data.Close)
                    self.ema50 = self.I(lambda x: x.ewm(span=50).mean(), self.data.Close)

                def next(self):
                    if self.ema20[-1] > self.ema50[-1] and self.ema20[-2] <= self.ema50[-2]:
                        self.buy()
                    elif self.ema20[-1] < self.ema50[-1] and self.ema20[-2] >= self.ema50[-2]:
                        self.sell()

            bt = Backtest(df, MAStrategy, cash=100000, commission=.002, exclusive_orders=True)
            stats = bt.run()
            st.write("📊 Backtest Summary", stats[['Return [%]', 'Win Rate [%]', '# Trades']])
            st.plotly_chart(bt.plot(open_browser=False), use_container_width=True)
        except Exception as e:
            st.warning(f"Backtesting error: {e}")


        # --- PROPHET FORECAST SECTION ---
        st.subheader("📆 30-Day Price Forecast (Prophet AI)")

        try:
            from prophet import Prophet

            # Prepare data for Prophet
            prophet_df = df.reset_index()[["Datetime", "Close"]].rename(columns={"Datetime": "ds", "Close": "y"}).dropna()
            prophet_df["ds"] = prophet_df["ds"].dt.tz_localize(None)  # 🚫 Remove timezone info


            if len(prophet_df) >= 60:  # Ensure enough data
                model = Prophet(daily_seasonality=True)
                model.fit(prophet_df)

                future = model.make_future_dataframe(periods=30)
                forecast = model.predict(future)

                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat"], name="Predicted Price", line=dict(color="orange")))
                fig2.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat_upper"], name="Upper Bound", line=dict(color="lightgreen"), opacity=0.3))
                fig2.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat_lower"], name="Lower Bound", line=dict(color="red"), opacity=0.3))
                fig2.add_trace(go.Scatter(x=prophet_df["ds"], y=prophet_df["y"], name="Actual", line=dict(color="white")))

                fig2.update_layout(
                    title=f"{stock_choice} – 30-Day Price Forecast (Prophet)",
                    template="plotly_dark",
                    height=500
                )
                st.plotly_chart(fig2, use_container_width=True)

                # Tomorrow forecast
                tomorrow_index = len(prophet_df) + 1
                if tomorrow_index < len(forecast):
                    tomorrow_pred = forecast.iloc[tomorrow_index]
                    st.metric("🔮 Tomorrow's Forecasted Price", f"₹ {round(tomorrow_pred['yhat'], 2)}")
                else:
                    st.warning("Insufficient forecast data to display tomorrow's price.")
            else:
                st.warning("📉 Not enough historical data to run 30-day Prophet forecast. Please use a longer time frame (e.g., 3M or 6M).")

        except Exception as e:
            st.error(f"Prophet Forecast Error: {e}")

    except Exception as e:
        st.error(f"Advanced Analysis Error: {e}")

