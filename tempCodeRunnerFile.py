# streamlit_ui_enhanced_voice.py
# Enhanced Streamlit UI with Voice Commands and News Features
# Run with: streamlit run streamlit_ui_enhanced_voice.py
# 
# New Features:
# - Voice command support (speech recognition)
# - Latest company news integration
# - News sentiment analysis dashboard
# - Voice feedback using text-to-speech
#
# Additional requirements:
# pip install SpeechRecognition pyttsx3 pyaudio

# ======= begin: early log/warning suppression (must be first) =======
import os, sys, warnings, logging
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")
NOISY = ["streamlit", "streamlit.runtime", "streamlit.runtime.scriptrunner", 
         "tornado", "urllib3", "matplotlib", "concurrent.futures"]
for name in NOISY:
    logging.getLogger(name).setLevel(logging.ERROR)
logging.getLogger().setLevel(logging.ERROR)
class SuppressScriptRunContextFilter(logging.Filter):
    def filter(self, record):
        try:
            msg = record.getMessage()
            if "missing ScriptRunContext" in msg:
                return False
        except Exception:
            pass
        return True
logging.getLogger().addFilter(SuppressScriptRunContextFilter())
# ======= end suppression block =======

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
import xgboost as xgb
from io import BytesIO
from datetime import datetime
import time as time_module

# Voice recognition imports
try:
    import speech_recognition as sr
    SPEECH_RECOGNITION_AVAILABLE = True
except ImportError:
    SPEECH_RECOGNITION_AVAILABLE = False
    st.warning("⚠️ SpeechRecognition not installed. Voice commands disabled. Install: pip install SpeechRecognition pyaudio")

try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False

# Import core bot functions
try:
    from stock_analysis_bot import (
        get_price_data,
        compute_indicators,
        make_features,
        train_model,
        predict_next_days,
        backtest_ml_strategy,
        save_model,
        load_model,
        get_stock_news,
        print_news_summary,
        get_company_name_from_ticker
    )
except Exception as e:
    st.error("Could not import functions from stock_analysis_bot.py. Make sure the enhanced version is in the same folder.")
    st.stop()

# Page config
st.set_page_config(page_title="Stockholm - AI Powered Stock Bot", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for better UI
st.markdown("""
<style>
    /* Hide Deploy button */
    .stDeployButton,
    .stAppDeployButton {
        visibility: hidden !important;
    }
    
    /* Hide Streamlit menu */
    #MainMenu {
        visibility: hidden;
    }
    
    /* Hide footer */
    footer {
        visibility: hidden;
    }
    
    .main-header {
        font-size: 6.5rem;
        font-weight: 900;
        letter-spacing: -2px;
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 50%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        margin-top: -0.5rem;
        font-family: 'Helvetica Neue', Arial, sans-serif;
        line-height: 1;
        text-transform: uppercase;
        text-shadow: 0 0 30px rgba(59, 130, 246, 0.3);
    }
    .sub-header {
        font-size: 1.3rem;
        color: #94a3b8;
        font-weight: 500;
        margin-top: 0rem;
        margin-bottom: 2.5rem;
        letter-spacing: 2px;
        text-transform: uppercase;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .voice-button {
        background-color: #FF4B4B;
        color: white;
        border-radius: 50%;
        width: 60px;
        height: 60px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">Stockholm</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-Powered Stock Analysis Platform</p>', unsafe_allow_html=True)

# --------------------------
# Voice Command Functions
# --------------------------

class VoiceAssistant:
    """Voice command handler for the stock bot."""
    
    def __init__(self):
        self.recognizer = sr.Recognizer() if SPEECH_RECOGNITION_AVAILABLE else None
        self.tts_engine = pyttsx3.init() if TTS_AVAILABLE else None
        if self.tts_engine:
            self.tts_engine.setProperty('rate', 150)
            self.tts_engine.setProperty('volume', 0.9)
    
    def speak(self, text):
        """Text to speech output."""
        if TTS_AVAILABLE and self.tts_engine:
            try:
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()
            except Exception as e:
                st.warning(f"TTS error: {e}")
    
    def listen(self, timeout=5):
        """Listen for voice command and return recognized text."""
        if not SPEECH_RECOGNITION_AVAILABLE or not self.recognizer:
            return None
        
        try:
            with sr.Microphone() as source:
                st.info("🎤 Listening... Speak now!")
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=5)
                
            text = self.recognizer.recognize_google(audio)
            return text.lower()
        except sr.WaitTimeoutError:
            st.warning("⏱️ No speech detected. Please try again.")
            return None
        except sr.UnknownValueError:
            st.warning("🤔 Could not understand audio. Please try again.")
            return None
        except Exception as e:
            st.error(f"Voice recognition error: {e}")
            return None
    
    def parse_command(self, text):
        """Parse voice command and extract action and ticker."""
        if not text:
            return None, None
        
        text = text.lower().strip()
        
        # Command patterns
        commands = {
            'fetch': ['fetch', 'get', 'load', 'show', 'data for', 'price for'],
            'news': ['news', 'latest news', 'company news', 'updates for', 'headlines'],
            'predict': ['predict', 'forecast', 'prediction for', 'future'],
            'train': ['train', 'train model', 'build model'],
            'backtest': ['backtest', 'test strategy', 'performance'],
            'indicators': ['indicators', 'technical indicators', 'technicals'],
        }
        
        # Detect command
        detected_command = None
        for cmd, patterns in commands.items():
            if any(pattern in text for pattern in patterns):
                detected_command = cmd
                break
        
        # Extract ticker (common Indian stock names)
        ticker_map = {
            'reliance': 'RELIANCE.NS',
            'tcs': 'TCS.NS',
            'infosys': 'INFY.NS',
            'hdfc': 'HDFCBANK.NS',
            'icici': 'ICICIBANK.NS',
            'wipro': 'WIPRO.NS',
            'bharti': 'BHARTIARTL.NS',
            'itc': 'ITC.NS',
            'hind copper': 'HINDCOPPER.NS',
            'hindalco': 'HINDALCO.NS',
            'tata steel': 'TATASTEEL.NS',
            'maruti': 'MARUTI.NS',
            'asian paints': 'ASIANPAINT.NS',
            'bajaj': 'BAJFINANCE.NS',
            'kotak': 'KOTAKBANK.NS',
        }
        
        detected_ticker = None
        for name, ticker in ticker_map.items():
            if name in text:
                detected_ticker = ticker
                break
        
        # If no predefined ticker, try to extract from text
        if not detected_ticker:
            words = text.split()
            for word in words:
                if word.isupper() or (len(word) > 2 and word.replace('.', '').replace('-', '').isalnum()):
                    detected_ticker = word.upper()
                    if not detected_ticker.endswith('.NS') and not detected_ticker.endswith('.BO'):
                        detected_ticker = detected_ticker + '.NS'
                    break
        
        return detected_command, detected_ticker


# Initialize voice assistant
voice_assistant = VoiceAssistant()

# --------------------------
# Session initialization
# --------------------------
if 'data_cache' not in st.session_state:
    st.session_state['data_cache'] = {}
if 'ind_cache' not in st.session_state:
    st.session_state['ind_cache'] = {}
if 'models' not in st.session_state:
    st.session_state['models'] = {}
if 'preds' not in st.session_state:
    st.session_state['preds'] = {}
if 'bt' not in st.session_state:
    st.session_state['bt'] = {}
if 'portfolio' not in st.session_state:
    st.session_state['portfolio'] = []
if 'news_cache' not in st.session_state:
    st.session_state['news_cache'] = {}
if 'favorites' not in st.session_state:
    st.session_state['favorites'] = ["RELIANCE.NS", "HINDCOPPER.NS", "TCS.NS", "INFY.NS"]
if 'voice_command_result' not in st.session_state:
    st.session_state['voice_command_result'] = None
if 'current_ticker' not in st.session_state:
    st.session_state['current_ticker'] = None

# --------------------------
# Utility helpers
# --------------------------
def cache_key(ticker, period, interval):
    return f"{ticker}__{period}__{interval}"

def fetch_and_cache(ticker, period='5y', interval='1d'):
    key = cache_key(ticker, period, interval)
    if key in st.session_state['data_cache']:
        return st.session_state['data_cache'][key]
    df = get_price_data(ticker, period=period, interval=interval)
    st.session_state['data_cache'][key] = df
    return df

def compute_and_cache_indicators(ticker, period='5y', interval='1d'):
    key = cache_key(ticker, period, interval) + "__ind"
    if key in st.session_state['ind_cache']:
        return st.session_state['ind_cache'][key]
    df = fetch_and_cache(ticker, period, interval)
    ind = compute_indicators(df)
    st.session_state['ind_cache'][key] = ind
    return ind

def get_and_cache_news(ticker, days=7):
    key = f"{ticker}__news__{days}"
    if key in st.session_state['news_cache']:
        cache_time, news_df = st.session_state['news_cache'][key]
        if (datetime.now() - cache_time).seconds < 3600:  # Cache for 1 hour
            return news_df
    
    # Try to get news, with better error handling
    try:
        news_df = get_stock_news(ticker, days=days)
        if news_df.empty:
            # If no news from API, create a sample message
            st.info(f"⚠️ No recent news found for {ticker} from Yahoo Finance. This could mean:")
            st.write("• The ticker might not have recent news coverage")
            st.write("• Yahoo Finance API might be temporarily unavailable")
            st.write("• Try a more popular ticker like RELIANCE.NS or TCS.NS")
    except Exception as e:
        st.error(f"Error fetching news: {e}")
        news_df = pd.DataFrame()
    
    st.session_state['news_cache'][key] = (datetime.now(), news_df)
    return news_df

def show_metric_cards(metrics: dict):
    if not metrics:
        return
    cols = st.columns(len(metrics))
    for (label, value), col in zip(metrics.items(), cols):
        display = value if isinstance(value, str) else (f"{value:.3f}" if (value is not None and not np.isnan(value)) else "N/A")
        col.metric(label, display)

def plot_candlestick(df: pd.DataFrame, ticker: str, show_sma=True):
    fig = go.Figure(data=[go.Candlestick(
        x=df.index,
        open=df.get('Open', df['Close']), high=df.get('High', df['Close']),
        low=df.get('Low', df['Close']), close=df['Close'], name=ticker)])
    if show_sma and 'SMA20' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA20'], mode='lines', name='SMA20', line=dict(color='orange')))
    if 'EMA20' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA20'], mode='lines', name='EMA20', line=dict(color='cyan')))
    fig.update_layout(xaxis_rangeslider_visible=False, height=540, template='plotly_dark')
    return fig

def plot_news_sentiment(news_df: pd.DataFrame):
    """Create sentiment distribution chart."""
    if news_df.empty or 'sentiment' not in news_df.columns:
        return None
    
    sentiment_counts = news_df['sentiment'].value_counts()
    colors = {'positive': '#00CC96', 'neutral': '#FFA15A', 'negative': '#EF553B'}
    
    fig = go.Figure(data=[
        go.Bar(
            x=sentiment_counts.index,
            y=sentiment_counts.values,
            marker_color=[colors.get(s, 'gray') for s in sentiment_counts.index],
            text=sentiment_counts.values,
            textposition='auto',
        )
    ])
    fig.update_layout(
        title="News Sentiment Distribution",
        xaxis_title="Sentiment",
        yaxis_title="Number of Articles",
        template='plotly_dark',
        height=300
    )
    return fig

def send_telegram(token: str, chat_id: str, message: str) -> bool:
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        r = requests.post(url, json={"chat_id": chat_id, "text": message})
        return r.status_code == 200
    except Exception:
        return False

# XGBoost training wrapper
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, roc_auc_score
def train_xgb_from_indicators(ind_df: pd.DataFrame, n_splits=5, seed=42):
    X, y = make_features(ind_df)
    tscv = TimeSeriesSplit(n_splits=n_splits)
    oof = np.zeros(len(y))
    metrics = []
    model = xgb.XGBClassifier(n_estimators=200, use_label_encoder=False, eval_metric='logloss', random_state=seed, verbosity=0)
    for fold, (tr, te) in enumerate(tscv.split(X), start=1):
        Xtr, Xte = X.iloc[tr], X.iloc[te]
        ytr, yte = y.iloc[tr], y.iloc[te]
        model.fit(Xtr, ytr)
        proba = model.predict_proba(Xte)[:,1]
        oof[te] = proba
        preds = (proba >= 0.5).astype(int)
        acc = accuracy_score(yte, preds)
        try: auc = roc_auc_score(yte, proba)
        except Exception: auc = np.nan
        metrics.append({'fold':fold, 'accuracy':acc, 'auc':auc})
    model.fit(X, y)
    report = {'cv_metrics': metrics, 'oof_accuracy': accuracy_score(y, (oof>=0.5).astype(int))}
    try:
        report['oof_auc'] = float(roc_auc_score(y, oof)) if len(np.unique(y))>1 else np.nan
    except Exception:
        report['oof_auc'] = np.nan
    return model, report

# --------------------------
# Voice Command Interface (NEW)
# --------------------------
st.sidebar.markdown("---")
st.sidebar.header("🎤 Voice Commands")

if SPEECH_RECOGNITION_AVAILABLE:
    st.sidebar.write("**Supported commands:**")
    st.sidebar.write("• 'Fetch data for [stock]'")
    st.sidebar.write("• 'Show news for [stock]'")
    st.sidebar.write("• 'Predict [stock]'")
    st.sidebar.write("• 'Train model for [stock]'")
    
    if st.sidebar.button("🎤 Start Voice Command", key="voice_btn", help="Click and speak"):
        with st.spinner("🎤 Listening..."):
            command_text = voice_assistant.listen(timeout=5)
            
        if command_text:
            st.sidebar.success(f"Heard: '{command_text}'")
            command, ticker = voice_assistant.parse_command(command_text)
            
            if command and ticker:
                st.session_state['voice_command_result'] = (command, ticker)
                st.session_state['current_ticker'] = ticker
                voice_assistant.speak(f"Executing {command} for {ticker.replace('.NS', '').replace('.BO', '')}")
                st.sidebar.success(f"✅ Command: {command} | Ticker: {ticker}")
                st.rerun()
            else:
                st.sidebar.error("❌ Could not understand command or ticker")
                voice_assistant.speak("Sorry, I could not understand the command")
else:
    st.sidebar.warning("Voice commands unavailable. Install: pip install SpeechRecognition pyaudio")

# Execute voice command if present
if st.session_state['voice_command_result']:
    command, ticker = st.session_state['voice_command_result']
    st.session_state['voice_command_result'] = None  # Clear after processing
    
    if command == 'fetch':
        try:
            df = fetch_and_cache(ticker)
            ind = compute_and_cache_indicators(ticker)
            st.success(f"✅ Fetched {len(df)} rows for {ticker}")
            voice_assistant.speak(f"Data loaded for {ticker.replace('.NS', '')}")
        except Exception as e:
            st.error(f"Error fetching {ticker}: {e}")
    
    elif command == 'news':
        try:
            news_df = get_and_cache_news(ticker, days=7)
            st.session_state['current_news'] = news_df
            st.success(f"✅ Fetched {len(news_df)} news articles for {ticker}")
            voice_assistant.speak(f"Found {len(news_df)} news articles")
        except Exception as e:
            st.error(f"Error fetching news: {e}")
    
    elif command == 'predict':
        try:
            ind = compute_and_cache_indicators(ticker)
            mkey = (ticker, 'rf')
            if mkey not in st.session_state['models']:
                model, _ = train_model(ind, n_splits=5)
                st.session_state['models'][mkey] = model
            else:
                model = st.session_state['models'][mkey]
            
            preds = predict_next_days(model, ind, days=7)
            st.session_state['preds'][ticker] = preds
            st.success(f"✅ Predictions generated for {ticker}")
            voice_assistant.speak("Predictions ready")
        except Exception as e:
            st.error(f"Error in prediction: {e}")

# --------------------------
# Sidebar controls & presets
# --------------------------
with st.sidebar:
    st.header("📊 Controls")
    default_ticker = st.session_state.get('current_ticker', "HINDCOPPER.NS")
    tickers_input = st.text_input("Tickers (space-separated)", value=default_ticker if default_ticker else "HINDCOPPER.NS")
    tickers = [t.strip().upper() for t in tickers_input.split()] if tickers_input and tickers_input.strip() else []
    period = st.selectbox("Period", ['1y','3y','5y','10y','max'], index=2)
    interval = st.selectbox("Interval", ['1d','1wk'], index=0)
    predict_days = st.number_input("Predict days", min_value=1, max_value=30, value=7)
    cv_splits = st.slider("CV splits", min_value=2, max_value=10, value=5)
    threshold = st.slider("Backtest threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
    
    st.markdown("---")
    st.subheader("⭐ Favorites")
    for fav in st.session_state['favorites']:
        if st.button(f"📌 {fav}", key=f"fav_{fav}"):
            st.session_state['current_ticker'] = fav
            tickers = [fav]
            st.rerun()

    newfav = st.text_input("Add favorite", "")
    if st.button("➕ Add"):
        if newfav.strip() and newfav.upper() not in st.session_state['favorites']:
            st.session_state['favorites'].append(newfav.strip().upper())
            st.success(f"Added {newfav.upper()}")

    st.markdown("---")
    st.subheader("📲 Telegram Alerts")
    tg_enable = st.checkbox("Enable alerts", value=False)
    tg_token = st.text_input("Bot token", type="password")
    tg_chat = st.text_input("Chat ID")
    tg_threshold = st.slider("Alert threshold", 0.0, 1.0, 0.75, 0.01)

    st.markdown("---")
    st.subheader("🤖 Model Options")
    use_xgb = st.checkbox("Train XGBoost", value=True)
    model_choice = st.radio("Active model", options=["RandomForest", "XGBoost"], index=0)

# --------------------------
# Main Layout: Tabs
# --------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Data & Charts", "📰 News & Sentiment", "🤖 ML & Predictions", "💼 Portfolio", "ℹ️ Help"])

# TAB 1: Data & Charts
with tab1:
    st.subheader("📊 Stock Data & Technical Analysis")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🔄 Fetch Data", use_container_width=True):
            if tickers:
                for t in tickers:
                    try:
                        df = fetch_and_cache(t, period, interval)
                        st.success(f"✅ Fetched {len(df)} rows for {t}")
                    except Exception as e:
                        st.error(f"❌ {t}: {e}")
    
    with col2:
        if st.button("📊 Compute Indicators", use_container_width=True):
            if tickers:
                for t in tickers:
                    try:
                        ind = compute_and_cache_indicators(t, period, interval)
                        st.success(f"✅ Indicators for {t}")
                    except Exception as e:
                        st.error(f"❌ {t}: {e}")
    
    with col3:
        show_table = st.checkbox("Show data table", value=False)
    
    if tickers:
        t = tickers[0]
        try:
            ind = compute_and_cache_indicators(t, period, interval)
            
            # Metrics row
            last_row = ind.iloc[-1]
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Close", f"₹{last_row['Close']:.2f}", 
                       f"{last_row.get('Return_1d', 0)*100:.2f}%")
            col2.metric("RSI", f"{last_row.get('RSI14', 0):.1f}")
            col3.metric("MACD", f"{last_row.get('MACD', 0):.2f}")
            col4.metric("Volume", f"{last_row.get('Volume', 0):,.0f}" if 'Volume' in last_row else "N/A")
            col5.metric("ATR", f"{last_row.get('ATR14', 0):.2f}")
            
            # Candlestick chart
            st.plotly_chart(plot_candlestick(ind, t), use_container_width=True)
            
            # Optional data table
            if show_table:
                st.dataframe(ind.tail(20), use_container_width=True)
                
        except Exception as e:
            st.info("👆 Click 'Fetch Data' and 'Compute Indicators' to see charts")

# TAB 2: News & Sentiment
with tab2:
    st.subheader("📰 Latest News & Sentiment Analysis")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        news_days = st.slider("News from last N days", 1, 30, 7)
    with col2:
        if st.button("🔄 Refresh News", use_container_width=True):
            if tickers:
                st.session_state['news_cache'] = {}  # Clear cache
                st.rerun()
    
    if tickers:
        t = tickers[0]
        try:
            news_df = get_and_cache_news(t, days=news_days)
            
            if not news_df.empty:
                # Sentiment summary
                col1, col2, col3, col4 = st.columns(4)
                total_news = len(news_df)
                sentiment_counts = news_df['sentiment'].value_counts()
                
                col1.metric("📰 Total Articles", total_news)
                col2.metric("📈 Positive", sentiment_counts.get('positive', 0), 
                           f"{sentiment_counts.get('positive', 0)/total_news*100:.0f}%")
                col3.metric("➡️ Neutral", sentiment_counts.get('neutral', 0),
                           f"{sentiment_counts.get('neutral', 0)/total_news*100:.0f}%")
                col4.metric("📉 Negative", sentiment_counts.get('negative', 0),
                           f"{sentiment_counts.get('negative', 0)/total_news*100:.0f}%")
                
                # Sentiment chart
                fig = plot_news_sentiment(news_df)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                
                # News list
                st.markdown("### Latest Headlines")
                for idx, row in news_df.head(15).iterrows():
                    sentiment_emoji = {'positive': '📈', 'negative': '📉', 'neutral': '➡️'}.get(row['sentiment'], '➡️')
                    
                    with st.expander(f"{sentiment_emoji} [{row['source']}] {row['title']}", expanded=False):
                        if row['description']:
                            st.write(row['description'])
                        st.write(f"**Published:** {row['published']}")
                        st.write(f"**Sentiment:** {row['sentiment'].capitalize()}")
                        if row['url']:
                            st.markdown(f"[Read full article]({row['url']})")
            else:
                st.info(f"No news found for {t}. Try a different ticker or time range.")
                
        except Exception as e:
            st.error(f"Error loading news: {e}")
    else:
        st.info("Enter a ticker to fetch news")

# TAB 3: ML & Predictions
with tab3:
    st.subheader("🤖 Machine Learning & Predictions")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🎯 Train Models", use_container_width=True):
            if tickers:
                for t in tickers:
                    try:
                        ind = compute_and_cache_indicators(t, period, '1d')
                        
                        # RandomForest
                        with st.spinner(f"Training RF for {t}..."):
                            rf_model, rf_report = train_model(ind, n_splits=cv_splits)
                            st.session_state['models'][(t, 'rf')] = rf_model
                            st.session_state['models'][(t, 'rf_report')] = rf_report
                        
                        # XGBoost
                        if use_xgb:
                            with st.spinner(f"Training XGB for {t}..."):
                                xgb_model, xgb_report = train_xgb_from_indicators(ind, n_splits=cv_splits)
                                st.session_state['models'][(t, 'xgb')] = xgb_model
                                st.session_state['models'][(t, 'xgb_report')] = xgb_report
                        
                        st.success(f"✅ Models trained for {t}")
                        
                        # Show metrics
                        st.write(f"**{t} Model Performance:**")
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.write("**RandomForest:**")
                            st.write(f"OOF Accuracy: {rf_report['oof_accuracy']:.3f}")
                            st.write(f"OOF AUC: {rf_report.get('oof_auc', 0):.3f}")
                        if use_xgb:
                            with col_b:
                                st.write("**XGBoost:**")
                                st.write(f"OOF Accuracy: {xgb_report['oof_accuracy']:.3f}")
                                st.write(f"OOF AUC: {xgb_report.get('oof_auc', 0):.3f}")
                        
                    except Exception as e:
                        st.error(f"❌ Training failed for {t}: {e}")
    
    with col2:
        if st.button("🔮 Predict Next Days", use_container_width=True):
            if tickers:
                for t in tickers:
                    try:
                        ind = compute_and_cache_indicators(t, period, '1d')
                        mkey = (t, 'xgb') if (model_choice == "XGBoost" and (t, 'xgb') in st.session_state['models']) else (t, 'rf')
                        
                        if mkey not in st.session_state['models']:
                            st.warning(f"Training {model_choice} for {t}...")
                            if model_choice == "XGBoost":
                                model, _ = train_xgb_from_indicators(ind)
                            else:
                                model, _ = train_model(ind)
                            st.session_state['models'][mkey] = model
                        else:
                            model = st.session_state['models'][mkey]
                        
                        preds = predict_next_days(model, ind, days=predict_days)
                        st.session_state['preds'][t] = preds
                        
                        st.success(f"✅ Predictions for {t}")
                        st.dataframe(preds, use_container_width=True)
                        
                        # Telegram alert
                        if tg_enable and tg_token and tg_chat:
                            top_p = preds['p_up'].iloc[0]
                            if top_p >= tg_threshold:
                                msg = f"🚨 ALERT {t}: Prediction={top_p:.2%} for {preds['date'].iloc[0]}"
                                if send_telegram(tg_token, tg_chat, msg):
                                    st.success("📲 Telegram alert sent!")
                        
                    except Exception as e:
                        st.error(f"❌ Prediction failed for {t}: {e}")
    
    with col3:
        if st.button("📊 Backtest Strategy", use_container_width=True):
            if tickers:
                for t in tickers:
                    try:
                        ind = compute_and_cache_indicators(t, period, '1d')
                        mkey = (t, 'xgb') if (model_choice == "XGBoost" and (t, 'xgb') in st.session_state['models']) else (t, 'rf')
                        
                        if mkey not in st.session_state['models']:
                            st.warning(f"Training model for backtest...")
                            model, _ = train_model(ind) if model_choice == "RandomForest" else train_xgb_from_indicators(ind)
                            st.session_state['models'][mkey] = model
                        else:
                            model = st.session_state['models'][mkey]
                        
                        bt_df, metrics = backtest_ml_strategy(model, ind, threshold=threshold)
                        st.session_state['bt'][t] = (bt_df, metrics)
                        
                        st.success(f"✅ Backtest for {t}")
                        
                        # Metrics
                        mlm = metrics.get('ML', {})
                        bhm = metrics.get('BuyHold', {})
                        
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.write("**ML Strategy:**")
                            st.metric("CAGR", f"{mlm.get('CAGR', 0):.2%}")
                            st.metric("Sharpe", f"{mlm.get('Sharpe', 0):.2f}")
                            st.metric("Max DD", f"{mlm.get('MaxDrawdown', 0):.2%}")
                        
                        with col_b:
                            st.write("**Buy & Hold:**")
                            st.metric("CAGR", f"{bhm.get('CAGR', 0):.2%}")
                            st.metric("Sharpe", f"{bhm.get('Sharpe', 0):.2f}")
                            st.metric("Max DD", f"{bhm.get('MaxDrawdown', 0):.2%}")
                        
                        # Equity curve
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=bt_df.index, y=bt_df['BuyHoldCurve'], 
                                               mode='lines', name='Buy & Hold', line=dict(color='blue')))
                        fig.add_trace(go.Scatter(x=bt_df.index, y=bt_df['MLCurve'], 
                                               mode='lines', name='ML Strategy', line=dict(color='green')))
                        fig.update_layout(title=f"{t} Strategy Comparison", 
                                        xaxis_title="Date", yaxis_title="Equity Curve",
                                        template='plotly_dark', height=400)
                        st.plotly_chart(fig, use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"❌ Backtest failed for {t}: {e}")
    
    # Show existing predictions if available
    if tickers and st.session_state['preds']:
        st.markdown("---")
        st.markdown("### 📋 Current Predictions")
        for t in tickers:
            if t in st.session_state['preds']:
                st.write(f"**{t}:**")
                st.dataframe(st.session_state['preds'][t], use_container_width=True)

# TAB 4: Portfolio
with tab4:
    st.subheader("💼 Portfolio & Paper Trading")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("#### Add Trade")
        with st.form("add_trade_form", clear_on_submit=True):
            col_a, col_b, col_c = st.columns(3)
            trade_ticker = col_a.text_input("Ticker", value=tickers[0] if tickers else "")
            trade_action = col_b.selectbox("Action", ["BUY", "SELL"])
            trade_qty = col_c.number_input("Quantity", min_value=1, value=1)
            
            col_d, col_e = st.columns(2)
            trade_price = col_d.number_input("Price (0 = use current)", value=0.0, min_value=0.0)
            trade_note = col_e.text_input("Note (optional)")
            
            submitted = st.form_submit_button("➕ Add Trade", use_container_width=True)
            
            if submitted and trade_ticker:
                use_price = trade_price
                if use_price == 0.0:
                    try:
                        df = fetch_and_cache(trade_ticker.upper())
                        use_price = float(df['Close'].iloc[-1])
                    except Exception:
                        st.error("Could not fetch current price. Enter manually.")
                        use_price = None
                
                if use_price:
                    st.session_state['portfolio'].append({
                        "datetime": datetime.now().isoformat(),
                        "ticker": trade_ticker.upper(),
                        "action": trade_action,
                        "qty": int(trade_qty),
                        "price": float(use_price),
                        "note": trade_note
                    })
                    st.success(f"✅ Trade added: {trade_action} {trade_qty} {trade_ticker} @ ₹{use_price:.2f}")
                    st.rerun()
    
    with col2:
        st.markdown("#### Portfolio Actions")
        if st.button("🗑️ Clear Portfolio", use_container_width=True):
            st.session_state['portfolio'] = []
            st.success("Portfolio cleared")
            st.rerun()
        
        if st.session_state['portfolio']:
            # Export trades
            trades_df = pd.DataFrame(st.session_state['portfolio'])
            csv = trades_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Trades CSV",
                data=csv,
                file_name=f"portfolio_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )
    
    # Display portfolio
    if st.session_state['portfolio']:
        st.markdown("---")
        st.markdown("### 📊 Trade History")
        trades_df = pd.DataFrame(st.session_state['portfolio'])
        st.dataframe(trades_df, use_container_width=True)
        
        # Holdings summary
        st.markdown("### 💰 Holdings Summary")
        holdings = {}
        for _, trade in trades_df.iterrows():
            ticker = trade['ticker']
            qty_signed = trade['qty'] if trade['action'] == 'BUY' else -trade['qty']
            cost = qty_signed * trade['price']
            
            if ticker not in holdings:
                holdings[ticker] = {'qty': 0, 'cost': 0.0, 'trades': 0}
            holdings[ticker]['qty'] += qty_signed
            holdings[ticker]['cost'] += cost
            holdings[ticker]['trades'] += 1
        
        # Calculate current value and P&L
        holdings_list = []
        for ticker, data in holdings.items():
            try:
                df = fetch_and_cache(ticker)
                current_price = float(df['Close'].iloc[-1])
                market_value = current_price * data['qty']
                pnl = market_value + data['cost']  # cost is negative for buys
                pnl_pct = (pnl / abs(data['cost'])) * 100 if data['cost'] != 0 else 0
                
                holdings_list.append({
                    'Ticker': ticker,
                    'Quantity': data['qty'],
                    'Avg Cost': f"₹{abs(data['cost']/data['qty']):.2f}" if data['qty'] != 0 else "N/A",
                    'Current Price': f"₹{current_price:.2f}",
                    'Market Value': f"₹{market_value:.2f}",
                    'P&L': f"₹{pnl:.2f}",
                    'P&L %': f"{pnl_pct:.2f}%",
                    'Trades': data['trades']
                })
            except Exception:
                holdings_list.append({
                    'Ticker': ticker,
                    'Quantity': data['qty'],
                    'Avg Cost': f"₹{abs(data['cost']/data['qty']):.2f}" if data['qty'] != 0 else "N/A",
                    'Current Price': "Error",
                    'Market Value': "Error",
                    'P&L': "Error",
                    'P&L %': "Error",
                    'Trades': data['trades']
                })
        
        if holdings_list:
            holdings_df = pd.DataFrame(holdings_list)
            st.dataframe(holdings_df, use_container_width=True)
    else:
        st.info("📝 No trades yet. Add your first trade above!")

# TAB 5: Help
with tab5:
    st.subheader("ℹ️ Help & Documentation")
    
    st.markdown("""
    ## 🎤 Voice Commands Guide
    
    ### Setup Required:
    ```bash
    pip install SpeechRecognition pyaudio pyttsx3
    ```
    
    ### Supported Voice Commands:
    - **"Fetch data for [stock name]"** - Downloads price data
    - **"Show news for [stock name]"** - Displays latest news
    - **"Predict [stock name]"** - Generates predictions
    - **"Train model for [stock name]"** - Trains ML model
    - **"Get indicators for [stock name]"** - Shows technical indicators
    
    ### Supported Stock Names:
    - Reliance, TCS, Infosys, HDFC, ICICI, Wipro
    - Bharti, ITC, Hind Copper, Hindalco, Tata Steel
    - Maruti, Asian Paints, Bajaj, Kotak
    - Or say the full ticker code (e.g., "RELIANCE.NS")
    
    ### Example Commands:
    - "Fetch data for Reliance"
    - "Show news for TCS"
    - "Predict Infosys for next 7 days"
    
    ---
    
    ## 📰 News Features
    
    ### News Sources:
    1. **Yahoo Finance** - Real-time stock news
    2. **MoneyControl** - Indian market specific news
    3. **NewsAPI** (optional) - Global news coverage
    
    ### Setup NewsAPI (Optional):
    1. Get free API key from: https://newsapi.org/register
    2. Set environment variable:
       ```bash
       export NEWSAPI_KEY=your_key_here
       ```
    
    ### Sentiment Analysis:
    - 📈 **Positive** - Bullish indicators, good news
    - ➡️ **Neutral** - Factual reporting, no clear sentiment
    - 📉 **Negative** - Bearish indicators, concerning news
    
    ---
    
    ## 🤖 ML Models
    
    ### Available Models:
    1. **Random Forest** - Ensemble model, robust
    2. **XGBoost** - Gradient boosting, high accuracy
    
    ### Features Used:
    - Lagged returns (1, 2, 3, 5, 10 days)
    - Technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands, ATR)
    - Volume changes
    
    ### Backtesting:
    - Walk-forward cross-validation
    - Transaction costs included (5 bps default)
    - Comparison with Buy & Hold strategy
    
    ---
    
    ## 💼 Portfolio Management
    
    ### Features:
    - Paper trading simulation
    - Real-time P&L calculation
    - Trade history export
    - Holdings summary with current prices
    
    ---
    
    ## 📲 Telegram Alerts
    
    ### Setup:
    1. Create a Telegram bot via @BotFather
    2. Get your bot token
    3. Get your chat ID (use @userinfobot)
    4. Enter credentials in sidebar
    
    ### Alert Triggers:
    - Prediction probability exceeds threshold
    - Customizable threshold (default 75%)
    
    ---
    
    ## 🛠️ Installation
    
    ```bash
    # Core dependencies
    pip install yfinance pandas numpy matplotlib seaborn scikit-learn ta joblib
    pip install streamlit plotly xgboost
    
    # News features
    pip install newsapi-python feedparser beautifulsoup4 requests textblob
    
    # Voice features
    pip install SpeechRecognition pyaudio pyttsx3
    
    # Telegram
    pip install requests
    ```
    
    ---
    
    ## 📞 Support
    
    For issues or questions:
    - Check that all dependencies are installed
    - Ensure stock_analysis_bot.py is in the same directory
    - For voice commands, check microphone permissions
    - For NewsAPI, verify your API key is valid
    
    ---
    
    ## ⚠️ Disclaimer
    
    This tool is for educational and research purposes only.
    - Not financial advice
    - Past performance doesn't guarantee future results
    - Always do your own research
    - Consult a financial advisor before trading
    """)
    
    st.markdown("---")
    st.info("💡 **Pro Tip**: Use voice commands for hands-free analysis while multitasking!")

# Footer
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("🇮🇳 **Made for Indian Stock Market**")
with col2:
    st.markdown("📊 **Real-time Data & News**")
with col3:
    st.markdown("🎤 **Voice-Powered Analysis**")

st.markdown("</div>", unsafe_allow_html=True)