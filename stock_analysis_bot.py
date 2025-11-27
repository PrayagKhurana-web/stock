"""
Stock Analysis Bot (Python) - Enhanced with News
---------------------------------

New Features:
- Company news fetching from multiple sources
- Latest stock updates and announcements
- News sentiment analysis
- Voice command support (in streamlit UI)

Requirements (add to existing):
pip install -U newsapi-python feedparser beautifulsoup4 requests textblob

For NewsAPI, get free key from: https://newsapi.org/register
Set env var: NEWSAPI_KEY=your_key_here
"""
from __future__ import annotations
import os
import sys
import argparse
import warnings
warnings.filterwarnings("ignore")

from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import TimeSeriesSplit
import joblib

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")

# News libraries
try:
    from newsapi import NewsApiClient
    NEWSAPI_AVAILABLE = True
except Exception:
    NEWSAPI_AVAILABLE = False

try:
    import feedparser
    import requests
    from bs4 import BeautifulSoup
    NEWS_SCRAPE_AVAILABLE = True
except Exception:
    NEWS_SCRAPE_AVAILABLE = False

try:
    from textblob import TextBlob
    SENTIMENT_AVAILABLE = True
except Exception:
    SENTIMENT_AVAILABLE = False

try:
    from ta.trend import EMAIndicator, SMAIndicator, MACD
    from ta.momentum import RSIIndicator
    from ta.volatility import BollingerBands, AverageTrueRange
    TA_AVAILABLE = True
except Exception:
    TA_AVAILABLE = False

# -----------------------------
# News Fetching Layer (NEW)
# -----------------------------

def get_company_name_from_ticker(ticker: str) -> str:
    """Extract company name from ticker for better news search."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return info.get('longName', ticker.replace('.NS', '').replace('.BO', ''))
    except Exception:
        return ticker.replace('.NS', '').replace('.BO', '')


def fetch_news_newsapi(query: str, days: int = 7) -> List[Dict]:
    """Fetch news using NewsAPI (requires API key)."""
    if not NEWSAPI_AVAILABLE:
        return []
    
    api_key = os.getenv('NEWSAPI_KEY')
    if not api_key:
        print("NewsAPI key not found. Set NEWSAPI_KEY env variable.")
        return []
    
    try:
        newsapi = NewsApiClient(api_key=api_key)
        from_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        
        articles = newsapi.get_everything(
            q=query,
            from_param=from_date,
            language='en',
            sort_by='publishedAt',
            page_size=20
        )
        
        news_list = []
        for article in articles.get('articles', []):
            news_list.append({
                'title': article.get('title', ''),
                'description': article.get('description', ''),
                'source': article.get('source', {}).get('name', 'Unknown'),
                'url': article.get('url', ''),
                'published': article.get('publishedAt', ''),
                'sentiment': get_sentiment(article.get('title', '') + ' ' + article.get('description', ''))
            })
        return news_list
    except Exception as e:
        print(f"NewsAPI error: {e}")
        return []


def fetch_news_moneycontrol(ticker: str) -> List[Dict]:
    """Scrape news from Moneycontrol (Indian stock news)."""
    if not NEWS_SCRAPE_AVAILABLE:
        return []
    
    try:
        company = get_company_name_from_ticker(ticker)
        search_url = f"https://www.moneycontrol.com/news/tags/{company.lower().replace(' ', '-')}.html"
        
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(search_url, headers=headers, timeout=10)
        
        if response.status_code != 200:
            return []
        
        soup = BeautifulSoup(response.content, 'html.parser')
        articles = soup.find_all('li', class_='clearfix')
        
        news_list = []
        for article in articles[:10]:
            try:
                title_tag = article.find('h2')
                link_tag = title_tag.find('a') if title_tag else None
                desc_tag = article.find('p')
                
                if link_tag:
                    news_list.append({
                        'title': link_tag.text.strip(),
                        'description': desc_tag.text.strip() if desc_tag else '',
                        'source': 'MoneyControl',
                        'url': link_tag['href'],
                        'published': datetime.now().isoformat(),
                        'sentiment': get_sentiment(link_tag.text.strip())
                    })
            except Exception:
                continue
        
        return news_list
    except Exception as e:
        print(f"MoneyControl scraping error: {e}")
        return []


def fetch_news_yahoo_finance(ticker: str) -> List[Dict]:
    """Fetch news from Yahoo Finance using yfinance."""
    try:
        stock = yf.Ticker(ticker)
        news = stock.news
        
        news_list = []
        for item in news[:15]:
            news_list.append({
                'title': item.get('title', ''),
                'description': item.get('summary', ''),
                'source': item.get('publisher', 'Yahoo Finance'),
                'url': item.get('link', ''),
                'published': datetime.fromtimestamp(item.get('providerPublishTime', 0)).isoformat(),
                'sentiment': get_sentiment(item.get('title', '') + ' ' + item.get('summary', ''))
            })
        return news_list
    except Exception as e:
        print(f"Yahoo Finance news error: {e}")
        return []


def get_sentiment(text: str) -> str:
    """Analyze sentiment of text using TextBlob."""
    if not SENTIMENT_AVAILABLE or not text:
        return 'neutral'
    
    try:
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        
        if polarity > 0.1:
            return 'positive'
        elif polarity < -0.1:
            return 'negative'
        else:
            return 'neutral'
    except Exception:
        return 'neutral'


def get_stock_news(ticker: str, days: int = 7) -> pd.DataFrame:
    """
    Aggregate news from multiple sources for a given ticker.
    Returns DataFrame with columns: title, description, source, url, published, sentiment
    """
    all_news = []
    
    # Try Yahoo Finance first (most reliable for stocks)
    print(f"Fetching news for {ticker} from Yahoo Finance...")
    all_news.extend(fetch_news_yahoo_finance(ticker))
    
    # Try MoneyControl for Indian stocks
    if ticker.endswith('.NS') or ticker.endswith('.BO'):
        print(f"Fetching news from MoneyControl...")
        all_news.extend(fetch_news_moneycontrol(ticker))
    
    # Try NewsAPI if available
    company_name = get_company_name_from_ticker(ticker)
    print(f"Fetching news from NewsAPI for {company_name}...")
    all_news.extend(fetch_news_newsapi(company_name, days=days))
    
    if not all_news:
        print("No news found.")
        return pd.DataFrame()
    
    # Remove duplicates based on title
    seen_titles = set()
    unique_news = []
    for item in all_news:
        title = item['title'].lower().strip()
        if title and title not in seen_titles:
            seen_titles.add(title)
            unique_news.append(item)
    
    df = pd.DataFrame(unique_news)
    if 'published' in df.columns:
        df['published'] = pd.to_datetime(df['published'])
        df = df.sort_values('published', ascending=False)
    
    return df


def print_news_summary(news_df: pd.DataFrame, max_items: int = 10):
    """Print formatted news summary."""
    if news_df.empty:
        print("No news available.")
        return
    
    print(f"\n{'='*80}")
    print(f"LATEST NEWS ({len(news_df)} articles)")
    print(f"{'='*80}\n")
    
    for idx, row in news_df.head(max_items).iterrows():
        sentiment_emoji = {'positive': '📈', 'negative': '📉', 'neutral': '➡️'}.get(row['sentiment'], '➡️')
        print(f"{sentiment_emoji} [{row['source']}] {row['title']}")
        if row['description']:
            print(f"   {row['description'][:150]}...")
        print(f"   🔗 {row['url']}")
        print(f"   📅 {row['published']}\n")
    
    # Sentiment summary
    sentiment_counts = news_df['sentiment'].value_counts()
    print(f"\nSentiment Summary:")
    for sentiment, count in sentiment_counts.items():
        print(f"  {sentiment.capitalize()}: {count}")
    print(f"{'='*80}\n")


# -----------------------------
# Data Access Layer (Existing)
# -----------------------------

def _fetch_yahoo(ticker: str, period: str = "5y", interval: str = "1d") -> pd.DataFrame:
    data = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)
    if data is None or len(data) == 0:
        raise ValueError(f"No data returned from Yahoo for {ticker}.")

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [c[0] if isinstance(c, tuple) else c for c in data.columns]

    data = data.rename(columns={
        'open':'Open','high':'High','low':'Low','close':'Close','adj close':'Adj Close','volume':'Volume'
    })
    data = data.rename(columns=str.title)

    if 'Close' not in data.columns and 'Adj Close' in data.columns:
        data['Close'] = data['Adj Close']

    needed = {'Close','High','Low'}
    if not needed.issubset(set(data.columns)):
        raise ValueError(f"Downloaded data missing required columns {needed - set(data.columns)} for {ticker}.")

    data.index = pd.to_datetime(data.index)
    data = data[~data.index.duplicated(keep='first')]
    if 'Volume' in data.columns and (data['Volume'] > 0).any():
        data = data[data['Volume'] > 0]
    if {'Open','High','Low','Close'}.issubset(data.columns):
        eq = (data['Open']==data['High']) & (data['High']==data['Low']) & (data['Low']==data['Close'])
        data = data[~eq]

    return data


def _fetch_alpha_vantage(ticker: str, interval: str = "Daily") -> pd.DataFrame:
    key = os.getenv("ALPHA_VANTAGE_KEY")
    if not key:
        raise RuntimeError("ALPHA_VANTAGE_KEY not set.")
    try:
        from alpha_vantage.timeseries import TimeSeries
    except ImportError:
        raise RuntimeError("alpha_vantage package not installed. pip install alpha_vantage")

    ts = TimeSeries(key=key, output_format='pandas')
    data, _ = ts.get_daily_adjusted(symbol=ticker, outputsize='full')
    data.index = pd.to_datetime(data.index)
    data = data.sort_index()
    data = data.rename(columns={
        '1. open': 'Open','2. high': 'High','3. low': 'Low','4. close': 'Close','5. adjusted close': 'Adj Close','6. volume': 'Volume'
    })
    data['Close'] = data['Adj Close']
    data = data[['Open','High','Low','Close','Volume']]
    return data


def get_price_data(ticker: str, period: str = "5y", interval: str = "1d") -> pd.DataFrame:
    try:
        return _fetch_yahoo(ticker, period=period, interval=interval)
    except Exception as e:
        print(f"Yahoo fetch failed: {e}. Trying Alpha Vantage (Daily) if configured...")
        try:
            return _fetch_alpha_vantage(ticker, interval="Daily")
        except Exception as e2:
            raise RuntimeError(f"Could not fetch data from any source: {e2}")

# -----------------------------
# Feature Engineering (Existing)
# -----------------------------

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if 'Close' not in df.columns and 'Adj Close' in df.columns:
        df['Close'] = df['Adj Close']

    if not {'Close','High','Low'}.issubset(df.columns):
        raise ValueError("DataFrame must include Close, High, Low columns")

    if TA_AVAILABLE:
        df['SMA20'] = SMAIndicator(df['Close'], window=20).sma_indicator()
        df['EMA20'] = EMAIndicator(df['Close'], window=20).ema_indicator()
        macd = MACD(df['Close'])
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        rsi = RSIIndicator(df['Close'], window=14)
        df['RSI14'] = rsi.rsi()
        bb = BollingerBands(df['Close'], window=20, window_dev=2)
        df['BB_High'] = bb.bollinger_hband()
        df['BB_Low'] = bb.bollinger_lband()
        atr = AverageTrueRange(df['High'], df['Low'], df['Close'], window=14)
        df['ATR14'] = atr.average_true_range()
    else:
        df['SMA20'] = df['Close'].rolling(20).mean()
        df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
        delta = df['Close'].diff()
        up = delta.clip(lower=0).rolling(14).mean()
        down = (-delta.clip(upper=0)).rolling(14).mean()
        rs = up / (down.replace(0, np.nan))
        df['RSI14'] = 100 - (100 / (1 + rs))
        df['MACD'] = df['Close'].ewm(span=12, adjust=False).mean() - df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        std20 = df['Close'].rolling(20).std()
        df['BB_High'] = df['SMA20'] + 2*std20
        df['BB_Low'] = df['SMA20'] - 2*std20
        tr = np.maximum(df['High']-df['Low'], np.maximum(abs(df['High']-df['Close'].shift()), abs(df['Low']-df['Close'].shift())))
        df['ATR14'] = tr.rolling(14).mean()

    df['Return_1d'] = df['Close'].pct_change()
    for lag in [1,2,3,5,10]:
        df[f'Return_lag_{lag}'] = df['Return_1d'].shift(lag)
    if 'Volume' in df.columns:
        df['Vol_Chg'] = df['Volume'].pct_change().replace([np.inf, -np.inf], np.nan)
    else:
        df['Vol_Chg'] = np.nan

    df = df.dropna()
    return df

# [Rest of the existing functions remain the same: plotting, ML, backtesting, etc.]
# Keeping them intact...

def plot_price(df: pd.DataFrame, ticker: str):
    plt.figure(figsize=(12,6))
    plt.plot(df.index, df['Close'], label=f'{ticker} Close')
    plt.title(f'{ticker} Close Price')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_indicators(df: pd.DataFrame, ticker: str):
    fig, axes = plt.subplots(3, 1, figsize=(12,10), sharex=True)

    axes[0].plot(df.index, df['Close'], label='Close')
    axes[0].plot(df.index, df['SMA20'], label='SMA20', alpha=0.8)
    axes[0].plot(df.index, df['EMA20'], label='EMA20', alpha=0.8)
    axes[0].fill_between(df.index, df['BB_Low'], df['BB_High'], alpha=0.15, label='Bollinger Bands')
    axes[0].legend()
    axes[0].set_title(f'{ticker} Price & Bands')

    axes[1].plot(df.index, df['MACD'], label='MACD')
    axes[1].plot(df.index, df['MACD_Signal'], label='Signal')
    axes[1].legend()
    axes[1].set_title('MACD')

    axes[2].plot(df.index, df['RSI14'], label='RSI14')
    axes[2].axhline(70, linestyle='--', alpha=0.6)
    axes[2].axhline(30, linestyle='--', alpha=0.6)
    axes[2].legend()
    axes[2].set_title('RSI')

    plt.tight_layout()
    plt.show()


def make_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    feats = df[[
        'Return_lag_1','Return_lag_2','Return_lag_3','Return_lag_5','Return_lag_10',
        'SMA20','EMA20','MACD','MACD_Signal','RSI14','BB_High','BB_Low','ATR14','Vol_Chg'
    ]].copy()
    for col in ['SMA20','EMA20','BB_High','BB_Low','ATR14']:
        feats[col] = feats[col] / df['Close']
    y = (df['Return_1d'].shift(-1) > 0).astype(int)
    feats, y = feats.iloc[:-1], y.iloc[:-1]
    return feats, y


def train_model(df: pd.DataFrame, n_splits: int = 5, random_state: int = 42) -> Tuple[Pipeline, dict]:
    X, y = make_features(df)

    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(n_estimators=400, max_depth=None, min_samples_leaf=2, random_state=random_state, n_jobs=-1))
    ])

    tscv = TimeSeriesSplit(n_splits=n_splits)
    oof_preds = np.zeros(len(y))
    fold = 0
    metrics = []
    for train_idx, test_idx in tscv.split(X):
        fold += 1
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
        pipe.fit(X_tr, y_tr)
        proba = pipe.predict_proba(X_te)[:,1]
        oof_preds[test_idx] = proba
        preds = (proba >= 0.5).astype(int)
        acc = accuracy_score(y_te, preds)
        try:
            auc = roc_auc_score(y_te, proba)
        except ValueError:
            auc = np.nan
        metrics.append({'fold': fold, 'accuracy': acc, 'auc': auc})

    pipe.fit(X, y)
    report = {
        'cv_metrics': metrics,
        'oof_accuracy': accuracy_score(y, (oof_preds>=0.5).astype(int)),
        'oof_auc': float(roc_auc_score(y, oof_preds)) if len(np.unique(y))>1 else np.nan
    }
    return pipe, report


def predict_next_days(model: Pipeline, df: pd.DataFrame, days: int = 5) -> pd.DataFrame:
    df_work = df.copy()
    rows = []
    for _ in range(days):
        feats, _ = make_features(df_work)
        proba = model.predict_proba(feats.iloc[[-1]])[0,1]
        direction = 'Up' if proba >= 0.5 else 'Down'
        rows.append({'date': df_work.index[-1] + pd.Timedelta(days=1), 'p_up': float(proba), 'direction': direction})
        last_ret = df_work['Return_1d'].iloc[-20:].std()
        next_ret = last_ret if proba>=0.5 else -last_ret
        next_close = df_work['Close'].iloc[-1] * (1 + next_ret)
        new_row = df_work.iloc[[-1]].copy()
        new_row.index = [df_work.index[-1] + pd.Timedelta(days=1)]
        new_row['Close'] = next_close
        df_work = pd.concat([df_work, new_row])
        df_work = compute_indicators(df_work)
    return pd.DataFrame(rows)


def performance_metrics(equity_curve: pd.Series, freq: int = 252) -> dict:
    returns = equity_curve.pct_change().dropna()
    cagr = (equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (freq/len(equity_curve)) - 1
    sharpe = np.sqrt(freq) * returns.mean() / (returns.std() + 1e-12)
    rollmax = equity_curve.cummax()
    drawdown = equity_curve / rollmax - 1
    max_dd = drawdown.min()
    return { 'CAGR': cagr, 'Sharpe': sharpe, 'MaxDrawdown': max_dd }


def backtest_ml_strategy(model: Pipeline, df: pd.DataFrame, threshold: float = 0.5, fee_bps: float = 5.0) -> Tuple[pd.DataFrame, dict]:
    X, y = make_features(df)
    proba = model.predict_proba(X)[:,1]

    signal = pd.Series((proba >= threshold).astype(int), index=X.index, name='Signal')
    ret = df.loc[X.index, 'Return_1d']
    trades = (signal.diff().fillna(0).abs() > 0).astype(int)
    fee = trades * (fee_bps/10000.0)
    strat_ret = signal * ret - fee

    bh_curve = (1+ret).cumprod()
    strat_curve = (1+strat_ret).cumprod()

    bh_metrics = performance_metrics(bh_curve)
    strat_metrics = performance_metrics(strat_curve)

    out = pd.DataFrame({
        'Close': df.loc[X.index, 'Close'],
        'BuyHoldCurve': bh_curve,
        'MLCurve': strat_curve,
        'Signal': signal,
        'ProbUp': proba
    })
    return out, { 'BuyHold': bh_metrics, 'ML': strat_metrics }


def print_cv_report(report: dict):
    print("Cross-Validation Metrics (TimeSeriesSplit):")
    for m in report['cv_metrics']:
        print(f"  Fold {m['fold']} -> Acc: {m['accuracy']:.3f} | AUC: {m['auc']:.3f}")
    print(f"OOF Accuracy: {report['oof_accuracy']:.3f}")
    print(f"OOF AUC: {report['oof_auc']:.3f}")


def save_model(model: Pipeline, ticker: str, path: str = "models") -> str:
    os.makedirs(path, exist_ok=True)
    fname = os.path.join(path, f"{ticker.replace('.', '_')}_rf.pkl")
    joblib.dump(model, fname)
    return fname


def load_model(ticker: str, path: str = "models") -> Pipeline:
    fname = os.path.join(path, f"{ticker.replace('.', '_')}_rf.pkl")
    if not os.path.exists(fname):
        raise FileNotFoundError(f"Model not found at {fname}. Train first.")
    return joblib.load(fname)

# -----------------------------
# CLI commands (Enhanced with news)
# -----------------------------

def cmd_fetch(args):
    df = get_price_data(args.ticker, period=args.period, interval=args.interval)
    print(df.tail())


def cmd_indicators(args):
    df = get_price_data(args.ticker, period=args.period, interval=args.interval)
    df = compute_indicators(df)
    print(df.tail())


def cmd_plot(args):
    df = get_price_data(args.ticker, period=args.period, interval=args.interval)
    df = compute_indicators(df)
    if args.kind == 'price':
        plot_price(df, args.ticker)
    else:
        plot_indicators(df, args.ticker)


def cmd_train(args):
    df = get_price_data(args.ticker, period=args.period, interval='1d')
    df = compute_indicators(df)
    model, report = train_model(df, n_splits=args.splits, random_state=args.seed)
    print_cv_report(report)
    if args.save:
        path = save_model(model, args.ticker)
        print(f"Model saved to {path}")


def cmd_predict(args):
    try:
        model = load_model(args.ticker)
    except Exception:
        print("No saved model found, training on the fly...")
        df = get_price_data(args.ticker, period=args.period, interval='1d')
        df = compute_indicators(df)
        model, _ = train_model(df)
    df = get_price_data(args.ticker, period=args.period, interval='1d')
    df = compute_indicators(df)
    preds = predict_next_days(model, df, days=args.days)
    print(preds)


def cmd_backtest(args):
    df = get_price_data(args.ticker, period=args.period, interval='1d')
    df = compute_indicators(df)
    model, report = train_model(df, n_splits=args.splits, random_state=args.seed)
    print_cv_report(report)
    bt, metrics = backtest_ml_strategy(model, df, threshold=args.threshold, fee_bps=args.fee_bps)
    print("\nPerformance Metrics:")
    for k, v in metrics.items():
        print(f"{k}: CAGR={v['CAGR']:.2%} | Sharpe={v['Sharpe']:.2f} | MaxDD={v['MaxDrawdown']:.2%}")
    if args.plot:
        plt.figure(figsize=(12,6))
        plt.plot(bt.index, bt['BuyHoldCurve'], label='Buy & Hold')
        plt.plot(bt.index, bt['MLCurve'], label='ML Strategy')
        plt.title(f"{args.ticker} Strategy vs Buy&Hold")
        plt.xlabel('Date')
        plt.ylabel('Equity Curve (normalized)')
        plt.legend()
        plt.tight_layout()
        plt.show()


def cmd_report(args):
    df = get_price_data(args.ticker, period=args.period, interval='1d')
    df = compute_indicators(df)
    model, report = train_model(df, n_splits=args.splits, random_state=args.seed)
    print_cv_report(report)
    last = df.iloc[-1]
    print("\nLatest Snapshot:")
    print(last[['Close','SMA20','EMA20','RSI14','MACD','MACD_Signal','BB_Low','BB_High','ATR14']])
    plt.figure(figsize=(12,6))
    sns.histplot(df['Return_1d'].dropna(), bins=60, kde=True)
    plt.title(f"{args.ticker} Daily Returns Distribution")
    plt.xlabel('Daily Return')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()


def cmd_news(args):
    """NEW: Fetch and display latest news for a ticker."""
    news_df = get_stock_news(args.ticker, days=args.days)
    print_news_summary(news_df, max_items=args.max_items)


def cmd_bot(args):
    print("\nWelcome to the Stock Analysis Bot! Type 'help' to see commands. Type 'quit' to exit.\n")
    cached_df = None
    cached_ticker = None
    model = None
    while True:
        try:
            q = input("bot> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break
        if q.lower() in {"quit","exit"}:
            print("Bye!")
            break
        if q.lower() in {"help","?"}:
            print("""
Commands:
  load <TICKER> [PERIOD]
  indicators
  plot [price|full]
  train [splits]
  predict [days]
  backtest [threshold]
  news [days]          # NEW: show latest news
  stats
  save
  help | quit
  
Examples:
  load RELIANCE.NS 10y
  train 5
  backtest 0.55
  predict 7
  news 7
""")
            continue
        if q.startswith("load "):
            parts = q.split()
            cached_ticker = parts[1]
            per = parts[2] if len(parts) > 2 else '5y'
            cached_df = get_price_data(cached_ticker, period=per, interval='1d')
            cached_df = compute_indicators(cached_df)
            print(f"Loaded {cached_ticker} with {len(cached_df)} rows.")
            continue
        if q.startswith("plot"):
            if cached_df is None:
                print("Load a ticker first.")
                continue
            kind = q.split()[1] if len(q.split())>1 else 'full'
            if kind == 'price':
                plot_price(cached_df, cached_ticker)
            else:
                plot_indicators(cached_df, cached_ticker)
            continue
        if q.startswith("indicators"):
            if cached_df is None:
                print("Load a ticker first.")
                continue
            print(cached_df.tail()[['Close','SMA20','EMA20','RSI14','MACD','MACD_Signal','BB_Low','BB_High','ATR14']])
            continue
        if q.startswith("train"):
            if cached_df is None:
                print("Load a ticker first.")
                continue
            parts = q.split()
            splits = int(parts[1]) if len(parts)>1 else 5
            model, report = train_model(cached_df, n_splits=splits)
            print_cv_report(report)
            continue
        if q.startswith("predict"):
            if cached_df is None:
                print("Load a ticker first.")
                continue
            if model is None:
                model, _ = train_model(cached_df)
            parts = q.split()
            days = int(parts[1]) if len(parts)>1 else 5
            print(predict_next_days(model, cached_df, days=days))
            continue
        if q.startswith("backtest"):
            if cached_df is None:
                print("Load a ticker first.")
                continue
            if model is None:
                model, _ = train_model(cached_df)
            parts = q.split()
            thr = float(parts[1]) if len(parts)>1 else 0.5
            bt, metrics = backtest_ml_strategy(model, cached_df, threshold=thr)
            print("Performance:")
            for k, v in metrics.items():
                print(f"{k}: CAGR={v['CAGR']:.2%} | Sharpe={v['Sharpe']:.2f} | MaxDD={v['MaxDrawdown']:.2%}")
            plt.figure(figsize=(12,6))
            plt.plot(bt.index, bt['BuyHoldCurve'], label='Buy & Hold')
            plt.plot(bt.index, bt['MLCurve'], label='ML Strategy')
            plt.title(f"{cached_ticker} Strategy vs Buy&Hold")
            plt.xlabel('Date')
            plt.ylabel('Equity Curve (normalized)')
            plt.legend()
            plt.tight_layout()
            plt.show()
            continue
        if q.startswith("news"):
            if cached_ticker is None:
                print("Load a ticker first.")
                continue
            parts = q.split()
            days = int(parts[1]) if len(parts)>1 else 7
            news_df = get_stock_news(cached_ticker, days=days)
            print_news_summary(news_df, max_items=10)
            continue
        if q.startswith("stats"):
            if cached_df is None:
                print("Load a ticker first.")
                continue
            print(cached_df.describe().T)
            continue
        if q.startswith("save"):
            if model is None or cached_ticker is None:
                print("Nothing to save. Train a model first.")
                continue
            path = save_model(model, cached_ticker)
            print(f"Saved model to {path}")
            continue
        print("Unknown command. Type 'help' for options.")


# -----------------------------
# Parser
# -----------------------------

def build_parser():
    p = argparse.ArgumentParser(description="Stock Analysis Bot (Python)")
    sub = p.add_subparsers(dest='cmd')

    pf = sub.add_parser('fetch', help='Fetch raw data')
    pf.add_argument('--ticker', required=True)
    pf.add_argument('--period', default='5y')
    pf.add_argument('--interval', default='1d')
    pf.set_defaults(func=cmd_fetch)

    pi = sub.add_parser('indicators', help='Compute & show indicators')
    pi.add_argument('--ticker', required=True)
    pi.add_argument('--period', default='5y')
    pi.add_argument('--interval', default='1d')
    pi.set_defaults(func=cmd_indicators)

    pp = sub.add_parser('plot', help='Plot price or indicators')
    pp.add_argument('--ticker', required=True)
    pp.add_argument('--period', default='5y')
    pp.add_argument('--interval', default='1d')
    pp.add_argument('--kind', choices=['price','full'], default='full')
    pp.set_defaults(func=cmd_plot)

    pt = sub.add_parser('train', help='Train ML model')
    pt.add_argument('--ticker', required=True)
    pt.add_argument('--period', default='10y')
    pt.add_argument('--splits', type=int, default=5)
    pt.add_argument('--seed', type=int, default=42)
    pt.add_argument('--save', action='store_true')
    pt.set_defaults(func=cmd_train)

    pdp = sub.add_parser('predict', help='Predict next N days (direction prob)')
    pdp.add_argument('--ticker', required=True)
    pdp.add_argument('--period', default='10y')
    pdp.add_argument('--days', type=int, default=5)
    pdp.set_defaults(func=cmd_predict)

    pb = sub.add_parser('backtest', help='Backtest ML strategy vs Buy&Hold')
    pb.add_argument('--ticker', required=True)
    pb.add_argument('--period', default='10y')
    pb.add_argument('--splits', type=int, default=5)
    pb.add_argument('--seed', type=int, default=42)
    pb.add_argument('--threshold', type=float, default=0.5)
    pb.add_argument('--fee_bps', type=float, default=5.0, help='Round-trip fee in bps (0.01% = 1 bps)')
    pb.add_argument('--plot', action='store_true')
    pb.set_defaults(func=cmd_backtest)

    pr = sub.add_parser('report', help='Quick EDA + latest snapshot + CV metrics')
    pr.add_argument('--ticker', required=True)
    pr.add_argument('--period', default='5y')
    pr.add_argument('--splits', type=int, default=5, help='Number of cross-validation splits')
    pr.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    pr.set_defaults(func=cmd_report)

    # NEW: News command
    pn = sub.add_parser('news', help='Fetch latest news for a ticker')
    pn.add_argument('--ticker', required=True)
    pn.add_argument('--days', type=int, default=7, help='News from last N days')
    pn.add_argument('--max_items', type=int, default=10, help='Max news items to display')
    pn.set_defaults(func=cmd_news)

    pbm = sub.add_parser('bot', help='Interactive bot mode')
    pbm.set_defaults(func=cmd_bot)

    return p


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, 'func'):
        parser.print_help()
        return 0
    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())