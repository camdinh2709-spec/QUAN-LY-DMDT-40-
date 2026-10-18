# ==============================================================================
# HOSE V4 QUANTITATIVE PORTFOLIO ENGINE — STREAMLIT WEB APP
# ==============================================================================
# Nested Walk-Forward FA/TA -> Top N -> EW / MPT-LW / MinVar / HRP / Blend50
# Market Regime & Target Cash Policy (Mục tiêu tiền mặt tối đa 40%)
# ==============================================================================

import io
import os
import re
import math
import time
import warnings
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from scipy.optimize import minimize
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
from sklearn.covariance import LedoitWolf

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------------------
# 1. STREAMLIT CONFIG & CUSTOM STYLES
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="HOSE V4 Quant Portfolio Backtester",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        text-align: center;
    }
    .metric-val {
        font-size: 1.6rem;
        font-weight: bold;
        color: #0f172a;
    }
    .metric-lbl {
        font-size: 0.85rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        white-space: pre-wrap;
        background-color: #f8fafc;
        border-radius: 6px 6px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #e0e7ff !important;
        color: #1e3a8a !important;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------------------
# 2. HELPER FUNCTIONS & DATA STANDARDIZATION
# ------------------------------------------------------------------------------
TRADING_DAYS = 252

def clean_num(series):
    """Làm sạch chuỗi số chứa dấu phẩy phân cách phần nghìn."""
    if series is None:
        return pd.Series(dtype=float)
    if series.dtype == object or series.dtype == str:
        s = series.astype(str).str.replace(',', '').str.strip()
        return pd.to_numeric(s, errors='coerce')
    return pd.to_numeric(series, errors='coerce')

def parse_dates_safely(s):
    """Ưu tiên parse mm/dd/yyyy sau đó là dd/mm/yyyy."""
    s = pd.Series(s)
    x = pd.to_datetime(s, format="%m/%d/%Y", errors="coerce")
    if x.notna().mean() < 0.80:
        x = pd.to_datetime(s, errors="coerce", dayfirst=False)
    if x.notna().mean() < 0.80:
        x = pd.to_datetime(s, errors="coerce", dayfirst=True)
    return x

def map_vietnamese_columns(columns):
    """Tự động nhận diện và ánh xạ tên cột tiếng Việt hoặc tiếng Anh."""
    mapping = {}
    for c in columns:
        cl = str(c).lower().strip()
        # Ticker
        if any(k in cl for k in ['mã', 'ticker', 'symbol', 'code']) and not any(k in cl for k in ['mở', 'm?', 'khớp', 'kh?p']):
            mapping[c] = 'Ticker'
        # Date
        elif any(k in cl for k in ['ngày', 'date', 'ngay', 'time']):
            mapping[c] = 'Date'
        # Adjusted close
        elif any(k in cl for k in ['điều chỉnh', 'dieu chinh', '?i?u', 'adj']):
            mapping[c] = 'adj_close'
        # Close
        elif any(k in cl for k in ['đóng cửa', 'dong cua', '?óng', 'close']):
            mapping[c] = 'Close'
        # Open
        elif any(k in cl for k in ['mở cửa', 'mo cua', 'm? c', 'open']):
            mapping[c] = 'Open'
        # High
        elif any(k in cl for k in ['cao nhất', 'cao nhat', 'cao', 'high']):
            mapping[c] = 'High'
        # Low
        elif any(k in cl for k in ['thấp nhất', 'thap nhat', 'th?p', 'low']):
            mapping[c] = 'Low'
        # Volume
        elif any(k in cl for k in ['khớp lệnh', 'kh?p l', 'volume', 'vol']) and not any(k in cl for k in ['giá trị', 'gi tr', 'tr?']):
            mapping[c] = 'Volume'
    return mapping

@st.cache_data(show_spinner=False)
def load_and_standardize_data(file_source, use_adjusted=True):
    """Đọc dữ liệu, chuẩn hóa cột OHLCV và điều chỉnh giá (nếu có)."""
    if isinstance(file_source, str) or isinstance(file_source, Path):
        try:
            raw = pd.read_csv(file_source, encoding='utf-8', low_memory=False)
        except Exception:
            raw = pd.read_csv(file_source, encoding='latin1', low_memory=False)
    else:
        # Streamlit UploadedFile
        fname = getattr(file_source, 'name', '').lower()
        if fname.endswith(('.xlsx', '.xls')):
            raw = pd.read_excel(file_source)
        else:
            try:
                raw = pd.read_csv(file_source, encoding='utf-8', low_memory=False)
            except Exception:
                file_source.seek(0)
                raw = pd.read_csv(file_source, encoding='latin1', low_memory=False)

    col_map = map_vietnamese_columns(raw.columns)
    df = raw.rename(columns=col_map)

    required = ["Ticker", "Date", "Open", "High", "Low", "Close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Thiếu các cột bắt buộc: {missing}. Các cột hiện có: {list(raw.columns)}")

    out = pd.DataFrame({
        "Ticker": df["Ticker"].astype(str).str.strip().str.upper(),
        "Date": parse_dates_safely(df["Date"]),
        "Open": clean_num(df["Open"]),
        "High": clean_num(df["High"]),
        "Low": clean_num(df["Low"]),
        "Close": clean_num(df["Close"]),
        "Volume": clean_num(df["Volume"]).fillna(0).clip(lower=0) if "Volume" in df.columns else 0.0,
    })

    if "adj_close" in df.columns and use_adjusted:
        out["adj_close"] = clean_num(df["adj_close"])
        adj_ratio = (out["adj_close"] / out["Close"].replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(1.0)
        out["Open"] = out["Open"] * adj_ratio
        out["High"] = out["High"] * adj_ratio
        out["Low"] = out["Low"] * adj_ratio
        out["Close"] = out["adj_close"]

    out = out.dropna(subset=["Ticker", "Date", "Open", "High", "Low", "Close"])
    out = out[(out[["Open", "High", "Low", "Close"]] > 0).all(axis=1)]
    valid_ohlc = (
        (out["High"] >= out[["Open", "Close", "Low"]].max(axis=1)) &
        (out["Low"] <= out[["Open", "Close", "High"]].min(axis=1))
    )
    out = out.loc[valid_ohlc]
    out = out.drop_duplicates(["Ticker", "Date"], keep="last")
    data = out.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    return data

def build_price_dict(data):
    return {
        t: g[["Date", "Open", "High", "Low", "Close", "Volume"]]
             .drop_duplicates("Date").sort_values("Date").set_index("Date")
        for t, g in data.groupby("Ticker")
    }

def build_synthetic_market_proxy(price_dict, exclude=None):
    exclude = set(exclude or [])
    closes = [px["Close"].rename(t) for t, px in price_dict.items() if t not in exclude]
    m = pd.concat(closes, axis=1).sort_index()
    ret = m.pct_change(fill_method=None)
    ewret = ret.mean(axis=1, skipna=True).fillna(0)
    close = 1000 * (1 + ewret).cumprod()
    return pd.DataFrame({"Open": close, "High": close, "Low": close, "Close": close, "Volume": 0.0}, index=close.index)


# ------------------------------------------------------------------------------
# 3. TECHNICAL INDICATORS & FEATURE ENGINEERING
# ------------------------------------------------------------------------------
def ema(s, span):
    return s.ewm(span=span, adjust=False, min_periods=span).mean()

def rsi_wilder(close, window=14):
    d = close.diff()
    gain, loss = d.clip(lower=0), -d.clip(upper=0)
    ag = gain.ewm(alpha=1/window, adjust=False, min_periods=window).mean()
    al = loss.ewm(alpha=1/window, adjust=False, min_periods=window).mean()
    rs = ag / al.replace(0, np.nan)
    rsi = 100 - 100/(1+rs)
    rsi.loc[(al == 0) & (ag > 0)] = 100
    rsi.loc[(ag == 0) & (al > 0)] = 0
    return rsi

def atr_wilder(df, window=14):
    prev = df["Close"].shift(1)
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - prev).abs(),
        (df["Low"] - prev).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/window, adjust=False, min_periods=window).mean()

def adx_wilder(df, window=14):
    high, low = df["High"], df["Low"]
    up, down = high.diff(), -low.diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)
    atr = atr_wilder(df, window)
    plus_di = 100 * plus_dm.ewm(alpha=1/window, adjust=False, min_periods=window).mean() / atr.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(alpha=1/window, adjust=False, min_periods=window).mean() / atr.replace(0, np.nan)
    dx = 100 * (plus_di-minus_di).abs() / (plus_di+minus_di).replace(0, np.nan)
    adx = dx.ewm(alpha=1/window, adjust=False, min_periods=window).mean()
    return adx, plus_di, minus_di

def calculate_ta_features(df):
    out = df.copy().sort_index()
    out["EMA20"] = ema(out["Close"], 20)
    out["EMA50"] = ema(out["Close"], 50)
    out["EMA200"] = ema(out["Close"], 200)

    ema12, ema26 = ema(out["Close"], 12), ema(out["Close"], 26)
    out["MACD"] = ema12 - ema26
    out["MACD_Signal"] = ema(out["MACD"], 9)

    out["ADX"], out["Plus_DI"], out["Minus_DI"] = adx_wilder(out, 14)
    out["ATR"] = atr_wilder(out, 14)
    out["ATR_pct"] = 100 * out["ATR"] / out["Close"].replace(0, np.nan)
    out["ATR_pct_Q90"] = out["ATR_pct"].rolling(126, min_periods=40).quantile(0.90)

    out["Volume_MA20"] = out["Volume"].rolling(20, min_periods=20).mean()
    out["Volume_Ratio"] = out["Volume"] / out["Volume_MA20"].replace(0, np.nan)
    out["OBV"] = (np.sign(out["Close"].diff()).fillna(0) * out["Volume"]).cumsum()
    out["OBV_EMA20"] = ema(out["OBV"], 20)

    out["RSI"] = rsi_wilder(out["Close"], 14)
    out["ROC10"] = out["Close"].pct_change(10) * 100
    out["High_52W"] = out["High"].rolling(252, min_periods=150).max()
    out["Close_to_52W_High"] = out["Close"] / out["High_52W"].replace(0, np.nan)
    out["Avg_Traded_Value_60"] = (out["Close"] * out["Volume"]).rolling(60, min_periods=20).mean()

    out["Momentum63"] = out["Close"].pct_change(63)
    out["Momentum126"] = out["Close"].pct_change(126)
    out["Momentum252"] = out["Close"].pct_change(252)
    out["Vol63"] = out["Close"].pct_change().rolling(63, min_periods=40).std() * np.sqrt(TRADING_DAYS)
    rollmax = out["Close"].rolling(126, min_periods=60).max()
    out["DD126"] = out["Close"] / rollmax - 1

    # TA Score 0-10
    out["Trend_Score"] = (
        (out["Close"] > out["EMA50"]).astype(int)*2 +
        (out["EMA50"] > out["EMA200"]).astype(int)*2 +
        (out["MACD"] > out["MACD_Signal"]).astype(int)
    ).clip(0, 5)
    momentum_core = ((out["RSI"] >= 45) & (out["RSI"] <= 75)) | (out["ROC10"] > 0)
    out["Momentum_Score"] = (
        momentum_core.astype(int)*2 +
        (out["Close_to_52W_High"] >= 0.85).astype(int)
    ).clip(0, 3)
    out["Volume_Score"] = ((out["Volume_Ratio"] >= 1.0) | (out["OBV"] > out["OBV_EMA20"])).astype(int)
    out["Volatility_Score"] = (
        out["ATR_pct"].notna() & out["ATR_pct_Q90"].notna() &
        (out["ATR_pct"] <= out["ATR_pct_Q90"])
    ).astype(int)
    out["TA_Score"] = (
        out["Trend_Score"] + out["Momentum_Score"] +
        out["Volume_Score"] + out["Volatility_Score"]
    ).clip(0, 10).astype(int)
    return out

def stateful_active_from_score(score, buy_threshold, exit_threshold):
    state = 0
    vals = []
    for v in score.fillna(-np.inf):
        if v >= buy_threshold:
            state = 1
        elif v <= exit_threshold:
            state = 0
        vals.append(state)
    return pd.Series(vals, index=score.index, dtype=int)


# ------------------------------------------------------------------------------
# 4. METRICS & SIMULATION
# ------------------------------------------------------------------------------
def annualized_sharpe(r):
    r = pd.Series(r).replace([np.inf, -np.inf], np.nan).dropna()
    return np.sqrt(TRADING_DAYS)*r.mean()/r.std(ddof=1) if len(r) > 1 and r.std(ddof=1) > 1e-12 else np.nan

def annualized_sortino(r):
    r = pd.Series(r).replace([np.inf, -np.inf], np.nan).dropna()
    neg = r[r < 0]
    return np.sqrt(TRADING_DAYS)*r.mean()/neg.std(ddof=1) if len(neg) > 1 and neg.std(ddof=1) > 1e-12 else np.nan

def metrics_from_equity(eq):
    eq = pd.Series(eq).dropna()
    if len(eq) < 2:
        return {}
    r = eq.pct_change().dropna()
    total = eq.iloc[-1]/eq.iloc[0] - 1
    years = max((eq.index[-1]-eq.index[0]).days/365.25, 1/365.25)
    cagr = (eq.iloc[-1]/eq.iloc[0])**(1/years) - 1
    dd = eq/eq.cummax() - 1
    mdd = dd.min()
    vol = r.std(ddof=1)*np.sqrt(TRADING_DAYS) if len(r) > 1 else np.nan
    return {
        "Return [%]": total * 100,
        "CAGR [%]": cagr * 100,
        "Sharpe": annualized_sharpe(r),
        "Sortino": annualized_sortino(r),
        "Volatility [%]": vol * 100 if pd.notna(vol) else np.nan,
        "MaxDD [%]": mdd * 100 if pd.notna(mdd) else np.nan,
        "Calmar": cagr/abs(mdd) if pd.notna(mdd) and mdd < 0 else np.nan,
    }

def simulate_single_stock(features, start, end, buy_threshold, exit_threshold,
                          initial_cash=1_000_000.0, commission=0.0015, slippage=0.0005):
    active_close = stateful_active_from_score(features["TA_Score"], buy_threshold, exit_threshold)
    active_open = active_close.shift(1).fillna(0).astype(int)
    x = features.loc[(features.index >= pd.Timestamp(start)) & (features.index <= pd.Timestamp(end))].copy()
    active_open = active_open.reindex(x.index).fillna(0).astype(int)
    if x.empty:
        return {}

    cash, shares, state, trades = float(initial_cash), 0.0, 0, 0
    eq = []
    for dt, row in x.iterrows():
        d = int(active_open.loc[dt])
        op, cl = float(row["Open"]), float(row["Close"])
        if d != state:
            if d == 1:
                px = op*(1+slippage)
                qty = math.floor(cash/(px*(1+commission)))
                if qty > 0:
                    cash -= qty*px*(1+commission)
                    shares = qty
                    state = 1
                    trades += 1
            else:
                px = op*(1-slippage)
                cash += shares*px*(1-commission)
                shares = 0
                state = 0
                trades += 1
        eq.append((dt, cash + shares*cl))

    m = metrics_from_equity(pd.Series(dict(eq)).sort_index())
    m["Trades"] = trades
    return m

def make_equal_folds(dates, start, end, n_folds=3):
    d = pd.DatetimeIndex(dates)
    d = d[(d >= pd.Timestamp(start)) & (d <= pd.Timestamp(end))]
    if len(d) < n_folds*20:
        n_folds = max(1, len(d)//20)
    chunks = np.array_split(d, n_folds)
    folds = []
    for i, c in enumerate(chunks, 1):
        if len(c) >= 20:
            folds.append({"Fold": f"F{i}", "start": pd.Timestamp(c[0]), "end": pd.Timestamp(c[-1])})
    return folds

def build_universe(price_dict, asof, benchmark_ticker=None, liquidity_lookback=60,
                   liquidity_min_percentile=0.20, min_history_floor=180, min_history_cap=400):
    all_dates = set()
    for px in price_dict.values():
        all_dates.update(px.loc[:asof].index)
    pre_sessions = len(all_dates)
    min_hist = min(min_history_cap, max(min_history_floor, int(0.55*pre_sessions)))

    rows = []
    for t, px in price_dict.items():
        if t == benchmark_ticker:
            continue
        h = px.loc[:asof]
        if h.empty:
            continue
        avg_value = (h["Close"]*h["Volume"]).tail(liquidity_lookback).mean()
        rows.append({
            "Ticker": t,
            "HistorySessions": len(h),
            "AvgTradedValue": avg_value,
            "EligibleHistory": len(h) >= min_hist,
        })

    u = pd.DataFrame(rows)
    if u.empty:
        return u
    ok = u["EligibleHistory"] & u["AvgTradedValue"].notna()
    u.loc[ok, "LiquidityPercentile"] = u.loc[ok, "AvgTradedValue"].rank(pct=True)
    u["PassLiquidity"] = u["LiquidityPercentile"].fillna(0) >= liquidity_min_percentile
    u["Candidate"] = u["EligibleHistory"] & u["PassLiquidity"]
    return u.sort_values("AvgTradedValue", ascending=False).reset_index(drop=True)

def choose_global_ta_thresholds(feature_store, candidates, folds,
                                buy_choices=[5, 6, 7], exit_choices=[2, 3, 4],
                                stability_penalty=0.20):
    trials = []
    for b, e in product(buy_choices, exit_choices):
        if e >= b:
            continue
        stock_scores = []
        for t in candidates:
            ms = []
            for f in folds:
                m = simulate_single_stock(feature_store[t], f["start"], f["end"], b, e)
                if m:
                    sh = m.get("Sharpe", np.nan)
                    sh = -3.0 if not np.isfinite(sh) else float(sh)
                    ms.append((sh, float(m.get("Return [%]", 0)), float(m.get("MaxDD [%]", -100))))
            if ms:
                a = np.array(ms, float)
                stock_scores.append([np.nanmean(a[:,0]), np.nanmean(a[:,1]), np.nanmin(a[:,2])])
        if not stock_scores:
            continue

        z = np.array(stock_scores, float)
        med_sh = float(np.nanmedian(z[:,0]))
        sd_sh = float(np.nanstd(z[:,0]))
        med_ret = float(np.nanmedian(z[:,1]))
        med_dd = float(np.nanmedian(z[:,2]))
        score = med_sh - stability_penalty*sd_sh + 0.005*med_ret + 0.003*med_dd
        trials.append({
            "BuyThreshold": b,
            "ExitThreshold": e,
            "GlobalCVScore": score,
            "MedianSharpe": med_sh,
            "SharpeStd": sd_sh,
            "MedianReturnPct": med_ret,
            "MedianMaxDDPct": med_dd,
            "NStocks": len(stock_scores),
        })

    tr = pd.DataFrame(trials).sort_values("GlobalCVScore", ascending=False).reset_index(drop=True)
    if tr.empty:
        return 6, 3, tr
    best = tr.iloc[0]
    return int(best["BuyThreshold"]), int(best["ExitThreshold"]), tr


# ------------------------------------------------------------------------------
# 5. AS-OF RANKING & SELECTION
# ------------------------------------------------------------------------------
def trailing_stock_quality(features, asof, b, e, lookback=126):
    h = features.loc[:asof]
    if len(h) < 40:
        return {"TrailingSharpe": -3.0, "TrailingReturnPct": -100.0, "TrailingMaxDDPct": -100.0}
    sub_idx = h.index[-min(lookback, len(h)):]
    m = simulate_single_stock(features, sub_idx[0], sub_idx[-1], b, e)
    sh = m.get("Sharpe", np.nan) if m else np.nan
    return {
        "TrailingSharpe": -3.0 if not np.isfinite(sh) else float(sh),
        "TrailingReturnPct": float(m.get("Return [%]", -100.0)) if m else -100.0,
        "TrailingMaxDDPct": float(m.get("MaxDD [%]", -100.0)) if m else -100.0,
    }

def rank_and_select_asof(feature_store, price_dict, benchmark_features, benchmark_ticker,
                         asof, b, e, top_n=5, corr_lookback=126, max_corr=0.80):
    universe = build_universe(price_dict, asof, benchmark_ticker)
    if universe.empty:
        raise ValueError(f"Universe rỗng tại {asof}")
    candidates = universe.loc[universe["Candidate"], "Ticker"].tolist()
    bench = benchmark_features.loc[:asof]
    bench_m126 = bench["Close"].pct_change(126).iloc[-1] if len(bench) > 126 else np.nan
    uidx = universe.set_index("Ticker")

    rows = []
    for t in candidates:
        feat = feature_store[t].loc[:asof]
        if feat.empty:
            continue
        snap = feat.iloc[-1]
        tq = trailing_stock_quality(feature_store[t], asof, b, e)
        rows.append({
            "Ticker": t,
            **tq,
            "TA_Score": float(snap.get("TA_Score", np.nan)),
            "Momentum63": float(snap.get("Momentum63", np.nan)),
            "Momentum126": float(snap.get("Momentum126", np.nan)),
            "Momentum252": float(snap.get("Momentum252", np.nan)),
            "RelativeStrength126": (
                float(snap.get("Momentum126", np.nan)-bench_m126)
                if pd.notna(bench_m126) else float(snap.get("Momentum126", np.nan))
            ),
            "Vol63": float(snap.get("Vol63", np.nan)),
            "RSI": float(snap.get("RSI", np.nan)),
            "LiquidityPercentile": float(uidx.loc[t, "LiquidityPercentile"]),
        })

    r = pd.DataFrame(rows)
    if r.empty:
        raise ValueError(f"Không có cổ phiếu đủ điều kiện tại {asof}")

    r["PassTA"] = r["TA_Score"].fillna(0) >= 4
    if int(r["PassTA"].sum()) >= top_n:
        r = r[r["PassTA"]].copy()

    vol_cap = r["Vol63"].quantile(0.75)
    dd_floor = r["TrailingMaxDDPct"].quantile(0.15)
    r["PassRiskGate"] = (
        (r["Vol63"].isna() | (r["Vol63"] <= vol_cap)) &
        (r["TrailingMaxDDPct"].isna() | (r["TrailingMaxDDPct"] >= dd_floor)) &
        (r["RSI"].isna() | (r["RSI"] <= 80.0))
    )
    if int(r["PassRiskGate"].sum()) >= top_n:
        r = r[r["PassRiskGate"]].copy()

    r["P_Trailing"] = r["TrailingSharpe"].rank(pct=True)
    r["P_TA"] = r["TA_Score"].rank(pct=True)
    r["P_RS"] = r["RelativeStrength126"].rank(pct=True)
    r["P_LowVol"] = (-r["Vol63"]).rank(pct=True)
    r["P_Liq"] = r["LiquidityPercentile"].rank(pct=True)
    mommix = 0.6*r["Momentum126"].fillna(r["Momentum63"]) + 0.4*r["Momentum63"].fillna(0)
    r["P_Momentum"] = mommix.rank(pct=True)
    r["TA_Composite"] = (
        0.25*r["P_Trailing"] +
        0.20*r["P_TA"] +
        0.25*r["P_RS"] +
        0.10*r["P_LowVol"] +
        0.05*r["P_Liq"] +
        0.15*r["P_Momentum"]
    ) * 100

    r["FinalScore"] = r["TA_Composite"]
    filter_mode = "TA"
    r = r.sort_values(["FinalScore", "TrailingSharpe"], ascending=False).reset_index(drop=True)

    corr_tickers = r["Ticker"].tolist()
    close_list = [feature_store[t].loc[:asof, "Close"].rename(t) for t in corr_tickers]
    corr_px = pd.concat(close_list, axis=1, join="inner").tail(corr_lookback+1) if close_list else pd.DataFrame()
    corr = corr_px.pct_change(fill_method=None).dropna().corr() if len(corr_px) > 20 else pd.DataFrame()

    selected = []
    for _, row in r.iterrows():
        t = row["Ticker"]
        if selected and not corr.empty:
            too_close = any(
                t in corr.index and x in corr.columns and
                pd.notna(corr.loc[t, x]) and abs(corr.loc[t, x]) > max_corr
                for x in selected
            )
            if too_close:
                continue
        selected.append(t)
        if len(selected) >= top_n:
            break

    if len(selected) < top_n:
        for t in r["Ticker"]:
            if t not in selected:
                selected.append(t)
            if len(selected) >= top_n:
                break

    locked = r[r["Ticker"].isin(selected)].copy()
    locked["SelectedOrder"] = locked["Ticker"].map({t:i+1 for i,t in enumerate(selected)})
    locked["FilterModeUsed"] = filter_mode
    locked["AsOfDate"] = pd.Timestamp(asof)
    locked = locked.sort_values("SelectedOrder")
    return r, locked, selected, filter_mode

def first_session_each_period(calendar, start, end, freq="Q"):
    idx = pd.DatetimeIndex(calendar)
    idx = idx[(idx >= pd.Timestamp(start)) & (idx <= pd.Timestamp(end))]
    if len(idx) == 0:
        return []
    s = pd.Series(idx, index=idx)
    key = idx.to_period(freq)
    firsts = s.groupby(key).first().tolist()
    if idx[0] not in firsts:
        firsts = [idx[0]] + firsts
    return sorted(set(pd.Timestamp(x) for x in firsts))

def previous_session(calendar, dt):
    idx = pd.DatetimeIndex(calendar)
    prev = idx[idx < pd.Timestamp(dt)]
    return pd.Timestamp(prev[-1]) if len(prev) else None

def build_selection_schedule(feature_store, price_dict, benchmark_ticker, benchmark_features,
                             calendar, start, end, b, e, top_n=5, selection_freq="Q"):
    effective_dates = first_session_each_period(calendar, start, end, selection_freq)
    rows = []
    details = {}
    for eff in effective_dates:
        asof = previous_session(calendar, eff)
        if asof is None:
            continue
        ranking, locked, selected, mode = rank_and_select_asof(
            feature_store, price_dict, benchmark_features, benchmark_ticker,
            asof, b, e, top_n=top_n
        )
        details[pd.Timestamp(eff)] = {"ranking": ranking, "locked": locked}
        row = {
            "EffectiveDate": pd.Timestamp(eff),
            "AsOfDate": pd.Timestamp(asof),
            "FilterMode": mode,
        }
        for i in range(top_n):
            row[f"Top{i+1}"] = selected[i] if i < len(selected) else None
        rows.append(row)

    schedule = pd.DataFrame(rows).sort_values("EffectiveDate").reset_index(drop=True)
    return schedule, details

def selected_tickers_asof(selection_schedule, effective_date, top_n=5):
    x = selection_schedule[selection_schedule["EffectiveDate"] <= pd.Timestamp(effective_date)]
    if x.empty:
        return []
    row = x.iloc[-1]
    return [row[f"Top{i+1}"] for i in range(top_n) if pd.notna(row.get(f"Top{i+1}"))]


# ------------------------------------------------------------------------------
# 6. PORTFOLIO ALLOCATION & OPTIMIZATION
# ------------------------------------------------------------------------------
def close_matrix(price_dict, tickers, end=None, start=None):
    series = [price_dict[t]["Close"].rename(t) for t in tickers if t in price_dict]
    if not series:
        return pd.DataFrame()
    m = pd.concat(series, axis=1, join="inner").sort_index().dropna(how="any")
    if start is not None:
        m = m.loc[m.index >= pd.Timestamp(start)]
    if end is not None:
        m = m.loc[m.index <= pd.Timestamp(end)]
    return m

def equal_weights(tickers):
    tickers = list(tickers)
    return pd.Series(1/len(tickers), index=tickers, dtype=float) if tickers else pd.Series(dtype=float)

def robust_mu(prices, lookback=252, shrinkage=0.50):
    r = prices.pct_change(fill_method=None).dropna().tail(lookback)
    if r.empty:
        return pd.Series(0.0, index=prices.columns)
    span = max(20, min(lookback, len(r)))
    raw = r.ewm(span=span, adjust=False).mean().iloc[-1] * TRADING_DAYS
    grand = float(raw.mean())
    return shrinkage*raw + (1-shrinkage)*grand

def lw_cov(prices, lookback=252):
    r = prices.pct_change(fill_method=None).dropna().tail(lookback)
    if len(r) < 30:
        return r.cov()*TRADING_DAYS
    est = LedoitWolf().fit(r.values)
    return pd.DataFrame(est.covariance_*TRADING_DAYS, index=r.columns, columns=r.columns)

def max_sharpe_weights(mu, cov, max_weight=0.50, l2_gamma=0.03):
    tickers = list(mu.index)
    n = len(tickers)
    if n == 0:
        return pd.Series(dtype=float)
    max_weight = max(max_weight, 1/n)
    muv = mu.values.astype(float)
    cv = cov.reindex(index=tickers, columns=tickers).fillna(0).values.astype(float) + np.eye(n)*1e-8

    def obj(w):
        ret = float(w@muv)
        vol = math.sqrt(max(float(w@cv@w), 1e-12))
        return -(ret/vol) + l2_gamma*float(np.sum(w*w))

    x0 = np.repeat(1/n, n)
    bounds = [(0, max_weight)]*n
    cons = ({"type":"eq", "fun":lambda w:np.sum(w)-1},)
    res = minimize(obj, x0, method="SLSQP", bounds=bounds, constraints=cons,
                   options={"maxiter":1000, "ftol":1e-10})
    if not res.success or np.any(~np.isfinite(res.x)):
        return equal_weights(tickers)
    w = np.clip(res.x, 0, None)
    return pd.Series(w/w.sum(), index=tickers) if w.sum() > 0 else equal_weights(tickers)

def min_variance_weights(cov, max_weight=0.50):
    tickers = list(cov.index)
    n = len(tickers)
    if n == 0:
        return pd.Series(dtype=float)
    max_weight = max(max_weight, 1/n)
    cv = cov.values.astype(float) + np.eye(n)*1e-8

    def obj(w):
        return float(w@cv@w)

    x0 = np.repeat(1/n, n)
    bounds = [(0, max_weight)]*n
    cons = ({"type":"eq", "fun":lambda w:np.sum(w)-1},)
    res = minimize(obj, x0, method="SLSQP", bounds=bounds, constraints=cons,
                   options={"maxiter":1000, "ftol":1e-12})
    if not res.success:
        return equal_weights(tickers)
    w = np.clip(res.x, 0, None)
    return pd.Series(w/w.sum(), index=tickers)

def _cluster_var(cov, items):
    sub = cov.loc[items, items].values
    ivp = 1/np.diag(sub)
    ivp = ivp/ivp.sum()
    return float(ivp@sub@ivp)

def _quasi_diag(link):
    link = link.astype(int)
    sort_ix = pd.Series([link[-1,0], link[-1,1]])
    num_items = int(link[-1,3])
    while sort_ix.max() >= num_items:
        sort_ix.index = range(0, sort_ix.shape[0]*2, 2)
        df0 = sort_ix[sort_ix >= num_items]
        i = df0.index
        j = df0.values-num_items
        sort_ix.loc[i] = link[j,0]
        df1 = pd.Series(link[j,1], index=i+1)
        sort_ix = pd.concat([sort_ix, df1]).sort_index()
        sort_ix.index = range(sort_ix.shape[0])
    return sort_ix.tolist()

def hrp_weights(prices, method="single"):
    r = prices.pct_change(fill_method=None).dropna()
    if len(r) < 30 or prices.shape[1] < 2:
        return equal_weights(prices.columns)
    cov = r.cov()*TRADING_DAYS
    corr = r.corr().clip(-1,1)
    dist = np.sqrt((1-corr)/2)
    link = linkage(squareform(dist.values, checks=False), method=method)
    ordered = corr.index[_quasi_diag(link)].tolist()

    w = pd.Series(1.0, index=ordered)
    clusters = [ordered]
    while clusters:
        nxt = []
        for c in clusters:
            if len(c) <= 1:
                continue
            s = len(c)//2
            c0, c1 = c[:s], c[s:]
            v0, v1 = _cluster_var(cov, c0), _cluster_var(cov, c1)
            a = 1-v0/(v0+v1) if v0+v1 > 0 else .5
            w[c0] *= a
            w[c1] *= (1-a)
            nxt += [c0, c1]
        clusters = nxt
    return w.reindex(prices.columns).fillna(0)/w.sum()

def compute_weights(model_family, prices):
    tickers = list(prices.columns)
    ew = equal_weights(tickers)
    if len(tickers) == 0 or model_family == "EW":
        return ew
    cov = lw_cov(prices, 252)
    if model_family == "MINVAR_LW":
        return min_variance_weights(cov)
    if model_family == "HRP":
        return hrp_weights(prices.tail(253))
    mu = robust_mu(prices, 252)
    mpt = max_sharpe_weights(mu, cov)
    if model_family == "MPT_LW":
        return mpt
    if model_family == "BLEND50":
        w = 0.5*mpt + 0.5*ew
        return w/w.sum()
    raise ValueError("Model family không hỗ trợ: " + str(model_family))

def build_market_regime(benchmark_features, span=200, slope_lookback=20, use_slope=True):
    x = benchmark_features.copy()
    ema_m = x["Close"].ewm(span=span, adjust=False, min_periods=span).mean()
    slope = ema_m - ema_m.shift(slope_lookback)
    risk_close = x["Close"] > ema_m
    if use_slope:
        risk_close = risk_close & (slope > 0)
    return pd.DataFrame({
        "EMA_Market": ema_m,
        "EMA_Slope": slope,
        "RiskOn_Close": risk_close.fillna(False),
        "RiskOn_Open": risk_close.shift(1).fillna(False),
    }, index=x.index)

def build_stock_open_masks(feature_store, tickers, b, e):
    masks = {}
    for t in tickers:
        active_close = stateful_active_from_score(feature_store[t]["TA_Score"], b, e)
        masks[t] = active_close.shift(1).fillna(0).astype(bool)
    return pd.DataFrame(masks).sort_index()

def build_dynamic_weight_schedule(price_dict, selection_schedule, calendar, start, end,
                                  model_family, allocation_freq="M", top_n=5):
    rebalance_dates = first_session_each_period(calendar, start, end, allocation_freq)
    rows = []
    all_tickers = set()
    for eff in rebalance_dates:
        asof = previous_session(calendar, eff)
        if asof is None:
            continue
        top = selected_tickers_asof(selection_schedule, eff, top_n=top_n)
        if not top:
            continue
        all_tickers.update(top)
        prices = close_matrix(price_dict, top, end=asof)
        if prices.empty or len(prices) < 30:
            w = equal_weights(top)
        else:
            w = compute_weights(model_family, prices.tail(253))
        row = {"EffectiveDate": pd.Timestamp(eff), "AsOfDate": pd.Timestamp(asof)}
        row.update({t: float(w.get(t, 0.0)) for t in top})
        rows.append(row)
    meta = pd.DataFrame(rows).sort_values("EffectiveDate").reset_index(drop=True)
    tickers = sorted(all_tickers)
    weights = meta.set_index("EffectiveDate").reindex(columns=tickers).fillna(0.0)
    return weights, meta[["EffectiveDate", "AsOfDate"]]


# ------------------------------------------------------------------------------
# 7. EVENT-DRIVEN PORTFOLIO ENGINE
# ------------------------------------------------------------------------------
def run_portfolio_engine_dynamic(price_dict, weight_schedule, calendar, start, end,
                                 market_regime_open=None, stock_active_open=None,
                                 use_market_regime=True, use_stock_mask=True,
                                 individual_trailing_stop=False, individual_stop_pct=0.12,
                                 portfolio_trailing_stop=False, portfolio_stop_pct=0.15,
                                 initial_cash=1_000_000_000.0, commission=0.0015,
                                 slippage=0.0005, fractional=False, cash_policy="cash40"):
    tickers = list(weight_schedule.columns)
    idx = pd.DatetimeIndex(calendar)
    idx = idx[(idx >= pd.Timestamp(start)) & (idx <= pd.Timestamp(end))]
    if len(idx) == 0:
        raise ValueError("Không có ngày giao dịch trong kỳ")

    opens = pd.DataFrame(index=idx, columns=tickers, dtype=float)
    closes = pd.DataFrame(index=idx, columns=tickers, dtype=float)
    for t in tickers:
        if t in price_dict:
            opens[t] = price_dict[t]["Open"].reindex(idx)
            closes[t] = price_dict[t]["Close"].reindex(idx)

    ws = weight_schedule.sort_index().reindex(columns=tickers)
    ws = ws.reindex(ws.index.union(idx)).sort_index().ffill().reindex(idx).fillna(0.0)

    regime = (
        pd.Series(True, index=idx)
        if market_regime_open is None
        else pd.Series(market_regime_open).reindex(idx).ffill().fillna(False).astype(bool)
    )
    masks = (
        pd.DataFrame(True, index=idx, columns=tickers)
        if stock_active_open is None
        else stock_active_open.reindex(index=idx, columns=tickers).fillna(False).astype(bool)
    )

    cash = float(initial_cash)
    shares = pd.Series(0.0, index=tickers)
    last_close = pd.Series(np.nan, index=tickers)
    last_target = pd.Series(np.nan, index=tickers)
    entry_peak = pd.Series(np.nan, index=tickers)
    stop_blocked = pd.Series(False, index=tickers)
    portfolio_peak = initial_cash
    portfolio_blocked = False

    orders, eq_rows, exposure_rows = [], [], []
    turnover, rebals = 0.0, 0

    for dt in idx:
        op = opens.loc[dt].astype(float)
        cl = closes.loc[dt].astype(float)
        valid_close = cl.notna()
        last_close.loc[valid_close] = cl.loc[valid_close]
        mark_open = op.where(op.notna(), last_close)
        mark_close = cl.where(cl.notna(), last_close)

        base = ws.loc[dt].clip(lower=0)
        if base.sum() > 1 + 1e-8:
            base = base/base.sum()

        risk_on = bool(regime.loc[dt]) if use_market_regime else True
        active = masks.loc[dt].astype(bool) if use_stock_mask else pd.Series(True, index=tickers)
        for t in tickers:
            if not active[t]:
                stop_blocked[t] = False

        target = base.copy()
        if use_stock_mask:
            target *= active.astype(float)
        if individual_trailing_stop:
            target[stop_blocked] = 0.0
        if use_market_regime and not risk_on:
            target[:] = 0.0
        if portfolio_trailing_stop and portfolio_blocked:
            target[:] = 0.0

        target = target.clip(lower=0)
        if target.sum() > 1 + 1e-8:
            target /= target.sum()

        effective_policy = cash_policy
        min_exposure = {"original": 0.0, "cash40": 0.60, "full": 1.0}.get(effective_policy, 0.60)
        base_total = float(base.sum())
        current_total = float(target.sum())
        if min_exposure > current_total + 1e-10 and base_total > current_total + 1e-10:
            feasible_exposure = min(min_exposure, base_total)
            alpha = (feasible_exposure - current_total) / (base_total - current_total)
            target = (1-alpha)*target + alpha*base
            target = target.clip(lower=0)

        changed = last_target.isna().any() or (target-last_target.fillna(0)).abs().max() > 1e-10
        if changed:
            equity_open = cash + float((shares * mark_open.fillna(0)).sum())
            desired = shares.copy()
            for t in tickers:
                if pd.isna(op[t]) or op[t] <= 0:
                    continue
                if target[t] <= 0:
                    desired[t] = 0.0
                else:
                    px = op[t]*(1+slippage)
                    raw_qty = equity_open*target[t]/(px*(1+commission))
                    desired[t] = raw_qty if fractional else math.floor(raw_qty)

            for t in tickers:
                d = desired[t] - shares[t]
                if d < -1e-12 and pd.notna(op[t]) and op[t] > 0:
                    qty = -d
                    px = op[t]*(1-slippage)
                    notional = qty*px
                    fee = notional*commission
                    cash += notional-fee
                    shares[t] -= qty
                    turnover += notional
                    orders.append({"Date":dt, "Ticker":t, "Side":"SELL", "Shares":qty, "ExecPrice":px, "Fee":fee, "Reason":"TARGET"})
                    if shares[t] <= 1e-12:
                        entry_peak[t] = np.nan

            for t in tickers:
                d = desired[t] - shares[t]
                if d > 1e-12 and pd.notna(op[t]) and op[t] > 0:
                    px = op[t]*(1+slippage)
                    unit = px*(1+commission)
                    qty = min(d, cash/unit) if fractional else min(d, math.floor(cash/unit))
                    if qty > 0:
                        notional = qty*px
                        fee = notional*commission
                        was_zero = shares[t] <= 1e-12
                        cash -= notional+fee
                        shares[t] += qty
                        turnover += notional
                        orders.append({"Date":dt, "Ticker":t, "Side":"BUY", "Shares":qty, "ExecPrice":px, "Fee":fee, "Reason":"TARGET"})
                        if was_zero and pd.notna(mark_close[t]):
                            entry_peak[t] = mark_close[t]

            last_target = target.copy()
            rebals += 1

        if individual_trailing_stop:
            for t in tickers:
                if shares[t] > 0 and pd.notna(mark_close[t]):
                    entry_peak[t] = max(entry_peak[t] if pd.notna(entry_peak[t]) else mark_close[t], mark_close[t])
                    if mark_close[t] <= entry_peak[t]*(1-individual_stop_pct):
                        stop_blocked[t] = True

        equity = float(cash + (shares*mark_close.fillna(0)).sum())
        portfolio_peak = max(portfolio_peak, equity)
        if portfolio_trailing_stop and equity <= portfolio_peak*(1-portfolio_stop_pct):
            portfolio_blocked = True

        invested = float((shares*mark_close.fillna(0)).sum())
        exposure = invested/equity if equity > 0 else 0.0
        exposure_rows.append(exposure)
        eq_rows.append((dt, equity))

    eq = pd.Series(dict(eq_rows)).sort_index()
    st_res = metrics_from_equity(eq)
    st_res.update({
        "#Orders": len(orders),
        "#Rebalances": rebals,
        "Turnover [%]": 100*turnover/initial_cash,
        "AvgExposure [%]": 100*np.mean(exposure_rows),
        "AvgCash [%]": 100*(1-np.mean(exposure_rows)),
    })
    return st_res, eq, pd.DataFrame(orders)

def benchmark_nav(benchmark_px, start, end, initial_cash=1_000_000_000.0):
    x = benchmark_px.loc[(benchmark_px.index >= pd.Timestamp(start)) & (benchmark_px.index <= pd.Timestamp(end)), "Close"].dropna()
    if x.empty:
        return pd.Series(dtype=float)
    return initial_cash*x/x.iloc[0]

ACTIVE_MODEL_SPECS = {
    "DYN_EW_REGIME":      {"family":"EW",        "regime":True, "mask":True},
    "DYN_MPT_LW_REGIME":  {"family":"MPT_LW",    "regime":True, "mask":True},
    "DYN_MINVAR_REGIME":  {"family":"MINVAR_LW", "regime":True, "mask":True},
    "DYN_HRP_REGIME":     {"family":"HRP",       "regime":True, "mask":True},
    "DYN_BLEND50_REGIME": {"family":"BLEND50",   "regime":True, "mask":True},
}

def model_validation_score(stats, benchmark_stats):
    sh = stats.get("Sharpe", np.nan)
    sh = -3 if not np.isfinite(sh) else sh
    so = stats.get("Sortino", np.nan)
    so = -3 if not np.isfinite(so) else so
    ret = stats.get("Return [%]", -100)
    dd = stats.get("MaxDD [%]", -100)
    turnover = stats.get("Turnover [%]", 0)
    bret = benchmark_stats.get("Return [%]", 0)
    return 0.45*sh + 0.15*so + 0.012*(ret-bret) + 0.004*dd - 0.0005*turnover


# ------------------------------------------------------------------------------
# 8. SIDEBAR CONTROLS
# ------------------------------------------------------------------------------
st.sidebar.markdown("### ⚙️ Cấu Hình Chiến Lược & Dữ Liệu")

# Data Source
default_csv = None
for f in os.listdir('.'):
    if f.endswith('.csv') and ('hose' in f.lower() or 'stock' in f.lower() or 'data' in f.lower()):
        default_csv = f
        break

data_source_mode = st.sidebar.radio(
    "Nguồn dữ liệu:",
    ["📁 Dữ liệu mặc định trong repo" if default_csv else "Dữ liệu mẫu", "📤 Tải file mới lên (CSV / Excel)"]
)

uploaded_file = None
if "Tải file mới" in data_source_mode:
    uploaded_file = st.sidebar.file_uploader("Chọn file CSV hoặc Excel", type=["csv", "xlsx", "xls"])
    if uploaded_file is None:
        st.sidebar.info("💡 Chưa tải file. Ứng dụng sẽ dùng dữ liệu HOSE mẫu mặc định.")
        data_path = default_csv
    else:
        data_path = uploaded_file
else:
    data_path = default_csv

use_adjusted = st.sidebar.checkbox("Sử dụng Giá điều chỉnh (Adjusted Price)", value=True,
                                   help="Khuyến nghị cho chứng khoán Việt Nam để loại bỏ khoảng nhảy giá do chia cổ tức/thưởng.")

st.sidebar.markdown("---")
st.sidebar.markdown("### 💰 Tham Số Danh Mục")

initial_cash = st.sidebar.number_input("Vốn ban đầu (VND)", value=1_000_000_000, step=100_000_000, format="%d")
top_n = st.sidebar.slider("Số lượng cổ phiếu Top N", min_value=3, max_value=10, value=5, step=1)
cash_policy = st.sidebar.selectbox(
    "Chính sách tiền mặt (Cash Policy):",
    options=["cash40", "original", "full"],
    index=0,
    format_func=lambda x: {
        "cash40": "🛡️ cash40 — Tiền mặt tối đa 40% (Cổ phiếu >= 60%)",
        "original": "🔒 original — Giữ nguyên phòng vệ gốc (Cash tới 100%)",
        "full": "⚡ full — Đầu tư 100% cổ phiếu khi giải ngân"
    }[x],
    help="cash40 là cải tiến V4: khi thị trường xấu hoặc bị lọc, tỷ trọng tiền mặt tối đa không quá 40%."
)

with st.sidebar.expander("🛠️ Chi tiết Phí, Chu kỳ & Risk Overlay"):
    commission_pct = st.number_input("Phí giao dịch (%)", value=0.15, step=0.01) / 100.0
    slippage_pct = st.number_input("Trượt giá Slippage (%)", value=0.05, step=0.01) / 100.0
    selection_freq = st.selectbox("Chu kỳ tái chọn Top N", options=["Q", "M"], index=0, format_func=lambda x: "Hàng Quý (Q)" if x=="Q" else "Hàng Tháng (M)")
    rebalance_freq = st.selectbox("Chu kỳ tái cân bằng trọng số", options=["M"], index=0, format_func=lambda x: "Hàng Tháng (M)")
    use_market_regime = st.checkbox("Kích hoạt Market Regime (EMA200)", value=True)
    use_stock_mask = st.checkbox("Kích hoạt Stock Active Mask", value=True)
    enable_stop_loss = st.checkbox("Kích hoạt Trailing Stop Loss", value=False)
    stop_loss_pct = st.slider("Mức cắt lỗ Trailing Stop (%)", 5, 25, 12) / 100.0 if enable_stop_loss else 0.12

run_button = st.sidebar.button("🚀 Chạy Kiểm Định Chiến Lược", type="primary", use_container_width=True)


# ------------------------------------------------------------------------------
# 9. MAIN CONTENT & EXECUTION PIPELINE
# ------------------------------------------------------------------------------
st.markdown('<div class="main-title">📈 HOSE V4 Quantitative Portfolio Backtester</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Hệ thống kiểm định danh mục định lượng Walk-Forward không rò rỉ dữ liệu tương lai (No Future Leakage) • Mục tiêu tiền mặt tối đa 40%</div>', unsafe_allow_html=True)

if not data_path:
    st.error("❌ Không tìm thấy file dữ liệu mặc định và chưa có file nào được tải lên. Hãy tải lên file CSV hoặc Excel tại sidebar.")
    st.stop()

# Execution Trigger or Session State
if "backtest_results" not in st.session_state or run_button:
    with st.spinner("Đang chuẩn bị và nạp dữ liệu..."):
        try:
            data = load_and_standardize_data(data_path, use_adjusted=use_adjusted)
        except Exception as ex:
            st.error(f"Lỗi khi đọc file dữ liệu: {ex}")
            st.stop()

    status_ph = st.empty()
    progress_bar = st.progress(0.1)

    status_ph.info("1/5. Xây dựng Price Dictionary và Benchmark Proxy...")
    price_dict = build_price_dict(data)
    benchmark_aliases = {"VNINDEX", "VN-INDEX", "VNI", "^VNINDEX", "VNINDEXINDEX"}
    benchmark_ticker = next((t for t in price_dict if t in benchmark_aliases), None)
    if benchmark_ticker is None:
        benchmark_ticker = "__MARKET_PROXY__"
        price_dict[benchmark_ticker] = build_synthetic_market_proxy(price_dict)

    # Resolve splits
    dates = pd.DatetimeIndex(sorted(pd.to_datetime(data["Date"].unique())))
    n = len(dates)
    last_year = dates[-1].year
    last_year_dates = dates[dates.year == last_year]
    pre_last_year_count = int((dates < last_year_dates[0]).sum()) if len(last_year_dates) else 0

    if len(last_year_dates) >= 180 and pre_last_year_count >= 252 + 63:
        test_start, test_end = last_year_dates[0], last_year_dates[-1]
        pretest = dates[dates < test_start]
    else:
        test_n = max(100, min(252, int(round(0.20 * n))))
        test_start, test_end = dates[-test_n], dates[-1]
        pretest = dates[dates < test_start]

    val_n = max(63, min(252, int(round(0.25 * len(pretest)))))
    val_n = min(val_n, len(pretest) - 252)
    val_start, val_end = pretest[-val_n], pretest[-1]
    core_end = pretest[-val_n - 1]

    progress_bar.progress(0.3)
    status_ph.info("2/5. Tính toán chỉ báo kỹ thuật & TA Score (0–10) cho toàn Universe...")
    feature_store = {t: calculate_ta_features(px) for t, px in price_dict.items()}

    progress_bar.progress(0.5)
    status_ph.info("3/5. Cross-validation trên Core Train để khóa ngưỡng BUY/EXIT chung...")
    core_universe = build_universe(price_dict, core_end, benchmark_ticker)
    core_candidates = core_universe.loc[core_universe["Candidate"], "Ticker"].tolist()
    core_dates = dates[dates <= core_end]
    inner_n = max(63, min(126, int(0.30*len(core_dates))))
    inner_start = core_dates[-inner_n]
    threshold_folds = make_equal_folds(core_dates, inner_start, core_end, n_folds=3)
    best_buy, best_exit, threshold_trials = choose_global_ta_thresholds(feature_store, core_candidates, threshold_folds)

    progress_bar.progress(0.7)
    status_ph.info("4/5. Chạy Nested Walk-Forward trên Validation để chọn Champion Model...")
    val_selection, val_selection_details = build_selection_schedule(
        feature_store, price_dict, benchmark_ticker, feature_store[benchmark_ticker],
        dates, val_start, val_end, best_buy, best_exit, top_n=top_n, selection_freq=selection_freq
    )
    regime_val = build_market_regime(feature_store[benchmark_ticker])
    union_val_tickers = sorted({val_selection.loc[i, f"Top{j+1}"] for i in val_selection.index for j in range(top_n) if pd.notna(val_selection.loc[i, f"Top{j+1}"])})
    masks_val = build_stock_open_masks(feature_store, union_val_tickers, best_buy, best_exit)
    bnav_val = benchmark_nav(price_dict[benchmark_ticker], val_start, val_end, initial_cash=initial_cash)
    bstats_val = metrics_from_equity(bnav_val)

    val_rows = []
    for name, spec in ACTIVE_MODEL_SPECS.items():
        sched, _ = build_dynamic_weight_schedule(price_dict, val_selection, dates, val_start, val_end, spec["family"], allocation_freq=rebalance_freq, top_n=top_n)
        st_val, eq_val, _ = run_portfolio_engine_dynamic(
            price_dict, sched, dates, val_start, val_end,
            market_regime_open=regime_val["RiskOn_Open"],
            stock_active_open=masks_val,
            use_market_regime=(use_market_regime and spec["regime"]),
            use_stock_mask=(use_stock_mask and spec["mask"]),
            individual_trailing_stop=enable_stop_loss,
            individual_stop_pct=stop_loss_pct,
            initial_cash=initial_cash,
            commission=commission_pct,
            slippage=slippage_pct,
            cash_policy=cash_policy,
        )
        st_val["Model"] = name
        st_val["ValidationScore"] = model_validation_score(st_val, bstats_val)
        val_rows.append(st_val)

    val_res = pd.DataFrame(val_rows).set_index("Model")
    champion = val_res["ValidationScore"].idxmax()

    progress_bar.progress(0.9)
    status_ph.info(f"5/5. Chạy Out-of-Sample Final Test cho Champion ({champion}) và Benchmarks...")
    test_selection, test_selection_details = build_selection_schedule(
        feature_store, price_dict, benchmark_ticker, feature_store[benchmark_ticker],
        dates, test_start, test_end, best_buy, best_exit, top_n=top_n, selection_freq=selection_freq
    )
    regime_test = build_market_regime(feature_store[benchmark_ticker])
    union_test_tickers = sorted({test_selection.loc[i, f"Top{j+1}"] for i in test_selection.index for j in range(top_n) if pd.notna(test_selection.loc[i, f"Top{j+1}"])})
    masks_test = build_stock_open_masks(feature_store, union_test_tickers, best_buy, best_exit)

    bnav_test = benchmark_nav(price_dict[benchmark_ticker], test_start, test_end, initial_cash=initial_cash)
    bstats_test = metrics_from_equity(bnav_test)
    bstats_test["Model"] = "VNINDEX_or_MarketBenchmark"
    test_rows = [bstats_test]
    test_curves = {"VNINDEX_or_MarketBenchmark": bnav_test}
    test_orders_map = {}

    # Benchmark Static Buy & Hold
    start_dt = pd.Timestamp(test_start)
    top_bh = selected_tickers_asof(test_selection, start_dt, top_n=top_n)
    bh_sched = pd.DataFrame([equal_weights(top_bh)], index=[start_dt])
    bh_stats, bh_eq, bh_od = run_portfolio_engine_dynamic(
        price_dict, bh_sched, dates, test_start, test_end,
        use_market_regime=False, use_stock_mask=False, cash_policy="original",
        initial_cash=initial_cash, commission=commission_pct, slippage=slippage_pct
    )
    bh_stats["Model"] = "STATIC_EW_BH"
    test_rows.append(bh_stats)
    test_curves["STATIC_EW_BH"] = bh_eq
    test_orders_map["STATIC_EW_BH"] = bh_od

    # Dynamic EW no regime
    ew_sched, _ = build_dynamic_weight_schedule(price_dict, test_selection, dates, test_start, test_end, "EW", allocation_freq=rebalance_freq, top_n=top_n)
    ew_stats, ew_eq, ew_od = run_portfolio_engine_dynamic(
        price_dict, ew_sched, dates, test_start, test_end,
        use_market_regime=False, use_stock_mask=False, cash_policy="original",
        initial_cash=initial_cash, commission=commission_pct, slippage=slippage_pct
    )
    ew_stats["Model"] = "DYN_EW_NO_REGIME"
    test_rows.append(ew_stats)
    test_curves["DYN_EW_NO_REGIME"] = ew_eq
    test_orders_map["DYN_EW_NO_REGIME"] = ew_od

    # All active models
    for name, spec in ACTIVE_MODEL_SPECS.items():
        sched, _ = build_dynamic_weight_schedule(price_dict, test_selection, dates, test_start, test_end, spec["family"], allocation_freq=rebalance_freq, top_n=top_n)
        st_test, eq_test, od_test = run_portfolio_engine_dynamic(
            price_dict, sched, dates, test_start, test_end,
            market_regime_open=regime_test["RiskOn_Open"],
            stock_active_open=masks_test,
            use_market_regime=use_market_regime,
            use_stock_mask=use_stock_mask,
            individual_trailing_stop=enable_stop_loss,
            individual_stop_pct=stop_loss_pct,
            initial_cash=initial_cash,
            commission=commission_pct,
            slippage=slippage_pct,
            cash_policy=cash_policy,
        )
        label = "LOCKED_CHAMPION" if name == champion else "DIAG_" + name
        st_test["Model"] = label
        st_test["OriginalModel"] = name
        test_rows.append(st_test)
        test_curves[label] = eq_test
        test_orders_map[label] = od_test

    test_res = pd.DataFrame(test_rows).set_index("Model")
    if "STATIC_EW_BH" in test_res.index:
        test_res["ExcessCAGR_vs_StaticEW [pp]"] = test_res["CAGR [%]"] - test_res.loc["STATIC_EW_BH", "CAGR [%]"]
    bname = "VNINDEX_or_MarketBenchmark"
    test_res["ExcessCAGR_vs_Benchmark [pp]"] = test_res["CAGR [%]"] - test_res.loc[bname, "CAGR [%]"]
    test_res["DD_Improvement_vs_Benchmark [pp]"] = test_res["MaxDD [%]"] - test_res.loc[bname, "MaxDD [%]"]

    # Save to session
    st.session_state["backtest_results"] = {
        "data": data,
        "champion": champion,
        "best_buy": best_buy,
        "best_exit": best_exit,
        "threshold_trials": threshold_trials,
        "split_info": {
            "dates": dates,
            "core_end": core_end,
            "val_start": val_start,
            "val_end": val_end,
            "test_start": test_start,
            "test_end": test_end,
        },
        "val_selection": val_selection,
        "val_results": val_res,
        "test_selection": test_selection,
        "test_selection_details": test_selection_details,
        "test_results": test_res,
        "test_curves": test_curves,
        "test_orders_map": test_orders_map,
        "benchmark_ticker": benchmark_ticker,
    }

    progress_bar.progress(1.0)
    status_ph.success("✅ Đã hoàn tất kiểm định toàn diện thành công!")
    time.sleep(0.5)
    status_ph.empty()
    progress_bar.empty()

# Retrieve stored results
res = st.session_state["backtest_results"]
champion = res["champion"]
test_res = res["test_results"]
test_curves = res["test_curves"]
test_orders_map = res["test_orders_map"]
val_res = res["val_results"]
split_info = res["split_info"]


# ------------------------------------------------------------------------------
# 10. METRICS CARDS & TABS
# ------------------------------------------------------------------------------
champ_row = test_res.loc["LOCKED_CHAMPION"]
bench_row = test_res.loc["VNINDEX_or_MarketBenchmark"]
bh_row = test_res.loc["STATIC_EW_BH"]

c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-lbl">🏆 Champion Model</div>
        <div class="metric-val" style="font-size: 1.15rem; color: #1e3a8a;">{champion}</div>
        <span style="font-size:0.8rem; color:#10b981;">Đã khóa từ Validation</span>
    </div>
    """, unsafe_allow_html=True)
with c2:
    ret_diff = champ_row['Return [%]'] - bench_row['Return [%]']
    color = "#10b981" if ret_diff >= 0 else "#ef4444"
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-lbl">Lợi Nhuận Tích Lũy</div>
        <div class="metric-val" style="color: {color};">{champ_row['Return [%]']:.2f}%</div>
        <span style="font-size:0.8rem; color:#6b7280;">Benchmark: {bench_row['Return [%]']:.2f}%</span>
    </div>
    """, unsafe_allow_html=True)
with c3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-lbl">CAGR (Năm)</div>
        <div class="metric-val">{champ_row['CAGR [%]']:.2f}%</div>
        <span style="font-size:0.8rem; color:#6b7280;">B&H: {bh_row['CAGR [%]']:.2f}%</span>
    </div>
    """, unsafe_allow_html=True)
with c4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-lbl">Sharpe / Sortino</div>
        <div class="metric-val">{champ_row['Sharpe']:.2f} / {champ_row['Sortino']:.2f}</div>
        <span style="font-size:0.8rem; color:#6b7280;">Bmk Sharpe: {bench_row['Sharpe']:.2f}</span>
    </div>
    """, unsafe_allow_html=True)
with c5:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-lbl">Max Drawdown</div>
        <div class="metric-val" style="color: #ef4444;">{champ_row['MaxDD [%]']:.2f}%</div>
        <span style="font-size:0.8rem; color:#6b7280;">Tiền mặt TB: {champ_row['AvgCash [%]']:.1f}%</span>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Đường Cong NAV & Hiệu Suất",
    "💼 Lịch Sử Tái Cơ Cấu & Top N",
    "⚖️ Validation & Chọn Champion",
    "🛡️ Audit No Future Leakage",
    "📥 Xuất Báo Cáo & Dữ Liệu"
])

# ------------------------------------------------------------------------------
# TAB 1: BIỂU ĐỒ NAV & HIỆU SUẤT
# ------------------------------------------------------------------------------
with tab1:
    st.markdown("#### 📈 Diễn Biến Đường Cong NAV Chuẩn Hóa (Final Test)")

    fig_nav = go.Figure()
    # Benchmark
    if "VNINDEX_or_MarketBenchmark" in test_curves:
        s = test_curves["VNINDEX_or_MarketBenchmark"]
        fig_nav.add_trace(go.Scatter(
            x=s.index, y=s/s.iloc[0], name="📊 VN-Index / Market Benchmark",
            line=dict(color="#111827", width=2.5, dash="solid")
        ))
    if "STATIC_EW_BH" in test_curves:
        s = test_curves["STATIC_EW_BH"]
        fig_nav.add_trace(go.Scatter(
            x=s.index, y=s/s.iloc[0], name="📌 STATIC_EW_BH (Equal Weight - Buy & Hold)",
            line=dict(color="#3b82f6", width=2, dash="dash")
        ))
    if "DYN_EW_NO_REGIME" in test_curves:
        s = test_curves["DYN_EW_NO_REGIME"]
        fig_nav.add_trace(go.Scatter(
            x=s.index, y=s/s.iloc[0], name="⚡ DYN_EW_NO_REGIME (Không hạ tỷ trọng)",
            line=dict(color="#f97316", width=2, dash="dot")
        ))
    if "LOCKED_CHAMPION" in test_curves:
        s = test_curves["LOCKED_CHAMPION"]
        fig_nav.add_trace(go.Scatter(
            x=s.index, y=s/s.iloc[0], name=f"🏆 LOCKED_CHAMPION ({champion})",
            line=dict(color="#10b981", width=3.5)
        ))

    # Diagnostics
    for k in test_curves:
        if k.startswith("DIAG_"):
            s = test_curves[k]
            clean_lbl = k.replace("DIAG_", "")
            fig_nav.add_trace(go.Scatter(
                x=s.index, y=s/s.iloc[0], name=f"▫️ {clean_lbl}",
                line=dict(width=1.2), opacity=0.65
            ))

    fig_nav.update_layout(
        xaxis_title="Thời gian",
        yaxis_title="Tỷ lệ NAV Chuẩn Hóa (Bắt đầu = 1.0)",
        hovermode="x unified",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=60, b=20),
        height=520,
    )
    st.plotly_chart(fig_nav, use_container_width=True)

    # Underwater Drawdown Chart
    st.markdown("#### 🌊 Biểu Đồ Sụt Giảm Tài Sản (Underwater Drawdown)")
    fig_dd = go.Figure()
    if "VNINDEX_or_MarketBenchmark" in test_curves:
        s = test_curves["VNINDEX_or_MarketBenchmark"]
        dd_bench = (s / s.cummax() - 1) * 100
        fig_dd.add_trace(go.Scatter(
            x=dd_bench.index, y=dd_bench, name="VNINDEX Drawdown",
            fill='tozeroy', line=dict(color="#9ca3af", width=1.5), opacity=0.4
        ))
    if "LOCKED_CHAMPION" in test_curves:
        s = test_curves["LOCKED_CHAMPION"]
        dd_champ = (s / s.cummax() - 1) * 100
        fig_dd.add_trace(go.Scatter(
            x=dd_champ.index, y=dd_champ, name=f"Champion ({champion}) Drawdown",
            fill='tozeroy', line=dict(color="#ef4444", width=2), opacity=0.6
        ))
    fig_dd.update_layout(
        xaxis_title="Thời gian",
        yaxis_title="Drawdown (%)",
        hovermode="x unified",
        template="plotly_white",
        height=320,
        margin=dict(l=20, r=20, t=20, b=20)
    )
    st.plotly_chart(fig_dd, use_container_width=True)

    # Bảng chi tiết Metrics
    st.markdown("#### 📋 Bảng So Sánh Chỉ Số Toàn Diện Giữa Các Mô Hình")
    cols_display = [
        "Return [%]", "CAGR [%]", "Sharpe", "Sortino", "MaxDD [%]",
        "Calmar", "Turnover [%]", "AvgExposure [%]", "AvgCash [%]",
        "ExcessCAGR_vs_Benchmark [pp]", "DD_Improvement_vs_Benchmark [pp]"
    ]
    formatted_df = test_res[[c for c in cols_display if c in test_res.columns]].copy()
    st.dataframe(formatted_df.style.format("{:.2f}", na_rep="-").background_gradient(
        subset=["Return [%]", "CAGR [%]", "Sharpe"], cmap="Greens"
    ).background_gradient(subset=["MaxDD [%]"], cmap="Reds_r"), use_container_width=True)


# ------------------------------------------------------------------------------
# TAB 2: LỊCH SỬ PHÂN BỔ & TOP N
# ------------------------------------------------------------------------------
with tab2:
    st.markdown("#### 🗓️ Lịch Sử Tái Lựa Chọn Danh Mục Top N (Walk-Forward Schedule)")
    st.markdown("""
    > Mỗi kỳ tái lựa chọn danh mục, mô hình chỉ được dùng dữ liệu tính đến phiên `t-1` (AsOfDate).
    > Các cổ phiếu lọt vào danh mục phải vượt qua cả bộ lọc **Thanh khoản**, **TA Score**, **Risk Gate** và **Đa dạng hóa Correlation**.
    """)

    test_sel = res["test_selection"]
    st.dataframe(test_sel, use_container_width=True)

    st.markdown("#### 🔍 Xem Chi Tiết Điểm Số Cổ Phiếu Tại Kỳ Lựa Chọn")
    chosen_eff = st.selectbox("Chọn ngày hiệu lực (Effective Date):", options=test_sel["EffectiveDate"].dt.strftime("%Y-%m-%d").tolist())
    if chosen_eff:
        chosen_ts = pd.Timestamp(chosen_eff)
        if chosen_ts in res["test_selection_details"]:
            locked_detail = res["test_selection_details"][chosen_ts]["locked"]
            cols_show = ["Ticker", "SelectedOrder", "FinalScore", "TA_Composite", "TrailingSharpe", "RelativeStrength126", "Vol63", "RSI"]
            st.dataframe(locked_detail[[c for c in cols_show if c in locked_detail.columns]], use_container_width=True)

    # Lệnh giao dịch
    st.markdown("#### 🛒 Lịch Sử Khớp Lệnh Của Champion (Orders Log)")
    champ_orders = test_orders_map.get("LOCKED_CHAMPION", pd.DataFrame())
    if not champ_orders.empty:
        st.dataframe(champ_orders.head(100), use_container_width=True)
    else:
        st.info("Không có lệnh giao dịch phát sinh.")


# ------------------------------------------------------------------------------
# TAB 3: VALIDATION & CHỌN CHAMPION
# ------------------------------------------------------------------------------
with tab3:
    st.markdown("#### ⚖️ Kết Quả Đua Mô Hình Trên Tập Validation")
    st.markdown("""
    **Quy tắc:** Chỉ tập Validation được phép quyết định mô hình Champion.
    Điểm `ValidationScore` ưu tiên Sharpe, Sortino, Excess Return và phạt nặng Turnover cao.
    Sau khi chọn được Champion, mô hình này bị khóa cứng và đưa sang Final Test để đánh giá khách quan.
    """)

    val_table = val_res.copy()
    st.dataframe(val_table.style.format("{:.3f}", na_rep="-").highlight_max(subset=["ValidationScore"], color="#bbf7d0"), use_container_width=True)

    st.markdown("#### 🎯 Quá Trình Tối Ưu Ngưỡng BUY/EXIT Trên Core Train")
    st.markdown(f"""
    - **BUY Threshold tối ưu:** `{res['best_buy']}`
    - **EXIT Threshold tối ưu:** `{res['best_exit']}`
    """)
    if "threshold_trials" in res and not res["threshold_trials"].empty:
        st.dataframe(res["threshold_trials"].head(10).style.format("{:.3f}", na_rep="-"), use_container_width=True)


# ------------------------------------------------------------------------------
# TAB 4: AUDIT RÒ RỈ DỮ LIỆU (LEAKAGE AUDIT)
# ------------------------------------------------------------------------------
with tab4:
    st.markdown("#### 🛡️ Bảng Audit Chống Rò Rỉ Thông Tin Tương Lai (No Future Leakage)")
    st.markdown("""
    Trong nghiên cứu định lượng tài chính, lỗi **Look-ahead bias / Future Leakage** là nguyên nhân số một khiến backtest rất đẹp nhưng thực chiến thất bại.
    Hệ thống HOSE V4 tuân thủ nghiêm ngặt các ranh giới thông tin:
    """)

    sp = res["split_info"]
    leakage_df = pd.DataFrame([
        ["Khóa ngưỡng TA BUY/EXIT", str(sp["core_end"].date()), False, "Chỉ tối ưu trên tập Core Train qua Cross-Validation."],
        ["Khóa họ mô hình Champion", str(sp["val_end"].date()), False, "Chọn mô hình tốt nhất trên Validation, khóa cứng trước khi sang Test."],
        ["Tái chọn Top N hàng quý", "Phiên t-1", False, "Mỗi ngày EffectiveDate chỉ dùng dữ liệu đóng cửa phiên trước đó."],
        ["Tối ưu trọng số hàng tháng", "Phiên t-1", False, "Ma trận Covariance (Ledoit-Wolf) chỉ tính bằng giá đến t-1."],
        ["Market Regime & Tín hiệu lệnh", "Phiên t-1 Close -> t Open", False, "Tín hiệu đóng nến hôm trước chỉ được khớp tại giá Mở cửa hôm sau."],
        ["Đánh giá Out-of-Sample Final Test", str(sp["test_end"].date()), True, "Tập Test chỉ dùng để đo lường, tuyệt đối không chỉnh sửa tham số."]
    ], columns=["Công đoạn nghiên cứu", "Thời điểm chốt thông tin (Point-in-time)", "Dùng tập Test để chỉnh tham số?", "Quy tắc kiểm soát"])

    st.dataframe(leakage_df, use_container_width=True)


# ------------------------------------------------------------------------------
# TAB 5: XUẤT BÁO CÁO & DỮ LIỆU
# ------------------------------------------------------------------------------
with tab5:
    st.markdown("#### 📥 Xuất Báo Cáo Kết Quả Định Lượng Hoàn Chỉnh")
    st.write("Tải xuống toàn bộ bảng dữ liệu, lịch sử tái cơ cấu, kết quả đánh giá các mô hình dưới dạng file Excel chuẩn hoặc file CSV:")

    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
        test_res.reset_index().to_excel(writer, sheet_name="FinalTestResults", index=False)
        val_res.reset_index().to_excel(writer, sheet_name="ValidationResults", index=False)
        res["test_selection"].to_excel(writer, sheet_name="TestSelectionHistory", index=False)
        res["val_selection"].to_excel(writer, sheet_name="ValSelectionHistory", index=False)
        if "threshold_trials" in res and not res["threshold_trials"].empty:
            res["threshold_trials"].to_excel(writer, sheet_name="CoreThresholdTrials", index=False)
        leakage_df.to_excel(writer, sheet_name="LeakageAudit", index=False)
        champ_orders = test_orders_map.get("LOCKED_CHAMPION", pd.DataFrame())
        if not champ_orders.empty:
            champ_orders.to_excel(writer, sheet_name="ChampionOrders", index=False)

    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        st.download_button(
            label="📊 Tải Báo Cáo Excel Đầy Đủ (.xlsx)",
            data=excel_buffer.getvalue(),
            file_name=f"HOSE_V4_Backtest_Report_{time.strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )

    with col_dl2:
        # Export daily NAV curves to CSV
        curves_df = pd.DataFrame(test_curves)
        csv_buffer = curves_df.to_csv(index=True, encoding="utf-8-sig")
        st.download_button(
            label="📈 Tải Chuỗi NAV Hàng Ngày (.csv)",
            data=csv_buffer,
            file_name=f"HOSE_V4_Daily_NAV_Curves_{time.strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )

    st.success("Tất cả các tệp đều được xuất với mã hóa UTF-8 chuẩn xác, sẵn sàng cho việc phân tích và báo cáo.")
