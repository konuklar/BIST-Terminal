# -*- coding: utf-8 -*-
"""
📊 ADVANCED BIST Risk Budgeting System (Streamlit Cloud)
Enhanced with:
- BIST100 benchmark integration (Yahoo Finance only)
- Benchmark-relative (active) risk contributions panel
- Stress scenarios module (FX shock / Rate shock proxies from Yahoo Finance)
- PyPortfolioOpt portfolio strategies (Mean-Variance / HRP / Black-Litterman / CLA where available)
- VaR / CVaR / ES + rolling risk contributions
- Professional visualizations + robust data alignment + forward-fill only (NO synthetic data)

Footer statement:
"The Quantitative Analysis Performed by LabGen25@Istanbul by Murat KONUKLAR 2026"
"""

from __future__ import annotations

import math
import warnings
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, date
from io import BytesIO
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
from scipy.optimize import minimize

warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("yfinance").setLevel(logging.ERROR)

# ------------------------------------------------------------
# Optional: PyPortfolioOpt
# ------------------------------------------------------------
try:
    from pypfopt import (
        EfficientFrontier,
        risk_models,
        expected_returns,
        BlackLittermanModel,
        CLA,
    )
    from pypfopt.hierarchical_risk_parity import HRPOpt
    PYPFOPT_AVAILABLE = True
except Exception:
    PYPFOPT_AVAILABLE = False

# ------------------------------------------------------------
# Page configuration
# ------------------------------------------------------------
st.set_page_config(
    page_title="BIST Advanced Risk Budgeting",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------
# Custom CSS (keep layout feel)
# ------------------------------------------------------------
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.3rem;
        color: #1E3A8A;
        font-weight: 800;
        margin-bottom: 0.25rem;
    }
    .sub-header {
        font-size: 1.35rem;
        color: #2563EB;
        font-weight: 700;
        margin-top: 0.8rem;
        margin-bottom: 0.25rem;
    }
    .data-source-badge {
        background-color: #1E3A8A;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 1rem;
        font-size: 0.85rem;
        display: inline-block;
        margin: 0.2rem 0 1rem 0;
    }
    .note-box {
        background-color: #F3F4F6;
        border: 1px solid #E5E7EB;
        padding: 0.75rem 1rem;
        border-radius: 0.6rem;
    }
    .small-muted {
        color: #6B7280;
        font-size: 0.85rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------
# Universe (BIST 50 constituents - static list, ASCII tickers)
# TRALTIN removed as requested; ASTOR included as requested.
# ------------------------------------------------------------
BIST50_TICKERS = [
    "AEFES.IS", "AKBNK.IS", "ALARK.IS", "ARCLK.IS", "ASELS.IS",
    "ASTOR.IS", "BIMAS.IS", "BRSAN.IS", "BTCIM.IS", "CCOLA.IS",
    "CIMSA.IS", "DOAS.IS", "DOHOL.IS", "DSTKF.IS", "EKGYO.IS",
    "ENKAI.IS", "EREGL.IS", "FROTO.IS", "GARAN.IS", "GUBRF.IS",
    "HALKB.IS", "HEKTS.IS", "ISCTR.IS", "KCHOL.IS", "KONTR.IS",
    "KRDMD.IS", "KUYAS.IS", "MAVI.IS", "MGROS.IS", "MIATK.IS",
    "OYAKC.IS", "PASEU.IS", "PETKM.IS", "PGSUS.IS", "SAHOL.IS",
    "SASA.IS", "SISE.IS", "SOKM.IS", "TAVHL.IS", "TCELL.IS",
    "THYAO.IS", "TOASO.IS", "TRMET.IS", "TSKB.IS", "TTKOM.IS",
    "TUPRS.IS", "ULKER.IS", "VAKBN.IS", "VESTL.IS", "YKBNK.IS",
]

# Benchmark candidates (Yahoo Finance only)
BENCHMARK_CANDIDATES = ["^XU100", "XU100.IS"]  # try both, keep what works

# Stress factor proxies (Yahoo Finance only)
FX_FACTOR = "TRY=X"     # USD/TRY (Yahoo)
RATE_FACTOR = "^TNX"    # US 10Y yield proxy (Yahoo). Turkey 10Y is not reliably available on Yahoo.

# Sector map (for caps). Extend anytime; unknown -> "Other".
SECTOR_MAP = {
    "AKBNK.IS": "Banking", "GARAN.IS": "Banking", "HALKB.IS": "Banking", "ISCTR.IS": "Banking",
    "VAKBN.IS": "Banking", "YKBNK.IS": "Banking", "TSKB.IS": "Banking",
    "KCHOL.IS": "Holding", "SAHOL.IS": "Holding", "ALARK.IS": "Holding",
    "BIMAS.IS": "Retail", "MGROS.IS": "Retail", "SOKM.IS": "Retail",
    "TCELL.IS": "Telecom", "TTKOM.IS": "Telecom",
    "THYAO.IS": "Aviation", "PGSUS.IS": "Aviation", "TAVHL.IS": "Aviation",
    "TUPRS.IS": "Energy", "PETKM.IS": "Petrochemical", "EREGL.IS": "Iron&Steel", "KRDMD.IS": "Iron&Steel",
    "ASELS.IS": "Defense", "KONTR.IS": "Technology", "MIATK.IS": "Technology", "ASTOR.IS": "Technology",
    "ULKER.IS": "Consumer", "CCOLA.IS": "Consumer", "AEFES.IS": "Consumer",
    "SASA.IS": "Chemicals", "GUBRF.IS": "Chemicals", "HEKTS.IS": "Chemicals",
    "ARCLK.IS": "Industrial", "FROTO.IS": "Automotive", "TOASO.IS": "Automotive",
    "BRSAN.IS": "Industrial", "OYAKC.IS": "Industrial", "CIMSA.IS": "Industrial", "BTCIM.IS": "Industrial",
    "SISE.IS": "Industrial",
    "DOAS.IS": "Industrial", "DOHOL.IS": "Holding", "EKGYO.IS": "RealEstate",
    "ENKAI.IS": "Industrial", "KUYAS.IS": "Other", "DSTKF.IS": "Other",
    "MAVI.IS": "Retail", "TRMET.IS": "Other", "PASEU.IS": "Other",
}

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def _tz_strip_index(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
        df = df.copy()
        df.index = df.index.tz_localize(None)
    return df

def _coerce_numeric_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out

def sanitize_for_excel(obj: Any) -> pd.DataFrame:
    if obj is None:
        return pd.DataFrame()
    if isinstance(obj, pd.Series):
        obj = obj.to_frame()
    if isinstance(obj, dict):
        try:
            obj = pd.DataFrame([obj])
        except Exception:
            obj = pd.DataFrame({"value": [str(obj)]})
    if not isinstance(obj, pd.DataFrame):
        return pd.DataFrame({"value": [str(obj)]})

    df = obj.copy()

    for col in df.columns:
        if pd.api.types.is_datetime64tz_dtype(df[col]):
            df[col] = df[col].dt.tz_localize(None)
        if pd.api.types.is_period_dtype(df[col]) or pd.api.types.is_timedelta64_dtype(df[col]):
            df[col] = df[col].astype(str)

    df = df.replace([np.inf, -np.inf], np.nan)
    df = _tz_strip_index(df)

    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].map(lambda x: "" if pd.isna(x) else str(x))

    return df

def to_excel_bytes(sheets: Dict[str, Any]) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        for name, obj in sheets.items():
            df = sanitize_for_excel(obj)
            if df is None or df.empty:
                continue
            df.to_excel(writer, sheet_name=str(name)[:31], index=False)
    return output.getvalue()

# ------------------------------------------------------------
# Yahoo Finance fetcher (NO synthetic data; forward-fill only)
# ------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_yahoo_prices(
    tickers: List[str],
    start: date,
    end: date,
) -> Tuple[pd.DataFrame, List[str], Dict[str, str]]:
    start_str = pd.to_datetime(start).strftime("%Y-%m-%d")
    end_str = pd.to_datetime(end).strftime("%Y-%m-%d")
    meta = {"mode": "batch", "note": ""}

    def extract_close(data: pd.DataFrame) -> pd.DataFrame:
        if data is None or data.empty:
            return pd.DataFrame()
        if isinstance(data.columns, pd.MultiIndex):
            lvl0 = data.columns.get_level_values(0)
            if "Close" in lvl0:
                closes = data["Close"]
            elif "Adj Close" in lvl0:
                closes = data["Adj Close"]
            else:
                try:
                    closes = data.xs("Close", axis=1, level=-1)
                except Exception:
                    return pd.DataFrame()
            if isinstance(closes, pd.Series):
                closes = closes.to_frame()
            return closes
        if "Close" in data.columns:
            return pd.DataFrame({tickers[0]: data["Close"]})
        if "Adj Close" in data.columns:
            return pd.DataFrame({tickers[0]: data["Adj Close"]})
        return pd.DataFrame()

    # Batch first
    try:
        raw = yf.download(
            tickers=tickers,
            start=start_str,
            end=end_str,
            interval="1d",
            group_by="column",
            auto_adjust=True,
            progress=False,
            threads=True,
            timeout=30,
        )
        closes = extract_close(raw)
    except Exception as e:
        closes = pd.DataFrame()
        meta["note"] = f"Batch download failed: {e}"

    # Per-ticker fallback (still Yahoo)
    if closes.empty:
        meta["mode"] = "per_ticker"
        series = {}
        for t in tickers:
            try:
                d = yf.download(
                    tickers=t,
                    start=start_str,
                    end=end_str,
                    interval="1d",
                    auto_adjust=True,
                    progress=False,
                    threads=False,
                    timeout=30,
                )
                if d is None or d.empty:
                    continue
                if "Close" in d.columns:
                    series[t] = d["Close"]
                elif "Adj Close" in d.columns:
                    series[t] = d["Adj Close"]
            except Exception:
                continue
        closes = pd.DataFrame(series)

    closes = closes.sort_index()
    closes = _tz_strip_index(closes)
    closes = _coerce_numeric_df(closes)

    dropped = [t for t in tickers if t not in closes.columns or closes[t].dropna().empty]
    closes = closes.drop(columns=dropped, errors="ignore")
    return closes, dropped, meta

def clean_prices_forward_fill(
    prices: pd.DataFrame,
    min_obs: int,
    max_missing_frac: float,
    ffill_limit: int,
) -> Tuple[pd.DataFrame, List[str], Dict[str, Any]]:
    info = {"min_obs": min_obs, "max_missing_frac": max_missing_frac, "ffill_limit": ffill_limit}
    if prices is None or prices.empty:
        return pd.DataFrame(), [], info

    p = prices.copy()

    # Missing fraction BEFORE ffill (for reporting)
    pre_miss = p.isna().mean()

    # Forward-fill only
    p = p.ffill(limit=ffill_limit)

    # Drop all-NaN columns
    p = p.dropna(axis=1, how="all")

    dropped = []

    # Drop assets that were too missing originally
    bad_missing = pre_miss[pre_miss > max_missing_frac].index.tolist()
    dropped.extend(bad_missing)
    p = p.drop(columns=bad_missing, errors="ignore")

    # Drop assets with too few observations after ffill
    counts = p.count()
    bad_counts = counts[counts < min_obs].index.tolist()
    dropped.extend(bad_counts)
    p = p.drop(columns=bad_counts, errors="ignore")

    # Strict intersection: keep rows where all assets have prices
    p = p.dropna(how="any")

    info.update(
        {
            "start_effective": str(p.index.min()) if not p.empty else None,
            "end_effective": str(p.index.max()) if not p.empty else None,
            "n_assets": int(p.shape[1]),
            "n_days": int(p.shape[0]),
        }
    )
    return p, sorted(list(set(dropped))), info

def returns_from_prices(prices: pd.DataFrame) -> pd.DataFrame:
    r = prices.pct_change().dropna(how="any")
    return _coerce_numeric_df(r)

# ------------------------------------------------------------
# Risk analytics
# ------------------------------------------------------------
@dataclass
class RiskResults:
    risk_table: pd.DataFrame
    portfolio_metrics: Dict[str, Any]
    cov_annual: pd.DataFrame
    port_returns: pd.Series

class RiskEngine:
    @staticmethod
    def portfolio_vol(cov_annual: np.ndarray, w: np.ndarray) -> float:
        return float(np.sqrt(np.maximum(w @ cov_annual @ w, 0.0)))

    @staticmethod
    def risk_contributions(cov_annual: pd.DataFrame, w: np.ndarray, tickers: List[str]) -> pd.DataFrame:
        cov = cov_annual.values
        vol = RiskEngine.portfolio_vol(cov, w)
        if vol <= 0:
            vol = 1e-12
        mrc = (cov @ w) / vol
        crc = w * mrc
        pct = (crc / vol) * 100.0
        indiv_vol = np.sqrt(np.diag(cov))
        df = pd.DataFrame({
            "Symbol": tickers,
            "Sector": [SECTOR_MAP.get(t, "Other") for t in tickers],
            "Weight": w,
            "Individual_Volatility": indiv_vol,
            "Marginal_Risk_Contribution": mrc,
            "Component_Risk": crc,
            "Risk_Contribution_%": pct
        }).sort_values("Risk_Contribution_%", ascending=False).reset_index(drop=True)
        df["Risk_Rank"] = np.arange(1, len(df) + 1)
        return df

    @staticmethod
    def tail_var_es(returns: pd.Series, alpha: float) -> Dict[str, float]:
        var = float(returns.quantile(alpha))
        es = float(returns[returns <= var].mean()) if (returns <= var).any() else float("nan")
        return {"VaR": var, "ES": es}

    @staticmethod
    def compute(returns: pd.DataFrame, weights: np.ndarray) -> RiskResults:
        tickers = returns.columns.tolist()
        cov_annual = returns.cov() * 252.0
        cov_annual = cov_annual.fillna(0.0)
        port_returns = (returns * weights).sum(axis=1)

        port_var = float(weights @ cov_annual.values @ weights)
        port_vol = float(np.sqrt(np.maximum(port_var, 0.0)))

        indiv_vol = np.sqrt(np.diag(cov_annual.values))
        weighted_avg_vol = float(np.sum(weights * indiv_vol)) if len(indiv_vol) else float("nan")
        diversification_ratio = float(weighted_avg_vol / port_vol) if port_vol > 0 else float("nan")

        risk_df = RiskEngine.risk_contributions(cov_annual, weights, tickers)

        tail95 = RiskEngine.tail_var_es(port_returns, 0.05)
        tail99 = RiskEngine.tail_var_es(port_returns, 0.01)

        portfolio_metrics = {
            "volatility": port_vol,
            "variance": port_var,
            "n_assets": len(tickers),
            "diversification_ratio": diversification_ratio,
            "max_risk_contrib": float(risk_df.iloc[0]["Risk_Contribution_%"]) if len(risk_df) else float("nan"),
            "max_risk_asset": str(risk_df.iloc[0]["Symbol"]) if len(risk_df) else "",
            "VaR_95_1d": tail95["VaR"],
            "ES_95_1d": tail95["ES"],
            "VaR_99_1d": tail99["VaR"],
            "ES_99_1d": tail99["ES"],
        }
        return RiskResults(risk_table=risk_df, portfolio_metrics=portfolio_metrics,
                           cov_annual=cov_annual, port_returns=port_returns)

# Rolling RC%
@st.cache_data(ttl=3600, show_spinner=False)
def rolling_risk_contributions(returns: pd.DataFrame, weights: Tuple[float, ...], window: int) -> pd.DataFrame:
    w = np.array(weights, dtype=float)
    tickers = returns.columns.tolist()
    if len(returns) < window + 5:
        return pd.DataFrame()
    out = []
    for i in range(window, len(returns) + 1):
        sub = returns.iloc[i - window:i]
        cov = sub.cov().values * 252.0
        port_var = float(w @ cov @ w)
        port_vol = float(np.sqrt(np.maximum(port_var, 0.0)))
        if port_vol <= 0:
            continue
        mrc = (cov @ w) / port_vol
        crc = w * mrc
        pct = (crc / port_vol) * 100.0
        out.append(pct)
    idx = returns.index[window - 1:]
    rr = pd.DataFrame(out, index=idx[:len(out)], columns=tickers)
    rr = rr.replace([np.inf, -np.inf], np.nan).dropna(how="all")
    return rr

# Active risk contributions
def active_risk_contributions(asset_returns: pd.DataFrame, weights: np.ndarray, benchmark_returns: pd.Series) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    common = asset_returns.index.intersection(benchmark_returns.index)
    R = asset_returns.loc[common].copy()
    b = benchmark_returns.loc[common].copy()
    port = (R * weights).sum(axis=1)
    active = port - b

    cov = pd.concat([R, active.rename("ACTIVE")], axis=1).cov() * 252.0
    cov_assets_active = cov.loc[R.columns, "ACTIVE"].values

    active_var = float(active.var() * 252.0)
    active_vol = float(np.sqrt(np.maximum(active_var, 0.0)))
    if active_vol <= 0:
        active_vol = 1e-12

    mrc = cov_assets_active / active_vol
    crc = weights * mrc
    pct = (crc / active_vol) * 100.0

    df = pd.DataFrame({"Symbol": R.columns, "Active_MRC": mrc, "Active_CRC": crc, "Active_Risk_%": pct})
    df = df.sort_values("Active_Risk_%", ascending=False).reset_index(drop=True)

    meta = {
        "tracking_error": active_vol,
        "active_mean": float(active.mean() * 252.0),
        "information_ratio": (float(active.mean() * 252.0) / active_vol) if active_vol > 0 else np.nan,
    }
    return df, meta

# Stress betas (FX + Rate proxy)
def estimate_factor_betas(asset_returns: pd.DataFrame, fx_returns: pd.Series, rate_changes: pd.Series, min_obs: int = 100) -> pd.DataFrame:
    common = asset_returns.index.intersection(fx_returns.index).intersection(rate_changes.index)
    R = asset_returns.loc[common]
    FX = fx_returns.loc[common]
    RT = rate_changes.loc[common]

    rows = []
    for t in R.columns:
        y = R[t].dropna()
        X = pd.concat([FX.rename("FX"), RT.rename("RATE")], axis=1).loc[y.index].dropna()
        y = y.loc[X.index]
        if len(y) < min_obs:
            rows.append((t, np.nan, np.nan, len(y)))
            continue
        Xmat = np.column_stack([np.ones(len(X)), X["FX"].values, X["RATE"].values])
        coef, *_ = np.linalg.lstsq(Xmat, y.values, rcond=None)
        rows.append((t, float(coef[1]), float(coef[2]), int(len(y))))
    return pd.DataFrame(rows, columns=["Symbol", "Beta_FX", "Beta_Rate", "Obs"])

def stress_scenario_impact(weights: np.ndarray, betas: pd.DataFrame, fx_shock: float, rate_shock_bp: float) -> Dict[str, Any]:
    b = betas.set_index("Symbol")
    # bp -> yield decimal (100bp = 0.01)
    rate_shock_dec = rate_shock_bp / 10000.0
    beta_fx = b["Beta_FX"].fillna(0.0).values
    beta_rt = b["Beta_Rate"].fillna(0.0).values
    shock_vec = beta_fx * fx_shock + beta_rt * rate_shock_dec
    impact = float(np.sum(weights * shock_vec))
    contrib = weights * shock_vec
    df = pd.DataFrame({"Symbol": b.index, "Shock_Return": shock_vec, "Contribution": contrib}).sort_values("Contribution", ascending=False)
    return {"portfolio_impact": impact, "details": df, "fx_shock": fx_shock, "rate_shock_bp": rate_shock_bp}

# ------------------------------------------------------------
# Optimization
# ------------------------------------------------------------
def clip_and_renormalize(w: np.ndarray, max_w: float) -> np.ndarray:
    w = np.clip(w, 0.0, max_w)
    s = w.sum()
    return w / s if s > 0 else np.ones_like(w) / len(w)

def apply_sector_caps_post(w: np.ndarray, tickers: List[str], sector_caps: Dict[str, float]) -> np.ndarray:
    if not sector_caps:
        return w
    w = w.copy()
    for _ in range(10):
        sector_sums = {}
        for i, t in enumerate(tickers):
            sec = SECTOR_MAP.get(t, "Other")
            sector_sums[sec] = sector_sums.get(sec, 0.0) + w[i]
        violated = [(sec, sector_sums[sec], cap) for sec, cap in sector_caps.items() if sector_sums.get(sec, 0.0) > cap + 1e-9]
        if not violated:
            break
        for sec, actual, cap in violated:
            idx = [i for i, t in enumerate(tickers) if SECTOR_MAP.get(t, "Other") == sec]
            if not idx or actual <= 0:
                continue
            scale = cap / actual
            w[idx] *= scale
            leftover = 1.0 - w.sum()
            other_idx = [i for i in range(len(tickers)) if i not in idx]
            if other_idx:
                base = w[other_idx].sum()
                if base > 0:
                    w[other_idx] += (w[other_idx] / base) * leftover
        if w.sum() > 0:
            w /= w.sum()
    return w

def optimize_portfolio(prices: pd.DataFrame, returns: pd.DataFrame, method: str, max_weight: float, sector_caps: Dict[str, float]) -> Tuple[np.ndarray, str]:
    tickers = returns.columns.tolist()
    n = len(tickers)

    if method == "Equal Weight":
        return np.ones(n) / n, ""

    if method == "Risk Parity (SciPy)":
        cov = returns.cov().values * 252.0

        def obj(w):
            w = clip_and_renormalize(w, max_weight)
            port_var = float(w @ cov @ w)
            port_vol = math.sqrt(max(port_var, 1e-18))
            mrc = (cov @ w) / port_vol
            rc = w * mrc
            target = port_vol / n
            return float(np.sum((rc - target) ** 2))

        x0 = np.ones(n) / n
        cons = {"type": "eq", "fun": lambda x: np.sum(x) - 1.0}
        bnds = [(0.0, max_weight) for _ in range(n)]
        res = minimize(obj, x0, method="SLSQP", bounds=bnds, constraints=cons, options={"ftol": 1e-10, "maxiter": 2000})
        w = res.x if res.success else x0
        w = clip_and_renormalize(w, max_weight)
        w = apply_sector_caps_post(w, tickers, sector_caps)
        return w, "Risk parity via SciPy (sector caps post-processed)."

    # PyPortfolioOpt routes
    if not PYPFOPT_AVAILABLE:
        return np.ones(n) / n, "PyPortfolioOpt missing; defaulted to equal weight."

    mu = expected_returns.mean_historical_return(prices, frequency=252)
    S = risk_models.CovarianceShrinkage(prices).ledoit_wolf()

    try:
        if method in ["Min Volatility", "Max Sharpe", "Efficient Return", "Efficient Risk", "Black-Litterman"]:
            if method == "Black-Litterman":
                market_caps = pd.Series(1.0, index=tickers)
                bl = BlackLittermanModel(S, pi="market", market_caps=market_caps)
                mu = bl.bl_returns()
                S = bl.bl_cov()

            ef = EfficientFrontier(mu, S)
            ef.add_constraint(lambda w: w >= 0)
            ef.add_constraint(lambda w: w <= max_weight)

            # sector caps linear constraints
            for sec, cap in sector_caps.items():
                idx = [i for i, t in enumerate(tickers) if SECTOR_MAP.get(t, "Other") == sec]
                if idx:
                    ef.add_constraint(lambda w, idx=idx, cap=cap: w[idx].sum() <= cap)

            if method == "Min Volatility":
                ef.min_volatility()
            elif method == "Efficient Return":
                ef.efficient_return(float(np.nanmedian(mu.values)))
            elif method == "Efficient Risk":
                asset_vol = np.sqrt(np.diag(S.values))
                ef.efficient_risk(float(np.nanmedian(asset_vol)))
            else:
                ef.max_sharpe(risk_free_rate=0.0)

            wd = ef.clean_weights()
            w = np.array([wd.get(t, 0.0) for t in tickers], dtype=float)
            w = w / w.sum() if w.sum() > 0 else np.ones(n) / n
            return w, ""

        if method == "HRP":
            hrp = HRPOpt(returns.T)
            hrp.optimize()
            wd = hrp.clean_weights()
            w = np.array([wd.get(t, 0.0) for t in tickers], dtype=float)
            w = clip_and_renormalize(w, max_weight)
            w = apply_sector_caps_post(w, tickers, sector_caps)
            return w, "HRP (caps post-processed)."

        if method == "CLA":
            cla = CLA(mu, S)
            cla.min_volatility()
            wd = cla.clean_weights()
            w = np.array([wd.get(t, 0.0) for t in tickers], dtype=float)
            w = clip_and_renormalize(w, max_weight)
            w = apply_sector_caps_post(w, tickers, sector_caps)
            return w, "CLA (caps post-processed)."

    except Exception as e:
        return np.ones(n) / n, f"Optimization error, fallback to equal weight: {e}"

    return np.ones(n) / n, "Unknown method; defaulted to equal weight."

# ------------------------------------------------------------
# App
# ------------------------------------------------------------
def main():
    st.markdown('<div class="main-header">📊 Advanced BIST Risk Budgeting System</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="note-box">
        <b>Data policy:</b> This app fetches <b>prices only from Yahoo Finance</b> via <code>yfinance</code>.
        If a ticker has missing days, the app applies <b>forward-fill only</b> (limited) and then enforces strict date alignment.
        <br/><span class="small-muted">No synthetic series is generated. If Yahoo returns no usable history, the ticker is dropped and reported.</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="data-source-badge">📡 Data Source: Yahoo Finance (via yfinance)</div>', unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown("## ⚙️ Controls")

        c1, c2 = st.columns(2)
        with c1:
            start_date = st.date_input("Start", value=date(2020, 1, 1), max_value=date.today() - timedelta(days=5))
        with c2:
            end_date = st.date_input("End", value=date.today(), max_value=date.today())

        if start_date >= end_date:
            st.error("Start date must be before end date.")
            st.stop()

        st.markdown("### 🧺 Universe")
        tickers = BIST50_TICKERS.copy()
        st.caption("TRALTIN removed. ASTOR included. Invalid/no-data tickers auto-drop.")

        st.markdown("### 🧠 Portfolio Strategy")
        strategy = st.selectbox(
            "Optimization method",
            ["Equal Weight", "Risk Parity (SciPy)", "Min Volatility", "Max Sharpe", "Efficient Return", "Efficient Risk", "HRP", "CLA", "Black-Litterman"],
            index=0,
        )

        st.markdown("### 🧱 Constraints")
        max_weight = st.slider("Max weight per stock", 0.02, 0.25, 0.12, 0.01)

        enable_sector_caps = st.checkbox("Enable sector caps", value=False)
        sector_caps = {}
        if enable_sector_caps:
            sector_caps["Banking"] = st.slider("Banking cap", 0.10, 0.60, 0.35, 0.01)
            sector_caps["Holding"] = st.slider("Holding cap", 0.05, 0.50, 0.25, 0.01)
            sector_caps["Retail"] = st.slider("Retail cap", 0.05, 0.50, 0.20, 0.01)
            sector_caps["Aviation"] = st.slider("Aviation cap", 0.05, 0.50, 0.25, 0.01)
            sector_caps["Industrial"] = st.slider("Industrial cap", 0.05, 0.70, 0.40, 0.01)
            sector_caps["Other"] = st.slider("Other cap", 0.05, 0.80, 0.60, 0.01)

        st.markdown("### 📈 Data cleaning")
        ffill_limit = st.slider("Forward-fill limit (days)", 1, 10, 5, 1)
        min_obs = st.slider("Minimum observations per asset", 60, 500, 180, 10)
        max_missing_frac = st.slider("Max missing fraction (pre-ffill)", 0.0, 0.8, 0.25, 0.05)

        st.markdown("### 📊 Rolling analytics")
        roll_window = st.slider("Rolling window (days)", 60, 504, 252, 21)
        top_n_roll = st.slider("Top-N assets in rolling chart", 5, 20, 10, 1)

        st.markdown("### 🧭 Benchmark")
        include_benchmark = st.checkbox("Include BIST100 benchmark", value=True)

        st.markdown("### 🧪 Stress scenarios")
        enable_stress = st.checkbox("Enable stress panel", value=True)
        fx_shock_pct = st.slider("FX shock (USDTRY move)", -20.0, 20.0, 5.0, 0.5)
        rate_shock_bp = st.slider("Rate shock (proxy bp)", -300, 300, 100, 25)

        if not PYPFOPT_AVAILABLE and strategy not in ["Equal Weight", "Risk Parity (SciPy)"]:
            st.warning("PyPortfolioOpt not available. Mean-variance/HRP/BL/CLA may fallback.")

    # Fetch prices
    with st.spinner("📥 Fetching prices from Yahoo Finance..."):
        prices_raw, dropped_raw, meta_fetch = fetch_yahoo_prices(tickers, start_date, end_date)

    if prices_raw.empty:
        st.error("❌ No data received from Yahoo Finance for the selected range/universe.")
        st.stop()

    prices, dropped_clean, clean_info = clean_prices_forward_fill(prices_raw, min_obs, max_missing_frac, ffill_limit)

    if prices.empty or prices.shape[1] < 2:
        st.error("❌ Not enough assets after cleaning (need at least 2).")
        st.write("Dropped/no-data tickers (raw fetch):", dropped_raw)
        st.write("Dropped tickers (cleaning filters):", dropped_clean)
        st.stop()

    returns = returns_from_prices(prices)

    # Benchmark
    benchmark_prices = None
    bench_used = None
    benchmark_returns = None
    if include_benchmark:
        for b in BENCHMARK_CANDIDATES:
            bp_raw, bd_raw, _ = fetch_yahoo_prices([b], start_date, end_date)
            if not bp_raw.empty and b in bp_raw.columns and bp_raw[b].dropna().shape[0] >= min_obs:
                benchmark_prices = bp_raw[[b]].ffill(limit=ffill_limit).dropna()
                bench_used = b
                break

        if benchmark_prices is not None:
            common_idx = prices.index.intersection(benchmark_prices.index)
            prices = prices.loc[common_idx]
            returns = returns_from_prices(prices)
            benchmark_returns = benchmark_prices.loc[common_idx, bench_used].pct_change().dropna()

    # Data summary
    st.markdown('<div class="sub-header">✅ Data Summary</div>', unsafe_allow_html=True)
    s1, s2, s3, s4 = st.columns(4)
    with s1:
        st.metric("Assets (final)", int(returns.shape[1]))
    with s2:
        st.metric("Trading days", int(returns.shape[0]))
    with s3:
        st.metric("Start (effective)", str(prices.index.min())[:10])
    with s4:
        st.metric("End (effective)", str(prices.index.max())[:10])

    with st.expander("🔎 Data diagnostics (what was fetched & what got dropped)"):
        st.write("Fetch mode:", meta_fetch.get("mode"))
        if meta_fetch.get("note"):
            st.write("Fetch note:", meta_fetch.get("note"))
        st.write("Universe requested:", len(tickers))
        st.write("Raw columns received:", int(prices_raw.shape[1]))
        st.write("Dropped/no-data tickers (raw fetch):", dropped_raw)
        st.write("Dropped tickers (cleaning filters):", dropped_clean)
        st.write("Final tickers:", returns.columns.tolist())
        if include_benchmark:
            st.write("Benchmark used:", bench_used)

    # Optimization
    st.markdown('<div class="sub-header">🧠 Portfolio Optimization</div>', unsafe_allow_html=True)
    w, opt_note = optimize_portfolio(prices, returns, strategy, max_weight, sector_caps if enable_sector_caps else {})
    weights = pd.Series(w, index=returns.columns, name="Weight").sort_values(ascending=False)

    c1, c2, c3 = st.columns([1.4, 1, 1])
    with c1:
        st.write("**Selected strategy:**", strategy)
        st.write("**PyPortfolioOpt available:**", "✅" if PYPFOPT_AVAILABLE else "❌")
        if opt_note:
            st.info(opt_note)
    with c2:
        st.metric("Max weight cap", f"{max_weight:.0%}")
    with c3:
        st.metric("Non-zero weights", int((weights > 1e-6).sum()))

    st.dataframe((weights * 100).round(2).rename("Weight %").to_frame(), use_container_width=True, height=260)

    # Risk metrics
    rr = RiskEngine.compute(returns, w)
    risk_metrics = rr.risk_table
    pm = rr.portfolio_metrics

    st.markdown('<div class="sub-header">📌 Key Portfolio Metrics</div>', unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Volatility (Ann.)", f"{pm['volatility']:.2%}")
    with m2:
        st.metric("Diversification Ratio", f"{pm['diversification_ratio']:.2f}")
    with m3:
        st.metric("Top Risk Contributor", pm["max_risk_asset"], f"{pm['max_risk_contrib']:.1f}%")
    with m4:
        st.metric("VaR 95% (1d, hist)", f"{pm['VaR_95_1d']:.2%}")

    # Risk contribution chart
    st.markdown('<div class="sub-header">🎯 Risk Contribution Analysis</div>', unsafe_allow_html=True)
    sorted_df = risk_metrics.sort_values("Risk_Contribution_%", ascending=True)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=sorted_df["Symbol"],
        x=sorted_df["Risk_Contribution_%"],
        orientation="h",
        text=sorted_df["Risk_Contribution_%"].round(1).astype(str) + "%",
        textposition="outside",
    ))
    eq_target = 100.0 / len(sorted_df)
    fig.add_vline(x=eq_target, line_dash="dash", line_color="red", annotation_text=f"Equal {eq_target:.1f}%")
    fig.update_layout(title="Risk Contribution by Asset", xaxis_title="Risk Contribution (%)", height=720, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, use_container_width=True)

    # Detailed table
    st.markdown('<div class="sub-header">📋 Detailed Risk Metrics</div>', unsafe_allow_html=True)
    ddf = risk_metrics.copy()
    ddf["Weight"] = (ddf["Weight"] * 100).round(2)
    ddf["Risk_Contribution_%"] = ddf["Risk_Contribution_%"].round(2)
    ddf["Individual_Volatility"] = (ddf["Individual_Volatility"] * 100).round(2)
    st.dataframe(ddf[["Risk_Rank", "Symbol", "Sector", "Weight", "Risk_Contribution_%", "Individual_Volatility"]], use_container_width=True, height=520)

    # Rolling RC
    st.markdown('<div class="sub-header">📈 Rolling Risk Contributions</div>', unsafe_allow_html=True)
    rr_pct = rolling_risk_contributions(returns, tuple(w.tolist()), roll_window)
    if rr_pct.empty:
        st.info("Not enough data for rolling contributions at this window.")
    else:
        latest = rr_pct.iloc[-1].sort_values(ascending=False)
        top_syms = latest.head(top_n_roll).index.tolist()
        fig_rr = go.Figure()
        for sym in top_syms:
            fig_rr.add_trace(go.Scatter(x=rr_pct.index, y=rr_pct[sym], mode="lines", name=sym))
        fig_rr.update_layout(title=f"Rolling % Risk Contributions (window={roll_window}d) — Top {top_n_roll}", yaxis_title="Contribution (%)", height=460)
        st.plotly_chart(fig_rr, use_container_width=True)

    # Sector analysis
    st.markdown('<div class="sub-header">🏭 Sector Risk Analysis</div>', unsafe_allow_html=True)
    sector_analysis = risk_metrics.groupby("Sector").agg({"Risk_Contribution_%": "sum", "Weight": "sum"}).reset_index()
    sector_analysis["Weight"] = sector_analysis["Weight"] * 100
    sector_analysis = sector_analysis.sort_values("Risk_Contribution_%", ascending=False)
    st.dataframe(sector_analysis, use_container_width=True, height=280)

    # Active risk contributions
    st.markdown('<div class="sub-header">🧭 Benchmark-Relative (Active) Risk Contributions</div>', unsafe_allow_html=True)
    active_df = pd.DataFrame()
    active_meta = {}
    if benchmark_returns is None:
        st.info("Benchmark not available in this run (Yahoo returned no usable benchmark history).")
    else:
        active_df, active_meta = active_risk_contributions(returns, w, benchmark_returns)
        a1, a2, a3 = st.columns(3)
        with a1:
            st.metric("Tracking Error (Ann.)", f"{active_meta['tracking_error']:.2%}")
        with a2:
            st.metric("Active Return (Ann.)", f"{active_meta['active_mean']:.2%}")
        with a3:
            st.metric("Information Ratio", f"{active_meta['information_ratio']:.2f}")

        top_active = active_df.head(15).sort_values("Active_Risk_%", ascending=True)
        fig_a = go.Figure()
        fig_a.add_trace(go.Bar(y=top_active["Symbol"], x=top_active["Active_Risk_%"], orientation="h"))
        fig_a.update_layout(title="Top Active Risk Contributors (vs BIST100)", height=520)
        st.plotly_chart(fig_a, use_container_width=True)
        st.dataframe(active_df, use_container_width=True, height=320)

    # Stress scenarios
    st.markdown('<div class="sub-header">🧪 Stress Scenarios</div>', unsafe_allow_html=True)
    stress_results = {}
    betas_df = pd.DataFrame()
    if enable_stress:
        with st.spinner("📥 Fetching stress factors from Yahoo..."):
            factors_raw, factors_dropped, _ = fetch_yahoo_prices([FX_FACTOR, RATE_FACTOR], start_date, end_date)

        if factors_raw.empty or FX_FACTOR not in factors_raw.columns:
            st.warning("FX factor (USDTRY) not available from Yahoo in this run; stress results limited.")
        else:
            factors_raw = factors_raw.ffill(limit=ffill_limit).dropna()
            common = returns.index.intersection(factors_raw.index)
            fx = factors_raw.loc[common, FX_FACTOR].pct_change().dropna()
            if RATE_FACTOR in factors_raw.columns:
                tnx = factors_raw.loc[common, RATE_FACTOR]
                rate_yield_dec = (tnx / 1000.0)
                rate_changes = rate_yield_dec.diff().dropna()
            else:
                rate_changes = pd.Series(0.0, index=fx.index)

            common2 = returns.index.intersection(fx.index).intersection(rate_changes.index)
            R = returns.loc[common2]
            fx = fx.loc[common2]
            rate_changes = rate_changes.loc[common2]

            betas_df = estimate_factor_betas(R, fx, rate_changes, min_obs=min(120, max(60, len(common2)//2)))
            betas_df = betas_df.set_index("Symbol").reindex(returns.columns).reset_index()

            stress_results = stress_scenario_impact(w, betas_df, fx_shock_pct/100.0, rate_shock_bp)
            st.write(f"Scenario: FX {fx_shock_pct:+.1f}% (USDTRY), Rate {rate_shock_bp:+d} bp (proxy {RATE_FACTOR}).")
            st.metric("Estimated 1-day portfolio impact", f"{stress_results['portfolio_impact']:+.2%}")
            det = stress_results["details"].copy()
            det["Shock_Return"] = (det["Shock_Return"] * 100).round(3)
            det["Contribution"] = (det["Contribution"] * 100).round(3)
            st.dataframe(det.head(20), use_container_width=True, height=320)
    else:
        st.info("Stress scenarios disabled.")

    # Export
    st.markdown('<div class="sub-header">📦 Export</div>', unsafe_allow_html=True)
    sheets = {
        "Portfolio_Weights": weights.reset_index().rename(columns={"index": "Symbol"}),
        "Risk_Metrics": risk_metrics,
        "Sector_Analysis": sector_analysis,
        "Portfolio_Summary": pd.DataFrame([pm]),
    }
    if not active_df.empty:
        sheets["Active_Risk"] = active_df
        sheets["Active_Summary"] = pd.DataFrame([active_meta])
    if not rr_pct.empty:
        sheets["Rolling_RC_pct"] = rr_pct.tail(250).reset_index().rename(columns={"index": "Date"})
    if enable_stress and stress_results:
        sheets["Stress_Details"] = stress_results["details"].reset_index(drop=True)
        if not betas_df.empty:
            sheets["Factor_Betas"] = betas_df

    xlsx = to_excel_bytes(sheets)
    st.download_button(
        "📥 Download Full Report (Excel)",
        data=xlsx,
        file_name=f"bist_risk_budgeting_report_{datetime.now().strftime('%Y%m%d')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    st.markdown("---")
    st.markdown(
        "<div class='small-muted'><b>The Quantitative Analysis Performed by LabGen25@Istanbul by Murat KONUKLAR 2026</b></div>",
        unsafe_allow_html=True,
    )

    if not PYPFOPT_AVAILABLE:
        st.warning(
            "PyPortfolioOpt could not be imported. This usually means the Streamlit Cloud build "
            "failed to install cvxpy/solver dependencies under your Python version. Use the pinned requirements "
            "provided with this app and set Python to 3.11 in Streamlit Cloud settings."
        )

if __name__ == "__main__":
    main()
