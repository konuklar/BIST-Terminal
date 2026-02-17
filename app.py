
# ============================================================
# 📊 ADVANCED BIST Risk Budgeting System (Streamlit Cloud)
# Yahoo Finance ONLY • No Synthetic Series • Forward-Fill Only
#
# Includes:
# - BIST50 (baseline hardcoded universe) + ASTOR.IS (forced) ; excludes KOZAL, TRALTIN
# - BIST100 benchmark (XU100.IS primary, ^XU100 fallback)
# - PyPortfolioOpt optimization (EF, HRP, Black-Litterman, + objectives)
# - Constraints: max weight per stock, sector caps
# - Risk budgeting: MRC/CRC, rolling risk contributions
# - Tail risk: VaR / CVaR / ES (historical) + horizons
# - Active risk (benchmark-relative) contributions (tracking error decomposition)
# - Stress scenarios: FX shock & rate shock (factor-betas from Yahoo)
# - Robust data fetching: batch → chunked → per-ticker history
# - Safe Excel export (fixes pandas/xlsxwriter ValueError)
#
# Signature:
# "The Quantitative Analysis Performed by LabGen25@Istanbul by Murat KONUKLAR 2026"
# ============================================================

from __future__ import annotations

import json
import math
import re
import time
import warnings
from dataclasses import dataclass
from datetime import datetime, date, timedelta
from io import BytesIO
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.optimize import minimize

warnings.filterwarnings("ignore")

# ------------------------------------------------------------
# Optional packages (hard requirement per user: PyPortfolioOpt)
# ------------------------------------------------------------
PYPFOPT_AVAILABLE = False
PYPFOPT_IMPORT_ERROR = None
try:
    import cvxpy as cp
    from pypfopt import EfficientFrontier, risk_models, expected_returns, objective_functions
    from pypfopt.black_litterman import BlackLittermanModel
    from pypfopt.hierarchical_risk_parity import HRPOpt
    PYPFOPT_AVAILABLE = True
except Exception as e:
    PYPFOPT_AVAILABLE = False
    PYPFOPT_IMPORT_ERROR = str(e)

# ------------------------------------------------------------
# Streamlit config + CSS
# ------------------------------------------------------------
st.set_page_config(page_title="BIST Risk Budgeting Terminal", page_icon="📊", layout="wide", initial_sidebar_state="expanded")

SIGNATURE = "The Quantitative Analysis Performed by LabGen25@Istanbul by Murat KONUKLAR 2026"

st.markdown(
    """
<style>
.main-header { font-size: 2.2rem; font-weight: 800; color: #0B2A6B; margin: 0.2rem 0 0.6rem 0;}
.sub-header { font-size: 1.25rem; font-weight: 700; color: #1D4ED8; margin: 0.75rem 0 0.4rem 0;}
.badge { display:inline-block; padding: .25rem .7rem; border-radius: 999px; background: #0B2A6B; color: white; font-size: 0.82rem; }
.kpi { background: #F3F4F6; padding: 0.9rem; border-radius: 0.7rem; border: 1px solid rgba(0,0,0,.06); }
.small { font-size: 0.9rem; color: rgba(0,0,0,.65); }
.warn { color: #B91C1C; font-weight: 700; }
.ok { color: #047857; font-weight: 700; }
code { font-size: 0.9rem; }
</style>
""",
    unsafe_allow_html=True,
)

# ------------------------------------------------------------
# Universe + metadata (baseline BIST50-style list)
# NOTE: BIST50 constituents change over time; we keep a robust baseline list.
# We strictly use Yahoo Finance data. We exclude KOZAL and TRALTIN. We force ASTOR.IS.
# ------------------------------------------------------------
BASE_UNIVERSE_BIST50 = [
    "AKBNK.IS","ALARK.IS","ARCLK.IS","ASELS.IS","ASTOR.IS",
    "BIMAS.IS","CCOLA.IS","DOAS.IS","EGEEN.IS","EKGYO.IS",
    "ENKAI.IS","EREGL.IS","FROTO.IS","GARAN.IS","GUBRF.IS",
    "HEKTS.IS","ISCTR.IS","KCHOL.IS","KRDMD.IS","MGROS.IS",
    "ODAS.IS","OTKAR.IS","PETKM.IS","PGSUS.IS","SAHOL.IS",
    "SASA.IS","SISE.IS","SOKM.IS","TCELL.IS","THYAO.IS",
    "TKFEN.IS","TOASO.IS","TSKB.IS","TTKOM.IS","TTRAK.IS",
    "TUPRS.IS","ULKER.IS","VAKBN.IS","VESTL.IS","YKBNK.IS",
    "KONTR.IS","KLSER.IS","CIMSA.IS","KOZAA.IS","ENJSA.IS",
    "BRSAN.IS","BAGFS.IS","KMPUR.IS","AKSEN.IS","AEFES.IS"
]

# Explicit exclusions per user
EXCLUDED = {"KOZAL.IS", "TRALTIN.IS", "TRALTIN", "TRALT", "TRALTIN.IS"}
BASE_UNIVERSE_BIST50 = [t for t in BASE_UNIVERSE_BIST50 if t not in EXCLUDED]

BENCHMARK_CANDIDATES = ["XU100.IS", "^XU100"]  # Yahoo Finance has both; XU100.IS usually more stable with yfinance
BIST50_INDEX_CANDIDATES = ["XU050.IS", "^XU050"]  # Optional (for display only)

FX_USDTRY_CANDIDATES = ["TRY=X"]        # USDTRY on Yahoo
FX_EURTRY_CANDIDATES = ["EURTRY=X"]     # EURTRY on Yahoo
RATE_CANDIDATES = ["TR10YT=RR", "TR10YT=XX", "^TNX"]  # Turkey 10Y sometimes appears as TR10YT=RR on some feeds; fallback to ^TNX

# Sector mapping (approx, for caps). You can edit in-app.
SECTOR_MAP = {
    "AKBNK.IS":"Banking","GARAN.IS":"Banking","ISCTR.IS":"Banking","VAKBN.IS":"Banking","YKBNK.IS":"Banking","TSKB.IS":"Banking",
    "ARCLK.IS":"Industrial","ALARK.IS":"Holding/Infra","ENKAI.IS":"Holding/Infra","KCHOL.IS":"Holding/Infra","SAHOL.IS":"Holding/Infra",
    "ASELS.IS":"Defense","BIMAS.IS":"Retail","MGROS.IS":"Retail","SOKM.IS":"Retail","ULKER.IS":"Food & Beverage","CCOLA.IS":"Food & Beverage","AEFES.IS":"Food & Beverage",
    "EKGYO.IS":"Real Estate",
    "EREGL.IS":"Iron & Steel","KRDMD.IS":"Iron & Steel","BRSAN.IS":"Iron & Steel",
    "FROTO.IS":"Automotive","TOASO.IS":"Automotive","DOAS.IS":"Automotive","OTKAR.IS":"Automotive","TTRAK.IS":"Automotive",
    "PETKM.IS":"Petrochemical","TUPRS.IS":"Energy",
    "PGSUS.IS":"Aviation","THYAO.IS":"Aviation",
    "SASA.IS":"Chemicals","CIMSA.IS":"Cement","SISE.IS":"Materials/Glass",
    "TTKOM.IS":"Telecom","TCELL.IS":"Telecom",
    "ASTOR.IS":"Industrial","AKSEN.IS":"Energy","ENJSA.IS":"Energy",
    "HEKTS.IS":"Chemicals","GUBRF.IS":"Chemicals",
    "EGEEN.IS":"Industrial","TKFEN.IS":"Industrial",
    "KOZAA.IS":"Mining",
    "ODAS.IS":"Energy",
    "KONTR.IS":"Industrial","KLSER.IS":"Industrial","KMPUR.IS":"Industrial","BAGFS.IS":"Industrial"
}

# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------
def _now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def _ensure_list(x) -> List[str]:
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return list(x)
    return [str(x)]

def _normalize_ticker(t: str) -> str:
    t = (t or "").strip().upper()
    # keep indices (start with ^) and FX symbols as-is
    if t.startswith("^") or t.endswith("=X") or t.endswith("=RR") or t.endswith("=XX"):
        return t
    # already has .IS
    if "." in t:
        return t
    return f"{t}.IS"

def _sanitize_sheet_name(name: str) -> str:
    # Excel sheet name rules: max 31 chars, no : \ / ? * [ ]
    bad = r'[:\\/*?\[\]]'
    name = re.sub(bad, " ", str(name))
    name = name.strip()
    if not name:
        name = "Sheet"
    return name[:31]

def _excel_safe_df(df: pd.DataFrame) -> pd.DataFrame:
    """Fix ValueError in pandas Excel formatter by converting problematic cell types."""
    if df is None:
        return pd.DataFrame()
    out = df.copy()
    # Ensure tz-naive datetimes
    for c in out.columns:
        if pd.api.types.is_datetime64tz_dtype(out[c]):
            out[c] = out[c].dt.tz_convert(None)
    # Convert object cells that are list/dict/ndarray to JSON strings
    if len(out.columns) > 0:
        obj_cols = [c for c in out.columns if out[c].dtype == "object"]
        for c in obj_cols:
            def _coerce(v):
                if isinstance(v, (dict, list, tuple, set, np.ndarray)):
                    try:
                        return json.dumps(v, ensure_ascii=False)
                    except Exception:
                        return str(v)
                if isinstance(v, (pd.Timestamp, datetime, date)):
                    try:
                        return pd.to_datetime(v).to_pydatetime().replace(tzinfo=None).isoformat(sep=" ")
                    except Exception:
                        return str(v)
                return v
            out[c] = out[c].map(_coerce)
    return out

def to_excel_bytes(sheets: Dict[str, pd.DataFrame]) -> bytes:
    """Write multiple dataframes to a single Excel file safely."""
    bio = BytesIO()
    with pd.ExcelWriter(bio, engine="xlsxwriter") as writer:
        # Cover page
        cover = pd.DataFrame({
            "Label": ["Signature", "Generated"],
            "Value": [SIGNATURE, _now_str()]
        })
        _excel_safe_df(cover).to_excel(writer, sheet_name="Cover", index=False)

        for name, df in sheets.items():
            safe_name = _sanitize_sheet_name(name)
            safe_df = _excel_safe_df(df)
            safe_df.to_excel(writer, sheet_name=safe_name, index=False)
    bio.seek(0)
    return bio.read()

def _pick_first_working_ticker(candidates: List[str], start: str, end: str) -> Optional[str]:
    for t in candidates:
        try:
            d = yf.download(t, start=start, end=end, interval="1d", auto_adjust=True, progress=False, threads=False)
            if d is not None and not d.empty:
                return t
        except Exception:
            pass
    return None

# ------------------------------------------------------------
# Yahoo Finance fetching (robust) — NO synthetic data
# ------------------------------------------------------------
@dataclass
class FetchReport:
    mode: str
    requested: List[str]
    received_cols: List[str]
    dropped_raw: List[str]
    dropped_clean: List[str]
    notes: List[str]

def _extract_close_from_download(df: pd.DataFrame, tickers: List[str]) -> pd.DataFrame:
    """Extract a Close/Adj Close panel from yfinance.download output, robust to MultiIndex layouts."""
    if df is None or df.empty:
        return pd.DataFrame()

    # Single ticker — columns are OHLCV
    if len(tickers) == 1 and isinstance(df.columns, pd.Index):
        if "Adj Close" in df.columns:
            return pd.DataFrame({tickers[0]: df["Adj Close"]})
        if "Close" in df.columns:
            return pd.DataFrame({tickers[0]: df["Close"]})
        # fallback: first numeric column
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if num_cols:
            return pd.DataFrame({tickers[0]: df[num_cols[0]]})
        return pd.DataFrame()

    # MultiIndex cases
    if isinstance(df.columns, pd.MultiIndex):
        # Case A: level 0 = fields, level 1 = tickers
        if "Adj Close" in df.columns.get_level_values(0):
            out = df.xs("Adj Close", axis=1, level=0, drop_level=True)
            return out
        if "Close" in df.columns.get_level_values(0):
            out = df.xs("Close", axis=1, level=0, drop_level=True)
            return out

        # Case B: level 1 = fields, level 0 = tickers
        if "Adj Close" in df.columns.get_level_values(1):
            out = df.xs("Adj Close", axis=1, level=1, drop_level=True)
            return out
        if "Close" in df.columns.get_level_values(1):
            out = df.xs("Close", axis=1, level=1, drop_level=True)
            return out

    # If not MultiIndex, attempt to find 'Close' column groups
    if "Adj Close" in df.columns:
        return df[["Adj Close"]].rename(columns={"Adj Close": tickers[0]})
    if "Close" in df.columns:
        return df[["Close"]].rename(columns={"Close": tickers[0]})

    return pd.DataFrame()

@st.cache_data(ttl=60 * 45, show_spinner=False)
def fetch_prices_yahoo(
    tickers: List[str],
    start: str,
    end: str,
    chunk_size: int = 25,
    max_retries: int = 3,
    pause: float = 1.5,
) -> Tuple[pd.DataFrame, FetchReport]:
    """Fetch close prices for a list of tickers from Yahoo Finance using yfinance.
    Strategy: batch → chunked → per-ticker history. No synthetic series."""
    tickers = [t for t in tickers if t and t not in EXCLUDED]
    tickers = [_normalize_ticker(t) for t in tickers]
    tickers = list(dict.fromkeys(tickers))  # de-dup

    report = FetchReport(
        mode="batch",
        requested=tickers,
        received_cols=[],
        dropped_raw=[],
        dropped_clean=[],
        notes=[]
    )

    def _try_download(tks: List[str], threads: bool) -> pd.DataFrame:
        last_err = None
        for attempt in range(max_retries):
            try:
                raw = yf.download(
                    tickers=tks,
                    start=start,
                    end=end,
                    interval="1d",
                    auto_adjust=True,
                    group_by="column",
                    progress=False,
                    threads=threads,
                    timeout=30
                )
                close = _extract_close_from_download(raw, tks)
                if close is not None and not close.empty:
                    return close
            except Exception as e:
                last_err = e
            time.sleep(pause * (2 ** attempt))
        if last_err is not None:
            report.notes.append(f"download error (last): {last_err}")
        return pd.DataFrame()

    # 1) Batch download
    close = _try_download(tickers, threads=False)  # threads=False is often more reliable for Yahoo throttling

    # 2) Chunked download
    if close.empty and len(tickers) > 1:
        report.mode = "chunked"
        parts = []
        for i in range(0, len(tickers), chunk_size):
            chunk = tickers[i:i + chunk_size]
            d = _try_download(chunk, threads=False)
            if not d.empty:
                parts.append(d)
        if parts:
            close = pd.concat(parts, axis=1)

    # 3) Per-ticker history fallback
    if close.empty:
        report.mode = "per_ticker_history"
        data = {}
        for t in tickers:
            try:
                hist = yf.Ticker(t).history(start=start, end=end, interval="1d", auto_adjust=True)
                if hist is not None and not hist.empty:
                    # With auto_adjust=True, adjusted close is typically in "Close"
                    if "Close" in hist.columns:
                        data[t] = hist["Close"]
                    elif "Adj Close" in hist.columns:
                        data[t] = hist["Adj Close"]
            except Exception as e:
                report.notes.append(f"{t} history error: {e}")
        if data:
            close = pd.DataFrame(data).sort_index()

    if close is None or close.empty:
        report.received_cols = []
        report.dropped_raw = tickers
        return pd.DataFrame(), report

    # Standardize index & columns
    close = close.copy()
    close.index = pd.to_datetime(close.index).tz_localize(None)
    close = close.sort_index()
    close = close.loc[:, ~close.columns.duplicated()].copy()

    # Raw drop list (empty columns)
    dropped_raw = [c for c in tickers if c not in close.columns]
    report.dropped_raw = dropped_raw
    report.received_cols = close.columns.tolist()

    return close, report

def clean_prices(
    prices: pd.DataFrame,
    ffill_limit: int = 5,
    min_obs: int = 80,
    max_missing_pct: float = 0.25,
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """Forward fill only, drop too-missing columns, enforce minimum observations."""
    if prices is None or prices.empty:
        return pd.DataFrame(), [], []

    df = prices.copy()

    # drop all-NaN cols
    df = df.dropna(axis=1, how="all")

    # Forward fill only (NO synthetic, NO backfill)
    df = df.ffill(limit=ffill_limit)

    # drop columns still with too much missing
    missing_pct = df.isna().mean()
    drop_missing = missing_pct[missing_pct > max_missing_pct].index.tolist()

    # min observations
    obs = df.notna().sum()
    drop_minobs = obs[obs < min_obs].index.tolist()

    drop_cols = sorted(set(drop_missing + drop_minobs))
    cleaned = df.drop(columns=drop_cols, errors="ignore")

    # drop rows all nan
    cleaned = cleaned.dropna(how="all")

    return cleaned, drop_missing, drop_minobs

def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    r = prices.pct_change()
    r = r.replace([np.inf, -np.inf], np.nan)
    # after ffill, we can require complete rows for clean covariance math
    r = r.dropna(how="any")
    return r

# ------------------------------------------------------------
# Risk Analytics: MRC/CRC, VaR/CVaR/ES, rolling contributions, active risk
# ------------------------------------------------------------
def portfolio_returns(returns: pd.DataFrame, w: np.ndarray) -> pd.Series:
    return (returns.values @ w).astype(float).reshape(-1)

def annualize_vol(series: pd.Series, freq: int = 252) -> float:
    return float(series.std() * np.sqrt(freq))

def historical_var_cvar_es(r: pd.Series, alpha: float = 0.05) -> Tuple[float, float, float]:
    r = r.dropna().astype(float)
    if len(r) < 10:
        return np.nan, np.nan, np.nan
    var = np.quantile(r, alpha)
    tail = r[r <= var]
    cvar = float(tail.mean()) if len(tail) else np.nan
    es = cvar  # historical ES equals CVaR under this convention
    return float(var), float(cvar), float(es)

def risk_contributions(returns: pd.DataFrame, w: np.ndarray) -> Tuple[pd.DataFrame, Dict[str, float], pd.DataFrame]:
    """Compute MRC/CRC on annualized covariance."""
    tickers = returns.columns.tolist()
    n = len(tickers)

    cov = returns.cov().values * 252
    w = np.asarray(w).reshape(-1)
    w = w / w.sum()

    port_var = float(w @ cov @ w)
    port_vol = float(np.sqrt(max(port_var, 0.0)))

    indiv_vol = np.sqrt(np.clip(np.diag(cov), 0, None))

    # Marginal contribution
    mrc = (cov @ w) / (port_vol + 1e-12)
    crc = w * mrc
    pct = (crc / (port_vol + 1e-12)) * 100.0

    df = pd.DataFrame({
        "Symbol": tickers,
        "Sector": [SECTOR_MAP.get(t, "Other") for t in tickers],
        "Weight": w,
        "Individual_Volatility": indiv_vol,
        "Marginal_Risk_Contribution": mrc,
        "Component_Risk": crc,
        "Risk_Contribution_%": pct,
    }).sort_values("Risk_Contribution_%", ascending=False).reset_index(drop=True)
    df["Risk_Rank"] = np.arange(1, len(df) + 1)

    # Diversification ratio
    weighted_avg_vol = float(np.sum(w * indiv_vol))
    div_ratio = weighted_avg_vol / (port_vol + 1e-12)

    metrics = {
        "portfolio_vol": port_vol,
        "portfolio_var": port_var,
        "diversification_ratio": div_ratio,
        "n_assets": n,
        "avg_indiv_vol": float(np.mean(indiv_vol)) if len(indiv_vol) else np.nan,
        "top_risk_asset": df.iloc[0]["Symbol"] if len(df) else None,
        "top_risk_pct": float(df.iloc[0]["Risk_Contribution_%"]) if len(df) else np.nan,
    }
    cov_df = pd.DataFrame(cov, index=tickers, columns=tickers)
    return df, metrics, cov_df

def rolling_risk_contributions(returns: pd.DataFrame, w: np.ndarray, window: int = 60) -> pd.DataFrame:
    """Rolling component risk contributions (annualized) with fixed weights."""
    tickers = returns.columns.tolist()
    w = np.asarray(w).reshape(-1)
    w = w / w.sum()
    out = []
    for i in range(window, len(returns) + 1):
        rwin = returns.iloc[i - window:i]
        cov = rwin.cov().values * 252
        port_var = float(w @ cov @ w)
        port_vol = float(np.sqrt(max(port_var, 0.0)))
        mrc = (cov @ w) / (port_vol + 1e-12)
        crc = w * mrc
        pct = (crc / (port_vol + 1e-12)) * 100.0
        out.append(pd.Series(pct, index=tickers, name=returns.index[i - 1]))
    if not out:
        return pd.DataFrame()
    return pd.DataFrame(out)

def active_risk_contributions(asset_returns: pd.DataFrame, benchmark_returns: pd.Series, w: np.ndarray) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Tracking error decomposition based on active returns per asset: (r_i - r_b)."""
    w = np.asarray(w).reshape(-1)
    w = w / w.sum()

    common = asset_returns.index.intersection(benchmark_returns.index)
    R = asset_returns.loc[common].copy()
    b = benchmark_returns.loc[common].astype(float)

    # Active return per asset
    A = R.sub(b, axis=0)
    A = A.dropna(how="any")
    if len(A) < 20 or A.shape[1] < 2:
        return pd.DataFrame(), {"tracking_error": np.nan}

    covA = A.cov().values * 252
    te_var = float(w @ covA @ w)
    te = float(np.sqrt(max(te_var, 0.0)))

    mrc = (covA @ w) / (te + 1e-12)
    crc = w * mrc
    pct = (crc / (te + 1e-12)) * 100.0

    df = pd.DataFrame({
        "Symbol": A.columns,
        "Sector": [SECTOR_MAP.get(t, "Other") for t in A.columns],
        "Weight": w,
        "Active_Risk_Contribution_%": pct,
        "Active_Component_Risk": crc,
        "Active_MRC": mrc
    }).sort_values("Active_Risk_Contribution_%", ascending=False).reset_index(drop=True)

    stats = {"tracking_error": te, "tracking_error_var": te_var}
    return df, stats

# ------------------------------------------------------------
# Optimization (PyPortfolioOpt + fallback)
# ------------------------------------------------------------
def build_sector_indices(tickers: List[str]) -> Dict[str, List[int]]:
    idx = {}
    for i, t in enumerate(tickers):
        s = SECTOR_MAP.get(t, "Other")
        idx.setdefault(s, []).append(i)
    return idx

def optimize_weights(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    method: str,
    rf: float,
    max_w: float,
    sector_caps: Dict[str, float],
    target_return: Optional[float] = None,
    target_vol: Optional[float] = None,
) -> Tuple[np.ndarray, Dict[str, float], Dict[str, Any]]:
    """Return (weights, perf, debug). Always returns weights summing to 1."""
    tickers = returns.columns.tolist()
    n = len(tickers)

    debug = {"method": method, "used_pypfopt": False, "notes": []}

    # Fallback equal weights
    w_eq = np.ones(n) / n

    if method == "Equal Weight":
        perf = {
            "expected_return": float(returns.mean().values @ w_eq * 252),
            "volatility": float(np.sqrt(w_eq @ (returns.cov().values * 252) @ w_eq)),
            "sharpe_ratio": np.nan
        }
        perf["sharpe_ratio"] = (perf["expected_return"] - rf) / (perf["volatility"] + 1e-12)
        return w_eq, perf, debug

    # Risk parity (SLSQP) fallback uses only numpy/scipy
    if method == "Risk Parity (SLSQP)":
        cov = returns.cov().values * 252

        def obj(x):
            x = np.clip(x, 0, max_w)
            x = x / (x.sum() + 1e-12)
            port_var = x @ cov @ x
            port_vol = np.sqrt(max(port_var, 0.0))
            mrc = (cov @ x) / (port_vol + 1e-12)
            rc = x * mrc
            target = port_vol / n
            return float(np.sum((rc - target) ** 2))

        cons = [{"type": "eq", "fun": lambda x: float(np.sum(x) - 1.0)}]
        bnds = [(0.0, max_w) for _ in range(n)]
        x0 = w_eq.copy()

        res = minimize(obj, x0, method="SLSQP", bounds=bnds, constraints=cons, options={"maxiter": 2000, "ftol": 1e-10})
        w = res.x if res.success else w_eq
        w = np.clip(w, 0, max_w)
        w = w / (w.sum() + 1e-12)

        perf = {
            "expected_return": float(returns.mean().values @ w * 252),
            "volatility": float(np.sqrt(w @ (returns.cov().values * 252) @ w)),
        }
        perf["sharpe_ratio"] = (perf["expected_return"] - rf) / (perf["volatility"] + 1e-12)
        debug["notes"].append(f"SLSQP success={res.success}")
        return w, perf, debug

    # PyPortfolioOpt route (required by user)
    if not PYPFOPT_AVAILABLE:
        debug["notes"].append("PyPortfolioOpt not available; returning Equal Weight.")
        return w_eq, {"expected_return": np.nan, "volatility": np.nan, "sharpe_ratio": np.nan}, debug

    debug["used_pypfopt"] = True

    # Expected returns and covariance
    mu = expected_returns.mean_historical_return(prices, frequency=252)
    S = risk_models.CovarianceShrinkage(prices).ledoit_wolf()

    ef = EfficientFrontier(mu, S)

    # Constraints: long-only + per-asset max
    ef.add_constraint(lambda w: w >= 0)
    ef.add_constraint(lambda w: w <= max_w)

    # Sector caps (sum of weights in sector <= cap)
    sector_idx = build_sector_indices(tickers)
    for sec, idxs in sector_idx.items():
        cap = float(sector_caps.get(sec, 1.0))
        if cap < 1.0 - 1e-9:  # only add if binding
            ef.add_constraint(lambda w, idxs=idxs, cap=cap: cp.sum(w[idxs]) <= cap)

    # Optimization
    try:
        if method == "Max Sharpe":
            ef.max_sharpe(risk_free_rate=rf)
        elif method == "Min Volatility":
            ef.min_volatility()
        elif method == "Max Utility":
            ef.max_quadratic_utility(risk_aversion=1.0)
        elif method == "Efficient Return":
            if target_return is None:
                target_return = float(mu.mean())
            ef.efficient_return(target_return)
        elif method == "Efficient Risk":
            if target_vol is None:
                # set a mild target
                target_vol = float(np.sqrt(np.diag(S)).mean())
            ef.efficient_risk(target_vol)
        elif method == "Min CVaR":
            # Optional objective: requires objective_functions and returns series
            try:
                ef.add_objective(objective_functions.CVaR, returns=returns)
                ef.min_volatility()
            except Exception:
                ef.min_volatility()
                debug["notes"].append("CVaR objective not available; used min_vol.")
        elif method == "Min Semivariance":
            try:
                ef.add_objective(objective_functions.semivariance, returns=returns)
                ef.min_volatility()
            except Exception:
                ef.min_volatility()
                debug["notes"].append("Semivariance objective not available; used min_vol.")
        else:
            ef.max_sharpe(risk_free_rate=rf)

        w_dict = ef.clean_weights()
        w = np.array([w_dict.get(t, 0.0) for t in tickers], dtype=float)
        w = np.clip(w, 0, max_w)
        w = w / (w.sum() + 1e-12)

        pret, pvol, psr = ef.portfolio_performance(risk_free_rate=rf, verbose=False)
        perf = {"expected_return": float(pret), "volatility": float(pvol), "sharpe_ratio": float(psr)}
        return w, perf, debug

    except Exception as e:
        debug["notes"].append(f"PyPortfolioOpt failed: {e}")
        return w_eq, {"expected_return": np.nan, "volatility": np.nan, "sharpe_ratio": np.nan}, debug

# ------------------------------------------------------------
# Stress scenarios (factor-beta from Yahoo)
# ------------------------------------------------------------
def _ols_beta(y: np.ndarray, x: np.ndarray) -> float:
    """OLS beta of y on x (no intercept), robust."""
    y = np.asarray(y).astype(float)
    x = np.asarray(x).astype(float)
    mask = np.isfinite(y) & np.isfinite(x)
    y = y[mask]
    x = x[mask]
    if len(y) < 30:
        return np.nan
    vx = np.var(x)
    if vx <= 1e-12:
        return np.nan
    return float(np.cov(y, x)[0, 1] / vx)

def estimate_factor_betas(asset_returns: pd.DataFrame, fx_ret: pd.Series, rate_ret: pd.Series) -> pd.DataFrame:
    """Compute each asset beta to FX and rate factors."""
    common = asset_returns.index
    if fx_ret is not None:
        common = common.intersection(fx_ret.index)
    if rate_ret is not None:
        common = common.intersection(rate_ret.index)

    R = asset_returns.loc[common].copy()
    betas = []
    for c in R.columns:
        y = R[c].values
        b_fx = _ols_beta(y, fx_ret.loc[common].values) if fx_ret is not None else np.nan
        b_rt = _ols_beta(y, rate_ret.loc[common].values) if rate_ret is not None else np.nan
        betas.append((c, b_fx, b_rt, SECTOR_MAP.get(c, "Other")))
    return pd.DataFrame(betas, columns=["Symbol", "Beta_FX_USDTRY", "Beta_RATE", "Sector"])

def scenario_impact(weights: np.ndarray, betas_df: pd.DataFrame, fx_shock: float, rate_shock: float) -> Tuple[pd.DataFrame, float]:
    """1-day linear factor shock impact per asset and portfolio."""
    df = betas_df.copy()
    df["Weight"] = weights
    df["Shock_FX"] = fx_shock
    df["Shock_RATE"] = rate_shock
    df["Impact_Est"] = df["Beta_FX_USDTRY"] * fx_shock + df["Beta_RATE"] * rate_shock
    df["Weighted_Impact"] = df["Weight"] * df["Impact_Est"]
    port = float(df["Weighted_Impact"].sum())
    df = df.sort_values("Weighted_Impact", ascending=True)
    return df, port

# ------------------------------------------------------------
# UI
# ------------------------------------------------------------
st.markdown(f'<div class="main-header">📊 BIST Risk Budgeting Terminal</div>', unsafe_allow_html=True)
st.markdown(f'<div class="badge">Yahoo Finance ONLY • No Synthetic Series</div> &nbsp; <span class="small">{SIGNATURE}</span>', unsafe_allow_html=True)
st.markdown("", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### ⚙️ Controls")
    today = datetime.now().date()
    default_start = date(2020, 1, 1)
    start_date = st.date_input("Start date", value=default_start, max_value=today - timedelta(days=5))
    end_date = st.date_input("End date", value=today, max_value=today)

    st.markdown("### 📦 Universe")
    universe_mode = st.selectbox("Universe", ["BIST50 (baseline)", "Custom list"], index=0)
    if universe_mode == "Custom list":
        user_txt = st.text_area("Tickers (comma-separated, e.g. AKBNK, GARAN, THYAO)", value=",".join([t.replace(".IS","") for t in BASE_UNIVERSE_BIST50[:20]]))
        universe = [_normalize_ticker(x) for x in user_txt.split(",") if x.strip()]
    else:
        universe = BASE_UNIVERSE_BIST50.copy()

    # Enforce explicit exclusions and ASTOR inclusion
    universe = [t for t in universe if t not in EXCLUDED and t != "KOZAL.IS"]
    if "ASTOR.IS" not in universe:
        universe = ["ASTOR.IS"] + universe

    st.markdown("### 🧹 Data cleaning")
    ffill_limit = st.slider("Forward-fill limit (days)", min_value=0, max_value=15, value=5, help="Only forward-fill is used. No backfill, no synthetic.")
    min_obs = st.slider("Min observations per asset", min_value=30, max_value=400, value=80)
    max_missing_pct = st.slider("Max missing % per asset", min_value=0.0, max_value=0.8, value=0.25, step=0.05)

    st.markdown("### 🧠 Optimization")
    rf = st.number_input("Risk-free rate (annual, decimal)", value=0.15, min_value=0.0, max_value=1.0, step=0.01)
    max_w = st.slider("Max weight per stock", min_value=0.02, max_value=0.30, value=0.15, step=0.01)

    methods = ["Equal Weight", "Risk Parity (SLSQP)", "Max Sharpe", "Min Volatility", "Max Utility", "Efficient Return", "Efficient Risk", "Min CVaR", "Min Semivariance"]
    opt_method = st.selectbox("Method", methods, index=0)

    # Sector caps UI
    st.markdown("### 🏭 Sector caps")
    # Build sector list from current universe
    _secs = sorted({SECTOR_MAP.get(t, "Other") for t in universe})
    sector_caps = {}
    with st.expander("Edit sector caps (sum of weights per sector ≤ cap)"):
        st.caption("Set caps ≤ 100%. Leave at 100% to disable.")
        for s in _secs:
            sector_caps[s] = st.slider(f"{s} cap", 0.05, 1.00, 1.00, 0.05)
    if not sector_caps:
        sector_caps = {s: 1.0 for s in _secs}

    st.markdown("### 🧪 Rolling + Tail risk")
    roll_window = st.slider("Rolling window (days)", 20, 252, 60, 5)
    var_alpha = st.selectbox("VaR level", ["95%", "99%"], index=0)
    alpha = 0.05 if var_alpha == "95%" else 0.01

    st.markdown("### ⚡ Stress shocks")
    fx_shock = st.slider("FX shock (USDTRY return, 1-day)", -0.20, 0.20, 0.05, 0.01)
    rate_shock = st.slider("Rate shock (proxy return, 1-day)", -0.10, 0.10, 0.02, 0.01)

    st.markdown("---")
    st.markdown(f"**Updated:** {_now_str()}")

run = st.button("▶ Run analysis", type="primary")

# ------------------------------------------------------------
# Execution
# ------------------------------------------------------------
if not run:
    st.info("Set parameters on the left and click **Run analysis**.")
    st.stop()

# Date validation
if start_date >= end_date:
    st.error("Start date must be before end date.")
    st.stop()

start = str(start_date)
end = str(end_date)

# Pick benchmark that actually works on Yahoo
benchmark_ticker = _pick_first_working_ticker(BENCHMARK_CANDIDATES, start, end)

if benchmark_ticker is None:
    st.warning("Benchmark not available from Yahoo for this range. Benchmark-relative panels will be disabled.")
else:
    st.success(f"Benchmark selected: {benchmark_ticker}")

# Data fetch
with st.spinner("Fetching prices from Yahoo Finance (batch → chunked → per-ticker fallback)..."):
    prices_raw, frep = fetch_prices_yahoo(universe + ([benchmark_ticker] if benchmark_ticker else []), start, end)

if prices_raw is None or prices_raw.empty:
    st.error("No data received from Yahoo Finance for the selected range/universe.")
    st.markdown("**What this means:** Yahoo returned an empty dataset. This can happen due to invalid tickers, delistings, or temporary Yahoo throttling.")
    st.markdown("Try: (1) shorten the date range, (2) reduce universe size, (3) rerun a minute later.")
    st.stop()

# Split benchmark out
bench_prices = None
asset_prices_raw = prices_raw.copy()
if benchmark_ticker and benchmark_ticker in prices_raw.columns:
    bench_prices = prices_raw[benchmark_ticker].copy()
    asset_prices_raw = prices_raw.drop(columns=[benchmark_ticker], errors="ignore")

# Clean asset prices
asset_prices, drop_missing, drop_minobs = clean_prices(asset_prices_raw, ffill_limit=ffill_limit, min_obs=min_obs, max_missing_pct=max_missing_pct)
dropped_clean = sorted(set(drop_missing + drop_minobs))

# Report
with st.expander("📌 Data diagnostics (click to expand)", expanded=False):
    st.write({"fetch_mode": frep.mode, "requested": len(frep.requested), "received_cols": len(frep.received_cols)})
    st.markdown("**Dropped/no-data tickers (raw fetch):**")
    st.code(str(frep.dropped_raw))
    st.markdown("**Dropped by cleaning:**")
    st.code(str(dropped_clean))
    if frep.notes:
        st.markdown("**Notes:**")
        st.write(frep.notes)
    st.markdown("**Final universe:**")
    st.write(asset_prices.columns.tolist())
    st.markdown("**Effective date range (after ffill & drops):**")
    st.write({"start": str(asset_prices.index.min().date()) if not asset_prices.empty else None,
              "end": str(asset_prices.index.max().date()) if not asset_prices.empty else None})

if asset_prices.shape[1] < 2:
    st.error("❌ Not enough assets after cleaning (need at least 2).")
    st.markdown("Increase missing tolerance / reduce min observations / reduce universe, then rerun.")
    st.stop()

# Returns (strict alignment)
returns = compute_returns(asset_prices)

if returns.empty or returns.shape[0] < 30:
    st.error("❌ Returns matrix is too small after alignment. Reduce min observations or shorten universe.")
    st.stop()

# Benchmark returns (if available)
bench_returns = None
if bench_prices is not None and not bench_prices.empty:
    bench_prices = bench_prices.ffill(limit=ffill_limit)
    bench_returns = bench_prices.pct_change().dropna()
    bench_returns = bench_returns.replace([np.inf, -np.inf], np.nan).dropna()
    # align to asset returns
    common = returns.index.intersection(bench_returns.index)
    returns = returns.loc[common].copy()
    bench_returns = bench_returns.loc[common].copy()

# Optimization
if not PYPFOPT_AVAILABLE:
    st.warning("⚠️ PyPortfolioOpt is not available in this environment. Check requirements/build logs.")
    if PYPFOPT_IMPORT_ERROR:
        st.caption(f"Import error: {PYPFOPT_IMPORT_ERROR}")
    st.caption("The app will still run, but PyPortfolioOpt strategies will fall back where possible.")
else:
    st.success("✅ PyPortfolioOpt is available and will be used for supported strategies.")

with st.spinner("Optimizing portfolio weights..."):
    w, perf, dbg = optimize_weights(
        prices=asset_prices,
        returns=returns,
        method=opt_method,
        rf=rf,
        max_w=max_w,
        sector_caps=sector_caps,
    )

# Risk contributions
risk_df, port_metrics, cov_df = risk_contributions(returns, w)
port_ret_series = pd.Series(portfolio_returns(returns, w), index=returns.index, name="Portfolio")

# Tail risk metrics
var1, cvar1, es1 = historical_var_cvar_es(port_ret_series, alpha=alpha)
# 10-day horizon (sqrt scaling for VaR & ES; conservative, standard practice)
h = 10
var10 = var1 * np.sqrt(h) if np.isfinite(var1) else np.nan
cvar10 = cvar1 * np.sqrt(h) if np.isfinite(cvar1) else np.nan
es10 = es1 * np.sqrt(h) if np.isfinite(es1) else np.nan

# Rolling risk contributions
roll_rc = rolling_risk_contributions(returns, w, window=roll_window)

# Active risk contributions
active_df = pd.DataFrame()
active_stats = {}
if bench_returns is not None and not bench_returns.empty:
    active_df, active_stats = active_risk_contributions(returns, bench_returns, w)

# Stress factors (Yahoo only)
fx_ticker = _pick_first_working_ticker(FX_USDTRY_CANDIDATES, start, end)
rate_ticker = _pick_first_working_ticker(RATE_CANDIDATES, start, end)

fx_ret = None
rate_ret = None

if fx_ticker:
    fx_prices, _ = fetch_prices_yahoo([fx_ticker], start, end)
    if not fx_prices.empty:
        fx_prices = fx_prices.ffill(limit=ffill_limit)
        fx_ret = fx_prices.iloc[:, 0].pct_change().dropna()

if rate_ticker:
    rt_prices, _ = fetch_prices_yahoo([rate_ticker], start, end)
    if not rt_prices.empty:
        rt_prices = rt_prices.ffill(limit=ffill_limit)
        # For yields, pct_change is a proxy; for ^TNX it's an index-like series
        rate_ret = rt_prices.iloc[:, 0].pct_change().dropna()

betas_df = estimate_factor_betas(returns, fx_ret, rate_ret) if (fx_ret is not None or rate_ret is not None) else pd.DataFrame()
scenario_df = pd.DataFrame()
scenario_port = np.nan
if not betas_df.empty:
    scenario_df, scenario_port = scenario_impact(w, betas_df, fx_shock, rate_shock)

# ------------------------------------------------------------
# Layout: Tabs
# ------------------------------------------------------------
tabs = st.tabs([
    "📦 Data",
    "🧠 Optimization",
    "🎯 Risk Budgeting",
    "📉 VaR / CVaR / ES",
    "🧭 Rolling Risk",
    "🧮 Active Risk vs BIST100",
    "⚡ Stress Scenarios",
    "📤 Export"
])

# --- Tab: Data
with tabs[0]:
    st.markdown('<div class="sub-header">📦 Data Summary</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Assets (final)", int(asset_prices.shape[1]))
    c2.metric("Trading days", int(returns.shape[0]))
    c3.metric("Start", str(returns.index.min().date()))
    c4.metric("End", str(returns.index.max().date()))

    st.caption("Prices are fetched only from Yahoo Finance via yfinance. Missing values are handled by forward-fill only (no backfill).")
    st.dataframe(asset_prices.tail(8), use_container_width=True)

    fig = go.Figure()
    sel = asset_prices.columns[:10]
    for t in sel:
        s = (asset_prices[t] / asset_prices[t].iloc[0]) * 100
        fig.add_trace(go.Scatter(x=s.index, y=s.values, name=t))
    fig.update_layout(title="Normalized Price (Top 10 assets)", height=420, template="plotly_white", legend_orientation="h")
    st.plotly_chart(fig, use_container_width=True)

# --- Tab: Optimization
with tabs[1]:
    st.markdown('<div class="sub-header">🧠 Optimization Results</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Method", opt_method)
    c2.metric("Exp. Return (ann.)", f"{perf.get('expected_return', np.nan):.2%}" if np.isfinite(perf.get("expected_return", np.nan)) else "n/a")
    c3.metric("Volatility (ann.)", f"{perf.get('volatility', np.nan):.2%}" if np.isfinite(perf.get("volatility", np.nan)) else "n/a")
    c4.metric("Sharpe (rf)", f"{perf.get('sharpe_ratio', np.nan):.2f}" if np.isfinite(perf.get("sharpe_ratio", np.nan)) else "n/a")

    st.caption(f"Max weight per stock: {max_w:.0%}. Sector caps apply where set below 100%.")
    if dbg.get("notes"):
        st.info("Optimizer notes: " + " | ".join(dbg["notes"]))

    w_df = pd.DataFrame({"Symbol": returns.columns, "Sector": [SECTOR_MAP.get(t, "Other") for t in returns.columns], "Weight": w})
    w_df = w_df.sort_values("Weight", ascending=False)
    st.dataframe(w_df, use_container_width=True, hide_index=True)

    fig = px.bar(w_df.head(20), x="Weight", y="Symbol", color="Sector", orientation="h", title="Top Weights (Top 20)")
    fig.update_layout(height=520, template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

# --- Tab: Risk Budgeting
with tabs[2]:
    st.markdown('<div class="sub-header">🎯 Risk Budgeting (MRC / CRC)</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Portfolio Vol (ann.)", f"{port_metrics['portfolio_vol']:.2%}")
    c2.metric("Diversification Ratio", f"{port_metrics['diversification_ratio']:.2f}")
    c3.metric("Top risk asset", str(port_metrics["top_risk_asset"]))
    c4.metric("Top risk %", f"{port_metrics['top_risk_pct']:.1f}%")

    top = risk_df.sort_values("Risk_Contribution_%", ascending=True)
    fig = go.Figure(go.Bar(
        x=top["Risk_Contribution_%"],
        y=top["Symbol"],
        orientation="h",
        marker=dict(color=top["Risk_Contribution_%"], colorscale="RdYlGn_r", showscale=True),
        text=top["Risk_Contribution_%"].round(1).astype(str) + "%",
        textposition="outside",
    ))
    eq = 100 / len(top)
    fig.add_vline(x=eq, line_dash="dash", line_color="red", annotation_text=f"Equal risk {eq:.1f}%")
    fig.update_layout(title="Risk Contribution by Asset (%)", height=780, template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        risk_df[["Risk_Rank","Symbol","Sector","Weight","Individual_Volatility","Marginal_Risk_Contribution","Risk_Contribution_%"]],
        use_container_width=True,
        hide_index=True
    )

# --- Tab: VaR/CVaR/ES
with tabs[3]:
    st.markdown('<div class="sub-header">📉 Tail Risk (Historical)</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    c1.metric(f"VaR ({var_alpha}, 1d)", f"{var1:.2%}" if np.isfinite(var1) else "n/a")
    c2.metric(f"CVaR/ES ({var_alpha}, 1d)", f"{cvar1:.2%}" if np.isfinite(cvar1) else "n/a")
    c3.metric("Portfolio Vol (ann.)", f"{port_metrics['portfolio_vol']:.2%}")

    c4, c5, c6 = st.columns(3)
    c4.metric(f"VaR ({var_alpha}, {h}d √t)", f"{var10:.2%}" if np.isfinite(var10) else "n/a")
    c5.metric(f"CVaR/ES ({var_alpha}, {h}d √t)", f"{cvar10:.2%}" if np.isfinite(cvar10) else "n/a")
    c6.metric("Mean daily return", f"{port_ret_series.mean():.3%}")

    fig = go.Figure()
    fig.add_trace(go.Histogram(x=port_ret_series * 100, nbinsx=60, name="Daily returns (%)"))
    if np.isfinite(var1):
        fig.add_vline(x=var1 * 100, line_dash="dash", line_color="red", annotation_text="VaR")
    if np.isfinite(cvar1):
        fig.add_vline(x=cvar1 * 100, line_dash="dot", line_color="darkred", annotation_text="CVaR/ES")
    fig.update_layout(title="Portfolio Returns Distribution", height=480, template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

# --- Tab: Rolling Risk
with tabs[4]:
    st.markdown('<div class="sub-header">🧭 Rolling Risk Contributions</div>', unsafe_allow_html=True)
    st.caption("Rolling component risk contributions (%) computed on a rolling covariance window with fixed weights.")

    if roll_rc is None or roll_rc.empty:
        st.warning("Rolling risk contributions not available (increase date range or reduce window).")
    else:
        # Show top contributors time series
        latest = roll_rc.iloc[-1].sort_values(ascending=False).head(6).index.tolist()
        fig = go.Figure()
        for t in latest:
            fig.add_trace(go.Scatter(x=roll_rc.index, y=roll_rc[t], name=t))
        fig.update_layout(title=f"Rolling Risk Contribution % (Top 6 at last date, window={roll_window})", height=520, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(roll_rc.tail(10), use_container_width=True)

# --- Tab: Active Risk vs BIST100
with tabs[5]:
    st.markdown('<div class="sub-header">🧮 Benchmark-Relative (Active) Risk Contributions</div>', unsafe_allow_html=True)
    if bench_returns is None or active_df is None or active_df.empty:
        st.warning("Benchmark-relative active risk panel is unavailable (benchmark missing or insufficient overlap).")
    else:
        te = active_stats.get("tracking_error", np.nan)
        st.metric("Tracking Error (ann.)", f"{te:.2%}" if np.isfinite(te) else "n/a")

        top = active_df.sort_values("Active_Risk_Contribution_%", ascending=True).head(20)
        fig = go.Figure(go.Bar(
            x=top["Active_Risk_Contribution_%"],
            y=top["Symbol"],
            orientation="h",
            marker=dict(color=top["Active_Risk_Contribution_%"], colorscale="RdYlGn_r"),
        ))
        fig.update_layout(title="Active Risk Contribution by Asset (%) — Tracking Error Decomposition", height=640, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(active_df, use_container_width=True, hide_index=True)

# --- Tab: Stress Scenarios
with tabs[6]:
    st.markdown('<div class="sub-header">⚡ Stress Scenarios (FX Shock / Rate Shock)</div>', unsafe_allow_html=True)
    st.caption("Factor betas are estimated from Yahoo Finance time series only. Scenario impact is a 1-day linear approximation.")

    c1, c2, c3 = st.columns(3)
    c1.write(f"FX factor: `{fx_ticker}`" if fx_ticker else "FX factor: n/a")
    c2.write(f"Rate factor: `{rate_ticker}`" if rate_ticker else "Rate factor: n/a")
    c3.metric("Portfolio 1-day impact (est.)", f"{scenario_port:.2%}" if np.isfinite(scenario_port) else "n/a")

    if betas_df is None or betas_df.empty:
        st.warning("Not enough factor data to estimate betas (FX/rate ticker missing or no overlap).")
    else:
        st.dataframe(betas_df.sort_values("Beta_FX_USDTRY", ascending=False), use_container_width=True, hide_index=True)

        if scenario_df is not None and not scenario_df.empty:
            fig = px.bar(scenario_df.tail(20), x="Weighted_Impact", y="Symbol", color="Sector", orientation="h",
                         title="Weighted Scenario Impact (bottom 20 shown)")
            fig.update_layout(height=620, template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

# --- Tab: Export
with tabs[7]:
    st.markdown('<div class="sub-header">📤 Export</div>', unsafe_allow_html=True)
    st.caption("Exports are Excel-safe (timezone-aware datetimes and object cells are sanitized).")

    sheets = {
        "Weights": w_df,
        "Risk_Metrics": risk_df,
        "Covariance": cov_df.reset_index().rename(columns={"index":"Symbol"}),
        "Portfolio_Returns": port_ret_series.reset_index().rename(columns={"index":"Date", "Portfolio":"Return"}),
        "Rolling_Risk": roll_rc.reset_index().rename(columns={"index":"Date"}) if roll_rc is not None and not roll_rc.empty else pd.DataFrame(),
        "Active_Risk": active_df if active_df is not None else pd.DataFrame(),
        "Betas": betas_df if betas_df is not None else pd.DataFrame(),
        "Scenario": scenario_df if scenario_df is not None else pd.DataFrame(),
        "Data_Diagnostics": pd.DataFrame({
            "key": ["fetch_mode","benchmark","requested","received","dropped_raw","dropped_clean","start","end","signature"],
            "value": [
                frep.mode,
                benchmark_ticker or "",
                len(frep.requested),
                len(frep.received_cols),
                json.dumps(frep.dropped_raw, ensure_ascii=False),
                json.dumps(dropped_clean, ensure_ascii=False),
                str(returns.index.min().date()),
                str(returns.index.max().date()),
                SIGNATURE
            ]
        })
    }

    xlsx = to_excel_bytes({k: v for k, v in sheets.items()})
    st.download_button(
        "📥 Download full report (Excel)",
        data=xlsx,
        file_name=f"bist_risk_budgeting_report_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    st.markdown("**Notes**")
    st.markdown(
        "- Data source is Yahoo Finance via yfinance only.\n"
        "- If Yahoo returns empty for a ticker, it is dropped (no synthetic replacement).\n"
        "- Forward-fill is used to bridge short holiday gaps only (limited by the slider)."
    )

