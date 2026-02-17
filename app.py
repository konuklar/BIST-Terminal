
# =============================================================
# 📊 ADVANCED BIST Risk Budgeting System (Streamlit Community Cloud)
#
# HARD REQUIREMENTS (per user):
# - Use ONLY original Yahoo Finance data via yfinance (NO synthetic series).
# - If data gaps exist: use forward-filling (no simulated interpolation).
# - Be careful in data alignment across assets/benchmark/factors.
# - Universe change: remove TRALT/TRALTIN; include ASTOR.IS instead.
# - PyPortfolioOpt MUST be utilized (pinned in requirements) with multiple portfolio strategies.
# - Add:
#   (i) Benchmark-relative (active) risk contributions vs BIST100 (tracking error decomposition)
#   (ii) Stress scenarios module (rate shock / FX shock) using factor betas
# - Include signature text in UI + exports:
#   "The Quantitative Analysis Performed by LabGen25@Istanbul by Murat KONUKLAR 2026"
# =============================================================

from __future__ import annotations

import io
import json
import time
import logging
import warnings
from dataclasses import dataclass
from datetime import date, datetime
from typing import Dict, List, Optional, Tuple, Iterable

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import minimize
from scipy import stats

warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("yfinance").setLevel(logging.CRITICAL)
logging.getLogger("urllib3").setLevel(logging.CRITICAL)

# Optional packages (but required in requirements for Cloud build)
try:
    from pypfopt import EfficientFrontier, risk_models, expected_returns, BlackLittermanModel
    from pypfopt.hierarchical_risk_parity import HRPOpt
    # Optional extensions
    try:
        from pypfopt import EfficientSemivariance
    except Exception:
        EfficientSemivariance = None
    try:
        from pypfopt import EfficientCVaR
    except Exception:
        EfficientCVaR = None
    try:
        from pypfopt import EfficientCDaR
    except Exception:
        EfficientCDaR = None
    PYPFOPT_AVAILABLE = True
except Exception:
    PYPFOPT_AVAILABLE = False
    EfficientSemivariance = None
    EfficientCVaR = None
    EfficientCDaR = None

try:
    import quantstats as qs
    QUANTSTATS_AVAILABLE = True
except Exception:
    QUANTSTATS_AVAILABLE = False

SIGNATURE_TEXT = "The Quantitative Analysis Performed by LabGen25@Istanbul by Murat KONUKLAR 2026"

# -------------------------
# Streamlit page + styling
# -------------------------
st.set_page_config(
    page_title="Advanced BIST Risk Budgeting",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
.main-header { font-size: 2.2rem; font-weight: 800; margin-bottom: 0.15rem; }
.badge { display:inline-block; padding:0.22rem 0.7rem; border-radius:999px;
        background:#0f2a5f; color:#fff; font-size:0.85rem; margin:0.2rem 0 0.3rem 0;}
.sig { display:block; padding:0.35rem 0.7rem; border-radius:12px; background:#f3f4f6;
       color:#111827; font-size:0.92rem; margin:0.25rem 0 0.8rem 0; font-weight:650; }
.sub-header { font-size: 1.35rem; font-weight: 750; margin: 0.85rem 0 0.25rem 0; }
.small-note { color:#6b7280; font-size:0.92rem; }
.kpi-note { color:#6b7280; font-size:0.85rem; margin-top:-6px; }
</style>
""",
    unsafe_allow_html=True,
)

# =============================================================
# Helpers
# =============================================================

def normalize_tickers(tickers: Iterable[str], suffix: str = ".IS") -> List[str]:
    out: List[str] = []
    for t in tickers:
        if t is None:
            continue
        s = str(t).strip().upper()
        if not s:
            continue
        # allow Yahoo index tickers like ^XU100 or TRY=X
        if s.startswith("^") or "=" in s:
            out.append(s)
            continue
        if "." in s:
            out.append(s)
        else:
            out.append(s + suffix)

    # unique preserve order
    seen = set()
    uniq = []
    for s in out:
        if s not in seen:
            uniq.append(s)
            seen.add(s)
    return uniq

def parse_custom_tickers(text: str) -> List[str]:
    if text is None:
        return []
    parts = [p.strip() for p in str(text).replace(",", "\n").splitlines()]
    parts = [p for p in parts if p]
    return normalize_tickers(parts)

def safe_float(x, default=np.nan) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)

def portfolio_returns(returns: pd.DataFrame, weights: np.ndarray) -> pd.Series:
    w = np.asarray(weights, dtype=float).reshape(-1)
    w = np.clip(w, 0, None)
    w = w / (w.sum() if w.sum() != 0 else 1.0)
    pr = returns.values @ w
    return pd.Series(pr, index=returns.index, name="Portfolio")

def sanitize_for_excel(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fixes common Excel export issues:
    - Excel doesn't support timezone-aware datetimes
    - Convert Period and complex objects to strings
    """
    x = df.copy()

    # index
    try:
        if isinstance(x.index, pd.DatetimeIndex) and x.index.tz is not None:
            x.index = x.index.tz_localize(None)
    except Exception:
        pass

    # datetime tz columns
    for c in x.columns:
        try:
            if pd.api.types.is_datetime64tz_dtype(x[c].dtype):
                x[c] = x[c].dt.tz_localize(None)
        except Exception:
            pass

    # Period -> string
    for c in x.columns:
        if pd.api.types.is_period_dtype(x[c].dtype):
            x[c] = x[c].astype(str)

    # object cells that might contain timestamps/dicts/lists
    def _clean_cell(v):
        if isinstance(v, pd.Timestamp):
            try:
                if v.tz is not None:
                    return v.tz_localize(None)
            except Exception:
                return v.to_pydatetime()
            return v.to_pydatetime()
        if isinstance(v, (dict, list, tuple, set)):
            return json.dumps(list(v)) if not isinstance(v, dict) else json.dumps(v)
        return v

    obj_cols = [c for c in x.columns if x[c].dtype == "object"]
    for c in obj_cols:
        x[c] = x[c].map(_clean_cell)

    return x

def to_excel_bytes(sheets: Dict[str, pd.DataFrame]) -> bytes:
    """
    Robust Excel export:
    - sanitizes every sheet for Excel compatibility
    - includes signature sheet
    """
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="xlsxwriter") as writer:
        # signature sheet first
        sig_df = pd.DataFrame({"Signature": [SIGNATURE_TEXT], "GeneratedAt": [datetime.now()]})
        sanitize_for_excel(sig_df).to_excel(writer, sheet_name="Signature", index=False)

        for name, df in sheets.items():
            safe_name = str(name)[:31]
            df2 = sanitize_for_excel(df)
            df2.to_excel(writer, sheet_name=safe_name, index=False)

    bio.seek(0)
    return bio.read()

def linear_beta(y: pd.Series, X: pd.DataFrame) -> pd.Series:
    """
    OLS beta via numpy (no statsmodels dependency):
    y = a + X b
    """
    df = pd.concat([y, X], axis=1).dropna()
    if df.shape[0] < max(30, X.shape[1] + 5):
        return pd.Series([np.nan] * X.shape[1], index=X.columns)

    yy = df.iloc[:, 0].values.reshape(-1, 1)
    XX = df.iloc[:, 1:].values
    # add intercept
    XX = np.column_stack([np.ones((XX.shape[0], 1)), XX])
    try:
        b = np.linalg.lstsq(XX, yy, rcond=None)[0].reshape(-1)
        # b[0] intercept, rest betas
        return pd.Series(b[1:], index=df.columns[1:])
    except Exception:
        return pd.Series([np.nan] * (XX.shape[1] - 1), index=df.columns[1:])

# =============================================================
# Metadata
# =============================================================

@dataclass(frozen=True)
class AssetMeta:
    name: str
    sector: str

# =============================================================
# Data Fetching (Yahoo only; NO synthetic series)
# =============================================================

class BISTDataFetcher:
    """
    Robust Yahoo data fetch:
    1) yf.download batch
    2) per-ticker history fallback
    Cleaning rule: forward-fill only (per requirement).
    Alignment: outer-join date union then forward-fill; then drop leading all-NaNs.
    """

    # Stable default list (TRALT removed; ASTOR added)
    DEFAULT_20 = [
        "AKBNK.IS","ARCLK.IS","ASELS.IS","ASTOR.IS","BIMAS.IS",
        "EKGYO.IS","EREGL.IS","FROTO.IS","GARAN.IS","HALKB.IS",
        "ISCTR.IS","KCHOL.IS","KRDMD.IS","PETKM.IS","PGSUS.IS",
        "SAHOL.IS","SASA.IS","TCELL.IS","THYAO.IS","TOASO.IS",
    ]

    BENCHMARKS = {
        "BIST100 (^XU100)": "^XU100",
        "BIST50 (^XU050)": "^XU050",
        "BIST30 (^XU030)": "^XU030",
        "None": "",
    }

    # Basic mapping for display (expandable)
    META: Dict[str, AssetMeta] = {
        "AKBNK.IS": AssetMeta("Akbank", "Banking"),
        "ARCLK.IS": AssetMeta("Arcelik", "Industrial"),
        "ASELS.IS": AssetMeta("Aselsan", "Defense"),
        "ASTOR.IS": AssetMeta("Astor Enerji", "Industrial"),
        "BIMAS.IS": AssetMeta("BIM", "Retail"),
        "EKGYO.IS": AssetMeta("Emlak Konut", "Real Estate"),
        "EREGL.IS": AssetMeta("Eregli Demir Celik", "Iron & Steel"),
        "FROTO.IS": AssetMeta("Ford Otosan", "Automotive"),
        "GARAN.IS": AssetMeta("Garanti BBVA", "Banking"),
        "HALKB.IS": AssetMeta("Halkbank", "Banking"),
        "ISCTR.IS": AssetMeta("Is Bankasi", "Banking"),
        "KCHOL.IS": AssetMeta("Koc Holding", "Holding"),
        "KRDMD.IS": AssetMeta("Kardemir", "Iron & Steel"),
        "PETKM.IS": AssetMeta("Petkim", "Petrochemical"),
        "PGSUS.IS": AssetMeta("Pegasus", "Aviation"),
        "SAHOL.IS": AssetMeta("Sabanci Holding", "Holding"),
        "SASA.IS": AssetMeta("SASA Polyester", "Chemicals"),
        "TCELL.IS": AssetMeta("Turkcell", "Telecom"),
        "THYAO.IS": AssetMeta("Turkish Airlines", "Aviation"),
        "TOASO.IS": AssetMeta("Tofas", "Automotive"),
    }

    @staticmethod
    @st.cache_data(ttl=3600, show_spinner=False)
    def fetch_bist_list_online(mode: str = "BIST50") -> List[str]:
        """
        Auto-scrape (best-effort) from a public table; fallback is DEFAULT_20.
        Note: component lists can change; this is a convenience, not a guarantee.
        """
        mode = str(mode).upper()
        url = "https://www.oyakyatirim.com.tr/piyasa-verileri/XU050" if mode == "BIST50" else "https://www.oyakyatirim.com.tr/piyasa-verileri/XU100"
        try:
            tables = pd.read_html(url, flavor="lxml")
            df = tables[0]
            sym_col = None
            for c in df.columns:
                if str(c).strip().lower() in ("sembol", "symbol", "kod", "code"):
                    sym_col = c
                    break
            if sym_col is None:
                sym_col = df.columns[0]
            symbols = df[sym_col].astype(str).str.strip().tolist()
            lst = normalize_tickers(symbols, suffix=".IS")
            # enforce removal of TRALT / replace with ASTOR if present
            lst = [t for t in lst if t not in ("TRALT.IS", "TRALTIN.IS", "TRALTIN")]
            if "ASTOR.IS" not in lst:
                lst = ["ASTOR.IS"] + lst
            return lst
        except Exception:
            return BISTDataFetcher.DEFAULT_20

    @staticmethod
    def yahoo_health_check() -> Tuple[bool, str]:
        try:
            test = yf.download("THYAO.IS", period="5d", interval="1d", auto_adjust=True, progress=False, threads=False)
            if test is None or test.empty:
                return False, "Health check returned 0 rows (Yahoo blocked/rate-limited?)."
            return True, f"Health check OK (rows={len(test)})."
        except Exception as e:
            return False, f"Health check error: {e}"

    @staticmethod
    def _extract_close_panel(data: pd.DataFrame, tickers: List[str]) -> pd.DataFrame:
        if data is None or data.empty:
            return pd.DataFrame()
        if isinstance(data.columns, pd.MultiIndex):
            lvl0 = data.columns.get_level_values(0)
            if "Close" in lvl0:
                panel = data["Close"].copy()
            elif "Adj Close" in lvl0:
                panel = data["Adj Close"].copy()
            else:
                return pd.DataFrame()
        else:
            # single ticker structure
            if "Close" in data.columns and len(tickers) == 1:
                panel = pd.DataFrame({tickers[0]: data["Close"]})
            elif "Adj Close" in data.columns and len(tickers) == 1:
                panel = pd.DataFrame({tickers[0]: data["Adj Close"]})
            else:
                return pd.DataFrame()
        if isinstance(panel, pd.Series):
            panel = panel.to_frame()
        panel = panel.dropna(axis=1, how="all")
        return panel

    @staticmethod
    def _align_and_ffill(prices: pd.DataFrame, ffill_limit: int) -> pd.DataFrame:
        """
        Alignment: union of all dates (outer join already in DataFrame columns),
        then forward-fill per column (no backfill).
        """
        x = prices.copy()
        x = x.sort_index()
        x = x.replace([np.inf, -np.inf], np.nan)

        # forward fill only (requirement)
        x = x.ffill(limit=int(ffill_limit))

        # drop leading rows where everything is NaN
        first_valid = x.notna().any(axis=1)
        if first_valid.any():
            x = x.loc[first_valid.idxmax():]
        x = x.dropna(axis=1, how="all")
        return x

    @staticmethod
    def _best_effort_clean_panel(prices: pd.DataFrame, min_obs: int, max_missing_frac: float) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        prices = prices.copy()
        prices = prices.dropna(axis=1, how="all")

        obs = prices.notna().sum().sort_values(ascending=False)
        diag = pd.DataFrame({
            "ObsCount": obs,
            "Start": [prices[c].first_valid_index() for c in obs.index],
            "End": [prices[c].last_valid_index() for c in obs.index],
            "MissingFrac": [float(prices[c].isna().mean()) for c in obs.index],
        }).reset_index().rename(columns={"index": "Ticker"})

        # filter
        keep = diag["Ticker"][(diag["ObsCount"] >= int(min_obs)) & (diag["MissingFrac"] <= float(max_missing_frac))].tolist()
        prices = prices[keep] if keep else prices.iloc[:, 0:0]

        rets = prices.pct_change().replace([np.inf, -np.inf], np.nan).dropna(how="all")
        return prices, rets, diag

    @staticmethod
    def _prune_for_covariance(returns: pd.DataFrame, min_assets: int = 2, max_iter: int = 25) -> Tuple[pd.DataFrame, pd.DataFrame]:
        rets = returns.copy()
        for _ in range(max_iter):
            cov = rets.cov() * 252.0
            bad = cov.columns[cov.isna().any()].tolist()
            if not bad:
                return rets, cov
            rets = rets.drop(columns=bad, errors="ignore")
            if rets.shape[1] < min_assets:
                break
        return rets, rets.cov() * 252.0

    @staticmethod
    @st.cache_data(ttl=3600, show_spinner=False)
    def fetch_prices(
        tickers: Tuple[str, ...],
        start: str,
        end: str,
        min_obs: int,
        max_missing_frac: float,
        max_retries: int,
        ffill_limit: int,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], pd.DataFrame]:
        tickers_l = list(tickers)
        dropped: List[str] = []

        # 1) batch
        data = pd.DataFrame()
        last_err = None
        for k in range(int(max_retries)):
            try:
                data = yf.download(
                    tickers=tickers_l,
                    start=start,
                    end=end,
                    interval="1d",
                    auto_adjust=True,
                    group_by="column",
                    progress=False,
                    threads=False,  # more stable on cloud
                    timeout=30,
                )
                if data is not None and not data.empty:
                    break
            except Exception as e:
                last_err = e
            time.sleep(1.5 * (2 ** k))

        prices = BISTDataFetcher._extract_close_panel(data, tickers_l)

        # 2) per ticker fallback
        missing = [t for t in tickers_l if t not in prices.columns]
        if prices.empty or missing:
            frames: Dict[str, pd.Series] = {}
            for t in tickers_l:
                try:
                    hist = yf.Ticker(t).history(start=start, end=end, interval="1d", auto_adjust=True)
                    if hist is None or hist.empty or "Close" not in hist.columns:
                        dropped.append(t)
                        continue
                    frames[t] = hist["Close"].rename(t)
                except Exception:
                    dropped.append(t)
                time.sleep(0.12)
            if frames:
                prices = pd.concat(frames.values(), axis=1).sort_index()

        if prices is None or prices.empty:
            if last_err:
                raise ValueError(f"No data from Yahoo (batch+fallback failed). Last error: {last_err}")
            raise ValueError("No data from Yahoo (batch+fallback failed).")

        # align & forward-fill only
        prices = BISTDataFetcher._align_and_ffill(prices, ffill_limit=ffill_limit)

        # quality filter
        prices, rets, diag = BISTDataFetcher._best_effort_clean_panel(prices, min_obs=min_obs, max_missing_frac=max_missing_frac)
        if rets.shape[1] < 2:
            raise ValueError("Not enough valid tickers after cleaning (need at least 2).")

        # covariance prune
        rets, cov = BISTDataFetcher._prune_for_covariance(rets, min_assets=2)
        prices = prices[rets.columns.tolist()]

        if rets.shape[1] < 2:
            raise ValueError("Not enough valid tickers after covariance pruning (need at least 2).")

        return prices, rets, sorted(list(set(dropped))), diag

# =============================================================
# Risk + Performance Analytics
# =============================================================

class RiskEngine:
    @staticmethod
    def risk_contributions(returns: pd.DataFrame, weights: np.ndarray) -> Tuple[pd.DataFrame, Dict[str, float], pd.DataFrame]:
        cols = list(returns.columns)
        n = len(cols)

        w = np.asarray(weights, dtype=float).reshape(-1)
        w = np.clip(w, 0, None)
        w = w / (w.sum() if w.sum() != 0 else 1.0)

        cov_df = returns.cov() * 252.0
        cov = cov_df.values

        port_var = float(w @ cov @ w)
        port_vol = float(np.sqrt(max(port_var, 0.0)))

        indiv_vol = np.sqrt(np.clip(np.diag(cov), 0, None))

        if port_vol > 0:
            mrc = (cov @ w) / port_vol
            crc = w * mrc
            rc_pct = (crc / port_vol) * 100.0
        else:
            mrc = np.zeros(n)
            crc = np.zeros(n)
            rc_pct = np.zeros(n)

        rm = pd.DataFrame({
            "Symbol": cols,
            "Weight": w,
            "Individual_Volatility": indiv_vol,
            "Marginal_Risk_Contribution": mrc,
            "Component_Risk": crc,
            "Risk_Contribution_%": rc_pct,
        }).sort_values("Risk_Contribution_%", ascending=False).reset_index(drop=True)
        rm["Risk_Rank"] = np.arange(1, len(rm) + 1)

        wavg_vol = float(np.sum(w * indiv_vol))
        div_ratio = (wavg_vol / port_vol) if port_vol > 0 else np.nan

        pm = {
            "volatility": float(port_vol),
            "avg_volatility": float(np.nanmean(indiv_vol)),
            "diversification_ratio": float(div_ratio),
            "n_assets": int(n),
            "max_risk_contrib": float(rm.iloc[0]["Risk_Contribution_%"]),
            "max_risk_symbol": str(rm.iloc[0]["Symbol"]),
        }
        return rm, pm, cov_df

    @staticmethod
    def active_risk_contributions_vs_benchmark(
        asset_returns: pd.DataFrame,
        weights: np.ndarray,
        benchmark_returns: pd.Series,
    ) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """
        Tracking error (active risk) decomposition:
        active = w' r_assets - 1 * r_benchmark
        Contribution is computed on extended covariance with benchmark leg weight = -1.
        """
        if benchmark_returns is None or benchmark_returns.dropna().empty:
            raise ValueError("Benchmark returns are empty.")

        # strict alignment to avoid leakage
        ext = pd.concat([asset_returns, benchmark_returns.rename("BENCH")], axis=1, join="inner").dropna(how="any")
        if ext.shape[0] < 60 or ext.shape[1] < 3:
            raise ValueError("Not enough aligned history for active risk contributions.")

        cols_assets = list(asset_returns.columns)
        n = len(cols_assets)

        w_assets = np.asarray(weights, dtype=float).reshape(-1)
        w_assets = np.clip(w_assets, 0, None)
        w_assets = w_assets / (w_assets.sum() if w_assets.sum() != 0 else 1.0)

        # extended weights: assets + benchmark leg
        w_ext = np.concatenate([w_assets, np.array([-1.0])])
        cov = ext.cov() * 252.0
        covv = cov.values

        te_var = float(w_ext @ covv @ w_ext)
        te_vol = float(np.sqrt(max(te_var, 0.0)))

        if te_vol <= 0:
            raise ValueError("Tracking error volatility is non-positive.")

        mrc = (covv @ w_ext) / te_vol
        crc = w_ext * mrc
        rc_pct = (crc / te_vol) * 100.0

        symbols = cols_assets + ["BENCH (leg=-1)"]
        df = pd.DataFrame({
            "Leg": symbols,
            "Weight": w_ext,
            "Marginal_Contribution": mrc,
            "Component_Contribution": crc,
            "Active_Risk_Contribution_%": rc_pct,
        }).sort_values("Active_Risk_Contribution_%", ascending=False).reset_index(drop=True)

        te = {
            "tracking_error_vol": te_vol,
            "active_var": te_var,
            "n_obs": int(ext.shape[0]),
        }
        return df, te

    @staticmethod
    def constrained_risk_parity(
        cov_matrix: pd.DataFrame,
        tickers: List[str],
        sectors: Optional[List[str]] = None,
        min_weight: float = 0.0,
        max_weight: float = 0.10,
        sector_caps: Optional[Dict[str, float]] = None,
    ) -> np.ndarray:
        """
        Constrained risk parity (SLSQP):
        - bounds per asset: [min_weight, max_weight]
        - sector caps: sum_{i in sector s} w_i <= cap_s
        """
        n = len(tickers)
        cov = cov_matrix.values if isinstance(cov_matrix, pd.DataFrame) else np.asarray(cov_matrix)

        min_weight = float(min_weight)
        max_weight = float(max_weight)
        if min_weight < 0 or max_weight <= 0 or max_weight > 1 or min_weight >= 1:
            raise ValueError("Invalid min/max weight.")
        if min_weight > max_weight:
            raise ValueError("min_weight cannot exceed max_weight.")
        if min_weight * n > 1.0 + 1e-9:
            raise ValueError("Infeasible constraints: min_weight * N > 1. Reduce min_weight or universe size.")
        if max_weight * n < 1.0 - 1e-9:
            raise ValueError("Infeasible constraints: max_weight * N < 1. Increase max_weight or reduce universe size.")

        def obj(x):
            x = np.asarray(x, dtype=float)
            x = np.clip(x, min_weight, max_weight)
            x = x / (x.sum() if x.sum() != 0 else 1.0)
            var = float(x @ cov @ x)
            vol = float(np.sqrt(max(var, 0.0)))
            if vol <= 0:
                return 1e9
            mrc = (cov @ x) / vol
            rc = x * mrc
            target = vol / n
            return float(np.sum((rc - target) ** 2))

        x0 = np.ones(n) / n
        x0 = np.clip(x0, min_weight, max_weight)
        x0 = x0 / x0.sum()

        bounds = [(min_weight, max_weight) for _ in range(n)]
        constraints = [{"type": "eq", "fun": lambda x: np.sum(x) - 1.0}]

        if sector_caps and sectors and len(sectors) == n:
            caps = {str(k): float(v) for k, v in sector_caps.items()}
            for sname, cap in caps.items():
                cap = float(cap)
                if cap <= 0 or cap > 1:
                    raise ValueError(f"Sector cap for {sname} must be in (0, 1].")
                idx = [i for i, sec in enumerate(sectors) if sec == sname]
                if not idx:
                    continue
                constraints.append({
                    "type": "ineq",
                    "fun": (lambda x, idx=idx, cap=cap: cap - float(np.sum(np.asarray(x)[idx])))
                })

        res = minimize(
            obj,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-10, "maxiter": 5000},
        )

        if res.success and np.all(np.isfinite(res.x)):
            w = np.asarray(res.x, dtype=float)
            w = np.clip(w, min_weight, max_weight)
            w = w / (w.sum() if w.sum() != 0 else 1.0)
            return w

        return x0

def aggregate_horizon_returns(r: pd.Series, horizon_days: int) -> pd.Series:
    if horizon_days <= 1:
        return r.dropna()
    lr = np.log1p(r.dropna())
    agg_lr = lr.rolling(horizon_days).sum().dropna()
    return np.expm1(agg_lr)

def var_es_table(
    r: pd.Series,
    conf_levels: Tuple[float, ...],
    horizon_days: int,
    methods: Tuple[str, ...],
) -> pd.DataFrame:
    r_h = aggregate_horizon_returns(r, horizon_days).replace([np.inf, -np.inf], np.nan).dropna()
    if r_h.empty:
        raise ValueError("Return series empty after cleaning.")

    mu = float(r_h.mean())
    sigma = float(r_h.std(ddof=1))
    s = float(stats.skew(r_h, bias=False)) if len(r_h) > 10 else 0.0
    k = float(stats.kurtosis(r_h, fisher=False, bias=False)) if len(r_h) > 10 else 3.0

    rows = []
    for cl in conf_levels:
        alpha = 1.0 - float(cl)
        if alpha <= 0 or alpha >= 1:
            continue

        if "historical" in methods:
            q = float(r_h.quantile(alpha))
            var_h = -q
            es_h = -float(r_h[r_h <= q].mean())
            rows.append({"Method": "Historical", "CL": cl, "HorizonDays": horizon_days, "VaR": var_h, "ES(CVaR)": es_h})

        if "parametric" in methods:
            z = float(stats.norm.ppf(alpha))
            qn = mu + sigma * z
            var_p = -qn
            es_p = -(mu - sigma * (stats.norm.pdf(z) / alpha))
            rows.append({"Method": "Parametric(N)", "CL": cl, "HorizonDays": horizon_days, "VaR": var_p, "ES(CVaR)": es_p})

        if "modified" in methods:
            z = float(stats.norm.ppf(alpha))
            z_cf = (
                z
                + (1.0 / 6.0) * (z**2 - 1.0) * s
                + (1.0 / 24.0) * (z**3 - 3.0 * z) * (k - 3.0)
                - (1.0 / 36.0) * (2.0 * z**3 - 5.0 * z) * (s**2)
            )
            qcf = mu + sigma * z_cf
            var_m = -qcf
            es_m = -(mu - sigma * (stats.norm.pdf(z_cf) / alpha))
            rows.append({"Method": "Modified(CF)", "CL": cl, "HorizonDays": horizon_days, "VaR": var_m, "ES(CVaR)": es_m})

    out = pd.DataFrame(rows).sort_values(["Method", "CL"]).reset_index(drop=True)
    return out

def rolling_risk_contributions_pct(
    returns: pd.DataFrame,
    weights: np.ndarray,
    window: int,
    step: int,
    annualize: float = 252.0,
) -> pd.DataFrame:
    if window < 20:
        raise ValueError("Rolling window too small (>=20).")
    step = max(1, int(step))

    cols = list(returns.columns)
    n = len(cols)

    w = np.asarray(weights, dtype=float).reshape(-1)
    w = np.clip(w, 0, None)
    w = w / (w.sum() if w.sum() != 0 else 1.0)
    if w.shape[0] != n:
        raise ValueError("weights length must match number of assets.")

    idx = returns.index
    out_dates, out_rc = [], []

    for end_i in range(window - 1, len(idx), step):
        sub = returns.iloc[end_i - window + 1 : end_i + 1].dropna(how="any")
        if len(sub) < max(20, window // 3):
            continue

        cov = sub.cov().values * annualize
        port_var = float(w @ cov @ w)
        port_vol = float(np.sqrt(max(port_var, 0.0)))
        if port_vol <= 0:
            rc_pct = np.zeros(n)
        else:
            mrc = (cov @ w) / port_vol
            crc = w * mrc
            rc_pct = (crc / port_vol) * 100.0

        out_dates.append(idx[end_i])
        out_rc.append(rc_pct)

    if not out_rc:
        return pd.DataFrame()

    rc = pd.DataFrame(np.vstack(out_rc), index=pd.DatetimeIndex(out_dates), columns=cols)
    rc = rc.reindex(pd.DatetimeIndex(idx)).ffill().dropna(how="all")
    return rc

def performance_report(port: pd.Series, benchmark: Optional[pd.Series] = None) -> pd.Series:
    r = port.dropna().copy()
    if r.empty:
        return pd.Series(dtype=float)

    total_return = (1 + r).prod() - 1
    cagr = (1 + total_return) ** (252.0 / len(r)) - 1 if len(r) > 1 else np.nan
    ann_ret = r.mean() * 252.0
    ann_vol = r.std(ddof=1) * np.sqrt(252.0)
    downside = r[r < 0].std(ddof=1) * np.sqrt(252.0) if (r < 0).any() else np.nan
    sharpe = ann_ret / ann_vol if ann_vol and ann_vol > 0 else np.nan
    sortino = ann_ret / downside if downside and downside > 0 else np.nan

    wealth = (1 + r).cumprod()
    dd = wealth / wealth.cummax() - 1
    max_dd = float(dd.min()) if not dd.empty else np.nan
    avg_dd = float(dd[dd < 0].mean()) if (dd < 0).any() else 0.0

    out = {
        "Total Return": float(total_return),
        "CAGR": float(cagr) if np.isfinite(cagr) else np.nan,
        "Annualized Return": float(ann_ret),
        "Annualized Volatility": float(ann_vol),
        "Sharpe Ratio": float(sharpe) if np.isfinite(sharpe) else np.nan,
        "Sortino Ratio": float(sortino) if np.isfinite(sortino) else np.nan,
        "Skewness": float(r.skew()),
        "Kurtosis": float(r.kurtosis()),
        "Max Drawdown": float(max_dd),
        "Avg Drawdown": float(avg_dd),
    }

    if benchmark is not None and not benchmark.dropna().empty:
        b = benchmark.dropna()
        idx = r.index.intersection(b.index)
        if len(idx) >= 30:
            rr = r.loc[idx].values
            bb = b.loc[idx].values
            cov = float(np.cov(rr, bb, ddof=1)[0, 1])
            var_b = float(np.var(bb, ddof=1))
            beta = cov / var_b if var_b > 0 else np.nan
            alpha = (np.mean(rr) - beta * np.mean(bb)) * 252.0 if np.isfinite(beta) else np.nan
            te = float(np.std(rr - bb, ddof=1) * np.sqrt(252.0))
            ir = ((np.mean(rr - bb) * 252.0) / te) if te > 0 else np.nan
            out.update({
                "Beta vs Benchmark": float(beta) if np.isfinite(beta) else np.nan,
                "Alpha vs Benchmark": float(alpha) if np.isfinite(alpha) else np.nan,
                "Tracking Error": float(te),
                "Information Ratio": float(ir) if np.isfinite(ir) else np.nan,
            })
    return pd.Series(out)

# =============================================================
# PyPortfolioOpt wrapper: multiple portfolio strategies
# =============================================================

def pypfopt_optimize(
    method: str,
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    risk_free_rate: float,
    min_weight: float,
    max_weight: float,
    sector_caps: Optional[Dict[str, float]],
    sectors: Optional[List[str]],
    target_return: Optional[float] = None,
    target_volatility: Optional[float] = None,
    utility_gamma: float = 1.0,
    bl_views_json: str = "",
) -> Tuple[np.ndarray, Dict[str, float], str]:
    """
    Returns: weights, metrics, note
    """
    note = ""
    if not PYPFOPT_AVAILABLE:
        return np.ones(returns.shape[1]) / returns.shape[1], {}, "PyPortfolioOpt import failed (should be installed). Using Equal Weight."

    tickers = list(returns.columns)
    n = len(tickers)

    # expected returns
    try:
        mu = expected_returns.mean_historical_return(prices, frequency=252)
    except Exception:
        mu = returns.mean() * 252.0

    # covariance (shrinked)
    try:
        S = risk_models.CovarianceShrinkage(returns).ledoit_wolf()
    except Exception:
        S = risk_models.sample_cov(returns, frequency=252)

    def _add_constraints(ef):
        ef.add_constraint(lambda w: w >= 0)
        if min_weight > 0:
            ef.add_constraint(lambda w: w >= float(min_weight))
        ef.add_constraint(lambda w: w <= float(max_weight))
        if sector_caps and sectors and len(sectors) == n:
            for sec, cap in sector_caps.items():
                idx = [i for i, s in enumerate(sectors) if s == sec]
                if idx:
                    ef.add_constraint(lambda w, idx=idx, cap=float(cap): cap - sum(w[i] for i in idx))

    method_key = str(method).lower().strip()

    try:
        if method_key in ("max sharpe", "max_sharpe", "max-sharpe"):
            ef = EfficientFrontier(mu, S)
            _add_constraints(ef)
            ef.max_sharpe(risk_free_rate=risk_free_rate)
            w = ef.clean_weights()
            perf = ef.portfolio_performance(risk_free_rate=risk_free_rate, verbose=False)
            weights = np.array([w[t] for t in tickers], dtype=float)
            metrics = {"expected_return": float(perf[0]), "volatility": float(perf[1]), "sharpe_ratio": float(perf[2])}
            return weights, metrics, note

        if method_key in ("min volatility", "min_volatility", "min-volatility"):
            ef = EfficientFrontier(mu, S)
            _add_constraints(ef)
            ef.min_volatility()
            w = ef.clean_weights()
            perf = ef.portfolio_performance(risk_free_rate=risk_free_rate, verbose=False)
            weights = np.array([w[t] for t in tickers], dtype=float)
            metrics = {"expected_return": float(perf[0]), "volatility": float(perf[1]), "sharpe_ratio": float(perf[2])}
            return weights, metrics, note

        if method_key in ("max utility", "max_quadratic_utility", "quadratic utility", "utility"):
            ef = EfficientFrontier(mu, S)
            _add_constraints(ef)
            ef.max_quadratic_utility(risk_aversion=float(utility_gamma))
            w = ef.clean_weights()
            perf = ef.portfolio_performance(risk_free_rate=risk_free_rate, verbose=False)
            weights = np.array([w[t] for t in tickers], dtype=float)
            metrics = {"expected_return": float(perf[0]), "volatility": float(perf[1]), "sharpe_ratio": float(perf[2])}
            return weights, metrics, note

        if method_key in ("efficient return", "efficient_return"):
            if target_return is None:
                raise ValueError("Target return is required for efficient_return.")
            ef = EfficientFrontier(mu, S)
            _add_constraints(ef)
            ef.efficient_return(float(target_return))
            w = ef.clean_weights()
            perf = ef.portfolio_performance(risk_free_rate=risk_free_rate, verbose=False)
            weights = np.array([w[t] for t in tickers], dtype=float)
            metrics = {"expected_return": float(perf[0]), "volatility": float(perf[1]), "sharpe_ratio": float(perf[2])}
            return weights, metrics, note

        if method_key in ("efficient risk", "efficient_risk"):
            if target_volatility is None:
                raise ValueError("Target volatility is required for efficient_risk.")
            ef = EfficientFrontier(mu, S)
            _add_constraints(ef)
            ef.efficient_risk(float(target_volatility))
            w = ef.clean_weights()
            perf = ef.portfolio_performance(risk_free_rate=risk_free_rate, verbose=False)
            weights = np.array([w[t] for t in tickers], dtype=float)
            metrics = {"expected_return": float(perf[0]), "volatility": float(perf[1]), "sharpe_ratio": float(perf[2])}
            return weights, metrics, note

        if method_key in ("hrp", "hierarchical risk parity"):
            hrp = HRPOpt(returns.T)
            hrp.optimize()
            w = hrp.clean_weights()
            perf = hrp.portfolio_performance(risk_free_rate=risk_free_rate, verbose=False)
            weights = np.array([w.get(t, 0.0) for t in tickers], dtype=float)
            weights = np.clip(weights, 0, None)
            weights = weights / weights.sum()
            metrics = {"expected_return": float(perf[0]), "volatility": float(perf[1]), "sharpe_ratio": float(perf[2])}
            return weights, metrics, note

        if method_key in ("black-litterman", "black litterman", "bl"):
            views = {}
            if bl_views_json.strip():
                try:
                    views = json.loads(bl_views_json)
                    if not isinstance(views, dict):
                        views = {}
                except Exception:
                    views = {}
            if not views:
                note = "BL views missing/invalid; ran Max Sharpe on standard estimates instead."
                return pypfopt_optimize("max_sharpe", prices, returns, risk_free_rate, min_weight, max_weight, sector_caps, sectors)

            market_caps = pd.Series(np.ones(n), index=tickers)
            bl = BlackLittermanModel(S, pi="market", market_caps=market_caps, absolute_views=views)
            post_mu = bl.bl_returns()
            post_S = bl.bl_cov()
            ef = EfficientFrontier(post_mu, post_S)
            _add_constraints(ef)
            ef.max_sharpe(risk_free_rate=risk_free_rate)
            w = ef.clean_weights()
            perf = ef.portfolio_performance(risk_free_rate=risk_free_rate, verbose=False)
            weights = np.array([w[t] for t in tickers], dtype=float)
            metrics = {"expected_return": float(perf[0]), "volatility": float(perf[1]), "sharpe_ratio": float(perf[2])}
            return weights, metrics, note

        if method_key in ("cvar", "efficientcvar") and EfficientCVaR is not None:
            ef = EfficientCVaR(mu, returns)  # uses return scenarios
            _add_constraints(ef)
            ef.min_cvar()
            w = ef.clean_weights()
            weights = np.array([w[t] for t in tickers], dtype=float)
            # approximate perf using mean/cov
            port_ret = float((returns.mean() * 252.0).values @ weights)
            port_vol = float(np.sqrt(weights @ (returns.cov() * 252.0).values @ weights))
            metrics = {"expected_return": port_ret, "volatility": port_vol, "sharpe_ratio": port_ret / port_vol if port_vol > 0 else np.nan}
            note = "EfficientCVaR used (min CVaR)."
            return weights, metrics, note

        if method_key in ("semivariance", "efficientsemivariance") and EfficientSemivariance is not None:
            ef = EfficientSemivariance(mu, returns)
            _add_constraints(ef)
            ef.min_semivariance()
            w = ef.clean_weights()
            weights = np.array([w[t] for t in tickers], dtype=float)
            port_ret = float((returns.mean() * 252.0).values @ weights)
            port_vol = float(np.sqrt(weights @ (returns.cov() * 252.0).values @ weights))
            metrics = {"expected_return": port_ret, "volatility": port_vol, "sharpe_ratio": port_ret / port_vol if port_vol > 0 else np.nan}
            note = "EfficientSemivariance used (min semivariance)."
            return weights, metrics, note

        if method_key in ("cdar", "efficientcdar") and EfficientCDaR is not None:
            ef = EfficientCDaR(mu, returns)
            _add_constraints(ef)
            ef.min_cdar()
            w = ef.clean_weights()
            weights = np.array([w[t] for t in tickers], dtype=float)
            port_ret = float((returns.mean() * 252.0).values @ weights)
            port_vol = float(np.sqrt(weights @ (returns.cov() * 252.0).values @ weights))
            metrics = {"expected_return": port_ret, "volatility": port_vol, "sharpe_ratio": port_ret / port_vol if port_vol > 0 else np.nan}
            note = "EfficientCDaR used (min CDaR)."
            return weights, metrics, note

        return np.ones(n) / n, {}, "Unknown PyPortfolioOpt method; using Equal Weight."

    except Exception as e:
        return np.ones(n) / n, {}, f"PyPortfolioOpt failed: {e}. Using Equal Weight."

# =============================================================
# Plotly visuals
# =============================================================

def fig_cum_and_drawdown(port: pd.Series, bench: Optional[pd.Series]) -> go.Figure:
    pr = port.dropna()
    cum = (1 + pr).cumprod()
    dd = cum / cum.cummax() - 1

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.10,
                        subplot_titles=("Cumulative Growth (1.0=Start)", "Drawdown"))
    fig.add_trace(go.Scatter(x=cum.index, y=cum.values, name="Portfolio"), row=1, col=1)

    if bench is not None and not bench.dropna().empty:
        b = bench.dropna()
        idx = cum.index.intersection(b.index)
        if len(idx) > 10:
            bcum = (1 + b.loc[idx]).cumprod()
            fig.add_trace(go.Scatter(x=bcum.index, y=bcum.values, name="Benchmark", line=dict(dash="dash")), row=1, col=1)

    fig.add_trace(go.Scatter(x=dd.index, y=(dd.values * 100), name="Drawdown %", fill="tozeroy"), row=2, col=1)
    fig.update_layout(height=560, hovermode="x unified", margin=dict(l=10, r=10, t=60, b=10))
    fig.update_yaxes(title_text="Growth", row=1, col=1)
    fig.update_yaxes(title_text="DD (%)", row=2, col=1)
    return fig

def fig_risk_contrib_bar(risk_df: pd.DataFrame, name_map: Dict[str, str], top_n: int = 20, title: str = "Risk Contribution by Asset") -> go.Figure:
    df = risk_df.copy()
    df["Label"] = df["Symbol"].map(lambda s: name_map.get(s, s))
    metric_col = "Risk_Contribution_%" if "Risk_Contribution_%" in df.columns else df.columns[-1]
    df = df.sort_values(metric_col, ascending=True).tail(top_n)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=df["Label"], x=df[metric_col], orientation="h",
        marker=dict(color=df[metric_col], colorscale="RdYlGn_r", showscale=True,
                    colorbar=dict(title="Risk %")),
        text=df[metric_col].round(1).astype(str) + "%",
        textposition="outside",
        name="Contribution",
    ))
    fig.update_layout(height=680, title=title, margin=dict(l=10, r=10, t=60, b=10))
    return fig

def fig_corr_heatmap(returns: pd.DataFrame, max_assets: int = 35) -> go.Figure:
    cols = list(returns.columns)
    if len(cols) > max_assets:
        cols = cols[:max_assets]
    corr = returns[cols].corr()
    fig = go.Figure(data=go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.columns, colorscale="RdBu", zmid=0
    ))
    fig.update_layout(height=720, title=f"Correlation Heatmap (first {len(cols)} assets)", margin=dict(l=10, r=10, t=60, b=10))
    return fig

def fig_var_table(var_df: pd.DataFrame) -> go.Figure:
    df = var_df.copy()
    df["VaR_%"] = df["VaR"] * 100.0
    df["ES_%"] = df["ES(CVaR)"] * 100.0
    df["CL"] = df["CL"].map(lambda x: f"{int(round(x*100))}%")
    fig = go.Figure(data=[go.Table(
        header=dict(values=["Method", "CL", "HorizonDays", "VaR (%)", "ES/CVaR (%)"]),
        cells=dict(values=[
            df["Method"].tolist(),
            df["CL"].tolist(),
            df["HorizonDays"].astype(int).tolist(),
            [f"{x:.2f}" for x in df["VaR_%"].tolist()],
            [f"{x:.2f}" for x in df["ES_%"].tolist()],
        ])
    )])
    fig.update_layout(height=320, title="VaR / ES (loss magnitude)", margin=dict(l=10, r=10, t=60, b=10))
    return fig

def fig_rolling_rc(rc_pct: pd.DataFrame, name_map: Dict[str, str], top_n: int = 10) -> go.Figure:
    if rc_pct is None or rc_pct.empty:
        return go.Figure()
    avg = rc_pct.mean(axis=0).sort_values(ascending=False)
    top_cols = avg.head(top_n).index.tolist()
    fig = go.Figure()
    for c in top_cols:
        fig.add_trace(go.Scatter(x=rc_pct.index, y=rc_pct[c], mode="lines", name=name_map.get(c, c)))
    fig.update_layout(height=520, title=f"Rolling Risk Contributions (%) — Top {top_n}",
                      hovermode="x unified", margin=dict(l=10, r=10, t=60, b=10))
    return fig

# =============================================================
# Stress module (rate shock / FX shock) using factor betas
# =============================================================

def compute_factor_betas(
    asset_returns: pd.DataFrame,
    factor_returns: pd.DataFrame,
) -> pd.DataFrame:
    """
    Returns betas per asset vs each factor (aligned, no look-ahead).
    """
    betas = []
    for a in asset_returns.columns:
        b = linear_beta(asset_returns[a], factor_returns)
        row = {"Symbol": a}
        for f in factor_returns.columns:
            row[f"Beta_{f}"] = safe_float(b.get(f))
        betas.append(row)
    return pd.DataFrame(betas)

def stress_impact(
    weights: pd.Series,
    betas_df: pd.DataFrame,
    fx_factor: str,
    rate_factor: str,
    fx_shock_pct: float,
    rate_shock_pct: float,
) -> pd.DataFrame:
    """
    Apply shock (as % factor return shock) to estimate asset impacts:
    impact_i ≈ beta_fx_i * fx_shock + beta_rate_i * rate_shock
    """
    df = betas_df.copy()
    df["Weight"] = df["Symbol"].map(weights.to_dict()).fillna(0.0)

    bfx = df.get(f"Beta_{fx_factor}", np.nan)
    br = df.get(f"Beta_{rate_factor}", np.nan)

    fx_shock = float(fx_shock_pct) / 100.0
    rate_shock = float(rate_shock_pct) / 100.0

    df["ShockImpact"] = bfx * fx_shock + br * rate_shock
    df["WeightedImpact"] = df["Weight"] * df["ShockImpact"]

    out = df[["Symbol", "Weight", f"Beta_{fx_factor}", f"Beta_{rate_factor}", "ShockImpact", "WeightedImpact"]].copy()
    out = out.sort_values("WeightedImpact", ascending=False)
    return out

# =============================================================
# App
# =============================================================

def main():
    st.markdown('<div class="main-header">📊 Advanced BIST Risk Budgeting System</div>', unsafe_allow_html=True)
    st.markdown('<div class="badge">📡 Data Source: Yahoo Finance (yfinance)</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sig">{SIGNATURE_TEXT}</div>', unsafe_allow_html=True)

    fetcher = BISTDataFetcher()
    ok, msg = fetcher.yahoo_health_check()

    with st.sidebar:
        st.markdown("## ⚙️ Controls")
        st.markdown(f"**Yahoo Health:** {'✅' if ok else '⚠️'}")
        st.caption(msg)

        st.markdown("### Universe")
        uni_mode = st.selectbox("Select universe", ["AUTO_BIST50", "AUTO_BIST100", "Default 20 (stable)", "Custom list"])
        custom_txt = ""
        if uni_mode == "Custom list":
            custom_txt = st.text_area(
                "Tickers (comma/newline). You may omit .IS.",
                value="AKBNK,ARCLK,ASELS,ASTOR,BIMAS,EKGYO,EREGL,FROTO,GARAN,HALKB,ISCTR,KCHOL,KRDMD,PETKM,PGSUS,SAHOL,SASA,TCELL,THYAO,TOASO",
            )

        st.markdown("### Benchmark")
        bench_choice = st.selectbox("Benchmark", list(fetcher.BENCHMARKS.keys()), index=0)
        benchmark_ticker = fetcher.BENCHMARKS[bench_choice]

        st.markdown("### Dates")
        today = date.today()
        start_date = st.date_input("Start date", value=date(2022, 1, 1))
        end_date = st.date_input("End date", value=today)
        if start_date >= end_date:
            st.error("Start date must be before end date.")
            st.stop()

        st.markdown("### Data quality & alignment")
        min_obs = st.slider("Min observations per ticker", 30, 400, 120, step=10)
        max_missing_frac = st.slider("Max missing fraction", 0.05, 0.80, 0.35, step=0.05)
        ffill_limit = st.slider("Forward-fill limit (days)", 1, 20, 5, step=1)
        max_retries = st.slider("Yahoo retries", 1, 6, 3, step=1)

        st.divider()

        st.markdown("### Portfolio method (PyPortfolioOpt enabled)")
        if not PYPFOPT_AVAILABLE:
            st.error("PyPortfolioOpt is not available in this environment. Check requirements/build logs.")
        methods = [
            "Equal Weight",
            "Risk Parity (Constrained)",
            "PyPortfolioOpt: Max Sharpe",
            "PyPortfolioOpt: Min Volatility",
            "PyPortfolioOpt: Efficient Return",
            "PyPortfolioOpt: Efficient Risk",
            "PyPortfolioOpt: Max Utility",
            "PyPortfolioOpt: HRP",
            "PyPortfolioOpt: Black-Litterman",
        ]
        if EfficientCVaR is not None:
            methods.append("PyPortfolioOpt: CVaR (min)")
        if EfficientSemivariance is not None:
            methods.append("PyPortfolioOpt: Semivariance (min)")
        if EfficientCDaR is not None:
            methods.append("PyPortfolioOpt: CDaR (min)")
        method = st.selectbox("Optimization", methods, index=0)

        st.markdown("### Constraints")
        min_weight = st.slider("Min weight per stock", 0.0, 0.05, 0.0, step=0.005)
        max_weight = st.slider("Max weight per stock", 0.02, 0.30, 0.10, step=0.01)

        st.markdown("### Sector caps (optional)")
        st.caption("Upload sector map CSV with columns: ticker, sector. Then caps JSON, e.g. {\"Banking\":0.25}.")
        sector_file = st.file_uploader("Sector map CSV", type=["csv"])
        sector_caps_json = st.text_area("Sector caps JSON", value="", height=80)

        st.markdown("### Risk-free rate (annual, for Sharpe)")
        risk_free_rate = st.slider("Risk-free rate", 0.00, 0.60, 0.15, step=0.01)

        # PyPortfolioOpt extra knobs
        target_return = None
        target_vol = None
        utility_gamma = 1.0
        bl_views = ""

        if "Efficient Return" in method:
            st.caption("Target return is annualized (e.g., 0.35 = 35% expected annual return).")
            target_return = st.slider("Target annual return", 0.05, 1.50, 0.35, step=0.01)

        if "Efficient Risk" in method:
            st.caption("Target volatility is annualized (e.g., 0.35 = 35% annual vol).")
            target_vol = st.slider("Target annual volatility", 0.05, 1.50, 0.35, step=0.01)

        if "Max Utility" in method:
            utility_gamma = st.slider("Utility risk aversion (gamma)", 0.1, 10.0, 1.0, step=0.1)

        if "Black-Litterman" in method:
            st.caption("JSON dict of absolute expected returns (annual), e.g. {\"AKBNK.IS\":0.30,\"THYAO.IS\":0.25}")
            bl_views = st.text_area("BL Views JSON", value="", height=90)

        st.divider()

        st.markdown("### VaR / ES")
        var_horizon = st.slider("Horizon (days)", 1, 20, 1, step=1)
        cl_sel = st.multiselect("Confidence levels", options=[0.90, 0.95, 0.99], default=[0.95, 0.99])
        methods_sel = st.multiselect("Methods", options=["historical", "parametric", "modified"], default=["historical", "parametric", "modified"])

        st.divider()

        st.markdown("### Rolling risk contributions")
        compute_rolling = st.checkbox("Compute rolling RC", value=True)
        roll_window = st.slider("Window", 20, 260, 63, step=5)
        roll_step = st.slider("Step", 1, 20, 5, step=1)
        roll_topn = st.slider("Top N", 3, 25, 10, step=1)

        st.divider()

        st.markdown("### Stress scenarios (rate shock / FX shock)")
        st.caption("Uses factor betas estimated from Yahoo series (no simulation). Shocks are applied as % factor return shocks.")
        fx_factor = st.text_input("FX factor ticker (USD/TRY)", value="TRY=X")
        rate_factor = st.text_input("Rate factor ticker (proxy)", value="^TNX")
        fx_shock = st.slider("FX shock (%)", -30.0, 30.0, 5.0, step=0.5)
        rate_shock = st.slider("Rate shock (%)", -10.0, 10.0, 1.0, step=0.25)

        st.divider()
        show_quantstats = st.checkbox("Generate QuantStats HTML (optional)", value=False, disabled=not QUANTSTATS_AVAILABLE)
        run_btn = st.button("🚀 Run", use_container_width=True)

    if not run_btn:
        st.info("Configure on the left, then click **Run**.")
        st.caption("Tip: If Yahoo blocks your Cloud app, reduce universe size or try a later start date.")
        return

    # --------
    # Universe
    # --------
    if uni_mode == "Default 20 (stable)":
        universe = fetcher.DEFAULT_20
        uni_note = "Default 20"
    elif uni_mode == "AUTO_BIST100":
        universe = fetcher.fetch_bist_list_online("BIST100")
        uni_note = "AUTO_BIST100 (scraped best-effort)"
    elif uni_mode == "AUTO_BIST50":
        universe = fetcher.fetch_bist_list_online("BIST50")
        uni_note = "AUTO_BIST50 (scraped best-effort)"
    else:
        universe = parse_custom_tickers(custom_txt)
        uni_note = "Custom"

    # enforce removal of TRALT and presence of ASTOR
    universe = [t for t in normalize_tickers(universe) if t not in ("TRALT.IS", "TRALTIN.IS")]
    if "ASTOR.IS" not in universe:
        universe = ["ASTOR.IS"] + universe

    # ------
    # Sector map (optional)
    # ------
    sector_map: Optional[Dict[str, str]] = None
    if sector_file is not None:
        try:
            df_sec = pd.read_csv(sector_file)
            tcol = scol = None
            for c in df_sec.columns:
                lc = str(c).strip().lower()
                if lc in ("ticker", "symbol", "sembol", "kod", "code"):
                    tcol = c
                if lc in ("sector", "sektor", "industry"):
                    scol = c
            if tcol is None or scol is None:
                raise ValueError("CSV must include ticker and sector columns.")
            sector_map = {}
            for _, row in df_sec[[tcol, scol]].dropna().iterrows():
                tk = normalize_tickers([row[tcol]])[0]
                if tk in ("TRALT.IS", "TRALTIN.IS"):
                    continue
                sector_map[tk] = str(row[scol])
        except Exception as e:
            st.error(f"Sector CSV parse error: {e}")
            st.stop()

    sector_caps: Optional[Dict[str, float]] = None
    if sector_caps_json.strip():
        try:
            x = json.loads(sector_caps_json)
            if not isinstance(x, dict):
                raise ValueError("Sector caps must be a JSON object/dict.")
            sector_caps = {str(k): float(v) for k, v in x.items()}
        except Exception as e:
            st.error(f"Sector caps JSON error: {e}")
            st.stop()

    # --------
    # Download data
    # --------
    st.markdown('<div class="sub-header">📥 Data download</div>', unsafe_allow_html=True)
    st.caption(f"Universe: **{uni_note}** • requested tickers: **{len(universe)}**")

    fetch_tickers = universe.copy()
    if benchmark_ticker and benchmark_ticker not in fetch_tickers:
        fetch_tickers += [benchmark_ticker]

    # stress factors
    fx_factor_n = normalize_tickers([fx_factor], suffix=".IS")[0] if fx_factor else ""
    rate_factor_n = normalize_tickers([rate_factor], suffix=".IS")[0] if rate_factor else ""
    factor_tickers = []
    if fx_factor_n and fx_factor_n not in fetch_tickers:
        factor_tickers.append(fx_factor_n)
    if rate_factor_n and rate_factor_n not in fetch_tickers and rate_factor_n != fx_factor_n:
        factor_tickers.append(rate_factor_n)

    fetch_tickers_all = fetch_tickers + factor_tickers

    with st.spinner("Downloading from Yahoo Finance..."):
        try:
            prices_all, returns_all, dropped, diag = fetcher.fetch_prices(
                tuple(fetch_tickers_all),
                start=str(start_date),
                end=str(end_date),
                min_obs=int(min_obs),
                max_missing_frac=float(max_missing_frac),
                max_retries=int(max_retries),
                ffill_limit=int(ffill_limit),
            )
        except Exception as e:
            st.error(f"❌ Data error: {e}")
            st.info("Try: later start date, lower Min observations, or use Default 20.")
            st.stop()

    st.markdown('<div class="sub-header">🔎 Diagnostics</div>', unsafe_allow_html=True)
    st.dataframe(diag.sort_values("ObsCount", ascending=False), use_container_width=True, height=300)
    if dropped:
        st.warning(f"⚠️ Dropped/no-data tickers (raw fetch): {dropped}")

    # Split benchmark vs assets
    benchmark_prices = None
    benchmark_returns = None
    if benchmark_ticker and benchmark_ticker in prices_all.columns:
        benchmark_prices = prices_all[benchmark_ticker].dropna()
        benchmark_returns = benchmark_prices.pct_change().dropna()

    # factors
    factor_prices = {}
    if fx_factor_n and fx_factor_n in prices_all.columns:
        factor_prices[fx_factor_n] = prices_all[fx_factor_n].dropna()
    if rate_factor_n and rate_factor_n in prices_all.columns:
        factor_prices[rate_factor_n] = prices_all[rate_factor_n].dropna()

    factor_returns = None
    if factor_prices:
        # careful alignment later (inner join)
        factor_returns = pd.DataFrame({k: v.pct_change() for k, v in factor_prices.items()}).dropna(how="all")

    # Assets only
    drop_cols = [c for c in [benchmark_ticker, fx_factor_n, rate_factor_n] if c]
    prices = prices_all.drop(columns=drop_cols, errors="ignore")
    returns = returns_all.drop(columns=drop_cols, errors="ignore")

    if returns.shape[1] < 2:
        st.error("❌ Not enough assets after cleaning (need at least 2).")
        st.stop()

    st.success(f"✅ Loaded {returns.shape[1]} assets • {len(returns)} trading days")

    # Build sector list aligned with returns.columns
    loaded = list(returns.columns)
    sectors = []
    for t in loaded:
        if sector_map and t in sector_map:
            sectors.append(str(sector_map[t]))
        else:
            sectors.append(fetcher.META.get(t, AssetMeta(t, "Other")).sector)

    name_map = {t: fetcher.META.get(t, AssetMeta(t, "Other")).name for t in loaded}

    # =========================================================
    # Optimize weights
    # =========================================================
    st.markdown('<div class="sub-header">⚖️ Portfolio optimization</div>', unsafe_allow_html=True)

    weights = np.ones(len(loaded)) / len(loaded)
    opt_metrics: Dict[str, float] = {}
    opt_note = ""

    # initial risk metrics for covariance
    risk_df0, pm0, cov_df = RiskEngine.risk_contributions(returns, weights)

    try:
        if method == "Equal Weight":
            weights = np.ones(len(loaded)) / len(loaded)

        elif method == "Risk Parity (Constrained)":
            weights = RiskEngine.constrained_risk_parity(
                cov_matrix=cov_df,
                tickers=loaded,
                sectors=sectors,
                min_weight=float(min_weight),
                max_weight=float(max_weight),
                sector_caps=sector_caps,
            )

        else:
            # PyPortfolioOpt: always used for these strategies
            pmethod = method.split(":", 1)[1].strip()
            weights, opt_metrics, opt_note = pypfopt_optimize(
                method=pmethod,
                prices=prices[loaded],
                returns=returns[loaded],
                risk_free_rate=float(risk_free_rate),
                min_weight=float(min_weight),
                max_weight=float(max_weight),
                sector_caps=sector_caps,
                sectors=sectors,
                target_return=target_return,
                target_volatility=target_vol,
                utility_gamma=float(utility_gamma),
                bl_views_json=bl_views,
            )

        weights = np.asarray(weights, dtype=float).reshape(-1)
        weights = np.clip(weights, 0, None)
        weights = weights / (weights.sum() if weights.sum() != 0 else 1.0)

        # soft post-check bounds (do not create infeasible solutions)
        weights = np.clip(weights, float(min_weight), float(max_weight))
        weights = weights / weights.sum()

    except Exception as e:
        st.warning(f"Optimization failed: {e}. Falling back to Equal Weight.")
        weights = np.ones(len(loaded)) / len(loaded)

    if opt_note:
        st.info(opt_note)

    # =========================================================
    # Risk metrics + performance
    # =========================================================
    risk_df, pm, cov_df2 = RiskEngine.risk_contributions(returns, weights)
    port = portfolio_returns(returns.fillna(0.0), weights)

    # align benchmark to portfolio (inner join)
    bench_aligned = None
    port_aligned = port
    if benchmark_returns is not None and not benchmark_returns.empty:
        idx = port.index.intersection(benchmark_returns.index)
        if len(idx) >= 60:
            bench_aligned = benchmark_returns.loc[idx]
            port_aligned = port.loc[idx]

    perf = performance_report(port_aligned, bench_aligned)

    # VaR/ES
    var_tbl = var_es_table(
        port,
        conf_levels=tuple(cl_sel) if cl_sel else (0.95, 0.99),
        horizon_days=int(var_horizon),
        methods=tuple(methods_sel) if methods_sel else ("historical",),
    )

    # rolling RC
    rc_roll = pd.DataFrame()
    if compute_rolling:
        with st.spinner("Computing rolling risk contributions..."):
            try:
                rc_roll = rolling_risk_contributions_pct(returns, weights, window=int(roll_window), step=int(roll_step))
            except Exception:
                rc_roll = pd.DataFrame()

    # Active risk contributions
    active_rc = None
    active_stats = None
    if bench_aligned is not None:
        try:
            active_rc, active_stats = RiskEngine.active_risk_contributions_vs_benchmark(returns.loc[port_aligned.index], weights, bench_aligned)
        except Exception:
            active_rc, active_stats = None, None

    # Stress scenarios
    stress_betas = None
    stress_table = None
    if factor_returns is not None and not factor_returns.empty:
        # align factors to assets (inner join)
        fact = factor_returns.copy()
        common = returns.index.intersection(fact.index)
        if len(common) >= 90 and fact.shape[1] >= 1:
            fact = fact.loc[common].dropna(how="any")
            asset_r = returns.loc[fact.index].dropna(how="any")
            if asset_r.shape[0] >= 90 and asset_r.shape[1] >= 2:
                stress_betas = compute_factor_betas(asset_r, fact)
                weights_s = pd.Series(weights, index=loaded, name="Weight")
                # pick factor keys (must exist)
                fx_k = fx_factor_n if fx_factor_n in fact.columns else fact.columns[0]
                rate_k = rate_factor_n if (rate_factor_n in fact.columns and rate_factor_n != fx_k) else (fact.columns[1] if fact.shape[1] > 1 else fx_k)
                stress_table = stress_impact(weights_s, stress_betas, fx_k, rate_k, fx_shock, rate_shock)

    # =========================================================
    # KPIs
    # =========================================================
    st.markdown('<div class="sub-header">📌 Key Metrics</div>', unsafe_allow_html=True)

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Ann. Volatility", f"{pm['volatility']*100:.2f}%")
    k2.metric("Diversification Ratio", f"{pm['diversification_ratio']:.2f}")
    k3.metric("Top Risk Contributor", name_map.get(pm["max_risk_symbol"], pm["max_risk_symbol"]), f"{pm['max_risk_contrib']:.1f}%")
    k4.metric("Sharpe (approx)", f"{safe_float(perf.get('Sharpe Ratio', np.nan)):.2f}")
    k5.metric("Max Drawdown", f"{safe_float(perf.get('Max Drawdown', np.nan))*100:.2f}%")

    if bench_choice != "None" and ("Beta vs Benchmark" in perf.index):
        st.caption(
            f"Benchmark: {bench_choice} • Beta={safe_float(perf.get('Beta vs Benchmark')):.2f} • "
            f"TE={safe_float(perf.get('Tracking Error'))*100:.2f}% • IR={safe_float(perf.get('Information Ratio')):.2f}"
        )

    # =========================================================
    # Tabs
    # =========================================================
    tabs = st.tabs([
        "Performance",
        "Risk Contributions",
        "Active Risk vs Benchmark",
        "VaR / ES",
        "Correlation",
        "Rolling RC",
        "Stress Scenarios",
        "Weights & Tables",
        "Export",
    ])

    with tabs[0]:
        st.plotly_chart(fig_cum_and_drawdown(port_aligned, bench_aligned), use_container_width=True)
        st.markdown("**Performance Report (QuantStats-style)**")
        pr_df = perf.to_frame(name="Value").reset_index().rename(columns={"index": "Metric"})
        pr_df["Value"] = pr_df["Value"].apply(lambda x: f"{x:.6f}" if isinstance(x, (int, float, np.floating)) and np.isfinite(x) else str(x))
        st.dataframe(pr_df, use_container_width=True, height=420)

        if show_quantstats and QUANTSTATS_AVAILABLE:
            st.markdown("**QuantStats HTML Report (may take time)**")
            try:
                html = qs.reports.html(port_aligned, benchmark=bench_aligned, output=None, rf=float(risk_free_rate), title="BIST Portfolio Report")
                st.components.v1.html(html, height=900, scrolling=True)
            except Exception as e:
                st.warning(f"QuantStats report failed: {e}")

    with tabs[1]:
        st.plotly_chart(fig_risk_contrib_bar(risk_df, name_map, top_n=min(30, len(risk_df))), use_container_width=True)
        rm_out = risk_df.copy()
        rm_out["Company"] = rm_out["Symbol"].map(lambda s: name_map.get(s, s))
        rm_out["Sector"] = [sectors[loaded.index(s)] if s in loaded else "Other" for s in rm_out["Symbol"]]
        st.dataframe(
            rm_out[["Risk_Rank","Symbol","Company","Sector","Weight","Risk_Contribution_%","Individual_Volatility","Marginal_Risk_Contribution"]],
            use_container_width=True,
            height=560
        )

    with tabs[2]:
        st.markdown("**Active Risk (Tracking Error) Contributions**")
        if active_rc is None:
            st.info("Active risk panel requires benchmark data and sufficient aligned history.")
        else:
            st.caption(f"Aligned observations: {active_stats.get('n_obs', '-')}, Tracking Error Vol: {active_stats.get('tracking_error_vol', np.nan)*100:.2f}%")
            # build label map with benchmark leg
            am = dict(name_map)
            am["BENCH (leg=-1)"] = f"{bench_choice} leg (-1)"
            st.plotly_chart(
                fig_risk_contrib_bar(
                    active_rc.rename(columns={"Leg": "Symbol", "Active_Risk_Contribution_%": "Risk_Contribution_%"}),
                    am,
                    top_n=min(30, len(active_rc)),
                    title="Active Risk Contribution (%) — assets + benchmark leg",
                ),
                use_container_width=True,
            )
            st.dataframe(active_rc, use_container_width=True, height=520)

    with tabs[3]:
        st.plotly_chart(fig_var_table(var_tbl), use_container_width=True)
        st.caption("VaR/ES are shown as **loss magnitudes**. Example: VaR=0.03 means a 3% loss threshold.")

    with tabs[4]:
        st.plotly_chart(fig_corr_heatmap(returns, max_assets=35), use_container_width=True)

    with tabs[5]:
        if compute_rolling and (rc_roll is not None) and (not rc_roll.empty):
            st.plotly_chart(fig_rolling_rc(rc_roll, name_map, top_n=int(roll_topn)), use_container_width=True)
            st.caption("Rolling RC (%) is computed with rolling covariance and fixed weights.")
        else:
            st.info("Rolling RC is empty (try smaller window/step, or ensure enough data).")

    with tabs[6]:
        st.markdown("**Stress Scenarios (FX + Rate)**")
        if stress_table is None or stress_betas is None:
            st.info("Stress requires factor data (FX/rate tickers) and enough aligned history.")
        else:
            st.caption("Shocks apply as % factor return shocks. Results are linear beta approximations (no simulation).")
            st.dataframe(stress_table, use_container_width=True, height=520)
            port_shock = stress_table["WeightedImpact"].sum()
            st.metric("Estimated Portfolio Impact", f"{port_shock*100:.2f}%")

    with tabs[7]:
        w_df = pd.DataFrame({
            "Symbol": loaded,
            "Company": [name_map.get(t, t) for t in loaded],
            "Sector": sectors,
            "Weight(%)": np.round(weights * 100.0, 6),
        }).sort_values("Weight(%)", ascending=False)
        st.dataframe(w_df, use_container_width=True, height=520)

        st.markdown("**Download CSVs**")
        c1, c2 = st.columns(2)
        with c1:
            st.download_button(
                "⬇️ Weights (CSV)",
                data=w_df.to_csv(index=False).encode("utf-8"),
                file_name=f"bist_weights_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
            )
        with c2:
            risk_export = rm_out.copy()
            st.download_button(
                "⬇️ Risk metrics (CSV)",
                data=risk_export.to_csv(index=False).encode("utf-8"),
                file_name=f"bist_risk_metrics_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
            )

    with tabs[8]:
        st.markdown("**Excel report (one file)**")
        sheets = {
            "Weights": pd.DataFrame({
                "Symbol": loaded,
                "Company": [name_map.get(t, t) for t in loaded],
                "Sector": sectors,
                "Weight": weights,
            }),
            "RiskMetrics": rm_out.copy(),
            "PerfReport": perf.to_frame(name="Value").reset_index().rename(columns={"index": "Metric"}),
            "VaR_ES": var_tbl.copy(),
            "Diagnostics": diag.copy(),
        }
        if bench_aligned is not None:
            sheets["BenchmarkReturns"] = pd.DataFrame({"Date": bench_aligned.index, "BenchmarkReturn": bench_aligned.values})
        if active_rc is not None:
            sheets["ActiveRiskRC"] = active_rc.copy()
        if compute_rolling and rc_roll is not None and not rc_roll.empty:
            sheets["RollingRC"] = rc_roll.reset_index().rename(columns={"index": "Date"})
        if stress_betas is not None:
            sheets["StressBetas"] = stress_betas.copy()
        if stress_table is not None:
            sheets["StressImpact"] = stress_table.copy()

        try:
            xlsx = to_excel_bytes({k: v for k, v in sheets.items()})
            st.download_button(
                "⬇️ Download full report (Excel)",
                data=xlsx,
                file_name=f"bist_advanced_report_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        except Exception as e:
            st.error(f"Excel export failed: {e}")
            st.info("If this persists on Cloud, check logs for the specific sheet/value causing the issue.")

        st.caption(SIGNATURE_TEXT)

if __name__ == "__main__":
    main()
