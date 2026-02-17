
# =============================================================
# 📊 ADVANCED BIST Risk Budgeting System (Streamlit Cloud)
# Single-file app.py
# -------------------------------------------------------------
# Includes:
# - BIST50/BIST100 universe (auto-scrape option) + custom tickers
# - BIST100 benchmark integration
# - Constrained Risk Parity (max weight per stock, sector caps)
# - VaR / ES(CVaR): historical, parametric, modified (Cornish–Fisher)
# - Rolling risk contributions (%RC)
# - PyPortfolioOpt methods (if installed): Max Sharpe, Min Vol, HRP, Black-Litterman
# - QuantStats-style performance metrics (internal; QuantStats optional)
# - Professional Plotly dashboards
# - Robust multi-source data fetching (batch + per-ticker fallback)
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

# Optional packages
try:
    from pypfopt import EfficientFrontier, risk_models, expected_returns, BlackLittermanModel
    from pypfopt.hierarchical_risk_parity import HRPOpt
    PYPFOPT_AVAILABLE = True
except Exception:
    PYPFOPT_AVAILABLE = False

try:
    import quantstats as qs
    QUANTSTATS_AVAILABLE = True
except Exception:
    QUANTSTATS_AVAILABLE = False

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
.main-header { font-size: 2.2rem; font-weight: 800; margin-bottom: 0.25rem; }
.badge { display:inline-block; padding:0.22rem 0.7rem; border-radius:999px;
        background:#0f2a5f; color:#fff; font-size:0.85rem; margin:0.2rem 0 0.6rem 0;}
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

def apply_aliases(tickers: List[str]) -> Tuple[List[str], Dict[str, str]]:
    """
    Yahoo sometimes breaks/renames BIST tickers.
    We apply a minimal alias set to prevent known failures.
    """
    alias = {
        "KOZAL.IS": "TRALT.IS",  # common Yahoo mismatch
        "KOZAL": "TRALT.IS",
    }
    used = {}
    mapped = []
    for t in tickers:
        t2 = alias.get(t, t)
        if t2 != t:
            used[t] = t2
        mapped.append(t2)
    return normalize_tickers(mapped), used

def safe_float(x, default=np.nan) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)

def annualize_return(daily_mean: float) -> float:
    return daily_mean * 252.0

def annualize_vol(daily_std: float) -> float:
    return daily_std * np.sqrt(252.0)

def portfolio_returns(returns: pd.DataFrame, weights: np.ndarray) -> pd.Series:
    w = np.asarray(weights, dtype=float).reshape(-1)
    w = np.clip(w, 0, None)
    w = w / (w.sum() if w.sum() != 0 else 1.0)
    pr = returns.values @ w
    return pd.Series(pr, index=returns.index, name="Portfolio")

def to_excel_bytes(sheets: Dict[str, pd.DataFrame]) -> bytes:
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="xlsxwriter") as writer:
        for name, df in sheets.items():
            df.to_excel(writer, sheet_name=str(name)[:31], index=False)
    bio.seek(0)
    return bio.read()

# =============================================================
# Metadata
# =============================================================

@dataclass(frozen=True)
class AssetMeta:
    name: str
    sector: str

# =============================================================
# Data Fetching
# =============================================================

class BISTDataFetcher:
    """
    Robust multi-source fetch:
    1) yf.download batch
    2) per-ticker history fallback
    3) best-effort cleaning (does NOT require full intersection)
    """

    # "Core" list for stability (you can still use AUTO lists)
    DEFAULT_20 = [
        "AKBNK.IS","ARCLK.IS","ASELS.IS","BIMAS.IS","EKGYO.IS",
        "EREGL.IS","FROTO.IS","GARAN.IS","HALKB.IS","ISCTR.IS",
        "KCHOL.IS","TRALT.IS","KRDMD.IS","PETKM.IS","PGSUS.IS",
        "SAHOL.IS","SASA.IS","TCELL.IS","THYAO.IS","TOASO.IS",
    ]

    BENCHMARKS = {
        "BIST100 (^XU100)": "^XU100",
        "BIST50 (^XU050)": "^XU050",
        "BIST30 (^XU030)": "^XU030",
        "None": "",
    }

    # Basic mapping for a nicer UI (expand as you like)
    META: Dict[str, AssetMeta] = {
        "AKBNK.IS": AssetMeta("Akbank", "Banking"),
        "ARCLK.IS": AssetMeta("Arcelik", "Industrial"),
        "ASELS.IS": AssetMeta("Aselsan", "Defense"),
        "BIMAS.IS": AssetMeta("BIM", "Retail"),
        "EKGYO.IS": AssetMeta("Emlak Konut", "Real Estate"),
        "EREGL.IS": AssetMeta("Eregli Demir Celik", "Iron & Steel"),
        "FROTO.IS": AssetMeta("Ford Otosan", "Automotive"),
        "GARAN.IS": AssetMeta("Garanti BBVA", "Banking"),
        "HALKB.IS": AssetMeta("Halkbank", "Banking"),
        "ISCTR.IS": AssetMeta("Is Bankasi", "Banking"),
        "KCHOL.IS": AssetMeta("Koc Holding", "Holding"),
        "TRALT.IS": AssetMeta("Turk Altin Isletmeleri", "Mining"),
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
    def fetch_bist_list_online(mode: str = "BIST50", timeout: int = 20) -> List[str]:
        """
        Auto-scrape:
        - Uses OYAK pages commonly listing XU050/XU100.
        - If parsing fails, returns DEFAULT_20.
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
            return normalize_tickers(symbols, suffix=".IS")
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
    def _best_effort_clean_panel(prices: pd.DataFrame, min_obs: int, max_missing_frac: float) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Returns:
        - cleaned prices
        - cleaned returns
        - diagnostics
        """
        prices = prices.copy()
        prices = prices.replace([np.inf, -np.inf], np.nan).sort_index()
        prices = prices.dropna(axis=1, how="all")

        # limited ffill/bfill for holidays / short gaps
        prices = prices.ffill(limit=5).bfill(limit=2)

        obs = prices.notna().sum().sort_values(ascending=False)
        diag = pd.DataFrame({
            "ObsCount": obs,
            "Start": [prices[c].first_valid_index() for c in obs.index],
            "End": [prices[c].last_valid_index() for c in obs.index],
            "MissingFrac": [float(prices[c].isna().mean()) for c in obs.index],
        })

        keep = diag.index[(diag["ObsCount"] >= min_obs) & (diag["MissingFrac"] <= max_missing_frac)].tolist()
        prices = prices[keep]

        rets = prices.pct_change().replace([np.inf, -np.inf], np.nan)
        rets = rets.dropna(how="all")

        # drop columns that still have too many NaN in returns
        ret_obs = rets.notna().sum()
        keep2 = ret_obs.index[ret_obs >= (min_obs - 1)].tolist()
        rets = rets[keep2]
        prices = prices[keep2]

        return prices, rets, diag

    @staticmethod
    def _prune_for_covariance(returns: pd.DataFrame, min_assets: int = 2, max_iter: int = 25) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Remove columns that break covariance (NaNs).
        """
        rets = returns.copy()
        for _ in range(max_iter):
            cov = rets.cov() * 252.0
            bad = cov.columns[cov.isna().any()].tolist()
            if not bad:
                return rets, cov
            rets = rets.drop(columns=bad, errors="ignore")
            if rets.shape[1] < min_assets:
                break
        cov = rets.cov() * 252.0
        return rets, cov

    @staticmethod
    @st.cache_data(ttl=3600, show_spinner=False)
    def fetch_prices(
        tickers: Tuple[str, ...],
        start: str,
        end: str,
        min_obs: int,
        max_missing_frac: float,
        max_retries: int,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], pd.DataFrame]:
        tickers_l = list(tickers)
        dropped: List[str] = []

        # 1) batch attempt with retries
        data = pd.DataFrame()
        last_err = None
        for k in range(max_retries):
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

        # 2) per-ticker fallback if batch yields nothing or missing tickers
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
                time.sleep(0.15)
            if frames:
                prices = pd.concat(frames.values(), axis=1).sort_index()

        if prices is None or prices.empty:
            if last_err:
                raise ValueError(f"No data from Yahoo (batch+fallback failed). Last error: {last_err}")
            raise ValueError("No data from Yahoo (batch+fallback failed).")

        prices, rets, diag = BISTDataFetcher._best_effort_clean_panel(prices, min_obs=min_obs, max_missing_frac=max_missing_frac)
        if rets.shape[1] < 2:
            raise ValueError("Not enough valid tickers after cleaning (need at least 2).")

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
    """
    QuantStats-style key metrics (internal implementation).
    """
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
# PyPortfolioOpt wrappers (optional)
# =============================================================

def pypfopt_optimize(
    method: str,
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    risk_free_rate: float,
    max_weight: float,
    sector_caps: Optional[Dict[str, float]],
    sectors: Optional[List[str]],
    bl_views_json: str = "",
) -> Tuple[np.ndarray, Dict[str, float], str]:
    """
    Returns: weights, metrics, note
    """
    note = ""
    if not PYPFOPT_AVAILABLE:
        return np.ones(returns.shape[1]) / returns.shape[1], {}, "PyPortfolioOpt not installed; using Equal Weight."

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

    method = str(method).lower().strip()
    try:
        if method in ("max sharpe", "max_sharpe", "max-sharpe"):
            ef = EfficientFrontier(mu, S)
            ef.add_constraint(lambda w: w >= 0)
            ef.add_constraint(lambda w: w <= float(max_weight))
            if sector_caps and sectors and len(sectors) == n:
                for sec, cap in sector_caps.items():
                    idx = [i for i, s in enumerate(sectors) if s == sec]
                    if idx:
                        ef.add_constraint(lambda w, idx=idx, cap=float(cap): cap - sum(w[i] for i in idx))
            ef.max_sharpe(risk_free_rate=risk_free_rate)
            w = ef.clean_weights()
            perf = ef.portfolio_performance(risk_free_rate=risk_free_rate, verbose=False)
            weights = np.array([w[t] for t in tickers], dtype=float)
            metrics = {"expected_return": float(perf[0]), "volatility": float(perf[1]), "sharpe_ratio": float(perf[2])}
            return weights, metrics, note

        if method in ("min volatility", "min_volatility", "min-volatility"):
            ef = EfficientFrontier(mu, S)
            ef.add_constraint(lambda w: w >= 0)
            ef.add_constraint(lambda w: w <= float(max_weight))
            if sector_caps and sectors and len(sectors) == n:
                for sec, cap in sector_caps.items():
                    idx = [i for i, s in enumerate(sectors) if s == sec]
                    if idx:
                        ef.add_constraint(lambda w, idx=idx, cap=float(cap): cap - sum(w[i] for i in idx))
            ef.min_volatility()
            w = ef.clean_weights()
            perf = ef.portfolio_performance(risk_free_rate=risk_free_rate, verbose=False)
            weights = np.array([w[t] for t in tickers], dtype=float)
            metrics = {"expected_return": float(perf[0]), "volatility": float(perf[1]), "sharpe_ratio": float(perf[2])}
            return weights, metrics, note

        if method in ("hrp", "hierarchical risk parity"):
            # HRPOpt expects returns as rows (assets x time)
            hrp = HRPOpt(returns.T)
            hrp.optimize()
            w = hrp.clean_weights()
            perf = hrp.portfolio_performance(risk_free_rate=risk_free_rate, verbose=False)
            weights = np.array([w.get(t, 0.0) for t in tickers], dtype=float)
            weights = np.clip(weights, 0, None)
            weights = weights / weights.sum()
            metrics = {"expected_return": float(perf[0]), "volatility": float(perf[1]), "sharpe_ratio": float(perf[2])}
            return weights, metrics, note

        if method in ("black-litterman", "black litterman", "bl"):
            # views JSON example: {"AKBNK.IS": 0.30, "THYAO.IS": 0.25}
            views = {}
            if bl_views_json.strip():
                try:
                    views = json.loads(bl_views_json)
                    if not isinstance(views, dict):
                        views = {}
                except Exception:
                    views = {}
            if not views:
                # no views => behave like Max Sharpe on market implied returns
                note = "BL views missing/invalid; ran Max Sharpe on standard estimates instead."
                return pypfopt_optimize("max_sharpe", prices, returns, risk_free_rate, max_weight, sector_caps, sectors, "")

            market_caps = pd.Series(np.ones(n), index=tickers)
            bl = BlackLittermanModel(S, pi="market", market_caps=market_caps, absolute_views=views)
            post_mu = bl.bl_returns()
            post_S = bl.bl_cov()
            ef = EfficientFrontier(post_mu, post_S)
            ef.add_constraint(lambda w: w >= 0)
            ef.add_constraint(lambda w: w <= float(max_weight))
            ef.max_sharpe(risk_free_rate=risk_free_rate)
            w = ef.clean_weights()
            perf = ef.portfolio_performance(risk_free_rate=risk_free_rate, verbose=False)
            weights = np.array([w[t] for t in tickers], dtype=float)
            metrics = {"expected_return": float(perf[0]), "volatility": float(perf[1]), "sharpe_ratio": float(perf[2])}
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

def fig_risk_contrib_bar(risk_df: pd.DataFrame, name_map: Dict[str, str], top_n: int = 20) -> go.Figure:
    df = risk_df.copy()
    df["Label"] = df["Symbol"].map(lambda s: name_map.get(s, s))
    df = df.sort_values("Risk_Contribution_%", ascending=True).tail(top_n)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=df["Label"], x=df["Risk_Contribution_%"], orientation="h",
        marker=dict(color=df["Risk_Contribution_%"], colorscale="RdYlGn_r", showscale=True,
                    colorbar=dict(title="Risk %")),
        text=df["Risk_Contribution_%"].round(1).astype(str) + "%",
        textposition="outside",
        name="Risk Contribution",
    ))
    eq = 100.0 / max(1, len(risk_df))
    fig.add_vline(x=eq, line_dash="dash", line_color="red", opacity=0.7,
                  annotation_text=f"Equal RC (~{eq:.1f}%)")
    fig.update_layout(height=680, title="Risk Contribution by Asset", margin=dict(l=10, r=10, t=60, b=10))
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
# App
# =============================================================

def main():
    st.markdown('<div class="main-header">📊 Advanced BIST Risk Budgeting System</div>', unsafe_allow_html=True)
    st.markdown('<div class="badge">📡 Data Source: Yahoo Finance (yfinance)</div>', unsafe_allow_html=True)

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
            custom_txt = st.text_area("Tickers (comma/newline). You may omit .IS.", value="AKBNK,ARCLK,ASELS,BIMAS,EKGYO,EREGL,FROTO,GARAN,HALKB,ISCTR,KCHOL,KOZAL,KRDMD,PETKM,PGSUS,SAHOL,SASA,TCELL,THYAO,TOASO")

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

        st.markdown("### Data quality")
        min_obs = st.slider("Min observations per ticker", 30, 400, 120, step=10)
        max_missing_frac = st.slider("Max missing fraction", 0.05, 0.80, 0.35, step=0.05)
        max_retries = st.slider("Yahoo retries", 1, 6, 3, step=1)

        st.divider()

        st.markdown("### Portfolio method")
        methods = ["Equal Weight", "Risk Parity (Constrained)"]
        if PYPFOPT_AVAILABLE:
            methods += ["PyPortfolioOpt: Max Sharpe", "PyPortfolioOpt: Min Volatility", "PyPortfolioOpt: HRP", "PyPortfolioOpt: Black-Litterman"]
        else:
            st.caption("PyPortfolioOpt not installed → only Equal Weight + Risk Parity available.")
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

        bl_views = ""
        if "Black-Litterman" in method:
            st.markdown("### BL Views (optional)")
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

    universe, alias_used = apply_aliases(universe)

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
                if lc in ("ticker", "symbol", "sembol", "kod"):
                    tcol = c
                if lc in ("sector", "sektor", "industry"):
                    scol = c
            if tcol is None or scol is None:
                raise ValueError("CSV must include ticker and sector columns.")
            sector_map = {}
            for _, row in df_sec[[tcol, scol]].dropna().iterrows():
                tk = normalize_tickers([row[tcol]])[0]
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
    if alias_used:
        st.caption(f"Ticker aliases applied: {alias_used}")

    fetch_tickers = universe.copy()
    if benchmark_ticker and benchmark_ticker not in fetch_tickers:
        fetch_tickers += [benchmark_ticker]

    with st.spinner("Downloading from Yahoo Finance..."):
        try:
            prices_all, returns_all, dropped, diag = fetcher.fetch_prices(
                tuple(fetch_tickers),
                start=str(start_date),
                end=str(end_date),
                min_obs=int(min_obs),
                max_missing_frac=float(max_missing_frac),
                max_retries=int(max_retries),
            )
        except Exception as e:
            st.error(f"❌ Data error: {e}")
            st.info("Try: later start date (e.g. 2022-01-01), lower Min observations, or use Default 20.")
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
        prices = prices_all.drop(columns=[benchmark_ticker], errors="ignore")
        returns = returns_all.drop(columns=[benchmark_ticker], errors="ignore")
    else:
        prices = prices_all
        returns = returns_all

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

        elif method.startswith("PyPortfolioOpt"):
            pmethod = method.split(":", 1)[1].strip()
            weights, opt_metrics, opt_note = pypfopt_optimize(
                method=pmethod,
                prices=prices[loaded],
                returns=returns[loaded],
                risk_free_rate=float(risk_free_rate),
                max_weight=float(max_weight),
                sector_caps=sector_caps,
                sectors=sectors,
                bl_views_json=bl_views,
            )

        weights = np.asarray(weights, dtype=float).reshape(-1)
        weights = np.clip(weights, 0, None)
        weights = weights / (weights.sum() if weights.sum() != 0 else 1.0)

        # Enforce bounds post-normalization (softly)
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

    # align benchmark to portfolio
    bench_aligned = None
    if benchmark_returns is not None and not benchmark_returns.empty:
        idx = port.index.intersection(benchmark_returns.index)
        if len(idx) >= 30:
            bench_aligned = benchmark_returns.loc[idx]
            port_aligned = port.loc[idx]
        else:
            bench_aligned = None
            port_aligned = port
    else:
        port_aligned = port

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

    if benchmark_ticker and ("Beta vs Benchmark" in perf.index):
        st.caption(f"Benchmark: {bench_choice} • Beta={safe_float(perf.get('Beta vs Benchmark')):.2f} • IR={safe_float(perf.get('Information Ratio')):.2f}")

    # =========================================================
    # Charts
    # =========================================================
    tabs = st.tabs([
        "Performance", "Risk Contributions", "VaR / ES", "Correlation", "Rolling RC", "Weights & Tables", "Export"
    ])

    with tabs[0]:
        st.plotly_chart(fig_cum_and_drawdown(port_aligned, bench_aligned), use_container_width=True)
        st.markdown("**Performance Report (QuantStats-style)**")
        pr_df = perf.to_frame(name="Value").reset_index().rename(columns={"index": "Metric"})
        pr_df["Value"] = pr_df["Value"].apply(lambda x: f"{x:.4f}" if isinstance(x, (int, float, np.floating)) and np.isfinite(x) else str(x))
        st.dataframe(pr_df, use_container_width=True, height=420)

        if show_quantstats and QUANTSTATS_AVAILABLE:
            st.markdown("**QuantStats HTML Report (may take ~10–30s)**")
            try:
                # quantstats expects returns in decimal daily
                html = qs.reports.html(port_aligned, benchmark=bench_aligned, output=None, rf=float(risk_free_rate), title="BIST Portfolio Report")
                st.components.v1.html(html, height=900, scrolling=True)
            except Exception as e:
                st.warning(f"QuantStats report failed: {e}")

    with tabs[1]:
        st.plotly_chart(fig_risk_contrib_bar(risk_df, name_map, top_n=min(30, len(risk_df))), use_container_width=True)
        st.dataframe(risk_df.merge(
            pd.DataFrame({"Symbol": loaded, "Company": [name_map.get(t, t) for t in loaded], "Sector": sectors}),
            on="Symbol", how="left"
        )[["Risk_Rank","Symbol","Company","Sector","Weight","Risk_Contribution_%","Individual_Volatility","Marginal_Risk_Contribution"]]
          .sort_values("Risk_Rank"), use_container_width=True, height=560)

    with tabs[2]:
        st.plotly_chart(fig_var_table(var_tbl), use_container_width=True)
        st.caption("VaR/ES are shown as **loss magnitudes**. Example: VaR=0.03 means a 3% loss threshold.")

    with tabs[3]:
        st.plotly_chart(fig_corr_heatmap(returns, max_assets=35), use_container_width=True)

    with tabs[4]:
        if compute_rolling and (rc_roll is not None) and (not rc_roll.empty):
            st.plotly_chart(fig_rolling_rc(rc_roll, name_map, top_n=int(roll_topn)), use_container_width=True)
            st.caption("Rolling RC (%) is computed with covariance over the rolling window and fixed weights.")
        else:
            st.info("Rolling RC is empty (try smaller window/step, or ensure enough data).")

    with tabs[5]:
        w_df = pd.DataFrame({
            "Symbol": loaded,
            "Company": [name_map.get(t, t) for t in loaded],
            "Sector": sectors,
            "Weight(%)": np.round(weights * 100.0, 4),
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
            risk_export = risk_df.copy()
            risk_export["Company"] = risk_export["Symbol"].map(lambda s: name_map.get(s, s))
            risk_export["Sector"] = sectors
            st.download_button(
                "⬇️ Risk metrics (CSV)",
                data=risk_export.to_csv(index=False).encode("utf-8"),
                file_name=f"bist_risk_metrics_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
            )

    with tabs[6]:
        st.markdown("**Excel report (one file)**")
        sheets = {
            "Weights": pd.DataFrame({
                "Symbol": loaded,
                "Company": [name_map.get(t, t) for t in loaded],
                "Sector": sectors,
                "Weight": weights,
            }),
            "RiskMetrics": risk_df.copy(),
            "PerfReport": perf.to_frame(name="Value").reset_index().rename(columns={"index": "Metric"}),
            "VaR_ES": var_tbl.copy(),
        }
        if bench_aligned is not None:
            sheets["Benchmark"] = pd.DataFrame({"Date": bench_aligned.index, "BenchmarkReturn": bench_aligned.values})
        if compute_rolling and rc_roll is not None and not rc_roll.empty:
            rc_exp = rc_roll.reset_index().rename(columns={"index": "Date"})
            sheets["RollingRC"] = rc_exp

        xlsx = to_excel_bytes({k: v for k, v in sheets.items()})
        st.download_button(
            "⬇️ Download full report (Excel)",
            data=xlsx,
            file_name=f"bist_advanced_report_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
        st.caption("If you need a PDF report too, tell me your preferred layout (A4 portrait/landscape).")

if __name__ == "__main__":
    main()
