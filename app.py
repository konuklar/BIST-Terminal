# -*- coding: utf-8 -*-
"""
📊 ADVANCED BIST Risk Budgeting System (Streamlit Cloud) - ENHANCED VERSION
Enhanced with:
- Robust multi-strategy data fetching with fallbacks
- Advanced risk metrics (tail risk, drawdown analysis, factor models)
- Professional visualizations with interactive dashboards
- Monte Carlo simulation for stress testing
- Enhanced PyPortfolioOpt integration
- Performance attribution and benchmarking
- Export capabilities with comprehensive reports

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
from typing import Dict, List, Tuple, Any, Optional
import concurrent.futures
import time

import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.optimize import minimize, differential_evolution
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import squareform

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
except Exception as e:
    PYPFOPT_AVAILABLE = False
    print(f"PyPortfolioOpt import error: {e}")

# ------------------------------------------------------------
# Page configuration
# ------------------------------------------------------------
st.set_page_config(
    page_title="BIST Advanced Risk Budgeting System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------
# Custom CSS
# ------------------------------------------------------------
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        font-weight: 800;
        margin-bottom: 0.5rem;
        background: linear-gradient(90deg, #1E3A8A 0%, #2563EB 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2563EB;
        font-weight: 700;
        margin-top: 1.2rem;
        margin-bottom: 0.5rem;
        border-bottom: 2px solid #E5E7EB;
        padding-bottom: 0.3rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 0.8rem;
        padding: 1.2rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        border-left: 4px solid #2563EB;
    }
    .warning-card {
        background-color: #FEF3C7;
        border-radius: 0.8rem;
        padding: 1rem;
        border-left: 4px solid #F59E0B;
    }
    .data-source-badge {
        background-color: #1E3A8A;
        color: white;
        padding: 0.4rem 1rem;
        border-radius: 2rem;
        font-size: 0.9rem;
        display: inline-block;
        margin: 0.5rem 0 1rem 0;
    }
    .note-box {
        background-color: #F3F4F6;
        border: 1px solid #E5E7EB;
        padding: 0.75rem 1rem;
        border-radius: 0.6rem;
        font-size: 0.9rem;
    }
    .small-muted {
        color: #6B7280;
        font-size: 0.85rem;
        font-style: italic;
    }
    .footer {
        text-align: center;
        padding: 2rem 0 0 0;
        color: #6B7280;
        font-size: 0.9rem;
        border-top: 1px solid #E5E7EB;
        margin-top: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------
# Universe (BIST 50 constituents - enhanced with metadata)
# ------------------------------------------------------------
BIST50_TICKERS = [
    "AEFES.IS", "AKBNK.IS", "ALARK.IS", "ARCLK.IS", "ASELS.IS",
    "ASTOR.IS", "BIMAS.IS", "BRSAN.IS", "BTCIM.IS", "CCOLA.IS",
    "CIMSA.IS", "DOAS.IS", "DOHOL.IS", "EKGYO.IS", "ENKAI.IS",
    "EREGL.IS", "FROTO.IS", "GARAN.IS", "GUBRF.IS", "HALKB.IS",
    "HEKTS.IS", "ISCTR.IS", "KCHOL.IS", "KONTR.IS", "KRDMD.IS",
    "MAVI.IS", "MGROS.IS", "MIATK.IS", "OYAKC.IS", "PGSUS.IS",
    "SAHOL.IS", "SASA.IS", "SISE.IS", "SOKM.IS", "TAVHL.IS",
    "TCELL.IS", "THYAO.IS", "TOASO.IS", "TSKB.IS", "TTKOM.IS",
    "TUPRS.IS", "ULKER.IS", "VAKBN.IS", "VESTL.IS", "YKBNK.IS",
]

# Enhanced metadata with company names and sectors
COMPANY_METADATA = {
    "AEFES.IS": {"name": "Anadolu Efes", "sector": "Consumer", "market_cap": "Large"},
    "AKBNK.IS": {"name": "Akbank", "sector": "Banking", "market_cap": "Large"},
    "ALARK.IS": {"name": "Alarko Holding", "sector": "Holding", "market_cap": "Large"},
    "ARCLK.IS": {"name": "Arçelik", "sector": "Industrial", "market_cap": "Large"},
    "ASELS.IS": {"name": "Aselsan", "sector": "Defense", "market_cap": "Large"},
    "ASTOR.IS": {"name": "Astor Enerji", "sector": "Technology", "market_cap": "Mid"},
    "BIMAS.IS": {"name": "BİM Mağazalar", "sector": "Retail", "market_cap": "Large"},
    "BRSAN.IS": {"name": "Borusan Mannesmann", "sector": "Industrial", "market_cap": "Mid"},
    "BTCIM.IS": {"name": "Batıçim", "sector": "Industrial", "market_cap": "Mid"},
    "CCOLA.IS": {"name": "Coca-Cola İçecek", "sector": "Consumer", "market_cap": "Large"},
    "CIMSA.IS": {"name": "Çimsa", "sector": "Industrial", "market_cap": "Mid"},
    "DOAS.IS": {"name": "Doğuş Otomotiv", "sector": "Automotive", "market_cap": "Large"},
    "DOHOL.IS": {"name": "Doğan Holding", "sector": "Holding", "market_cap": "Large"},
    "EKGYO.IS": {"name": "Emlak Konut", "sector": "Real Estate", "market_cap": "Large"},
    "ENKAI.IS": {"name": "Enka İnşaat", "sector": "Industrial", "market_cap": "Large"},
    "EREGL.IS": {"name": "Ereğli Demir Çelik", "sector": "Iron & Steel", "market_cap": "Large"},
    "FROTO.IS": {"name": "Ford Otosan", "sector": "Automotive", "market_cap": "Large"},
    "GARAN.IS": {"name": "Garanti BBVA", "sector": "Banking", "market_cap": "Large"},
    "GUBRF.IS": {"name": "Gübre Fabrikaları", "sector": "Chemicals", "market_cap": "Mid"},
    "HALKB.IS": {"name": "Halkbank", "sector": "Banking", "market_cap": "Large"},
    "HEKTS.IS": {"name": "Hektaş", "sector": "Chemicals", "market_cap": "Mid"},
    "ISCTR.IS": {"name": "İş Bankası", "sector": "Banking", "market_cap": "Large"},
    "KCHOL.IS": {"name": "Koç Holding", "sector": "Holding", "market_cap": "Large"},
    "KONTR.IS": {"name": "Kontrolmatik", "sector": "Technology", "market_cap": "Mid"},
    "KRDMD.IS": {"name": "Kardemir", "sector": "Iron & Steel", "market_cap": "Mid"},
    "MAVI.IS": {"name": "Mavi Giyim", "sector": "Retail", "market_cap": "Mid"},
    "MGROS.IS": {"name": "Migros", "sector": "Retail", "market_cap": "Large"},
    "MIATK.IS": {"name": "Mia Teknoloji", "sector": "Technology", "market_cap": "Mid"},
    "OYAKC.IS": {"name": "Oyak Çimento", "sector": "Industrial", "market_cap": "Large"},
    "PGSUS.IS": {"name": "Pegasus", "sector": "Aviation", "market_cap": "Large"},
    "SAHOL.IS": {"name": "Sabancı Holding", "sector": "Holding", "market_cap": "Large"},
    "SASA.IS": {"name": "SASA Polyester", "sector": "Chemicals", "market_cap": "Large"},
    "SISE.IS": {"name": "Şişecam", "sector": "Industrial", "market_cap": "Large"},
    "SOKM.IS": {"name": "Şok Marketler", "sector": "Retail", "market_cap": "Large"},
    "TAVHL.IS": {"name": "TAV Havalimanları", "sector": "Aviation", "market_cap": "Large"},
    "TCELL.IS": {"name": "Turkcell", "sector": "Telecom", "market_cap": "Large"},
    "THYAO.IS": {"name": "Türk Hava Yolları", "sector": "Aviation", "market_cap": "Large"},
    "TOASO.IS": {"name": "Tofaş", "sector": "Automotive", "market_cap": "Large"},
    "TSKB.IS": {"name": "TSKB", "sector": "Banking", "market_cap": "Mid"},
    "TTKOM.IS": {"name": "Türk Telekom", "sector": "Telecom", "market_cap": "Large"},
    "TUPRS.IS": {"name": "Tüpraş", "sector": "Energy", "market_cap": "Large"},
    "ULKER.IS": {"name": "Ülker", "sector": "Consumer", "market_cap": "Large"},
    "VAKBN.IS": {"name": "Vakıfbank", "sector": "Banking", "market_cap": "Large"},
    "VESTL.IS": {"name": "Vestel", "sector": "Technology", "market_cap": "Large"},
    "YKBNK.IS": {"name": "Yapı Kredi", "sector": "Banking", "market_cap": "Large"},
}

# Benchmark and factor tickers
BENCHMARK_CANDIDATES = ["^XU100", "XU100.IS"]
FX_FACTOR = "TRY=X"
RATE_FACTOR = "^TNX"
GOLD_FACTOR = "GC=F"
OIL_FACTOR = "CL=F"

# ------------------------------------------------------------
# Enhanced Data Fetcher with Multi-Strategy and Fallbacks
# ------------------------------------------------------------
class EnhancedDataFetcher:
    """Robust data fetcher with multiple strategies and fallbacks"""
    
    @staticmethod
    @st.cache_data(ttl=3600, show_spinner=False)
    def fetch_with_retry(
        tickers: List[str],
        start: date,
        end: date,
        max_retries: int = 3,
        parallel: bool = True
    ) -> Tuple[pd.DataFrame, List[str], Dict[str, Any]]:
        """Fetch data with multiple strategies and retry logic"""
        
        start_str = pd.to_datetime(start).strftime("%Y-%m-%d")
        end_str = pd.to_datetime(end).strftime("%Y-%m-%d")
        
        metadata = {
            "strategy": "unknown",
            "attempts": 0,
            "success": False,
            "errors": []
        }
        
        # Strategy 1: Batch download
        for attempt in range(max_retries):
            metadata["attempts"] += 1
            try:
                data = yf.download(
                    tickers=tickers,
                    start=start_str,
                    end=end_str,
                    interval="1d",
                    auto_adjust=True,
                    group_by="column",
                    progress=False,
                    threads=True,
                    timeout=30
                )
                
                if data is not None and not data.empty:
                    closes = EnhancedDataFetcher._extract_closes(data, tickers)
                    if not closes.empty:
                        metadata["strategy"] = "batch"
                        metadata["success"] = True
                        closes = EnhancedDataFetcher._clean_index(closes)
                        dropped = [t for t in tickers if t not in closes.columns]
                        return closes, dropped, metadata
                        
            except Exception as e:
                metadata["errors"].append(f"Batch attempt {attempt + 1}: {str(e)}")
                time.sleep(2 ** attempt)  # Exponential backoff
        
        # Strategy 2: Parallel download if batch fails
        if parallel and len(tickers) > 1:
            try:
                closes = EnhancedDataFetcher._fetch_parallel(tickers, start_str, end_str)
                if not closes.empty:
                    metadata["strategy"] = "parallel"
                    metadata["success"] = True
                    closes = EnhancedDataFetcher._clean_index(closes)
                    dropped = [t for t in tickers if t not in closes.columns]
                    return closes, dropped, metadata
            except Exception as e:
                metadata["errors"].append(f"Parallel fetch: {str(e)}")
        
        # Strategy 3: Sequential download as last resort
        closes = EnhancedDataFetcher._fetch_sequential(tickers, start_str, end_str)
        if not closes.empty:
            metadata["strategy"] = "sequential"
            metadata["success"] = True
            closes = EnhancedDataFetcher._clean_index(closes)
            dropped = [t for t in tickers if t not in closes.columns]
            return closes, dropped, metadata
        
        return pd.DataFrame(), tickers, metadata
    
    @staticmethod
    def _extract_closes(data: pd.DataFrame, tickers: List[str]) -> pd.DataFrame:
        """Extract close prices from various Yahoo Finance return formats"""
        if data is None or data.empty:
            return pd.DataFrame()
        
        if isinstance(data.columns, pd.MultiIndex):
            # Multi-index case (multiple tickers)
            for price_type in ["Close", "Adj Close"]:
                if price_type in data.columns.get_level_values(0):
                    closes = data[price_type]
                    if isinstance(closes, pd.Series):
                        closes = closes.to_frame()
                    return closes
        else:
            # Single ticker case
            if "Close" in data.columns:
                return pd.DataFrame({tickers[0]: data["Close"]})
            elif "Adj Close" in data.columns:
                return pd.DataFrame({tickers[0]: data["Adj Close"]})
        
        return pd.DataFrame()
    
    @staticmethod
    def _fetch_parallel(tickers: List[str], start: str, end: str, max_workers: int = 5) -> pd.DataFrame:
        """Parallel fetch for better performance"""
        
        def fetch_single(ticker):
            try:
                ticker_data = yf.download(
                    tickers=ticker,
                    start=start,
                    end=end,
                    interval="1d",
                    auto_adjust=True,
                    progress=False,
                    threads=False
                )
                if ticker_data is not None and not ticker_data.empty:
                    if "Close" in ticker_data.columns:
                        return ticker, ticker_data["Close"]
                    elif "Adj Close" in ticker_data.columns:
                        return ticker, ticker_data["Adj Close"]
            except:
                pass
            return ticker, None
        
        results = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_ticker = {executor.submit(fetch_single, t): t for t in tickers}
            for future in concurrent.futures.as_completed(future_to_ticker):
                ticker, series = future.result()
                if series is not None:
                    results[ticker] = series
        
        if results:
            return pd.DataFrame(results)
        return pd.DataFrame()
    
    @staticmethod
    def _fetch_sequential(tickers: List[str], start: str, end: str) -> pd.DataFrame:
        """Sequential fetch as last resort"""
        results = {}
        for ticker in tickers:
            try:
                ticker_data = yf.download(
                    tickers=ticker,
                    start=start,
                    end=end,
                    interval="1d",
                    auto_adjust=True,
                    progress=False
                )
                if ticker_data is not None and not ticker_data.empty:
                    if "Close" in ticker_data.columns:
                        results[ticker] = ticker_data["Close"]
            except:
                continue
            time.sleep(0.2)  # Rate limiting
        
        return pd.DataFrame(results) if results else pd.DataFrame()
    
    @staticmethod
    def _clean_index(df: pd.DataFrame) -> pd.DataFrame:
        """Clean datetime index"""
        if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
            df = df.copy()
            df.index = df.index.tz_localize(None)
        return df


# ------------------------------------------------------------
# Advanced Risk Analytics Class
# ------------------------------------------------------------
@dataclass
class AdvancedRiskMetrics:
    """Comprehensive risk metrics calculation"""
    
    @staticmethod
    def calculate_drawdowns(returns: pd.Series) -> pd.DataFrame:
        """Calculate drawdown metrics including underwater period"""
        wealth = (1 + returns).cumprod()
        peak = wealth.cummax()
        drawdown = (wealth - peak) / peak
        
        # Find drawdown periods
        is_drawdown = drawdown < 0
        drawdown_start = None
        drawdown_periods = []
        
        for i, val in enumerate(is_drawdown):
            if val and drawdown_start is None:
                drawdown_start = i
            elif not val and drawdown_start is not None:
                drawdown_periods.append((drawdown_start, i))
                drawdown_start = None
        
        # Calculate max drawdown duration
        if drawdown_periods:
            max_duration = max(end - start for start, end in drawdown_periods)
        else:
            max_duration = 0
        
        return pd.DataFrame({
            'Drawdown': drawdown,
            'Wealth_Index': wealth,
            'Peak': peak,
            'Max_Drawdown': drawdown.min(),
            'Max_Duration_Days': max_duration
        })
    
    @staticmethod
    def tail_risk_metrics(returns: pd.Series) -> Dict[str, float]:
        """Calculate comprehensive tail risk measures"""
        sorted_returns = returns.sort_values()
        
        metrics = {}
        
        # VaR at different confidence levels
        for conf in [0.95, 0.99]:
            alpha = 1 - conf
            var = sorted_returns.quantile(alpha)
            cvar = sorted_returns[sorted_returns <= var].mean()
            
            metrics[f'VaR_{int(conf*100)}'] = var
            metrics[f'CVaR_{int(conf*100)}'] = cvar
        
        # Expected Shortfall (ES) at different levels
        for conf in [0.95, 0.99]:
            alpha = 1 - conf
            var = sorted_returns.quantile(alpha)
            es = sorted_returns[sorted_returns <= var].mean()
            metrics[f'ES_{int(conf*100)}'] = es
        
        # Tail Ratio (right tail / left tail)
        right_tail = sorted_returns.quantile(0.95)
        left_tail = abs(sorted_returns.quantile(0.05))
        metrics['Tail_Ratio'] = right_tail / left_tail if left_tail != 0 else np.nan
        
        # Pain Index and Ulcer Index
        wealth = (1 + returns).cumprod()
        drawdown = wealth / wealth.cummax() - 1
        metrics['Pain_Index'] = abs(drawdown).mean() * 100
        metrics['Ulcer_Index'] = np.sqrt((drawdown ** 2).mean()) * 100
        
        # Distribution metrics
        metrics['Skewness'] = returns.skew()
        metrics['Kurtosis'] = returns.kurtosis()
        metrics['Jarque_Bera'] = stats.jarque_bera(returns.dropna())[0]
        
        return metrics
    
    @staticmethod
    def rolling_risk_metrics(returns: pd.Series, windows: List[int] = [21, 63, 252]) -> pd.DataFrame:
        """Calculate rolling risk metrics for multiple windows"""
        results = {}
        
        for window in windows:
            rolling_vol = returns.rolling(window).std() * np.sqrt(252)
            rolling_sharpe = returns.rolling(window).mean() / returns.rolling(window).std() * np.sqrt(252)
            rolling_var = returns.rolling(window).quantile(0.05)
            rolling_skew = returns.rolling(window).skew()
            
            results[f'Vol_{window}d'] = rolling_vol
            results[f'Sharpe_{window}d'] = rolling_sharpe
            results[f'VaR95_{window}d'] = rolling_var
            results[f'Skew_{window}d'] = rolling_skew
        
        return pd.DataFrame(results)
    
    @staticmethod
    def stability_metrics(returns: pd.Series) -> Dict[str, float]:
        """Calculate strategy stability metrics"""
        # Rolling Sharpe stability
        rolling_sharpe = returns.rolling(252).mean() / returns.rolling(252).std() * np.sqrt(252)
        sharpe_std = rolling_sharpe.std()
        sharpe_mean = rolling_sharpe.mean()
        
        # Hit rate (percentage of positive days)
        hit_rate = (returns > 0).mean()
        
        # Gain to Pain ratio
        avg_gain = returns[returns > 0].mean() if any(returns > 0) else 0
        avg_loss = abs(returns[returns < 0].mean()) if any(returns < 0) else 1
        gain_pain_ratio = avg_gain / avg_loss if avg_loss > 0 else np.nan
        
        # Calmar ratio (CAGR / Max Drawdown)
        cagr = (1 + returns).prod() ** (252 / len(returns)) - 1
        max_dd = AdvancedRiskMetrics.calculate_drawdowns(returns)['Max_Drawdown'].iloc[0]
        calmar_ratio = cagr / abs(max_dd) if max_dd != 0 else np.nan
        
        return {
            'Sharpe_Stability': sharpe_std / sharpe_mean if sharpe_mean != 0 else np.nan,
            'Hit_Rate': hit_rate,
            'Gain_Pain_Ratio': gain_pain_ratio,
            'Calmar_Ratio': calmar_ratio,
            'Avg_Gain': avg_gain,
            'Avg_Loss': avg_loss
        }


# ------------------------------------------------------------
# Monte Carlo Simulation for Stress Testing
# ------------------------------------------------------------
class MonteCarloSimulator:
    """Monte Carlo simulation for portfolio stress testing"""
    
    @staticmethod
    def simulate_returns(
        returns: pd.DataFrame,
        weights: np.ndarray,
        n_simulations: int = 10000,
        horizon_days: int = 252
    ) -> Dict[str, Any]:
        """Run Monte Carlo simulation for portfolio returns"""
        
        # Calculate parameters
        mu = returns.mean() * 252
        sigma = returns.cov() * 252
        
        # Generate correlated random returns
        n_assets = len(returns.columns)
        simulated_returns = np.random.multivariate_normal(
            mean=mu,
            cov=sigma,
            size=(n_simulations, horizon_days)
        )
        
        # Calculate portfolio returns for each simulation
        portfolio_returns = simulated_returns @ weights
        
        # Calculate cumulative returns
        cumulative_returns = np.cumprod(1 + portfolio_returns, axis=1)
        
        # Calculate metrics
        final_values = cumulative_returns[:, -1]
        
        results = {
            'final_values': final_values,
            'mean_final': np.mean(final_values),
            'median_final': np.median(final_values),
            'std_final': np.std(final_values),
            'var_95': np.percentile(final_values, 5),
            'var_99': np.percentile(final_values, 1),
            'cvar_95': np.mean(final_values[final_values <= np.percentile(final_values, 5)]),
            'probability_loss': np.mean(final_values < 1) * 100,
            'probability_double': np.mean(final_values > 2) * 100,
            'all_simulations': cumulative_returns
        }
        
        return results
    
    @staticmethod
    def plot_simulations(sim_results: Dict[str, Any], n_show: int = 100) -> go.Figure:
        """Plot Monte Carlo simulation results"""
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Simulated Paths', 'Distribution of Final Values',
                           'Value at Risk (VaR)', 'Probability Distribution'),
            specs=[[{'secondary_y': False}, {'secondary_y': False}],
                   [{'secondary_y': False}, {'secondary_y': False}]]
        )
        
        # Simulated paths
        for i in range(min(n_show, len(sim_results['all_simulations']))):
            fig.add_trace(
                go.Scatter(
                    y=sim_results['all_simulations'][i],
                    mode='lines',
                    line=dict(width=0.5, color='rgba(100,100,255,0.2)'),
                    showlegend=False
                ),
                row=1, col=1
            )
        
        # Distribution of final values
        fig.add_trace(
            go.Histogram(x=sim_results['final_values'], nbinsx=50, name='Distribution'),
            row=1, col=2
        )
        
        # VaR visualization
        fig.add_trace(
            go.Box(y=sim_results['final_values'], name='Final Values'),
            row=2, col=1
        )
        
        # Probability distribution
        fig.add_trace(
            go.Violin(y=sim_results['final_values'], name='Distribution', box_visible=True),
            row=2, col=2
        )
        
        fig.update_layout(
            height=800,
            title_text="Monte Carlo Simulation Results",
            showlegend=False
        )
        
        return fig


# ------------------------------------------------------------
# Enhanced Risk Engine with Advanced Metrics
# ------------------------------------------------------------
@dataclass
class EnhancedRiskResults:
    risk_table: pd.DataFrame
    portfolio_metrics: Dict[str, Any]
    cov_annual: pd.DataFrame
    port_returns: pd.Series
    advanced_metrics: Dict[str, Any]
    drawdown_analysis: pd.DataFrame
    tail_metrics: Dict[str, float]

class EnhancedRiskEngine:
    """Enhanced risk engine with comprehensive metrics"""
    
    @staticmethod
    def portfolio_vol(cov_annual: np.ndarray, w: np.ndarray) -> float:
        return float(np.sqrt(np.maximum(w @ cov_annual @ w, 0.0)))
    
    @staticmethod
    def risk_contributions(cov_annual: pd.DataFrame, w: np.ndarray, tickers: List[str]) -> pd.DataFrame:
        cov = cov_annual.values
        vol = EnhancedRiskEngine.portfolio_vol(cov, w)
        if vol <= 0:
            vol = 1e-12
        
        mrc = (cov @ w) / vol
        crc = w * mrc
        pct = (crc / vol) * 100.0
        indiv_vol = np.sqrt(np.diag(cov))
        
        df = pd.DataFrame({
            "Symbol": tickers,
            "Company": [COMPANY_METADATA.get(t, {}).get("name", t) for t in tickers],
            "Sector": [COMPANY_METADATA.get(t, {}).get("sector", "Other") for t in tickers],
            "Market_Cap": [COMPANY_METADATA.get(t, {}).get("market_cap", "Mid") for t in tickers],
            "Weight": w,
            "Individual_Volatility": indiv_vol,
            "Marginal_Risk_Contribution": mrc,
            "Component_Risk": crc,
            "Risk_Contribution_%": pct,
            "Risk_Weight_Ratio": pct / (w * 100) if any(w > 0) else 0
        }).sort_values("Risk_Contribution_%", ascending=False).reset_index(drop=True)
        
        df["Risk_Rank"] = np.arange(1, len(df) + 1)
        return df
    
    @staticmethod
    def compute(returns: pd.DataFrame, weights: np.ndarray) -> EnhancedRiskResults:
        tickers = returns.columns.tolist()
        cov_annual = returns.cov() * 252.0
        cov_annual = cov_annual.fillna(0.0)
        port_returns = (returns * weights).sum(axis=1)
        
        port_var = float(weights @ cov_annual.values @ weights)
        port_vol = float(np.sqrt(np.maximum(port_var, 0.0)))
        
        indiv_vol = np.sqrt(np.diag(cov_annual.values))
        weighted_avg_vol = float(np.sum(weights * indiv_vol)) if len(indiv_vol) else float("nan")
        diversification_ratio = float(weighted_avg_vol / port_vol) if port_vol > 0 else float("nan")
        
        risk_df = EnhancedRiskEngine.risk_contributions(cov_annual, weights, tickers)
        
        # Advanced metrics
        advanced = AdvancedRiskMetrics()
        drawdown_analysis = advanced.calculate_drawdowns(port_returns)
        tail_metrics = advanced.tail_risk_metrics(port_returns)
        stability_metrics = advanced.stability_metrics(port_returns)
        rolling_metrics = advanced.rolling_risk_metrics(port_returns)
        
        # Calculate returns metrics
        total_return = (1 + port_returns).prod() - 1
        cagr = (1 + total_return) ** (252 / len(port_returns)) - 1
        
        portfolio_metrics = {
            "volatility": port_vol,
            "variance": port_var,
            "total_return": total_return,
            "cagr": cagr,
            "n_assets": len(tickers),
            "diversification_ratio": diversification_ratio,
            "max_risk_contrib": float(risk_df.iloc[0]["Risk_Contribution_%"]) if len(risk_df) else float("nan"),
            "max_risk_asset": str(risk_df.iloc[0]["Symbol"]) if len(risk_df) else "",
            "max_drawdown": float(drawdown_analysis['Max_Drawdown'].iloc[0]),
            "max_drawdown_duration": int(drawdown_analysis['Max_Duration_Days'].iloc[0]),
        }
        
        # Add tail metrics to portfolio metrics
        portfolio_metrics.update(tail_metrics)
        portfolio_metrics.update(stability_metrics)
        
        return EnhancedRiskResults(
            risk_table=risk_df,
            portfolio_metrics=portfolio_metrics,
            cov_annual=cov_annual,
            port_returns=port_returns,
            advanced_metrics=rolling_metrics.to_dict() if not rolling_metrics.empty else {},
            drawdown_analysis=drawdown_analysis,
            tail_metrics=tail_metrics
        )


# ------------------------------------------------------------
# Enhanced Visualizations
# ------------------------------------------------------------
class EnhancedVisualizer:
    """Enhanced visualization suite"""
    
    @staticmethod
    def create_dashboard(
        returns: pd.DataFrame,
        weights: np.ndarray,
        risk_results: EnhancedRiskResults,
        benchmark_returns: Optional[pd.Series] = None
    ) -> go.Figure:
        """Create comprehensive risk dashboard"""
        
        port_returns = risk_results.port_returns
        
        fig = make_subplots(
            rows=4, cols=3,
            subplot_titles=(
                'Cumulative Returns', 'Drawdown Analysis', 'Rolling Volatility',
                'Risk Contribution', 'Monthly Returns Heatmap', 'Rolling Sharpe',
                'Correlation Matrix', 'VaR Distribution', 'Risk Metrics',
                'Sector Allocation', 'Top Holdings', 'Performance Summary'
            ),
            specs=[
                [{'secondary_y': True}, {'secondary_y': False}, {'secondary_y': False}],
                [{'secondary_y': False}, {'secondary_y': False}, {'secondary_y': False}],
                [{'secondary_y': False}, {'secondary_y': False}, {'secondary_y': False}],
                [{'secondary_y': False}, {'secondary_y': False}, {'type': 'table'}]
            ],
            vertical_spacing=0.08,
            horizontal_spacing=0.1
        )
        
        # 1. Cumulative Returns
        cum_ret = (1 + port_returns).cumprod()
        fig.add_trace(
            go.Scatter(x=cum_ret.index, y=cum_ret.values, name='Portfolio', line=dict(color='blue', width=2)),
            row=1, col=1
        )
        
        if benchmark_returns is not None:
            common_idx = cum_ret.index.intersection(benchmark_returns.index)
            bench_cum = (1 + benchmark_returns[common_idx]).cumprod()
            fig.add_trace(
                go.Scatter(x=bench_cum.index, y=bench_cum.values, name='BIST100',
                          line=dict(color='gray', width=1, dash='dash')),
                row=1, col=1
            )
        
        # 2. Drawdown Analysis
        drawdown = risk_results.drawdown_analysis['Drawdown'] * 100
        fig.add_trace(
            go.Scatter(x=drawdown.index, y=drawdown.values, fill='tozeroy',
                      name='Drawdown %', line=dict(color='red')),
            row=1, col=2
        )
        
        # Mark maximum drawdown
        max_dd_idx = drawdown.idxmin()
        fig.add_trace(
            go.Scatter(x=[max_dd_idx], y=[drawdown.min()], mode='markers',
                      marker=dict(size=10, color='darkred', symbol='x'),
                      name=f'Max DD: {drawdown.min():.1f}%'),
            row=1, col=2
        )
        
        # 3. Rolling Volatility
        roll_vol = port_returns.rolling(60).std() * np.sqrt(252) * 100
        fig.add_trace(
            go.Scatter(x=roll_vol.index, y=roll_vol.values, name='60d Rolling Vol %',
                      line=dict(color='orange', width=2)),
            row=1, col=3
        )
        
        # 4. Risk Contribution
        risk_df = risk_results.risk_table.nlargest(15, 'Risk_Contribution_%')
        fig.add_trace(
            go.Bar(
                y=risk_df['Company'],
                x=risk_df['Risk_Contribution_%'],
                orientation='h',
                name='Risk Contribution',
                marker=dict(
                    color=risk_df['Risk_Contribution_%'],
                    colorscale='RdYlGn_r',
                    showscale=True
                )
            ),
            row=2, col=1
        )
        
        # 5. Monthly Returns Heatmap
        monthly_ret = port_returns.resample('M').apply(lambda x: (1 + x).prod() - 1) * 100
        years = monthly_ret.index.year.unique()
        months = range(1, 13)
        
        heatmap_data = []
        for y in sorted(years):
            year_data = []
            for m in months:
                val = monthly_ret[(monthly_ret.index.year == y) & (monthly_ret.index.month == m)]
                year_data.append(val.iloc[0] if not val.empty else 0)
            heatmap_data.append(year_data)
        
        fig.add_trace(
            go.Heatmap(
                z=heatmap_data,
                x=[f'Month {m}' for m in months],
                y=sorted(years),
                colorscale='RdYlGn',
                zmid=0,
                showscale=False,
                hovertemplate='Year: %{y}<br>Month: %{x}<br>Return: %{z:.1f}%<extra></extra>'
            ),
            row=2, col=2
        )
        
        # 6. Rolling Sharpe
        roll_sharpe = port_returns.rolling(252).mean() / port_returns.rolling(252).std() * np.sqrt(252)
        fig.add_trace(
            go.Scatter(x=roll_sharpe.index, y=roll_sharpe.values, name='1Y Rolling Sharpe',
                      line=dict(color='purple', width=2)),
            row=2, col=3
        )
        fig.add_hline(y=1, line_dash="dash", line_color="green", row=2, col=3,
                     annotation_text="Good Sharpe")
        
        # 7. Correlation Matrix (Top 15 assets)
        top_assets = risk_results.risk_table.nlargest(15, 'Weight')['Symbol'].tolist()
        corr_matrix = returns[top_assets].corr()
        
        fig.add_trace(
            go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='RdBu',
                zmid=0,
                text=corr_matrix.values.round(2),
                texttemplate='%{text}',
                textfont={"size": 8},
                showscale=False
            ),
            row=3, col=1
        )
        
        # 8. VaR Distribution
        var_95 = risk_results.tail_metrics.get('VaR_95', 0) * 100
        var_99 = risk_results.tail_metrics.get('VaR_99', 0) * 100
        
        fig.add_trace(
            go.Histogram(x=port_returns * 100, nbinsx=50, name='Returns Distribution',
                        marker_color='lightblue'),
            row=3, col=2
        )
        fig.add_vline(x=var_95, line_dash="dash", line_color="red", row=3, col=2,
                     annotation_text=f"VaR95: {var_95:.1f}%")
        fig.add_vline(x=var_99, line_dash="dash", line_color="darkred", row=3, col=2,
                     annotation_text=f"VaR99: {var_99:.1f}%")
        
        # 9. Risk Metrics Summary
        metrics_text = [
            f"Volatility: {risk_results.portfolio_metrics['volatility']*100:.1f}%",
            f"Total Return: {risk_results.portfolio_metrics['total_return']*100:.1f}%",
            f"CAGR: {risk_results.portfolio_metrics['cagr']*100:.1f}%",
            f"Sharpe: {risk_results.portfolio_metrics.get('Sharpe_Ratio', 0):.2f}",
            f"Max DD: {risk_results.portfolio_metrics['max_drawdown']*100:.1f}%",
            f"Div Ratio: {risk_results.portfolio_metrics['diversification_ratio']:.2f}",
            f"VaR95: {risk_results.tail_metrics.get('VaR_95', 0)*100:.1f}%",
            f"CVaR95: {risk_results.tail_metrics.get('CVaR_95', 0)*100:.1f}%"
        ]
        
        fig.add_trace(
            go.Table(
                header=dict(values=['Metric', 'Value']),
                cells=dict(values=[
                    [m.split(':')[0] for m in metrics_text],
                    [m.split(':')[1].strip() for m in metrics_text]
                ])
            ),
            row=3, col=3
        )
        
        # 10. Sector Allocation
        sector_data = risk_results.risk_table.groupby('Sector')['Weight'].sum() * 100
        fig.add_trace(
            go.Pie(
                labels=sector_data.index,
                values=sector_data.values,
                hole=0.3,
                textinfo='label+percent',
                showlegend=False
            ),
            row=4, col=1
        )
        
        # 11. Top Holdings
        top_holdings = risk_results.risk_table.nlargest(5, 'Weight')[['Company', 'Weight']]
        fig.add_trace(
            go.Bar(
                x=top_holdings['Weight'] * 100,
                y=top_holdings['Company'],
                orientation='h',
                marker_color='green'
            ),
            row=4, col=2
        )
        
        # 12. Performance Summary Table
        perf_summary = pd.DataFrame({
            'Metric': ['Alpha', 'Beta', 'Info Ratio', 'Tracking Error'],
            'Value': ['N/A', 'N/A', 'N/A', 'N/A']
        })
        
        fig.add_trace(
            go.Table(
                header=dict(values=['Metric', 'Value']),
                cells=dict(values=[perf_summary['Metric'], perf_summary['Value']])
            ),
            row=4, col=3
        )
        
        # Update layout
        fig.update_layout(
            height=1600,
            title_text="BIST Advanced Risk Dashboard",
            title_font_size=24,
            showlegend=False,
            template='plotly_white'
        )
        
        return fig
    
    @staticmethod
    def plot_efficient_frontier(
        returns: pd.DataFrame,
        risk_free_rate: float = 0.15
    ) -> go.Figure:
        """Plot efficient frontier with optimal portfolios"""
        
        if not PYPFOPT_AVAILABLE:
            fig = go.Figure()
            fig.add_annotation(
                text="PyPortfolioOpt not installed. Install for efficient frontier visualization.",
                showarrow=False,
                font=dict(size=14)
            )
            return fig
        
        try:
            mu = expected_returns.mean_historical_return(returns)
            S = risk_models.CovarianceShrinkage(returns).ledoit_wolf()
            
            # Generate efficient frontier
            ef = EfficientFrontier(mu, S)
            
            # Get frontier points
            target_returns = np.linspace(mu.min(), mu.max() * 1.5, 50)
            volatilities = []
            
            for target in target_returns:
                try:
                    ef = EfficientFrontier(mu, S)
                    ef.efficient_return(target)
                    volatilities.append(ef.portfolio_performance()[1])
                except:
                    volatilities.append(np.nan)
            
            # Get optimal portfolios
            ef = EfficientFrontier(mu, S)
            ef.add_constraint(lambda x: x >= 0)
            
            # Max Sharpe
            ef.max_sharpe(risk_free_rate=risk_free_rate)
            max_sharpe_ret, max_sharpe_vol, max_sharpe_sr = ef.portfolio_performance()
            
            # Min Volatility
            ef.min_volatility()
            min_vol_ret, min_vol_vol, min_vol_sr = ef.portfolio_performance()
            
            # Create figure
            fig = go.Figure()
            
            # Efficient frontier
            fig.add_trace(go.Scatter(
                x=volatilities,
                y=target_returns,
                mode='lines',
                name='Efficient Frontier',
                line=dict(color='blue', width=2)
            ))
            
            # Max Sharpe portfolio
            fig.add_trace(go.Scatter(
                x=[max_sharpe_vol],
                y=[max_sharpe_ret],
                mode='markers+text',
                name='Max Sharpe',
                marker=dict(size=15, color='red', symbol='star'),
                text=['Max Sharpe'],
                textposition='top center'
            ))
            
            # Min Volatility portfolio
            fig.add_trace(go.Scatter(
                x=[min_vol_vol],
                y=[min_vol_ret],
                mode='markers+text',
                name='Min Volatility',
                marker=dict(size=15, color='green', symbol='diamond'),
                text=['Min Vol'],
                textposition='bottom center'
            ))
            
            # Individual assets
            fig.add_trace(go.Scatter(
                x=np.sqrt(np.diag(S)),
                y=mu,
                mode='markers+text',
                name='Individual Assets',
                marker=dict(size=8, color='gray'),
                text=returns.columns,
                textposition='top center',
                textfont=dict(size=8)
            ))
            
            fig.update_layout(
                title='Efficient Frontier - BIST Portfolio',
                xaxis_title='Annualized Volatility',
                yaxis_title='Annualized Return',
                height=600,
                template='plotly_white',
                showlegend=True
            )
            
            return fig
            
        except Exception as e:
            fig = go.Figure()
            fig.add_annotation(
                text=f"Error generating efficient frontier: {str(e)}",
                showarrow=False,
                font=dict(size=12)
            )
            return fig


# ------------------------------------------------------------
# Main Application
# ------------------------------------------------------------
def main():
    # Header
    st.markdown('<div class="main-header">📊 Advanced BIST Risk Budgeting System</div>', unsafe_allow_html=True)
    
    # Data policy notice
    st.markdown(
        """
        <div class="note-box">
        <b>📡 Data Policy:</b> This app fetches prices exclusively from <b>Yahoo Finance</b> via <code>yfinance</code>.
        Missing data is handled with <b>forward-fill only</b> (limited) followed by strict date alignment.
        <br/><span class="small-muted">No synthetic data is generated. Tickers with insufficient history are automatically dropped.</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    st.markdown('<div class="data-source-badge">📡 Data Source: Yahoo Finance</div>', unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown("## ⚙️ Configuration Panel")
        
        with st.expander("📅 Date Range", expanded=True):
            c1, c2 = st.columns(2)
            with c1:
                start_date = st.date_input(
                    "Start",
                    value=date(2020, 1, 1),
                    max_value=date.today() - timedelta(days=30)
                )
            with c2:
                end_date = st.date_input(
                    "End",
                    value=date.today(),
                    max_value=date.today()
                )
        
        if start_date >= end_date:
            st.error("❌ Start date must be before end date.")
            st.stop()
        
        with st.expander("🎯 Portfolio Strategy", expanded=True):
            strategy = st.selectbox(
                "Optimization Method",
                [
                    "Equal Weight",
                    "Risk Parity (SciPy)",
                    "Min Volatility",
                    "Max Sharpe",
                    "Efficient Return",
                    "Efficient Risk",
                    "HRP (Hierarchical Risk Parity)",
                    "CLA (Critical Line Algorithm)",
                    "Black-Litterman"
                ],
                index=0,
                help="Select portfolio optimization method"
            )
            
            max_weight = st.slider(
                "Max Weight per Asset",
                min_value=0.02,
                max_value=0.30,
                value=0.12,
                step=0.01,
                help="Maximum allocation to any single stock"
            )
        
        with st.expander("🏭 Sector Constraints", expanded=False):
            enable_sector_caps = st.checkbox("Enable Sector Caps", value=False)
            sector_caps = {}
            if enable_sector_caps:
                cols = st.columns(2)
                with cols[0]:
                    sector_caps["Banking"] = st.slider("Banking", 0.10, 0.60, 0.35, 0.01)
                    sector_caps["Holding"] = st.slider("Holding", 0.05, 0.50, 0.25, 0.01)
                    sector_caps["Retail"] = st.slider("Retail", 0.05, 0.50, 0.20, 0.01)
                with cols[1]:
                    sector_caps["Industrial"] = st.slider("Industrial", 0.05, 0.70, 0.40, 0.01)
                    sector_caps["Aviation"] = st.slider("Aviation", 0.05, 0.50, 0.25, 0.01)
                    sector_caps["Other"] = st.slider("Other", 0.05, 0.80, 0.60, 0.01)
        
        with st.expander("📊 Data Quality", expanded=False):
            ffill_limit = st.slider("Forward-fill Limit (days)", 1, 10, 5, 1)
            min_obs = st.slider("Minimum Observations", 60, 500, 180, 10)
            max_missing_frac = st.slider("Max Missing Fraction", 0.0, 0.8, 0.25, 0.05)
        
        with st.expander("📈 Analytics Settings", expanded=False):
            roll_window = st.slider("Rolling Window (days)", 60, 504, 252, 21)
            top_n_roll = st.slider("Top N for Rolling Chart", 5, 20, 10, 1)
            
            include_benchmark = st.checkbox("Include BIST100 Benchmark", value=True)
            enable_stress = st.checkbox("Enable Stress Testing", value=True)
            
            if enable_stress:
                fx_shock_pct = st.slider("FX Shock (USDTRY %)", -30.0, 30.0, 5.0, 0.5)
                rate_shock_bp = st.slider("Rate Shock (bp)", -500, 500, 100, 25)
        
        if not PYPFOPT_AVAILABLE and strategy not in ["Equal Weight", "Risk Parity (SciPy)"]:
            st.warning(
                "⚠️ PyPortfolioOpt not available. "
                "Advanced optimization methods will fallback to equal weight."
            )
        
        run_button = st.button("🚀 Run Analysis", type="primary", use_container_width=True)

    # Main content area
    if run_button:
        with st.spinner("📥 Fetching data from Yahoo Finance..."):
            # Fetch data using enhanced fetcher
            fetcher = EnhancedDataFetcher()
            prices_raw, dropped_raw, fetch_metadata = fetcher.fetch_with_retry(
                BIST50_TICKERS, start_date, end_date
            )
        
        if prices_raw.empty:
            st.error("❌ No data received from Yahoo Finance.")
            with st.expander("🔍 Debug Information"):
                st.json(fetch_metadata)
            st.stop()
        
        # Clean prices
        with st.spinner("🧹 Cleaning and aligning data..."):
            prices, dropped_clean, clean_info = EnhancedDataFetcher.clean_prices_forward_fill(
                prices_raw, min_obs, max_missing_frac, ffill_limit
            )
        
        if prices.empty or prices.shape[1] < 2:
            st.error("❌ Insufficient assets after cleaning.")
            col1, col2 = st.columns(2)
            with col1:
                st.write("Dropped (raw fetch):", dropped_raw)
            with col2:
                st.write("Dropped (cleaning):", dropped_clean)
            st.stop()
        
        # Calculate returns
        returns = prices.pct_change().dropna()
        
        # Fetch benchmark if requested
        benchmark_returns = None
        if include_benchmark:
            for b in BENCHMARK_CANDIDATES:
                bench_data, _, _ = fetcher.fetch_with_retry([b], start_date, end_date)
                if not bench_data.empty and b in bench_data.columns:
                    benchmark_returns = bench_data[b].pct_change().dropna()
                    break
        
        # Data summary
        st.markdown('<div class="sub-header">✅ Data Summary</div>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Assets (final)", returns.shape[1])
        with col2:
            st.metric("Trading Days", returns.shape[0])
        with col3:
            st.metric("Date Range", f"{returns.index[0].strftime('%Y-%m-%d')}")
        with col4:
            st.metric("to", f"{returns.index[-1].strftime('%Y-%m-%d')}")
        
        with st.expander("🔍 Data Quality Details"):
            st.write("**Fetch Strategy:**", fetch_metadata.get("strategy", "unknown"))
            st.write("**Dropped Tickers:**", dropped_raw + dropped_clean)
            st.write("**Final Tickers:**", returns.columns.tolist())
            if include_benchmark and benchmark_returns is not None:
                st.write("**Benchmark:**", "BIST100 (XU100)")
        
        # Optimize portfolio
        with st.spinner("🧮 Optimizing portfolio..."):
            w, opt_note = EnhancedPortfolioOptimizer.optimize(
                prices, returns, strategy, max_weight, sector_caps if enable_sector_caps else {}
            )
        
        # Calculate risk metrics
        risk_engine = EnhancedRiskEngine()
        risk_results = risk_engine.compute(returns, w)
        
        # Display results
        st.markdown('<div class="sub-header">📊 Portfolio Analysis Results</div>', unsafe_allow_html=True)
        
        # Key metrics in cards
        cols = st.columns(4)
        with cols[0]:
            st.metric(
                "Annual Volatility",
                f"{risk_results.portfolio_metrics['volatility']:.2%}",
                help="Annualized portfolio volatility"
            )
        with cols[1]:
            st.metric(
                "Total Return",
                f"{risk_results.portfolio_metrics['total_return']:.2%}",
                help="Cumulative total return over the period"
            )
        with cols[2]:
            st.metric(
                "Max Drawdown",
                f"{risk_results.portfolio_metrics['max_drawdown']:.2%}",
                help="Maximum peak-to-trough decline"
            )
        with cols[3]:
            st.metric(
                "Diversification Ratio",
                f"{risk_results.portfolio_metrics['diversification_ratio']:.2f}",
                help="Weighted avg vol / portfolio vol"
            )
        
        # Second row of metrics
        cols = st.columns(4)
        with cols[0]:
            st.metric(
                "VaR (95%)",
                f"{risk_results.tail_metrics.get('VaR_95', 0):.2%}",
                help="Value at Risk at 95% confidence"
            )
        with cols[1]:
            st.metric(
                "CVaR (95%)",
                f"{risk_results.tail_metrics.get('CVaR_95', 0):.2%}",
                help="Conditional VaR at 95% confidence"
            )
        with cols[2]:
            st.metric(
                "Sharpe Ratio",
                f"{risk_results.portfolio_metrics.get('Sharpe_Ratio', 0):.2f}",
                help="Risk-adjusted return"
            )
        with cols[3]:
            st.metric(
                "Calmar Ratio",
                f"{risk_results.portfolio_metrics.get('Calmar_Ratio', 0):.2f}",
                help="CAGR / Max Drawdown"
            )
        
        # Create and display dashboard
        visualizer = EnhancedVisualizer()
        dashboard = visualizer.create_dashboard(
            returns, w, risk_results, benchmark_returns
        )
        st.plotly_chart(dashboard, use_container_width=True)
        
        # Efficient Frontier (if applicable)
        if strategy in ["Min Volatility", "Max Sharpe", "Efficient Return", "Efficient Risk"]:
            st.markdown('<div class="sub-header">📈 Efficient Frontier</div>', unsafe_allow_html=True)
            ef_plot = visualizer.plot_efficient_frontier(prices if prices is not None else returns)
            st.plotly_chart(ef_plot, use_container_width=True)
        
        # Detailed risk table
        st.markdown('<div class="sub-header">📋 Detailed Risk Metrics</div>', unsafe_allow_html=True)
        display_df = risk_results.risk_table.copy()
        display_df['Weight'] = display_df['Weight'] * 100
        display_df['Individual_Volatility'] = display_df['Individual_Volatility'] * 100
        display_df['Risk_Contribution_%'] = display_df['Risk_Contribution_%'].round(2)
        
        st.dataframe(
            display_df[[
                'Risk_Rank', 'Symbol', 'Company', 'Sector', 'Market_Cap',
                'Weight', 'Risk_Contribution_%', 'Individual_Volatility',
                'Risk_Weight_Ratio'
            ]].round(2),
            use_container_width=True,
            height=500
        )
        
        # Rolling risk contributions
        st.markdown('<div class="sub-header">📈 Rolling Risk Contributions</div>', unsafe_allow_html=True)
        
        @st.cache_data
        def calc_rolling_rc(returns_df, weights_tuple, window):
            w = np.array(weights_tuple)
            tickers = returns_df.columns.tolist()
            results = []
            dates = []
            
            for i in range(window, len(returns_df) + 1):
                sub = returns_df.iloc[i-window:i]
                cov = sub.cov() * 252
                port_var = w @ cov.values @ w
                port_vol = np.sqrt(max(port_var, 1e-12))
                mrc = (cov.values @ w) / port_vol
                rc_pct = (w * mrc / port_vol) * 100
                results.append(rc_pct)
                dates.append(returns_df.index[i-1])
            
            if results:
                return pd.DataFrame(results, index=dates, columns=tickers)
            return pd.DataFrame()
        
        rolling_rc = calc_rolling_rc(returns, tuple(w.tolist()), roll_window)
        
        if not rolling_rc.empty:
            latest = rolling_rc.iloc[-1].sort_values(ascending=False)
            top_symbols = latest.head(top_n_roll).index.tolist()
            
            fig_rolling = go.Figure()
            for sym in top_symbols:
                fig_rolling.add_trace(go.Scatter(
                    x=rolling_rc.index,
                    y=rolling_rc[sym],
                    mode='lines',
                    name=COMPANY_METADATA.get(sym, {}).get('name', sym),
                    line=dict(width=1.5)
                ))
            
            fig_rolling.update_layout(
                title=f'Rolling Risk Contribution (Window: {roll_window} days)',
                xaxis_title='Date',
                yaxis_title='Risk Contribution (%)',
                height=500,
                hovermode='x unified',
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=1.05
                )
            )
            
            st.plotly_chart(fig_rolling, use_container_width=True)
        else:
            st.info("Insufficient data for rolling calculations.")
        
        # Monte Carlo Simulation
        if enable_stress:
            st.markdown('<div class="sub-header">🎲 Monte Carlo Simulation</div>', unsafe_allow_html=True)
            
            with st.spinner("Running Monte Carlo simulation..."):
                mc_simulator = MonteCarloSimulator()
                sim_results = mc_simulator.simulate_returns(
                    returns, w, n_simulations=5000, horizon_days=252
                )
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Median Final Value", f"{sim_results['median_final']:.2f}x")
                with col2:
                    st.metric("VaR (95%)", f"{sim_results['var_95']:.2f}x")
                with col3:
                    st.metric("CVaR (95%)", f"{sim_results['cvar_95']:.2f}x")
                with col4:
                    st.metric("Loss Probability", f"{sim_results['probability_loss']:.1f}%")
                
                sim_plot = mc_simulator.plot_simulations(sim_results)
                st.plotly_chart(sim_plot, use_container_width=True)
        
        # Export functionality
        st.markdown('<div class="sub-header">📦 Export Report</div>', unsafe_allow_html=True)
        
        # Prepare export data
        export_data = {
            'Portfolio_Weights': pd.DataFrame({
                'Symbol': risk_results.risk_table['Symbol'],
                'Company': risk_results.risk_table['Company'],
                'Weight_%': risk_results.risk_table['Weight'] * 100,
                'Sector': risk_results.risk_table['Sector']
            }),
            'Risk_Metrics': risk_results.risk_table,
            'Portfolio_Summary': pd.DataFrame([risk_results.portfolio_metrics]),
            'Drawdown_Analysis': risk_results.drawdown_analysis,
            'Tail_Risk_Metrics': pd.DataFrame([risk_results.tail_metrics])
        }
        
        if not rolling_rc.empty:
            export_data['Rolling_Risk'] = rolling_rc.tail(500).reset_index()
        
        if enable_stress and 'sim_results' in locals():
            export_data['Monte_Carlo'] = pd.DataFrame({
                'Final_Values': sim_results['final_values']
            })
        
        # Convert to Excel
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            for sheet_name, df in export_data.items():
                if df is not None and not df.empty:
                    df.to_excel(writer, sheet_name=sheet_name[:31], index=False)
        
        st.download_button(
            "📥 Download Full Report (Excel)",
            data=output.getvalue(),
            file_name=f"bist_risk_report_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
    
    else:
        # Welcome screen
        st.info("👈 Configure parameters in the sidebar and click **Run Analysis** to start.")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            ### 🎯 Features
            - BIST50 Universe
            - Multiple optimization strategies
            - Advanced risk metrics
            - Sector constraints
            """)
        with col2:
            st.markdown("""
            ### 📊 Analytics
            - Risk decomposition
            - VaR / CVaR analysis
            - Drawdown analysis
            - Monte Carlo simulation
            """)
        with col3:
            st.markdown("""
            ### 📈 Visualizations
            - Interactive dashboards
            - Efficient frontier
            - Rolling metrics
            - Export capabilities
            """)
    
    # Footer
    st.markdown(
        """
        <div class="footer">
        <b>The Quantitative Analysis Performed by LabGen25@Istanbul by Murat KONUKLAR 2026</b>
        <br/>
        <span class="small-muted">Data Source: Yahoo Finance | Risk Models: PyPortfolioOpt, SciPy</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    # PyPortfolioOpt warning if needed
    if not PYPFOPT_AVAILABLE:
        st.sidebar.warning(
            "⚠️ **PyPortfolioOpt not available**\n\n"
            "This limits optimization options. To enable full functionality, "
            "ensure Python 3.11 and proper dependencies in Streamlit Cloud."
        )

if __name__ == "__main__":
    main()
