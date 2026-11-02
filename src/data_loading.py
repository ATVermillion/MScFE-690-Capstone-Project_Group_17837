"""
Data layer for the MScFE 690 capstone (Group 17837).

Mirrors sections 1-3 of notebooks/Capstone_Project.ipynb:
- adjusted daily prices for the SPY core and the aggressive / defensive satellite sleeves (yfinance)
- macro series from FRED (API key read from the FRED_API_KEY environment variable)
- equal-weighted sleeve returns and a cumulative-growth chart on a log scale

Run from the repository root:  python src/data_loading.py
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import yfinance as yf

# --- Universe definition (see config/config.yaml) ---
CORE_TICKER = "SPY"
AGGRESSIVE_TICKERS = ["NVDA", "AMD", "AVGO", "TSLA", "AMZN", "META", "NFLX", "CRM", "GS", "CAT"]
DEFENSIVE_TICKERS = ["TLT", "GLD"]
TICKERS = [CORE_TICKER] + AGGRESSIVE_TICKERS + DEFENSIVE_TICKERS

START_DATE = "2015-01-01"
END_DATE = "2026-10-01"   # exclusive in yfinance: data to 30 Sep 2026
FRED_SERIES = {"y10": "DGS10", "spread_10y2y": "T10Y2Y", "fedfunds": "DFF"}


def load_prices(tickers=TICKERS, start=START_DATE, end=END_DATE):
    """Download adjusted close prices and volumes; returns (prices, volumes)."""
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    prices = raw["Close"].dropna(how="all")
    volumes = raw["Volume"].reindex(prices.index)
    return prices, volumes


def fred(series_id, start=START_DATE, end=END_DATE, api_key=None):
    """Download one FRED series as a float Series indexed by date."""
    api_key = api_key or os.environ.get("FRED_API_KEY")
    if not api_key:
        raise RuntimeError("Set the FRED_API_KEY environment variable (free key from fred.stlouisfed.org).")
    params = {"series_id": series_id, "api_key": api_key, "file_type": "json",
              "observation_start": start, "observation_end": end}
    r = requests.get("https://api.stlouisfed.org/fred/series/observations", params=params, timeout=30)
    if r.status_code != 200:
        # report the status only: the full URL would expose the API key
        raise RuntimeError(f"FRED request for {series_id} failed (HTTP {r.status_code})")
    obs = pd.DataFrame(r.json()["observations"])
    values = pd.to_numeric(obs["value"], errors="coerce")       # "." marks missing days
    return pd.Series(values.values, index=pd.to_datetime(obs["date"]), name=series_id)


def load_macro(start=START_DATE, end=END_DATE):
    """Daily macro panel: 10-year yield, 10y-2y spread, effective fed funds rate."""
    return pd.DataFrame({name: fred(sid, start, end) for name, sid in FRED_SERIES.items()})


def sleeve_returns(prices):
    """Daily simple returns of the core and the two equal-weighted satellite sleeves."""
    simple = np.exp(np.log(prices / prices.shift(1)).dropna()) - 1
    return pd.DataFrame({
        "Core (SPY)": simple[CORE_TICKER],
        "Aggressive sleeve": simple[AGGRESSIVE_TICKERS].mean(axis=1),
        "Defensive sleeve": simple[DEFENSIVE_TICKERS].mean(axis=1),
    })


def plot_cumulative_growth(sleeves, save_path="results/figures/growth_core_aggressive_defensive.png"):
    """Cumulative growth of $1 for the core and both sleeves, log scale, labelled axes."""
    growth = (1 + sleeves).cumprod()
    colors = {"Core (SPY)": "black", "Aggressive sleeve": "tab:red", "Defensive sleeve": "tab:blue"}
    fig, ax = plt.subplots(figsize=(10, 5))
    for col in growth.columns:
        growth[col].plot(ax=ax, label=col, color=colors[col], linewidth=2 if col == "Core (SPY)" else 1.5)
    ax.set_title("Cumulative Growth of $1: SPY Core vs. Aggressive and Defensive Satellite Sleeves")
    ax.set_xlabel("Date")
    ax.set_ylabel("Growth of $1 (log scale)")
    ax.set_yscale("log")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=100)
    return fig


if __name__ == "__main__":
    prices, _ = load_prices()
    print(f"Prices: {prices.shape}, {prices.index.min().date()} to {prices.index.max().date()}")
    sleeves = sleeve_returns(prices)
    plot_cumulative_growth(sleeves)
    print("Saved results/figures/growth_core_aggressive_defensive.png")
    if os.environ.get("FRED_API_KEY"):
        macro = load_macro()
        print(f"Macro: {macro.shape[0]} days, columns {list(macro.columns)}")
    else:
        print("FRED_API_KEY not set: skipped macro download")
