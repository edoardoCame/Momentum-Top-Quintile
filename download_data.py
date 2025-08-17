#!/usr/bin/env python3
"""
Scarica serie storiche da yfinance e salva singoli file Parquet per ciascun ticker.

Default: commodities e coppie FX più liquide, dal 2000-01-01 ad oggi.

Usage:
  python download_data.py --outdir data

Opzioni principali:
  --start, --end, --interval, --tickers-file

I file verranno salvati in <outdir>/<ticker>.parquet
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import time
from typing import List

import pandas as pd
import yfinance as yf

DEFAULT_START = "2000-01-01"

DEFAULT_TICKERS = [
    # Commodities (Yahoo Finance futures symbols)
    "GC=F",  # Gold
    "SI=F",  # Silver
    "CL=F",  # Crude Oil WTI
    "BZ=F",  # Brent Oil
    "NG=F",  # Natural Gas
    "HG=F",  # Copper
    "ZC=F",  # Corn
    "ZW=F",  # Wheat
    "ZS=F",  # Soybeans
    "KC=F",  # Coffee
    "CT=F",  # Cotton
    "SB=F",  # Sugar

    # FX (most liquid)
    "EURUSD=X",
    "USDJPY=X",
    "GBPUSD=X",
    "USDCHF=X",
    "AUDUSD=X",
    "USDCAD=X",
    "NZDUSD=X",
    "EURJPY=X",
    "EURGBP=X",
    # Additional liquid FX pairs
    "EURAUD=X",
    "EURCAD=X",
    "GBPJPY=X",
    "CADJPY=X",
    "AUDJPY=X",
    "USDNOK=X",
    "USDSEK=X",
    "USDMXN=X",
    "USDZAR=X",
    "USDHKD=X",
    # Common crosses (no USD)
    "EURCHF=X",
    "EURNZD=X",
    "GBPAUD=X",
    "GBPCAD=X",
    "GBPNZD=X",
    "GBPCHF=X",
    "AUDNZD=X",
    "AUDCAD=X",
    "CADCHF=X",
    "NZDJPY=X",
    "NZDCHF=X",
    "CHFJPY=X",

    # Bond ETFs - Treasury
    "TLT",   # iShares 20+ Year Treasury Bond ETF
    "IEF",   # iShares 7-10 Year Treasury Bond ETF
    "SHY",   # iShares 1-3 Year Treasury Bond ETF
    "GOVT",  # iShares U.S. Treasury Bond ETF
    "BIL",   # SPDR Bloomberg Barclays 1-3 Month T-Bill ETF
    "SCHR",  # Schwab Intermediate-Term Treasury ETF
    "SCHQ",  # Schwab Long Treasury ETF

    # Bond ETFs - Corporate & Aggregate
    "AGG",   # iShares Core US Aggregate Bond ETF
    "BND",   # Vanguard Total Bond Market ETF
    "LQD",   # iShares iBoxx $ Investment Grade Corporate Bond ETF
    "VCIT",  # Vanguard Intermediate-Term Corporate Bond ETF
    "VCSH",  # Vanguard Short-Term Corporate Bond ETF
    "HYG",   # iShares iBoxx $ High Yield Corporate Bond ETF

    # US Stock Market Index ETFs
    "SPY",   # SPDR S&P 500 ETF Trust
    "QQQ",   # Invesco QQQ Trust (Nasdaq 100)
    "VTI",   # Vanguard Total Stock Market ETF
    "IVV",   # iShares Core S&P 500 ETF
    "VOO",   # Vanguard S&P 500 ETF
    "SCHX",  # Schwab US Large-Cap ETF
    "VXF",   # Vanguard Extended Market ETF
    "IWM",   # iShares Russell 2000 ETF (Small Cap)

    # International Stock ETFs
    "EFA",   # iShares MSCI EAFE ETF (Developed Markets)
    "EEM",   # iShares MSCI Emerging Markets ETF
    "VGK",   # Vanguard FTSE Europe ETF
    "VEA",   # Vanguard FTSE Developed Markets ETF
    "VXUS",  # Vanguard Total International Stock ETF
    "IEFA",  # iShares Core MSCI EAFE IMI Index ETF
    "IEMG",  # iShares Core MSCI Emerging Markets IMI Index ETF

    # Sector ETFs
    "XLE",   # Energy Select Sector SPDR Fund
    "XLF",   # Financial Select Sector SPDR Fund
    "XLK",   # Technology Select Sector SPDR Fund
]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def read_tickers_file(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
    return lines


def fetch_and_save_ticker(ticker: str, start: str, end: str, interval: str, outdir: str, max_retries: int = 3) -> None:
    # build a filesystem-safe filename by replacing '=' with '_'
    safe_name = ticker.replace("=", "_")
    outpath = os.path.join(outdir, f"{safe_name}.parquet")
    if os.path.exists(outpath):
        print(f"Skipping {ticker} -> {os.path.basename(outpath)} (file exists)")
        return

    attempt = 0
    while attempt < max_retries:
        try:
            print(f"Downloading {ticker} (start={start} end={end} interval={interval})...")
            df = yf.download(ticker, start=start, end=end, interval=interval, progress=False, threads=False)
            if df is None or df.empty:
                print(f"Warning: no data for {ticker}")
                # create empty dataframe with standard columns to keep schema
                df = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Adj Close", "Volume"]) 

            # ensure DatetimeIndex
            if not isinstance(df.index, pd.DatetimeIndex):
                try:
                    df.index = pd.to_datetime(df.index)
                except Exception:
                    pass

            # save as parquet with pyarrow (fast and widely compatible)
            df.to_parquet(outpath, engine="pyarrow", index=True)
            print(f"Saved {outpath} (rows={len(df)})")
            return
        except Exception as exc:
            attempt += 1
            wait = 2 ** attempt
            print(f"Error downloading {ticker}: {exc} (attempt {attempt}/{max_retries}), retrying in {wait}s...")
            time.sleep(wait)

    print(f"Failed to download {ticker} after {max_retries} attempts")


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Scarica dati storici da yfinance e salva in Parquet per ticker singoli.")
    parser.add_argument("--outdir", "-o", default="/Users/edoardocamerinelli/Desktop/TSMOM/data", help="Directory di output (default: /Users/edoardocamerinelli/Desktop/TSMOM/data)")
    parser.add_argument("--start", "-s", default=DEFAULT_START, help="Data di inizio (YYYY-MM-DD)")
    parser.add_argument("--end", "-e", default=dt.date.today().isoformat(), help="Data di fine (YYYY-MM-DD)")
    parser.add_argument("--interval", "-i", default="1d", help="Intervallo (1d, 1wk, 1mo, etc.)")
    parser.add_argument("--tickers-file", "-f", help="File con lista di tickers (uno per riga). Se presente, sovrascrive la lista default.")
    parser.add_argument("--tickers", "-t", help="Lista separata da virgola di tickers (sovrascrive default e file)")
    args = parser.parse_args(argv)

    start = args.start
    end = args.end
    interval = args.interval
    outdir = args.outdir

    ensure_dir(outdir)

    if args.tickers:
        tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
    elif args.tickers_file:
        tickers = read_tickers_file(args.tickers_file)
    else:
        tickers = DEFAULT_TICKERS

    print(f"Tickers to download: {len(tickers)} items")

    for ticker in tickers:
        fetch_and_save_ticker(ticker, start, end, interval, outdir)

    print("Done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
