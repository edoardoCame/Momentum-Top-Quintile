#!/usr/bin/env python3
"""Median-sign momentum and minimal cross-sectional median strategy.

Vectorized, no-lookahead implementation using parquet files in `data/`.

Three strategy variants are provided:
 - ts: time-series sign median (s_t)
 - cs: cross-sectional median on L-period cumulative returns
 - hybrid: require agreement between ts sign and cs direction

Defaults: L=63 (approx. quarter), 252 trading days/year for ann. metrics.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import sys


DATA_DIR = Path(__file__).parent / "data"


def load_prices(data_dir=DATA_DIR):
    # find parquet files under data/*/*.parquet
    files = sorted(data_dir.rglob("*.parquet"))
    series = []
    names = []
    for f in files:
        try:
            df = pd.read_parquet(f)
        except Exception as e:
            print(f"skipping {f} (read error): {e}")
            continue
        # try to detect price column
        col = None
        if isinstance(df, pd.DataFrame):
            # common column names
            for c in ("Close", "close", "Adj Close", "adjclose", "adj_close", "price", "Price"):
                if c in df.columns:
                    col = c
                    break
            if col is None:
                # if single-column DF, take that
                if df.shape[1] == 1:
                    col = df.columns[0]
                else:
                    # fallback: look for a column that is numeric
                    numeric = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]
                    if numeric:
                        col = numeric[0]
        else:
            # parquet may be a Series
            pass
        if col is not None:
            s = df[col].copy()
        else:
            # try to coerce to series: prefer first numeric column if any
            try:
                if isinstance(df, pd.DataFrame) and df.shape[1] >= 1:
                    s = df.iloc[:, 0].copy()
                else:
                    s = pd.Series(df).squeeze()
            except Exception:
                print(f"cannot extract price from {f}, skipping")
                continue
        # if we somehow still have a DataFrame, take first column
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0]
        s.index = pd.to_datetime(s.index)
        s = s.sort_index()
        ticker = f.stem
        # ensure it's a Series and set name to ticker
        try:
            s = s.rename(ticker)
        except Exception:
            s.name = ticker
        series.append(s)
        names.append(ticker)
    if not series:
        raise RuntimeError(f"no price series found under {data_dir}")
    prices = pd.concat(series, axis=1).dropna(how="all")
    prices = prices.sort_index()
    return prices


def compute_returns(prices):
    # avoid deprecated default fill_method warning; do not forward-fill before pct_change
    return prices.pct_change(fill_method=None)


def compute_s_t(returns, L):
    # median of signs of past L returns, using only r_{t-1}..r_{t-L}
    s = returns.shift(1).rolling(window=L, min_periods=L).apply(lambda x: np.median(np.sign(x)), raw=True)
    # round to -1,0,1 cleanly
    s = s.round().astype('Int64')
    return s


def compute_cumret(returns, L):
    # L-period compounded return using past returns r_{t-1}..r_{t-L}
    def cumprod_minus1(x):
        return np.prod(1 + x) - 1
    cum = returns.shift(1).rolling(window=L, min_periods=L).apply(cumprod_minus1, raw=True)
    return cum


def cs_positions(cumret):
    # cross-sectional median cut: +1 if > median, -1 if < median, 0 else
    med = cumret.median(axis=1)
    # subtract median row-wise
    pos = cumret.sub(med, axis=0)
    pos = pos.apply(np.sign)
    # convert to integer
    pos = pos.astype('Int64')
    return pos


def normalize_positions(pos):
    # pos is DataFrame of -1,0,1
    posf = pos.astype(float)
    longs = posf.where(posf > 0, 0.0)
    shorts = posf.where(posf < 0, 0.0)
    # normalize longs to sum to +1 and shorts to sum to -1 (so net zero when both present)
    long_sum = longs.sum(axis=1).replace(0, np.nan)
    short_sum = shorts.sum(axis=1).replace(0, np.nan)
    longs = longs.div(long_sum, axis=0).fillna(0.0)
    # For shorts: divide by absolute value of short_sum to normalize magnitude, 
    # keeping the negative sign
    shorts = shorts.div(-short_sum, axis=0).fillna(0.0) * -1.0  # Fix: multiply by -1 to keep shorts negative
    weights = longs + shorts
    return weights


def strategy_returns(weights, returns):
    # returns: same index/columns as weights. Elementwise product, sum across assets.
    strat_ret = (weights * returns).sum(axis=1)
    return strat_ret


def perf_stats(ser, trading_days=252):
    ser = ser.dropna()
    cum = (1 + ser).cumprod() - 1
    total_ret = cum.iloc[-1]
    ann_ret = (1 + total_ret) ** (trading_days / len(ser)) - 1 if len(ser) > 0 else np.nan
    ann_vol = ser.std() * np.sqrt(trading_days)
    sharpe = ann_ret / ann_vol if ann_vol and not np.isnan(ann_vol) else np.nan
    # max drawdown
    nav = (1 + ser).cumprod()
    peak = nav.cummax()
    dd = (nav - peak) / peak
    mdd = dd.min()
    return dict(total_return=total_ret, ann_return=ann_ret, ann_vol=ann_vol, sharpe=sharpe, max_drawdown=mdd)


def run(L=63, variant='hybrid'):
    prices = load_prices()
    returns = compute_returns(prices)
    print(f"Loaded {prices.shape[1]} tickers: {list(prices.columns)}")
    print("Rebalance frequency: daily (positions computed each trading day using past data only)")
    s_t = compute_s_t(returns, L)
    cumret = compute_cumret(returns, L)
    pos_cs = cs_positions(cumret)
    pos_ts = s_t

    if variant == 'cs':
        pos = pos_cs
    elif variant == 'ts':
        pos = pos_ts
    elif variant == 'hybrid':
        # require sign agreement: sign(cumret) equals s_t and s_t != 0
        sign_cum = np.sign(cumret).astype('Int64')
        agree = (s_t == sign_cum) & (s_t != 0)
        pos = s_t.where(agree, other=0).astype('Int64')
    else:
        raise ValueError('unknown variant')

    # treat NaN positions as 0 (insufficient history / missing data)
    pos = pos.fillna(0)

    weights = normalize_positions(pos)
    strat_ret = strategy_returns(weights, returns)

    stats = perf_stats(strat_ret)

    print(f"Variant: {variant}, L={L}")
    for k, v in stats.items():
        print(f"{k}: {v}")

    return dict(prices=prices, returns=returns, s_t=s_t, cumret=cumret, pos=pos, weights=weights, stats=stats)


if __name__ == '__main__':
    import argparse

    p = argparse.ArgumentParser(description='Median-sign momentum minimal strategy')
    p.add_argument('--L', type=int, default=63, help='lookback window in days')
    p.add_argument('--variant', choices=['ts', 'cs', 'hybrid'], default='hybrid')
    args = p.parse_args()
    run(L=args.L, variant=args.variant)