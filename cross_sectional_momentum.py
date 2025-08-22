import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def load_commodities_data(data_dir='data/commodities'):
    """Load and synchronize commodity parquet files into a single DataFrame with Close prices."""
    data_path = Path(data_dir)
    parquet_files = list(data_path.glob('*.parquet'))
    
    all_prices = {}
    
    for file in parquet_files:
        filename_ticker = file.stem
        df = pd.read_parquet(file)
        
        # Check if the data has multi-level columns
        if df.columns.nlevels > 1:
            # Get the actual ticker from the column (level 1)
            actual_ticker = df.columns.levels[1][0]
            
            # Access Close price using the actual ticker from the data
            if 'Close' in df.columns.levels[0]:
                close_price = df[('Close', actual_ticker)]
                # Use the filename (without extension) as the key for consistency
                all_prices[filename_ticker] = close_price
        else:
            # Handle non-multi-level columns (if any exist)
            if 'Close' in df.columns:
                close_price = df['Close']
                all_prices[filename_ticker] = close_price
    
    prices_df = pd.DataFrame(all_prices)
    prices_df.index = pd.to_datetime(prices_df.index)
    prices_df = prices_df.sort_index()
    
    return prices_df

def calculate_cross_sectional_momentum_signals(prices_df, lookback_days=1):
    """Calculate cross-sectional momentum signals based on rolling cumulative returns.
    
    Parameters:
    -----------
    prices_df : pd.DataFrame
        Daily price data
    lookback_days : int
        Number of days to look back for momentum calculation (default: 1)
    
    Returns:
    --------
    tuple: (daily_returns, momentum_signals)
        daily_returns: Daily returns for all assets
        momentum_signals: Rolling cumulative returns over lookback period, shifted to avoid lookahead
    """
    # Use daily prices directly
    daily_prices = prices_df.copy()
    
    # Calculate daily returns
    daily_returns = daily_prices.pct_change()
    
    if lookback_days == 1:
        # Single day momentum (original behavior)
        momentum_signals = daily_returns.shift(1)
    else:
        # Rolling cumulative returns over lookback period
        def rolling_cumulative_return(x):
            if len(x) < 1:
                return np.nan
            return (1 + x).prod() - 1  # Cumulative return over the period
        
        momentum_signals = daily_returns.rolling(window=lookback_days).apply(
            rolling_cumulative_return, raw=True
        )
        
        # Shift by 1 period to avoid look-ahead bias
        # Day t uses momentum calculated from day t-lookback_days to day t-1
        momentum_signals = momentum_signals.shift(1)
    
    return daily_returns, momentum_signals

def select_top_quantile_equal_weight(momentum_signals, quantile=0.25):
    """Select top quantile performers and apply equal weighting."""
    weights = pd.DataFrame(0.0, index=momentum_signals.index, columns=momentum_signals.columns)
    
    for date in momentum_signals.index:
        # Get valid signals for this date
        valid_signals = momentum_signals.loc[date].dropna()
        
        if len(valid_signals) > 0:
            # Calculate how many assets to select (top quantile)
            n_select = max(1, int(len(valid_signals) * quantile))
            
            # Select top performers based on weekly returns
            top_assets = valid_signals.nlargest(n_select).index
            
            # Apply equal weighting
            if len(top_assets) > 0:
                equal_weight = 1.0 / len(top_assets)
                weights.loc[date, top_assets] = equal_weight
    
    return weights

def calculate_benchmark_equal_weight(prices_df):
    """Calculate equal-weight benchmark across all commodities."""
    # Use daily prices directly
    daily_prices = prices_df.copy()
    
    # Create equal weight allocation
    n_assets = len(daily_prices.columns)
    equal_weight = 1.0 / n_assets
    
    # Create weights dataframe
    weights = pd.DataFrame(equal_weight, 
                          index=daily_prices.index, 
                          columns=daily_prices.columns)
    
    return weights

def backtest_cross_sectional_strategy(prices_df, weights_df, drawdown_filter=None):
    """Backtest the cross-sectional momentum strategy with daily rebalancing.
    
    Parameters:
    -----------
    prices_df : pd.DataFrame
        Daily price data
    weights_df : pd.DataFrame
        Strategy weights
    drawdown_filter : pd.Series, optional
        Binary filter to apply during high drawdown periods
    """
    # Use daily data directly
    daily_prices = prices_df.copy()
    daily_returns = daily_prices.pct_change()
    
    # Align weights and returns
    aligned_weights = weights_df.reindex(daily_returns.index, method='ffill')
    
    # Apply drawdown filter if provided
    if drawdown_filter is not None:
        # Align filter with returns
        aligned_filter = drawdown_filter.reindex(daily_returns.index, method='ffill')
        # Zero out weights when filter is 0 (stop investing)
        aligned_weights = aligned_weights.multiply(aligned_filter, axis=0)
    
    # Calculate portfolio returns: weights at t-1 * returns at t
    portfolio_returns = (aligned_weights.shift(1) * daily_returns).sum(axis=1)
    
    # Calculate cumulative returns
    portfolio_cumulative = (1 + portfolio_returns).cumprod()
    
    return portfolio_returns, portfolio_cumulative

def calculate_rolling_drawdown_filter(returns, lookback_days=90, drawdown_threshold=0.10):
    """Calculate rolling drawdown filter to stop investing during high drawdown periods.
    
    Parameters:
    -----------
    returns : pd.Series
        Daily returns series
    lookback_days : int
        Number of days to look back for drawdown calculation (default: 90)
    drawdown_threshold : float
        Drawdown threshold to trigger filter (default: 0.10 = 10%)
    
    Returns:
    --------
    pd.Series
        Binary filter (1 = invest, 0 = stop investing)
    """
    # Use lookback_days directly for daily data
    lookback_period = lookback_days
    
    # Calculate cumulative returns
    cumulative_returns = (1 + returns).cumprod()
    
    # Calculate rolling maximum and drawdown
    rolling_max = cumulative_returns.rolling(window=lookback_period, min_periods=1).max()
    rolling_drawdown = (cumulative_returns - rolling_max) / rolling_max
    
    # Create filter: 1 if drawdown is within threshold, 0 if exceeds threshold
    filter_signal = (rolling_drawdown >= -drawdown_threshold).astype(int)
    
    # Shift by 1 period to avoid lookahead bias
    filter_signal_shifted = filter_signal.shift(1)
    
    # Fill first NaN with 1 (allow investing at start)
    filter_signal_shifted.iloc[0] = 1
    
    return filter_signal_shifted

def calculate_performance_metrics(returns):
    """Calculate comprehensive performance metrics for daily returns."""
    returns_clean = returns.dropna()
    
    metrics = {}
    
    # Basic metrics (annualized for daily returns)
    metrics['Total Return'] = (1 + returns_clean).prod() - 1
    metrics['Annualized Return'] = (1 + returns_clean).prod() ** (252 / len(returns_clean)) - 1
    metrics['Volatility'] = returns_clean.std() * np.sqrt(252)
    
    # Handle division by zero for Sharpe ratio
    if metrics['Volatility'] != 0:
        metrics['Sharpe Ratio'] = metrics['Annualized Return'] / metrics['Volatility']
    else:
        metrics['Sharpe Ratio'] = 0.0
    
    # Drawdown analysis
    cumulative = (1 + returns_clean).cumprod()
    running_max = cumulative.expanding().max()
    drawdowns = (cumulative - running_max) / running_max
    metrics['Max Drawdown'] = drawdowns.min()
    
    # Additional metrics
    metrics['Positive Days'] = (returns_clean > 0).mean()
    metrics['Average Daily Return'] = returns_clean.mean()
    metrics['Best Day'] = returns_clean.max()
    metrics['Worst Day'] = returns_clean.min()
    
    return metrics


def run_cross_sectional_momentum_strategy(data_dir='data/commodities', quantile=0.25, lookback_days=1):
    """Run the complete cross-sectional momentum strategy pipeline with daily rebalancing.
    
    Parameters:
    -----------
    data_dir : str
        Directory containing commodity data
    quantile : float
        Top quantile to select (default: 0.25 for top 25%)
    lookback_days : int
        Number of days to look back for momentum calculation (default: 1)
    """
    print("Loading commodities data...")
    prices = load_commodities_data(data_dir)
    
    print(f"Calculating cross-sectional momentum signals (lookback: {lookback_days} days)...")
    daily_returns, momentum_signals = calculate_cross_sectional_momentum_signals(prices, lookback_days)
    
    print("Selecting top quantile with equal weighting...")
    strategy_weights = select_top_quantile_equal_weight(momentum_signals, quantile)
    
    print("Creating equal-weight benchmark...")
    benchmark_weights = calculate_benchmark_equal_weight(prices)
    
    print("Running strategy backtest...")
    strategy_returns, strategy_cumulative = backtest_cross_sectional_strategy(prices, strategy_weights)
    
    print("Running benchmark backtest...")
    benchmark_returns, benchmark_cumulative = backtest_cross_sectional_strategy(prices, benchmark_weights)
    
    print("Calculating performance metrics...")
    strategy_metrics = calculate_performance_metrics(strategy_returns)
    benchmark_metrics = calculate_performance_metrics(benchmark_returns)
    
    return {
        'prices': prices,
        'daily_returns': daily_returns,
        'momentum_signals': momentum_signals,
        'strategy_weights': strategy_weights,
        'benchmark_weights': benchmark_weights,
        'strategy_returns': strategy_returns,
        'strategy_cumulative': strategy_cumulative,
        'benchmark_returns': benchmark_returns,
        'benchmark_cumulative': benchmark_cumulative,
        'strategy_metrics': strategy_metrics,
        'benchmark_metrics': benchmark_metrics,
        'lookback_days': lookback_days
    }