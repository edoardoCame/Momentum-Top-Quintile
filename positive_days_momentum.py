import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def load_all_assets_data(data_dir='data'):
    """Load and synchronize all asset parquet files into a single DataFrame with Close prices."""
    data_path = Path(data_dir)
    all_prices = {}
    
    # Process all subdirectories
    for asset_class_dir in data_path.iterdir():
        if asset_class_dir.is_dir():
            parquet_files = list(asset_class_dir.glob('*.parquet'))
            
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
                        # Use asset_class_filename format for unique identification
                        key = f"{asset_class_dir.name}_{filename_ticker}"
                        all_prices[key] = close_price
                else:
                    # Handle non-multi-level columns
                    if 'Close' in df.columns:
                        close_price = df['Close']
                        key = f"{asset_class_dir.name}_{filename_ticker}"
                        all_prices[key] = close_price
    
    prices_df = pd.DataFrame(all_prices)
    prices_df.index = pd.to_datetime(prices_df.index)
    prices_df = prices_df.sort_index()
    
    return prices_df

def calculate_positive_days_signals(prices_df, lookback_days=5):
    """Calculate positive days momentum signals based on count of positive days in lookback period.
    
    Parameters:
    -----------
    prices_df : pd.DataFrame
        Daily price data
    lookback_days : int
        Number of days to look back for positive days counting (default: 5)
    
    Returns:
    --------
    tuple: (daily_returns, positive_days_signals)
        daily_returns: Daily returns for all assets
        positive_days_signals: Count of positive days in lookback period, shifted to avoid lookahead
    """
    daily_prices = prices_df.copy()
    
    # Calculate daily returns
    daily_returns = daily_prices.pct_change()
    
    # Create binary positive/negative indicator
    positive_days = (daily_returns > 0).astype(int)
    
    # Count positive days in rolling window
    positive_days_count = positive_days.rolling(window=lookback_days, min_periods=1).sum()
    
    # Shift by 1 period to avoid look-ahead bias
    # Day t uses positive days count calculated from day t-lookback_days to day t-1
    positive_days_signals = positive_days_count.shift(1)
    
    return daily_returns, positive_days_signals

def select_long_short_positive_days(positive_days_signals, quantile=0.25):
    """Select top quantile for long positions and bottom quantile for short positions based on positive days count.
    
    Parameters:
    -----------
    positive_days_signals : pd.DataFrame
        Count of positive days signals
    quantile : float
        Quantile to select for long (top) and short (bottom) positions
    
    Returns:
    --------
    pd.DataFrame
        Weights with positive values for long positions, negative for short positions
    """
    weights = pd.DataFrame(0.0, index=positive_days_signals.index, columns=positive_days_signals.columns)
    
    for date in positive_days_signals.index:
        # Get valid signals for this date
        valid_signals = positive_days_signals.loc[date].dropna()
        
        if len(valid_signals) > 0:
            # Calculate how many assets to select for each side
            n_select = max(1, int(len(valid_signals) * quantile))
            
            # Select top performers for long positions
            long_assets = valid_signals.nlargest(n_select).index
            
            # Select bottom performers for short positions
            short_assets = valid_signals.nsmallest(n_select).index
            
            # Apply equal weighting (normalized so total long + short = 1)
            if len(long_assets) > 0 and len(short_assets) > 0:
                # Equal weight for longs and shorts, each side gets 50% of capital
                long_weight = 0.5 / len(long_assets)
                short_weight = -0.5 / len(short_assets)
                
                weights.loc[date, long_assets] = long_weight
                weights.loc[date, short_assets] = short_weight
    
    return weights

def calculate_benchmark_equal_weight(prices_df):
    """Calculate equal-weight benchmark across all assets."""
    daily_prices = prices_df.copy()
    
    # Create equal weight allocation
    n_assets = len(daily_prices.columns)
    equal_weight = 1.0 / n_assets
    
    # Create weights dataframe
    weights = pd.DataFrame(equal_weight, 
                          index=daily_prices.index, 
                          columns=daily_prices.columns)
    
    return weights

def backtest_positive_days_strategy(prices_df, weights_df):
    """Backtest the positive days momentum strategy with daily rebalancing.
    
    Parameters:
    -----------
    prices_df : pd.DataFrame
        Daily price data
    weights_df : pd.DataFrame
        Strategy weights
    """
    daily_prices = prices_df.copy()
    daily_returns = daily_prices.pct_change()
    
    # Align weights and returns
    aligned_weights = weights_df.reindex(daily_returns.index, method='ffill')
    
    # Calculate portfolio returns: weights at t-1 * returns at t
    portfolio_returns = (aligned_weights.shift(1) * daily_returns).sum(axis=1)
    
    # Calculate cumulative returns
    portfolio_cumulative = (1 + portfolio_returns).cumprod()
    
    return portfolio_returns, portfolio_cumulative

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

def run_positive_days_momentum_strategy(data_dir='data', quantile=0.25, lookback_days=5):
    """Run the complete positive days momentum long-short strategy pipeline with daily rebalancing.
    
    Parameters:
    -----------
    data_dir : str
        Directory containing asset data
    quantile : float
        Top/bottom quantile to select for long/short positions (default: 0.25 for top/bottom 25%)
    lookback_days : int
        Number of days to look back for positive days counting (default: 5)
    """
    print("Loading all assets data...")
    prices = load_all_assets_data(data_dir)
    
    print(f"Calculating positive days momentum signals (lookback: {lookback_days} days)...")
    daily_returns, positive_days_signals = calculate_positive_days_signals(prices, lookback_days)
    
    print("Selecting top quantile for long and bottom quantile for short positions...")
    strategy_weights = select_long_short_positive_days(positive_days_signals, quantile)
    
    print("Creating equal-weight benchmark...")
    benchmark_weights = calculate_benchmark_equal_weight(prices)
    
    print("Running strategy backtest...")
    strategy_returns, strategy_cumulative = backtest_positive_days_strategy(prices, strategy_weights)
    
    print("Running benchmark backtest...")
    benchmark_returns, benchmark_cumulative = backtest_positive_days_strategy(prices, benchmark_weights)
    
    print("Calculating performance metrics...")
    strategy_metrics = calculate_performance_metrics(strategy_returns)
    benchmark_metrics = calculate_performance_metrics(benchmark_returns)
    
    return {
        'prices': prices,
        'daily_returns': daily_returns,
        'positive_days_signals': positive_days_signals,
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