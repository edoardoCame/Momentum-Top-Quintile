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

def calculate_cross_sectional_momentum_signals(prices_df):
    """Calculate cross-sectional momentum signals based on weekly returns."""
    # Convert to weekly prices (Monday resampling)
    weekly_prices = prices_df.resample('W-MON').last()
    
    # Calculate weekly returns
    weekly_returns = weekly_prices.pct_change()
    
    # Shift by 1 period to avoid look-ahead bias
    # Monday t uses performance from week t-1
    shifted_returns = weekly_returns.shift(1)
    
    return weekly_returns, shifted_returns

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
    # Convert to weekly prices
    weekly_prices = prices_df.resample('W-MON').last()
    
    # Create equal weight allocation
    n_assets = len(weekly_prices.columns)
    equal_weight = 1.0 / n_assets
    
    # Create weights dataframe
    weights = pd.DataFrame(equal_weight, 
                          index=weekly_prices.index, 
                          columns=weekly_prices.columns)
    
    return weights

def backtest_cross_sectional_strategy(prices_df, weights_df):
    """Backtest the cross-sectional momentum strategy with weekly rebalancing."""
    # Convert to weekly data
    weekly_prices = prices_df.resample('W-MON').last()
    weekly_returns = weekly_prices.pct_change()
    
    # Align weights and returns
    aligned_weights = weights_df.reindex(weekly_returns.index, method='ffill')
    
    # Calculate portfolio returns: weights at t-1 * returns at t
    portfolio_returns = (aligned_weights.shift(1) * weekly_returns).sum(axis=1)
    
    # Calculate cumulative returns
    portfolio_cumulative = (1 + portfolio_returns).cumprod()
    
    return portfolio_returns, portfolio_cumulative

def calculate_performance_metrics(returns):
    """Calculate comprehensive performance metrics for weekly returns."""
    returns_clean = returns.dropna()
    
    metrics = {}
    
    # Basic metrics (annualized)
    metrics['Total Return'] = (1 + returns_clean).prod() - 1
    metrics['Annualized Return'] = (1 + returns_clean).prod() ** (52 / len(returns_clean)) - 1
    metrics['Volatility'] = returns_clean.std() * np.sqrt(52)
    
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
    metrics['Positive Weeks'] = (returns_clean > 0).mean()
    metrics['Average Weekly Return'] = returns_clean.mean()
    metrics['Best Week'] = returns_clean.max()
    metrics['Worst Week'] = returns_clean.min()
    
    return metrics

def run_cross_sectional_momentum_strategy(data_dir='data/commodities', quantile=0.25):
    """Run the complete cross-sectional momentum strategy pipeline."""
    print("Loading commodities data...")
    prices = load_commodities_data(data_dir)
    
    print("Calculating cross-sectional momentum signals...")
    weekly_returns, momentum_signals = calculate_cross_sectional_momentum_signals(prices)
    
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
        'weekly_returns': weekly_returns,
        'momentum_signals': momentum_signals,
        'strategy_weights': strategy_weights,
        'benchmark_weights': benchmark_weights,
        'strategy_returns': strategy_returns,
        'strategy_cumulative': strategy_cumulative,
        'benchmark_returns': benchmark_returns,
        'benchmark_cumulative': benchmark_cumulative,
        'strategy_metrics': strategy_metrics,
        'benchmark_metrics': benchmark_metrics
    }