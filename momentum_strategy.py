import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def load_commodities_data(data_dir='data/commodities'):
    """Load and synchronize commodity parquet files into a single DataFrame with Close prices."""
    return load_all_data(data_dir)

def load_all_data(data_dir='data'):
    """Load and synchronize all parquet files into a single DataFrame with Close prices."""
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

def calculate_monthly_returns(prices_df):
    """Convert daily prices to monthly returns using month-end resampling."""
    monthly_prices = prices_df.resample('M').last()
    monthly_returns = monthly_prices.pct_change()
    return monthly_returns

def calculate_weekly_returns(prices_df):
    """Convert daily prices to weekly returns using Monday resampling."""
    weekly_prices = prices_df.resample('W-MON').last()
    weekly_returns = weekly_prices.pct_change()
    return weekly_returns

def calculate_momentum_signals(monthly_returns, lookback_months=12):
    """Calculate cumulative return signals with proper shift to avoid look-ahead bias."""
    def rolling_cumulative_return(x):
        if len(x) < 1:
            return np.nan
        return (1 + x).prod() - 1  # Cumulative return over the period
    
    momentum_scores = monthly_returns.rolling(window=lookback_months).apply(
        rolling_cumulative_return, raw=True
    )
    
    # CRITICAL: Shift signals by 1 period to avoid look-ahead bias
    # Signals at time t use data available until t-1
    shifted_signals = momentum_scores.shift(1)
    
    return shifted_signals

def calculate_weekly_momentum_signals(weekly_returns, lookback_weeks=1):
    """Calculate weekly momentum signals with proper shift to avoid look-ahead bias."""
    def rolling_cumulative_return(x):
        if len(x) < 1:
            return np.nan
        return (1 + x).prod() - 1  # Cumulative return over the period
    
    momentum_scores = weekly_returns.rolling(window=lookback_weeks).apply(
        rolling_cumulative_return, raw=True
    )
    
    # CRITICAL: Shift signals by 1 period to avoid look-ahead bias
    # Monday t uses signals calculated on week t-1
    shifted_signals = momentum_scores.shift(1)
    
    return shifted_signals

def calculate_risk_parity_weights(selected_assets, monthly_returns, date, lookback_months=12):
    """Calculate risk parity weights for selected assets."""
    # Get historical returns for selected assets up to the rebalancing date
    end_date = date
    start_date = monthly_returns.index[max(0, monthly_returns.index.get_loc(date) - lookback_months)]
    
    historical_returns = monthly_returns.loc[start_date:end_date, selected_assets]
    
    # Calculate volatilities (standard deviation)
    volatilities = historical_returns.std()
    
    # Handle edge cases
    volatilities = volatilities.fillna(1.0)  # If no data, use equal weight
    volatilities = volatilities.replace(0.0, volatilities[volatilities > 0].min() if (volatilities > 0).any() else 1.0)
    
    # Calculate inverse volatility weights
    inverse_vol = 1.0 / volatilities
    risk_parity_weights = inverse_vol / inverse_vol.sum()
    
    return risk_parity_weights

def calculate_weekly_risk_parity_weights(selected_assets, weekly_returns, date, lookback_weeks=4):
    """Calculate risk parity weights for selected assets using weekly data."""
    # Get historical returns for selected assets up to the rebalancing date
    end_date = date
    start_date = weekly_returns.index[max(0, weekly_returns.index.get_loc(date) - lookback_weeks)]
    
    historical_returns = weekly_returns.loc[start_date:end_date, selected_assets]
    
    # Calculate volatilities (standard deviation)
    volatilities = historical_returns.std()
    
    # Handle edge cases
    volatilities = volatilities.fillna(1.0)  # If no data, use equal weight
    volatilities = volatilities.replace(0.0, volatilities[volatilities > 0].min() if (volatilities > 0).any() else 1.0)
    
    # Calculate inverse volatility weights
    inverse_vol = 1.0 / volatilities
    risk_parity_weights = inverse_vol / inverse_vol.sum()
    
    return risk_parity_weights

def select_top_quantile(momentum_signals, monthly_returns, quantile=0.25):
    """Select top quantile performers and create risk parity allocation."""
    weights = pd.DataFrame(0.0, index=momentum_signals.index, columns=momentum_signals.columns)
    
    for date in momentum_signals.index:
        valid_signals = momentum_signals.loc[date].dropna()
        
        if len(valid_signals) > 0:
            # Select top performers based on cumulative returns
            n_select = max(1, int(len(valid_signals) * quantile))
            top_assets = valid_signals.nlargest(n_select).index
            
            # Calculate risk parity weights for selected assets
            if len(top_assets) > 0:
                rp_weights = calculate_risk_parity_weights(top_assets, monthly_returns, date)
                weights.loc[date, top_assets] = rp_weights
    
    return weights

def backtest_strategy(prices_df, weights_df):
    """Backtest the momentum strategy with monthly rebalancing."""
    monthly_prices = prices_df.resample('M').last()
    monthly_returns = monthly_prices.pct_change()
    
    # Align weights and returns
    aligned_weights = weights_df.reindex(monthly_returns.index, method='ffill')
    
    # Calculate portfolio returns
    portfolio_returns = (aligned_weights.shift(1) * monthly_returns).sum(axis=1)
    
    # Calculate cumulative returns
    portfolio_cumulative = (1 + portfolio_returns).cumprod()
    
    return portfolio_returns, portfolio_cumulative

def backtest_weekly_strategy(prices_df, weights_df):
    """Backtest the momentum strategy with weekly rebalancing."""
    weekly_prices = prices_df.resample('W-MON').last()
    weekly_returns = weekly_prices.pct_change()
    
    # Align weights and returns
    aligned_weights = weights_df.reindex(weekly_returns.index, method='ffill')
    
    # Calculate portfolio returns
    portfolio_returns = (aligned_weights.shift(1) * weekly_returns).sum(axis=1)
    
    # Calculate cumulative returns
    portfolio_cumulative = (1 + portfolio_returns).cumprod()
    
    return portfolio_returns, portfolio_cumulative

def calculate_performance_metrics(returns, benchmark_returns=None):
    """Calculate comprehensive performance metrics."""
    returns_clean = returns.dropna()
    
    metrics = {}
    
    # Basic metrics
    metrics['Total Return'] = (1 + returns_clean).prod() - 1
    metrics['Annualized Return'] = (1 + returns_clean).prod() ** (12 / len(returns_clean)) - 1
    metrics['Volatility'] = returns_clean.std() * np.sqrt(12)
    metrics['Sharpe Ratio'] = metrics['Annualized Return'] / metrics['Volatility']
    
    # Drawdown analysis
    cumulative = (1 + returns_clean).cumprod()
    running_max = cumulative.expanding().max()
    drawdowns = (cumulative - running_max) / running_max
    metrics['Max Drawdown'] = drawdowns.min()
    
    # Additional metrics
    metrics['Positive Months'] = (returns_clean > 0).mean()
    metrics['Average Monthly Return'] = returns_clean.mean()
    metrics['Best Month'] = returns_clean.max()
    metrics['Worst Month'] = returns_clean.min()
    
    # Benchmark comparison if provided
    if benchmark_returns is not None:
        bench_clean = benchmark_returns.dropna()
        aligned_dates = returns_clean.index.intersection(bench_clean.index)
        
        if len(aligned_dates) > 0:
            port_aligned = returns_clean.reindex(aligned_dates)
            bench_aligned = bench_clean.reindex(aligned_dates)
            
            metrics['Benchmark Total Return'] = (1 + bench_aligned).prod() - 1
            metrics['Benchmark Annualized Return'] = (1 + bench_aligned).prod() ** (12 / len(bench_aligned)) - 1
            metrics['Benchmark Volatility'] = bench_aligned.std() * np.sqrt(12)
            metrics['Benchmark Sharpe'] = metrics['Benchmark Annualized Return'] / metrics['Benchmark Volatility']
            
            bench_cumulative = (1 + bench_aligned).cumprod()
            bench_running_max = bench_cumulative.expanding().max()
            bench_drawdowns = (bench_cumulative - bench_running_max) / bench_running_max
            metrics['Benchmark Max Drawdown'] = bench_drawdowns.min()
            
            # Excess return metrics
            excess_returns = port_aligned - bench_aligned
            metrics['Excess Return'] = excess_returns.mean() * 12
            metrics['Information Ratio'] = excess_returns.mean() / excess_returns.std() * np.sqrt(12)
    
    return metrics

def run_full_strategy(data_dir='data', lookback_months=12, top_quantile=0.25):
    """Run the complete momentum strategy pipeline."""
    print("Loading data...")
    prices = load_all_data(data_dir)
    
    print("Calculating monthly returns...")
    monthly_returns = calculate_monthly_returns(prices)
    
    print("Calculating momentum signals...")
    momentum_signals = calculate_momentum_signals(monthly_returns, lookback_months)
    
    print("Selecting top performers...")
    weights = select_top_quantile(momentum_signals, monthly_returns, top_quantile)
    
    print("Running backtest...")
    portfolio_returns, portfolio_cumulative = backtest_strategy(prices, weights)
    
    # Calculate benchmark (SPY) performance if available
    benchmark_returns = None
    if 'SPY' in monthly_returns.columns:
        benchmark_returns = monthly_returns['SPY']
    
    print("Calculating performance metrics...")
    metrics = calculate_performance_metrics(portfolio_returns, benchmark_returns)
    
    return {
        'prices': prices,
        'monthly_returns': monthly_returns,
        'momentum_signals': momentum_signals,
        'weights': weights,
        'portfolio_returns': portfolio_returns,
        'portfolio_cumulative': portfolio_cumulative,
        'metrics': metrics
    }