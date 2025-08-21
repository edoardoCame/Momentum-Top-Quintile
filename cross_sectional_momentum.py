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

def backtest_cross_sectional_strategy(prices_df, weights_df, drawdown_filter=None):
    """Backtest the cross-sectional momentum strategy with weekly rebalancing.
    
    Parameters:
    -----------
    prices_df : pd.DataFrame
        Daily price data
    weights_df : pd.DataFrame
        Strategy weights
    drawdown_filter : pd.Series, optional
        Binary filter to apply during high drawdown periods
    """
    # Convert to weekly data
    weekly_prices = prices_df.resample('W-MON').last()
    weekly_returns = weekly_prices.pct_change()
    
    # Align weights and returns
    aligned_weights = weights_df.reindex(weekly_returns.index, method='ffill')
    
    # Apply drawdown filter if provided
    if drawdown_filter is not None:
        # Align filter with returns
        aligned_filter = drawdown_filter.reindex(weekly_returns.index, method='ffill')
        # Zero out weights when filter is 0 (stop investing)
        aligned_weights = aligned_weights.multiply(aligned_filter, axis=0)
    
    # Calculate portfolio returns: weights at t-1 * returns at t
    portfolio_returns = (aligned_weights.shift(1) * weekly_returns).sum(axis=1)
    
    # Calculate cumulative returns
    portfolio_cumulative = (1 + portfolio_returns).cumprod()
    
    return portfolio_returns, portfolio_cumulative

def calculate_rolling_drawdown_filter(returns, lookback_days=90, drawdown_threshold=0.10):
    """Calculate rolling drawdown filter to stop investing during high drawdown periods.
    
    Parameters:
    -----------
    returns : pd.Series
        Weekly returns series
    lookback_days : int
        Number of days to look back for drawdown calculation (default: 90)
    drawdown_threshold : float
        Drawdown threshold to trigger filter (default: 0.10 = 10%)
    
    Returns:
    --------
    pd.Series
        Binary filter (1 = invest, 0 = stop investing)
    """
    # Convert lookback days to weeks (approximately)
    lookback_weeks = int(lookback_days / 7)
    
    # Calculate cumulative returns
    cumulative_returns = (1 + returns).cumprod()
    
    # Calculate rolling maximum and drawdown
    rolling_max = cumulative_returns.rolling(window=lookback_weeks, min_periods=1).max()
    rolling_drawdown = (cumulative_returns - rolling_max) / rolling_max
    
    # Create filter: 1 if drawdown is within threshold, 0 if exceeds threshold
    filter_signal = (rolling_drawdown >= -drawdown_threshold).astype(int)
    
    # Shift by 1 period to avoid lookahead bias
    filter_signal_shifted = filter_signal.shift(1)
    
    # Fill first NaN with 1 (allow investing at start)
    filter_signal_shifted.iloc[0] = 1
    
    return filter_signal_shifted

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

def extract_avg_correlation(corr_matrices_series):
    """Extract average off-diagonal correlations from rolling correlation matrices."""
    avg_correlations = []
    
    for date, corr_matrix in corr_matrices_series.groupby(level=0):
        if corr_matrix.shape[0] > 1:  # Need at least 2x2 matrix
            # Get off-diagonal elements (exclude diagonal = 1.0)
            mask = ~np.eye(corr_matrix.shape[0], dtype=bool)
            off_diagonal = corr_matrix.values[mask]
            # Calculate mean of off-diagonal correlations
            avg_corr = np.nanmean(off_diagonal)
            avg_correlations.append(avg_corr)
        else:
            avg_correlations.append(np.nan)
    
    dates = corr_matrices_series.index.get_level_values(0).unique()
    return pd.Series(avg_correlations, index=dates)

def calculate_correlation_regime_filter(weekly_returns, short_window=4, long_window=52, threshold=1.2):
    """Calculate correlation regime filter based on rolling correlations - no lookahead bias."""
    
    # Calculate rolling correlation matrices (vectorized)
    short_corr_matrices = weekly_returns.rolling(window=short_window, min_periods=short_window).corr()
    long_corr_matrices = weekly_returns.rolling(window=long_window, min_periods=long_window).corr()
    
    # Extract average correlations for each time period
    short_corr_avg = extract_avg_correlation(short_corr_matrices)
    long_corr_avg = extract_avg_correlation(long_corr_matrices)
    
    # Calculate correlation ratio
    corr_ratio = short_corr_avg / long_corr_avg
    
    # Shift by 1 period to avoid look-ahead bias
    corr_ratio_shifted = corr_ratio.shift(1)
    
    # Create regime signal: 1 = momentum regime, -1 = contrarian regime
    regime_signal = np.where(corr_ratio_shifted > threshold, 1, -1)
    
    return pd.Series(regime_signal, index=weekly_returns.index, name='regime')

def select_bottom_quantile_equal_weight(momentum_signals, quantile=0.25):
    """Select bottom quantile performers (worst) for contrarian strategy with equal weighting."""
    weights = pd.DataFrame(0.0, index=momentum_signals.index, columns=momentum_signals.columns)
    
    for date in momentum_signals.index:
        # Get valid signals for this date
        valid_signals = momentum_signals.loc[date].dropna()
        
        if len(valid_signals) > 0:
            # Calculate how many assets to select (bottom quantile)
            n_select = max(1, int(len(valid_signals) * quantile))
            
            # Select worst performers based on weekly returns (for mean reversion)
            bottom_assets = valid_signals.nsmallest(n_select).index
            
            # Apply equal weighting
            if len(bottom_assets) > 0:
                equal_weight = 1.0 / len(bottom_assets)
                weights.loc[date, bottom_assets] = equal_weight
    
    return weights

def adaptive_cross_sectional_strategy(momentum_signals, regime_filter, quantile=0.25):
    """Adaptive strategy: momentum when regime=1, long/short contrarian when regime=-1."""
    weights = pd.DataFrame(0.0, index=momentum_signals.index, columns=momentum_signals.columns)
    
    for date in momentum_signals.index:
        # Skip if no regime signal available
        if date not in regime_filter.index or pd.isna(regime_filter.loc[date]):
            continue
            
        # Get valid signals for this date
        valid_signals = momentum_signals.loc[date].dropna()
        
        if len(valid_signals) > 0:
            # Calculate how many assets to select
            n_select = max(1, int(len(valid_signals) * quantile))
            
            # Momentum regime: long top performers only
            if regime_filter.loc[date] == 1:
                top_assets = valid_signals.nlargest(n_select).index
                if len(top_assets) > 0:
                    equal_weight = 1.0 / len(top_assets)
                    weights.loc[date, top_assets] = equal_weight
                    
            # Contrarian regime: long worst performers + short top performers
            else:
                bottom_assets = valid_signals.nsmallest(n_select).index  # Long worst performers
                top_assets = valid_signals.nlargest(n_select).index      # Short top performers
                
                # Long bottom quintile (expect bounce)
                if len(bottom_assets) > 0:
                    long_weight = 0.5 / len(bottom_assets)  # 50% allocated to longs
                    weights.loc[date, bottom_assets] = long_weight
                
                # Short top quintile (expect reversal)
                if len(top_assets) > 0:
                    short_weight = -0.5 / len(top_assets)  # 50% allocated to shorts
                    weights.loc[date, top_assets] = short_weight
    
    return weights

def run_adaptive_cross_sectional_strategy(data_dir='data/commodities', quantile=0.25, 
                                         short_window=4, long_window=52, threshold=1.2):
    """Run the adaptive regime-switching cross-sectional strategy."""
    print("Loading commodities data...")
    prices = load_commodities_data(data_dir)
    
    print("Calculating cross-sectional momentum signals...")
    weekly_returns, momentum_signals = calculate_cross_sectional_momentum_signals(prices)
    
    print("Calculating correlation regime filter...")
    regime_filter = calculate_correlation_regime_filter(weekly_returns, short_window, long_window, threshold)
    
    print("Creating adaptive strategy weights...")
    adaptive_weights = adaptive_cross_sectional_strategy(momentum_signals, regime_filter, quantile)
    
    print("Creating momentum-only strategy weights...")
    momentum_weights = select_top_quantile_equal_weight(momentum_signals, quantile)
    
    print("Creating contrarian-only strategy weights...")
    contrarian_weights = select_bottom_quantile_equal_weight(momentum_signals, quantile)
    
    print("Creating equal-weight benchmark...")
    benchmark_weights = calculate_benchmark_equal_weight(prices)
    
    print("Running backtests...")
    adaptive_returns, adaptive_cumulative = backtest_cross_sectional_strategy(prices, adaptive_weights)
    momentum_returns, momentum_cumulative = backtest_cross_sectional_strategy(prices, momentum_weights)
    contrarian_returns, contrarian_cumulative = backtest_cross_sectional_strategy(prices, contrarian_weights)
    benchmark_returns, benchmark_cumulative = backtest_cross_sectional_strategy(prices, benchmark_weights)
    
    print("Calculating performance metrics...")
    adaptive_metrics = calculate_performance_metrics(adaptive_returns)
    momentum_metrics = calculate_performance_metrics(momentum_returns)
    contrarian_metrics = calculate_performance_metrics(contrarian_returns)
    benchmark_metrics = calculate_performance_metrics(benchmark_returns)
    
    return {
        'prices': prices,
        'weekly_returns': weekly_returns,
        'momentum_signals': momentum_signals,
        'regime_filter': regime_filter,
        'adaptive_weights': adaptive_weights,
        'momentum_weights': momentum_weights,
        'contrarian_weights': contrarian_weights,
        'benchmark_weights': benchmark_weights,
        'adaptive_returns': adaptive_returns,
        'adaptive_cumulative': adaptive_cumulative,
        'momentum_returns': momentum_returns,
        'momentum_cumulative': momentum_cumulative,
        'contrarian_returns': contrarian_returns,
        'contrarian_cumulative': contrarian_cumulative,
        'benchmark_returns': benchmark_returns,
        'benchmark_cumulative': benchmark_cumulative,
        'adaptive_metrics': adaptive_metrics,
        'momentum_metrics': momentum_metrics,
        'contrarian_metrics': contrarian_metrics,
        'benchmark_metrics': benchmark_metrics
    }

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