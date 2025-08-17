# Multi-Asset Momentum Strategy with Risk Parity

## Overview

This repository implements a sophisticated momentum trading strategy that combines Sharpe ratio-based asset selection with risk parity allocation across a diversified multi-asset universe. The strategy systematically identifies top-performing assets based on risk-adjusted returns and allocates capital using inverse volatility weighting to achieve balanced risk contributions.

## Strategy Description

### Core Components

1. **Multi-Asset Universe**: 74+ instruments across:
   - **Equities**: US market ETFs (SPY, QQQ, VTI, IWM, etc.)
   - **International Equities**: Developed and emerging market ETFs (EFA, EEM, VEA, etc.)
   - **Fixed Income**: Treasury and corporate bond ETFs (TLT, AGG, HYG, etc.)
   - **Currencies**: Major FX pairs (EURUSD, USDJPY, GBPUSD, etc.)
   - **Commodities**: Futures contracts (Gold, Oil, Copper, Agricultural products)
   - **Sectors**: Technology, Energy, Financial sector ETFs

2. **Momentum Signal**: 12-month rolling Sharpe ratio
   - Risk-adjusted momentum measurement
   - Accounts for both returns and volatility
   - Proper signal shifting to avoid look-ahead bias

3. **Asset Selection**: Top 25% quantile performers
   - Dynamic selection based on momentum ranking
   - Typically holds 15-20 assets simultaneously
   - Monthly rebalancing frequency

4. **Risk Parity Allocation**: Inverse volatility weighting
   - Equal risk contribution from each selected asset
   - Reduces concentration risk
   - Adapts to changing market volatilities

### Key Features

- ✅ **No Look-Ahead Bias**: Signals properly shifted by one period
- ✅ **Risk-Adjusted Selection**: Uses Sharpe ratios instead of raw returns
- ✅ **Diversified Universe**: Multiple asset classes and geographies
- ✅ **Risk Management**: Built-in volatility-based position sizing
- ✅ **Systematic Approach**: Fully quantitative and reproducible

## Performance Summary

**Period**: January 2000 - August 2025 (308 months)

| Metric | Strategy | SPY Benchmark |
|--------|----------|---------------|
| Total Return | 172.77% | 627.71% |
| Annualized Return | 3.99% | 8.07% |
| Volatility | 5.40% | 15.21% |
| Sharpe Ratio | 0.738 | 0.530 |
| Maximum Drawdown | -10.41% | -50.78% |
| Positive Months | 63.3% | - |

**Key Insights:**
- Lower but more consistent returns with significantly reduced volatility
- Superior risk-adjusted performance (higher Sharpe ratio)
- Dramatic reduction in maximum drawdown (-10.4% vs -50.8%)
- Strategy prioritizes capital preservation and steady growth

## Repository Structure

```
mom_allassets/
├── data/                          # Historical price data (Parquet format)
│   ├── SPY.parquet               # S&P 500 ETF
│   ├── EURUSD_X.parquet          # EUR/USD exchange rate
│   ├── GC_F.parquet              # Gold futures
│   └── ... (70+ other instruments)
├── download_data.py              # Data acquisition script
├── momentum_strategy.py          # Core strategy implementation
├── momentum_analysis.ipynb       # Comprehensive analysis notebook
└── README.md                     # This documentation
```

## Installation & Setup

### Prerequisites

```bash
# Python 3.8+ required
pip install pandas numpy matplotlib seaborn yfinance pyarrow
```

### Quick Start

1. **Download Data**:
```bash
python download_data.py --outdir data
```

2. **Run Strategy Analysis**:
```python
from momentum_strategy import run_full_strategy

# Execute complete strategy pipeline
results = run_full_strategy(
    data_dir='data',
    lookback_months=12,
    top_quantile=0.25
)

# Extract key results
portfolio_returns = results['portfolio_returns']
metrics = results['metrics']
```

3. **View Detailed Analysis**:
Open `momentum_analysis.ipynb` in Jupyter for comprehensive visualizations and performance analysis.

## Core Functionality

### Data Management (`download_data.py`)

- Downloads historical data from Yahoo Finance
- Supports custom date ranges and intervals
- Saves individual Parquet files for each instrument
- Includes retry logic and error handling
- Default coverage: 2000-01-01 to present

```bash
# Basic usage
python download_data.py --outdir data

# Custom options
python download_data.py --start 2010-01-01 --end 2023-12-31 --outdir custom_data
```

### Strategy Engine (`momentum_strategy.py`)

Key functions:

- `load_all_data()`: Synchronizes price data across all assets
- `calculate_momentum_signals()`: Computes 12-month Sharpe ratios
- `select_top_quantile()`: Identifies top performers for allocation
- `calculate_risk_parity_weights()`: Inverse volatility weighting
- `backtest_strategy()`: Complete portfolio simulation
- `calculate_performance_metrics()`: Comprehensive statistics

### Analysis Notebook (`momentum_analysis.ipynb`)

Comprehensive analysis including:

- Data availability timeline and coverage
- Strategy performance metrics and comparisons
- Equity curve visualization and drawdown analysis
- Monthly returns heatmap and seasonality
- Risk parity allocation effectiveness
- Risk contribution analysis
- Asset selection frequency and patterns

## Strategy Configuration

### Parameters

- **Lookback Period**: 12 months (configurable)
- **Selection Quantile**: Top 25% (configurable)
- **Rebalancing**: Monthly
- **Allocation Method**: Risk parity (inverse volatility)
- **Signal Processing**: 1-period lag to avoid look-ahead bias

### Customization Options

```python
# Modify strategy parameters
results = run_full_strategy(
    data_dir='data',
    lookback_months=6,     # Shorter momentum period
    top_quantile=0.20      # More selective (top 20%)
)
```

## Data Sources & Coverage

### Asset Classes Included

1. **US Equities** (13 instruments):
   - Broad market: SPY, VTI, QQQ, IVV, VOO
   - Size factors: IWM (small cap), VXF (extended market)
   - Large cap: SCHX

2. **International Equities** (7 instruments):
   - Developed markets: EFA, VEA, VGK, IEFA
   - Emerging markets: EEM, IEMG, VXUS

3. **Fixed Income** (13 instruments):
   - Treasury: TLT, IEF, SHY, GOVT, BIL, SCHR, SCHQ
   - Corporate: AGG, BND, LQD, VCIT, VCSH, HYG

4. **Currencies** (26 pairs):
   - Major pairs: EURUSD, USDJPY, GBPUSD, USDCHF, AUDUSD
   - Cross rates: EURGBP, EURJPY, GBPJPY, AUDNZD
   - Exotic pairs: USDMXN, USDZAR, USDHKD

5. **Commodities** (12 futures):
   - Precious metals: Gold (GC), Silver (SI)
   - Energy: Crude Oil (CL), Brent (BZ), Natural Gas (NG)
   - Industrial: Copper (HG)
   - Agriculture: Corn (ZC), Wheat (ZW), Soybeans (ZS), Coffee (KC), Cotton (CT), Sugar (SB)

6. **Sectors** (3 instruments):
   - XLE (Energy), XLF (Financial), XLK (Technology)

### Data Quality

- **Coverage**: 2000-2025 (25+ years)
- **Frequency**: Daily prices converted to monthly returns
- **Source**: Yahoo Finance via yfinance library
- **Format**: Parquet files for efficient storage and loading
- **Completeness**: Handles missing data and different inception dates

## Risk Management

### Built-in Risk Controls

1. **Volatility-Based Sizing**: Risk parity allocation reduces concentration
2. **Diversification**: Multi-asset universe spanning global markets
3. **Momentum Filtering**: Only invests in positively trending assets
4. **Regular Rebalancing**: Monthly updates to maintain target allocations
5. **Maximum Drawdown Control**: Historical max drawdown of 10.4%

### Risk Metrics Monitored

- Portfolio volatility and correlations
- Individual asset risk contributions
- Concentration measures (Herfindahl index)
- Drawdown duration and magnitude
- Rolling Sharpe ratios

## Limitations & Considerations

### Known Limitations

1. **Transaction Costs**: Not explicitly modeled in backtest
2. **Capacity**: Strategy may face constraints with large AUM
3. **Implementation**: Requires access to diverse asset classes
4. **Market Regimes**: Performance may vary across different market environments

### Important Notes

- Past performance does not guarantee future results
- Strategy assumes sufficient liquidity in all instruments
- Monthly rebalancing may not be optimal in all market conditions
- Risk parity allocation assumes volatility persistence

## Contributing

### Development Setup

1. Clone repository
2. Install dependencies: `pip install -r requirements.txt`
3. Download data: `python download_data.py`
4. Run tests: `python -m pytest tests/`

### Code Style

- Follow PEP 8 conventions
- Use type hints where applicable
- Add docstrings to all functions
- Include unit tests for new functionality

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This code is for educational and research purposes only. It does not constitute investment advice, and you should not rely on it as such. Trading and investing involve substantial risk of loss and are not suitable for all investors. Past performance is not indicative of future results.

## Contact

For questions, suggestions, or collaboration opportunities, please open an issue in this repository.

---

*Last updated: August 2025*