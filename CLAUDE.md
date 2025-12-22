# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MBB Stock Predictor - Deep learning system to predict MBB (MB Bank) stock prices on HOSE exchange using LSTM + Attention and generate investment signals.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run dashboard
streamlit run app/dashboard.py

# Run tests
pytest tests/ -v

# Test modules
python -m src.data.collector
python -m src.features.technical
python -m src.models.lstm_attention
```

## Project Structure

```
src/
├── data/
│   ├── collector.py      # Vnstock API - fetch MBB data (2015-2024)
│   └── preprocessor.py   # Scaling, sequences, train/val/test split
├── features/
│   └── technical.py      # 28 technical indicators (pandas-ta)
├── models/
│   ├── lstm_attention.py # LSTM + Self-Attention architecture
│   ├── trainer.py        # Training with EarlyStopping, ReduceLR
│   └── predictor.py      # Inference and signal generation
├── strategy/
│   ├── signals.py        # BUY/SELL/HOLD signal generation
│   └── risk.py           # Position sizing, stop loss
└── utils/
    ├── config.py         # All hyperparameters centralized
    └── metrics.py        # MAE, RMSE, MAPE, Sharpe, Drawdown
```

## Architecture

```
Vnstock API → collector.py → preprocessor.py → technical.py
                                                    ↓
                                            lstm_attention.py
                                                    ↓
                                             predictor.py → signals.py → dashboard.py
```

## Key Decisions (from brainstorming)

| Category | Decision |
|----------|----------|
| Data | MBB only, 2015-2024, Daily + Weekly |
| Features | 28 features (technical + market + flow) |
| Model | LSTM(128) + Attention + LSTM(64) |
| Loss | Huber Loss |
| Lookback | 60 days |
| Signal threshold | ±5% |
| Stop loss | -5% fixed |
| Position size | 5% portfolio |
| Retrain | Monthly |
| Deploy | Streamlit Cloud |

## Data Source

```python
from vnstock import Vnstock

stock = Vnstock().stock(symbol="MBB", source="VCI")
df = stock.quote.history(start="2015-01-01", end="2024-12-23", interval="1D")
```

## Target Metrics

- MAPE < 10%
- Direction Accuracy > 55%
- Sharpe Ratio > 1.0
- Max Drawdown < 20%

## Development Status

- [x] Project structure created
- [x] All skeleton files created
- [ ] Data collection & EDA
- [ ] Feature engineering pipeline
- [ ] Model training
- [ ] Backtesting
- [ ] Dashboard integration
