# MBB Stock Predictor - Project Plan

> Deep Learning dự đoán giá cổ phiếu MBB (MB Bank) và đưa ra kế hoạch đầu tư

---

## 1. Tổng quan dự án

### Problem Statement
Xây dựng hệ thống sử dụng Deep Learning để dự đoán giá cổ phiếu MBB (mã cổ phiếu của Ngân hàng TMCP Quân đội) trên sàn HOSE, từ đó đưa ra kế hoạch đầu tư hợp lý.

### Goals
| Goal | Mô tả |
|------|-------|
| **Primary** | Dự đoán giá cổ phiếu MBB ngày tiếp theo (t+1) |
| **Secondary** | Đưa ra tín hiệu mua/bán với threshold ±5% |
| **Tertiary** | Xây dựng dashboard Streamlit để theo dõi và phân tích |

### Success Metrics
- **Model Accuracy**: MAPE < 10%, Direction Accuracy > 55%
- **Strategy Performance**: Sharpe Ratio > 1, Max Drawdown < 20%
- **System Reliability**: Retrain hàng tháng, deploy trên Streamlit Cloud

### Constraints
- **Technical**: Python, TensorFlow/Keras, LSTM + Attention
- **Data**: Sử dụng Vnstock (miễn phí), chỉ MBB
- **Skill Level**: Người mới bắt đầu với Deep Learning

---

## 2. Nguồn dữ liệu

### Primary Data Source
**Vnstock Python** - Thư viện Python miễn phí để lấy dữ liệu chứng khoán Việt Nam
- Website: https://vnstocks.com
- Cài đặt: `pip install vnstock`

### Các nguồn data khác (tham khảo)
| Nguồn | Chi phí | Link |
|-------|---------|------|
| SSI FastConnect | Cần tài khoản | https://guide.ssi.com.vn |
| FiinGroup DataFeed | Trả phí | https://datafeed.fiingroup.vn |
| WiFeed | Trả phí | https://wifeed.vn |

### Data Requirements

#### 1. Dữ liệu giá (OHLCV)
```
- Open: Giá mở cửa
- High: Giá cao nhất
- Low: Giá thấp nhất
- Close: Giá đóng cửa
- Volume: Khối lượng giao dịch
- Timeframe: Daily + Weekly
- Lịch sử: 2015-2024 (10 năm, ~2500 trading days)
- Phạm vi: Chỉ MBB (model chuyên biệt)
```

#### 2. Technical Indicators (Base - 15 features)
```
Moving Averages:
- SMA (5, 20, 50)
- EMA (12, 26)

Momentum:
- RSI (14)
- MACD, MACD_Signal

Volatility:
- Bollinger Bands upper/lower (20, 2)
- ATR (14)

Changes:
- Price_Change (%)
- Volume_Change (%)
```

#### 3. Technical Indicators bổ sung (Group A - 5 features)
```
- ADX (14) - Trend strength
- OBV - On Balance Volume
- Stochastic %K, %D
```

#### 4. Market Context (Group B - 4 features)
```
- VN-Index daily return
- VN30 daily return
- USD/VND exchange rate change
```

#### 5. Trading Flow (Group C - 2 features)
```
- Foreign net buy/sell volume
- Proprietary trading net
```

**Total Features: ~28 features**
**Lookback Period: 60 ngày**

---

## 3. Technical Stack

```yaml
Core:
  Python: ">=3.10"

Data Collection:
  vnstock: ">=2.0.0"
  pandas: ">=2.0.0"
  numpy: ">=1.24.0"

Technical Analysis:
  pandas-ta: ">=0.3.14"

Machine Learning:
  scikit-learn: ">=1.3.0"
  tensorflow: ">=2.15.0"

Visualization:
  matplotlib: ">=3.7.0"
  seaborn: ">=0.12.0"
  plotly: ">=5.18.0"

Dashboard:
  streamlit: ">=1.29.0"

Utilities:
  python-dotenv: ">=1.0.0"
  joblib: ">=1.3.0"
```

---

## 4. Kiến trúc hệ thống

### High-Level Architecture
```
┌─────────────────────────────────────────────────────────────────┐
│                        MBB STOCK PREDICTOR                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐     │
│  │  DATA LAYER  │────▶│ PROCESSING   │────▶│  ML/DL LAYER │     │
│  │              │     │    LAYER     │     │              │     │
│  │ - Vnstock    │     │ - Cleaning   │     │ - LSTM Model │     │
│  │ - OHLCV      │     │ - Features   │     │ - Prediction │     │
│  │ - Indicators │     │ - Scaling    │     │ - Evaluation │     │
│  └──────────────┘     └──────────────┘     └──────────────┘     │
│                                                  │               │
│                                                  ▼               │
│                              ┌──────────────────────────────┐   │
│                              │     INVESTMENT STRATEGY      │   │
│                              │                              │   │
│                              │  - Buy/Sell Signals          │   │
│                              │  - Risk Assessment           │   │
│                              │  - Portfolio Suggestion      │   │
│                              └──────────────────────────────┘   │
│                                                  │               │
│                                                  ▼               │
│                              ┌──────────────────────────────┐   │
│                              │       OUTPUT / UI            │   │
│                              │                              │   │
│                              │  - Dashboard (Streamlit)     │   │
│                              │  - Reports                   │   │
│                              │  - Alerts                    │   │
│                              └──────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Project Structure
```
mbb-stock-predictor/
├── data/
│   ├── raw/                    # Dữ liệu gốc từ API
│   ├── processed/              # Dữ liệu đã xử lý
│   └── models/                 # Saved models
│
├── src/
│   ├── data/
│   │   ├── collector.py        # Thu thập dữ liệu từ Vnstock
│   │   └── preprocessor.py     # Xử lý và chuẩn bị data
│   │
│   ├── features/
│   │   ├── technical.py        # Tính chỉ số kỹ thuật
│   │   └── fundamental.py      # Features tài chính
│   │
│   ├── models/
│   │   ├── lstm_model.py       # LSTM architecture
│   │   ├── trainer.py          # Training logic
│   │   └── predictor.py        # Inference
│   │
│   ├── strategy/
│   │   ├── signals.py          # Buy/Sell signals
│   │   └── risk.py             # Risk management
│   │
│   └── utils/
│       ├── config.py           # Configurations
│       └── metrics.py          # Evaluation metrics
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_backtesting.ipynb
│
├── app/
│   └── dashboard.py            # Streamlit dashboard
│
├── tests/
├── requirements.txt
└── README.md
```

---

## 5. LSTM + Attention Model Architecture

### Model Design
```
┌─────────────────────────────────────────┐
│ Input Layer                             │
│ Shape: (sequence_length, num_features)  │
│ Example: (60, 28) - 60 ngày, 28 features│
└────────────────────┬────────────────────┘
                     ▼
┌─────────────────────────────────────────┐
│ LSTM Layer 1                            │
│ Units: 128, return_sequences=True       │
└────────────────────┬────────────────────┘
                     ▼
┌─────────────────────────────────────────┐
│ Self-Attention Layer                    │
│ Focus on important time steps           │
└────────────────────┬────────────────────┘
                     ▼
┌─────────────────────────────────────────┐
│ LSTM Layer 2                            │
│ Units: 64, return_sequences=False       │
│ + Dropout(0.2)                          │
└────────────────────┬────────────────────┘
                     ▼
┌─────────────────────────────────────────┐
│ Dense Layer                             │
│ Units: 32, activation='relu'            │
│ + Dropout(0.2)                          │
└────────────────────┬────────────────────┘
                     ▼
┌─────────────────────────────────────────┐
│ Output Layer                            │
│ Units: 1 (predicted price t+1)          │
└─────────────────────────────────────────┘
```

### Input Features (28 features)
| Nhóm | Features | Số lượng |
|------|----------|----------|
| OHLCV | Close, Volume | 2 |
| Moving Averages | SMA_5, SMA_20, SMA_50, EMA_12, EMA_26 | 5 |
| Momentum | RSI_14, MACD, MACD_Signal | 3 |
| Volatility | BB_upper, BB_lower, ATR_14 | 3 |
| Changes | Price_Change, Volume_Change | 2 |
| Technical+ | ADX, OBV, Stochastic_K, Stochastic_D | 4 |
| Market | VNIndex_Return, VN30_Return, USDVND_Change | 3 |
| Flow | Foreign_Net, Proprietary_Net | 2 |
| **Total** | | **28** |

### Hyperparameters
```python
HYPERPARAMETERS = {
    # Data
    'sequence_length': 60,      # Số ngày lookback
    'num_features': 28,

    # Model Architecture
    'lstm_units_1': 128,
    'lstm_units_2': 64,
    'attention_heads': 4,       # Multi-head attention
    'dropout_rate': 0.2,
    'dense_units': 32,

    # Training
    'learning_rate': 0.001,
    'batch_size': 32,
    'epochs': 100,
    'optimizer': 'adam',
    'loss': 'huber',            # Robust với outliers
    'early_stopping_patience': 15,
    'reduce_lr_patience': 5,

    # Validation
    'train_ratio': 0.70,        # 2015-2021
    'val_ratio': 0.15,          # 2022
    'test_ratio': 0.15,         # 2023-2024
}
```

### Retrain Strategy
- **Frequency**: Hàng tháng
- **Method**: Walk-forward validation
- **Data**: Rolling window với data mới nhất

---

## 6. Investment Strategy

### Signal Generation
```python
# Chiến lược conservative với threshold ±5%
def generate_signals(predicted_price, current_price):
    """
    Generate trading signals based on predicted price change.

    Threshold: ±5% (conservative)
    """
    price_change = (predicted_price - current_price) / current_price

    if price_change >= 0.05:
        return "BUY"
    elif price_change <= -0.05:
        return "SELL"
    else:
        return "HOLD"
```

### Risk Management
```python
RISK_CONFIG = {
    # Position Sizing
    'position_size': 0.05,      # Fixed 5% portfolio per trade

    # Stop Loss
    'stop_loss': 0.05,          # Fixed -5% từ entry price

    # Take Profit
    'take_profit': None,        # Based on predicted price/signal change

    # Trade Rules
    'max_holding_days': 20,     # Tối đa giữ 20 ngày nếu không có signal
    'min_days_between_trades': 1,
}
```

### Trading Rules
1. **Entry**: Khi signal = BUY và chưa có position
2. **Exit**: Khi signal = SELL hoặc hit stop loss
3. **Hold**: Giữ position nếu signal = HOLD
4. **Re-evaluate**: Hàng ngày sau khi có prediction mới

---

## 7. Evaluation Metrics

### Model Performance
| Metric | Target | Mô tả |
|--------|--------|-------|
| MAE | < 500 VND | Mean Absolute Error |
| RMSE | < 800 VND | Root Mean Square Error |
| MAPE | < 10% | Mean Absolute Percentage Error |
| Direction Accuracy | > 55% | Dự đoán đúng xu hướng |

### Strategy Performance
| Metric | Target | Mô tả |
|--------|--------|-------|
| Total Return | > Buy & Hold | Lợi nhuận tổng |
| Sharpe Ratio | > 1.0 | Risk-adjusted return |
| Max Drawdown | < 20% | Mức giảm tối đa |
| Win Rate | > 50% | Tỷ lệ giao dịch có lãi |
| Profit Factor | > 1.5 | Gross profit / Gross loss |

---

## 8. Development Roadmap

### Milestone 1: Foundation
- [ ] Setup project structure
- [ ] Install dependencies
- [ ] Configure environment
- [ ] Thu thập dữ liệu MBB (5-10 năm)
- [ ] EDA notebook
- **Deliverable**: Data exploration notebook

### Milestone 2: Feature Engineering
- [ ] Tính technical indicators
- [ ] Feature selection
- [ ] Data preprocessing pipeline
- [ ] Train/validation/test split
- **Deliverable**: Feature engineering pipeline

### Milestone 3: Model Development
- [ ] Implement LSTM model
- [ ] Training pipeline
- [ ] Hyperparameter tuning
- [ ] Model evaluation
- [ ] Cross-validation
- **Deliverable**: Trained model với baseline metrics

### Milestone 4: Strategy & Backtesting
- [ ] Implement trading signals
- [ ] Backtest framework
- [ ] Risk management rules
- [ ] Performance analysis
- **Deliverable**: Backtest report

### Milestone 5: Deployment
- [ ] Streamlit dashboard
- [ ] Auto data update
- [ ] Alert system
- [ ] Documentation
- **Deliverable**: Production-ready application

---

## 9. Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| **Overfitting** | High | Cross-validation, regularization, early stopping |
| **Data leakage** | High | Strict train/test split, walk-forward validation |
| **Market regime change** | Medium | Regular model retraining, ensemble methods |
| **API changes** | Medium | Abstract data layer, backup sources |
| **Black swan events** | High | Risk management, position limits |

---

## 10. Important Notes

### Disclaimers
1. **Dự đoán giá cổ phiếu có độ không chắc chắn cao** - không có model nào có thể dự đoán chính xác 100%
2. **Kết quả backtest không đảm bảo lợi nhuận tương lai**
3. **Nên sử dụng như công cụ hỗ trợ** - không nên dựa hoàn toàn vào model để ra quyết định đầu tư
4. **Luôn quản lý rủi ro** - không đầu tư quá số tiền có thể chấp nhận mất

### Best Practices
- Bắt đầu với paper trading trước khi dùng tiền thật
- Theo dõi và đánh giá model liên tục
- Kết hợp phân tích cơ bản với kỹ thuật
- Cập nhật model định kỳ với dữ liệu mới

---

## 11. Resources

### Documentation
- [Vnstock Documentation](https://vnstocks.com)
- [TensorFlow LSTM Tutorial](https://www.tensorflow.org/tutorials/structured_data/time_series)
- [Pandas-TA Documentation](https://github.com/twopirllc/pandas-ta)

### Learning Resources
- [Deep Learning for Time Series](https://machinelearningmastery.com/deep-learning-for-time-series-forecasting/)
- [LSTM for Stock Prediction](https://www.kaggle.com/code/raoulma/ny-stock-price-prediction-rnn-lstm-gru)

### Vietnam Stock Market
- [HOSE](https://www.hsx.vn)
- [MBB Stock Info](https://simplize.vn/co-phieu/MBB)

---

*Plan created: 2024-12-23*
*Last updated: 2024-12-23*
