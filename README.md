# MBB Stock Predictor

Dự đoán giá cổ phiếu MBB (MB Bank) sử dụng Deep Learning.

## Tính năng

- Dự đoán giá ngày tiếp theo với LSTM + Attention
- Tín hiệu giao dịch BUY/SELL/HOLD
- Dashboard theo dõi realtime

## Cài đặt

```bash
pip install -r requirements.txt
```

## Sử dụng

```bash
# Chạy dashboard
streamlit run app/dashboard.py

# Chạy tests
pytest tests/ -v
```

## Tech Stack

- Python 3.10+
- TensorFlow / Keras
- Vnstock (data)
- Streamlit (dashboard)

## License

MIT
