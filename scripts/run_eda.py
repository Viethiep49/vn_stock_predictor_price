"""Quick EDA script for MBB data."""
import pandas as pd
import numpy as np
import sys
sys.path.insert(0, '.')

# Load data
print("Loading MBB data...")
df = pd.read_csv('data/raw/mbb_daily.csv', parse_dates=['time'])
df = df.set_index('time').sort_index()

vnindex = pd.read_csv('data/raw/vnindex_daily.csv', parse_dates=['time']).set_index('time')
vn30 = pd.read_csv('data/raw/vn30_daily.csv', parse_dates=['time']).set_index('time')

print("=" * 60)
print("MBB STOCK DATA SUMMARY")
print("=" * 60)
print(f"Period: {df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}")
print(f"Total trading days: {len(df)}")

print(f"\nPrice Range:")
print(f"  Min: {df['close'].min():.2f}")
print(f"  Max: {df['close'].max():.2f}")
print(f"  Current: {df['close'].iloc[-1]:.2f}")

# Calculate returns
df['returns'] = df['close'].pct_change()
print(f"\nDaily Returns:")
print(f"  Mean: {df['returns'].mean()*100:.4f}%")
print(f"  Std: {df['returns'].std()*100:.4f}%")
print(f"  Annualized Return: {df['returns'].mean()*252*100:.2f}%")
print(f"  Annualized Volatility: {df['returns'].std()*np.sqrt(252)*100:.2f}%")

# Correlation
vnindex['returns'] = vnindex['close'].pct_change()
vn30['returns'] = vn30['close'].pct_change()

merged = pd.DataFrame({
    'MBB': df['returns'],
    'VNINDEX': vnindex['returns'],
    'VN30': vn30['returns']
}).dropna()

print(f"\nCorrelation with VNINDEX: {merged['MBB'].corr(merged['VNINDEX']):.4f}")
print(f"Correlation with VN30: {merged['MBB'].corr(merged['VN30']):.4f}")

# Missing values
print(f"\nMissing values: {df.isnull().sum().sum()}")
print(f"Duplicate dates: {df.index.duplicated().sum()}")

# Save clean data
df_clean = df[['open', 'high', 'low', 'close', 'volume']].copy()
df_clean.to_csv('data/processed/mbb_clean.csv')
print("\n" + "=" * 60)
print("Clean data saved to data/processed/mbb_clean.csv")
print("=" * 60)
