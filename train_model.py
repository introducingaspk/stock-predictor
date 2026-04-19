import pandas as pd
import yfinance as yf
from sklearn.linear_model import LinearRegression
import joblib

# ✅ Download real stock data (1 year)
df = yf.Ticker("AAPL").history(period="1y")

# Keep only needed column
df = df[["Close"]]

# ✅ Create features
df["MA10"] = df["Close"].rolling(10).mean()
df["MA50"] = df["Close"].rolling(50).mean()
df["Lag1"] = df["Close"].shift(1)

# ✅ Target = next day price
df["Target"] = df["Close"].shift(-1)

# Drop missing values
df = df.dropna()

# Features & label
X = df[["MA10", "MA50", "Lag1"]]
y = df["Target"]

# ✅ Train model
model = LinearRegression()
model.fit(X, y)

# ✅ Save model
joblib.dump(model, "model.pkl")

print("✅ Real model trained and saved")