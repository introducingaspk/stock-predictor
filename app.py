from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib
import pandas as pd
import requests

app = Flask(__name__)
CORS(app)

# ✅ Load model
model = joblib.load("model.pkl")


@app.route('/')
def home():
    return "Stock Prediction API Running ✅"


# ✅ Fetch stock data (Yahoo Finance API)
def get_stock_data(symbol):
    try:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?range=3mo&interval=1d"

        headers = {
            "User-Agent": "Mozilla/5.0"
        }

        response = requests.get(url, headers=headers, timeout=10)

        if response.status_code != 200:
            return None, None

        data = response.json()

        if "chart" not in data or data["chart"]["result"] is None:
            return None, None

        result = data["chart"]["result"][0]

        closes = result["indicators"]["quote"][0]["close"]
        timestamps = result["timestamp"]

        # ✅ Clean data
        clean_prices = []
        clean_dates = []

        for price, ts in zip(closes, timestamps):
            if price is not None:
                clean_prices.append(price)
                clean_dates.append(ts)

        return clean_prices, clean_dates

    except Exception as e:
        print("Fetch Error:", e)
        return None, None


# ✅ Prediction API
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "Invalid JSON"}), 400

        symbol = data.get("symbol", "").upper().strip()

        if not symbol:
            return jsonify({"error": "Stock symbol required"}), 400

        prices, _ = get_stock_data(symbol)

        if prices is None or len(prices) < 50:
            return jsonify({"error": "Invalid stock symbol"}), 400

        df = pd.DataFrame(prices, columns=["Close"])

        # ✅ Features
        df["MA10"] = df["Close"].rolling(10).mean()
        df["MA50"] = df["Close"].rolling(50).mean()

        df = df.dropna()

        if df.empty:
            return jsonify({"error": "Not enough data"}), 400

        latest = df.iloc[-1]

        features = np.array([[latest["MA10"], latest["MA50"], latest["Close"]]])
        prediction = model.predict(features)

        return jsonify({
            "symbol": symbol,
            "current_price": float(latest["Close"]),
            "predicted_price": float(prediction[0])
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ✅ NEW: History API for chart
@app.route("/history/<symbol>", methods=["GET"])
def history(symbol):
    try:
        symbol = symbol.upper().strip()

        prices, timestamps = get_stock_data(symbol)

        if prices is None or len(prices) == 0:
            return jsonify({"error": "Invalid stock symbol"}), 400

        return jsonify({
            "symbol": symbol,
            "prices": prices,
            "timestamps": timestamps
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)