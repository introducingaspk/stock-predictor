from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib   # ✅ ADD THIS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# ✅ LOAD MODEL
model = joblib.load("model.pkl")

@app.route('/')
def home():
    return "Stock Prediction API Running ✅"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    
    MA10 = data["MA10"]
    MA50 = data["MA50"]
    Lag1 = data["Lag1"]

    features = np.array([[MA10, MA50, Lag1]])
    prediction = model.predict(features)

    return jsonify({
        "predicted_price": float(prediction[0])
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)