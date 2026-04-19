import yfinance as yf
import matplotlib.pyplot as plt

# Step 1: download data
data = yf.download("AAPL", start="2020-01-01", end="2024-01-01")

# Step 2: visualize
data['Close'].plot(figsize=(10,5))
plt.title("AAPL Stock Price")
plt.show()

# ---------------- ML PART ---------------- #

# Step 3: create features
data['MA10'] = data['Close'].rolling(10).mean()
data['MA50'] = data['Close'].rolling(50).mean()
data['Lag1'] = data['Close'].shift(1)

data = data.dropna()

# Step 4: prepare data
X = data[['MA10', 'MA50', 'Lag1']]
y = data['Close'].values.ravel()

# Step 5: split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# Step 6: train model
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)

# Step 7: predict
predictions = model.predict(X_test)

# Step 8: evaluate
from sklearn.metrics import mean_squared_error, r2_score

rmse = mean_squared_error(y_test, predictions) ** 0.5
r2 = r2_score(y_test, predictions)

print("RMSE:", rmse)
print("R2 Score:", r2)

# Step 9: plot results
plt.figure(figsize=(10,5))
plt.plot(y_test, label="Actual")
plt.plot(predictions, label="Predicted")
plt.legend()
plt.title("Actual vs Predicted")
plt.show()
import pickle

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved ✅")
from sklearn.ensemble import RandomForestRegressor

rf_model = RandomForestRegressor(n_estimators=100)
rf_model.fit(X_train, y_train)

rf_predictions = rf_model.predict(X_test)

rf_rmse = mean_squared_error(y_test, rf_predictions) ** 0.5
rf_r2 = r2_score(y_test, rf_predictions)

print("RF RMSE:", rf_rmse)
print("RF R2:", rf_r2)