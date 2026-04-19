import yfinance as yf
import matplotlib.pyplot as plt

# Step 1: download data
data = yf.download("AAPL", start="2020-01-01", end="2024-01-01")

# Step 2: plot
data['Close'].plot(figsize=(10,5))
plt.title("AAPL Stock Price")
plt.show()

# Step 3: save
data.to_excel("stock_data.xlsx", engine="openpyxl")

print("Done successfully!")