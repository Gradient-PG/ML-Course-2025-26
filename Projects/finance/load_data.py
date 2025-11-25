# Try to predict opening price for the next day of any publicly traded company you'd like.
# Here is how you can load the data of nvidia from the past year:

import yfinance as yf

ticker_symbol = "NVDA"
ticker = yf.Ticker(ticker_symbol)
historical_data = ticker.history(period="1y")
historical_data