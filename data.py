import pandas as pd
import yfinance as yf

def download_close(tickers, start, end):
    px = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)["Close"]
    if isinstance(px, pd.Series):
        px = px.to_frame()
    return px.dropna(how="all")

def month_end_prices(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.resample("M").last()
