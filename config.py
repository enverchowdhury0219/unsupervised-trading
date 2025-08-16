UNIVERSE = [
    "AAPL","MSFT","AMZN","GOOGL","META","NVDA","AVGO","PEP","COST","LLY",
    "JPM","XOM","UNH","V","MA","HD","PG","ABBV","WMT","KO"
]
BENCH = "SPY"
START = "2015-01-01"
END = None            # to today
TOP_N = 5
TC_BPS = 5            # simple round-trip turnover cost in bps
SEED = 42
TITLE = f"12-1 Momentum (Top {TOP_N}) vs {BENCH} — Monthly, EW"
