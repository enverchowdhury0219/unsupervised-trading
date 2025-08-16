import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# ----------------------
# Config
# ----------------------
TICKERS = ["AAPL","MSFT","AMZN","GOOGL","META","NVDA","TSLA","JPM","XOM","UNH"]
BENCH   = "SPY"
START   = "2016-01-01"
END     = "2025-08-01"
N_CLUSTERS = 3
TRAIN_SPLIT = 0.6      # first 60% of days = training, rest = test
RANDOM_STATE = 42

# ----------------------
# Data
# ----------------------
def fetch_prices(tickers, start, end):
    close_cols, vol_cols = [], []
    for t in tickers:
        df = yf.download(t, start=start, end=end, auto_adjust=True, progress=False)
        if df.empty:
            continue

        s_close = df["Close"].copy()
        s_close.name = t                       # <- set series name safely
        close_cols.append(s_close)

        s_vol = df["Volume"].copy()
        s_vol.name = t                         # <- set series name safely
        vol_cols.append(s_vol)

    if not close_cols:
        raise RuntimeError("No price data downloaded. Check tickers/dates or network.")

    close = pd.concat(close_cols, axis=1).dropna(how="all")
    volume = pd.concat(vol_cols, axis=1).reindex(close.index)
    return close, volume

# ----------------------
# Features
# ----------------------
def make_features(close, volume):
    rets = close.pct_change()
    feat = {
        "mom5":  close.pct_change(5),
        "mom20": close.pct_change(20),
        "vol20": rets.rolling(20).std(),
        "volz20": (volume - volume.rolling(20).mean()) / volume.rolling(20).std(),
    }
    pieces = []
    for name, df in feat.items():
        pieces.append(df.stack().rename(name))
    X = pd.concat(pieces, axis=1).dropna()
    X.index = X.index.set_names(["Date","Ticker"])

    fwd1d = close.pct_change().shift(-1).stack().rename("fwd1d")
    fwd1d.index = fwd1d.index.set_names(["Date","Ticker"])
    y = fwd1d.reindex(X.index).dropna()

    # Align X to y after shift
    X = X.loc[y.index]
    return X, y

# ----------------------
# Train / Test split (by date)
# ----------------------
def time_split_index(dates, train_split=0.6):
    uniq = pd.Index(sorted(dates.unique()))
    cut = int(len(uniq) * train_split)
    split_date = uniq[cut]
    return split_date

# ----------------------
# Label clusters by their average forward return (train only)
# ----------------------
def label_clusters_and_best(X_train, y_train, n_clusters=3, random_state=42):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_train.values)

    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    km.fit(Xs)

    clusters = pd.Series(km.predict(Xs), index=X_train.index, name="cluster")
    cluster_perf = y_train.groupby(clusters).mean().sort_values(ascending=False)
    best_cluster = int(cluster_perf.index[0])

    return km, scaler, best_cluster, cluster_perf

# ----------------------
# Build daily signals and backtest
# ----------------------
def build_signals_and_backtest(X_test, y_all, km, scaler, best_cluster):
    Xt = scaler.transform(X_test.values)
    preds = pd.Series(km.predict(Xt), index=X_test.index, name="cluster")
    sig = (preds == best_cluster).astype(int).rename("signal")

    # Wide signal matrix (rows=Date, cols=Ticker)
    S = sig.unstack("Ticker").fillna(0)

    # Next-day returns aligned to the same (Date, Ticker) index
    Y = y_all.reindex(X_test.index).unstack("Ticker")

    # Equal-weight daily return among selected tickers; if none selected -> 0
    selected_count = S.sum(axis=1)
    # Avoid division by zero: where count==0, portfolio return = 0
    port_ret = (S * Y).sum(axis=1)
    port_ret = np.where(selected_count.values > 0, port_ret / selected_count.values, 0.0)
    port_ret = pd.Series(port_ret, index=S.index, name="strategy")

    return port_ret

def perf_stats(returns, freq=252):
    # Avoid division by zero on empty std
    mu = returns.mean() * freq
    sigma = returns.std(ddof=1) * np.sqrt(freq)
    sharpe = mu / sigma if sigma > 0 else np.nan
    curve = (1 + returns).cumprod()
    peak = curve.cummax()
    mdd = ((curve / peak) - 1).min()
    cagr = curve.iloc[-1] ** (freq / len(returns)) - 1 if len(returns) > 0 else np.nan
    return {"CAGR": cagr, "Sharpe": sharpe, "MaxDD": mdd}

# ----------------------
# Main
# ----------------------
def main():
    close, volume = fetch_prices(TICKERS, START, END)
    bench_close, _ = fetch_prices([BENCH], START, END)
    if bench_close.empty or close.empty:
        raise RuntimeError("No data pulled. Check tickers or dates.")
    bench = bench_close[BENCH].pct_change().rename("SPY")

    X, y = make_features(close, volume)
    split_date = time_split_index(X.index.get_level_values("Date"), TRAIN_SPLIT)

    X_train = X.loc[(X.index.get_level_values("Date") <= split_date)]
    y_train = y.loc[X_train.index]

    X_test  = X.loc[(X.index.get_level_values("Date") > split_date)]
    y_test  = y.loc[X_test.index]

    km, scaler, best_cluster, cluster_perf = label_clusters_and_best(
        X_train, y_train, n_clusters=N_CLUSTERS, random_state=RANDOM_STATE
    )

    strat_ret = build_signals_and_backtest(X_test, y, km, scaler, best_cluster)

    # Align benchmark to test period dates
    bench_ret = bench.reindex(strat_ret.index).fillna(0.0)

    # Stats
    s_stats = perf_stats(strat_ret)
    b_stats = perf_stats(bench_ret)

    print("\n=== Cluster train performance (mean next-day return) ===")
    print(cluster_perf.sort_index())
    print(f"\nBest cluster (train): {best_cluster}")
    print("\n=== Test Stats ===")
    print("Strategy:", {k: round(v, 4) for k,v in s_stats.items()})
    print("SPY:",      {k: round(v, 4) for k,v in b_stats.items()})

    # Plot equity curves
    strat_curve = (1 + strat_ret).cumprod().rename("Strategy")
    spy_curve   = (1 + bench_ret).cumprod().rename("SPY")

    ax = strat_curve.plot(figsize=(10,5))
    spy_curve.plot(ax=ax)
    ax.set_title("Unsupervised K-Means Cluster Strategy vs SPY")
    ax.set_ylabel("Growth of $1")
    ax.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
