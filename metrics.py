import numpy as np

def perf_stats(curve):
    r = curve.pct_change().dropna()
    ann = lambda x: (1 + x.mean())**12 - 1
    vol = lambda x: x.std() * np.sqrt(12)

    def mdd(series):
        roll_max = series.cummax()
        dd = series / roll_max - 1
        return float(dd.min())

    cols = list(curve.columns)
    strat = cols[0]
    bench = cols[1]

    s_ret, s_vol = ann(r[strat]), vol(r[strat])
    b_ret, b_vol = ann(r[bench]), vol(r[bench])
    sharpe = (r[strat].mean()*12) / (r[strat].std()*np.sqrt(12))

    return {
        "Start": str(curve.index.min().date()),
        "End": str(curve.index.max().date()),
        "Strategy Ann Return": s_ret,
        "Strategy Ann Vol": s_vol,
        "Strategy Sharpe (naive)": sharpe,
        "Strategy Max Drawdown": mdd(curve[strat]),
        f"{bench} Ann Return": b_ret,
        f"{bench} Ann Vol": b_vol,
        f"{bench} Max Drawdown": mdd(curve[bench]),
    }
