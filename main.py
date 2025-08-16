import numpy as np
import random

from config import UNIVERSE, BENCH, START, END, TOP_N, TC_BPS, TITLE, SEED
from data import download_close, month_end_prices
from features import momentum_12_1
from backtest import run_backtest
from metrics import perf_stats
from plotting import plot_curve

if __name__ == "__main__":
    random.seed(SEED); np.random.seed(SEED)

    print("Downloading data...")
    px = download_close(UNIVERSE, START, END)
    bench_px = download_close([BENCH], START, END)[BENCH].to_frame()

    print("Computing features...")
    mom = momentum_12_1(px)

    print("Backtesting...")
    curve = run_backtest(
        daily_prices=px,
        bench_daily=bench_px,
        momentum=mom,
        bench_symbol=BENCH,
        top_n=TOP_N,
        tc_bps=TC_BPS,
    )

    print("\n=== Performance ===")
    s = perf_stats(curve)
    for k, v in s.items():
        if k in ("Start","End"):
            print(f"{k:>24}: {v}")
        else:
            print(f"{k:>24}: {v: .2%}")

    plot_curve(curve, title=TITLE)
