import pandas as pd
from .strategy import pick_top_by_momentum, turnover_cost

def run_backtest(daily_prices: pd.DataFrame,
                 bench_daily: pd.DataFrame,
                 momentum: pd.DataFrame,
                 bench_symbol: str,
                 top_n: int,
                 tc_bps: float) -> pd.DataFrame:
    mpx = daily_prices.resample("M").last().dropna(how="all", axis=1)
    bench_mpx = bench_daily.resample("M").last().dropna()
    mret = mpx.pct_change()
    bench_ret = bench_mpx.pct_change()

    # need first 12 months of history
    rebals = mpx.index[12:]

    eq, beq = 1.0, 1.0
    curve, bcurve, dates = [], [], []
    prev_hold: set = set()

    for i in range(len(rebals) - 1):
        t = rebals[i]
        t_next = rebals[i + 1]

        picks = pick_top_by_momentum(momentum.loc[t], top_n=top_n)
        cost = turnover_cost(prev_hold, set(picks), top_n, tc_bps)
        prev_hold = set(picks)

        # realize next month’s return
        next_ret = mret.loc[t_next, picks].mean()
        if pd.isna(next_ret):
            continue

        eq *= (1 - cost) * (1 + next_ret)

        b = bench_ret.loc[t_next, bench_symbol]
        beq *= (1 + b)

        dates.append(t_next)
        curve.append(eq)
        bcurve.append(beq)

    out = pd.DataFrame({"Strategy": curve, bench_symbol: bcurve}, index=pd.to_datetime(dates))
    return out
