import pandas as pd

def pick_top_by_momentum(momentum_at_t: pd.Series, top_n: int) -> list:
    s = momentum_at_t.dropna().sort_values(ascending=False)
    return s.index[:top_n].tolist()

def turnover_cost(prev_hold: set, new_hold: set, top_n: int, tc_bps: float) -> float:
    if not prev_hold:
        return 0.0
    overlap = len(prev_hold & new_hold)
    turnover_frac = 1 - overlap / float(top_n)
    return (tc_bps / 10000.0) * turnover_frac
