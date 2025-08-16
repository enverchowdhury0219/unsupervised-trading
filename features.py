import numpy as np
import pandas as pd

def momentum_12_1(daily_prices: pd.DataFrame) -> pd.DataFrame:
    """
    12-1 momentum: product of months t-12..t-2 (exclude last month).
    Implemented on monthly returns, then aligned to month-ends.
    """
    mpx = daily_prices.resample("M").last()
    mret = mpx.pct_change()
    # rolling 12 months, drop the most recent month in the window
    mom = (1 + mret).rolling(12).apply(lambda x: np.prod(x[:-1]) - 1, raw=True)
    return mom
