"""Forward-return targets.

Timing convention: a signal computed at the close of day t can only be traded at
the next opportunity. We assume execution at the close of t+1 (one full day of
delay, which is conservative), so the return attributed to the signal of day t is

    fwd_ret_h(t) = adj_close[t+1+h] / adj_close[t+1] - 1

Using adj_close[t] as the entry price instead would mean trading at the same close
the signal was computed from: a subtle but common form of look-ahead.
"""

from __future__ import annotations

import pandas as pd

HORIZONS = (1, 5, 21)


def forward_returns(adj_close: pd.DataFrame, horizon: int, delay: int = 1) -> pd.DataFrame:
    entry = adj_close.shift(-delay)
    exit_ = adj_close.shift(-(delay + horizon))
    return exit_ / entry - 1.0
