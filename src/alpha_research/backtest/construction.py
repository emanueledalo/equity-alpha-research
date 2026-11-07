"""From a composite score to target portfolio weights.

Long-short (market-neutral) book
--------------------------------
1. Raw weights from ranks: w_i proportional to rank(score_i) - average rank. Ranks
   make the book robust to score outliers; using the whole cross-section (not just
   the extreme quintiles) diversifies stock-specific risk.
   Optionally divide by each stock's idiosyncratic volatility (risk-scaled weights).
2. Neutralise: remove from w its projection on the risk exposures A
   (a constant column, market beta, and optionally statistical factors):
        w <- w - A (A'A)^-1 A'w
   The constant column makes sum(w) = 0 (dollar neutral), the beta column makes
   sum(w * beta) = 0 (beta neutral), and so on for every column of A.
   Stage 3 showed why: raw low-vol / low-beta long-shorts carry a market beta of
   about -0.5, so in a bull market they lose money for reasons that have nothing to
   do with stock selection.
3. Scale to the target gross exposure (default 2 = 100% long + 100% short) and cap
   each position at `max_weight`. Capping breaks neutrality slightly, so we alternate
   clip -> neutralise -> rescale until the caps hold.

Long-only book
--------------
Equal weight in the top quintile of the score. Benchmarked against the equal-weighted
universe, it is the version relevant to a long-only asset manager.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ConstructionConfig:
    gross: float = 2.0          # 100% long + 100% short
    max_weight: float = 0.02    # at most 2% of capital in one stock
    risk_scaled: bool = False   # w ~ score / idiosyncratic vol instead of w ~ score
    iterations: int = 50        # clip / neutralise rounds (converges in a few)


def neutralise(w: np.ndarray, exposures: np.ndarray) -> np.ndarray:
    """Remove the components of w along the columns of `exposures` (least squares residual)."""
    coef, *_ = np.linalg.lstsq(exposures, w, rcond=None)
    return w - exposures @ coef


def long_short_weights(score: pd.Series, exposures: pd.DataFrame, cfg: ConstructionConfig = ConstructionConfig(),
                       idio: pd.Series | None = None) -> pd.Series:
    """Neutral, capped long-short weights for one rebalance date.

    score:     composite score per ticker (NaN = not tradable today)
    exposures: (tickers x factors) risk exposures the book must be neutral to,
               e.g. columns [const, beta, pc1, ...] from risk.exposures
    idio:      idiosyncratic volatility per ticker, used only if cfg.risk_scaled.
               Why divide by it: the optimal weight under a factor model is
               alpha_i / sigma_i^2 and, following Grinold, alpha_i = IC * sigma_i * z_i,
               so w_i ~ z_i / sigma_i: the same conviction gets a smaller position in
               a more volatile stock.
    """
    s = score.dropna()
    if len(s) < 20:
        return pd.Series(dtype=float)

    ranks = s.rank().to_numpy()
    w = (ranks - ranks.mean()) / (len(s) / 2)          # centred ranks, roughly in [-1, 1]
    if cfg.risk_scaled and idio is not None:
        vol = idio.reindex(s.index)
        vol = vol.fillna(vol.median()).clip(vol.quantile(0.05), vol.quantile(0.95))
        w = w / vol.to_numpy()

    A = exposures.reindex(s.index).fillna(0.0).to_numpy()

    for _ in range(cfg.iterations):
        w = neutralise(w, A)
        w = w * cfg.gross / np.abs(w).sum()
        if np.abs(w).max() <= cfg.max_weight * 1.001:  # caps hold (0.1% tolerance): done
            break
        w = np.clip(w, -cfg.max_weight, cfg.max_weight)
    return pd.Series(w, index=s.index)


def long_only_weights(score: pd.Series, top: float = 0.2) -> pd.Series:
    """Equal weight in the top `top` share of stocks by score."""
    s = score.dropna()
    chosen = s[s.rank(pct=True) > 1 - top]
    return pd.Series(1.0 / len(chosen), index=chosen.index) if len(chosen) else pd.Series(dtype=float)


def benchmark_weights(score: pd.Series) -> pd.Series:
    """Equal weight on every stock with a score: the long-only book's benchmark."""
    s = score.dropna()
    return pd.Series(1.0 / len(s), index=s.index) if len(s) else pd.Series(dtype=float)


def build_targets(scores: pd.DataFrame, func) -> pd.DataFrame:
    """Apply func(date, score_row) on every rebalance date -> (dates x tickers) target weights."""
    rows = {date: func(date, scores.loc[date]) for date in scores.index}
    return pd.DataFrame(rows).T.reindex(columns=scores.columns).fillna(0.0)
