"""How much weight each signal gets in the composite.

All methods see only the history of signal returns f (holding periods x signals)
that is *known* at the rebalance date; `walk_forward` enforces this.

Methods
-------
equal         1/K for every signal. The hard-to-beat baseline: no estimation error.
ic_weighted   Proportional to each signal's historical mean payoff, floored at 0
              (a signal is never flipped against its literature sign on in-sample evidence).
bayes         Hierarchical (empirical-Bayes) shrinkage of the mean payoffs, then
              mean-variance weights with a Ledoit-Wolf covariance. See `bayes_posterior`.
ridge         Pooled ridge regression of stock returns on all signals (ML baseline);
              computed in combination/composite.py because it needs stock-level data.

Weights are normalised to sum(|w|) = 1; only relative weights matter because the
composite is re-standardised.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def normalise(w: pd.Series) -> pd.Series:
    total = w.abs().sum()
    return w / total if total > 0 else w * 0.0


def equal_weights(f: pd.DataFrame) -> pd.Series:
    return normalise(pd.Series(1.0, index=f.columns))


def ic_weights(f: pd.DataFrame) -> pd.Series:
    return normalise(f.mean().clip(lower=0))


PRIOR_IC = 0.02      # literature-informed prior for the *average* signal's rank IC (monthly)
PRIOR_IC_SD = 0.02   # wide: the prior allows anything from about -0.02 to 0.06


def bayes_posterior(f: pd.DataFrame, scale: float, prior_ic: float = PRIOR_IC,
                    prior_ic_sd: float = PRIOR_IC_SD) -> pd.DataFrame:
    """Hierarchical normal model for the mean payoff mu_k of each signal.

        f_k,m ~ N(mu_k, sigma_k^2)        observed monthly payoffs of signal k
        mu_k  ~ N(mu_0, tau^2)            premia of different signals share a common level
        mu_0  ~ N(m_lit, v_lit)           ... whose prior comes from the literature

    Level 3 (the prior on mu_0) is literature-informed and deliberately sceptical:
    published anomalies lose roughly half their strength after publication
    (McLean & Pontiff, 2016) and are weaker in large caps, so the average signal is
    expected to have a rank IC of ~0.02 rather than the 0.05+ often reported.
    A payoff f is a regression slope on a unit-variance signal, so f = IC * sigma_cs,
    where sigma_cs (`scale`) is the cross-sectional volatility of monthly stock returns.

    Estimation (closed form, refitted every month):
      * m_k = sample mean of f_k, s_k^2 = its sampling variance (periods do not overlap).
      * tau^2 by method of moments: max(Var(m_k) - mean(s_k^2), 0): how much the true
        premia really differ once sampling noise is removed.
      * mu_0 posterior: combine the data estimate of the common level with the
        literature prior. The data estimate is the plain average of the m_k (each signal
        is one draw from the population of published anomalies) and its variance
        1' Sigma 1 / (K^2 n) + tau^2 / K uses the full covariance Sigma of the payoffs:
        12 signals that are up to 94% correlated are far less than 12 independent pieces
        of evidence, and treating them as independent would make the data look
        overconfident and swamp the prior.
      * mu_k posterior: post_var = 1 / (1/s_k^2 + 1/tau^2),
                        post_mean = post_var * (m_k / s_k^2 + mu_0 / tau^2).

    A signal with a noisy record is pulled towards the common level; a long, stable
    record keeps more of its own estimate. If tau^2 = 0 (the data cannot tell the
    signals apart) every signal gets mu_0. This is the partial pooling a hierarchical
    Stan model would produce, in closed form so it can be refitted every month.
    """
    n = f.notna().sum()
    m = f.mean()
    s2 = f.var(ddof=1) / n
    tau2 = max(m.var(ddof=1) - s2.mean(), 0.0)

    m_lit, v_lit = prior_ic * scale, (prior_ic_sd * scale) ** 2
    k = f.shape[1]
    cov_of_means = np.cov(f.dropna().to_numpy(), rowvar=False) / n.min()
    mu0_hat = m.mean()
    mu0_hat_var = cov_of_means.sum() / k ** 2 + tau2 / k
    mu0_var = 1 / (1 / mu0_hat_var + 1 / v_lit)
    mu0 = mu0_var * (mu0_hat / mu0_hat_var + m_lit / v_lit)

    if tau2 == 0:
        post_mean = pd.Series(mu0, index=f.columns)
        post_var = pd.Series(mu0_var, index=f.columns)
        shrink = pd.Series(1.0, index=f.columns)
    else:
        post_var = 1 / (1 / s2 + 1 / tau2)
        post_mean = post_var * (m / s2 + mu0 / tau2)
        shrink = post_var / tau2   # share of the estimate coming from the common level
    return pd.DataFrame({"sample_mean": m, "se": np.sqrt(s2), "post_mean": post_mean,
                         "post_sd": np.sqrt(post_var), "shrinkage": shrink,
                         "mu0": mu0, "tau": np.sqrt(tau2), "sample_ic": m / scale,
                         "post_ic": post_mean / scale})


def nonneg_mean_variance(cov: np.ndarray, mu: np.ndarray, iters: int = 5000) -> np.ndarray:
    """argmax_w  mu'w - 1/2 w' cov w  subject to w >= 0 (projected gradient descent).

    Non-negativity keeps every signal on its literature side: the model may decide
    a signal is worth nothing, but never bets against it on in-sample evidence.
    If no signal has a positive posterior premium, the answer is w = 0: do not trade.
    """
    w = np.zeros(len(mu))
    step = 1.0 / np.linalg.eigvalsh(cov).max()
    for _ in range(iters):
        w_new = np.maximum(w - step * (cov @ w - mu), 0.0)
        if np.abs(w_new - w).max() < 1e-12:
            break
        w = w_new
    return w


def ledoit_wolf(x: np.ndarray) -> tuple[np.ndarray, float]:
    """Ledoit & Wolf (2004) shrinkage of the sample covariance towards a scaled identity.

    With 12 strongly correlated signals and ~100 monthly observations the sample
    covariance is noisy and its inverse amplifies the noise; shrinking towards
    mu * I keeps the inverse well-conditioned. Returns (covariance, shrinkage in [0, 1]).
    """
    x = x - x.mean(axis=0)
    n, p = x.shape
    S = x.T @ x / n
    mu = np.trace(S) / p
    target = mu * np.eye(p)
    d2 = np.sum((S - target) ** 2)
    b2_bar = sum(np.sum((np.outer(row, row) - S) ** 2) for row in x) / n ** 2
    b2 = min(b2_bar, d2)
    delta = b2 / d2 if d2 > 0 else 1.0
    return delta * target + (1 - delta) * S, float(delta)


def bayes_weights(f: pd.DataFrame, scale: float, prior_ic: float = PRIOR_IC,
                  prior_ic_sd: float = PRIOR_IC_SD) -> pd.Series:
    """Non-negative mean-variance weights with shrunk means and shrunk covariance.

    The covariance term accounts for redundancy: two signals that are 94% correlated
    (low_vol, idio_vol) share one bet instead of getting double weight.
    """
    f = f.dropna()
    post = bayes_posterior(f, scale, prior_ic, prior_ic_sd)
    cov, _ = ledoit_wolf(f.to_numpy())
    w = nonneg_mean_variance(cov, post["post_mean"].to_numpy())
    return normalise(pd.Series(w, index=f.columns))


METHODS = {"equal": equal_weights, "ic_weighted": ic_weights, "bayes": bayes_weights}


def walk_forward(f: pd.DataFrame, known_at: pd.Series, dates: pd.DatetimeIndex, method: str,
                 min_periods: int = 36, scale: pd.Series | None = None,
                 prior: tuple[float, float] = (PRIOR_IC, PRIOR_IC_SD)) -> pd.DataFrame:
    """Weights at each date in `dates`, estimated only from periods known by that date.

    `known_at[m]` is the date period m's return is fully realised. The first
    `min_periods` known periods are a burn-in; no weights (and no trading) before.
    `scale` (cross-sectional volatility of each period's stock returns) is needed by
    the Bayesian prior only.
    """
    func = METHODS[method]
    known_at = known_at.reindex(f.index)
    rows = {}
    for date in dates:
        known = (known_at <= date).to_numpy()
        hist = f[known].dropna(how="all")
        if len(hist) < min_periods:
            continue
        if method == "bayes":
            rows[date] = func(hist, scale=float(scale[known].mean()), prior_ic=prior[0], prior_ic_sd=prior[1])
        else:
            rows[date] = func(hist)
    return pd.DataFrame(rows).T
