"""Is a long-short return new alpha, or a known factor in disguise?

Regress the daily Q5-Q1 return on the Fama-French five factors plus momentum:

    r_t = alpha + b1 MKT + b2 SMB + b3 HML + b4 RMW + b5 CMA + b6 MOM + e_t

A signal whose returns are fully explained by the factors (alpha ~ 0) adds
nothing a cheap factor ETF would not give. The loadings also tell us what the
signal really is: e.g. low_vol typically loads on RMW (profitability) and has a
negative market beta.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from alpha_research.evaluation.stats import ols_newey_west

FACTORS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "MOM"]
TRADING_DAYS = 252


def factor_regression(daily: pd.Series, factors: pd.DataFrame, lags: int = 5) -> pd.DataFrame:
    """Coefficients and Newey-West t-stats; alpha is annualised. Rows: alpha + factors."""
    df = pd.concat([daily.rename("y"), factors[FACTORS]], axis=1, join="inner").dropna()
    X = np.column_stack([np.ones(len(df)), df[FACTORS].to_numpy()])
    beta, se = ols_newey_west(df["y"].to_numpy(), X, lags)
    out = pd.DataFrame({"coef": beta, "t_stat": beta / se}, index=["alpha", *FACTORS])
    out.loc["alpha", "coef"] *= TRADING_DAYS
    resid = df["y"].to_numpy() - X @ beta
    out.attrs["r2"] = 1 - resid.var() / df["y"].var()
    out.attrs["n"] = len(df)
    return out
