"""Pure-Python signal / factor generators.

Produces panels of synthetic signals, factor loadings, and benchmarks
for testing alpha / risk / portfolio code without external data.
"""

from __future__ import annotations

import math
from datetime import date, timedelta
from typing import Sequence

import numpy as np
import polars as pl


def _date_range(n_dates: int, start: date | None = None) -> list[date]:
    if start is None:
        start = date(2020, 1, 1)
    return [start + timedelta(days=i) for i in range(n_dates)]


def _symbols(n_assets: int) -> list[str]:
    return [f"A{i:04d}" for i in range(n_assets)]


def generate_signal(
    n_dates: int = 252,
    n_assets: int = 50,
    ic: float = 0.05,
    return_vol: float = 0.02,
    seed: int | None = None,
    start: date | None = None,
    symbols: Sequence[str] | None = None,
) -> pl.DataFrame:
    """Generate a long-form panel ``[date, symbol, signal, fwd_returns]``.

    The signal is constructed so that its cross-sectional Pearson IC
    against ``fwd_returns`` is approximately ``ic`` per date.

    Args:
        n_dates: Number of dates in the panel.
        n_assets: Number of assets per date.
        ic: Target per-date Pearson information coefficient. Must be
            in ``(-1, 1)``. Larger magnitudes produce a more predictive
            signal.
        return_vol: Cross-sectional standard deviation of forward
            returns.
        seed: RNG seed.
        start: First date in the panel. Defaults to ``2020-01-01``.
        symbols: Optional list of symbols of length ``n_assets``.

    Returns:
        Polars DataFrame with columns ``date``, ``symbol``, ``signal``,
        ``fwd_returns``. Length is ``n_dates * n_assets``.

    Raises:
        ValueError: If ``ic`` is not in ``(-1, 1)`` or ``symbols``
            length mismatches ``n_assets``.
    """
    if not -1.0 < ic < 1.0:
        raise ValueError(f"ic must lie in (-1, 1), got {ic}")
    syms = list(symbols) if symbols is not None else _symbols(n_assets)
    if len(syms) != n_assets:
        raise ValueError(f"symbols length {len(syms)} != n_assets {n_assets}")

    rng = np.random.default_rng(seed)
    fwd = rng.normal(0.0, return_vol, (n_dates, n_assets))
    fwd_z = (fwd - fwd.mean(axis=1, keepdims=True)) / fwd.std(axis=1, keepdims=True, ddof=0)
    noise = rng.normal(0.0, 1.0, (n_dates, n_assets))
    signal = ic * fwd_z + math.sqrt(1.0 - ic * ic) * noise

    dates = _date_range(n_dates, start)
    date_col = np.repeat(np.array(dates, dtype="datetime64[D]"), n_assets)
    symbol_col = np.tile(np.array(syms), n_dates)
    return pl.DataFrame(
        {
            "date": date_col,
            "symbol": symbol_col,
            "signal": signal.flatten(),
            "fwd_returns": fwd.flatten(),
        }
    ).with_columns(pl.col("date").cast(pl.Date))


def generate_factor_loadings(
    n_assets: int = 50,
    factors: Sequence[str] = ("market", "value", "momentum", "size", "quality"),
    seed: int | None = None,
    symbols: Sequence[str] | None = None,
) -> pl.DataFrame:
    """Generate Barra-style factor loadings.

    Args:
        n_assets: Number of assets.
        factors: Factor names. Each loading is drawn iid ``N(0, 1)``
            then standardised cross-sectionally to mean zero and unit
            std. The ``market`` factor (if present) is set to 1.0 for
            every asset.
        seed: RNG seed.
        symbols: Optional list of symbols of length ``n_assets``.

    Returns:
        Polars DataFrame with column ``symbol`` plus one column per
        factor.

    Raises:
        ValueError: If ``symbols`` length mismatches ``n_assets``.
    """
    syms = list(symbols) if symbols is not None else _symbols(n_assets)
    if len(syms) != n_assets:
        raise ValueError(f"symbols length {len(syms)} != n_assets {n_assets}")

    rng = np.random.default_rng(seed)
    cols: dict[str, list | np.ndarray] = {"symbol": syms}
    for f in factors:
        if f == "market":
            cols[f] = np.ones(n_assets)
        else:
            x = rng.normal(0.0, 1.0, n_assets)
            x = (x - x.mean()) / x.std(ddof=0)
            cols[f] = x
    return pl.DataFrame(cols)


def generate_benchmark(
    n_dates: int = 252,
    annual_return: float = 0.08,
    annual_vol: float = 0.16,
    periods_per_year: int = 252,
    seed: int | None = None,
    start: date | None = None,
) -> pl.DataFrame:
    """Generate a benchmark return series.

    Args:
        n_dates: Number of dates.
        annual_return: Target annualised mean return.
        annual_vol: Target annualised volatility.
        periods_per_year: Periods per year.
        seed: RNG seed.
        start: First date. Defaults to ``2020-01-01``.

    Returns:
        Polars DataFrame with columns ``date`` (Date) and
        ``benchmark`` (Float64 simple returns).
    """
    rng = np.random.default_rng(seed)
    mu = annual_return / periods_per_year
    sigma = annual_vol / math.sqrt(periods_per_year)
    rets = rng.normal(mu, sigma, n_dates)
    dates = _date_range(n_dates, start)
    return pl.DataFrame({"date": pl.Series(dates).cast(pl.Date), "benchmark": rets})
