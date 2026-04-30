"""Synthetic financial data generation.

The Rust core emits Apache Arrow ``RecordBatch`` values via the pyarrow
PyCapsule interface; this Python layer wraps each generator so that the
public API returns polars ``DataFrame`` objects.
"""

from __future__ import annotations

from typing import Optional

import polars as pl

from ._signals import (
    generate_benchmark,
    generate_factor_loadings,
    generate_signal,
)
from .finance_datagen import (
    GARCHGenerator as _RustGARCH,
    GBMGenerator as _RustGBM,
    HestonGenerator as _RustHeston,
    ohlc_from_close as _rust_ohlc_from_close,
)

__version__ = "0.1.0"

__all__ = [
    "GBMGenerator",
    "HestonGenerator",
    "GARCHGenerator",
    "ohlc_from_close",
    "generate_signal",
    "generate_factor_loadings",
    "generate_benchmark",
]


def _rb_to_polars(batch) -> pl.DataFrame:
    return pl.from_arrow(batch)


class GBMGenerator:
    r"""Geometric Brownian Motion price generator.

    Discretizes the SDE :math:`dS_t = \mu S_t\, dt + \sigma S_t\, dW_t`
    exactly in log-space. Returns a polars ``DataFrame`` with columns
    ``[timestamp, symbol, price]`` of length ``n_steps + 1``.

    Args:
        s0: Initial price. Must be strictly positive.
        mu: Drift (annualized).
        sigma: Volatility (annualized). Must be non-negative.
        dt: Model time step in years (e.g. ``1/252`` for daily).
        n_steps: Number of return draws. The output has ``n_steps + 1``
            rows.
        symbol: Value written into the ``symbol`` column.
        start_ms: First timestamp, epoch milliseconds (UTC).
        step_ms: Spacing of the timestamp grid in milliseconds.
        seed: Optional ChaCha8 seed for reproducible output.

    Raises:
        ValueError: If any parameter is out of range.
    """

    def __init__(
        self,
        s0: float = 100.0,
        mu: float = 0.05,
        sigma: float = 0.2,
        dt: float = 1.0 / 252.0,
        n_steps: int = 252,
        symbol: str = "SYM",
        start_ms: int = 0,
        step_ms: int = 86_400_000,
        seed: Optional[int] = None,
    ) -> None:
        self._inner = _RustGBM(
            s0=s0,
            mu=mu,
            sigma=sigma,
            dt=dt,
            n_steps=n_steps,
            symbol=symbol,
            start_ms=start_ms,
            step_ms=step_ms,
            seed=seed,
        )

    def generate(self) -> pl.DataFrame:
        """Simulate the path and return it as a polars ``DataFrame``.

        Returns:
            DataFrame with columns ``[timestamp, symbol, price]``.
        """
        return _rb_to_polars(self._inner.record_batch())


class HestonGenerator:
    r"""Heston (1993) stochastic-volatility price generator.

    Two-factor SDE

    .. math::

        dS_t &= \mu S_t\,dt + \sqrt{v_t}\,S_t\,dW_t^{S} \\
        dv_t &= \kappa(\theta - v_t)\,dt + \xi\sqrt{v_t}\,dW_t^{v}

    with :math:`\mathrm{Corr}(dW^S, dW^v) = \rho`. Discretized with
    full-truncation Euler on the variance, log-Euler on the price.
    Returns a polars ``DataFrame`` with columns
    ``[timestamp, symbol, price, variance]``.

    Args:
        s0: Initial price; must be > 0.
        v0: Initial variance; must be >= 0.
        mu: Drift.
        kappa: Mean-reversion speed; must be >= 0.
        theta: Long-run variance; must be >= 0.
        xi: Vol-of-vol; must be >= 0.
        rho: Leverage correlation; must satisfy ``abs(rho) <= 1``.
        dt: Model time step in years.
        n_steps: Number of draws.
        symbol: Value written into the ``symbol`` column.
        start_ms: First timestamp (epoch ms, UTC).
        step_ms: Timestamp grid spacing in ms.
        seed: Optional ChaCha8 seed.

    Raises:
        ValueError: If any parameter is out of range.
    """

    def __init__(
        self,
        s0: float = 100.0,
        v0: float = 0.04,
        mu: float = 0.05,
        kappa: float = 2.0,
        theta: float = 0.04,
        xi: float = 0.3,
        rho: float = -0.7,
        dt: float = 1.0 / 252.0,
        n_steps: int = 252,
        symbol: str = "SYM",
        start_ms: int = 0,
        step_ms: int = 86_400_000,
        seed: Optional[int] = None,
    ) -> None:
        self._inner = _RustHeston(
            s0=s0,
            v0=v0,
            mu=mu,
            kappa=kappa,
            theta=theta,
            xi=xi,
            rho=rho,
            dt=dt,
            n_steps=n_steps,
            symbol=symbol,
            start_ms=start_ms,
            step_ms=step_ms,
            seed=seed,
        )

    def generate(self) -> pl.DataFrame:
        """Simulate the path and return it as a polars ``DataFrame``.

        Returns:
            DataFrame with columns
            ``[timestamp, symbol, price, variance]``. The ``variance``
            column is non-negative (clamped at zero).
        """
        return _rb_to_polars(self._inner.record_batch())


class GARCHGenerator:
    r"""GARCH(1,1) discrete-time return generator.

    .. math::

        r_t &= \mu + \sigma_t Z_t,\quad Z_t \sim \mathcal{N}(0,1) \\
        \sigma_t^2 &= \omega + \alpha\,\varepsilon_{t-1}^2 + \beta\,\sigma_{t-1}^2

    When the process is stationary (``alpha + beta < 1``) the initial
    variance is set to the unconditional value
    ``omega / (1 - alpha - beta)``; otherwise it falls back to ``omega``.
    Returns a polars ``DataFrame`` with columns
    ``[timestamp, symbol, price, return, sigma]``; the first row has
    ``return = 0``.

    Args:
        s0: Initial price; must be > 0.
        mu: Mean log-return.
        omega: Constant variance term; must be >= 0.
        alpha: Shock weight; must be >= 0.
        beta: Persistence weight; must be >= 0.
        n_steps: Number of return draws.
        symbol: Value written into the ``symbol`` column.
        start_ms: First timestamp (epoch ms, UTC).
        step_ms: Timestamp grid spacing in ms.
        seed: Optional ChaCha8 seed.

    Raises:
        ValueError: If any parameter is out of range.
    """

    def __init__(
        self,
        s0: float = 100.0,
        mu: float = 0.0,
        omega: float = 1e-6,
        alpha: float = 0.05,
        beta: float = 0.90,
        n_steps: int = 252,
        symbol: str = "SYM",
        start_ms: int = 0,
        step_ms: int = 86_400_000,
        seed: Optional[int] = None,
    ) -> None:
        self._inner = _RustGARCH(
            s0=s0,
            mu=mu,
            omega=omega,
            alpha=alpha,
            beta=beta,
            n_steps=n_steps,
            symbol=symbol,
            start_ms=start_ms,
            step_ms=step_ms,
            seed=seed,
        )

    def generate(self) -> pl.DataFrame:
        """Simulate the path and return it as a polars ``DataFrame``.

        Returns:
            DataFrame with columns
            ``[timestamp, symbol, price, return, sigma]``.
        """
        return _rb_to_polars(self._inner.record_batch())


def ohlc_from_close(
    close,
    intrabar_vol: float = 0.005,
    base_volume: float = 1_000_000.0,
    vol_factor: float = 5e7,
    symbol: str = "SYM",
    start_ms: int = 0,
    step_ms: int = 86_400_000,
    seed: Optional[int] = None,
) -> pl.DataFrame:
    """Construct an OHLCV bar series from a close-price series.

    For each bar ``i``::

        open_i  = close_{i-1}                        (open_0 = close_0)
        high_i  = max(open_i, close_i) * (1 + |U_1| * intrabar_vol)
        low_i   = min(open_i, close_i) * (1 - |U_2| * intrabar_vol)
        ret_i   = log(close_i / close_{i-1})         (ret_0 = 0)
        vol_i   = base_volume + vol_factor * |ret_i|

    where ``U_1, U_2 ~ Uniform(0, 1)`` are independent. The synthesized
    bars satisfy ``high >= max(open, close)`` and
    ``low <= min(open, close)``.

    Args:
        close: Close-price series. Accepts a list, iterable, numpy
            array, or polars ``Series`` of floats. Must be non-empty.
        intrabar_vol: Per-bar high/low envelope width.
        base_volume: Floor volume.
        vol_factor: Volume sensitivity to absolute log-return.
        symbol: Value written into the ``symbol`` column.
        start_ms: First timestamp (epoch ms, UTC).
        step_ms: Timestamp grid spacing in ms.
        seed: Optional ChaCha8 seed.

    Returns:
        polars ``DataFrame`` with columns
        ``[timestamp, symbol, open, high, low, close, volume]``.

    Raises:
        ValueError: If ``close`` is empty or any parameter is out of
            range.
    """
    if isinstance(close, pl.Series):
        close = close.to_list()
    else:
        close = list(close)
    batch = _rust_ohlc_from_close(
        close,
        intrabar_vol=intrabar_vol,
        base_volume=base_volume,
        vol_factor=vol_factor,
        symbol=symbol,
        start_ms=start_ms,
        step_ms=step_ms,
        seed=seed,
    )
    return _rb_to_polars(batch)
