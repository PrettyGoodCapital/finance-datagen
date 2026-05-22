"""Synthetic financial data generation.

The Rust core emits Apache Arrow ``RecordBatch`` values via the pyarrow
PyCapsule interface; this Python layer wraps each generator so that the
public API returns polars ``DataFrame`` objects.
"""

from __future__ import annotations

from typing import Annotated, Optional, Sequence

import polars as pl
from pydantic import Field

from ._base import DataGenerator, NonNegativeFloat, PositiveFloat, PositiveInt
from ._market_models import (
    MarketImpactCurveGenerator,
    MultiAssetGBMGenerator,
    RegimeSwitchingGenerator,
    generate_market_impact_curve,
    generate_multi_asset_gbm,
    generate_regime_switching,
)
from ._portfolio import PositionsGenerator, TransactionsGenerator, generate_positions, generate_transactions
from ._risk_models import (
    FactorCovarianceGenerator,
    FundamentalRiskModelGenerator,
    SpecificVarianceGenerator,
    StatisticalRiskModelGenerator,
    generate_factor_covariance,
    generate_fundamental_risk_model,
    generate_specific_variance,
    generate_statistical_risk_model,
)
from ._signals import (
    BenchmarkGenerator,
    FactorLoadingsGenerator,
    SignalGenerator,
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

__version__ = "0.1.1"

__all__ = [
    "DataGenerator",
    "GBMGenerator",
    "HestonGenerator",
    "GARCHGenerator",
    "SignalGenerator",
    "FactorLoadingsGenerator",
    "BenchmarkGenerator",
    "PositionsGenerator",
    "TransactionsGenerator",
    "MultiAssetGBMGenerator",
    "RegimeSwitchingGenerator",
    "MarketImpactCurveGenerator",
    "StatisticalRiskModelGenerator",
    "FundamentalRiskModelGenerator",
    "FactorCovarianceGenerator",
    "SpecificVarianceGenerator",
    "ohlc_from_close",
    "generate_gbm",
    "generate_heston",
    "generate_garch",
    "generate_signal",
    "generate_factor_loadings",
    "generate_benchmark",
    "generate_positions",
    "generate_transactions",
    "generate_multi_asset_gbm",
    "generate_regime_switching",
    "generate_market_impact_curve",
    "generate_statistical_risk_model",
    "generate_fundamental_risk_model",
    "generate_factor_covariance",
    "generate_specific_variance",
]


def _rb_to_polars(batch) -> pl.DataFrame:
    return pl.from_arrow(batch)


class GBMGenerator(DataGenerator[pl.DataFrame]):
    r"""Geometric Brownian Motion price generator.

    Discretizes the SDE :math:`dS_t = \mu S_t\, dt + \sigma S_t\, dW_t`
    exactly in log-space. Returns a polars ``DataFrame`` with columns
    ``[timestamp, symbol, price]`` of length ``n_steps + 1``.
    """

    s0: PositiveFloat = 100.0
    mu: float = 0.05
    sigma: NonNegativeFloat = 0.2
    dt: PositiveFloat = 1.0 / 252.0
    n_steps: PositiveInt = 252
    symbol: str = "SYM"
    start_ms: int = 0
    step_ms: int = 86_400_000
    seed: int | None = None

    def generate(self) -> pl.DataFrame:
        """Simulate the path and return it as a polars ``DataFrame``."""
        inner = _RustGBM(
            s0=self.s0,
            mu=self.mu,
            sigma=self.sigma,
            dt=self.dt,
            n_steps=self.n_steps,
            symbol=self.symbol,
            start_ms=self.start_ms,
            step_ms=self.step_ms,
            seed=self.seed,
        )
        return _rb_to_polars(inner.record_batch())


class HestonGenerator(DataGenerator[pl.DataFrame]):
    r"""Heston stochastic-volatility price generator."""

    s0: PositiveFloat = 100.0
    v0: NonNegativeFloat = 0.04
    mu: float = 0.05
    kappa: NonNegativeFloat = 2.0
    theta: NonNegativeFloat = 0.04
    xi: NonNegativeFloat = 0.3
    rho: Annotated[float, Field(ge=-1.0, le=1.0)] = -0.7
    dt: PositiveFloat = 1.0 / 252.0
    n_steps: PositiveInt = 252
    symbol: str = "SYM"
    start_ms: int = 0
    step_ms: int = 86_400_000
    seed: int | None = None

    def generate(self) -> pl.DataFrame:
        """Simulate the path and return it as a polars ``DataFrame``."""
        inner = _RustHeston(
            s0=self.s0,
            v0=self.v0,
            mu=self.mu,
            kappa=self.kappa,
            theta=self.theta,
            xi=self.xi,
            rho=self.rho,
            dt=self.dt,
            n_steps=self.n_steps,
            symbol=self.symbol,
            start_ms=self.start_ms,
            step_ms=self.step_ms,
            seed=self.seed,
        )
        return _rb_to_polars(inner.record_batch())


class GARCHGenerator(DataGenerator[pl.DataFrame]):
    r"""GARCH(1,1) discrete-time return generator."""

    s0: PositiveFloat = 100.0
    mu: float = 0.0
    omega: NonNegativeFloat = 1e-6
    alpha: NonNegativeFloat = 0.05
    beta: NonNegativeFloat = 0.90
    n_steps: PositiveInt = 252
    symbol: str = "SYM"
    start_ms: int = 0
    step_ms: int = 86_400_000
    seed: int | None = None

    def generate(self) -> pl.DataFrame:
        """Simulate the path and return it as a polars ``DataFrame``."""
        inner = _RustGARCH(
            s0=self.s0,
            mu=self.mu,
            omega=self.omega,
            alpha=self.alpha,
            beta=self.beta,
            n_steps=self.n_steps,
            symbol=self.symbol,
            start_ms=self.start_ms,
            step_ms=self.step_ms,
            seed=self.seed,
        )
        return _rb_to_polars(inner.record_batch())


def generate_gbm(
    s0: float = 100.0,
    mu: float = 0.05,
    sigma: float = 0.2,
    dt: float = 1.0 / 252.0,
    n_steps: int = 252,
    symbol: str = "SYM",
    start_ms: int = 0,
    step_ms: int = 86_400_000,
    seed: int | None = None,
) -> pl.DataFrame:
    """Generate a GBM price path."""
    return GBMGenerator(
        s0=s0,
        mu=mu,
        sigma=sigma,
        dt=dt,
        n_steps=n_steps,
        symbol=symbol,
        start_ms=start_ms,
        step_ms=step_ms,
        seed=seed,
    ).generate()


def generate_heston(
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
    seed: int | None = None,
) -> pl.DataFrame:
    """Generate a Heston price path."""
    return HestonGenerator(
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
    ).generate()


def generate_garch(
    s0: float = 100.0,
    mu: float = 0.0,
    omega: float = 1e-6,
    alpha: float = 0.05,
    beta: float = 0.90,
    n_steps: int = 252,
    symbol: str = "SYM",
    start_ms: int = 0,
    step_ms: int = 86_400_000,
    seed: int | None = None,
) -> pl.DataFrame:
    """Generate a GARCH price and return path."""
    return GARCHGenerator(
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
    ).generate()


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
    """Construct an OHLCV bar series from a close-price series."""
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
