"""Synthetic financial data generation.

The Rust core emits Apache Arrow ``RecordBatch`` values via the pyarrow
PyCapsule interface; this Python layer wraps each generator so that the
public API returns polars ``DataFrame`` objects.
"""

from __future__ import annotations

from typing import Annotated as _Annotated

import polars as _pl
from finance_enums import (
    Currency as _Currency,
    ExchangeCode as _ExchangeCode,
    InstrumentType as _InstrumentType,
    MarketType as _MarketType,
    VenueType as _VenueType,
    exchange_record as _exchange_record,
)
from pydantic import Field as _Field

from ._base import (
    DataGenerator,
    NonNegativeFloat as _NonNegativeFloat,
    PositiveFloat as _PositiveFloat,
    PositiveInt as _PositiveInt,
)
from ._market_models import (
    MarketImpactCurveGenerator,
    MultiAssetGBMGenerator,
    RegimeSwitchingGenerator,
    generate_market_impact_curve,
    generate_multi_asset_gbm,
    generate_regime_switching,
)
from ._portfolio import (
    ExecutionsGenerator,
    OrdersGenerator,
    PositionsGenerator,
    TransactionsGenerator,
    generate_executions,
    generate_orders,
    generate_positions,
    generate_transactions,
)
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

__version__ = "0.2.0"

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
    "OrdersGenerator",
    "ExecutionsGenerator",
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
    "generate_orders",
    "generate_executions",
    "generate_multi_asset_gbm",
    "generate_regime_switching",
    "generate_market_impact_curve",
    "generate_statistical_risk_model",
    "generate_fundamental_risk_model",
    "generate_factor_covariance",
    "generate_specific_variance",
]


def _rb_to_polars(batch) -> _pl.DataFrame:
    return _pl.from_arrow(batch)


def _with_optional_metadata(
    frame: _pl.DataFrame,
    *,
    currency: str | None = None,
    exchange: str | None = None,
    include_region: bool = False,
    instrument_type: str | None = None,
    market_type: str | None = None,
    venue_type: str | None = None,
) -> _pl.DataFrame:
    if currency is not None:
        _Currency(currency)
        frame = frame.with_columns(_pl.lit(currency).alias("currency"))

    if exchange is not None:
        _ExchangeCode(exchange)
        frame = frame.with_columns(_pl.lit(exchange).alias("exchange"))
        if include_region:
            record = _exchange_record(exchange)
            region = None if record is None else record.region
            frame = frame.with_columns(_pl.lit(region).alias("region"))

    if instrument_type is not None:
        _InstrumentType(instrument_type)
        frame = frame.with_columns(_pl.lit(instrument_type).alias("instrument_type"))

    if market_type is not None:
        _MarketType(market_type)
        frame = frame.with_columns(_pl.lit(market_type).alias("market_type"))

    if venue_type is not None:
        _VenueType(venue_type)
        frame = frame.with_columns(_pl.lit(venue_type).alias("venue_type"))

    return frame


class GBMGenerator(DataGenerator[_pl.DataFrame]):
    r"""Geometric Brownian Motion price generator.

    Discretizes the SDE :math:`dS_t = \mu S_t\, dt + \sigma S_t\, dW_t`
    exactly in log-space. Returns a polars ``DataFrame`` with columns
    ``[timestamp, symbol, price]`` of length ``n_steps + 1``.
    """

    s0: _PositiveFloat = 100.0
    mu: float = 0.05
    sigma: _NonNegativeFloat = 0.2
    dt: _PositiveFloat = 1.0 / 252.0
    n_steps: _PositiveInt = 252
    symbol: str = "SYM"
    start_ms: int = 0
    step_ms: int = 86_400_000
    seed: int | None = None
    currency: str | None = None
    exchange: str | None = None
    include_region: bool = False
    instrument_type: str | None = None
    market_type: str | None = None
    venue_type: str | None = None

    def generate(self) -> _pl.DataFrame:
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
        return _with_optional_metadata(
            _rb_to_polars(inner.record_batch()),
            currency=self.currency,
            exchange=self.exchange,
            include_region=self.include_region,
            instrument_type=self.instrument_type,
            market_type=self.market_type,
            venue_type=self.venue_type,
        )


class HestonGenerator(DataGenerator[_pl.DataFrame]):
    r"""Heston stochastic-volatility price generator."""

    s0: _PositiveFloat = 100.0
    v0: _NonNegativeFloat = 0.04
    mu: float = 0.05
    kappa: _NonNegativeFloat = 2.0
    theta: _NonNegativeFloat = 0.04
    xi: _NonNegativeFloat = 0.3
    rho: _Annotated[float, _Field(ge=-1.0, le=1.0)] = -0.7
    dt: _PositiveFloat = 1.0 / 252.0
    n_steps: _PositiveInt = 252
    symbol: str = "SYM"
    start_ms: int = 0
    step_ms: int = 86_400_000
    seed: int | None = None
    currency: str | None = None
    exchange: str | None = None
    include_region: bool = False
    instrument_type: str | None = None
    market_type: str | None = None
    venue_type: str | None = None

    def generate(self) -> _pl.DataFrame:
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
        return _with_optional_metadata(
            _rb_to_polars(inner.record_batch()),
            currency=self.currency,
            exchange=self.exchange,
            include_region=self.include_region,
            instrument_type=self.instrument_type,
            market_type=self.market_type,
            venue_type=self.venue_type,
        )


class GARCHGenerator(DataGenerator[_pl.DataFrame]):
    r"""GARCH(1,1) discrete-time return generator."""

    s0: _PositiveFloat = 100.0
    mu: float = 0.0
    omega: _NonNegativeFloat = 1e-6
    alpha: _NonNegativeFloat = 0.05
    beta: _NonNegativeFloat = 0.90
    n_steps: _PositiveInt = 252
    symbol: str = "SYM"
    start_ms: int = 0
    step_ms: int = 86_400_000
    seed: int | None = None
    currency: str | None = None
    exchange: str | None = None
    include_region: bool = False
    instrument_type: str | None = None
    market_type: str | None = None
    venue_type: str | None = None

    def generate(self) -> _pl.DataFrame:
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
        return _with_optional_metadata(
            _rb_to_polars(inner.record_batch()),
            currency=self.currency,
            exchange=self.exchange,
            include_region=self.include_region,
            instrument_type=self.instrument_type,
            market_type=self.market_type,
            venue_type=self.venue_type,
        )


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
    currency: str | None = None,
    exchange: str | None = None,
    include_region: bool = False,
    instrument_type: str | None = None,
    market_type: str | None = None,
    venue_type: str | None = None,
) -> _pl.DataFrame:
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
        currency=currency,
        exchange=exchange,
        include_region=include_region,
        instrument_type=instrument_type,
        market_type=market_type,
        venue_type=venue_type,
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
    currency: str | None = None,
    exchange: str | None = None,
    include_region: bool = False,
    instrument_type: str | None = None,
    market_type: str | None = None,
    venue_type: str | None = None,
) -> _pl.DataFrame:
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
        currency=currency,
        exchange=exchange,
        include_region=include_region,
        instrument_type=instrument_type,
        market_type=market_type,
        venue_type=venue_type,
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
    currency: str | None = None,
    exchange: str | None = None,
    include_region: bool = False,
    instrument_type: str | None = None,
    market_type: str | None = None,
    venue_type: str | None = None,
) -> _pl.DataFrame:
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
        currency=currency,
        exchange=exchange,
        include_region=include_region,
        instrument_type=instrument_type,
        market_type=market_type,
        venue_type=venue_type,
    ).generate()


def ohlc_from_close(
    close,
    intrabar_vol: float = 0.005,
    base_volume: float = 1_000_000.0,
    vol_factor: float = 5e7,
    symbol: str = "SYM",
    start_ms: int = 0,
    step_ms: int = 86_400_000,
    seed: int | None = None,
) -> _pl.DataFrame:
    """Construct an OHLCV bar series from a close-price series."""
    if isinstance(close, _pl.Series):
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


globals().pop("annotations", None)
globals().pop("finance_datagen", None)
