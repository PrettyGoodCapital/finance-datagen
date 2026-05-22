"""Synthetic market-model generators beyond single-asset paths."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated, Sequence

import numpy as np
import polars as pl
from pydantic import Field, field_validator, model_validator

from ._base import DataGenerator, NonNegativeFloat, PositiveFloat, PositiveInt


def _symbols(n_assets: int, symbols: Sequence[str] | None) -> list[str]:
    values = [f"A{i:04d}" for i in range(n_assets)] if symbols is None else list(symbols)
    if len(values) != n_assets:
        raise ValueError(f"symbols length {len(values)} != n_assets {n_assets}")
    return values


def _as_vector(name: str, value: float | Sequence[float], n_assets: int) -> np.ndarray:
    if isinstance(value, int | float):
        return np.full(n_assets, float(value))
    array = np.asarray(list(value), dtype=float)
    if array.shape != (n_assets,):
        raise ValueError(f"{name} must be scalar or length {n_assets}")
    return array


def _timestamp_grid(n_steps: int, start_ms: int, step_ms: int) -> list[datetime]:
    return [datetime.fromtimestamp((start_ms + i * step_ms) / 1000.0, tz=timezone.utc) for i in range(n_steps + 1)]


def _coerce_array_tuple(value):
    if isinstance(value, np.ndarray):
        return tuple(value.tolist())
    return value


class MultiAssetGBMGenerator(DataGenerator[pl.DataFrame]):
    """Generate correlated multi-asset GBM paths in long form."""

    n_steps: PositiveInt = 252
    n_assets: PositiveInt = 10
    s0: float | tuple[float, ...] = 100.0
    mu: float | tuple[float, ...] = 0.05
    sigma: float | tuple[float, ...] = 0.20
    dt: PositiveFloat = 1.0 / 252.0
    rho: Annotated[float, Field(gt=-1.0, lt=1.0)] = 0.30
    corr: tuple[tuple[float, ...], ...] | None = None
    symbols: tuple[str, ...] | None = None
    start_ms: int = 0
    step_ms: int = 86_400_000
    seed: int | None = None

    @field_validator("s0", "mu", "sigma", "corr", mode="before")
    @classmethod
    def _coerce_numpy_inputs(cls, value):
        return _coerce_array_tuple(value)

    @model_validator(mode="after")
    def _validate_inputs(self):
        s0 = _as_vector("s0", self.s0, self.n_assets)
        sigma = _as_vector("sigma", self.sigma, self.n_assets)
        _as_vector("mu", self.mu, self.n_assets)
        if (s0 <= 0).any() or (sigma < 0).any():
            raise ValueError("s0 must be positive and sigma must be non-negative")
        self._correlation()
        _symbols(self.n_assets, self.symbols)
        return self

    def _correlation(self) -> np.ndarray:
        if self.corr is None:
            matrix = np.full((self.n_assets, self.n_assets), self.rho)
            np.fill_diagonal(matrix, 1.0)
        else:
            matrix = np.asarray(self.corr, dtype=float)
            if matrix.shape != (self.n_assets, self.n_assets):
                raise ValueError(f"corr must have shape ({self.n_assets}, {self.n_assets})")
        if not np.allclose(matrix, matrix.T):
            raise ValueError("corr must be symmetric")
        try:
            np.linalg.cholesky(matrix)
        except np.linalg.LinAlgError as exc:
            raise ValueError("corr must be positive definite") from exc
        return matrix

    def generate(self) -> pl.DataFrame:
        """Return ``[timestamp, symbol, price, return]`` in long form."""
        rng = np.random.default_rng(self.seed)
        s0 = _as_vector("s0", self.s0, self.n_assets)
        mu = _as_vector("mu", self.mu, self.n_assets)
        sigma = _as_vector("sigma", self.sigma, self.n_assets)
        chol = np.linalg.cholesky(self._correlation())
        shocks = rng.normal(0.0, 1.0, size=(self.n_steps, self.n_assets)) @ chol.T
        log_returns = (mu - 0.5 * sigma * sigma) * self.dt + sigma * np.sqrt(self.dt) * shocks
        prices = np.vstack([s0, s0 * np.exp(np.cumsum(log_returns, axis=0))])
        returns = np.vstack([np.zeros(self.n_assets), log_returns])
        timestamps = _timestamp_grid(self.n_steps, self.start_ms, self.step_ms)
        timestamp_values = [timestamp for timestamp in timestamps for _ in range(self.n_assets)]

        return pl.DataFrame(
            {
                "timestamp": pl.Series(timestamp_values).cast(pl.Datetime("ms", "UTC")),
                "symbol": np.tile(np.array(_symbols(self.n_assets, self.symbols)), self.n_steps + 1),
                "price": prices.reshape(-1),
                "return": returns.reshape(-1),
            }
        )


class RegimeSwitchingGenerator(DataGenerator[pl.DataFrame]):
    """Generate a single price path with Markov switching return regimes."""

    s0: PositiveFloat = 100.0
    n_steps: PositiveInt = 252
    transition_matrix: tuple[tuple[float, ...], ...] = ((0.95, 0.05), (0.10, 0.90))
    regime_mu: tuple[float, ...] = (0.0004, -0.0003)
    regime_sigma: tuple[float, ...] = (0.008, 0.025)
    symbol: str = "SYM"
    start_ms: int = 0
    step_ms: int = 86_400_000
    seed: int | None = None

    @field_validator("transition_matrix", "regime_mu", "regime_sigma", mode="before")
    @classmethod
    def _coerce_numpy_inputs(cls, value):
        return _coerce_array_tuple(value)

    @model_validator(mode="after")
    def _validate_regimes(self):
        transition = np.asarray(self.transition_matrix, dtype=float)
        if transition.ndim != 2 or transition.shape[0] != transition.shape[1]:
            raise ValueError("transition matrix must be square")
        if not np.allclose(transition.sum(axis=1), 1.0):
            raise ValueError("transition matrix rows must sum to 1")
        if (transition < 0).any():
            raise ValueError("transition matrix probabilities must be non-negative")
        if len(self.regime_mu) != transition.shape[0] or len(self.regime_sigma) != transition.shape[0]:
            raise ValueError("regime_mu and regime_sigma must match transition matrix size")
        if (np.asarray(self.regime_sigma, dtype=float) < 0).any():
            raise ValueError("regime_sigma must be non-negative")
        return self

    def generate(self) -> pl.DataFrame:
        """Return ``[timestamp, symbol, price, return, regime]``."""
        rng = np.random.default_rng(self.seed)
        transition = np.asarray(self.transition_matrix, dtype=float)
        regime_mu = np.asarray(self.regime_mu, dtype=float)
        regime_sigma = np.asarray(self.regime_sigma, dtype=float)
        regimes = np.zeros(self.n_steps + 1, dtype=int)
        returns = np.zeros(self.n_steps + 1)
        prices = np.empty(self.n_steps + 1)
        prices[0] = self.s0

        for step in range(1, self.n_steps + 1):
            prior = regimes[step - 1]
            regimes[step] = int(rng.choice(np.arange(transition.shape[0]), p=transition[prior]))
            regime = regimes[step]
            returns[step] = rng.normal(regime_mu[regime], regime_sigma[regime])
            prices[step] = prices[step - 1] * np.exp(returns[step])

        return pl.DataFrame(
            {
                "timestamp": pl.Series(_timestamp_grid(self.n_steps, self.start_ms, self.step_ms)).cast(pl.Datetime("ms", "UTC")),
                "symbol": [self.symbol] * (self.n_steps + 1),
                "price": prices,
                "return": returns,
                "regime": regimes,
            }
        )


class MarketImpactCurveGenerator(DataGenerator[pl.DataFrame]):
    """Generate Almgren-Chriss-style impact curves by participation rate."""

    n_assets: PositiveInt | None = None
    symbols: tuple[str, ...] | None = None
    participation_rates: tuple[PositiveFloat, ...] = (
        0.01,
        0.042222222222222223,
        0.07444444444444444,
        0.10666666666666666,
        0.1388888888888889,
        0.1711111111111111,
        0.20333333333333334,
        0.23555555555555557,
        0.2677777777777778,
        0.30,
    )
    average_adv: PositiveFloat = 1_000_000.0
    average_volatility: PositiveFloat = 0.02
    temporary_impact_coef: NonNegativeFloat = 0.5
    permanent_impact_coef: NonNegativeFloat = 0.1
    seed: int | None = None

    @model_validator(mode="after")
    def _validate_symbols(self):
        if self.symbols is not None:
            if not self.symbols:
                raise ValueError("symbols must not be empty")
            if self.n_assets is not None and len(self.symbols) != self.n_assets:
                raise ValueError(f"symbols length {len(self.symbols)} != n_assets {self.n_assets}")
        return self

    def _resolved_symbols(self) -> list[str]:
        if self.symbols is not None:
            return list(self.symbols)
        return _symbols(10 if self.n_assets is None else self.n_assets, None)

    def generate(self) -> pl.DataFrame:
        """Return an impact curve for every symbol and participation rate."""
        rng = np.random.default_rng(self.seed)
        symbols = self._resolved_symbols()
        adv = rng.lognormal(np.log(self.average_adv), 0.35, size=len(symbols))
        vol = rng.lognormal(np.log(self.average_volatility), 0.25, size=len(symbols))

        rows = []
        for symbol, symbol_adv, symbol_vol in zip(symbols, adv, vol):
            for rate in self.participation_rates:
                temporary = self.temporary_impact_coef * symbol_vol * np.sqrt(rate) * 10_000.0
                permanent = self.permanent_impact_coef * symbol_vol * rate * 10_000.0
                rows.append((symbol, rate, symbol_adv, symbol_vol, temporary, permanent, temporary + permanent))

        return pl.DataFrame(
            rows,
            schema=[
                "symbol",
                "participation_rate",
                "adv",
                "volatility",
                "temporary_impact_bps",
                "permanent_impact_bps",
                "total_impact_bps",
            ],
            orient="row",
        )


def generate_multi_asset_gbm(
    n_steps: int = 252,
    n_assets: int = 10,
    s0: float | Sequence[float] = 100.0,
    mu: float | Sequence[float] = 0.05,
    sigma: float | Sequence[float] = 0.20,
    dt: float = 1.0 / 252.0,
    rho: float = 0.30,
    corr: Sequence[Sequence[float]] | None = None,
    symbols: Sequence[str] | None = None,
    start_ms: int = 0,
    step_ms: int = 86_400_000,
    seed: int | None = None,
) -> pl.DataFrame:
    """Generate correlated multi-asset GBM paths in long form."""
    return MultiAssetGBMGenerator(
        n_steps=n_steps,
        n_assets=n_assets,
        s0=s0 if isinstance(s0, int | float) else tuple(s0),
        mu=mu if isinstance(mu, int | float) else tuple(mu),
        sigma=sigma if isinstance(sigma, int | float) else tuple(sigma),
        dt=dt,
        rho=rho,
        corr=None if corr is None else tuple(tuple(row) for row in corr),
        symbols=None if symbols is None else tuple(symbols),
        start_ms=start_ms,
        step_ms=step_ms,
        seed=seed,
    ).generate()


def generate_regime_switching(
    s0: float = 100.0,
    n_steps: int = 252,
    transition_matrix: Sequence[Sequence[float]] = ((0.95, 0.05), (0.10, 0.90)),
    regime_mu: Sequence[float] = (0.0004, -0.0003),
    regime_sigma: Sequence[float] = (0.008, 0.025),
    symbol: str = "SYM",
    start_ms: int = 0,
    step_ms: int = 86_400_000,
    seed: int | None = None,
) -> pl.DataFrame:
    """Generate a single price path with Markov switching return regimes."""
    return RegimeSwitchingGenerator(
        s0=s0,
        n_steps=n_steps,
        transition_matrix=tuple(tuple(row) for row in transition_matrix),
        regime_mu=tuple(regime_mu),
        regime_sigma=tuple(regime_sigma),
        symbol=symbol,
        start_ms=start_ms,
        step_ms=step_ms,
        seed=seed,
    ).generate()


def generate_market_impact_curve(
    n_assets: int | None = None,
    symbols: Sequence[str] | None = None,
    participation_rates: Sequence[float] = (
        0.01,
        0.042222222222222223,
        0.07444444444444444,
        0.10666666666666666,
        0.1388888888888889,
        0.1711111111111111,
        0.20333333333333334,
        0.23555555555555557,
        0.2677777777777778,
        0.30,
    ),
    average_adv: float = 1_000_000.0,
    average_volatility: float = 0.02,
    temporary_impact_coef: float = 0.5,
    permanent_impact_coef: float = 0.1,
    seed: int | None = None,
) -> pl.DataFrame:
    """Generate market-impact curves by participation rate."""
    return MarketImpactCurveGenerator(
        n_assets=n_assets,
        symbols=None if symbols is None else tuple(symbols),
        participation_rates=tuple(participation_rates),
        average_adv=average_adv,
        average_volatility=average_volatility,
        temporary_impact_coef=temporary_impact_coef,
        permanent_impact_coef=permanent_impact_coef,
        seed=seed,
    ).generate()
