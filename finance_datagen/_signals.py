"""Signal, factor-loading, and benchmark generators."""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import date, timedelta
from typing import Annotated

import numpy as np
import polars as pl
from pydantic import Field, model_validator

from ._base import DataGenerator, NonNegativeFloat, PositiveFloat, PositiveInt


def _date_range(n_dates: int, start: date | None = None) -> list[date]:
    first = date(2020, 1, 1) if start is None else start
    return [first + timedelta(days=i) for i in range(n_dates)]


def _symbols(n_assets: int, symbols: Sequence[str] | None = None) -> list[str]:
    values = [f"A{i:04d}" for i in range(n_assets)] if symbols is None else list(symbols)
    if len(values) != n_assets:
        raise ValueError(f"symbols length {len(values)} != n_assets {n_assets}")
    return values


class SignalGenerator(DataGenerator[pl.DataFrame]):
    """Generate a long-form signal and forward-return panel."""

    n_dates: PositiveInt = 252
    n_assets: Annotated[int, Field(gt=1)] = 50
    ic: Annotated[float, Field(gt=-1.0, lt=1.0)] = 0.05
    return_vol: PositiveFloat = 0.02
    seed: int | None = None
    start: date | None = None
    symbols: tuple[str, ...] | None = None

    @model_validator(mode="after")
    def _validate_symbols(self):
        _symbols(self.n_assets, self.symbols)
        return self

    def generate(self) -> pl.DataFrame:
        """Return ``[date, symbol, signal, fwd_returns]``."""
        rng = np.random.default_rng(self.seed)
        fwd = rng.normal(0.0, self.return_vol, (self.n_dates, self.n_assets))
        fwd_z = (fwd - fwd.mean(axis=1, keepdims=True)) / fwd.std(axis=1, keepdims=True, ddof=0)
        noise = rng.normal(0.0, 1.0, (self.n_dates, self.n_assets))
        signal = self.ic * fwd_z + math.sqrt(1.0 - self.ic * self.ic) * noise

        dates = _date_range(self.n_dates, self.start)
        syms = _symbols(self.n_assets, self.symbols)
        return pl.DataFrame(
            {
                "date": np.repeat(np.array(dates, dtype="datetime64[D]"), self.n_assets),
                "symbol": np.tile(np.array(syms), self.n_dates),
                "signal": signal.flatten(),
                "fwd_returns": fwd.flatten(),
            }
        ).with_columns(pl.col("date").cast(pl.Date))


class FactorLoadingsGenerator(DataGenerator[pl.DataFrame]):
    """Generate Barra-style factor loadings."""

    n_assets: Annotated[int, Field(gt=1)] = 50
    factors: tuple[str, ...] = ("market", "value", "momentum", "size", "quality")
    seed: int | None = None
    symbols: tuple[str, ...] | None = None

    @model_validator(mode="after")
    def _validate_inputs(self):
        if not self.factors:
            raise ValueError("factors must not be empty")
        _symbols(self.n_assets, self.symbols)
        return self

    def generate(self) -> pl.DataFrame:
        """Return ``symbol`` plus one column per factor."""
        rng = np.random.default_rng(self.seed)
        cols: dict[str, list[str] | np.ndarray] = {"symbol": _symbols(self.n_assets, self.symbols)}
        for factor in self.factors:
            if factor == "market":
                cols[factor] = np.ones(self.n_assets)
            else:
                values = rng.normal(0.0, 1.0, self.n_assets)
                cols[factor] = (values - values.mean()) / values.std(ddof=0)
        return pl.DataFrame(cols)


class BenchmarkGenerator(DataGenerator[pl.DataFrame]):
    """Generate an independent Gaussian benchmark return series."""

    n_dates: PositiveInt = 252
    annual_return: float = 0.08
    annual_vol: NonNegativeFloat = 0.16
    periods_per_year: PositiveInt = 252
    seed: int | None = None
    start: date | None = None

    def generate(self) -> pl.DataFrame:
        """Return ``[date, benchmark]``."""
        rng = np.random.default_rng(self.seed)
        mu = self.annual_return / self.periods_per_year
        sigma = self.annual_vol / math.sqrt(self.periods_per_year)
        rets = rng.normal(mu, sigma, self.n_dates)
        dates = _date_range(self.n_dates, self.start)
        return pl.DataFrame({"date": pl.Series(dates).cast(pl.Date), "benchmark": rets})


def generate_signal(
    n_dates: int = 252,
    n_assets: int = 50,
    ic: float = 0.05,
    return_vol: float = 0.02,
    seed: int | None = None,
    start: date | None = None,
    symbols: Sequence[str] | None = None,
) -> pl.DataFrame:
    """Generate a long-form panel ``[date, symbol, signal, fwd_returns]``."""
    return SignalGenerator(
        n_dates=n_dates,
        n_assets=n_assets,
        ic=ic,
        return_vol=return_vol,
        seed=seed,
        start=start,
        symbols=None if symbols is None else tuple(symbols),
    ).generate()


def generate_factor_loadings(
    n_assets: int = 50,
    factors: Sequence[str] = ("market", "value", "momentum", "size", "quality"),
    seed: int | None = None,
    symbols: Sequence[str] | None = None,
) -> pl.DataFrame:
    """Generate Barra-style factor loadings."""
    return FactorLoadingsGenerator(
        n_assets=n_assets,
        factors=tuple(factors),
        seed=seed,
        symbols=None if symbols is None else tuple(symbols),
    ).generate()


def generate_benchmark(
    n_dates: int = 252,
    annual_return: float = 0.08,
    annual_vol: float = 0.16,
    periods_per_year: int = 252,
    seed: int | None = None,
    start: date | None = None,
) -> pl.DataFrame:
    """Generate a benchmark return series."""
    return BenchmarkGenerator(
        n_dates=n_dates,
        annual_return=annual_return,
        annual_vol=annual_vol,
        periods_per_year=periods_per_year,
        seed=seed,
        start=start,
    ).generate()
