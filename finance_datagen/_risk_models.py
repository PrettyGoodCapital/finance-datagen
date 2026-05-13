"""Synthetic factor-risk-model data generators."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Annotated, Sequence

import numpy as np
import polars as pl
from finance_enums import Sector
from pydantic import Field, model_validator

from ._base import DataGenerator, NonNegativeFloat, PositiveFloat, PositiveInt

_DEFAULT_SECTORS = tuple(member.value for member in Sector)


def _date_range(n_dates: int, start: date | None) -> list[date]:
    first = date(2020, 1, 1) if start is None else start
    return [first + timedelta(days=i) for i in range(n_dates)]


def _symbols(n_assets: int, symbols: Sequence[str] | None) -> list[str]:
    values = [f"A{i:04d}" for i in range(n_assets)] if symbols is None else list(symbols)
    if len(values) != n_assets:
        raise ValueError(f"symbols length {len(values)} != n_assets {n_assets}")
    return values


def _standardize(values: np.ndarray) -> np.ndarray:
    centered = values - values.mean()
    std = centered.std(ddof=0)
    if std == 0.0:
        return centered
    return centered / std


class SpecificVarianceGenerator(DataGenerator[pl.DataFrame]):
    """Generate a positive idiosyncratic variance vector."""

    n_assets: PositiveInt = 50
    target_vol: PositiveFloat = 0.25
    dispersion: NonNegativeFloat = 0.35
    seed: int | None = None
    symbols: tuple[str, ...] | None = None

    @model_validator(mode="after")
    def _validate_symbols(self):
        _symbols(self.n_assets, self.symbols)
        return self

    def generate(self) -> pl.DataFrame:
        """Return ``[symbol, specific_variance]``."""
        rng = np.random.default_rng(self.seed)
        mean = np.log(self.target_vol * self.target_vol) - 0.5 * self.dispersion * self.dispersion
        variance = rng.lognormal(mean=mean, sigma=self.dispersion, size=self.n_assets)
        return pl.DataFrame({"symbol": _symbols(self.n_assets, self.symbols), "specific_variance": variance})


class FactorCovarianceGenerator(DataGenerator[pl.DataFrame]):
    """Generate a symmetric positive semidefinite factor covariance matrix."""

    factors: tuple[str, ...] = ("market", "sector", "value", "momentum", "size", "quality", "low_vol", "growth")
    factor_vol: PositiveFloat = 0.16
    eigen_decay: Annotated[float, Field(gt=0.0, le=1.0)] = 0.75
    base_corr: Annotated[float, Field(gt=-1.0, lt=1.0)] = 0.25
    seed: int | None = None

    @model_validator(mode="after")
    def _validate_factors(self):
        if not self.factors:
            raise ValueError("factors must not be empty")
        return self

    def generate(self) -> pl.DataFrame:
        """Return a wide covariance matrix with a leading ``factor`` column."""
        _ = np.random.default_rng(self.seed)
        n_factors = len(self.factors)
        indices = np.arange(n_factors)
        corr = self.base_corr ** np.abs(indices[:, None] - indices[None, :])
        vol = self.factor_vol * self.eigen_decay ** (indices / 2.0)
        covariance = np.outer(vol, vol) * corr
        return pl.DataFrame({"factor": self.factors, **{factor: covariance[:, i] for i, factor in enumerate(self.factors)}})


class FundamentalRiskModelGenerator(DataGenerator[pl.DataFrame]):
    """Generate Barra-style sector and style-factor exposure data."""

    n_assets: Annotated[int, Field(gt=1)] = 50
    sectors: tuple[str, ...] = _DEFAULT_SECTORS
    style_factors: tuple[str, ...] = ("value", "momentum", "size", "quality", "low_vol", "growth")
    seed: int | None = None
    symbols: tuple[str, ...] | None = None

    @model_validator(mode="after")
    def _validate_inputs(self):
        if not self.sectors or not self.style_factors:
            raise ValueError("sectors and style_factors must not be empty")
        _symbols(self.n_assets, self.symbols)
        return self

    def generate(self) -> pl.DataFrame:
        """Return wide factor loadings with a positive specific variance."""
        rng = np.random.default_rng(self.seed)
        symbols = _symbols(self.n_assets, self.symbols)
        cols: dict[str, Sequence[float] | Sequence[str] | np.ndarray] = {
            "symbol": symbols,
            "sector": rng.choice(np.array(self.sectors), size=self.n_assets),
            "market": np.ones(self.n_assets),
        }
        for factor in self.style_factors:
            cols[factor] = _standardize(rng.normal(0.0, 1.0, self.n_assets))
        cols["specific_variance"] = (
            SpecificVarianceGenerator(n_assets=self.n_assets, seed=self.seed, symbols=tuple(symbols)).generate()["specific_variance"].to_numpy()
        )
        return pl.DataFrame(cols)


class StatisticalRiskModelGenerator(DataGenerator[dict[str, pl.DataFrame]]):
    """Generate PCA-style statistical factor model components."""

    n_dates: Annotated[int, Field(gt=1)] = 252
    n_assets: Annotated[int, Field(gt=1)] = 50
    n_factors: PositiveInt = 5
    factor_vol: PositiveFloat = 0.01
    idiosyncratic_vol: PositiveFloat = 0.015
    seed: int | None = None
    start: date | None = None
    symbols: tuple[str, ...] | None = None

    @model_validator(mode="after")
    def _validate_inputs(self):
        if self.n_factors >= self.n_assets:
            raise ValueError("n_factors must be smaller than n_assets")
        _symbols(self.n_assets, self.symbols)
        return self

    def generate(self) -> dict[str, pl.DataFrame]:
        """Return factor loadings, factor returns, and specific variance."""
        rng = np.random.default_rng(self.seed)
        true_factor_returns = rng.normal(0.0, self.factor_vol, size=(self.n_dates, self.n_factors))
        true_loadings = rng.normal(0.0, 1.0, size=(self.n_assets, self.n_factors))
        idiosyncratic = rng.normal(0.0, self.idiosyncratic_vol, size=(self.n_dates, self.n_assets))
        returns = true_factor_returns @ true_loadings.T + idiosyncratic
        centered = returns - returns.mean(axis=0, keepdims=True)

        u_matrix, singular_values, vt_matrix = np.linalg.svd(centered, full_matrices=False)
        factors = [f"factor_{i + 1}" for i in range(self.n_factors)]
        factor_scores = u_matrix[:, : self.n_factors] * singular_values[: self.n_factors]
        loadings = vt_matrix[: self.n_factors].T
        residual = centered - factor_scores @ vt_matrix[: self.n_factors]
        specific_variance = residual.var(axis=0, ddof=0)

        symbols = _symbols(self.n_assets, self.symbols)
        factor_loadings = pl.DataFrame({"symbol": symbols, **{factor: loadings[:, i] for i, factor in enumerate(factors)}})
        factor_returns = pl.DataFrame(
            {
                "date": pl.Series(_date_range(self.n_dates, self.start)).cast(pl.Date),
                **{factor: factor_scores[:, i] for i, factor in enumerate(factors)},
            }
        )
        specific = pl.DataFrame({"symbol": symbols, "specific_variance": specific_variance})
        return {"factor_loadings": factor_loadings, "factor_returns": factor_returns, "specific_variance": specific}


def generate_specific_variance(
    n_assets: int = 50,
    target_vol: float = 0.25,
    dispersion: float = 0.35,
    seed: int | None = None,
    symbols: Sequence[str] | None = None,
) -> pl.DataFrame:
    """Generate a positive idiosyncratic variance vector."""
    return SpecificVarianceGenerator(
        n_assets=n_assets,
        target_vol=target_vol,
        dispersion=dispersion,
        seed=seed,
        symbols=None if symbols is None else tuple(symbols),
    ).generate()


def generate_factor_covariance(
    factors: Sequence[str] = ("market", "sector", "value", "momentum", "size", "quality", "low_vol", "growth"),
    factor_vol: float = 0.16,
    eigen_decay: float = 0.75,
    base_corr: float = 0.25,
    seed: int | None = None,
) -> pl.DataFrame:
    """Generate a symmetric positive semidefinite factor covariance matrix."""
    return FactorCovarianceGenerator(
        factors=tuple(factors),
        factor_vol=factor_vol,
        eigen_decay=eigen_decay,
        base_corr=base_corr,
        seed=seed,
    ).generate()


def generate_fundamental_risk_model(
    n_assets: int = 50,
    sectors: Sequence[str] = _DEFAULT_SECTORS,
    style_factors: Sequence[str] = ("value", "momentum", "size", "quality", "low_vol", "growth"),
    seed: int | None = None,
    symbols: Sequence[str] | None = None,
) -> pl.DataFrame:
    """Generate Barra-style sector and style-factor exposure data."""
    return FundamentalRiskModelGenerator(
        n_assets=n_assets,
        sectors=tuple(sectors),
        style_factors=tuple(style_factors),
        seed=seed,
        symbols=None if symbols is None else tuple(symbols),
    ).generate()


def generate_statistical_risk_model(
    n_dates: int = 252,
    n_assets: int = 50,
    n_factors: int = 5,
    factor_vol: float = 0.01,
    idiosyncratic_vol: float = 0.015,
    seed: int | None = None,
    start: date | None = None,
    symbols: Sequence[str] | None = None,
) -> dict[str, pl.DataFrame]:
    """Generate PCA-style statistical factor model components."""
    return StatisticalRiskModelGenerator(
        n_dates=n_dates,
        n_assets=n_assets,
        n_factors=n_factors,
        factor_vol=factor_vol,
        idiosyncratic_vol=idiosyncratic_vol,
        seed=seed,
        start=start,
        symbols=None if symbols is None else tuple(symbols),
    ).generate()
