"""Tests for synthetic risk-model data generators."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from finance_enums import Sector

import finance_datagen as fd


def test_fundamental_risk_model_generator_schema_and_standardisation() -> None:
    generator = fd.FundamentalRiskModelGenerator(n_assets=40, seed=2)
    loadings = generator.generate()
    expected_sectors = tuple(member.name for member in Sector)

    expected = [
        "symbol",
        "sector",
        "market",
        "value",
        "momentum",
        "size",
        "quality",
        "low_vol",
        "growth",
        "specific_variance",
    ]
    assert loadings.columns == expected
    assert loadings.shape == (40, len(expected))
    assert generator.sectors == expected_sectors
    assert set(loadings["sector"].unique().to_list()) <= set(expected_sectors)
    assert all(v == 1.0 for v in loadings["market"].to_list())
    assert (loadings["specific_variance"] > 0).all()

    for factor in ["value", "momentum", "size", "quality", "low_vol", "growth"]:
        assert float(loadings[factor].mean()) == pytest.approx(0.0, abs=1e-9)
        assert float(loadings[factor].std(ddof=0)) == pytest.approx(1.0, abs=1e-9)


def test_factor_covariance_generator_is_symmetric_psd() -> None:
    factors = ("market", "sector_tech", "value", "momentum")
    covariance = fd.FactorCovarianceGenerator(factors=factors, seed=4).generate()

    assert covariance.columns == ["factor", *factors]
    assert covariance["factor"].to_list() == list(factors)
    matrix = covariance.drop("factor").to_numpy()
    assert np.allclose(matrix, matrix.T)
    assert np.linalg.eigvalsh(matrix).min() >= -1e-12
    assert np.diag(matrix).min() > 0


def test_specific_variance_generator_outputs_positive_vector() -> None:
    variances = fd.SpecificVarianceGenerator(n_assets=25, seed=8).generate()
    repeat = fd.SpecificVarianceGenerator(n_assets=25, seed=8).generate()

    assert variances.columns == ["symbol", "specific_variance"]
    assert variances.shape == (25, 2)
    assert (variances["specific_variance"] > 0).all()
    assert variances.equals(repeat)


def test_statistical_risk_model_generator_returns_model_components() -> None:
    model = fd.StatisticalRiskModelGenerator(n_dates=160, n_assets=12, n_factors=3, seed=11).generate()

    assert set(model) == {"factor_loadings", "factor_returns", "specific_variance"}
    assert model["factor_loadings"].columns == ["symbol", "factor_1", "factor_2", "factor_3"]
    assert model["factor_loadings"].shape == (12, 4)
    assert model["factor_returns"].columns == ["date", "factor_1", "factor_2", "factor_3"]
    assert model["factor_returns"].shape == (160, 4)
    assert model["factor_returns"]["date"].dtype == pl.Date
    assert model["specific_variance"].columns == ["symbol", "specific_variance"]
    assert (model["specific_variance"]["specific_variance"] > 0).all()


def test_statistical_risk_model_generator_is_deterministic() -> None:
    first = fd.StatisticalRiskModelGenerator(n_dates=40, n_assets=8, n_factors=2, seed=31).generate()
    second = fd.StatisticalRiskModelGenerator(n_dates=40, n_assets=8, n_factors=2, seed=31).generate()

    for key in first:
        assert first[key].equals(second[key])
