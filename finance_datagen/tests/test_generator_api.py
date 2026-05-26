"""Tests for the unified generator API surface."""

from __future__ import annotations

import polars as pl
import pytest

import finance_datagen as fd

GENERATOR_CLASSES = [
    fd.GBMGenerator,
    fd.HestonGenerator,
    fd.GARCHGenerator,
    fd.SignalGenerator,
    fd.FactorLoadingsGenerator,
    fd.BenchmarkGenerator,
    fd.PositionsGenerator,
    fd.TransactionsGenerator,
    fd.OrdersGenerator,
    fd.ExecutionsGenerator,
    fd.MultiAssetGBMGenerator,
    fd.RegimeSwitchingGenerator,
    fd.MarketImpactCurveGenerator,
    fd.StatisticalRiskModelGenerator,
    fd.FundamentalRiskModelGenerator,
    fd.FactorCovarianceGenerator,
    fd.SpecificVarianceGenerator,
]


def _assert_same_output(left, right) -> None:
    if isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert left[key].equals(right[key])
    else:
        assert left.equals(right)


def test_all_public_generators_share_pydantic_base() -> None:
    for generator_class in GENERATOR_CLASSES:
        assert issubclass(generator_class, fd.DataGenerator)
        generator = generator_class()
        assert hasattr(generator, "model_dump")


def test_package_root_exports_only_documented_public_api() -> None:
    public_names = {name for name in vars(fd) if not name.startswith("_")}

    assert public_names == set(fd.__all__)


def test_next_yields_generated_data_once() -> None:
    generator = fd.SignalGenerator(n_dates=3, n_assets=4, seed=9)

    yielded = next(generator)

    assert isinstance(yielded, pl.DataFrame)
    assert yielded.equals(generator.generate())
    with pytest.raises(StopIteration):
        next(generator)


def test_signal_generator_replaces_standalone_helper() -> None:
    model_output = fd.SignalGenerator(n_dates=20, n_assets=10, ic=0.2, seed=4).generate()
    helper_output = fd.generate_signal(n_dates=20, n_assets=10, ic=0.2, seed=4)

    assert model_output.equals(helper_output)


def test_pydantic_field_validation_rejects_invalid_parameters() -> None:
    with pytest.raises(ValueError, match="n_dates"):
        fd.SignalGenerator(n_dates=0)

    with pytest.raises(ValueError, match="ic"):
        fd.SignalGenerator(ic=1.2)


def test_convenience_functions_instantiate_matching_models() -> None:
    cases = [
        (fd.generate_gbm, fd.GBMGenerator, {"n_steps": 3, "seed": 1}),
        (fd.generate_heston, fd.HestonGenerator, {"n_steps": 3, "seed": 2}),
        (fd.generate_garch, fd.GARCHGenerator, {"n_steps": 3, "seed": 3}),
        (fd.generate_factor_loadings, fd.FactorLoadingsGenerator, {"n_assets": 5, "seed": 4}),
        (fd.generate_benchmark, fd.BenchmarkGenerator, {"n_dates": 5, "seed": 5}),
        (fd.generate_positions, fd.PositionsGenerator, {"n_dates": 2, "n_assets": 3, "seed": 6}),
        (fd.generate_transactions, fd.TransactionsGenerator, {"n_dates": 2, "n_assets": 3, "trades_per_day": 2, "seed": 7}),
        (fd.generate_orders, fd.OrdersGenerator, {"n_dates": 2, "n_assets": 3, "orders_per_day": 2, "seed": 8}),
        (fd.generate_executions, fd.ExecutionsGenerator, {"n_dates": 2, "n_assets": 3, "executions_per_day": 2, "seed": 9}),
        (fd.generate_multi_asset_gbm, fd.MultiAssetGBMGenerator, {"n_steps": 2, "n_assets": 3, "seed": 10}),
        (fd.generate_regime_switching, fd.RegimeSwitchingGenerator, {"n_steps": 3, "seed": 11}),
        (fd.generate_market_impact_curve, fd.MarketImpactCurveGenerator, {"n_assets": 3, "seed": 12}),
        (fd.generate_statistical_risk_model, fd.StatisticalRiskModelGenerator, {"n_dates": 6, "n_assets": 4, "n_factors": 2, "seed": 13}),
        (fd.generate_fundamental_risk_model, fd.FundamentalRiskModelGenerator, {"n_assets": 5, "seed": 14}),
        (fd.generate_factor_covariance, fd.FactorCovarianceGenerator, {"factors": ("market", "value"), "seed": 15}),
        (fd.generate_specific_variance, fd.SpecificVarianceGenerator, {"n_assets": 4, "seed": 16}),
    ]

    for helper, generator_class, kwargs in cases:
        _assert_same_output(helper(**kwargs), generator_class(**kwargs).generate())
