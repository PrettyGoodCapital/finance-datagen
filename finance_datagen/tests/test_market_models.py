"""Tests for multi-asset, regime-switching, and market-impact generators."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

import finance_datagen as fd


def test_multi_asset_gbm_generator_produces_correlated_panel() -> None:
    panel = fd.MultiAssetGBMGenerator(n_steps=400, n_assets=3, rho=0.65, seed=5).generate()

    assert panel.columns == ["timestamp", "symbol", "price", "return"]
    assert panel.shape == (401 * 3, 4)
    assert panel["timestamp"].dtype == pl.Datetime("ms", "UTC")
    assert (panel["price"] > 0).all()

    initial_returns = panel.group_by("symbol").agg(pl.col("return").first().alias("first_return")).sort("symbol")
    assert initial_returns["first_return"].to_list() == [0.0, 0.0, 0.0]

    wide_returns = panel.filter(pl.col("return") != 0.0).pivot(index="timestamp", on="symbol", values="return").sort("timestamp")
    corr = np.corrcoef(wide_returns.drop("timestamp").to_numpy().T)
    off_diagonal = corr[np.triu_indices(3, k=1)]
    assert float(off_diagonal.mean()) == pytest.approx(0.65, abs=0.15)


def test_multi_asset_gbm_generator_is_deterministic() -> None:
    first = fd.MultiAssetGBMGenerator(n_steps=20, n_assets=3, seed=19).generate()
    second = fd.MultiAssetGBMGenerator(n_steps=20, n_assets=3, seed=19).generate()

    assert first.equals(second)


def test_multi_asset_gbm_generator_validates_correlation_matrix() -> None:
    with pytest.raises(ValueError, match="positive definite"):
        fd.MultiAssetGBMGenerator(n_assets=3, rho=-0.9)


def test_regime_switching_generator_outputs_regimes() -> None:
    path = fd.RegimeSwitchingGenerator(n_steps=120, seed=13).generate()

    assert path.columns == ["timestamp", "symbol", "price", "return", "regime"]
    assert path.shape == (121, 5)
    assert path["timestamp"].dtype == pl.Datetime("ms", "UTC")
    assert (path["price"] > 0).all()
    assert set(path["regime"].unique().to_list()) <= {0, 1}
    assert path["return"][0] == 0.0


def test_regime_switching_generator_validates_transition_matrix() -> None:
    with pytest.raises(ValueError, match="transition"):
        fd.RegimeSwitchingGenerator(transition_matrix=[[0.8, 0.3], [0.2, 0.8]])


def test_market_impact_curve_generator_monotone_by_participation() -> None:
    curves = fd.MarketImpactCurveGenerator(symbols=["A", "B"], participation_rates=[0.01, 0.10, 0.25], seed=23).generate()

    assert curves.columns == [
        "symbol",
        "participation_rate",
        "adv",
        "volatility",
        "temporary_impact_bps",
        "permanent_impact_bps",
        "total_impact_bps",
    ]
    assert curves.shape == (6, 7)
    assert (curves["adv"] > 0).all()
    assert (curves["total_impact_bps"] > 0).all()

    diffs = curves.sort("symbol", "participation_rate").group_by("symbol").agg(pl.col("total_impact_bps").diff().drop_nulls().min())
    assert (diffs["total_impact_bps"] > 0).all()
