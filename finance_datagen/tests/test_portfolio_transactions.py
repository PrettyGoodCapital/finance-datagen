"""Tests for synthetic position and transaction generators."""

from __future__ import annotations

import polars as pl
import pytest
from finance_enums import PositionEffect, Side

import finance_datagen as fd


def test_positions_generator_shape_and_exposure() -> None:
    positions = fd.PositionsGenerator(n_dates=4, n_assets=5, gross_exposure=1.4, seed=7).generate()

    assert positions.columns == ["date", "symbol", "price", "quantity", "market_value", "weight"]
    assert positions.shape == (20, 6)
    assert positions["date"].dtype == pl.Date
    assert (positions["price"] > 0).all()

    gross = positions.group_by("date").agg(pl.col("weight").abs().sum().alias("gross")).sort("date")
    assert gross["gross"].to_list() == pytest.approx([1.4, 1.4, 1.4, 1.4])

    reconstructed = positions.select((pl.col("quantity") * pl.col("price") - pl.col("market_value")).abs().max()).item()
    assert reconstructed == pytest.approx(0.0, abs=1e-9)


def test_positions_generator_is_deterministic() -> None:
    first = fd.PositionsGenerator(n_dates=3, n_assets=4, seed=11).generate()
    second = fd.PositionsGenerator(n_dates=3, n_assets=4, seed=11).generate()

    assert first.equals(second)


def test_transactions_generator_schema_and_notional() -> None:
    transactions = fd.TransactionsGenerator(n_dates=3, n_assets=4, trades_per_day=6, seed=3).generate()

    assert transactions.columns == ["timestamp", "symbol", "amount", "price", "side", "position_effect", "notional", "commission", "fees", "bps"]
    assert transactions.shape == (18, 10)
    assert transactions["timestamp"].dtype == pl.Datetime("ms", "UTC")
    assert set(transactions["side"].unique().to_list()) <= {Side.Buy.value, Side.Sell.value}
    assert set(transactions["position_effect"].unique().to_list()) <= {PositionEffect.Open.value, PositionEffect.Close.value}
    assert (transactions.filter(pl.col("side") == Side.Buy.value)["amount"] > 0).all()
    assert (transactions.filter(pl.col("side") == Side.Sell.value)["amount"] < 0).all()
    assert (transactions["price"] > 0).all()
    assert (transactions["notional"] >= 0).all()

    max_diff = transactions.select((pl.col("amount").abs() * pl.col("price") - pl.col("notional")).abs().max()).item()
    assert max_diff == pytest.approx(0.0, abs=1e-9)


def test_transactions_generator_validates_shape() -> None:
    with pytest.raises(ValueError, match="trades_per_day"):
        fd.TransactionsGenerator(trades_per_day=0)

    with pytest.raises(ValueError, match="symbols length"):
        fd.TransactionsGenerator(n_assets=3, symbols=["A", "B"])
