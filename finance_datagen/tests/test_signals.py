"""Tests for signal / factor / benchmark generators."""

from __future__ import annotations

import math

import polars as pl
import pytest

import finance_datagen as fd


def test_generate_signal_shape():
    df = fd.generate_signal(n_dates=20, n_assets=10, ic=0.1, seed=0)
    assert df.shape == (200, 4)
    assert df.columns == ["date", "symbol", "signal", "fwd_returns"]
    assert df["date"].dtype == pl.Date


def test_generate_signal_ic_positive():
    df = fd.generate_signal(n_dates=200, n_assets=80, ic=0.3, seed=42)
    ic = df.group_by("date").agg(
        pl.corr(pl.col("signal"), pl.col("fwd_returns"), method="pearson").alias("ic"),
    )
    mean_ic = float(ic["ic"].mean())
    assert mean_ic == pytest.approx(0.3, abs=0.05)


def test_generate_signal_ic_zero():
    df = fd.generate_signal(n_dates=200, n_assets=80, ic=0.0, seed=1)
    ic = df.group_by("date").agg(
        pl.corr(pl.col("signal"), pl.col("fwd_returns"), method="pearson").alias("ic"),
    )
    assert abs(float(ic["ic"].mean())) < 0.05


def test_generate_signal_invalid_ic():
    with pytest.raises(ValueError):
        fd.generate_signal(ic=1.5)


def test_generate_signal_symbol_mismatch():
    with pytest.raises(ValueError):
        fd.generate_signal(n_assets=3, symbols=["A", "B"])


def test_generate_factor_loadings_default():
    df = fd.generate_factor_loadings(n_assets=20, seed=0)
    assert df.height == 20
    assert "symbol" in df.columns
    assert "market" in df.columns
    assert all(v == 1.0 for v in df["market"].to_list())


def test_generate_factor_loadings_standardised():
    df = fd.generate_factor_loadings(n_assets=200, factors=("value", "momentum"), seed=7)
    for col in ("value", "momentum"):
        s = df[col]
        assert float(s.mean()) == pytest.approx(0.0, abs=1e-9)
        assert float(s.std(ddof=0)) == pytest.approx(1.0, abs=1e-9)


def test_generate_benchmark_stats():
    df = fd.generate_benchmark(
        n_dates=2520,
        annual_return=0.10,
        annual_vol=0.20,
        seed=3,
    )
    assert df.shape == (2520, 2)
    assert df["date"].dtype == pl.Date
    rets = df["benchmark"]
    ann_ret = float(rets.mean()) * 252
    ann_vol = float(rets.std()) * math.sqrt(252)
    assert ann_ret == pytest.approx(0.10, abs=0.15)
    assert ann_vol == pytest.approx(0.20, abs=0.02)
