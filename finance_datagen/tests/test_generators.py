import polars as pl
import pytest

from finance_datagen import (
    GARCHGenerator,
    GBMGenerator,
    HestonGenerator,
    ohlc_from_close,
)

PRICE_COLS = ["timestamp", "symbol", "price"]


def test_gbm_basic():
    gen = GBMGenerator(seed=42, n_steps=10)
    df = gen.generate()
    assert isinstance(df, pl.DataFrame)
    assert df.columns == PRICE_COLS
    assert df.height == 11
    assert (df["price"] > 0).all()


def test_gbm_deterministic():
    a = GBMGenerator(seed=7, n_steps=20).generate()
    b = GBMGenerator(seed=7, n_steps=20).generate()
    assert a.equals(b)


def test_gbm_invalid_params():
    with pytest.raises(ValueError):
        GBMGenerator(s0=0.0)


def test_heston():
    df = HestonGenerator(seed=1, n_steps=50).generate()
    assert df.columns == ["timestamp", "symbol", "price", "variance"]
    assert df.height == 51
    assert (df["price"] > 0).all()
    assert (df["variance"] >= 0).all()


def test_heston_invalid_rho():
    with pytest.raises(ValueError):
        HestonGenerator(rho=1.5)


def test_garch():
    df = GARCHGenerator(seed=3, n_steps=100).generate()
    assert df.columns == ["timestamp", "symbol", "price", "return", "sigma"]
    assert df.height == 101
    assert (df["sigma"] >= 0).all()
    # First row has 0 return.
    assert df["return"][0] == 0.0


def test_ohlc_from_close():
    closes = [100.0 + 0.5 * i for i in range(20)]
    df = ohlc_from_close(closes, seed=99)
    assert df.columns == [
        "timestamp",
        "symbol",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]
    assert df.height == 20
    # Invariants: high >= max(open, close), low <= min(open, close).
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    o = df["open"].to_numpy()
    c = df["close"].to_numpy()
    assert (high >= c).all() and (high >= o).all()
    assert (low <= c).all() and (low <= o).all()


def test_pipeline_gbm_to_ohlc():
    gbm = GBMGenerator(seed=11, n_steps=50).generate()
    ohlcv = ohlc_from_close(gbm["price"], seed=11, symbol="GBM")
    assert ohlcv.height == gbm.height


def test_single_asset_generators_optional_metadata_columns():
    gbm = GBMGenerator(
        seed=1,
        n_steps=4,
        currency="USD",
        exchange="XNYS",
        include_region=True,
        instrument_type="Spot",
        market_type="Equities",
        venue_type="Exchange",
    ).generate()
    heston = HestonGenerator(seed=2, n_steps=4, currency="USD", instrument_type="Spot", market_type="Equities", venue_type="Exchange").generate()
    garch = GARCHGenerator(seed=3, n_steps=4, exchange="XNYS", include_region=True, market_type="Equities", venue_type="Exchange").generate()

    assert {"currency", "exchange", "region", "instrument_type", "market_type", "venue_type"} <= set(gbm.columns)
    assert {"currency", "instrument_type", "market_type", "venue_type"} <= set(heston.columns)
    assert {"exchange", "region", "market_type", "venue_type"} <= set(garch.columns)
    assert gbm["region"].n_unique() == 1
